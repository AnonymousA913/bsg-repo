from __future__ import annotations

"""Benchmark controlled CPN representations with a shared timing protocol.

The checkpoints are recovered from the Section 5.4 evaluation metadata so
that this benchmark uses exactly the models already reported in the paper.
Model loading and warm-up are excluded.  The measured end-to-end time starts
before environment construction and ends after a complete greedy schedule is
produced.  All formal tasks use batch size one to represent online control.
"""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent
        for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    EvaluationTask,
    evaluate_batch,
    load_evaluator,
    make_environment,
    ordered_test_paths,
    seed_everything,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
SECTION54_EVALUATION_ROOT = (
    PACKAGE_ROOT
    / "results"
    / "sec54_representation"
    / "evaluation"
)
OUTPUT_ROOT = PACKAGE_ROOT / "results" / "sec57_runtime" / "controlled"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_SEED = 9000
FORMAL_WARMUP_INSTANCES = 5
FORMAL_BENCHMARK_INSTANCES = 20
FORMAL_REPEATS = 5
SMOKE_WARMUP_INSTANCES = 1
SMOKE_BENCHMARK_INSTANCES = 2
SMOKE_REPEATS = 1

# =============================================================================
# USER CONFIGURATION
# Edit this block and run the file directly in PyCharm.
# =============================================================================

SELECTED_TASK = "sd1_10x5_bsg"
RUN_PROFILE = "formal"          # "smoke" or "formal"
RUN_ALL_MAIN_TASKS = True        # Run all 5 methods x 4 paper settings.
OVERWRITE_EXISTING = False       # False resumes/skips completed measurements.
LIST_TASKS_ONLY = False
CHECK_CONFIGURATION_ONLY = False

# =============================================================================


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    label: str
    distribution: str
    size: str
    method_directory: str
    namespace: str = "formal_500x20"
    checkpoint_label: str = "best"

    @property
    def source_metadata(self) -> Path:
        return (
            SECTION54_EVALUATION_ROOT
            / self.namespace
            / self.distribution
            / self.size
            / self.method_directory
            / f"seed_{TRAINING_SEED}"
            / self.checkpoint_label
            / "greedy_metadata.json"
        )


METHODS = (
    ("petrirl", "PetriRL", "masked_mlp"),
    ("pnc", "PNC", "pnc_ppo"),
    ("pngcn", "PN-GCN", "petri_net_gcn_ppo"),
    ("ag", "AG", "ag_ppo"),
    ("bsg", "BSG-RL", "generic_compact_v5"),
)

PAPER_SETTINGS = (
    ("SD2", "20x10"),
    ("SD1", "10x5"),
    ("SD1", "15x10"),
    ("SD2", "20x5"),

)


def _main_specs() -> Iterable[BenchmarkSpec]:
    for distribution, size in PAPER_SETTINGS:
        prefix = f"{distribution.lower()}_{size}"
        for suffix, label, directory in METHODS:
            yield BenchmarkSpec(
                key=f"{prefix}_{suffix}",
                label=label,
                distribution=distribution,
                size=size,
                method_directory=directory,
                checkpoint_label=f"{directory}_best",
            )


MAIN_SPECS = tuple(_main_specs())
TASKS: Dict[str, BenchmarkSpec] = {
    spec.key: spec for spec in MAIN_SPECS
}
TASKS.update({
    "sd1_10x5_bsg_s": BenchmarkSpec(
        key="sd1_10x5_bsg_s",
        label="BSG-RL-S",
        distribution="SD1",
        size="10x5",
        method_directory="bsg_rl_small",
        namespace="capacity_control_500_budget",
    ),
    "sd2_20x5_bsg_s": BenchmarkSpec(
        key="sd2_20x5_bsg_s",
        label="BSG-RL-S",
        distribution="SD2",
        size="20x5",
        method_directory="bsg_rl_small",
        namespace="capacity_control_500_budget",
    ),
})
MAIN_TASK_KEYS = tuple(spec.key for spec in MAIN_SPECS)


CSV_FIELDS = (
    "system",
    "task",
    "profile",
    "method",
    "distribution",
    "size",
    "training_seed",
    "repeat",
    "instance_index",
    "instance_id",
    "makespan",
    "decision_count",
    "parameters",
    "environment_construction_s",
    "reset_and_initial_graph_s",
    "tensorization_s",
    "network_inference_s",
    "action_selection_s",
    "environment_step_and_graph_s",
    "measured_rollout_s",
    "end_to_end_s",
    "expected_instances_per_repeat",
    "expected_repeats",
)


def _profile_values(profile: str) -> tuple[int, int, int]:
    if profile == "smoke":
        return (
            SMOKE_WARMUP_INSTANCES,
            SMOKE_BENCHMARK_INSTANCES,
            SMOKE_REPEATS,
        )
    return (
        FORMAL_WARMUP_INSTANCES,
        FORMAL_BENCHMARK_INSTANCES,
        FORMAL_REPEATS,
    )


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_rows_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_source(spec: BenchmarkSpec) -> tuple[Path, Path, dict]:
    if not spec.source_metadata.is_file():
        raise FileNotFoundError(
            f"Missing Section 5.4 metadata for {spec.key}: "
            f"{spec.source_metadata}"
        )
    metadata = json.loads(spec.source_metadata.read_text(encoding="utf-8"))
    checkpoint = Path(str(metadata["checkpoint"]))
    dataset = Path(str(metadata["test_dataset"]))
    if not checkpoint.is_absolute():
        checkpoint = (PROJECT_ROOT / checkpoint).resolve()
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (
        checkpoint.name != "best_checkpoint.pt"
        or checkpoint.parent.name != f"seed_{TRAINING_SEED}"
    ):
        raise ValueError(
            "Section 5.7 requires the Section 5.4 best checkpoint at "
            f"seed_{TRAINING_SEED}/best_checkpoint.pt: {checkpoint}"
        )
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    return checkpoint, dataset, metadata


def _evaluation_task(
    spec: BenchmarkSpec,
    checkpoint: Path,
    dataset: Path,
    required_instances: int,
    training_seed: int,
) -> EvaluationTask:
    return EvaluationTask(
        mode="formal",
        distribution=spec.distribution,
        size=spec.size,
        training_seed=training_seed,
        checkpoint=checkpoint,
        test_dataset=dataset,
        output_dir=OUTPUT_ROOT / spec.key,
        checkpoint_label="runtime",
        test_limit=required_instances,
        greedy_batch_size=1,
        device=DEVICE,
    )


def _run_one(trainer, training: Mapping[str, object], path: Path) -> tuple:
    _synchronize(DEVICE)
    total_started = time.perf_counter()
    construction_started = time.perf_counter()
    environment = make_environment(path, training)
    construction_seconds = time.perf_counter() - construction_started
    outcomes, timing = evaluate_batch(
        trainer, [environment], deterministic=True
    )
    _synchronize(DEVICE)
    total_seconds = time.perf_counter() - total_started
    if len(outcomes) != 1:
        raise RuntimeError("Batch-size-one timing returned multiple outcomes")
    return outcomes[0], timing, construction_seconds, total_seconds


def _check_finite(row: Mapping[str, object]) -> None:
    for field in CSV_FIELDS:
        if not field.endswith("_s"):
            continue
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid timing {field}={value}")


def run(spec: BenchmarkSpec, profile: str, overwrite: bool) -> Path:
    warmup_count, benchmark_count, repeats = _profile_values(profile)
    checkpoint, dataset, source_metadata = _load_source(spec)
    checkpoint_training_seed = TRAINING_SEED
    task = _evaluation_task(
        spec,
        checkpoint,
        dataset,
        warmup_count + benchmark_count,
        checkpoint_training_seed,
    )
    task.validate()
    paths = ordered_test_paths(task)
    warmup_paths = paths[:warmup_count]
    benchmark_paths = paths[warmup_count:]
    if len(benchmark_paths) != benchmark_count:
        raise RuntimeError("Unexpected benchmark instance count")

    output_dir = OUTPUT_ROOT / profile / spec.key
    output_path = output_dir / "per_instance_timing.csv"
    rows = [] if overwrite else _read_rows(output_path)
    completed = {
        (int(row["repeat"]), str(row["instance_id"])) for row in rows
    }

    seed_everything(checkpoint_training_seed)
    trainer, checkpoint_payload = load_evaluator(checkpoint, device=DEVICE)
    training = checkpoint_payload["training_config"]
    parameters = sum(
        parameter.numel()
        for parameter in trainer.policy.parameters()
        if parameter.requires_grad
    )

    print("Section 5.7 controlled runtime benchmark")
    print(f"  task       : {spec.key}")
    print(f"  method     : {spec.label}")
    print(f"  scenario   : {spec.distribution} {spec.size}")
    print(f"  device     : {DEVICE}")
    print(f"  warm-up    : {warmup_count} instances")
    print(f"  benchmark  : {benchmark_count} instances x {repeats} repeats")
    print(f"  checkpoint : {checkpoint}")

    print("  warming up ...", flush=True)
    for path in warmup_paths:
        _run_one(trainer, training, path)

    for repeat in range(1, repeats + 1):
        for instance_index, path in enumerate(benchmark_paths, start=1):
            key = (repeat, path.name)
            if key in completed:
                continue
            outcome, timing, construction_seconds, total_seconds = _run_one(
                trainer, training, path
            )
            row = {
                "system": "controlled_cpn",
                "task": spec.key,
                "profile": profile,
                "method": spec.label,
                "distribution": spec.distribution,
                "size": spec.size,
                "training_seed": checkpoint_training_seed,
                "repeat": repeat,
                "instance_index": instance_index,
                "instance_id": path.name,
                "makespan": outcome.makespan,
                "decision_count": outcome.decision_count,
                "parameters": parameters,
                "environment_construction_s": construction_seconds,
                "reset_and_initial_graph_s": timing.reset_and_initial_graph_s,
                "tensorization_s": timing.tensorization_s,
                "network_inference_s": timing.network_inference_s,
                "action_selection_s": timing.action_selection_s,
                "environment_step_and_graph_s": (
                    timing.environment_step_and_graph_s
                ),
                "measured_rollout_s": timing.end_to_end_s,
                "end_to_end_s": total_seconds,
                "expected_instances_per_repeat": benchmark_count,
                "expected_repeats": repeats,
            }
            _check_finite(row)
            rows.append(row)
            rows.sort(key=lambda item: (
                int(item["repeat"]), int(item["instance_index"])
            ))
            _write_rows_atomic(output_path, rows)
            completed.add(key)
            print(
                f"  repeat {repeat}/{repeats}, "
                f"instance {instance_index}/{benchmark_count}: "
                f"{total_seconds:.4f} s",
                flush=True,
            )

    metadata = {
        "protocol": "sec57_controlled_runtime_v1",
        "task": spec.key,
        "profile": profile,
        "method": spec.label,
        "distribution": spec.distribution,
        "size": spec.size,
        "device": DEVICE,
        "batch_size": 1,
        "greedy_decoding": True,
        "training_seed": checkpoint_training_seed,
        "warmup_instances": warmup_count,
        "benchmark_instances": benchmark_count,
        "repeats": repeats,
        "parameters": parameters,
        "checkpoint": str(checkpoint),
        "source_metadata": str(spec.source_metadata),
        "checkpoint_iteration": source_metadata.get("checkpoint_iteration"),
        "output": str(output_path),
    }
    _write_json_atomic(output_dir / "benchmark_metadata.json", metadata)
    return output_path


def main() -> int:
    if RUN_PROFILE not in {"smoke", "formal"}:
        raise ValueError("RUN_PROFILE must be 'smoke' or 'formal'")
    if LIST_TASKS_ONLY:
        for key, spec in TASKS.items():
            print(
                f"{key:24s}  {spec.label:9s}  "
                f"{spec.distribution} {spec.size}"
            )
        return 0
    if RUN_ALL_MAIN_TASKS:
        for key in MAIN_TASK_KEYS:
            if CHECK_CONFIGURATION_ONLY:
                checkpoint, dataset, _ = _load_source(TASKS[key])
                print(f"{key}: {checkpoint} | {dataset}")
                continue
            output = run(TASKS[key], RUN_PROFILE, OVERWRITE_EXISTING)
            print(f"Output: {output}")
        if CHECK_CONFIGURATION_ONLY:
            print("All main-task configuration checks passed.")
        return 0
    if SELECTED_TASK not in TASKS:
        raise ValueError(
            f"Unknown SELECTED_TASK={SELECTED_TASK!r}; "
            f"choose one of {tuple(TASKS)}"
        )
    spec = TASKS[SELECTED_TASK]
    checkpoint, dataset, _ = _load_source(spec)
    print(f"checkpoint: {checkpoint}")
    print(f"dataset   : {dataset}")
    if CHECK_CONFIGURATION_ONLY:
        print("Configuration check passed.")
        return 0
    output = run(spec, RUN_PROFILE, OVERWRITE_EXISTING)
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

