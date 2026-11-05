from __future__ import annotations

"""Shared evaluator for independent Section 5.4 representation scripts."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

import json
from pathlib import Path
import re
from statistics import mean, stdev
import traceback
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    EvaluationTask,
    read_csv,
    write_csv_atomic,
)
from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_runner import (
    run_greedy,
    run_sampling100,
)
from CPNModelV2.learning import (
    POLICY_AG_PPO,
    POLICY_BSG_RL_SMALL,
    POLICY_MASKED_MLP,
    POLICY_PETRI_NET_GCN_PPO,
    POLICY_PNC_PPO,
)
from CPNModelV2.experiments.sec52_fjsp_performance.train_generic_compact_10x5 import (
    POLICY_GENERIC_COMPACT,
)


# (label, method, distribution, training size, training seed, relative PT path)
CheckpointSelection = Tuple[str, str, str, str, int, Path]

SUPPORTED_METHODS = (
    POLICY_AG_PPO,
    POLICY_MASKED_MLP,
    POLICY_PNC_PPO,
    POLICY_PETRI_NET_GCN_PPO,
    POLICY_GENERIC_COMPACT,
    POLICY_BSG_RL_SMALL,
)
METHOD_DIRECTORY = {
    POLICY_AG_PPO: "ag_ppo",
    POLICY_MASKED_MLP: "masked_mlp",
    POLICY_PNC_PPO: "pnc_ppo",
    POLICY_PETRI_NET_GCN_PPO: "petri_net_gcn_ppo",
    POLICY_GENERIC_COMPACT: "generic_compact_v5",
    POLICY_BSG_RL_SMALL: "bsg_rl_small",
}
METHOD_LABEL = {
    POLICY_AG_PPO: "AG-PPO",
    POLICY_MASKED_MLP: "PetriRL-FJSP",
    POLICY_PNC_PPO: "PNC-PPO",
    POLICY_PETRI_NET_GCN_PPO: "PN-GCN-PPO",
    POLICY_GENERIC_COMPACT: "BSG-RL",
    POLICY_BSG_RL_SMALL: "BSG-RL-S",
}


# =============================================================================
# CONFIGURATION SUPPLIED BY AN INDEPENDENT WRAPPER
# =============================================================================

# Run Greedy first. Sampling-100 is supported and resumes after interruption.
DECODING_MODE = "greedy"  # greedy or sampling100

# Deliberately empty: running this shared module must not start an old experiment.
CHECKPOINTS_TO_EVALUATE: Sequence[CheckpointSelection] = ()

# "smoke": 2 instances and 4 sampling rollouts.
# "full": 100 instances and 100 sampling rollouts.
EVALUATION_PROFILE = "full"  # smoke or full
TEST_DATA_ROOT = Path("data")
EVALUATION_ROOT = Path("CPNModelV2/results/sec54_representation/evaluation")
RESULT_NAMESPACE = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GREEDY_BATCH_SIZE = 5
MAX_SAMPLING_BATCH_SIZE = 5
DEVICE_BATCH_BY_SIZE = {
    "10x5": 5,
    "20x5": 2,
    "15x10": 2,
    "20x10": 1,
}
DEFAULT_DEVICE_BATCH_SIZE = 1

# Completed rows are retained by default. Set True only when a result is known
# to be invalid and the selected decoder must be recomputed.
OVERWRITE = False
SKIP_MISSING_CHECKPOINTS = False
STOP_ON_ERROR = True

# Reject a checkpoint outside the paper's declared training budget. Set to
# None only for an explicitly separate checkpoint study.
MAX_CHECKPOINT_ITERATION: int | None = 500

# =============================================================================


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
SUPPORTED_DISTRIBUTIONS = ("SD1", "SD2")


def project_path(relative_path: Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Path must be project-relative: {path}")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes the project root: {path}") from error
    return resolved


def checkpoint_path(configured_path: Path) -> Path:
    """Resolve a user-selected checkpoint path.

    Checkpoints may be configured either as absolute paths or as paths relative
    to the project root. Other project inputs and outputs remain project-relative.
    """

    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else project_path(path)


def _checkpoint_header(path: Path) -> Tuple[str, int]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return (
        str(checkpoint.get("policy_variant", "")),
        int(checkpoint.get("iteration", -1)),
    )


def _validate_selection(selection: CheckpointSelection) -> None:
    if not isinstance(selection, (tuple, list)) or len(selection) != 6:
        raise ValueError(
            "Each checkpoint entry must be "
            "(label, method, distribution, size, seed, relative_pt_path)"
        )
    label, method, distribution, size, seed, relative_path = selection
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise ValueError(f"Invalid checkpoint label: {label!r}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported Section 5.4 method: {method}")
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution}")
    if not re.fullmatch(r"\d+x\d+", str(size)):
        raise ValueError(f"Invalid FJSP size: {size}")
    if not isinstance(seed, int):
        raise ValueError(f"Training seed must be an integer: {seed!r}")
    path = Path(relative_path)
    if path.name != "best_checkpoint.pt" or path.parent.name != f"seed_{seed}":
        raise ValueError(
            "Each checkpoint path must end with "
            f"seed_{seed}/best_checkpoint.pt: {path}"
        )
    checkpoint_path(path)


def evaluation_task(selection: CheckpointSelection) -> EvaluationTask:
    _validate_selection(selection)
    label, method, distribution, size, seed, relative_checkpoint = selection
    dataset_dir = size if distribution == "SD1" else f"{size}+mix"
    default_batch = DEVICE_BATCH_BY_SIZE.get(
        size, DEFAULT_DEVICE_BATCH_SIZE
    )
    result_root = project_path(EVALUATION_ROOT)
    if RESULT_NAMESPACE:
        result_root = result_root / RESULT_NAMESPACE
    return EvaluationTask(
        mode="formal",
        distribution=distribution,
        size=size,
        training_seed=seed,
        checkpoint=checkpoint_path(Path(relative_checkpoint)),
        test_dataset=project_path(TEST_DATA_ROOT / distribution / dataset_dir),
        output_dir=(
            result_root
            / distribution
            / size
            / METHOD_DIRECTORY[method]
            / f"seed_{seed}"
            / label
        ).resolve(),
        checkpoint_label=label,
        test_limit=2 if EVALUATION_PROFILE == "smoke" else 100,
        greedy_batch_size=min(MAX_GREEDY_BATCH_SIZE, default_batch),
        sampling_rollouts=4 if EVALUATION_PROFILE == "smoke" else 100,
        sampling_batch_size=min(MAX_SAMPLING_BATCH_SIZE, default_batch),
        device=DEVICE,
    )


def selected_evaluation_tasks(
) -> Iterable[Tuple[str, EvaluationTask]]:
    seen_outputs = set()
    for raw_selection in CHECKPOINTS_TO_EVALUATE:
        selection = tuple(raw_selection)
        task = evaluation_task(selection)
        if task.output_dir in seen_outputs:
            raise ValueError(
                f"Duplicate evaluation output directory: {task.output_dir}"
            )
        seen_outputs.add(task.output_dir)
        yield selection[1], task


def _print_result(task: EvaluationTask) -> None:
    metadata_name = (
        "greedy_metadata.json"
        if DECODING_MODE == "greedy"
        else "sampling100_metadata.json"
    )
    path = task.output_dir / metadata_name
    if not path.is_file():
        return
    metadata = json.loads(path.read_text(encoding="utf-8"))
    result_key = (
        "mean_makespan"
        if DECODING_MODE == "greedy"
        else "mean_best_makespan"
    )
    print(f"  result     : {float(metadata[result_key]):.3f}")
    print(f"  gap        : {float(metadata['aggregate_gap_percent']):.3f}%")
    print(f"  checkpoint : iteration {metadata['checkpoint_iteration']}")
    print(f"  output     : {task.output_dir}")


PUBLIC_GREEDY_FIELDS = (
    "distribution",
    "size",
    "method",
    "seed",
    "instance_id",
    "makespan",
    "reference_makespan",
    "relative_gap_percent",
    "checkpoint_label",
    "checkpoint_iteration",
    "data_status",
)

PUBLIC_INSTANCE_FIELDS = (
    "distribution",
    "size",
    "method",
    "instance_id",
    "seed_count",
    "mean_makespan",
    "std_across_seeds_makespan",
    "reference_makespan",
    "relative_gap_percent",
    "data_status",
)

PUBLIC_SEED_FIELDS = (
    "distribution",
    "size",
    "method",
    "seed",
    "checkpoint_label",
    "checkpoint_iteration",
    "test_instances",
    "mean_makespan",
    "mean_instance_relative_gap_percent",
    "aggregate_gap_percent",
    "data_status",
)


def export_greedy_distribution_tables(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Merge selected greedy outputs into plot- and table-ready CSV files."""

    selected = list(selected_evaluation_tasks())
    if not selected:
        raise ValueError("No selected evaluation tasks to aggregate")
    combinations = {
        (task.distribution, task.size, method)
        for method, task in selected
    }
    if len(combinations) != 1:
        raise ValueError(
            "Distribution export expects one scenario and one method; "
            f"found {sorted(combinations)}"
        )

    public_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    for method, task in selected:
        result_path = task.output_dir / "greedy_per_instance.csv"
        metadata_path = task.output_dir / "greedy_metadata.json"
        rows = read_csv(result_path)
        if len(rows) != task.test_limit:
            raise RuntimeError(
                f"{result_path} has {len(rows)} rows; expected {task.test_limit}"
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint_iteration = int(metadata["checkpoint_iteration"])
        label = METHOD_LABEL[method]

        makespans: List[float] = []
        references: List[float] = []
        gaps: List[float] = []
        for row in rows:
            if str(row.get("feasible", "")).lower() != "true":
                raise RuntimeError(
                    f"Infeasible result in {result_path}: {row.get('instance_id')}"
                )
            makespan = float(row["makespan"])
            reference = float(row["reference_makespan"])
            if reference <= 0:
                raise ValueError(
                    f"Non-positive reference for {row.get('instance_id')}: {reference}"
                )
            gap = (makespan - reference) / reference * 100.0
            makespans.append(makespan)
            references.append(reference)
            gaps.append(gap)
            public_rows.append(
                {
                    "distribution": task.distribution,
                    "size": task.size,
                    "method": label,
                    "seed": task.training_seed,
                    "instance_id": row["instance_id"],
                    "makespan": makespan,
                    "reference_makespan": reference,
                    "relative_gap_percent": gap,
                    "checkpoint_label": task.checkpoint_label,
                    "checkpoint_iteration": checkpoint_iteration,
                    "data_status": "observed",
                }
            )

        seed_rows.append(
            {
                "distribution": task.distribution,
                "size": task.size,
                "method": label,
                "seed": task.training_seed,
                "checkpoint_label": task.checkpoint_label,
                "checkpoint_iteration": checkpoint_iteration,
                "test_instances": len(rows),
                "mean_makespan": mean(makespans),
                "mean_instance_relative_gap_percent": mean(gaps),
                "aggregate_gap_percent": (
                    (mean(makespans) - mean(references))
                    / mean(references)
                    * 100.0
                ),
                "data_status": "observed",
            }
        )

    public_rows.sort(
        key=lambda row: (int(row["seed"]), str(row["instance_id"]))
    )
    seed_rows.sort(key=lambda row: int(row["seed"]))
    expected_seed_count = len({int(row["seed"]) for row in seed_rows})

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in public_rows:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    instance_rows: List[Dict[str, Any]] = []
    for instance_id, rows in sorted(grouped.items()):
        values = [float(row["makespan"]) for row in rows]
        references = {float(row["reference_makespan"]) for row in rows}
        if len(references) != 1:
            raise ValueError(
                f"Inconsistent references for instance {instance_id}: {references}"
            )
        reference = next(iter(references))
        value_mean = mean(values)
        instance_rows.append(
            {
                "distribution": rows[0]["distribution"],
                "size": rows[0]["size"],
                "method": rows[0]["method"],
                "instance_id": instance_id,
                "seed_count": len(rows),
                "mean_makespan": value_mean,
                "std_across_seeds_makespan": (
                    stdev(values) if len(values) > 1 else ""
                ),
                "reference_makespan": reference,
                "relative_gap_percent": (
                    (value_mean - reference) / reference * 100.0
                ),
                "data_status": (
                    "observed"
                    if len(rows) == expected_seed_count
                    else "incomplete_seeds"
                ),
            }
        )

    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_instance_path = output_dir / "test_distribution_per_seed_instance.csv"
    instance_summary_path = output_dir / "test_distribution_instance_summary.csv"
    seed_summary_path = output_dir / "test_distribution_seed_summary.csv"
    write_csv_atomic(per_seed_instance_path, public_rows, PUBLIC_GREEDY_FIELDS)
    write_csv_atomic(instance_summary_path, instance_rows, PUBLIC_INSTANCE_FIELDS)
    write_csv_atomic(seed_summary_path, seed_rows, PUBLIC_SEED_FIELDS)
    return per_seed_instance_path, instance_summary_path, seed_summary_path


def main() -> int:
    if DECODING_MODE not in {"greedy", "sampling100"}:
        raise ValueError("DECODING_MODE must be greedy or sampling100")
    if EVALUATION_PROFILE not in {"smoke", "full"}:
        raise ValueError("EVALUATION_PROFILE must be smoke or full")
    if not CHECKPOINTS_TO_EVALUATE:
        raise ValueError(
            "CHECKPOINTS_TO_EVALUATE cannot be empty; run an independent "
            "Section 5.4 evaluation wrapper"
        )
    if RESULT_NAMESPACE and (
        Path(RESULT_NAMESPACE).is_absolute()
        or Path(RESULT_NAMESPACE).name != RESULT_NAMESPACE
    ):
        raise ValueError("RESULT_NAMESPACE must be one directory name")
    selected = list(selected_evaluation_tasks())
    print("Section 5.4.1 controlled representation evaluation")
    print(f"  decoder       : {DECODING_MODE}")
    print(f"  test profile  : {EVALUATION_PROFILE}")
    print(f"  device        : {DEVICE}")
    print(f"  namespace     : {RESULT_NAMESPACE or '(default)'}")
    print(f"  checkpoints   : {len(selected)}")
    print(
        "  checkpoint cap: "
        + (
            "none"
            if MAX_CHECKPOINT_ITERATION is None
            else str(MAX_CHECKPOINT_ITERATION)
        )
    )

    # Validate all selections before evaluating the first checkpoint. This
    # avoids doing partial work because of a typo in a later configured path.
    checkpoint_headers: Dict[Path, Tuple[str, int]] = {}
    for expected_method, task in selected:
        if not task.checkpoint.is_file():
            if SKIP_MISSING_CHECKPOINTS:
                continue
            raise FileNotFoundError(task.checkpoint)
        observed_method, checkpoint_iteration = _checkpoint_header(task.checkpoint)
        checkpoint_headers[task.checkpoint] = (
            observed_method,
            checkpoint_iteration,
        )
        if observed_method != expected_method:
            raise ValueError(
                "Checkpoint method mismatch: "
                f"expected {expected_method}, found {observed_method}"
            )
        if (
            MAX_CHECKPOINT_ITERATION is not None
            and checkpoint_iteration > MAX_CHECKPOINT_ITERATION
        ):
            raise ValueError(
                f"Checkpoint iteration {checkpoint_iteration} exceeds the "
                f"configured cap {MAX_CHECKPOINT_ITERATION}: {task.checkpoint}"
            )

    failures: List[str] = []
    evaluated = 0
    for expected_method, task in selected:
        print(f"\n[{task.task_id}]")
        print(f"  file       : {task.checkpoint}")
        if not task.checkpoint.is_file():
            message = f"missing checkpoint: {task.checkpoint}"
            if SKIP_MISSING_CHECKPOINTS:
                print(f"  skipped: {message}")
                continue
            raise FileNotFoundError(task.checkpoint)
        try:
            observed_method, checkpoint_iteration = checkpoint_headers[
                task.checkpoint
            ]
            if observed_method != expected_method:
                raise ValueError(
                    "Checkpoint method mismatch: "
                    f"expected {expected_method}, found {observed_method}"
                )
            print(f"  iteration  : {checkpoint_iteration}")
            if DECODING_MODE == "greedy":
                run_greedy(task, overwrite=OVERWRITE)
            else:
                run_sampling100(task, overwrite=OVERWRITE)
            _print_result(task)
            evaluated += 1
        except Exception as error:
            failures.append(
                f"{task.task_id}: {type(error).__name__}: {error}"
            )
            traceback.print_exc()
            if STOP_ON_ERROR:
                raise
    print(f"\nEvaluation complete: {evaluated}/{len(selected)} checkpoints")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
