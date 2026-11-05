from __future__ import annotations

"""Measure Wang et al.'s released FJSP solver on the same local hardware.

This is a system-level benchmark.  Model loading and warm-up are excluded,
whereas instance parsing, environment initialization, greedy inference, and
environment updates are included in the end-to-end time.
"""

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
OUTPUT_ROOT = PACKAGE_ROOT / "results" / "sec57_runtime" / "wang"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SEED = 19000
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

SELECTED_TASK = "sd1_10x5_wang"
RUN_PROFILE = "formal"          # "smoke" or "formal"
RUN_ALL_TASKS = True             # Run all four paper settings sequentially.
OVERWRITE_EXISTING = False       # False resumes/skips completed measurements.
LIST_TASKS_ONLY = False
CHECK_CONFIGURATION_ONLY = False

# =============================================================================


@dataclass(frozen=True)
class WangSpec:
    key: str
    distribution: str
    size: str
    data_directory: Path
    checkpoint: Path


TASKS: Dict[str, WangSpec] = {
    "sd1_10x5_wang": WangSpec(
        key="sd1_10x5_wang",
        distribution="SD1",
        size="10x5",
        data_directory=PROJECT_ROOT / "data" / "SD1" / "10x5",
        checkpoint=PROJECT_ROOT / "trained_network" / "SD1" / "10x5.pth",
    ),
    "sd1_15x10_wang": WangSpec(
        key="sd1_15x10_wang",
        distribution="SD1",
        size="15x10",
        data_directory=PROJECT_ROOT / "data" / "SD1" / "15x10",
        checkpoint=PROJECT_ROOT / "trained_network" / "SD1" / "15x10.pth",
    ),
    "sd2_20x5_wang": WangSpec(
        key="sd2_20x5_wang",
        distribution="SD2",
        size="20x5",
        data_directory=PROJECT_ROOT / "data" / "SD2" / "20x5+mix",
        checkpoint=(
            PROJECT_ROOT / "trained_network" / "SD2" / "20x5+mix.pth"
        ),
    ),
    "sd2_20x10_wang": WangSpec(
        key="sd2_20x10_wang",
        distribution="SD2",
        size="20x10",
        data_directory=PROJECT_ROOT / "data" / "SD2" / "20x10+mix",
        checkpoint=(
            PROJECT_ROOT / "trained_network" / "SD2" / "20x10+mix.pth"
        ),
    ),
}


CSV_FIELDS = (
    "system",
    "task",
    "profile",
    "method",
    "distribution",
    "size",
    "test_seed",
    "repeat",
    "instance_index",
    "instance_id",
    "makespan",
    "decision_count",
    "parameters",
    "initialization_s",
    "solve_s",
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


def _wang_modules():
    """Import the released code without letting params.py consume our CLI."""
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        from params import configs
        from data_utils import text_to_matrix
        from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
        from model.main_model import DANIEL
        from common_utils import greedy_select_action, setup_seed
    finally:
        sys.argv = original_argv
    return (
        configs,
        text_to_matrix,
        FJSPEnvForSameOpNums,
        DANIEL,
        greedy_select_action,
        setup_seed,
    )


def _ordered_paths(directory: Path) -> List[Path]:
    paths = [path for path in directory.rglob("*") if path.is_file()]
    paths.sort(key=lambda path: int(re.findall(r"\d+", path.name)[0]))
    paths.sort(key=lambda path: int(re.findall(r"\d+", path.name)[-1]))
    return paths


def _synchronize() -> None:
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(DEVICE))


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


def _model_config(configs) -> None:
    configs.device = DEVICE
    configs.fea_j_input_dim = 10
    configs.fea_m_input_dim = 8
    configs.dropout_prob = 0.0
    configs.num_heads_OAB = [4, 4]
    configs.num_heads_MAB = [4, 4]
    configs.layer_fea_output_dim = [32, 8]
    configs.num_mlp_layers_actor = 3
    configs.hidden_dim_actor = 64
    configs.num_mlp_layers_critic = 3
    configs.hidden_dim_critic = 64


def _solve_one(
    path: Path,
    policy,
    text_to_matrix,
    environment_class,
    greedy_select_action,
) -> tuple[float, int, float, float, float]:
    _synchronize()
    total_started = time.perf_counter()
    initialization_started = time.perf_counter()
    lines = path.read_text(encoding="utf-8").splitlines(True)
    job_length, op_pt = text_to_matrix(lines)
    n_jobs = int(job_length.shape[0])
    n_operations, n_machines = op_pt.shape
    del n_operations
    environment = environment_class(n_j=n_jobs, n_m=n_machines)
    state = environment.set_initial_data([job_length], [op_pt])
    _synchronize()
    initialization_seconds = time.perf_counter() - initialization_started

    solve_started = time.perf_counter()
    decisions = 0
    while True:
        with torch.no_grad():
            probabilities, _ = policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
        action = greedy_select_action(probabilities)
        state, _, done = environment.step(actions=action.cpu().numpy())
        decisions += 1
        if bool(np.asarray(done).all()):
            break
    _synchronize()
    solve_seconds = time.perf_counter() - solve_started
    total_seconds = time.perf_counter() - total_started
    return (
        float(environment.current_makespan[0]),
        decisions,
        initialization_seconds,
        solve_seconds,
        total_seconds,
    )


def run(spec: WangSpec, profile: str, overwrite: bool) -> Path:
    warmup_count, benchmark_count, repeats = _profile_values(profile)
    paths = _ordered_paths(spec.data_directory)
    required = warmup_count + benchmark_count
    if len(paths) < required:
        raise ValueError(
            f"{spec.data_directory} has {len(paths)} files; {required} required"
        )
    warmup_paths = paths[:warmup_count]
    benchmark_paths = paths[warmup_count:required]

    (
        configs,
        text_to_matrix,
        environment_class,
        policy_class,
        greedy_select_action,
        setup_seed,
    ) = _wang_modules()
    _model_config(configs)
    setup_seed(TEST_SEED)
    policy = policy_class(configs).to(torch.device(DEVICE))
    try:
        state_dict = torch.load(
            spec.checkpoint,
            map_location=torch.device(DEVICE),
            weights_only=False,
        )
    except TypeError:
        state_dict = torch.load(
            spec.checkpoint, map_location=torch.device(DEVICE)
        )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    parameters = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )

    output_dir = OUTPUT_ROOT / profile / spec.key
    output_path = output_dir / "per_instance_timing.csv"
    rows = [] if overwrite else _read_rows(output_path)
    completed = {
        (int(row["repeat"]), str(row["instance_id"])) for row in rows
    }

    print("Section 5.7 Wang runtime benchmark")
    print(f"  task       : {spec.key}")
    print(f"  scenario   : {spec.distribution} {spec.size}")
    print(f"  device     : {DEVICE}")
    print(f"  warm-up    : {warmup_count} instances")
    print(f"  benchmark  : {benchmark_count} instances x {repeats} repeats")
    print(f"  checkpoint : {spec.checkpoint}")
    print("  warming up ...", flush=True)
    for path in warmup_paths:
        _solve_one(
            path,
            policy,
            text_to_matrix,
            environment_class,
            greedy_select_action,
        )

    for repeat in range(1, repeats + 1):
        for instance_index, path in enumerate(benchmark_paths, start=1):
            key = (repeat, path.name)
            if key in completed:
                continue
            makespan, decisions, initialization, solve, total = _solve_one(
                path,
                policy,
                text_to_matrix,
                environment_class,
                greedy_select_action,
            )
            values = (initialization, solve, total)
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Invalid timing values for {path.name}: {values}")
            row = {
                "system": "wang_native",
                "task": spec.key,
                "profile": profile,
                "method": "Wang et al.",
                "distribution": spec.distribution,
                "size": spec.size,
                "test_seed": TEST_SEED,
                "repeat": repeat,
                "instance_index": instance_index,
                "instance_id": path.name,
                "makespan": makespan,
                "decision_count": decisions,
                "parameters": parameters,
                "initialization_s": initialization,
                "solve_s": solve,
                "end_to_end_s": total,
                "expected_instances_per_repeat": benchmark_count,
                "expected_repeats": repeats,
            }
            rows.append(row)
            rows.sort(key=lambda item: (
                int(item["repeat"]), int(item["instance_index"])
            ))
            _write_rows_atomic(output_path, rows)
            completed.add(key)
            print(
                f"  repeat {repeat}/{repeats}, "
                f"instance {instance_index}/{benchmark_count}: "
                f"{total:.4f} s",
                flush=True,
            )

    metadata = {
        "protocol": "sec57_wang_runtime_v1",
        "task": spec.key,
        "profile": profile,
        "method": "Wang et al.",
        "distribution": spec.distribution,
        "size": spec.size,
        "device": DEVICE,
        "batch_size": 1,
        "greedy_decoding": True,
        "test_seed": TEST_SEED,
        "warmup_instances": warmup_count,
        "benchmark_instances": benchmark_count,
        "repeats": repeats,
        "parameters": parameters,
        "checkpoint": str(spec.checkpoint),
        "data_directory": str(spec.data_directory),
        "output": str(output_path),
    }
    _write_json_atomic(output_dir / "benchmark_metadata.json", metadata)
    return output_path


def main() -> int:
    if RUN_PROFILE not in {"smoke", "formal"}:
        raise ValueError("RUN_PROFILE must be 'smoke' or 'formal'")
    if LIST_TASKS_ONLY:
        for key, spec in TASKS.items():
            print(f"{key:22s}  {spec.distribution} {spec.size}")
        return 0
    if RUN_ALL_TASKS:
        selected_keys = tuple(TASKS)
    else:
        if SELECTED_TASK not in TASKS:
            raise ValueError(
                f"Unknown SELECTED_TASK={SELECTED_TASK!r}; "
                f"choose one of {tuple(TASKS)}"
            )
        selected_keys = (SELECTED_TASK,)

    for key in selected_keys:
        spec = TASKS[key]
        if not spec.data_directory.is_dir():
            raise FileNotFoundError(spec.data_directory)
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(spec.checkpoint)
        print(f"checkpoint: {spec.checkpoint}")
        print(f"dataset   : {spec.data_directory}")
        if CHECK_CONFIGURATION_ONLY:
            print(f"{key}: configuration check passed.")
            continue
        output = run(spec, RUN_PROFILE, OVERWRITE_EXISTING)
        print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

