from __future__ import annotations

"""Configurable Section 5.4.1 representation-baseline training entry."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import torch

import CPNModelV2.experiments.sec52_fjsp_performance.train_generic_compact_10x5 as training
from CPNModelV2.learning import (
    POLICY_AG_PPO,
    POLICY_MASKED_MLP,
    POLICY_PETRI_NET_GCN_PPO,
    POLICY_PNC_PPO,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

METHODS: Sequence[str] = (
    training.POLICY_GENERIC_COMPACT,
    POLICY_PETRI_NET_GCN_PPO,
    POLICY_PNC_PPO,
    POLICY_MASKED_MLP,
    POLICY_AG_PPO,
)
EXPERIMENTS: Sequence[Tuple[str, str]] = (
    ("SD1", "15x10"),
)
TRAINING_SEEDS = (9000, 9001, 9002, 9003, 9004)
RUN_GROUP = "formal_500x20"
RUN_LABEL = ""

# Same training protocol as the current Section 5.2 experiments.
TRAINING_PROFILE = "full"
ITERATIONS = 500
TRAINING_INSTANCES_PER_ITERATION = 20
VALIDATE_EVERY = 10
VALIDATION_INSTANCES = 100
CALIBRATION_INSTANCES = 8

# Shared model width. REFERENCE_LAYERS is interpreted as propagation depth for
# PNC/PN-GCN and Generic Compact V5; AG-PPO and Masked-MLP ignore it.
HIDDEN_DIM = 64
CATEGORY_DIM = 16
REFERENCE_LAYERS = 2
COMPETITION_LAYERS = 0
DROPOUT = 0.0

# Shared PPO and reward settings.
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 1024
GAMMA = 1.0
GAE_LAMBDA = 0.98
CLIP_RATIO = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5
REWARD_MODE = training.REWARD_REMAINING_WORK_POTENTIAL
REWARD_SCALE = 20.0
POTENTIAL_COEFFICIENT = 0.1

ROLLOUT_BATCH_BY_SIZE = {
    "10x5": 4,
    "20x5": 2,
    "15x10": 2,
    "20x10": 1,
}
PPO_MICROBATCH_BY_SIZE = {
    "10x5": 32,
    "20x5": 16,
    "15x10": 16,
    "20x10": 8,
}
EVALUATION_BATCH_BY_SIZE = {
    "10x5": 5,
    "20x5": 2,
    "15x10": 2,
    "20x10": 1,
}
SCENE_CONFIGS = {
    "10x5": (10, 5, 4, 6),
    "20x5": (20, 5, 4, 6),
    "15x10": (15, 10, 8, 12),
    "20x10": (20, 10, 8, 12),
}

# Same candidate-checkpoint policy as Sections 5.2 and 5.5.
CHECKPOINT_ARCHIVE_START = 50
CHECKPOINT_ARCHIVE_INTERVAL = 40
ARCHIVE_VALIDATION_IMPROVEMENTS = True
CHECKPOINT_ARCHIVE_DIRECTORY = "checkpoint_archive"

# Paths are relative to the FJSP-DRL-main-Wang project root.
VALIDATION_DATA_ROOT = Path("data/data_train_vali")
OUTPUT_ROOT = Path("CPNModelV2/results/sec54_representation/training")
WORKER_LOG_ROOT = Path("CPNModelV2/results/sec54_representation/worker_logs")

PROTOCOL_VERSION = "sec54_representation_training_v1"
INSTANCE_SEED_SCHEME = "sha256_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTO_RESUME = True
SKIP_COMPLETED = True
STOP_ON_ERROR = True
SAVE_EVERY_VALIDATION = True
FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT = True
PRINT_EVERY = 1
VALIDATE_INTERNAL_GRAPHS = False

# =============================================================================


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
SUPPORTED_DISTRIBUTIONS = ("SD1", "SD2")
SUPPORTED_METHODS = (
    POLICY_AG_PPO,
    POLICY_MASKED_MLP,
    POLICY_PNC_PPO,
    POLICY_PETRI_NET_GCN_PPO,
    training.POLICY_GENERIC_COMPACT,
)
METHOD_DIRECTORY = {
    POLICY_AG_PPO: "ag_ppo",
    POLICY_MASKED_MLP: "masked_mlp",
    POLICY_PNC_PPO: "pnc_ppo",
    POLICY_PETRI_NET_GCN_PPO: "petri_net_gcn_ppo",
    training.POLICY_GENERIC_COMPACT: "generic_compact_v5",
}


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


def _dataset_directory(distribution: str, size: str) -> str:
    return size if distribution == "SD1" else f"{size}+mix"


def configured_run(
    method: str,
    distribution: str,
    size: str,
    seed: int,
) -> training.TrainingConfig:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported Section 5.4 method: {method}")
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution}")
    if size not in SCENE_CONFIGS:
        raise ValueError(f"Unsupported size: {size}")
    jobs, machines, sd1_min_operations, sd1_max_operations = SCENE_CONFIGS[size]
    min_operations, max_operations = (
        (sd1_min_operations, sd1_max_operations)
        if distribution == "SD1"
        else (machines, machines)
    )
    output_root = project_path(OUTPUT_ROOT)
    if RUN_GROUP:
        output_root = output_root / RUN_GROUP
    output_dir = (
        output_root
        / distribution
        / size
        / METHOD_DIRECTORY[method]
        / f"seed_{seed}"
    ).resolve()
    validation_dataset = project_path(
        VALIDATION_DATA_ROOT
        / distribution
        / _dataset_directory(distribution, size)
    )
    base = training.configured_run()
    return replace(
        base,
        profile=TRAINING_PROFILE,
        policy_variant=method,
        seed=seed,
        device=DEVICE,
        output_dir=output_dir,
        validation_dataset=validation_dataset,
        iterations=ITERATIONS,
        training_instances_per_iteration=TRAINING_INSTANCES_PER_ITERATION,
        rollout_device_batch_size=min(
            ROLLOUT_BATCH_BY_SIZE[size],
            TRAINING_INSTANCES_PER_ITERATION,
        ),
        ppo_device_microbatch_size=PPO_MICROBATCH_BY_SIZE[size],
        validate_every=VALIDATE_EVERY,
        validation_instances=VALIDATION_INSTANCES,
        evaluation_batch_size=EVALUATION_BATCH_BY_SIZE[size],
        calibration_instances=CALIBRATION_INSTANCES,
        hidden_dim=HIDDEN_DIM,
        category_dim=CATEGORY_DIM,
        reference_layers=REFERENCE_LAYERS,
        competition_layers=COMPETITION_LAYERS,
        dropout=DROPOUT,
        reward_mode=REWARD_MODE,
        reward_scale=REWARD_SCALE,
        potential_coefficient=POTENTIAL_COEFFICIENT,
        learning_rate=LEARNING_RATE,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_ratio=CLIP_RATIO,
        value_coefficient=VALUE_COEFFICIENT,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        max_grad_norm=MAX_GRAD_NORM,
        auto_resume=AUTO_RESUME,
        save_every_validation=SAVE_EVERY_VALIDATION,
        print_every=PRINT_EVERY,
        checkpoint_archive_start=CHECKPOINT_ARCHIVE_START,
        checkpoint_archive_interval=CHECKPOINT_ARCHIVE_INTERVAL,
        archive_validation_improvements=ARCHIVE_VALIDATION_IMPROVEMENTS,
        checkpoint_archive_directory=CHECKPOINT_ARCHIVE_DIRECTORY,
        distribution=distribution,
        protocol_version=PROTOCOL_VERSION,
        instance_seed_scheme=INSTANCE_SEED_SCHEME,
        training_size=size,
        num_jobs=jobs,
        num_machines=machines,
        min_operations=min_operations,
        max_operations=max_operations,
        min_eligible_machines=1,
        max_eligible_machines=machines,
    )


def configured_tasks() -> List[Tuple[str, str, str, int, training.TrainingConfig]]:
    if not METHODS or not EXPERIMENTS or not TRAINING_SEEDS:
        raise ValueError("METHODS, EXPERIMENTS and TRAINING_SEEDS cannot be empty")
    keys = [
        (method, distribution, size, seed)
        for method in METHODS
        for distribution, size in EXPERIMENTS
        for seed in TRAINING_SEEDS
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/distribution/size/seed task")
    return [
        (*key, configured_run(*key))
        for key in keys
    ]


def _is_complete(config: training.TrainingConfig) -> bool:
    path = config.output_dir / "training_result.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    return (
        result.get("status") == "complete"
        and int(result.get("iterations", -1)) >= config.iterations
    )


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "task_id", "method", "distribution", "size", "seed", "status",
        "completed_iterations", "best_validation_makespan",
        "best_iteration", "output_dir", "error",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    training.FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT = (
        FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT
    )
    training.VALIDATE_INTERNAL_GRAPHS = VALIDATE_INTERNAL_GRAPHS
    tasks = configured_tasks()
    label = RUN_LABEL or "sec54_worker"
    status_path = project_path(WORKER_LOG_ROOT) / f"{label}_status.csv"
    print("Section 5.4.1 controlled representation training")
    print(f"  protocol   : {PROTOCOL_VERSION}")
    print(f"  methods    : {tuple(METHOD_DIRECTORY[item] for item in METHODS)}")
    print(f"  scenes     : {tuple(EXPERIMENTS)}")
    print(f"  iterations : {ITERATIONS}")
    print(f"  seeds      : {TRAINING_SEEDS}")
    print(f"  device     : {DEVICE}")
    print(f"  tasks      : {len(tasks)}")

    rows: List[Dict[str, Any]] = []
    failures = 0
    for method, distribution, size, seed, config in tasks:
        task_id = (
            f"{METHOD_DIRECTORY[method]}_{distribution.lower()}_{size}_seed{seed}"
        )
        row: Dict[str, Any] = {
            "task_id": task_id,
            "method": method,
            "distribution": distribution,
            "size": size,
            "seed": seed,
            "status": "pending",
            "completed_iterations": 0,
            "best_validation_makespan": "",
            "best_iteration": "",
            "output_dir": str(config.output_dir),
            "error": "",
        }
        rows.append(row)
        _write_rows(status_path, rows)
        print(f"\n[{task_id}]")
        if SKIP_COMPLETED and _is_complete(config):
            row["status"] = "skipped_complete"
            _write_rows(status_path, rows)
            print("  skipped: training already complete")
            continue
        try:
            row["status"] = "running"
            _write_rows(status_path, rows)
            result = training.run_training(config)
            row.update({
                "status": "complete",
                "completed_iterations": result["iterations"],
                "best_validation_makespan": result["best_validation_makespan"],
                "best_iteration": result["best_iteration"],
            })
        except Exception as error:
            failures += 1
            row.update({
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
            traceback.print_exc()
            if STOP_ON_ERROR:
                _write_rows(status_path, rows)
                raise
        _write_rows(status_path, rows)
    print(f"\nTraining worker complete: {len(tasks) - failures}/{len(tasks)}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
