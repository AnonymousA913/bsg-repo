from __future__ import annotations

"""Configurable Section 5.5 ablation-training entry.

Edit only the USER CONFIGURATION block. Each process is defined by explicit
scene, variant, and seed selections; no machine identity is part of the
experimental protocol.
"""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

import csv
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

import torch

import CPNModelV2.experiments.sec52_fjsp_performance.train_generic_compact_10x5 as base_training
import CPNModelV2.experiments.sec55_ablation.training_core as training_core
from CPNModelV2.experiments.sec55_ablation.training_core import (
    ABLATION_VARIANTS as SUPPORTED_ABLATION_VARIANTS,
    AblationTrainingConfig,
    run_ablation_training,
)
from CPNModelV2.learning.generic_compact_ablation_policy import (
    FLAT_BINDING_VARIANT,
    NO_REFERENCE_VARIANT,
    NO_REFERENCE_FLAT_BINDING_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
    normalize_ablation_variant,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Each process runs the Cartesian product of these scenes, variants, and seeds.
# Copy this entry script and select disjoint combinations for parallel runs.
EXPERIMENTS: Tuple[Tuple[str, str], ...] = (
    ("SD1", "10x5"),
)
ABLATIONS = (
    NO_REFERENCE_VARIANT,
    FLAT_BINDING_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
)
# Five independent training seeds used by the formal Section 5.5 protocol.
TRAINING_SEEDS = (9000, 9001, 9002, 9003, 9004)
RUN_GROUP = "formal_500x20"
RUN_LABEL = ""

# Main training budget: matched to the current Section 5.2 protocol.
TRAINING_PROFILE = "full"
ITERATIONS = 500
TRAINING_INSTANCES_PER_ITERATION = 20
VALIDATE_EVERY = 10
VALIDATION_INSTANCES = 100
CALIBRATION_INSTANCES = 8

# Model architecture. The ablation changes only the selected semantic module.
POLICY_VARIANT = base_training.POLICY_GENERIC_COMPACT
HIDDEN_DIM = 64
CATEGORY_DIM = 16
REFERENCE_LAYERS = 2
COMPETITION_LAYERS = 0
DROPOUT = 0.0

# PPO and reward.
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 1024
GAMMA = 1.0
GAE_LAMBDA = 0.98
CLIP_RATIO = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5
REWARD_MODE = base_training.REWARD_REMAINING_WORK_POTENTIAL
REWARD_SCALE = 20.0
POTENTIAL_COEFFICIENT = 0.1

# Per-size memory controls.
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

# Dataset dimensions and SD1 operation-count ranges.
SCENE_CONFIGS = {
    "10x5": (10, 5, 4, 6),
    "20x5": (20, 5, 4, 6),
    "15x10": (15, 10, 8, 12),
    "20x10": (20, 10, 8, 12),
}

# Candidate-checkpoint policy.
CHECKPOINT_ARCHIVE_START = 50
CHECKPOINT_ARCHIVE_INTERVAL = 40
ARCHIVE_VALIDATION_IMPROVEMENTS = True
CHECKPOINT_ARCHIVE_DIRECTORY = "checkpoint_archive"

# All editable paths are relative to the project root.
VALIDATION_DATA_ROOT = Path("data/data_train_vali")
RESERVED_TEST_DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("CPNModelV2/results/sec55_ablation/training")
WORKER_LOG_ROOT = Path("CPNModelV2/results/sec55_ablation/worker_logs")

PROTOCOL_VERSION = "sec55_ablation_training_v1"
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


def validation_dataset(distribution: str, size: str) -> Path:
    return project_path(
        VALIDATION_DATA_ROOT
        / distribution
        / _dataset_directory(distribution, size)
    )


def reserved_test_dataset(distribution: str, size: str) -> Path:
    return project_path(
        RESERVED_TEST_DATA_ROOT
        / distribution
        / _dataset_directory(distribution, size)
    )


def _output_dir(
    distribution: str,
    size: str,
    variant: str,
    seed: int,
) -> Path:
    root = project_path(OUTPUT_ROOT)
    if RUN_GROUP:
        root = root / RUN_GROUP
    return (root / distribution / size / variant / f"seed_{seed}").resolve()


def _task_id(distribution: str, size: str, variant: str, seed: int) -> str:
    short_variant = {
        NO_REFERENCE_VARIANT: "noref",
        NO_REFERENCE_FLAT_BINDING_VARIANT: "noref_flatbind",
        NO_RELATION_DIRECTION_VARIANT: "noreldir",
        FLAT_BINDING_VARIANT: "flatbind",
    }[variant]
    group = RUN_GROUP or "default"
    return f"{group}_{distribution.lower()}_{size}_{short_variant}_seed{seed}"


def configured_run(
    distribution: str,
    size: str,
    variant: str,
    seed: int,
) -> AblationTrainingConfig:
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution}")
    if size not in SCENE_CONFIGS:
        raise ValueError(f"Unsupported size: {size}")
    variant = normalize_ablation_variant(variant)
    if variant not in SUPPORTED_ABLATION_VARIANTS:
        raise ValueError(f"Unsupported Section 5.5 ablation: {variant}")

    jobs, machines, sd1_min_operations, sd1_max_operations = SCENE_CONFIGS[size]
    if distribution == "SD1":
        min_operations = sd1_min_operations
        max_operations = sd1_max_operations
    else:
        min_operations = machines
        max_operations = machines

    requested_device = DEVICE
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"

    base = base_training.configured_run()
    values = dict(base.__dict__)
    values.update({
        "profile": TRAINING_PROFILE,
        "policy_variant": POLICY_VARIANT,
        "seed": seed,
        "device": requested_device,
        "output_dir": _output_dir(distribution, size, variant, seed),
        "validation_dataset": validation_dataset(distribution, size),
        "iterations": ITERATIONS,
        "training_instances_per_iteration": TRAINING_INSTANCES_PER_ITERATION,
        "rollout_device_batch_size": min(
            ROLLOUT_BATCH_BY_SIZE[size],
            TRAINING_INSTANCES_PER_ITERATION,
        ),
        "ppo_device_microbatch_size": PPO_MICROBATCH_BY_SIZE[size],
        "validate_every": VALIDATE_EVERY,
        "validation_instances": VALIDATION_INSTANCES,
        "evaluation_batch_size": EVALUATION_BATCH_BY_SIZE[size],
        "calibration_instances": CALIBRATION_INSTANCES,
        "hidden_dim": HIDDEN_DIM,
        "category_dim": CATEGORY_DIM,
        "reference_layers": REFERENCE_LAYERS,
        "competition_layers": COMPETITION_LAYERS,
        "dropout": DROPOUT,
        "reward_mode": REWARD_MODE,
        "reward_scale": REWARD_SCALE,
        "potential_coefficient": POTENTIAL_COEFFICIENT,
        "learning_rate": LEARNING_RATE,
        "update_epochs": UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_ratio": CLIP_RATIO,
        "value_coefficient": VALUE_COEFFICIENT,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "max_grad_norm": MAX_GRAD_NORM,
        "auto_resume": AUTO_RESUME,
        "save_every_validation": SAVE_EVERY_VALIDATION,
        "print_every": PRINT_EVERY,
        "checkpoint_archive_start": CHECKPOINT_ARCHIVE_START,
        "checkpoint_archive_interval": CHECKPOINT_ARCHIVE_INTERVAL,
        "archive_validation_improvements": ARCHIVE_VALIDATION_IMPROVEMENTS,
        "checkpoint_archive_directory": CHECKPOINT_ARCHIVE_DIRECTORY,
        "distribution": distribution,
        "protocol_version": PROTOCOL_VERSION,
        "instance_seed_scheme": INSTANCE_SEED_SCHEME,
        "training_size": size,
        "num_jobs": jobs,
        "num_machines": machines,
        "min_operations": min_operations,
        "max_operations": max_operations,
        "min_eligible_machines": 1,
        "max_eligible_machines": machines,
    })
    return AblationTrainingConfig(
        **values,
        ablation_variant=variant,
        reserved_test_dataset=reserved_test_dataset(distribution, size),
    )


ConfiguredTask = Tuple[str, str, str, int, AblationTrainingConfig]


def configured_tasks() -> List[ConfiguredTask]:
    if not EXPERIMENTS or not ABLATIONS or not TRAINING_SEEDS:
        raise ValueError(
            "EXPERIMENTS, ABLATIONS, and TRAINING_SEEDS must not be empty"
        )
    keys = [
        (distribution, size, variant, seed)
        for distribution, size in EXPERIMENTS
        for variant in ABLATIONS
        for seed in TRAINING_SEEDS
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate distribution/size/variant/seed training task")
    return [
        (
            distribution,
            size,
            variant,
            seed,
            configured_run(distribution, size, variant, seed),
        )
        for distribution, size, variant, seed in keys
    ]


def _is_complete(output_dir: Path, target_iterations: int) -> bool:
    result_path = output_dir / "training_result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return (
        result.get("status") == "complete"
        and int(result.get("iterations", -1)) >= target_iterations
    )


def _hardware_metadata() -> Dict[str, Any]:
    cuda_name = ""
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": cuda_name,
        "requested_device": DEVICE,
    }


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "task_id", "distribution", "size", "variant", "seed", "profile",
        "status", "completed_iterations", "target_iterations",
        "best_validation_makespan", "best_iteration",
        "elapsed_seconds_this_call", "output_dir", "error",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _log_stem(tasks: List[ConfiguredTask]) -> str:
    if RUN_LABEL:
        return RUN_LABEL
    distribution, size, variant, seed, _ = tasks[0]
    suffix = _task_id(distribution, size, variant, seed)
    return suffix if len(tasks) == 1 else f"{suffix}_plus_{len(tasks) - 1}"


def main() -> None:
    training_core.VALIDATE_INTERNAL_GRAPHS = VALIDATE_INTERNAL_GRAPHS
    training_core.FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT = (
        FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT
    )
    tasks = configured_tasks()
    metadata = _hardware_metadata()
    log_root = project_path(WORKER_LOG_ROOT)
    log_stem = _log_stem(tasks)
    worker_log = log_root / f"{log_stem}_training_status.csv"
    metadata_path = log_root / f"{log_stem}_hardware.json"
    _write_json(metadata_path, metadata)

    print("Section 5.5 BSG-RL ablation training")
    print(f"  protocol   : {PROTOCOL_VERSION}")
    print(f"  group      : {RUN_GROUP or 'default'}")
    print(f"  profile    : {TRAINING_PROFILE}")
    print(f"  iterations : {ITERATIONS}")
    print(f"  instances  : {TRAINING_INSTANCES_PER_ITERATION}/iteration")
    print(f"  archive    : after {CHECKPOINT_ARCHIVE_START}, every "
          f"{CHECKPOINT_ARCHIVE_INTERVAL} iterations")
    print(f"  seeds      : {TRAINING_SEEDS}")
    print(f"  device     : {DEVICE}")
    print(f"  tasks      : {len(tasks)}")

    rows: List[Dict[str, Any]] = []
    for distribution, size, variant, seed, config in tasks:
        task_id = _task_id(distribution, size, variant, seed)
        print(f"    - {task_id}")
        row: Dict[str, Any] = {
            "task_id": task_id,
            "distribution": distribution,
            "size": size,
            "variant": variant,
            "seed": seed,
            "profile": TRAINING_PROFILE,
            "status": "pending",
            "completed_iterations": 0,
            "target_iterations": config.iterations,
            "best_validation_makespan": "",
            "best_iteration": "",
            "elapsed_seconds_this_call": 0.0,
            "output_dir": str(config.output_dir),
            "error": "",
        }
        rows.append(row)
        _write_csv(worker_log, rows)

        if SKIP_COMPLETED and _is_complete(config.output_dir, config.iterations):
            result = json.loads(
                (config.output_dir / "training_result.json").read_text(
                    encoding="utf-8"
                )
            )
            row.update({
                "status": "skipped_complete",
                "completed_iterations": result["iterations"],
                "best_validation_makespan": result[
                    "best_validation_makespan"
                ],
                "best_iteration": result["best_iteration"],
            })
            _write_csv(worker_log, rows)
            continue

        started = time.perf_counter()
        row["status"] = "running"
        _write_csv(worker_log, rows)
        try:
            result = run_ablation_training(config)
            row.update({
                "status": "complete",
                "completed_iterations": result["iterations"],
                "best_validation_makespan": result[
                    "best_validation_makespan"
                ],
                "best_iteration": result["best_iteration"],
                "elapsed_seconds_this_call": time.perf_counter() - started,
            })
            _write_json(config.output_dir / "execution_metadata.json", {
                **metadata,
                "task_id": task_id,
                "group": RUN_GROUP,
                "profile": TRAINING_PROFILE,
                "distribution": distribution,
                "size": size,
                "variant": variant,
                "seed": seed,
                "output_relative_to_project": str(
                    config.output_dir.relative_to(PROJECT_ROOT)
                ),
            })
        except KeyboardInterrupt:
            row.update({
                "status": "interrupted",
                "elapsed_seconds_this_call": time.perf_counter() - started,
                "error": "KeyboardInterrupt",
            })
            _write_csv(worker_log, rows)
            raise
        except Exception as error:
            row.update({
                "status": "failed",
                "elapsed_seconds_this_call": time.perf_counter() - started,
                "error": "".join(
                    traceback.format_exception_only(type(error), error)
                ).strip(),
            })
            _write_csv(worker_log, rows)
            traceback.print_exc()
            if STOP_ON_ERROR:
                raise
        finally:
            _write_csv(worker_log, rows)

    print(f"Training run complete: {worker_log}")


if __name__ == "__main__":
    main()
