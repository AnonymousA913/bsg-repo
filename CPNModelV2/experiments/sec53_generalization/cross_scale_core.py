from __future__ import annotations

"""Protocol and data helpers for Section 5.3.1 cross-scale evaluation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    CHECKPOINT_FORMAT,
    SUPPORTED_CHECKPOINT_FORMATS,
    write_csv_atomic,
    write_json_atomic,
)
from CPNModelV2.experiments.sec53_generalization.task_matrix import (
    CROSS_SCALE_DISTRIBUTIONS,
    CROSS_SCALE_SOURCE_SIZES,
    CROSS_SCALE_TARGET_SIZES,
    EVALUATION_SEEDS,
    PROTOCOL_VERSION,
)
from CPNModelV2.integration.public_fjsp_dataset import PublicFJSPDataset


CROSS_SCALE_PROTOCOL_VERSION = f"{PROTOCOL_VERSION}_cross_scale_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CrossScaleEvaluationTask:
    distribution: str
    source_size: str
    target_size: str
    training_seed: int
    checkpoint_mode: str
    checkpoint: Path
    test_dataset: Path
    reference_file: Path
    output_dir: Path
    checkpoint_label: str = ""
    profile: str = "full"
    test_limit: int = 100
    greedy_batch_size: int = 1
    sampling_rollouts: int = 100
    sampling_batch_size: int = 1
    device: str = "cpu"
    allow_diagnostic_seed: bool = False

    def validate(self) -> None:
        if self.distribution not in CROSS_SCALE_DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        if self.source_size not in CROSS_SCALE_SOURCE_SIZES:
            raise ValueError(f"Unsupported source size: {self.source_size}")
        if self.target_size not in CROSS_SCALE_TARGET_SIZES:
            raise ValueError(f"Unsupported target size: {self.target_size}")
        if (
            self.training_seed not in EVALUATION_SEEDS
            and not self.allow_diagnostic_seed
        ):
            raise ValueError(f"Unsupported training seed: {self.training_seed}")
        if self.checkpoint_mode not in {"pilot", "formal", "selected"}:
            raise ValueError(
                "checkpoint_mode must be pilot, formal, or selected"
            )
        if self.profile not in {"smoke", "full"}:
            raise ValueError("profile must be smoke or full")
        for name, value in {
            "test_limit": self.test_limit,
            "greedy_batch_size": self.greedy_batch_size,
            "sampling_rollouts": self.sampling_rollouts,
            "sampling_batch_size": self.sampling_batch_size,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.profile == "full":
            if self.test_limit != 100:
                raise ValueError("The full protocol requires 100 test instances")
            if self.sampling_rollouts != 100:
                raise ValueError("The full protocol requires Sampling-100")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.test_dataset.is_dir():
            raise FileNotFoundError(self.test_dataset)
        if not self.reference_file.is_file():
            raise FileNotFoundError(self.reference_file)

    @property
    def task_id(self) -> str:
        base = (
            f"cross_scale_{self.distribution.lower()}_"
            f"train_{self.source_size}_test_{self.target_size}_"
            f"seed_{self.training_seed}"
        )
        return f"{base}_{self.checkpoint_label}" if self.checkpoint_label else base

    @property
    def target_dimensions(self) -> Tuple[int, int]:
        jobs, machines = self.target_size.split("x")
        return int(jobs), int(machines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wang_order_key(path: Path) -> Tuple[int, int, str]:
    numbers = [int(value) for value in re.findall(r"\d+", path.name)]
    if not numbers:
        raise ValueError(f"Instance name contains no numeric index: {path.name}")
    # Equivalent to the stable two-pass ordering used by Wang's release code.
    return numbers[-1], numbers[0], path.name


def ordered_test_paths(task: CrossScaleEvaluationTask) -> List[Path]:
    jobs, machines = task.target_dimensions
    dataset = PublicFJSPDataset(
        task.test_dataset,
        expected_jobs=jobs,
        expected_machines=machines,
    )
    paths = sorted(dataset.paths, key=wang_order_key)
    if len(paths) != 100:
        raise ValueError(
            f"{task.test_dataset} contains {len(paths)} instances; "
            "the Wang protocol requires exactly 100"
        )
    return paths[: task.test_limit]


def reference_makespans(task: CrossScaleEvaluationTask) -> np.ndarray:
    values = np.asarray(np.load(task.reference_file), dtype=float)
    if values.ndim == 2:
        values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != 100:
        raise ValueError(
            f"Unexpected reference shape {values.shape}: {task.reference_file}"
        )
    return values[: task.test_limit]


def deterministic_evaluation_seed(
    task: CrossScaleEvaluationTask,
    instance_id: str,
    decoder: str,
    rollout_id: int,
) -> int:
    canonical = "|".join((
        CROSS_SCALE_PROTOCOL_VERSION,
        task.distribution,
        task.source_size,
        task.target_size,
        str(task.training_seed),
        instance_id,
        decoder,
        str(rollout_id),
    ))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def checkpoint_audit(
    task: CrossScaleEvaluationTask,
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    if checkpoint.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint.get('format')!r}"
        )
    training = checkpoint["training_config"]
    observed = (
        str(training.get("distribution", "SD1")),
        str(training.get("training_size", "")),
        int(training.get("seed", checkpoint.get("seed", -1))),
    )
    expected = (task.distribution, task.source_size, task.training_seed)
    if observed != expected:
        raise ValueError(
            f"Checkpoint/task mismatch: expected={expected}, observed={observed}"
        )
    return {
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "best_validation_makespan": float(
            checkpoint["best_validation_makespan"]
        ),
        "training_protocol_version": training.get(
            "protocol_version", "legacy"
        ),
        "schema_sha256": _mapping_sha256(checkpoint["schema"]),
        "normalizer_sha256": _mapping_sha256(checkpoint["normalizer"]),
    }


def initialize_output(task: CrossScaleEvaluationTask) -> Dict[str, Any]:
    task.validate()
    paths = ordered_test_paths(task)
    references = reference_makespans(task)
    identity = {
        "evaluation_protocol_version": CROSS_SCALE_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "profile": task.profile,
        "distribution": task.distribution,
        "source_size": task.source_size,
        "target_size": task.target_size,
        "training_seed": task.training_seed,
        "checkpoint_label": task.checkpoint_label,
        "checkpoint_mode": task.checkpoint_mode,
        "checkpoint": str(task.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(task.checkpoint),
        "test_dataset": str(task.test_dataset.resolve()),
        "reference_file": str(task.reference_file.resolve()),
        "reference_sha256": sha256_file(task.reference_file),
        "test_limit": task.test_limit,
        "sampling_rollouts": task.sampling_rollouts,
    }
    if task.allow_diagnostic_seed:
        identity["diagnostic_nonprotocol_seed"] = True
    config_path = task.output_dir / "evaluation_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError(
                "Existing output directory belongs to a different evaluation: "
                f"{task.output_dir}"
            )
    else:
        write_json_atomic(config_path, identity)

    manifest_rows = []
    for index, (path, reference) in enumerate(zip(paths, references), start=1):
        manifest_rows.append({
            "official_index": index,
            "distribution": task.distribution,
            "target_size": task.target_size,
            "instance_id": path.name,
            "instance_path": str(path.resolve().relative_to(PROJECT_ROOT)),
            "instance_sha256": sha256_file(path),
            "reference_makespan": float(reference),
        })
    write_csv_atomic(
        task.output_dir / "dataset_manifest.csv",
        manifest_rows,
        (
            "official_index", "distribution", "target_size", "instance_id",
            "instance_path", "instance_sha256", "reference_makespan",
        ),
    )
    return identity


def validate_resumed_rows(
    task: CrossScaleEvaluationTask,
    rows: Sequence[Mapping[str, Any]],
    *,
    sampling: bool,
) -> None:
    known = {path.name for path in ordered_test_paths(task)}
    identities = []
    for row in rows:
        observed = (
            row.get("distribution"),
            row.get("source_size"),
            row.get("target_size"),
            int(row.get("training_seed", -1)),
        )
        expected = (
            task.distribution,
            task.source_size,
            task.target_size,
            task.training_seed,
        )
        if observed != expected:
            raise ValueError(
                f"Resumed row does not match task: {observed} != {expected}"
            )
        instance_id = str(row["instance_id"])
        if instance_id not in known:
            raise ValueError(f"Unknown resumed instance: {instance_id}")
        identity: Any = instance_id
        if sampling:
            rollout_id = int(row["rollout_id"])
            if not 1 <= rollout_id <= task.sampling_rollouts:
                raise ValueError(f"Invalid resumed rollout id: {rollout_id}")
            identity = (instance_id, rollout_id)
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise ValueError("Resumed output contains duplicate rows")


def cross_scale_error_rows(
    task: CrossScaleEvaluationTask,
    instance_id: str,
    decoder: str,
    rollout_id: int,
    errors: Sequence[str],
) -> List[Dict[str, Any]]:
    return [{
        "distribution": task.distribution,
        "source_size": task.source_size,
        "target_size": task.target_size,
        "training_seed": task.training_seed,
        "instance_id": instance_id,
        "decoder": decoder,
        "rollout_id": rollout_id,
        "error": error,
    } for error in errors]


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
