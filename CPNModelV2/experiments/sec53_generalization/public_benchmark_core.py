from __future__ import annotations

"""Protocol and data helpers for Section 5.3.2 public benchmarks."""

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    CHECKPOINT_FORMAT,
    SUPPORTED_CHECKPOINT_FORMATS,
    write_csv_atomic,
    write_json_atomic,
)
from CPNModelV2.experiments.sec53_generalization.task_matrix import (
    CROSS_DISTRIBUTION_SOURCE_SIZES,
    EVALUATION_SEEDS,
    PROTOCOL_VERSION,
    PUBLIC_BENCHMARKS,
)
from CPNModelV2.integration.public_fjsp_dataset import PublicFJSPDataset


PUBLIC_PROTOCOL_VERSION = f"{PROTOCOL_VERSION}_cross_distribution_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_INSTANCES = {
    "Brandimarte": 10,
    "Hurink_rdata": 40,
    "Hurink_edata": 40,
    "Hurink_vdata": 40,
}


@dataclass(frozen=True)
class PublicBenchmarkEvaluationTask:
    source_size: str
    benchmark: str
    training_seed: int
    checkpoint_mode: str
    checkpoint: Path
    dataset_dir: Path
    bks_file: Path
    or_tools_file: Path
    output_dir: Path
    checkpoint_label: str = ""
    profile: str = "full"
    test_limit: int = 1
    greedy_batch_size: int = 1
    sampling_rollouts: int = 100
    sampling_batch_size: int = 1
    device: str = "cpu"
    source_distribution: str = "SD1"
    allow_diagnostic_seed: bool = False

    def validate(self) -> None:
        if self.source_distribution not in {"SD1", "SD2"}:
            raise ValueError(
                f"Unsupported source distribution: {self.source_distribution}"
            )
        if self.source_size not in CROSS_DISTRIBUTION_SOURCE_SIZES:
            raise ValueError(f"Unsupported source size: {self.source_size}")
        if self.benchmark not in PUBLIC_BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {self.benchmark}")
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
            if self.test_limit != EXPECTED_INSTANCES[self.benchmark]:
                raise ValueError(
                    "Full profile must evaluate the complete benchmark family"
                )
            if self.sampling_rollouts != 100:
                raise ValueError("The full protocol requires Sampling-100")
        for path in (
            self.checkpoint,
            self.dataset_dir,
            self.bks_file,
            self.or_tools_file,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

    @property
    def task_id(self) -> str:
        base = (
            f"cross_distribution_{self.source_distribution.lower()}_"
            f"train_{self.source_size}_"
            f"test_{self.benchmark.lower()}_seed_{self.training_seed}"
        )
        return f"{base}_{self.checkpoint_label}" if self.checkpoint_label else base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_records(task: PublicBenchmarkEvaluationTask) -> List[Any]:
    dataset = PublicFJSPDataset(task.dataset_dir)
    expected = EXPECTED_INSTANCES[task.benchmark]
    if len(dataset) != expected:
        raise ValueError(
            f"{task.dataset_dir} contains {len(dataset)} instances; "
            f"expected {expected}"
        )
    return list(dataset.records[: task.test_limit])


def bks_makespans(task: PublicBenchmarkEvaluationTask) -> np.ndarray:
    with task.bks_file.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["benchname"] == task.benchmark
        ]
    by_stem = {row["filename"].lower(): float(row["ub"]) for row in rows}
    records = ordered_records(task)
    missing = [
        record.path.stem for record in records
        if record.path.stem.lower() not in by_stem
    ]
    if missing:
        raise ValueError(f"BKS entries are missing for: {missing}")
    if len(rows) != EXPECTED_INSTANCES[task.benchmark]:
        raise ValueError(
            f"Unexpected BKS row count for {task.benchmark}: {len(rows)}"
        )
    return np.asarray(
        [by_stem[record.path.stem.lower()] for record in records],
        dtype=float,
    )


def or_tools_makespans(task: PublicBenchmarkEvaluationTask) -> np.ndarray:
    values = np.asarray(np.load(task.or_tools_file), dtype=float)
    if values.ndim == 2:
        values = values[:, 0]
    expected = EXPECTED_INSTANCES[task.benchmark]
    if values.ndim != 1 or values.shape[0] != expected:
        raise ValueError(
            f"Unexpected OR-Tools shape {values.shape}: {task.or_tools_file}"
        )
    return values[: task.test_limit]


def deterministic_evaluation_seed(
    task: PublicBenchmarkEvaluationTask,
    instance_id: str,
    decoder: str,
    rollout_id: int,
) -> int:
    canonical = "|".join((
        PUBLIC_PROTOCOL_VERSION,
        task.source_distribution,
        task.source_size,
        task.benchmark,
        str(task.training_seed),
        instance_id,
        decoder,
        str(rollout_id),
    ))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def checkpoint_audit(
    task: PublicBenchmarkEvaluationTask,
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
    expected = (
        task.source_distribution,
        task.source_size,
        task.training_seed,
    )
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


def initialize_output(task: PublicBenchmarkEvaluationTask) -> Dict[str, Any]:
    task.validate()
    records = ordered_records(task)
    bks_values = bks_makespans(task)
    or_values = or_tools_makespans(task)
    identity = {
        "evaluation_protocol_version": PUBLIC_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "profile": task.profile,
        "source_distribution": task.source_distribution,
        "source_size": task.source_size,
        "benchmark": task.benchmark,
        "training_seed": task.training_seed,
        "checkpoint_label": task.checkpoint_label,
        "checkpoint_mode": task.checkpoint_mode,
        "checkpoint": str(task.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(task.checkpoint),
        "dataset_dir": str(task.dataset_dir.resolve()),
        "bks_file": str(task.bks_file.resolve()),
        "bks_sha256": sha256_file(task.bks_file),
        "or_tools_file": str(task.or_tools_file.resolve()),
        "or_tools_sha256": sha256_file(task.or_tools_file),
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
    for index, (record, bks, or_value) in enumerate(
        zip(records, bks_values, or_values), start=1
    ):
        manifest_rows.append({
            "official_index": index,
            "benchmark": task.benchmark,
            "instance_id": record.path.name,
            "instance_path": str(
                record.path.resolve().relative_to(PROJECT_ROOT)
            ),
            "instance_sha256": sha256_file(record.path),
            "num_jobs": record.num_jobs,
            "num_machines": record.num_machines,
            "num_operations": record.num_operations,
            "bks_makespan": float(bks),
            "or_tools_makespan": float(or_value),
        })
    write_csv_atomic(
        task.output_dir / "dataset_manifest.csv",
        manifest_rows,
        (
            "official_index", "benchmark", "instance_id", "instance_path",
            "instance_sha256", "num_jobs", "num_machines",
            "num_operations", "bks_makespan", "or_tools_makespan",
        ),
    )
    return identity


def validate_resumed_rows(
    task: PublicBenchmarkEvaluationTask,
    rows: Sequence[Mapping[str, Any]],
    *,
    sampling: bool,
) -> None:
    known = {record.path.name for record in ordered_records(task)}
    identities = []
    for row in rows:
        observed = (
            row.get("source_distribution"),
            row.get("source_size"),
            row.get("benchmark"),
            int(row.get("training_seed", -1)),
        )
        expected = (
            task.source_distribution,
            task.source_size,
            task.benchmark,
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


def public_error_rows(
    task: PublicBenchmarkEvaluationTask,
    instance_id: str,
    decoder: str,
    rollout_id: int,
    errors: Sequence[str],
) -> List[Dict[str, Any]]:
    return [{
        "source_distribution": task.source_distribution,
        "source_size": task.source_size,
        "benchmark": task.benchmark,
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
