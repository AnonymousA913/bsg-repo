from __future__ import annotations

"""Preflight audit for the Section 5.5 ablation protocol."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

import hashlib
from pathlib import Path
from typing import Dict, Tuple

from CPNModelV2.experiments.sec52_fjsp_performance.train_generic_compact_10x5 import (
    _training_instance_seed,
)
from CPNModelV2.experiments.sec55_ablation import train_worker
from CPNModelV2.integration.equivalence_core import CanonicalInstance


PREFLIGHT_EXPERIMENTS = (("SD1", "10x5"), ("SD2", "20x10"))
PREFLIGHT_ABLATIONS = train_worker.SUPPORTED_ABLATION_VARIANTS
PREFLIGHT_SEEDS = (9000, 9001, 9002, 9003, 9004)


def _fjs_paths(directory: Path) -> Tuple[Path, ...]:
    return tuple(sorted(directory.glob("*.fjs")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_dataset_pair(distribution: str, size: str) -> Dict[str, int]:
    validation = train_worker.validation_dataset(distribution, size)
    test = train_worker.reserved_test_dataset(distribution, size)
    validation_paths = _fjs_paths(validation)
    test_paths = _fjs_paths(test)
    if len(validation_paths) != 100:
        raise ValueError(
            f"{distribution}-{size} validation set has {len(validation_paths)} "
            "FJS files; expected exactly 100"
        )
    if len(test_paths) != 100:
        raise ValueError(
            f"{distribution}-{size} test set has {len(test_paths)} FJS files; "
            "expected exactly 100"
        )

    expected_jobs, expected_machines = map(int, size.split("x"))
    for path in (*validation_paths, *test_paths):
        instance = CanonicalInstance.from_fjs(path)
        if (
            instance.num_jobs != expected_jobs
            or instance.num_machines != expected_machines
        ):
            raise ValueError(
                f"Dataset dimension mismatch: {path} -> "
                f"{instance.num_jobs}x{instance.num_machines}, expected {size}"
            )

    validation_hashes = {_sha256(path) for path in validation_paths}
    test_hashes = {_sha256(path) for path in test_paths}
    overlap = validation_hashes & test_hashes
    if overlap:
        raise ValueError(
            f"{distribution}-{size} validation/test leakage detected: "
            f"{len(overlap)} byte-identical FJS files"
        )
    return {
        "validation_files": len(validation_paths),
        "test_files": len(test_paths),
        "overlap": 0,
    }


def _audit_generation_seeds() -> int:
    per_run = (
        train_worker.ITERATIONS
        * train_worker.TRAINING_INSTANCES_PER_ITERATION
    )
    for distribution, size in PREFLIGHT_EXPERIMENTS:
        for seed in PREFLIGHT_SEEDS:
            first = train_worker.configured_run(
                distribution,
                size,
                PREFLIGHT_ABLATIONS[0],
                seed,
            )
            values = [
                _training_instance_seed(first, iteration, batch_index)
                for iteration in range(1, first.iterations + 1)
                for batch_index in range(first.training_instances_per_iteration)
            ]
            if len(set(values)) != len(values):
                raise ValueError(
                    f"Generation-seed collision for "
                    f"{distribution}-{size}-seed{seed}"
                )

            first_prefix = [
                _training_instance_seed(first, 1, index)
                for index in range(first.training_instances_per_iteration)
            ]
            for variant in PREFLIGHT_ABLATIONS[1:]:
                other = train_worker.configured_run(
                    distribution,
                    size,
                    variant,
                    seed,
                )
                other_prefix = [
                    _training_instance_seed(other, 1, index)
                    for index in range(other.training_instances_per_iteration)
                ]
                if other_prefix != first_prefix:
                    raise AssertionError(
                        "Ablation variant leaked into the matched training "
                        f"instance stream: {variant}"
                    )
    return per_run


def main() -> None:
    formal_tasks = (
        len(PREFLIGHT_EXPERIMENTS)
        * len(PREFLIGHT_ABLATIONS)
        * len(PREFLIGHT_SEEDS)
    )
    expected_tasks = 40
    if formal_tasks != expected_tasks:
        raise AssertionError(
            f"Expected {expected_tasks} formal ablation trainings, "
            f"got {formal_tasks}"
        )

    print("Section 5.5 ablation preflight")
    print(f"  tasks     : {formal_tasks} formal ablation trainings")
    print(f"  seeds     : {PREFLIGHT_SEEDS}")
    print(
        f"  protocol  : {train_worker.ITERATIONS} iterations x "
        f"{train_worker.TRAINING_INSTANCES_PER_ITERATION} instances"
    )
    for distribution, size in PREFLIGHT_EXPERIMENTS:
        summary = _audit_dataset_pair(distribution, size)
        print(
            f"  dataset   : {distribution}-{size} "
            f"validation={summary['validation_files']} "
            f"test={summary['test_files']} overlap=0"
        )

    per_run = _audit_generation_seeds()
    print(
        f"  streams   : all {per_run:,} generation seeds/run unique; "
        "ablations matched"
    )
    print("Preflight OK")


if __name__ == "__main__":
    main()
