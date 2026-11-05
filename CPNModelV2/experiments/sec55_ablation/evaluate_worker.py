from __future__ import annotations

"""Shared greedy evaluator for the independent Section 5.5 test scripts."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass
import json
from pathlib import Path
import traceback
from typing import Any, Dict, Iterable, List, Sequence

import torch

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    CHECKPOINT_FORMAT_SEC55_ABLATION,
    EvaluationTask,
    read_csv,
    write_csv_atomic,
)
from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_runner import (
    run_greedy,
)
from CPNModelV2.experiments.sec52_fjsp_performance.train_generic_compact_10x5 import (
    POLICY_GENERIC_COMPACT,
)
from CPNModelV2.learning.generic_compact_ablation_policy import (
    FLAT_BINDING_VARIANT,
    FULL_VARIANT,
    NO_REFERENCE_VARIANT,
    NO_REFERENCE_FLAT_BINDING_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
)


SUPPORTED_VARIANTS = (
    FULL_VARIANT,
    FLAT_BINDING_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
    NO_REFERENCE_VARIANT,
    NO_REFERENCE_FLAT_BINDING_VARIANT,
)
VARIANT_LABELS = {
    FULL_VARIANT: "BSG-RL",
    FLAT_BINDING_VARIANT: "w/o role-aware composition",
    NO_RELATION_DIRECTION_VARIANT: "w/o relation/direction semantics",
    NO_REFERENCE_VARIANT: "w/o reference propagation",
    NO_REFERENCE_FLAT_BINDING_VARIANT: "w/o reference/role semantics",
}
DEVICE_BATCH_BY_SIZE = {
    "10x5": 5,
    "15x10": 2,
    "20x5": 2,
    "20x10": 1,
}


@dataclass(frozen=True)
class CheckpointSelection:
    variant: str
    distribution: str
    size: str
    seed: int
    checkpoint: Path
    label: str = "best"


# =============================================================================
# CONFIGURATION SUPPLIED BY AN INDEPENDENT SCENE SCRIPT
# =============================================================================

CHECKPOINTS_TO_EVALUATE: Sequence[CheckpointSelection] = ()
EVALUATION_PROFILE = "full"  # full=100 test instances; smoke=2
TEST_DATA_ROOT = Path("data")
EVALUATION_ROOT = Path("CPNModelV2/results/sec55_ablation/evaluation")
RESULT_NAMESPACE = "formal_500_budget"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GREEDY_BATCH_SIZE = 5
MAX_CHECKPOINT_ITERATION = 500
OVERWRITE = False
SKIP_MISSING_CHECKPOINTS = True
STOP_ON_ERROR = True

# =============================================================================


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
STATUS_FIELDS = (
    "distribution",
    "size",
    "variant",
    "variant_label",
    "seed",
    "status",
    "checkpoint_iteration",
    "expected_instances",
    "completed_instances",
    "checkpoint_path",
    "result_path",
    "error",
)


def project_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes the project root: {path}") from error
    return resolved


def ablation_checkpoint(
    distribution: str,
    size: str,
    variant: str,
    seed: int,
) -> Path:
    return Path(
        "CPNModelV2/results/sec55_ablation/training/formal_500x20"
    ) / distribution / size / variant / f"seed_{seed}" / "best_checkpoint.pt"


def make_scene_selections(
    *,
    distribution: str,
    size: str,
    seeds: Sequence[int],
    full_checkpoints: Dict[int, Path],
    ablation_variants: Sequence[str] = (
        FLAT_BINDING_VARIANT,
        NO_RELATION_DIRECTION_VARIANT,
        NO_REFERENCE_VARIANT,
        NO_REFERENCE_FLAT_BINDING_VARIANT,
    ),
) -> tuple[CheckpointSelection, ...]:
    """Build a complete scene matrix from validation-selected checkpoints."""

    selections: List[CheckpointSelection] = []
    for seed in seeds:
        selections.append(CheckpointSelection(
            FULL_VARIANT, distribution, size, seed, full_checkpoints[seed]
        ))
        for variant in ablation_variants:
            if variant not in SUPPORTED_VARIANTS or variant == FULL_VARIANT:
                raise ValueError(f"Unsupported ablation variant: {variant}")
            selections.append(CheckpointSelection(
                variant,
                distribution,
                size,
                seed,
                ablation_checkpoint(distribution, size, variant, seed),
            ))
    return tuple(selections)


def _dataset_directory(distribution: str, size: str) -> str:
    return size if distribution == "SD1" else f"{size}+mix"


def evaluation_task(selection: CheckpointSelection) -> EvaluationTask:
    if selection.variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant: {selection.variant}")
    checkpoint = Path(selection.checkpoint)
    if (
        checkpoint.name != "best_checkpoint.pt"
        or checkpoint.parent.name != f"seed_{selection.seed}"
    ):
        raise ValueError(
            "Each checkpoint path must end with "
            f"seed_{selection.seed}/best_checkpoint.pt: {checkpoint}"
        )
    test_limit = 2 if EVALUATION_PROFILE == "smoke" else 100
    batch_size = min(
        MAX_GREEDY_BATCH_SIZE,
        DEVICE_BATCH_BY_SIZE.get(selection.size, 1),
    )
    output_dir = (
        project_path(EVALUATION_ROOT)
        / RESULT_NAMESPACE
        / selection.distribution
        / selection.size
        / selection.variant
        / f"seed_{selection.seed}"
        / selection.label
    )
    return EvaluationTask(
        mode="formal",
        distribution=selection.distribution,
        size=selection.size,
        training_seed=selection.seed,
        checkpoint=project_path(checkpoint),
        test_dataset=project_path(
            TEST_DATA_ROOT
            / selection.distribution
            / _dataset_directory(selection.distribution, selection.size)
        ),
        output_dir=output_dir,
        checkpoint_label=f"{selection.variant}_{selection.label}",
        test_limit=test_limit,
        greedy_batch_size=batch_size,
        device=DEVICE,
    )


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_checkpoint(
    selection: CheckpointSelection,
    checkpoint: Dict[str, Any],
) -> int:
    iteration = int(checkpoint.get("iteration", -1))
    if iteration < 1 or iteration > MAX_CHECKPOINT_ITERATION:
        raise ValueError(
            f"Checkpoint iteration {iteration} is outside 1--"
            f"{MAX_CHECKPOINT_ITERATION}: {selection.checkpoint}"
        )
    training = checkpoint.get("training_config", {})
    observed = (
        str(training.get("distribution", "")),
        str(training.get("training_size", "")),
        int(training.get("seed", checkpoint.get("seed", -1))),
    )
    expected = (selection.distribution, selection.size, selection.seed)
    if observed != expected:
        raise ValueError(
            f"Checkpoint/task mismatch: expected={expected}, observed={observed}"
        )
    if selection.variant == FULL_VARIANT:
        if checkpoint.get("format") == CHECKPOINT_FORMAT_SEC55_ABLATION:
            raise ValueError("The Full control must use a non-ablation checkpoint")
        if checkpoint.get("policy_variant") != POLICY_GENERIC_COMPACT:
            raise ValueError("The Full control is not a BSG-RL checkpoint")
    else:
        if checkpoint.get("format") != CHECKPOINT_FORMAT_SEC55_ABLATION:
            raise ValueError("Ablation checkpoint has an unsupported format")
        if checkpoint.get("ablation_variant") != selection.variant:
            raise ValueError(
                "Ablation variant mismatch: expected "
                f"{selection.variant}, found {checkpoint.get('ablation_variant')}"
            )
    return iteration


def _status_row(
    selection: CheckpointSelection,
    task: EvaluationTask,
    *,
    status: str,
    checkpoint_iteration: int | str = "",
    completed_instances: int = 0,
    error: str = "",
) -> Dict[str, Any]:
    return {
        "distribution": selection.distribution,
        "size": selection.size,
        "variant": selection.variant,
        "variant_label": VARIANT_LABELS[selection.variant],
        "seed": selection.seed,
        "status": status,
        "checkpoint_iteration": checkpoint_iteration,
        "expected_instances": task.test_limit,
        "completed_instances": completed_instances,
        "checkpoint_path": str(task.checkpoint),
        "result_path": str(task.output_dir / "greedy_per_instance.csv"),
        "error": error,
    }


def _write_status(rows: Sequence[Dict[str, Any]]) -> Path:
    first = CHECKPOINTS_TO_EVALUATE[0]
    path = (
        project_path(EVALUATION_ROOT)
        / RESULT_NAMESPACE
        / first.distribution
        / first.size
        / "evaluation_status.csv"
    )
    # Independent scripts are normally run once per variant. Preserve status
    # rows written by earlier variant runs and replace only matching
    # (variant, seed) records.
    merged: Dict[tuple[str, int], Dict[str, Any]] = {
        (row["variant"], int(row["seed"])): dict(row)
        for row in read_csv(path)
    }
    for row in rows:
        merged[(str(row["variant"]), int(row["seed"]))] = dict(row)
    variant_order = {
        variant: index for index, variant in enumerate(SUPPORTED_VARIANTS)
    }
    ordered = sorted(
        merged.values(),
        key=lambda row: (
            variant_order.get(str(row["variant"]), len(variant_order)),
            int(row["seed"]),
        ),
    )
    write_csv_atomic(path, ordered, STATUS_FIELDS)
    return path


def selected_tasks() -> Iterable[tuple[CheckpointSelection, EvaluationTask]]:
    seen = set()
    for selection in CHECKPOINTS_TO_EVALUATE:
        key = (selection.distribution, selection.size, selection.variant, selection.seed)
        if key in seen:
            raise ValueError(f"Duplicate evaluation selection: {key}")
        seen.add(key)
        yield selection, evaluation_task(selection)


def main() -> int:
    if EVALUATION_PROFILE not in {"smoke", "full"}:
        raise ValueError("EVALUATION_PROFILE must be 'smoke' or 'full'")
    if not CHECKPOINTS_TO_EVALUATE:
        raise ValueError("Run an independent Section 5.5 scene script")
    scenes = {
        (item.distribution, item.size) for item in CHECKPOINTS_TO_EVALUATE
    }
    if len(scenes) != 1:
        raise ValueError("An independent test script must contain one scene")

    statuses: List[Dict[str, Any]] = []
    failures = 0
    for selection, task in selected_tasks():
        print(
            f"\n[{selection.distribution} {selection.size} | "
            f"{selection.variant} | seed {selection.seed}]"
        )
        print(f"  checkpoint : {task.checkpoint}")
        if not task.checkpoint.is_file():
            row = _status_row(
                selection, task, status="missing_checkpoint"
            )
            statuses.append(row)
            _write_status(statuses)
            print("  skipped    : checkpoint is not available")
            if SKIP_MISSING_CHECKPOINTS:
                continue
            raise FileNotFoundError(task.checkpoint)

        try:
            checkpoint = _torch_load(task.checkpoint)
            iteration = _validate_checkpoint(selection, checkpoint)
            run_greedy(task, overwrite=OVERWRITE)
            result_path = task.output_dir / "greedy_per_instance.csv"
            completed = len(read_csv(result_path))
            status = "observed" if completed == task.test_limit else "partial"
            statuses.append(_status_row(
                selection,
                task,
                status=status,
                checkpoint_iteration=iteration,
                completed_instances=completed,
            ))
            metadata_path = task.output_dir / "greedy_metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                print(f"  mean       : {metadata['mean_makespan']:.3f}")
        except Exception as error:
            failures += 1
            completed = len(read_csv(task.output_dir / "greedy_per_instance.csv"))
            statuses.append(_status_row(
                selection,
                task,
                status="failed",
                completed_instances=completed,
                error=f"{type(error).__name__}: {error}",
            ))
            traceback.print_exc()
            _write_status(statuses)
            if STOP_ON_ERROR:
                break
        _write_status(statuses)

    status_path = _write_status(statuses)
    print(f"\nStatus table: {status_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
