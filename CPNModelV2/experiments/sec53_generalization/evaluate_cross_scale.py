from __future__ import annotations

"""Run Section 5.3.1 cross-scale tests for explicitly selected checkpoints."""

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
import traceback
from typing import Iterable, List, Sequence

import torch

from CPNModelV2.experiments.sec52_fjsp_performance.evaluate_worker import (
    CheckpointSelection,
    validate_selection,
)
from CPNModelV2.experiments.sec53_generalization.cross_scale_core import (
    CrossScaleEvaluationTask,
)
from CPNModelV2.experiments.sec53_generalization.cross_scale_runner import (
    run_greedy,
    run_sampling100,
)
from CPNModelV2.experiments.sec53_generalization.task_matrix import (
    CROSS_SCALE_TARGET_SIZES,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Entry scripts must explicitly select their validation-selected checkpoint.
CHECKPOINTS_TO_EVALUATE: Sequence[CheckpointSelection] = ()

TARGET_SIZES = ("30x10", "40x10")
DECODING_MODE = "greedy"  # greedy or sampling100
EVALUATION_PROFILE = "full"  # smoke or full
RESULT_NAMESPACE = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GREEDY_BATCH_BY_TARGET = {"30x10": 1, "40x10": 1}
SAMPLING_BATCH_BY_TARGET = {"30x10": 1, "40x10": 1}
DEFAULT_BATCH_SIZE = 1

OVERWRITE = False
SKIP_MISSING_CHECKPOINTS = False
STOP_ON_ERROR = True

# All editable paths are project-relative.
TEST_DATA_ROOT = Path("data")
REFERENCE_ROOT = Path("or_solution")
RESULT_ROOT = Path("CPNModelV2/results/sec53_generalization")

# =============================================================================


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent


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


def _dataset_path(distribution: str, target_size: str) -> Path:
    directory = (
        target_size if distribution == "SD1" else f"{target_size}+mix"
    )
    return project_path(TEST_DATA_ROOT / distribution / directory)


def _reference_path(distribution: str, target_size: str) -> Path:
    suffix = target_size if distribution == "SD1" else f"{target_size}+mix"
    return project_path(
        REFERENCE_ROOT / distribution / f"solution_{suffix}.npy"
    )


def evaluation_task(
    selection: CheckpointSelection,
    target_size: str,
) -> CrossScaleEvaluationTask:
    validate_selection(selection)
    if target_size not in CROSS_SCALE_TARGET_SIZES:
        raise ValueError(f"Unsupported target size: {target_size}")
    label, distribution, source_size, seed, relative_checkpoint = selection
    profile_root = project_path(RESULT_ROOT)
    if EVALUATION_PROFILE == "smoke":
        profile_root = profile_root / "smoke"
    suite_root = profile_root / "cross_scale"
    if RESULT_NAMESPACE:
        suite_root = suite_root / RESULT_NAMESPACE
    test_limit = 2 if EVALUATION_PROFILE == "smoke" else 100
    sampling_rollouts = 4 if EVALUATION_PROFILE == "smoke" else 100
    return CrossScaleEvaluationTask(
        distribution=distribution,
        source_size=source_size,
        target_size=target_size,
        training_seed=seed,
        checkpoint_mode="selected",
        checkpoint=project_path(relative_checkpoint),
        test_dataset=_dataset_path(distribution, target_size),
        reference_file=_reference_path(distribution, target_size),
        output_dir=(
            suite_root
            / distribution
            / f"train_{source_size}"
            / f"test_{target_size}"
            / f"seed_{seed}"
            / label
        ).resolve(),
        checkpoint_label=label,
        profile=EVALUATION_PROFILE,
        test_limit=test_limit,
        greedy_batch_size=GREEDY_BATCH_BY_TARGET.get(
            target_size, DEFAULT_BATCH_SIZE
        ),
        sampling_rollouts=sampling_rollouts,
        sampling_batch_size=SAMPLING_BATCH_BY_TARGET.get(
            target_size, DEFAULT_BATCH_SIZE
        ),
        device=DEVICE,
        allow_diagnostic_seed=False,
    )


def selected_evaluation_tasks() -> Iterable[CrossScaleEvaluationTask]:
    if isinstance(TARGET_SIZES, str):
        raise ValueError("TARGET_SIZES must be a tuple or list, not a string")
    seen_outputs = set()
    for raw_selection in CHECKPOINTS_TO_EVALUATE:
        selection = tuple(raw_selection)
        validate_selection(selection)
        for target_size in TARGET_SIZES:
            task = evaluation_task(selection, target_size)
            if task.output_dir in seen_outputs:
                raise ValueError(
                    f"Duplicate evaluation output directory: {task.output_dir}"
                )
            seen_outputs.add(task.output_dir)
            yield task


def _print_result(task: CrossScaleEvaluationTask) -> None:
    filename = (
        "greedy_metadata.json"
        if DECODING_MODE == "greedy"
        else "sampling100_metadata.json"
    )
    path = task.output_dir / filename
    if not path.is_file():
        return
    metadata = json.loads(path.read_text(encoding="utf-8"))
    key = (
        "mean_makespan"
        if DECODING_MODE == "greedy"
        else "mean_best_makespan"
    )
    print(f"  result     : {float(metadata[key]):.3f}")
    print(f"  gap        : {float(metadata['aggregate_gap_percent']):.3f}%")
    print(f"  checkpoint : iteration {metadata['checkpoint_iteration']}")
    print(f"  output     : {task.output_dir}")


def main() -> int:
    if DECODING_MODE not in {"greedy", "sampling100"}:
        raise ValueError("DECODING_MODE must be greedy or sampling100")
    if EVALUATION_PROFILE not in {"smoke", "full"}:
        raise ValueError("EVALUATION_PROFILE must be smoke or full")
    if not CHECKPOINTS_TO_EVALUATE:
        raise ValueError("CHECKPOINTS_TO_EVALUATE cannot be empty")
    if RESULT_NAMESPACE and (
        Path(RESULT_NAMESPACE).name != RESULT_NAMESPACE
        or Path(RESULT_NAMESPACE).is_absolute()
    ):
        raise ValueError("RESULT_NAMESPACE must be one directory name")

    selected = list(selected_evaluation_tasks())
    if not selected:
        raise RuntimeError("No cross-scale tasks were selected")
    print("Section 5.3.1 BSG-RL zero-shot cross-scale evaluation")
    print(f"  decoder      : {DECODING_MODE}")
    print(f"  profile      : {EVALUATION_PROFILE}")
    print(f"  device       : {DEVICE}")
    print(f"  checkpoints  : {len(CHECKPOINTS_TO_EVALUATE)}")
    print(f"  targets      : {TARGET_SIZES}")
    print(f"  tasks        : {len(selected)}")

    failures: List[str] = []
    evaluated = 0
    for task in selected:
        print(f"\n[{task.task_id}]")
        print(f"  file       : {task.checkpoint}")
        if not task.checkpoint.is_file():
            message = f"missing checkpoint: {task.checkpoint}"
            if SKIP_MISSING_CHECKPOINTS:
                print(f"  skipped: {message}")
                continue
            raise FileNotFoundError(task.checkpoint)
        try:
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

    print(f"\nEvaluation complete: {evaluated}/{len(selected)} tasks")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
