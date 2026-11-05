from __future__ import annotations

"""Evaluate explicitly selected Section 5.2 best checkpoints."""

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
import traceback
from typing import Iterable, List, Sequence, Tuple

import torch

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    EvaluationTask,
)
from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_runner import (
    run_greedy,
    run_sampling100,
)


# One entry is:
# (unique label, distribution, training size, training seed, relative PT path)
CheckpointSelection = Tuple[str, str, str, int, Path]


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Run Greedy first. Sampling-100 is much slower but resumes after interruption.
DECODING_MODE = "sampling100"  # greedy or sampling100

CHECKPOINTS_TO_EVALUATE: Sequence[CheckpointSelection] = [
    (
        "sd1_15x10_best",
        "SD1",
        "15x10",
        9000,
        Path(
            "CPNModelV2/results/sec52_fjsp_performance/training/"
            "formal_1000x20/SD1/15x10/seed_9000/best_checkpoint.pt"
        ),
    ),
]

# All editable paths are relative to the FJSP-DRL-main-Wang project root.
TEST_DATA_ROOT = Path("data")
EVALUATION_ROOT = Path(
    "CPNModelV2/results/sec52_fjsp_performance/evaluation"
)

# Optional top-level result folder, useful when separating architectures.
RESULT_NAMESPACE = ""

# "smoke" uses two instances and four sampling rollouts. "full" follows the
# paper protocol: 100 instances and 100 rollouts for Sampling-100.
EVALUATION_PROFILE = "full"  # smoke or full

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
# to be invalid and the selected decoder must be recomputed from scratch.
OVERWRITE = False
SKIP_MISSING_CHECKPOINTS = True
STOP_ON_ERROR = True

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


def checkpoint_path(relative_path: Path) -> Path:
    return project_path(relative_path)


def test_dataset(distribution: str, size: str) -> Path:
    directory = size if distribution == "SD1" else f"{size}+mix"
    return project_path(TEST_DATA_ROOT / distribution / directory)


def _validate_label(label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise ValueError(
            "Checkpoint label must contain only letters, digits, '.', '_', "
            f"or '-': {label!r}"
        )


def validate_selection(selection: CheckpointSelection) -> None:
    if not isinstance(selection, (tuple, list)) or len(selection) != 5:
        raise ValueError(
            "Each CHECKPOINTS_TO_EVALUATE entry must contain exactly: "
            "(label, distribution, size, seed, relative_pt_path)"
        )
    label, distribution, size, seed, relative_checkpoint = selection
    _validate_label(str(label))
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution: {distribution}")
    if not re.fullmatch(r"\d+x\d+", str(size)):
        raise ValueError(f"Invalid FJSP size: {size}")
    if not isinstance(seed, int):
        raise ValueError(f"Training seed must be an integer: {seed!r}")
    path = Path(relative_checkpoint)
    if path.name != "best_checkpoint.pt" or path.parent.name != f"seed_{seed}":
        raise ValueError(
            "Each checkpoint path must end with "
            f"seed_{seed}/best_checkpoint.pt: {path}"
        )
    project_path(path)


def evaluation_task(
    label: str,
    distribution: str,
    size: str,
    seed: int,
    relative_checkpoint: Path,
) -> EvaluationTask:
    selection = (label, distribution, size, seed, relative_checkpoint)
    validate_selection(selection)
    default_batch = DEVICE_BATCH_BY_SIZE.get(
        size, DEFAULT_DEVICE_BATCH_SIZE
    )
    test_limit = 2 if EVALUATION_PROFILE == "smoke" else 100
    sampling_rollouts = 4 if EVALUATION_PROFILE == "smoke" else 100
    result_root = project_path(EVALUATION_ROOT)
    if RESULT_NAMESPACE:
        result_root = result_root / RESULT_NAMESPACE
    return EvaluationTask(
        mode="formal",
        distribution=distribution,
        size=size,
        training_seed=seed,
        checkpoint=checkpoint_path(relative_checkpoint),
        test_dataset=test_dataset(distribution, size),
        output_dir=(
            result_root
            / distribution
            / size
            / f"seed_{seed}"
            / label
        ).resolve(),
        checkpoint_label=label,
        test_limit=test_limit,
        greedy_batch_size=min(MAX_GREEDY_BATCH_SIZE, default_batch),
        sampling_rollouts=sampling_rollouts,
        sampling_batch_size=min(MAX_SAMPLING_BATCH_SIZE, default_batch),
        device=DEVICE,
    )


def selected_evaluation_tasks() -> Iterable[EvaluationTask]:
    seen_outputs = set()
    for raw_selection in CHECKPOINTS_TO_EVALUATE:
        selection = tuple(raw_selection)
        validate_selection(selection)
        task = evaluation_task(*selection)
        if task.output_dir in seen_outputs:
            raise ValueError(
                f"Duplicate evaluation output directory: {task.output_dir}"
            )
        seen_outputs.add(task.output_dir)
        yield task


def print_result(task: EvaluationTask) -> None:
    name = (
        "greedy_metadata.json"
        if DECODING_MODE == "greedy"
        else "sampling100_metadata.json"
    )
    path = task.output_dir / name
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
    print("Section 5.2 BSG-RL independent evaluation")
    print(f"  decoder       : {DECODING_MODE}")
    print(f"  test profile  : {EVALUATION_PROFILE}")
    print(f"  device        : {DEVICE}")
    print(f"  namespace     : {RESULT_NAMESPACE or '(default)'}")
    print(f"  checkpoints   : {len(selected)}")

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
            print_result(task)
            evaluated += 1
        except Exception as error:
            failures.append(f"{task.task_id}: {type(error).__name__}: {error}")
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
