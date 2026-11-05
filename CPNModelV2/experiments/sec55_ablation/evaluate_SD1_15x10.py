from __future__ import annotations

"""Independent Section 5.5 greedy test for one SD1 15x10 variant."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

from pathlib import Path
from typing import Sequence

import torch

from CPNModelV2.experiments.sec55_ablation import evaluate_worker as base


# =============================================================================
# USER CONFIGURATION
# =============================================================================

DISTRIBUTION = "SD1"
SIZE = "15x10"

# Select one ablation variant for this independent run.
VARIANT = base.FLAT_BINDING_VARIANT
TRAINING_SEEDS = (9000, 9001, 9002, 9003, 9004)
CHECKPOINTS: Sequence[tuple[int, str, Path]] = tuple(
    (seed, "best", base.ablation_checkpoint(DISTRIBUTION, SIZE, VARIANT, seed))
    for seed in TRAINING_SEEDS
)


MAX_CHECKPOINT_ITERATION = 500
EVALUATION_PROFILE = "full"  # full=100 instances; smoke=2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GREEDY_BATCH_SIZE = 5

TEST_DATA_ROOT = Path("data")
EVALUATION_ROOT = Path("CPNModelV2/results/sec55_ablation/evaluation")
RESULT_NAMESPACE = "formal_500_budget"

# Existing rows are resumed. Set True only to recompute invalid results.
OVERWRITE = False
STOP_ON_ERROR = True

# =============================================================================


CHECKPOINTS_TO_EVALUATE: Sequence[base.CheckpointSelection] = tuple(
    base.CheckpointSelection(
        variant=VARIANT,
        distribution=DISTRIBUTION,
        size=SIZE,
        seed=seed,
        checkpoint=checkpoint,
        label=label,
    )
    for seed, label, checkpoint in CHECKPOINTS
)


def main() -> int:
    if len({seed for seed, _, _ in CHECKPOINTS}) != len(CHECKPOINTS):
        raise ValueError("Each configured checkpoint must use a unique seed")

    base.CHECKPOINTS_TO_EVALUATE = CHECKPOINTS_TO_EVALUATE
    base.EVALUATION_PROFILE = EVALUATION_PROFILE
    base.TEST_DATA_ROOT = TEST_DATA_ROOT
    base.EVALUATION_ROOT = EVALUATION_ROOT
    base.RESULT_NAMESPACE = RESULT_NAMESPACE
    base.DEVICE = DEVICE
    base.MAX_GREEDY_BATCH_SIZE = MAX_GREEDY_BATCH_SIZE
    base.MAX_CHECKPOINT_ITERATION = MAX_CHECKPOINT_ITERATION
    base.OVERWRITE = OVERWRITE
    base.SKIP_MISSING_CHECKPOINTS = False
    base.STOP_ON_ERROR = STOP_ON_ERROR
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
