from __future__ import annotations

"""Evaluate five BSG-RL seeds on the same 100 SD1 15x10 instances."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent
        for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

from pathlib import Path
from typing import Sequence

import torch

from CPNModelV2.experiments.sec54_representation import evaluate_worker as base
import CPNModelV2.experiments.sec52_fjsp_performance.train_generic_compact_10x5 as training


# =============================================================================
# USER CONFIGURATION
# =============================================================================

DISTRIBUTION = "SD1"
SIZE = "15x10"
METHOD = training.POLICY_GENERIC_COMPACT

# One entry is (training seed, checkpoint label, checkpoint path).
CHECKPOINTS: Sequence[tuple[int, str, Path]] = (
    (
        9000,
        "best",
        Path(
            "CPNModelV2/results/sec54_representation/training/formal_500x20/"
            "SD1/15x10/generic_compact_v5/seed_9000/"
            "best_checkpoint.pt"
        ),
    ),
    (
        9001,
        "best",
        Path(
            "CPNModelV2/results/sec54_representation/training/formal_500x20/"
            "SD1/15x10/generic_compact_v5/seed_9001/"
            "best_checkpoint.pt"
        ),
    ),
    (
        9002,
        "best",
        Path(
            "CPNModelV2/results/sec54_representation/training/formal_500x20/"
            "SD1/15x10/generic_compact_v5/seed_9002/"
            "best_checkpoint.pt"
        ),
    ),
    (
        9003,
        "best",
        Path(
            "CPNModelV2/results/sec54_representation/training/formal_500x20/"
            "SD1/15x10/generic_compact_v5/seed_9003/"
            "best_checkpoint.pt"
        ),
    ),
    (
        9004,
        "best",
        Path(
            "CPNModelV2/results/sec54_representation/training/formal_500x20/"
            "SD1/15x10/generic_compact_v5/seed_9004/"
            "best_checkpoint.pt"
        ),
    ),
)

MAX_CHECKPOINT_ITERATION = 500
EVALUATION_PROFILE = "full"  # full=100 instances; smoke=2 instances
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GREEDY_BATCH_SIZE = 5

TEST_DATA_ROOT = Path("data")
EVALUATION_ROOT = Path("CPNModelV2/results/sec54_representation/evaluation")
RESULT_NAMESPACE = "formal_500_budget"
AGGREGATE_OUTPUT_DIR = (
    EVALUATION_ROOT
    / RESULT_NAMESPACE
    / DISTRIBUTION
    / SIZE
    / training.POLICY_GENERIC_COMPACT
    / "source_tables"
)

# Existing completed rows are resumed. Set True only to recompute invalid data.
OVERWRITE = False
STOP_ON_ERROR = True

# =============================================================================


CHECKPOINTS_TO_EVALUATE: Sequence[base.CheckpointSelection] = tuple(
    (
        label,
        METHOD,
        DISTRIBUTION,
        SIZE,
        seed,
        checkpoint,
    )
    for seed, label, checkpoint in CHECKPOINTS
)


def main() -> int:
    if len({seed for seed, _, _ in CHECKPOINTS}) != len(CHECKPOINTS):
        raise ValueError("Each configured checkpoint must use a unique seed")

    base.DECODING_MODE = "greedy"
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

    status = base.main()
    if status != 0:
        return status

    outputs = base.export_greedy_distribution_tables(AGGREGATE_OUTPUT_DIR)
    print("\nAggregated source tables:")
    for path in outputs:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
