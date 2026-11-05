from __future__ import annotations

"""Independent Section 5.3.2 entry point for the SD1 15x10 checkpoint."""

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

from CPNModelV2.experiments.sec53_generalization import (
    evaluate_public_benchmarks as base,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

DECODING_MODE = "greedy"  # Run greedy before sampling100.
EVALUATION_PROFILE = "full"  # smoke or full
RUN_GROUP = "seed9000_main_v1"
BENCHMARKS = (
    "Brandimarte",
    "Hurink_rdata",
    "Hurink_edata",
    "Hurink_vdata",
)

CHECKPOINTS_TO_EVALUATE: Sequence[base.CheckpointSelection] = [
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


def main() -> int:
    base.DECODING_MODE = DECODING_MODE
    base.EVALUATION_PROFILE = EVALUATION_PROFILE
    base.RESULT_NAMESPACE = RUN_GROUP
    base.BENCHMARKS = BENCHMARKS
    base.CHECKPOINTS_TO_EVALUATE = CHECKPOINTS_TO_EVALUATE
    base.SKIP_MISSING_CHECKPOINTS = False
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
