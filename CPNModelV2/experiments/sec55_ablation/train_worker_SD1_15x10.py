from __future__ import annotations

"""Section 5.5 launcher for the SD1 15x10 scene."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    _project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

from CPNModelV2.experiments.sec55_ablation import train_worker as base


# =============================================================================
# USER CONFIGURATION
# =============================================================================

EXPERIMENTS = (("SD1", "15x10"),)
ABLATIONS = tuple(base.SUPPORTED_ABLATION_VARIANTS)
TRAINING_SEEDS = (9000, 9001, 9002, 9003, 9004)

# =============================================================================


def main() -> None:
    base.EXPERIMENTS = EXPERIMENTS
    base.ABLATIONS = ABLATIONS
    base.TRAINING_SEEDS = TRAINING_SEEDS
    base.main()


if __name__ == "__main__":
    main()
