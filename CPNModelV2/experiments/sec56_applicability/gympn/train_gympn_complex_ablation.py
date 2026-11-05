from __future__ import annotations

"""Targeted cross-domain ablations for Sections 5.5 and 5.6.

The formal design trains three controlled BSG-RL ablations on two
relation-rich GymPN L5 scenarios.  Full BSG-RL controls are reused from the
controlled Section 5.6 experiment and are not retrained here.
"""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "gympn-main"))

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple

import torch

from CPNModelV2.experiments.sec56_applicability.gympn import (
    shared_gympn_ppo_runner as runner,
)
from CPNModelV2.experiments.sec56_applicability.gympn import (
    train_gympn_complex_controlled as controlled,
)
from CPNModelV2.experiments.sec56_applicability.gympn.gympn_ablation_policy import (
    GymPNGenericCompactAblationPolicy,
)
from CPNModelV2.learning.generic_compact_ablation_policy import (
    FLAT_BINDING_VARIANT,
    NO_REFERENCE_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
    normalize_ablation_variant,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RUN_PROFILE = "full"  # Use "smoke" first; publish only "full".
TRAINING_SEEDS = (9000, 9001, 9002, 9003, 9004)

COMPLEXITY_LEVEL = "C2_5"  # Paper L5.
SCENARIOS_TO_TRAIN = (
    "sequence_disjoint",
    "parallel_joint",
)
VARIANTS_TO_TRAIN = (
    NO_REFERENCE_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
    FLAT_BINDING_VARIANT,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTO_RESUME = True

# The 100-iteration budget is adapted from the longer FJSP protocol.  A
# validation and immutable archive checkpoint are produced at 20, 40, 60, 80,
# and 100 iterations.  best_checkpoint.pt is updated only by validation.
FULL_ITERATIONS = 100
FULL_EPISODES_PER_SCENARIO = 2
FULL_VALIDATE_EVERY = 20
FULL_VALIDATION_EPISODES_PER_SCENARIO = 20
FULL_CALIBRATION_EPISODES_PER_SCENARIO = 2

SMOKE_ITERATIONS = 2
SMOKE_EPISODES_PER_SCENARIO = 1
SMOKE_VALIDATE_EVERY = 1
SMOKE_VALIDATION_EPISODES_PER_SCENARIO = 2
SMOKE_CALIBRATION_EPISODES_PER_SCENARIO = 1

CHECKPOINT_ARCHIVE_START = 0
CHECKPOINT_ARCHIVE_INTERVAL = 20
ARCHIVE_VALIDATION_IMPROVEMENTS = True
CHECKPOINT_ARCHIVE_DIRECTORY = "checkpoint_archive"

# Keep all architecture and PPO settings matched to the Full BSG-RL runs.
HIDDEN_DIM = 64
CATEGORY_DIM = 16
REFERENCE_LAYERS = 2
FACTOR_ROUNDS = 0
DROPOUT = 0.0
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
PPO_DEVICE_MICROBATCH_SIZE = 64
GAMMA = 1.0
GAE_LAMBDA = 0.98
CLIP_RATIO = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5
ROLLOUT_BATCH_SIZE = 8

# =============================================================================


CHECKPOINT_FORMAT = "gympn_complex_bsg_rl_ablation_checkpoint_v1"
OUTPUT_ROOT = (
    controlled.PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_relational_complexity" / "controlled_ablations"
)


def _profile_tag() -> str:
    return "full_v1" if RUN_PROFILE == "full" else RUN_PROFILE


def _run_name(variant: str, scenario_name: str, training_seed: int) -> str:
    return (
        f"{scenario_name}_{variant}_{COMPLEXITY_LEVEL.lower()}_"
        f"{_profile_tag()}_seed{training_seed}"
    )


def checkpoint_path(
    variant: str,
    scenario_name: str,
    training_seed: int,
    filename: str = "best_checkpoint.pt",
) -> Path:
    variant = normalize_ablation_variant(variant)
    return (
        OUTPUT_ROOT / variant / COMPLEXITY_LEVEL.lower()
        / _run_name(variant, scenario_name, training_seed)
        / f"seed_{training_seed}"
        / filename
    )


def _validate_configuration() -> None:
    if RUN_PROFILE not in {"smoke", "full"}:
        raise ValueError("RUN_PROFILE must be 'smoke' or 'full'")
    if TRAINING_SEEDS != (9000, 9001, 9002, 9003, 9004):
        raise ValueError("Formal training seeds must be 9000--9004")
    if len(set(SCENARIOS_TO_TRAIN)) != len(SCENARIOS_TO_TRAIN):
        raise ValueError("SCENARIOS_TO_TRAIN contains duplicates")
    unknown_scenarios = sorted(
        set(SCENARIOS_TO_TRAIN) - set(controlled.PAPER_SCENARIO_NAMES)
    )
    if unknown_scenarios:
        raise ValueError(f"Unknown GymPN scenarios: {unknown_scenarios}")
    if len(set(VARIANTS_TO_TRAIN)) != len(VARIANTS_TO_TRAIN):
        raise ValueError("VARIANTS_TO_TRAIN contains duplicates")
    for variant in VARIANTS_TO_TRAIN:
        normalize_ablation_variant(variant)


def _apply_protocol_to_controlled_module() -> None:
    """Set the shared protocol before controlled._configure_runner is called."""

    controlled.RUN_PROFILE = RUN_PROFILE
    controlled.DEVICE = DEVICE
    controlled.AUTO_RESUME = AUTO_RESUME
    controlled.ROLLOUT_BATCH_SIZE = ROLLOUT_BATCH_SIZE
    controlled.SMOKE_ITERATIONS = SMOKE_ITERATIONS
    controlled.SMOKE_EPISODES_PER_SCENARIO = SMOKE_EPISODES_PER_SCENARIO
    controlled.SMOKE_VALIDATE_EVERY = SMOKE_VALIDATE_EVERY
    controlled.SMOKE_VALIDATION_EPISODES_PER_SCENARIO = (
        SMOKE_VALIDATION_EPISODES_PER_SCENARIO
    )
    controlled.SMOKE_CALIBRATION_EPISODES_PER_SCENARIO = (
        SMOKE_CALIBRATION_EPISODES_PER_SCENARIO
    )
    controlled.FULL_ITERATIONS = FULL_ITERATIONS
    controlled.FULL_EPISODES_PER_SCENARIO = FULL_EPISODES_PER_SCENARIO
    controlled.FULL_VALIDATE_EVERY = FULL_VALIDATE_EVERY
    controlled.FULL_VALIDATION_EPISODES_PER_SCENARIO = (
        FULL_VALIDATION_EPISODES_PER_SCENARIO
    )
    controlled.FULL_CALIBRATION_EPISODES_PER_SCENARIO = (
        FULL_CALIBRATION_EPISODES_PER_SCENARIO
    )
    controlled.HIDDEN_DIM = HIDDEN_DIM
    controlled.CATEGORY_DIM = CATEGORY_DIM
    controlled.REFERENCE_LAYERS = REFERENCE_LAYERS
    controlled.FACTOR_ROUNDS = FACTOR_ROUNDS
    controlled.DROPOUT = DROPOUT
    controlled.LEARNING_RATE = LEARNING_RATE
    controlled.UPDATE_EPOCHS = UPDATE_EPOCHS
    controlled.MINIBATCH_SIZE = MINIBATCH_SIZE
    controlled.PPO_DEVICE_MICROBATCH_SIZE = PPO_DEVICE_MICROBATCH_SIZE
    controlled.GAMMA = GAMMA
    controlled.GAE_LAMBDA = GAE_LAMBDA
    controlled.CLIP_RATIO = CLIP_RATIO
    controlled.VALUE_COEFFICIENT = VALUE_COEFFICIENT
    controlled.ENTROPY_COEFFICIENT = ENTROPY_COEFFICIENT
    controlled.MAX_GRAD_NORM = MAX_GRAD_NORM

    runner.CHECKPOINT_ARCHIVE_START = CHECKPOINT_ARCHIVE_START
    runner.CHECKPOINT_ARCHIVE_INTERVAL = CHECKPOINT_ARCHIVE_INTERVAL
    runner.ARCHIVE_VALIDATION_IMPROVEMENTS = ARCHIVE_VALIDATION_IMPROVEMENTS
    runner.CHECKPOINT_ARCHIVE_DIRECTORY = CHECKPOINT_ARCHIVE_DIRECTORY


def _configure_ablation(
    variant: str,
    scenario_name: str,
    training_seed: int,
) -> runner.TrainingConfig:
    variant = normalize_ablation_variant(variant)
    _apply_protocol_to_controlled_module()
    config = controlled._configure_runner(
        "BSG_RL",
        COMPLEXITY_LEVEL,
        scenario_name,
        training_seed,
    )
    run_name = _run_name(variant, scenario_name, training_seed)
    output_dir = (
        OUTPUT_ROOT
        / variant
        / COMPLEXITY_LEVEL.lower()
        / run_name
        / f"seed_{training_seed}"
    ).resolve()

    runner.CHECKPOINT_FORMAT = CHECKPOINT_FORMAT
    runner.PolicyNetwork = partial(
        GymPNGenericCompactAblationPolicy,
        ablation_variant=variant,
    )
    runner.EXPERIMENT_METADATA = {
        **runner.EXPERIMENT_METADATA,
        "section": "5.5/5.6",
        "suite": "gympn_cross_domain_ablation_v1",
        "model": "BSG_RL",
        "ablation_variant": variant,
        "comparison_control": "existing_full_bsg_rl",
        "formal_training_iterations": FULL_ITERATIONS,
        "checkpoint_archive_interval": CHECKPOINT_ARCHIVE_INTERVAL,
    }
    return replace(config, output_dir=output_dir)


def _validate_existing_checkpoint(
    config: runner.TrainingConfig,
    variant: str,
) -> None:
    checkpoint = config.output_dir / "last_checkpoint.pt"
    if not checkpoint.is_file():
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint}")
    metadata = payload.get("experiment_metadata", {})
    if metadata.get("ablation_variant") != variant:
        raise ValueError(f"Checkpoint ablation mismatch: {checkpoint}")


def run_selected() -> List[Tuple[str, str, int, Any]]:
    _validate_configuration()
    seeds = TRAINING_SEEDS
    jobs = tuple(
        (variant, scenario_name, training_seed)
        for training_seed in seeds
        for scenario_name in SCENARIOS_TO_TRAIN
        for variant in VARIANTS_TO_TRAIN
    )
    print("Sections 5.5/5.6 GymPN cross-domain ablation training")
    print(f"  profile  : {RUN_PROFILE}")
    print(f"  seeds    : {seeds}")
    print(f"  level    : {COMPLEXITY_LEVEL}")
    print(f"  jobs     : {len(jobs)}")
    print(f"  archive  : every {CHECKPOINT_ARCHIVE_INTERVAL} iterations")
    results = []
    for variant, scenario_name, training_seed in jobs:
        print(
            f"  run      : {variant} / {scenario_name} / seed{training_seed}",
            flush=True,
        )
        config = _configure_ablation(
            variant,
            scenario_name,
            training_seed,
        )
        _validate_existing_checkpoint(config, variant)
        result = runner.run_training(config)
        results.append((variant, scenario_name, training_seed, result))
    return results


def main() -> int:
    run_selected()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
