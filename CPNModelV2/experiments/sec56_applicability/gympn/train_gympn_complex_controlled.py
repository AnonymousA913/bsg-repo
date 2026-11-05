from __future__ import annotations

"""Controlled BSG_RL versus AG_PPO training on the complex GymPN suite.

Both representations use the same PPO backend, episode streams, validation
seeds, budgets, and checkpoint selection.  Only the graph builder, tensorizer,
and encoder differ.  Models are trained independently per scenario because a
HAN Assignment Graph has a scenario-specific heterogeneous topology.
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
from pathlib import Path
import random
from typing import Any, Iterable, List, Tuple

import torch

from CPNModelV2.experiments.sec56_applicability.gympn import (
    shared_gympn_ppo_runner as runner,
)
from CPNModelV2.graph.generic_tensorizer import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.integration.bpm2024_assignment_graph import (
    AssignmentGraphFeatureSchema,
    AssignmentGraphTensorizer,
)
from CPNModelV2.integration.gympn_assignment_graph import (
    AssignmentGraphNoOpNormalizer,
    GymPNControlledAssignmentGraphPolicy,
    build_ag_complex_suite_environment,
)
from CPNModelV2.integration.gympn_bsg_adapter import (
    build_bsg_complex_suite_environment,
)
from CPNModelV2.integration.gympn_complex_scenarios import (
    COMPLEXITY_CONFIGS,
    PAPER_COMPLEXITY_LABELS,
    PAPER_COMPLEXITY_LEVELS,
    complexity_scenario_factory,
)
from CPNModelV2.integration.gympn_scenarios import PAPER_SCENARIO_NAMES
from CPNModelV2.integration.gympn_suite import shortest_effect_action
from CPNModelV2.integration.gympn_decision_adapter import GymPNDecisionEnv
from CPNModelV2.learning.gympn_bsg_policy import GymPNGenericCompactPolicy


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RUN_PROFILE = "full"  # "smoke", "pilot", or "full"
MODELS_TO_TRAIN = ("BSG_RL", "AG_PPO")
FORMAL_TRAINING_SEEDS = (9000, 9001, 9002, 9003, 9004)
TRAINING_SEEDS = FORMAL_TRAINING_SEEDS

# Cross-shaped formal design: all eight topologies at the C2 anchor, plus a
# five-level scale sweep on two structurally complementary workflows.
TOPOLOGY_LEVELS = (
    "C1",      # L1
    "C1_5",    # L2
    "C2",      # L3
)
SCALING_SCENARIOS = (
    "sequence_disjoint",
    "parallel_joint",
    "loop_disjoint",
    "exclusive_choice_joint",
)


def _deduplicate_tasks(
    tasks: Iterable[Tuple[str, str]],
) -> Tuple[Tuple[str, str], ...]:
    return tuple(dict.fromkeys(tasks))


TOPOLOGY_TASKS = tuple(
    (complexity_level, scenario_name)
    for complexity_level in TOPOLOGY_LEVELS
    for scenario_name in PAPER_SCENARIO_NAMES
)
SCALING_TASKS = tuple(
    (complexity_level, scenario_name)
    for complexity_level in PAPER_COMPLEXITY_LEVELS
    for scenario_name in SCALING_SCENARIOS
)
SCALING_SUPPLEMENT_LEVELS = (
    "C2_25",   # L4
    "C2_5",    # L5
)

SCALING_SUPPLEMENT_TASKS = tuple(
    (complexity_level, scenario_name)
    for complexity_level in SCALING_SUPPLEMENT_LEVELS
    for scenario_name in SCALING_SCENARIOS
)
PAPER_TRAINING_TASKS = _deduplicate_tasks(
    (*TOPOLOGY_TASKS, *SCALING_SUPPLEMENT_TASKS)
)
TRAINING_TASKS = PAPER_TRAINING_TASKS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTO_RESUME = True

PROFILE_HORIZONS = {"smoke": 30.0, "pilot": 100.0, "full": 100.0}
ROLLOUT_BATCH_SIZE = 8

SMOKE_ITERATIONS = 2
SMOKE_EPISODES_PER_SCENARIO = 1
SMOKE_VALIDATE_EVERY = 1
SMOKE_VALIDATION_EPISODES_PER_SCENARIO = 2
SMOKE_CALIBRATION_EPISODES_PER_SCENARIO = 1

PILOT_ITERATIONS = 50
PILOT_EPISODES_PER_SCENARIO = 2
PILOT_VALIDATE_EVERY = 5
PILOT_VALIDATION_EPISODES_PER_SCENARIO = 20
PILOT_CALIBRATION_EPISODES_PER_SCENARIO = 2

FULL_ITERATIONS = 200
FULL_EPISODES_PER_SCENARIO = 2
FULL_VALIDATE_EVERY = 20
FULL_VALIDATION_EPISODES_PER_SCENARIO = 20
FULL_CALIBRATION_EPISODES_PER_SCENARIO = 2

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

PACKAGE_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if parent.name == "CPNModelV2"
)
OUTPUT_ROOT = (
    PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_relational_complexity" / "controlled_models"
)
BASELINE_CACHE_ROOT = (
    PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_relational_complexity" / "shared_baselines"
)

# =============================================================================


CHECKPOINT_FORMATS = {
    "BSG_RL": "gympn_complex_bsg_rl_controlled_checkpoint_v2",
    "AG_PPO": "gympn_complex_ag_ppo_controlled_checkpoint_v1",
}
REPRESENTATION_REVISIONS = {
    "BSG_RL": "declared_token_schema_v2",
    "AG_PPO": "assignment_graph_v1",
}
ALGORITHM_IMPLEMENTATIONS = {
    "BSG_RL": "generic_compact_v5",
    "AG_PPO": "assignment_graph_han",
}


def _validate_configuration() -> None:
    if RUN_PROFILE not in PROFILE_HORIZONS:
        raise ValueError("RUN_PROFILE must be smoke, pilot, or full")
    unknown_models = sorted(set(MODELS_TO_TRAIN) - set(CHECKPOINT_FORMATS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    if not TRAINING_SEEDS:
        raise ValueError("TRAINING_SEEDS must not be empty")
    if len(set(TRAINING_SEEDS)) != len(TRAINING_SEEDS):
        raise ValueError("TRAINING_SEEDS must not contain duplicates")
    if not TRAINING_TASKS:
        raise ValueError("TRAINING_TASKS must not be empty")
    if len(set(TRAINING_TASKS)) != len(TRAINING_TASKS):
        raise ValueError("TRAINING_TASKS must not contain duplicates")
    unknown_levels = sorted({
        level for level, _scenario in TRAINING_TASKS
        if level not in COMPLEXITY_CONFIGS
    })
    if unknown_levels:
        raise ValueError(f"Unknown complexity levels: {unknown_levels}")
    unknown_scenarios = sorted({
        scenario for _level, scenario in TRAINING_TASKS
        if scenario not in PAPER_SCENARIO_NAMES
    })
    if unknown_scenarios:
        raise ValueError(f"Unknown GymPN scenarios: {unknown_scenarios}")


def _environment_builder(model_name: str, complexity_level: str):
    def build(
        scenario_name: str,
        *,
        horizon: float,
        case_multiplier: int = 1,
    ):
        if case_multiplier != 1:
            raise ValueError("Relational-complexity runs require case_multiplier=1")
        if model_name == "BSG_RL":
            return build_bsg_complex_suite_environment(
                scenario_name,
                complexity_level,
                horizon=horizon,
            )
        return build_ag_complex_suite_environment(
            scenario_name,
            complexity_level,
            horizon=horizon,
        )

    return build


def _evaluate_rule(
    complexity_level: str,
    scenario_name: str,
    *,
    horizon: float,
    seeds: Iterable[int],
    rule: str,
) -> List[float]:
    scores = []
    for seed in seeds:
        env = GymPNDecisionEnv(
            complexity_scenario_factory(scenario_name, complexity_level),
            horizon=horizon,
        )
        graph = env.reset(seed=int(seed))
        rng = random.Random(int(seed))
        while True:
            if rule == "random":
                action = rng.choice(list(graph.action_keys.values()))
            elif rule == "shortest_effect":
                action = shortest_effect_action(graph)
            else:
                raise ValueError(f"Unknown GymPN rule: {rule}")
            step = env.step(action)
            graph = step.observation
            if step.terminated:
                scores.append(float(step.info["completed_tasks"]))
                break
    return scores


def _configure_representation(model_name: str) -> None:
    if model_name == "BSG_RL":
        runner.PolicyNetwork = GymPNGenericCompactPolicy
        runner.GraphTensorizer = GenericGraphTensorizer
        runner.fit_graph_schema = GenericGraphFeatureSchema.fit
        runner.fit_numeric_feature_normalizer = (
            GenericNumericFeatureNormalizer.fit
        )
        runner.restore_graph_feature_schema = GenericGraphFeatureSchema.from_dict
        runner.restore_numeric_feature_normalizer = (
            GenericNumericFeatureNormalizer.from_dict
        )
        return
    runner.PolicyNetwork = GymPNControlledAssignmentGraphPolicy
    runner.GraphTensorizer = AssignmentGraphTensorizer
    runner.fit_graph_schema = AssignmentGraphFeatureSchema.fit
    runner.fit_numeric_feature_normalizer = AssignmentGraphNoOpNormalizer.fit
    runner.restore_graph_feature_schema = AssignmentGraphFeatureSchema.from_dict
    runner.restore_numeric_feature_normalizer = (
        AssignmentGraphNoOpNormalizer.from_dict
    )


def _configure_runner(
    model_name: str,
    complexity_level: str,
    scenario_name: str,
    training_seed: int,
) -> runner.TrainingConfig:
    horizon = PROFILE_HORIZONS[RUN_PROFILE]
    profile_output_tag = "full_v2" if RUN_PROFILE == "full" else RUN_PROFILE
    run_name = (
        f"{scenario_name}_{model_name.lower()}_{complexity_level.lower()}_"
        f"{profile_output_tag}_seed{training_seed}"
    )
    output_root = OUTPUT_ROOT / model_name.lower()
    if model_name == "BSG_RL":
        # v1 calibrated only from sampled markings and can fail when a rare
        # Token shape first appears late in an episode.  Keep those artifacts
        # untouched and write the deterministic declared-schema revision to a
        # separate directory.
        output_root = output_root / "declared_schema_v2"
    output_root = output_root / complexity_level.lower()
    runner.CHECKPOINT_FORMAT = CHECKPOINT_FORMATS[model_name]
    runner.EXPERIMENT_METADATA = {
        "section": "5.6",
        "suite": "gympn_relational_complexity_v1",
        "model": model_name,
        "algorithm_implementation": ALGORITHM_IMPLEMENTATIONS[model_name],
        "scenario": scenario_name,
        "complexity_level": complexity_level,
        "paper_complexity_level": PAPER_COMPLEXITY_LABELS.get(
            complexity_level, "extreme"
        ),
        "experimental_design": "topology_anchor_plus_scale_sweep_v1",
        "comparison_protocol": "shared_ppo_v2",
        "representation_revision": REPRESENTATION_REVISIONS[model_name],
        "streaming_validation": True,
        "validation_rng_isolated": True,
        "smoke_result_publishable": False,
    }
    runner.RUN_PROFILE = RUN_PROFILE
    runner.SEED = training_seed
    runner.DEVICE = DEVICE
    runner.AUTO_RESUME = AUTO_RESUME
    runner.HORIZON = horizon
    runner.CASE_MULTIPLIER = 1
    runner.ROLLOUT_BATCH_SIZE = ROLLOUT_BATCH_SIZE

    runner.SMOKE_ITERATIONS = SMOKE_ITERATIONS
    runner.SMOKE_EPISODES_PER_SCENARIO = SMOKE_EPISODES_PER_SCENARIO
    runner.SMOKE_VALIDATE_EVERY = SMOKE_VALIDATE_EVERY
    runner.SMOKE_VALIDATION_EPISODES_PER_SCENARIO = (
        SMOKE_VALIDATION_EPISODES_PER_SCENARIO
    )
    runner.SMOKE_CALIBRATION_EPISODES_PER_SCENARIO = (
        SMOKE_CALIBRATION_EPISODES_PER_SCENARIO
    )
    runner.PILOT_ITERATIONS = PILOT_ITERATIONS
    runner.PILOT_EPISODES_PER_SCENARIO = PILOT_EPISODES_PER_SCENARIO
    runner.PILOT_VALIDATE_EVERY = PILOT_VALIDATE_EVERY
    runner.PILOT_VALIDATION_EPISODES_PER_SCENARIO = (
        PILOT_VALIDATION_EPISODES_PER_SCENARIO
    )
    runner.PILOT_CALIBRATION_EPISODES_PER_SCENARIO = (
        PILOT_CALIBRATION_EPISODES_PER_SCENARIO
    )
    runner.FULL_ITERATIONS = FULL_ITERATIONS
    runner.FULL_EPISODES_PER_SCENARIO = FULL_EPISODES_PER_SCENARIO
    runner.FULL_VALIDATE_EVERY = FULL_VALIDATE_EVERY
    runner.FULL_VALIDATION_EPISODES_PER_SCENARIO = (
        FULL_VALIDATION_EPISODES_PER_SCENARIO
    )
    runner.FULL_CALIBRATION_EPISODES_PER_SCENARIO = (
        FULL_CALIBRATION_EPISODES_PER_SCENARIO
    )

    runner.HIDDEN_DIM = HIDDEN_DIM
    runner.CATEGORY_DIM = CATEGORY_DIM
    runner.REFERENCE_LAYERS = REFERENCE_LAYERS
    runner.FACTOR_ROUNDS = FACTOR_ROUNDS
    runner.DROPOUT = DROPOUT
    runner.LEARNING_RATE = LEARNING_RATE
    runner.UPDATE_EPOCHS = UPDATE_EPOCHS
    runner.MINIBATCH_SIZE = MINIBATCH_SIZE
    runner.PPO_DEVICE_MICROBATCH_SIZE = PPO_DEVICE_MICROBATCH_SIZE
    runner.GAMMA = GAMMA
    runner.GAE_LAMBDA = GAE_LAMBDA
    runner.CLIP_RATIO = CLIP_RATIO
    runner.VALUE_COEFFICIENT = VALUE_COEFFICIENT
    runner.ENTROPY_COEFFICIENT = ENTROPY_COEFFICIENT
    runner.MAX_GRAD_NORM = MAX_GRAD_NORM
    runner.OUTPUT_ROOT = output_root
    runner.BASELINE_CACHE_ROOT = BASELINE_CACHE_ROOT
    runner.RUN_NAME = run_name
    runner.PAPER_SCENARIO_NAMES = (scenario_name,)
    runner.build_paper_suite_environment = _environment_builder(
        model_name, complexity_level
    )
    runner.evaluate_rule = lambda name, *, horizon, seeds, rule: _evaluate_rule(
        complexity_level,
        name,
        horizon=horizon,
        seeds=seeds,
        rule=rule,
    )
    _configure_representation(model_name)
    config = runner.configured_run()
    return replace(
        config,
        output_dir=(output_root / run_name / f"seed_{training_seed}").resolve(),
    )


def run_selected() -> List[Tuple[str, str, str, int, Any]]:
    _validate_configuration()
    results = []
    print("Section 5.6 controlled GymPN representation comparison")
    print(f"  profile : {RUN_PROFILE}")
    print(f"  horizon : {PROFILE_HORIZONS[RUN_PROFILE]}")
    print(f"  seeds   : {TRAINING_SEEDS}")
    print(f"  tasks   : {len(TRAINING_TASKS)} per model and seed")
    for training_seed in TRAINING_SEEDS:
        for complexity_level, scenario_name in TRAINING_TASKS:
            for model_name in MODELS_TO_TRAIN:
                print(
                    f"  run     : {model_name} / seed {training_seed} / "
                    f"{complexity_level} / {scenario_name}",
                    flush=True,
                )
                if model_name == "BSG_RL":
                    print("  encoder : Generic Compact V5", flush=True)
                config = _configure_runner(
                    model_name,
                    complexity_level,
                    scenario_name,
                    training_seed,
                )
                result = runner.run_training(config)
                results.append((
                    model_name,
                    complexity_level,
                    scenario_name,
                    training_seed,
                    result,
                ))
    return results


def main() -> int:
    run_selected()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
