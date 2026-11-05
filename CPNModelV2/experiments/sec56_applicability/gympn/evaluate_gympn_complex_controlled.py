from __future__ import annotations

"""Final held-out evaluation for Section 5.6 controlled GymPN models.

Checkpoint selection must be completed from validation results before adding a
task here.  This script evaluates each selected checkpoint once on the same
100 held-out seeds and caches non-learning baselines across model families.
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

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import torch

from CPNModelV2.experiments.sec56_applicability.gympn import (
    shared_gympn_ppo_runner as runner,
)
from CPNModelV2.experiments.sec56_applicability.gympn import (
    train_gympn_complex_controlled as training,
)
from CPNModelV2.learning import PPOConfig, TokenBindingPPO


# =============================================================================
# USER CONFIGURATION
# Evaluate all paper tasks with their validation-selected best checkpoints.
# =============================================================================

MODELS_TO_EVALUATE = ("BSG_RL", "AG_PPO")
TRAINING_SEEDS_TO_EVALUATE = (9000, 9001, 9002, 9003, 9004)

TRAINING_TASKS_TO_EVALUATE = training.PAPER_TRAINING_TASKS
EVALUATION_TASKS: Tuple[Tuple[str, str, str, int], ...] = tuple(
    (model_name, complexity_level, scenario_name, training_seed)
    for complexity_level, scenario_name in TRAINING_TASKS_TO_EVALUATE
    for training_seed in TRAINING_SEEDS_TO_EVALUATE
    for model_name in MODELS_TO_EVALUATE
)
CHECKPOINT_FILENAME = "best_checkpoint.pt"
FORMAL_TRAINING_ITERATIONS = 200
BASE_EVALUATION_SEED = 19000
EVALUATION_EPISODES = 100
BATCH_SIZE = 8
SKIP_COMPLETED = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================


OUTPUT_ROOT = (
    training.PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_relational_complexity" / "controlled_evaluations"
)
BASELINE_CACHE_ROOT = (
    training.PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_relational_complexity" / "shared_test_baselines"
)


def _seeds() -> List[int]:
    return [
        BASE_EVALUATION_SEED * 1_000_000 + 90_000_000 + episode_index
        for episode_index in range(EVALUATION_EPISODES)
    ]


def _checkpoint_path(
    model_name: str,
    complexity_level: str,
    scenario_name: str,
    training_seed: int,
) -> Path:
    model_root = training.OUTPUT_ROOT / model_name.lower()
    if model_name == "BSG_RL":
        model_root = model_root / "declared_schema_v2"
    run_name = (
        f"{scenario_name}_{model_name.lower()}_{complexity_level.lower()}_"
        f"full_v2_seed{training_seed}"
    )
    return (
        model_root
        / complexity_level.lower()
        / run_name
        / f"seed_{training_seed}"
        / CHECKPOINT_FILENAME
    )


def _output_dir(task: Tuple[str, str, str, int]) -> Path:
    model_name, complexity_level, scenario_name, training_seed = task
    return (
        OUTPUT_ROOT / model_name.lower() / complexity_level.lower()
        / (
            f"{scenario_name}_full_v2_seed{training_seed}_"
            f"testseed{BASE_EVALUATION_SEED}"
        )
    )


def _completed_result(task: Tuple[str, str, str, int]) -> Dict[str, Any] | None:
    output_dir = _output_dir(task)
    per_episode_path = output_dir / "per_episode_results.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "evaluation_metadata.json"
    if not (
        per_episode_path.is_file()
        and summary_path.is_file()
        and metadata_path.is_file()
    ):
        return None
    try:
        with per_episode_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != EVALUATION_EPISODES:
            return None
        if [int(row["seed"]) for row in rows] != _seeds():
            return None
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    model_name, complexity_level, scenario_name, training_seed = task
    expected = {
        "model": model_name,
        "complexity_level": complexity_level,
        "scenario": scenario_name,
        "training_seed": training_seed,
        "base_evaluation_seed": BASE_EVALUATION_SEED,
        "evaluation_episodes": EVALUATION_EPISODES,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    return metadata


def _read_baseline_cache(path: Path) -> Dict[str, List[float]] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EVALUATION_EPISODES:
        return None
    expected_seeds = _seeds()
    if [int(row["seed"]) for row in rows] != expected_seeds:
        return None
    return {
        "random": [float(row["random"]) for row in rows],
        "shortest_effect": [float(row["shortest_effect"]) for row in rows],
    }


def _baseline_scores(
    complexity_level: str,
    scenario_name: str,
    *,
    horizon: float,
) -> Dict[str, List[float]]:
    cache_path = (
        BASELINE_CACHE_ROOT
        / f"horizon_{horizon:g}"
        / f"episodes_{EVALUATION_EPISODES}"
        / f"seed{BASE_EVALUATION_SEED}"
        / complexity_level.lower()
        / f"{scenario_name}.csv"
    )
    cached = _read_baseline_cache(cache_path)
    if cached is not None:
        return cached
    seeds = _seeds()
    scores = {
        rule: training._evaluate_rule(
            complexity_level,
            scenario_name,
            horizon=horizon,
            seeds=seeds,
            rule=rule,
        )
        for rule in ("random", "shortest_effect")
    }
    runner._write_csv(
        cache_path,
        [
            {
                "episode": index + 1,
                "seed": seed,
                "random": scores["random"][index],
                "shortest_effect": scores["shortest_effect"][index],
            }
            for index, seed in enumerate(seeds)
        ],
        ("episode", "seed", "random", "shortest_effect"),
    )
    return scores


def _evaluate_task(task: Tuple[str, str, str, int]) -> Dict[str, Any]:
    model_name, complexity_level, scenario_name, training_seed = task
    if SKIP_COMPLETED:
        completed = _completed_result(task)
        if completed is not None:
            print(
                f"  skip complete: {model_name}/{complexity_level}/"
                f"{scenario_name}/seed{training_seed}",
                flush=True,
            )
            return completed
    if model_name not in training.CHECKPOINT_FORMATS:
        raise ValueError(f"Unknown model: {model_name}")
    checkpoint_path = _checkpoint_path(*task)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != training.CHECKPOINT_FORMATS[model_name]:
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    if tuple(checkpoint.get("scenario_names", ())) != (scenario_name,):
        raise ValueError(f"Checkpoint scenario mismatch: {checkpoint_path}")
    metadata = checkpoint.get("experiment_metadata", {})
    if metadata.get("complexity_level") != complexity_level:
        raise ValueError(f"Checkpoint complexity mismatch: {checkpoint_path}")
    checkpoint_seed = int(checkpoint["training_config"]["seed"])
    if checkpoint_seed != training_seed:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {training_seed}, "
            f"found {checkpoint_seed}: {checkpoint_path}"
        )
    checkpoint_iteration = int(checkpoint["iteration"])
    exceeds_formal_budget = checkpoint_iteration > FORMAL_TRAINING_ITERATIONS
    if exceeds_formal_budget:
        raise ValueError(
            f"Checkpoint iteration {checkpoint_iteration} exceeds formal "
            f"budget {FORMAL_TRAINING_ITERATIONS}: {checkpoint_path}"
        )

    training._configure_representation(model_name)
    schema = runner.restore_graph_feature_schema(checkpoint["schema"])
    normalizer = runner.restore_numeric_feature_normalizer(
        checkpoint["normalizer"], schema
    )
    trainer = TokenBindingPPO(
        runner.PolicyNetwork(schema, **checkpoint["policy_config"]),
        runner.GraphTensorizer(
            schema,
            normalizer=normalizer,
            validate_graphs=False,
        ),
        config=PPOConfig(**checkpoint["ppo_config"]),
        device=DEVICE,
    )
    trainer.policy.load_state_dict(checkpoint["model_state_dict"])
    trainer.policy.eval()

    horizon = float(checkpoint["training_config"]["horizon"])
    environment_builder = training._environment_builder(
        model_name, complexity_level
    )
    seeds = _seeds()
    policy_scores: List[float] = []
    with runner._preserve_random_state():
        for start in range(0, len(seeds), BATCH_SIZE):
            chunk = seeds[start : start + BATCH_SIZE]
            results = trainer.evaluate_episodes_batched(
                [
                    environment_builder(
                        scenario_name,
                        horizon=horizon,
                        case_multiplier=1,
                    )
                    for _seed_value in chunk
                ],
                seeds=chunk,
                deterministic=True,
            )
            policy_scores.extend(-float(result.makespan) for result in results)
            print(
                f"  {model_name}/{complexity_level}/{scenario_name}: "
                f"{len(policy_scores)}/{EVALUATION_EPISODES}",
                flush=True,
            )
    baseline_scores = _baseline_scores(
        complexity_level, scenario_name, horizon=horizon
    )
    scores = {model_name: policy_scores, **baseline_scores}
    output_dir = _output_dir(task)
    runner._write_csv(
        output_dir / "per_episode_results.csv",
        [
            {
                "episode": index + 1,
                "seed": seed,
                **{method: values[index] for method, values in scores.items()},
            }
            for index, seed in enumerate(seeds)
        ],
        ("episode", "seed", model_name, "random", "shortest_effect"),
    )
    summary = [
        {
            "method": method,
            "mean_completed_tasks": mean(values),
            "std_completed_tasks": pstdev(values),
            "episodes": len(values),
        }
        for method, values in scores.items()
    ]
    runner._write_csv(
        output_dir / "summary.csv",
        summary,
        ("method", "mean_completed_tasks", "std_completed_tasks", "episodes"),
    )
    result = {
        "model": model_name,
        "complexity_level": complexity_level,
        "paper_complexity_level": training.PAPER_COMPLEXITY_LABELS.get(
            complexity_level, "extreme"
        ),
        "scenario": scenario_name,
        "training_seed": training_seed,
        "base_evaluation_seed": BASE_EVALUATION_SEED,
        "evaluation_episodes": EVALUATION_EPISODES,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_iteration": checkpoint_iteration,
        "formal_training_iterations": FORMAL_TRAINING_ITERATIONS,
        "output_dir": str(output_dir.resolve()),
    }
    runner._write_json(output_dir / "evaluation_metadata.json", result)
    return result


def main() -> int:
    if EVALUATION_EPISODES != 100:
        raise ValueError("Formal Section 5.6 evaluation requires 100 episodes")
    expected = set(training.PAPER_TRAINING_TASKS)
    selected = set(TRAINING_TASKS_TO_EVALUATE)
    if selected != expected:
        missing = sorted(expected - selected)
        extra = sorted(selected - expected)
        raise ValueError(
            f"Evaluation tasks do not match formal design; "
            f"missing={missing}, extra={extra}"
        )
    missing_checkpoints = [
        str(_checkpoint_path(*task))
        for task in EVALUATION_TASKS
        if not _checkpoint_path(*task).is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Missing selected checkpoints:\n" + "\n".join(missing_checkpoints)
        )
    print("Section 5.6 GymPN held-out evaluation", flush=True)
    print(f"  groups  : {len(TRAINING_TASKS_TO_EVALUATE)}", flush=True)
    print(f"  runs    : {len(EVALUATION_TASKS)}", flush=True)
    print(f"  device  : {DEVICE}", flush=True)
    for task in EVALUATION_TASKS:
        _evaluate_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
