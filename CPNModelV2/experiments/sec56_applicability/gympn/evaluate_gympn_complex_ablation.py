from __future__ import annotations

"""Held-out evaluation for the targeted GymPN BSG-RL ablations."""

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
    train_gympn_complex_ablation as training,
)
from CPNModelV2.experiments.sec56_applicability.gympn import (
    train_gympn_complex_controlled as controlled,
)
from CPNModelV2.experiments.sec56_applicability.gympn.gympn_ablation_policy import (
    GymPNGenericCompactAblationPolicy,
)
from CPNModelV2.graph import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.learning import PPOConfig, TokenBindingPPO


# =============================================================================
# USER CONFIGURATION
# =============================================================================

TRAINING_SEEDS_TO_EVALUATE = (9000, 9001, 9002, 9003, 9004)
VARIANTS_TO_EVALUATE = training.VARIANTS_TO_TRAIN
SCENARIOS_TO_EVALUATE = training.SCENARIOS_TO_TRAIN

BASE_EVALUATION_SEED = 19000
EVALUATION_EPISODES = 100
BATCH_SIZE = 8
SKIP_COMPLETED = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================


OUTPUT_ROOT = (
    controlled.PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_relational_complexity" / "ablation_evaluations"
)


def _seeds() -> List[int]:
    return [
        BASE_EVALUATION_SEED * 1_000_000 + 90_000_000 + episode_index
        for episode_index in range(EVALUATION_EPISODES)
    ]


def _tasks() -> Tuple[Tuple[str, str, int], ...]:
    return tuple(
        (variant, scenario_name, training_seed)
        for training_seed in TRAINING_SEEDS_TO_EVALUATE
        for scenario_name in SCENARIOS_TO_EVALUATE
        for variant in VARIANTS_TO_EVALUATE
    )


def _checkpoint_path(task: Tuple[str, str, int]) -> Path:
    variant, scenario_name, training_seed = task
    return training.checkpoint_path(
        variant,
        scenario_name,
        training_seed,
        filename="best_checkpoint.pt",
    )


def _output_dir(task: Tuple[str, str, int]) -> Path:
    variant, scenario_name, training_seed = task
    return (
        OUTPUT_ROOT / variant / training.COMPLEXITY_LEVEL.lower()
        / (
            f"{scenario_name}_{variant}_full_v1_seed{training_seed}_"
            f"testseed{BASE_EVALUATION_SEED}"
        )
    )


def _completed_result(task: Tuple[str, str, int]) -> Dict[str, Any] | None:
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
    variant, scenario_name, training_seed = task
    expected = {
        "ablation_variant": variant,
        "complexity_level": training.COMPLEXITY_LEVEL,
        "scenario": scenario_name,
        "training_seed": training_seed,
        "base_evaluation_seed": BASE_EVALUATION_SEED,
        "evaluation_episodes": EVALUATION_EPISODES,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    return metadata


def _evaluate_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    variant, scenario_name, training_seed = task
    if SKIP_COMPLETED:
        completed = _completed_result(task)
        if completed is not None:
            print(
                f"  skip complete: {variant}/{scenario_name}/seed{training_seed}",
                flush=True,
            )
            return completed

    checkpoint_path = _checkpoint_path(task)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format") != training.CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    if tuple(checkpoint.get("scenario_names", ())) != (scenario_name,):
        raise ValueError(f"Checkpoint scenario mismatch: {checkpoint_path}")
    checkpoint_metadata = checkpoint.get("experiment_metadata", {})
    if checkpoint_metadata.get("ablation_variant") != variant:
        raise ValueError(f"Checkpoint ablation mismatch: {checkpoint_path}")
    if (
        checkpoint_metadata.get("complexity_level")
        != training.COMPLEXITY_LEVEL
    ):
        raise ValueError(f"Checkpoint complexity mismatch: {checkpoint_path}")
    checkpoint_seed = int(checkpoint["training_config"]["seed"])
    if checkpoint_seed != training_seed:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {training_seed}, "
            f"found {checkpoint_seed}: {checkpoint_path}"
        )
    checkpoint_iteration = int(checkpoint["iteration"])
    if checkpoint_iteration > training.FULL_ITERATIONS:
        raise ValueError(
            f"Checkpoint iteration {checkpoint_iteration} exceeds the formal "
            f"budget {training.FULL_ITERATIONS}: {checkpoint_path}"
        )

    schema = GenericGraphFeatureSchema.from_dict(checkpoint["schema"])
    normalizer = GenericNumericFeatureNormalizer.from_dict(
        checkpoint["normalizer"],
        schema,
    )
    trainer = TokenBindingPPO(
        GymPNGenericCompactAblationPolicy(
            schema,
            **checkpoint["policy_config"],
            ablation_variant=variant,
        ),
        GenericGraphTensorizer(
            schema,
            normalizer=normalizer,
            validate_graphs=False,
        ),
        config=PPOConfig(**checkpoint["ppo_config"]),
        device=DEVICE,
    )
    trainer.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer.policy.eval()

    horizon = float(checkpoint["training_config"]["horizon"])
    environment_builder = controlled._environment_builder(
        "BSG_RL",
        training.COMPLEXITY_LEVEL,
    )
    seeds = _seeds()
    scores: List[float] = []
    with runner._preserve_random_state():
        for start in range(0, len(seeds), BATCH_SIZE):
            chunk = seeds[start : start + BATCH_SIZE]
            outcomes = trainer.evaluate_episodes_batched(
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
            scores.extend(-float(outcome.makespan) for outcome in outcomes)
            print(
                f"  {variant}/{scenario_name}/seed{training_seed}: "
                f"{len(scores)}/{EVALUATION_EPISODES}",
                flush=True,
            )

    output_dir = _output_dir(task)
    runner._write_csv(
        output_dir / "per_episode_results.csv",
        [
            {
                "episode": index + 1,
                "seed": seed,
                "completed_tasks": scores[index],
            }
            for index, seed in enumerate(seeds)
        ],
        ("episode", "seed", "completed_tasks"),
    )
    runner._write_csv(
        output_dir / "summary.csv",
        [{
            "variant": variant,
            "mean_completed_tasks": mean(scores),
            "std_completed_tasks": pstdev(scores),
            "episodes": len(scores),
        }],
        (
            "variant", "mean_completed_tasks",
            "std_completed_tasks", "episodes",
        ),
    )
    result = {
        "model": "BSG_RL",
        "ablation_variant": variant,
        "complexity_level": training.COMPLEXITY_LEVEL,
        "paper_complexity_level": "L5",
        "scenario": scenario_name,
        "training_seed": training_seed,
        "base_evaluation_seed": BASE_EVALUATION_SEED,
        "evaluation_episodes": EVALUATION_EPISODES,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_iteration": checkpoint_iteration,
        "formal_training_iterations": training.FULL_ITERATIONS,
        "output_dir": str(output_dir.resolve()),
    }
    runner._write_json(output_dir / "evaluation_metadata.json", result)
    return result


def main() -> int:
    if EVALUATION_EPISODES != 100:
        raise ValueError("Formal ablation evaluation requires 100 episodes")
    if TRAINING_SEEDS_TO_EVALUATE != (9000, 9001, 9002, 9003, 9004):
        raise ValueError("Formal evaluation seeds must be 9000--9004")
    tasks = _tasks()
    missing = [str(_checkpoint_path(task)) for task in tasks if not _checkpoint_path(task).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing selected checkpoints:\n" + "\n".join(missing)
        )
    print("Sections 5.5/5.6 GymPN ablation held-out evaluation")
    print(f"  runs    : {len(tasks)}")
    print(f"  device  : {DEVICE}")
    for task in tasks:
        _evaluate_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
