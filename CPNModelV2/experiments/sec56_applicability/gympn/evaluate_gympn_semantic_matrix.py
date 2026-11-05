from __future__ import annotations

"""Held-out evaluation of the GymPN scale--ambiguity matrix checkpoints."""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    _workspace_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "gympn-main").is_dir()
    )
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_workspace_root / "gympn-main"))

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from CPNModelV2.experiments.sec56_applicability.gympn import (
    shared_gympn_ppo_runner as runner,
)
from CPNModelV2.experiments.sec56_applicability.gympn import (
    train_gympn_semantic_matrix as training,
)
from CPNModelV2.integration.gympn_assignment_graph import GymPNAssignmentGraphEnv
from CPNModelV2.integration.gympn_bsg_adapter import GymPNBSGEnv
from CPNModelV2.integration.gympn_complex_scenarios import (
    PAPER_COMPLEXITY_LABELS,
    PAPER_COMPLEXITY_LEVELS,
)
from CPNModelV2.integration.gympn_semantic_matrix_scenarios import (
    BINDING_ROLE,
    FORMAL_AMBIGUITY_RATIOS,
    REFERENCE_DIRECTION,
    REFERENCE_TYPE,
    SEMANTIC_MATRIX_SCALES,
    semantic_matrix_scenario_factory,
)
from CPNModelV2.learning import PPOConfig, TokenBindingPPO


# =============================================================================
# USER CONFIGURATION
# =============================================================================

TRAINING_SEEDS_TO_EVALUATE = (9000, 9001, 9002, 9003, 9004)

# Keep this task block aligned with train_gympn_semantic_matrix.py:
#   core_matrix     : Full, AG-PPO, and the joint ablation at L1--L5 (75 tasks)
#   targeted_l3     : four single-component ablations at L3 (20 tasks)
#   full_bsg_matrix : four single-component ablations at L1--L5 (100 tasks)
TASK_BLOCK = "full_bsg_matrix"

AMBIGUITY_RATIOS = FORMAL_AMBIGUITY_RATIOS
BASE_EVALUATION_SEED = 19000
EVALUATION_EPISODES_PER_CELL = 100
CHECKPOINT_FILENAME = "best_checkpoint.pt"
SKIP_COMPLETED = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================


OUTPUT_ROOT = (
    training.PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_semantic_matrix" / "evaluation"
)


class _TraceMixin:
    """Record semantic-family and ambiguity labels without exposing them."""

    decision_records: List[Dict[str, Any]]

    def reset(self, *, seed: Optional[int] = None):
        self.decision_records = []
        return super().reset(seed=seed)

    def step(self, binding_key):
        binding_index = int(binding_key[2])
        participants, _time, transition = self.problem.pn_actions[binding_index]
        group_ids = {
            int(token.value["group_id"])
            for _place, token in participants
            if isinstance(token.value, Mapping) and "group_id" in token.value
        }
        if len(group_ids) != 1:
            raise ValueError(f"Cannot identify matrix decision group: {group_ids}")
        group_id = next(iter(group_ids))
        transition_name = str(transition._id)
        family = {
            "select_direction_candidate": REFERENCE_DIRECTION,
            "select_typed_candidate": REFERENCE_TYPE,
            "select_ordered_binding": BINDING_ROLE,
        }.get(transition_name)
        if family is None:
            raise ValueError(f"Unknown semantic-matrix transition: {transition_name}")
        ambiguous_groups = set(self.problem.gympn_semantic_ambiguity_groups)
        result = super().step(binding_key)
        self.decision_records.append({
            "group_id": group_id,
            "family": family,
            "ambiguous": group_id in ambiguous_groups,
            "correct": float(result.reward),
        })
        return result


class _TracedBSGEnv(_TraceMixin, GymPNBSGEnv):
    pass


class _TracedAGEnv(_TraceMixin, GymPNAssignmentGraphEnv):
    pass


def _evaluation_seeds() -> List[int]:
    return [
        BASE_EVALUATION_SEED * 1_000_000 + 90_000_000 + episode_index
        for episode_index in range(EVALUATION_EPISODES_PER_CELL)
    ]


def _selected_model_levels() -> Tuple[Tuple[str, str], ...]:
    if TASK_BLOCK == "core_matrix":
        return tuple(
            (model_key, level)
            for level in PAPER_COMPLEXITY_LEVELS
            for model_key in training.CORE_MATRIX_MODELS
        )
    if TASK_BLOCK == "targeted_l3":
        return tuple(
            (model_key, "C2")
            for model_key in training.TARGETED_VARIANTS
        )
    if TASK_BLOCK == "full_bsg_matrix":
        return tuple(
            (model_key, level)
            for level in PAPER_COMPLEXITY_LEVELS
            for model_key in training.TARGETED_VARIANTS
        )
    raise ValueError(
        "TASK_BLOCK must be core_matrix, targeted_l3, or full_bsg_matrix"
    )


def _tasks() -> Tuple[Tuple[str, str, int], ...]:
    return tuple(
        (model_key, level, training_seed)
        for training_seed in TRAINING_SEEDS_TO_EVALUATE
        for model_key, level in _selected_model_levels()
    )


def _checkpoint_path(task: Tuple[str, str, int]) -> Path:
    model_key, level, training_seed = task
    return training.checkpoint_path(
        model_key,
        level,
        training_seed,
        filename=CHECKPOINT_FILENAME,
    )


def _output_dir(task: Tuple[str, str, int]) -> Path:
    model_key, level, training_seed = task
    return (
        OUTPUT_ROOT / model_key / level.lower()
        / (
            f"semantic_matrix_{model_key}_{level.lower()}_"
            f"seed{training_seed}_testseed{BASE_EVALUATION_SEED}"
        )
    )


def _build_environment(model_key: str, level: str, ratio: float, horizon: float):
    factory = semantic_matrix_scenario_factory(
        level,
        ambiguity_ratios=(ratio,),
    )
    environment_class = (
        _TracedAGEnv if model_key == training.AG_PPO_MODEL else _TracedBSGEnv
    )
    return environment_class(factory, horizon=horizon, validate_graphs=False)


def _restore_trainer(
    task: Tuple[str, str, int],
    checkpoint: Mapping[str, Any],
) -> TokenBindingPPO:
    model_key, _level, _training_seed = task
    training._configure_representation(model_key)
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
    trainer.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer.policy.eval()
    return trainer


def _metric(records: List[Dict[str, Any]], predicate) -> Tuple[Any, int]:
    selected = [float(row["correct"]) for row in records if predicate(row)]
    if not selected:
        return "", 0
    return mean(selected), len(selected)


def _episode_row(
    result: Any,
    env: _TraceMixin,
    *,
    ratio: float,
    episode_index: int,
    seed: int,
) -> Dict[str, Any]:
    records = env.decision_records
    expected = int(env.problem.gympn_semantic_decision_groups)
    if len(records) != expected or result.decision_count != expected:
        raise ValueError(
            f"Expected {expected} decisions, got {len(records)} trace records "
            f"and {result.decision_count} PPO transitions"
        )
    overall, overall_count = _metric(records, lambda _row: True)
    ambiguous, ambiguous_count = _metric(records, lambda row: row["ambiguous"])
    clear, clear_count = _metric(records, lambda row: not row["ambiguous"])
    direction, direction_count = _metric(
        records, lambda row: row["family"] == REFERENCE_DIRECTION
    )
    relation_type, type_count = _metric(
        records, lambda row: row["family"] == REFERENCE_TYPE
    )
    binding_role, role_count = _metric(
        records, lambda row: row["family"] == BINDING_ROLE
    )
    return {
        "ambiguity_ratio": ratio,
        "episode": episode_index + 1,
        "seed": seed,
        "correct_decisions": float(result.episode_return),
        "overall_accuracy": overall,
        "ambiguous_accuracy": ambiguous,
        "unambiguous_accuracy": clear,
        "direction_accuracy": direction,
        "relation_type_accuracy": relation_type,
        "binding_role_accuracy": binding_role,
        "decision_count": overall_count,
        "ambiguous_count": ambiguous_count,
        "unambiguous_count": clear_count,
        "direction_count": direction_count,
        "relation_type_count": type_count,
        "binding_role_count": role_count,
    }


PER_EPISODE_FIELDS = (
    "ambiguity_ratio", "episode", "seed", "correct_decisions",
    "overall_accuracy", "ambiguous_accuracy", "unambiguous_accuracy",
    "direction_accuracy", "relation_type_accuracy", "binding_role_accuracy",
    "decision_count", "ambiguous_count", "unambiguous_count",
    "direction_count", "relation_type_count", "binding_role_count",
)


def _optional_values(rows: List[Dict[str, Any]], field: str) -> List[float]:
    return [float(row[field]) for row in rows if row[field] != ""]


def _summary_rows(per_episode_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for ratio in AMBIGUITY_RATIOS:
        rows = [row for row in per_episode_rows if row["ambiguity_ratio"] == ratio]
        summary: Dict[str, Any] = {
            "ambiguity_ratio": ratio,
            "episodes": len(rows),
        }
        for field in (
            "overall_accuracy",
            "ambiguous_accuracy",
            "unambiguous_accuracy",
            "direction_accuracy",
            "relation_type_accuracy",
            "binding_role_accuracy",
        ):
            values = _optional_values(rows, field)
            summary[f"mean_{field}"] = mean(values) if values else ""
            summary[f"std_{field}"] = pstdev(values) if values else ""
        result.append(summary)
    return result


SUMMARY_FIELDS = (
    "ambiguity_ratio", "mean_overall_accuracy", "std_overall_accuracy",
    "mean_ambiguous_accuracy", "std_ambiguous_accuracy",
    "mean_unambiguous_accuracy", "std_unambiguous_accuracy",
    "mean_direction_accuracy", "std_direction_accuracy",
    "mean_relation_type_accuracy", "std_relation_type_accuracy",
    "mean_binding_role_accuracy", "std_binding_role_accuracy", "episodes",
)


def _completed_result(task: Tuple[str, str, int]) -> Optional[Dict[str, Any]]:
    output_dir = _output_dir(task)
    episode_path = output_dir / "per_episode_results.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "evaluation_metadata.json"
    if not (episode_path.is_file() and summary_path.is_file() and metadata_path.is_file()):
        return None
    try:
        with episode_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected_rows = len(AMBIGUITY_RATIOS) * EVALUATION_EPISODES_PER_CELL
        if len(rows) != expected_rows:
            return None
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    model_key, level, training_seed = task
    expected = {
        "model_key": model_key,
        "complexity_level": level,
        "training_seed": training_seed,
        "base_evaluation_seed": BASE_EVALUATION_SEED,
        "evaluation_episodes_per_cell": EVALUATION_EPISODES_PER_CELL,
        "ambiguity_ratios": list(AMBIGUITY_RATIOS),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    return metadata


def _evaluate_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    model_key, level, training_seed = task
    if SKIP_COMPLETED:
        completed = _completed_result(task)
        if completed is not None:
            print(f"  skip complete: {model_key}/{level}/seed{training_seed}")
            return completed

    checkpoint_path = _checkpoint_path(task)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    family = training._model_family(model_key)
    if checkpoint.get("format") != training.CHECKPOINT_FORMATS[family]:
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    if tuple(checkpoint.get("scenario_names", ())) != (
        training.SEMANTIC_MATRIX_SCENARIO,
    ):
        raise ValueError(f"Checkpoint scenario mismatch: {checkpoint_path}")
    metadata = checkpoint.get("experiment_metadata", {})
    if metadata.get("complexity_level") != level:
        raise ValueError(f"Checkpoint scale mismatch: {checkpoint_path}")
    checkpoint_model_key = (
        training.AG_PPO_MODEL
        if metadata.get("model") == "AG_PPO"
        else metadata.get("ablation_variant")
    )
    if checkpoint_model_key != model_key:
        raise ValueError(f"Checkpoint model mismatch: {checkpoint_path}")
    checkpoint_seed = int(checkpoint["training_config"]["seed"])
    if checkpoint_seed != training_seed:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {training_seed}, "
            f"found {checkpoint_seed}: {checkpoint_path}"
        )
    checkpoint_iteration = int(checkpoint["iteration"])
    if checkpoint_iteration > training.FULL_ITERATIONS:
        raise ValueError(
            f"Checkpoint iteration {checkpoint_iteration} exceeds formal "
            f"budget {training.FULL_ITERATIONS}: {checkpoint_path}"
        )

    trainer = _restore_trainer(task, checkpoint)
    horizon = float(checkpoint["training_config"]["horizon"])
    seeds = _evaluation_seeds()
    per_episode_rows = []
    with runner._preserve_random_state():
        for ratio in AMBIGUITY_RATIOS:
            for episode_index, seed in enumerate(seeds):
                env = _build_environment(model_key, level, ratio, horizon)
                result = trainer.collect_episode(
                    env,
                    seed=seed,
                    deterministic=True,
                )
                per_episode_rows.append(_episode_row(
                    result,
                    env,
                    ratio=ratio,
                    episode_index=episode_index,
                    seed=seed,
                ))
            print(
                f"  {model_key}/{PAPER_COMPLEXITY_LABELS[level]}/"
                f"seed{training_seed}: rho={ratio:g} complete",
                flush=True,
            )

    output_dir = _output_dir(task)
    runner._write_csv(
        output_dir / "per_episode_results.csv",
        per_episode_rows,
        PER_EPISODE_FIELDS,
    )
    runner._write_csv(
        output_dir / "summary.csv",
        _summary_rows(per_episode_rows),
        SUMMARY_FIELDS,
    )
    result = {
        "model_key": model_key,
        "model": "AG_PPO" if model_key == training.AG_PPO_MODEL else "BSG_RL",
        "ablation_variant": None if model_key == training.AG_PPO_MODEL else model_key,
        "complexity_level": level,
        "paper_complexity_level": PAPER_COMPLEXITY_LABELS[level],
        "training_seed": training_seed,
        "base_evaluation_seed": BASE_EVALUATION_SEED,
        "evaluation_episodes_per_cell": EVALUATION_EPISODES_PER_CELL,
        "ambiguity_ratios": list(AMBIGUITY_RATIOS),
        "evaluation_protocol": "paired_scale_ambiguity_matrix_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_iteration": checkpoint_iteration,
        "formal_training_iterations": training.FULL_ITERATIONS,
        "decision_groups": SEMANTIC_MATRIX_SCALES[level].decision_groups,
        "output_dir": str(output_dir.resolve()),
    }
    runner._write_json(output_dir / "evaluation_metadata.json", result)
    return result


def main() -> int:
    if TRAINING_SEEDS_TO_EVALUATE != (9000, 9001, 9002, 9003, 9004):
        raise ValueError("Formal evaluation seeds must be 9000--9004")
    if tuple(AMBIGUITY_RATIOS) != FORMAL_AMBIGUITY_RATIOS:
        raise ValueError("Formal evaluation must retain the five matrix rows")
    tasks = _tasks()
    missing = [str(_checkpoint_path(task)) for task in tasks if not _checkpoint_path(task).is_file()]
    if missing:
        raise FileNotFoundError("Missing selected checkpoints:\n" + "\n".join(missing))
    print("GymPN scale--semantic-ambiguity held-out evaluation")
    print(f"  task block: {TASK_BLOCK}")
    print(f"  tasks    : {len(tasks)}")
    print(f"  cells    : {len(tasks) * len(AMBIGUITY_RATIOS)}")
    print(f"  episodes : {EVALUATION_EPISODES_PER_CELL} per cell")
    print(f"  device   : {DEVICE}")
    for task in tasks:
        _evaluate_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
