from __future__ import annotations

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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean, pstdev
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from CPNModelV2.graph import (
    GenericGraphFeatureSchema,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.integration.gympn_scenarios import PAPER_SCENARIO_NAMES
from CPNModelV2.integration.gympn_suite import (
    build_paper_suite_environment,
    evaluate_rule,
    macro_average,
)
from CPNModelV2.learning import PPOConfig, TokenBindingPPO


CHECKPOINT_FORMAT = "sec56_shared_ppo_unconfigured_checkpoint"
EXPERIMENT_METADATA: Dict[str, Any] = {}
BASELINE_CACHE_ROOT: Optional[Path] = None
SCORE_NAME = "completed_tasks"
BASELINE_RULES: Tuple[str, ...] = ("random", "shortest_effect")

# Representation components must be injected explicitly by the Section 5.6
# entry point.  In particular, BSG_RL injects Generic Compact V5; this module
# has no implicit policy or tensorizer fallback.
PolicyNetwork: Any = None
GraphTensorizer: Any = None
fit_graph_schema: Any = None


def fit_numeric_feature_normalizer(graphs: Sequence[Any], schema: Any) -> Any:
    """Fit the active representation's training-only numeric statistics."""

    return GenericNumericFeatureNormalizer.fit(graphs, schema)


def restore_graph_feature_schema(values: Mapping[str, Any]) -> Any:
    return GenericGraphFeatureSchema.from_dict(values)


def restore_numeric_feature_normalizer(
    values: Mapping[str, Any],
    schema: Any,
) -> Any:
    return GenericNumericFeatureNormalizer.from_dict(values, schema)

# =============================================================================
# USER CONFIGURATION: edit this section, then run this file directly.
# =============================================================================

RUN_PROFILE = "full"  # "smoke", "pilot", or "full"
SEED = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTO_RESUME = True

HORIZON = 10.0
CASE_MULTIPLIER = 1
ROLLOUT_BATCH_SIZE = 8

# Each iteration contains the same number of episodes from every scenario.
SMOKE_ITERATIONS = 2
SMOKE_EPISODES_PER_SCENARIO = 1
SMOKE_VALIDATE_EVERY = 1
SMOKE_VALIDATION_EPISODES_PER_SCENARIO = 2
SMOKE_CALIBRATION_EPISODES_PER_SCENARIO = 1

PILOT_ITERATIONS = 20
PILOT_EPISODES_PER_SCENARIO = 2
PILOT_VALIDATE_EVERY = 2
PILOT_VALIDATION_EPISODES_PER_SCENARIO = 10
PILOT_CALIBRATION_EPISODES_PER_SCENARIO = 2

FULL_ITERATIONS = 200
FULL_EPISODES_PER_SCENARIO = 4
FULL_VALIDATE_EVERY = 10
FULL_VALIDATION_EPISODES_PER_SCENARIO = 50
FULL_CALIBRATION_EPISODES_PER_SCENARIO = 4

HIDDEN_DIM = 64
CATEGORY_DIM = 16
REFERENCE_LAYERS = 0
FACTOR_ROUNDS = 2
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

# Optional archival checkpoints.  Recovery and model selection always use
# last_checkpoint.pt and best_checkpoint.pt; archives are immutable candidates
# for later validation-only checkpoint analysis.  An interval of zero keeps
# the historical Section 5.6 behavior unchanged.
CHECKPOINT_ARCHIVE_START = 0
CHECKPOINT_ARCHIVE_INTERVAL = 0
ARCHIVE_VALIDATION_IMPROVEMENTS = False
CHECKPOINT_ARCHIVE_DIRECTORY = "checkpoint_archive"

PACKAGE_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if parent.name == "CPNModelV2"
)
OUTPUT_ROOT = PACKAGE_ROOT / "results" / "gympn" / "paper_suite"
RUN_NAME = f"shared_ppo_{RUN_PROFILE}_seed{SEED}"

# =============================================================================


@dataclass(frozen=True)
class TrainingConfig:
    profile: str
    seed: int
    device: str
    output_dir: Path
    horizon: float
    case_multiplier: int
    iterations: int
    episodes_per_scenario: int
    rollout_batch_size: int
    validate_every: int
    validation_episodes_per_scenario: int
    calibration_episodes_per_scenario: int
    hidden_dim: int
    category_dim: int
    reference_layers: int
    factor_rounds: int
    dropout: float
    learning_rate: float
    update_epochs: int
    minibatch_size: int
    ppo_device_microbatch_size: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    auto_resume: bool
    checkpoint_archive_start: int
    checkpoint_archive_interval: int
    archive_validation_improvements: bool
    checkpoint_archive_directory: str


def configured_run() -> TrainingConfig:
    if not isinstance(CASE_MULTIPLIER, int) or isinstance(CASE_MULTIPLIER, bool):
        raise TypeError("CASE_MULTIPLIER must be an integer")
    if CASE_MULTIPLIER < 1:
        raise ValueError("CASE_MULTIPLIER must be at least 1")
    if CHECKPOINT_ARCHIVE_START < 0:
        raise ValueError("CHECKPOINT_ARCHIVE_START must be non-negative")
    if CHECKPOINT_ARCHIVE_INTERVAL < 0:
        raise ValueError("CHECKPOINT_ARCHIVE_INTERVAL must be non-negative")
    archive_dir = Path(CHECKPOINT_ARCHIVE_DIRECTORY)
    if (
        not CHECKPOINT_ARCHIVE_DIRECTORY
        or archive_dir.is_absolute()
        or archive_dir.name != CHECKPOINT_ARCHIVE_DIRECTORY
    ):
        raise ValueError(
            "CHECKPOINT_ARCHIVE_DIRECTORY must be one directory name"
        )
    profiles = {
        "smoke": (
            SMOKE_ITERATIONS,
            SMOKE_EPISODES_PER_SCENARIO,
            SMOKE_VALIDATE_EVERY,
            SMOKE_VALIDATION_EPISODES_PER_SCENARIO,
            SMOKE_CALIBRATION_EPISODES_PER_SCENARIO,
        ),
        "pilot": (
            PILOT_ITERATIONS,
            PILOT_EPISODES_PER_SCENARIO,
            PILOT_VALIDATE_EVERY,
            PILOT_VALIDATION_EPISODES_PER_SCENARIO,
            PILOT_CALIBRATION_EPISODES_PER_SCENARIO,
        ),
        "full": (
            FULL_ITERATIONS,
            FULL_EPISODES_PER_SCENARIO,
            FULL_VALIDATE_EVERY,
            FULL_VALIDATION_EPISODES_PER_SCENARIO,
            FULL_CALIBRATION_EPISODES_PER_SCENARIO,
        ),
    }
    try:
        iterations, episodes, validate_every, validation, calibration = (
            profiles[RUN_PROFILE]
        )
    except KeyError as exc:
        raise ValueError("RUN_PROFILE must be smoke, pilot, or full") from exc
    device = DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return TrainingConfig(
        profile=RUN_PROFILE,
        seed=SEED,
        device=device,
        output_dir=(OUTPUT_ROOT / RUN_NAME).resolve(),
        horizon=HORIZON,
        case_multiplier=CASE_MULTIPLIER,
        iterations=iterations,
        episodes_per_scenario=episodes,
        rollout_batch_size=ROLLOUT_BATCH_SIZE,
        validate_every=validate_every,
        validation_episodes_per_scenario=validation,
        calibration_episodes_per_scenario=calibration,
        hidden_dim=HIDDEN_DIM,
        category_dim=CATEGORY_DIM,
        reference_layers=REFERENCE_LAYERS,
        factor_rounds=FACTOR_ROUNDS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        ppo_device_microbatch_size=PPO_DEVICE_MICROBATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_ratio=CLIP_RATIO,
        value_coefficient=VALUE_COEFFICIENT,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        max_grad_norm=MAX_GRAD_NORM,
        auto_resume=AUTO_RESUME,
        checkpoint_archive_start=CHECKPOINT_ARCHIVE_START,
        checkpoint_archive_interval=CHECKPOINT_ARCHIVE_INTERVAL,
        archive_validation_improvements=ARCHIVE_VALIDATION_IMPROVEMENTS,
        checkpoint_archive_directory=CHECKPOINT_ARCHIVE_DIRECTORY,
    )


def _seed(config: TrainingConfig, offset: int, scenario_index: int, index: int) -> int:
    return (
        config.seed * 1_000_000
        + offset
        + scenario_index * 100_000
        + index
    )


def _specifications(
    config: TrainingConfig,
    *,
    count_per_scenario: int,
    offset: int,
) -> List[Tuple[str, int]]:
    return [
        (
            scenario_name,
            _seed(config, offset, scenario_index, episode_index),
        )
        for scenario_index, scenario_name in enumerate(PAPER_SCENARIO_NAMES)
        for episode_index in range(count_per_scenario)
    ]


def _collect_calibration_graphs(config: TrainingConfig) -> list:
    graphs = []
    specifications = _specifications(
        config,
        count_per_scenario=config.calibration_episodes_per_scenario,
        offset=10_000_000,
    )
    for scenario_name, seed in specifications:
        rng = random.Random(seed)
        env = build_paper_suite_environment(
            scenario_name,
            horizon=config.horizon,
            case_multiplier=config.case_multiplier,
        )
        graph = env.reset(seed=seed)
        while True:
            graphs.append(graph)
            step = env.step(rng.choice(list(graph.action_keys.values())))
            graph = step.observation
            if step.terminated:
                break
    return graphs


def _collect_training_episodes(
    trainer: TokenBindingPPO,
    config: TrainingConfig,
    *,
    iteration: int,
) -> Tuple[list, Dict[str, List[float]]]:
    specifications = _specifications(
        config,
        count_per_scenario=config.episodes_per_scenario,
        offset=20_000_000 + iteration * 10_000,
    )
    random.Random(config.seed + iteration).shuffle(specifications)
    episodes = []
    scores = {name: [] for name in PAPER_SCENARIO_NAMES}
    for start in range(0, len(specifications), config.rollout_batch_size):
        chunk = specifications[start : start + config.rollout_batch_size]
        results = trainer.collect_episodes_batched(
            [
                build_paper_suite_environment(
                    name,
                    horizon=config.horizon,
                    case_multiplier=config.case_multiplier,
                )
                for name, _seed_value in chunk
            ],
            seeds=[seed for _name, seed in chunk],
            deterministic=False,
        )
        episodes.extend(results)
        for (scenario_name, _seed_value), result in zip(chunk, results):
            scores[scenario_name].append(-float(result.makespan))
    return episodes, scores


def _evaluate_policy(
    trainer: TokenBindingPPO,
    config: TrainingConfig,
    *,
    count_per_scenario: int,
    offset: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, List[float]]:
    scores = {name: [] for name in PAPER_SCENARIO_NAMES}
    total = len(PAPER_SCENARIO_NAMES) * count_per_scenario
    completed = 0
    with _preserve_random_state():
        for scenario_index, scenario_name in enumerate(PAPER_SCENARIO_NAMES):
            seeds = [
                _seed(config, offset, scenario_index, episode_index)
                for episode_index in range(count_per_scenario)
            ]
            for start in range(0, len(seeds), config.rollout_batch_size):
                chunk = seeds[start : start + config.rollout_batch_size]
                results = trainer.evaluate_episodes_batched(
                    [
                        build_paper_suite_environment(
                            scenario_name,
                            horizon=config.horizon,
                            case_multiplier=config.case_multiplier,
                        )
                        for _seed_value in chunk
                    ],
                    seeds=chunk,
                    deterministic=True,
                )
                scores[scenario_name].extend(
                    -float(result.makespan) for result in results
                )
                completed += len(results)
                if progress_callback is not None:
                    progress_callback(completed, total)
    return scores


def _evaluate_baselines(
    config: TrainingConfig,
) -> List[Dict[str, Any]]:
    rows = []
    for scenario_index, scenario_name in enumerate(PAPER_SCENARIO_NAMES):
        seeds = [
            _seed(config, 70_000_000, scenario_index, episode_index)
            for episode_index in range(
                config.validation_episodes_per_scenario
            )
        ]
        for rule in BASELINE_RULES:
            scores = evaluate_rule(
                scenario_name,
                horizon=config.horizon,
                seeds=seeds,
                rule=rule,
            )
            rows.append({
                "scenario": scenario_name,
                "method": rule,
                f"mean_{SCORE_NAME}": mean(scores),
                f"std_{SCORE_NAME}": pstdev(scores),
                "episodes": len(scores),
            })
    return rows


@contextmanager
def _preserve_random_state() -> Iterator[None]:
    """Keep evaluation/calibration from changing the training RNG stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _baseline_cache_path(config: TrainingConfig) -> Optional[Path]:
    if BASELINE_CACHE_ROOT is None:
        return None
    complexity = str(EXPERIMENT_METADATA.get("complexity_level", "standard"))
    scenario_key = "__".join(str(name) for name in PAPER_SCENARIO_NAMES)
    safe_scenario = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in scenario_key
    )
    return (
        BASELINE_CACHE_ROOT
        / f"horizon_{config.horizon:g}"
        / f"episodes_{config.validation_episodes_per_scenario}"
        / f"seed{config.seed}"
        / complexity.lower()
        / f"{safe_scenario}.csv"
    )


def _evaluate_or_load_baselines(
    config: TrainingConfig,
) -> Tuple[List[Dict[str, Any]], bool]:
    cache_path = _baseline_cache_path(config)
    if cache_path is not None:
        cached = _read_csv(cache_path)
        expected_rows = len(PAPER_SCENARIO_NAMES) * len(BASELINE_RULES)
        if len(cached) == expected_rows:
            return cached, True
    with _preserve_random_state():
        rows = _evaluate_baselines(config)
    if cache_path is not None:
        _write_csv(
            cache_path,
            rows,
            (
                "scenario", "method", f"mean_{SCORE_NAME}",
                f"std_{SCORE_NAME}", "episodes",
            ),
        )
    return rows, False


def _write_status(
    config: TrainingConfig,
    phase: str,
    iteration: int,
    **details: Any,
) -> None:
    _write_json(config.output_dir / "run_status.json", {
        "status": "running",
        "phase": phase,
        "iteration": iteration,
        **details,
    })


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_temporary_path(path: Path) -> Path:
    """Return an atomic-write path that remains usable near Windows MAX_PATH."""

    preferred = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if os.name != "nt" or len(os.path.abspath(preferred)) < 248:
        return preferred

    # Section 5.6 result directories are intentionally descriptive.  A normal
    # ``<filename>.tmp.<pid>`` suffix can therefore push an otherwise valid
    # file over the legacy 260-character Windows path limit.  Keep the
    # temporary name unique per process and target while leaving the final
    # public filename unchanged.
    target_code = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:6]
    return path.parent / f"~{os.getpid():x}{target_code}.tmp"


def _best_iteration(history: Sequence[Mapping[str, Any]]) -> int:
    best_iterations = [
        int(row["iteration"])
        for row in history
        if row.get("is_best")
    ]
    return max(best_iterations, default=0)


def _save_checkpoint(
    path: Path,
    trainer: TokenBindingPPO,
    schema: GenericGraphFeatureSchema,
    normalizer: GenericNumericFeatureNormalizer,
    policy_config: Mapping[str, Any],
    ppo_config: PPOConfig,
    config: TrainingConfig,
    *,
    iteration: int,
    best_score: float,
    history: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": trainer.policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "schema": schema.to_dict(),
        "normalizer": normalizer.to_dict(),
        "policy_config": dict(policy_config),
        "ppo_config": asdict(ppo_config),
        "training_config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "scenario_names": list(PAPER_SCENARIO_NAMES),
        "score_name": SCORE_NAME,
        "experiment_metadata": dict(EXPERIMENT_METADATA),
        "iteration": iteration,
        "best_validation_macro_score": best_score,
        "best_iteration": _best_iteration(history),
        "history": list(history),
    }, temporary)
    os.replace(temporary, path)


def _periodic_checkpoint_due(
    config: TrainingConfig,
    iteration: int,
) -> bool:
    return (
        config.checkpoint_archive_interval > 0
        and iteration > config.checkpoint_archive_start
        and iteration % config.checkpoint_archive_interval == 0
    )


def _archive_checkpoint(
    trainer: TokenBindingPPO,
    schema: GenericGraphFeatureSchema,
    normalizer: GenericNumericFeatureNormalizer,
    policy_config: Mapping[str, Any],
    ppo_config: PPOConfig,
    config: TrainingConfig,
    *,
    iteration: int,
    best_score: float,
    history: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
    validation_score: Optional[float],
) -> Optional[Path]:
    if not reasons:
        return None
    archive_dir = config.output_dir / config.checkpoint_archive_directory
    archive_dir.mkdir(parents=True, exist_ok=True)
    filename_codes = []
    if "validation_best" in reasons:
        filename_codes.append("b")
    if "periodic" in reasons:
        filename_codes.append("p")
    filename_codes.extend(
        reason[:1].lower()
        for reason in reasons
        if reason not in {"validation_best", "periodic"}
    )
    # Section 5.6 run directories are already descriptive and can approach the
    # legacy Windows MAX_PATH limit.  Keep archive filenames deliberately
    # compact so both the final path and the atomic ``.tmp.<pid>`` path remain
    # below that limit.  Human-readable reasons stay in manifest.json.
    checkpoint_path = archive_dir / (
        f"c{iteration:03d}{''.join(filename_codes)}.pt"
    )
    _save_checkpoint(
        checkpoint_path,
        trainer,
        schema,
        normalizer,
        policy_config,
        ppo_config,
        config,
        iteration=iteration,
        best_score=best_score,
        history=history,
    )

    manifest_path = archive_dir / "manifest.json"
    manifest: Dict[str, Any] = {
        "selection_policy": (
            "Candidates only. Select a checkpoint using validation data "
            "before inspecting held-out test results."
        ),
        "metric": "validation_macro_score (higher is better)",
        "checkpoints": [],
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest.get("checkpoints", [])
        if int(entry["iteration"]) != iteration
    ]
    entries.append({
        "iteration": iteration,
        "file": checkpoint_path.name,
        "reasons": sorted(set(reasons)),
        "validation_macro_score": validation_score,
        "best_validation_macro_score": float(best_score),
        "best_iteration": _best_iteration(history),
    })
    entries.sort(key=lambda entry: int(entry["iteration"]))
    manifest["checkpoints"] = entries
    _write_json(manifest_path, manifest)
    return checkpoint_path


def run_training(config: TrainingConfig) -> Dict[str, Any]:
    if not callable(PolicyNetwork):
        raise RuntimeError("Section 5.6 PolicyNetwork was not configured")
    if not callable(GraphTensorizer):
        raise RuntimeError("Section 5.6 GraphTensorizer was not configured")
    if not callable(fit_graph_schema):
        raise RuntimeError("Section 5.6 graph-schema fitter was not configured")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    _write_status(config, "initialization", 0)
    _write_json(config.output_dir / "run_config.json", {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "scenario_names": list(PAPER_SCENARIO_NAMES),
        "score_name": SCORE_NAME,
        "experiment_metadata": dict(EXPERIMENT_METADATA),
    })

    policy_config = {
        "hidden_dim": config.hidden_dim,
        "category_dim": config.category_dim,
        "reference_layers": config.reference_layers,
        "factor_rounds": config.factor_rounds,
        "dropout": config.dropout,
    }
    ppo_config = PPOConfig(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_ratio=config.clip_ratio,
        value_coefficient=config.value_coefficient,
        entropy_coefficient=config.entropy_coefficient,
        learning_rate=config.learning_rate,
        update_epochs=config.update_epochs,
        minibatch_size=config.minibatch_size,
        device_microbatch_size=config.ppo_device_microbatch_size,
        max_grad_norm=config.max_grad_norm,
    )
    checkpoint_path = config.output_dir / "last_checkpoint.pt"
    checkpoint = None
    if config.auto_resume and checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")

    if checkpoint is None:
        _write_status(config, "calibration", 0)
        with _preserve_random_state():
            calibration = _collect_calibration_graphs(config)
        schema = fit_graph_schema(calibration)
        normalizer = fit_numeric_feature_normalizer(calibration, schema)
        start_iteration = 0
        best_score = -float("inf")
        history: List[Dict[str, Any]] = []
    else:
        schema = restore_graph_feature_schema(checkpoint["schema"])
        normalizer = restore_numeric_feature_normalizer(
            checkpoint["normalizer"], schema
        )
        policy_config = dict(checkpoint["policy_config"])
        ppo_config = PPOConfig(**checkpoint["ppo_config"])
        start_iteration = int(checkpoint["iteration"])
        best_score = float(checkpoint["best_validation_macro_score"])
        history = list(checkpoint.get("history", []))

    trainer = TokenBindingPPO(
        PolicyNetwork(schema, **policy_config),
        GraphTensorizer(
            schema,
            normalizer=normalizer,
            validate_graphs=False,
        ),
        config=ppo_config,
        device=config.device,
    )
    if checkpoint is not None:
        trainer.policy.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    _write_status(
        config,
        "checkpoint_loaded" if checkpoint is not None else "model_initialized",
        start_iteration,
        resumed=checkpoint is not None,
    )
    _write_status(config, "baseline_evaluation", start_iteration)
    baselines, baseline_cache_hit = _evaluate_or_load_baselines(config)
    _write_csv(
        config.output_dir / "baseline_summary.csv",
        baselines,
        (
            "scenario", "method", f"mean_{SCORE_NAME}",
            f"std_{SCORE_NAME}", "episodes",
        ),
    )
    _write_status(
        config,
        "baseline_complete",
        start_iteration,
        baseline_cache_hit=baseline_cache_hit,
    )
    print("Section 5.6 GymPN controlled shared-PPO training")
    print(f"  profile       : {config.profile}")
    print(f"  interpreter   : {os.sys.executable}")
    print(f"  device        : {config.device}")
    print(f"  scenarios     : {len(PAPER_SCENARIO_NAMES)}")
    print(f"  iterations    : {start_iteration} -> {config.iterations}")
    print(f"  output        : {config.output_dir}")

    scenario_history = [
        row
        for row in _read_csv(config.output_dir / "validation_by_scenario.csv")
        if int(row.get("iteration", 0)) <= start_iteration
    ]
    # A process may stop after last_checkpoint.pt and best_checkpoint.pt were
    # written but before its periodic archive was committed (for example, an
    # archive-path error).  When resuming exactly on an archive boundary,
    # recreate that immutable candidate before advancing to the next round.
    if start_iteration > 0 and _periodic_checkpoint_due(config, start_iteration):
        resume_row = next(
            (
                row
                for row in reversed(history)
                if int(row.get("iteration", -1)) == start_iteration
            ),
            {},
        )
        resume_reasons = ["periodic"]
        if (
            config.archive_validation_improvements
            and resume_row.get("is_best")
        ):
            resume_reasons.append("validation_best")
        resumed_archive = _archive_checkpoint(
            trainer,
            schema,
            normalizer,
            policy_config,
            ppo_config,
            config,
            iteration=start_iteration,
            best_score=best_score,
            history=history,
            reasons=resume_reasons,
            validation_score=(
                float(resume_row["validation_macro_score"])
                if "validation_macro_score" in resume_row else None
            ),
        )
        if resumed_archive is not None:
            print(
                f"  archive resume : {resumed_archive.name} "
                f"({', '.join(resume_reasons)})",
                flush=True,
            )
    for iteration in range(start_iteration + 1, config.iterations + 1):
        iteration_started = time.perf_counter()
        _write_status(
            config,
            "training_rollout",
            iteration,
            target_iterations=config.iterations,
        )
        episodes, train_scores = _collect_training_episodes(
            trainer,
            config,
            iteration=iteration,
        )
        _write_status(config, "ppo_update", iteration)
        metrics = trainer.update(episodes, seed=config.seed + iteration)
        training_seconds = time.perf_counter() - iteration_started
        row: Dict[str, Any] = {
            "iteration": iteration,
            "train_macro_score": macro_average(train_scores),
            "train_worst_scenario_score": min(
                mean(values) for values in train_scores.values()
            ),
            "train_mean_return": mean(
                episode.episode_return for episode in episodes
            ),
            **metrics,
            "iteration_seconds": training_seconds,
            "validation_seconds": 0.0,
        }
        del episodes
        if (
            iteration % config.validate_every == 0
            or iteration == config.iterations
        ):
            validation_started = time.perf_counter()
            validation_total = (
                len(PAPER_SCENARIO_NAMES)
                * config.validation_episodes_per_scenario
            )
            _write_status(
                config,
                "validation",
                iteration,
                validation_completed=0,
                validation_total=validation_total,
            )

            def report_validation_progress(completed: int, total: int) -> None:
                _write_status(
                    config,
                    "validation",
                    iteration,
                    validation_completed=completed,
                    validation_total=total,
                )

            validation = _evaluate_policy(
                trainer,
                config,
                count_per_scenario=(
                    config.validation_episodes_per_scenario
                ),
                offset=80_000_000,
                progress_callback=report_validation_progress,
            )
            row["validation_seconds"] = time.perf_counter() - validation_started
            validation_macro = macro_average(validation)
            row["validation_macro_score"] = validation_macro
            row["validation_worst_scenario_score"] = min(
                mean(values) for values in validation.values()
            )
            improved = validation_macro > best_score
            if improved:
                best_score = validation_macro
            row["best_validation_macro_score"] = best_score
            row["is_best"] = improved
            for scenario_name, values in validation.items():
                scenario_history.append({
                    "iteration": iteration,
                    "scenario": scenario_name,
                    f"mean_{SCORE_NAME}": mean(values),
                    f"std_{SCORE_NAME}": pstdev(values),
                    f"min_{SCORE_NAME}": min(values),
                    f"max_{SCORE_NAME}": max(values),
                    "episodes": len(values),
                })
            _write_csv(
                config.output_dir / "validation_by_scenario.csv",
                scenario_history,
                (
                    "iteration", "scenario", f"mean_{SCORE_NAME}",
                    f"std_{SCORE_NAME}", f"min_{SCORE_NAME}",
                    f"max_{SCORE_NAME}", "episodes",
                ),
            )
        row["iteration_total_seconds"] = time.perf_counter() - iteration_started
        history.append(row)
        _write_csv(
            config.output_dir / "training_history.csv",
            history,
            (
                "iteration", "train_macro_score",
                "train_worst_scenario_score", "train_mean_return",
                "loss", "policy_loss", "value_loss", "entropy",
                "approx_kl", "clip_fraction", "grad_norm",
                "iteration_seconds", "validation_seconds",
                "iteration_total_seconds", "validation_macro_score",
                "validation_worst_scenario_score",
                "best_validation_macro_score", "is_best",
            ),
        )
        _write_status(config, "checkpoint_saving", iteration)
        _save_checkpoint(
            checkpoint_path,
            trainer,
            schema,
            normalizer,
            policy_config,
            ppo_config,
            config,
            iteration=iteration,
            best_score=best_score,
            history=history,
        )
        if row.get("is_best"):
            _save_checkpoint(
                config.output_dir / "best_checkpoint.pt",
                trainer,
                schema,
                normalizer,
                policy_config,
                ppo_config,
                config,
                iteration=iteration,
                best_score=best_score,
                history=history,
            )
        archive_reasons = []
        if _periodic_checkpoint_due(config, iteration):
            archive_reasons.append("periodic")
        if (
            config.archive_validation_improvements
            and row.get("is_best")
            and iteration >= config.checkpoint_archive_start
        ):
            archive_reasons.append("validation_best")
        archived_path = _archive_checkpoint(
            trainer,
            schema,
            normalizer,
            policy_config,
            ppo_config,
            config,
            iteration=iteration,
            best_score=best_score,
            history=history,
            reasons=archive_reasons,
            validation_score=(
                float(row["validation_macro_score"])
                if "validation_macro_score" in row else None
            ),
        )
        if archived_path is not None:
            print(
                f"  archived       : {archived_path.name} "
                f"({', '.join(archive_reasons)})",
                flush=True,
            )
        validation_macro_text = (
            f"{row['validation_macro_score']:.3f}"
            if "validation_macro_score" in row else "not_run"
        )
        validation_worst_text = (
            f"{row['validation_worst_scenario_score']:.3f}"
            if "validation_worst_scenario_score" in row else "not_run"
        )
        print(
            f"  iter={iteration:03d} "
            f"train_macro={row['train_macro_score']:.3f} "
            f"val_macro={validation_macro_text} "
            f"worst={validation_worst_text} "
            f"train_seconds={row['iteration_seconds']:.2f} "
            f"total_seconds={row['iteration_total_seconds']:.2f}",
            flush=True,
        )
        _write_status(
            config,
            "iteration_complete",
            iteration,
            best_validation_macro_score=best_score,
            target_iterations=config.iterations,
        )

    result = {
        "status": "complete",
        "profile": config.profile,
        "iterations": config.iterations,
        "scenario_count": len(PAPER_SCENARIO_NAMES),
        "score_name": SCORE_NAME,
        "best_validation_macro_score": best_score,
        "output_dir": str(config.output_dir),
        "experiment_metadata": dict(EXPERIMENT_METADATA),
    }
    _write_json(config.output_dir / "training_result.json", result)
    _write_json(config.output_dir / "run_status.json", {
        **result,
        "phase": "complete",
    })
    return result


def main() -> None:
    run_training(configured_run())


if __name__ == "__main__":
    main()
