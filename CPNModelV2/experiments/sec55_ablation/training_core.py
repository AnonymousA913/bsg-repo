from __future__ import annotations

"""Isolated Section 5.5 training core for Generic Compact ablations.

This file intentionally reuses the frozen Section 5.2 data generation, reward,
PPO, graph-building, calibration, and logging helpers while substituting only
``GenericCompactAblationPolicyNetwork`` for the policy forward semantics.
No existing production source file is modified.
"""

from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import random
from statistics import mean, pstdev
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    evaluate_batch,
)
from CPNModelV2.experiments.sec52_fjsp_performance.train_generic_compact_10x5 import (
    CHECKPOINT_FORMAT as SEC52_CHECKPOINT_FORMAT,
    POLICY_GENERIC_COMPACT,
    TrainingConfig,
    TRAINING_FIELDS,
    VALIDATION_FIELDS,
    VALIDATE_INTERNAL_GRAPHS,
    _archive_checkpoint,
    _atomic_torch_save,
    _collect_calibration_graphs,
    _evaluate_spt,
    _periodic_checkpoint_due,
    _seed_everything,
    _training_instance_seed,
    _validation_rows,
    _write_csv,
    _write_json,
    _write_training_instance_manifest,
)
from CPNModelV2.graph import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.integration.fjsp_v5_generic_adapter import (
    FJSPV5GenericGraphBuilder,
)
from CPNModelV2.integration.public_fjsp_dataset import (
    FJSFileV5NetFactory,
    PublicFJSPDataset,
)
from CPNModelV2.learning import (
    CPNGraphEnv,
    PPOConfig,
    SongSD1Config,
    SongSD1V5NetFactory,
    TokenBindingPPO,
    WangSD2Config,
    WangSD2V5NetFactory,
)
from CPNModelV2.learning.generic_compact_ablation_policy import (
    FLAT_BINDING_VARIANT,
    GenericCompactAblationPolicyNetwork,
    NO_REFERENCE_VARIANT,
    NO_REFERENCE_FLAT_BINDING_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
    normalize_ablation_variant,
)


CHECKPOINT_FORMAT = "generic_compact_sec55_ablation_checkpoint_v1"
TRAINING_ENGINE_VERSION = "sec55_isolated_ablation_engine_v1"
MEMORY_CLEANUP_EVERY = 25
FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT = True
ABLATION_VARIANTS = (
    FLAT_BINDING_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
    NO_REFERENCE_VARIANT,
    NO_REFERENCE_FLAT_BINDING_VARIANT,
)
# ABLATION_VARIANTS = (
#     NO_REFERENCE_VARIANT,
#     NO_RELATION_DIRECTION_VARIANT,
#     FLAT_BINDING_VARIANT,
# )


@dataclass(frozen=True)
class AblationTrainingConfig(TrainingConfig):
    ablation_variant: str = NO_REFERENCE_VARIANT
    reserved_test_dataset: Path = Path(".")

    def validate(self) -> None:
        super().validate()
        normalize_ablation_variant(self.ablation_variant)
        if self.ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(
                "AblationTrainingConfig must use a Section 5.5 ablation variant"
            )
        if not self.reserved_test_dataset.is_dir():
            raise FileNotFoundError(self.reserved_test_dataset)
        if self.validation_dataset.resolve() == self.reserved_test_dataset.resolve():
            raise ValueError(
                "Validation and reserved test datasets must be different directories"
            )


def _make_factory(config: AblationTrainingConfig) -> Any:
    if config.distribution == "SD1":
        instance_config = SongSD1Config(
            num_jobs=config.num_jobs,
            num_machines=config.num_machines,
            min_operations=config.min_operations,
            max_operations=config.max_operations,
            min_eligible_machines=config.min_eligible_machines,
            max_eligible_machines=config.max_eligible_machines,
        )
        return SongSD1V5NetFactory(instance_config)
    instance_config = WangSD2Config(
        num_jobs=config.num_jobs,
        num_machines=config.num_machines,
        operations_per_job=config.num_machines,
        min_eligible_machines=config.min_eligible_machines,
        max_eligible_machines=config.max_eligible_machines,
    )
    return WangSD2V5NetFactory(instance_config)


def _make_environment(factory: Any, config: AblationTrainingConfig) -> CPNGraphEnv:
    return CPNGraphEnv(
        factory,
        graph_builder=FJSPV5GenericGraphBuilder(validate=VALIDATE_INTERNAL_GRAPHS),
        reward_scale=config.reward_scale,
        reward_mode=config.reward_mode,
        potential_coefficient=config.potential_coefficient,
    )


def _make_file_environment(path: Path, config: AblationTrainingConfig) -> CPNGraphEnv:
    return CPNGraphEnv(
        FJSFileV5NetFactory(path),
        graph_builder=FJSPV5GenericGraphBuilder(validate=VALIDATE_INTERNAL_GRAPHS),
        reward_scale=config.reward_scale,
        reward_mode=config.reward_mode,
        potential_coefficient=config.potential_coefficient,
    )


def _evaluate_policy_streaming(
    trainer: TokenBindingPPO,
    paths: Sequence[Path],
    config: AblationTrainingConfig,
) -> Dict[str, Any]:
    """Greedy validation without retaining PPO transition trajectories.

    The Section 5.2 training helper uses ``collect_episodes_batched`` for
    validation, which retains per-decision graphs because those episodes are a
    PPO data structure.  Here validation uses the already-audited streaming
    evaluator: only the current observation is retained during rollout.
    """

    rows: List[Dict[str, Any]] = []
    trainer.policy.eval()
    for start in range(0, len(paths), config.evaluation_batch_size):
        chunk = paths[start : start + config.evaluation_batch_size]
        environments = [_make_file_environment(path, config) for path in chunk]
        outcomes, _timing = evaluate_batch(
            trainer,
            environments,
            deterministic=True,
        )
        rows.extend({
            "instance": path.name,
            "makespan": float(outcome.makespan),
            "decisions": int(outcome.decision_count),
        } for path, outcome in zip(chunk, outcomes))
    values = [row["makespan"] for row in rows]
    return {
        "mean_makespan": mean(values),
        "min_makespan": min(values),
        "max_makespan": max(values),
        "per_instance": rows,
    }


def _checkpoint_payload(
    trainer: TokenBindingPPO,
    schema: GenericGraphFeatureSchema,
    normalizer: GenericNumericFeatureNormalizer,
    config: AblationTrainingConfig,
    ppo_config: PPOConfig,
    *,
    iteration: int,
    best_validation: float,
    best_iteration: int,
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    training_config = asdict(config)
    training_config.update({
        "output_dir": str(config.output_dir),
        "validation_dataset": str(config.validation_dataset),
        "reserved_test_dataset": str(config.reserved_test_dataset),
    })
    return {
        "format": CHECKPOINT_FORMAT,
        "training_engine_version": TRAINING_ENGINE_VERSION,
        "base_checkpoint_format": SEC52_CHECKPOINT_FORMAT,
        "policy_variant": POLICY_GENERIC_COMPACT,
        "ablation_variant": config.ablation_variant,
        "model_state_dict": trainer.policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "schema": schema.to_dict(),
        "normalizer": normalizer.to_dict(),
        "training_config": training_config,
        "policy_config": {
            "hidden_dim": config.hidden_dim,
            "category_dim": config.category_dim,
            "reference_layers": config.reference_layers,
            "competition_layers": config.competition_layers,
            "dropout": config.dropout,
        },
        "ppo_config": asdict(ppo_config),
        "iteration": iteration,
        "best_validation_makespan": best_validation,
        "best_iteration": best_iteration,
        "history": list(history),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _resume_contract(config: AblationTrainingConfig) -> Mapping[str, Any]:
    return {
        "ablation_variant": config.ablation_variant,
        "distribution": config.distribution,
        "protocol_version": config.protocol_version,
        "instance_seed_scheme": config.instance_seed_scheme,
        "training_size": config.training_size,
        "num_jobs": config.num_jobs,
        "num_machines": config.num_machines,
        "min_operations": config.min_operations,
        "max_operations": config.max_operations,
        "min_eligible_machines": config.min_eligible_machines,
        "max_eligible_machines": config.max_eligible_machines,
        "seed": config.seed,
        "training_instances_per_iteration": config.training_instances_per_iteration,
        "rollout_device_batch_size": config.rollout_device_batch_size,
        "ppo_device_microbatch_size": config.ppo_device_microbatch_size,
        "validation_dataset": str(config.validation_dataset),
        "validation_instances": config.validation_instances,
        "hidden_dim": config.hidden_dim,
        "category_dim": config.category_dim,
        "reference_layers": config.reference_layers,
        "competition_layers": config.competition_layers,
        "dropout": config.dropout,
        "reward_mode": config.reward_mode,
        "reward_scale": config.reward_scale,
        "potential_coefficient": config.potential_coefficient,
        "learning_rate": config.learning_rate,
        "update_epochs": config.update_epochs,
        "minibatch_size": config.minibatch_size,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_ratio": config.clip_ratio,
        "value_coefficient": config.value_coefficient,
        "entropy_coefficient": config.entropy_coefficient,
        "max_grad_norm": config.max_grad_norm,
    }


def _build_trainer(
    config: AblationTrainingConfig,
    factory: Any,
) -> Tuple[
    TokenBindingPPO,
    GenericGraphFeatureSchema,
    GenericNumericFeatureNormalizer,
    PPOConfig,
    int,
    float,
    int,
    List[Dict[str, Any]],
]:
    checkpoint_path = config.output_dir / "last_checkpoint.pt"
    checkpoint: Optional[Dict[str, Any]] = None
    if config.auto_resume and checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=config.device,
            weights_only=False,
        )
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
        if checkpoint.get("ablation_variant") != config.ablation_variant:
            raise ValueError("Checkpoint ablation variant differs from configuration")

    policy_config = {
        "hidden_dim": config.hidden_dim,
        "category_dim": config.category_dim,
        "reference_layers": config.reference_layers,
        "competition_layers": config.competition_layers,
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
    start_iteration = 0
    best_validation = float("inf")
    best_iteration = 0
    history: List[Dict[str, Any]] = []

    if checkpoint is None:
        # Identical calibration seeds to Section 5.2 for the same training seed.
        calibration_seeds = range(
            config.seed * 10_000,
            config.seed * 10_000 + config.calibration_instances,
        )
        calibration_graphs = _collect_calibration_graphs(factory, calibration_seeds)
        schema = GenericGraphFeatureSchema.fit(calibration_graphs)
        normalizer = GenericNumericFeatureNormalizer.fit(calibration_graphs, schema)
        del calibration_graphs
    else:
        saved = dict(checkpoint.get("training_config", {}))
        saved["ablation_variant"] = checkpoint.get("ablation_variant")
        expected = _resume_contract(config)
        changed = {
            name: (saved.get(name), current)
            for name, current in expected.items()
            if saved.get(name) != current
        }
        if changed:
            raise ValueError(
                "The ablation/PPO/reward configuration differs from the checkpoint. "
                f"Restore these values or use a new output directory: {changed}"
            )
        schema = GenericGraphFeatureSchema.from_dict(checkpoint["schema"])
        normalizer = GenericNumericFeatureNormalizer.from_dict(
            checkpoint["normalizer"], schema
        )
        policy_config = dict(checkpoint["policy_config"])
        ppo_config = PPOConfig(**checkpoint["ppo_config"])
        start_iteration = int(checkpoint["iteration"])
        best_validation = float(checkpoint["best_validation_makespan"])
        best_iteration = int(checkpoint["best_iteration"])
        history = list(checkpoint.get("history", []))

    policy = GenericCompactAblationPolicyNetwork(
        schema,
        **policy_config,
        ablation_variant=config.ablation_variant,
    )
    tensorizer = GenericGraphTensorizer(
        schema,
        normalizer=normalizer,
        validate_graphs=VALIDATE_INTERNAL_GRAPHS,
    )
    trainer = TokenBindingPPO(
        policy,
        tensorizer,
        config=ppo_config,
        device=config.device,
    )
    if checkpoint is not None:
        trainer.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_random_states"):
            torch.cuda.set_rng_state_all([
                state.cpu() for state in checkpoint["cuda_random_states"]
            ])
    return (
        trainer,
        schema,
        normalizer,
        ppo_config,
        start_iteration,
        best_validation,
        best_iteration,
        history,
    )


def _maybe_cleanup_memory(iteration: int, device: str) -> None:
    if iteration % MEMORY_CLEANUP_EVERY != 0:
        return
    gc.collect()
    # empty_cache releases unused cached blocks only; it does not change model
    # tensors or PPO data.  Calling it infrequently avoids needless allocator
    # churn while helping long 20x10 runs on constrained GPUs.
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_ablation_training(config: AblationTrainingConfig) -> Dict[str, Any]:
    config.validate()
    output_has_files = config.output_dir.is_dir() and any(config.output_dir.iterdir())
    checkpoint_path = config.output_dir / "last_checkpoint.pt"
    if output_has_files and not config.auto_resume:
        raise FileExistsError(
            f"Output already contains a previous run: {config.output_dir}"
        )
    if (
        FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT
        and output_has_files
        and not checkpoint_path.is_file()
    ):
        raise FileExistsError(
            f"Output is non-empty without a resumable checkpoint: {config.output_dir}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    _write_json(config.output_dir / "run_status.json", {
        "status": "running",
        "phase": "schema_or_checkpoint_initialization",
        "iteration": 0,
        "ablation_variant": config.ablation_variant,
    })

    dataset = PublicFJSPDataset(
        config.validation_dataset,
        limit=config.validation_instances,
        expected_jobs=config.num_jobs,
        expected_machines=config.num_machines,
    )
    if config.profile == "full" and len(dataset) != 100:
        raise ValueError(
            f"Formal Section 5.5 validation requires exactly 100 instances; "
            f"got {len(dataset)} from {config.validation_dataset}"
        )

    factory = _make_factory(config)
    # Reuse the frozen Section 5.2 manifest/seed function verbatim.  Its key
    # intentionally contains the 'sec52' namespace so the generated stream is
    # matched to the already-trained Full control.
    _write_training_instance_manifest(config)
    (
        trainer,
        schema,
        normalizer,
        ppo_config,
        start_iteration,
        best_validation,
        best_iteration,
        history,
    ) = _build_trainer(config, factory)

    _write_json(config.output_dir / "run_config.json", {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "validation_dataset": str(config.validation_dataset),
        "reserved_test_dataset": str(config.reserved_test_dataset),
        "ppo": asdict(ppo_config),
        "training_engine_version": TRAINING_ENGINE_VERSION,
        "matched_full_protocol": {
            "section": "5.2",
            "training_seeds": [9000, 9001, 9002, 9003, 9004],
            "instance_seed_scheme": config.instance_seed_scheme,
            "seed_function": "reuse_sec52_training_instance_seed",
        },
    })

    spt_result = _evaluate_spt(dataset.paths)
    if start_iteration == 0 and not history:
        validation = _evaluate_policy_streaming(trainer, dataset.paths, config)
        best_validation = float(validation["mean_makespan"])
        best_iteration = 0
        history.append({
            "protocol_version": config.protocol_version,
            "distribution": config.distribution,
            "training_size": config.training_size,
            "training_seed": config.seed,
            "iteration": 0,
            "completed_training_episodes": 0,
            "validation_mean_makespan": validation["mean_makespan"],
            "validation_std_makespan": pstdev(
                row["makespan"] for row in validation["per_instance"]
            ),
            "validation_min_makespan": validation["min_makespan"],
            "validation_max_makespan": validation["max_makespan"],
            "best_validation_makespan": best_validation,
            "best_iteration": 0,
            "is_best": True,
        })
        rows = _validation_rows(0, validation, spt_result)
        _write_csv(
            config.output_dir / "best_validation_per_instance.csv",
            rows,
            VALIDATION_FIELDS,
        )
        payload = _checkpoint_payload(
            trainer,
            schema,
            normalizer,
            config,
            ppo_config,
            iteration=0,
            best_validation=best_validation,
            best_iteration=0,
            history=history,
        )
        _atomic_torch_save(payload, config.output_dir / "best_checkpoint.pt")
        _atomic_torch_save(payload, checkpoint_path)

    _write_csv(config.output_dir / "training_history.csv", history, TRAINING_FIELDS)
    _write_csv(
        config.output_dir / "validation_history.csv",
        [row for row in history if "validation_mean_makespan" in row],
        TRAINING_FIELDS,
    )
    _write_csv(
        config.output_dir / "spt_validation_per_instance.csv",
        [{"instance": row["instance"], "spt_makespan": row["makespan"]}
         for row in spt_result["per_instance"]],
        ("instance", "spt_makespan"),
    )

    environments = [
        _make_environment(factory, config)
        for _ in range(config.rollout_device_batch_size)
    ]
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.policy.parameters()
        if parameter.requires_grad
    )

    print(
        f"Section 5.5 ablation PPO: {config.ablation_variant} | "
        f"{config.distribution} {config.training_size} | seed={config.seed}"
    )
    print(f"  device        : {config.device}")
    print(f"  iterations    : {start_iteration} -> {config.iterations}")
    print(f"  train budget  : {config.training_instances_per_iteration}/iteration")
    print(f"  rollout batch : {config.rollout_device_batch_size}")
    print(f"  PPO microbatch: {config.ppo_device_microbatch_size}")
    print(f"  parameters    : {trainable_parameters:,}")
    print(f"  validation    : {len(dataset)} fixed instances")
    print(f"  reserved test : {config.reserved_test_dataset}")

    run_started = time.perf_counter()
    try:
        for iteration in range(start_iteration + 1, config.iterations + 1):
            iteration_started = time.perf_counter()
            seeds = [
                _training_instance_seed(config, iteration, index)
                for index in range(config.training_instances_per_iteration)
            ]
            episodes = []
            for start in range(0, len(seeds), config.rollout_device_batch_size):
                seed_chunk = seeds[start : start + config.rollout_device_batch_size]
                episodes.extend(trainer.collect_episodes_batched(
                    environments[:len(seed_chunk)],
                    seeds=seed_chunk,
                ))
                _write_json(config.output_dir / "run_status.json", {
                    "status": "running",
                    "phase": "rollout",
                    "iteration": iteration,
                    "completed_training_instances": len(episodes),
                    "training_instances_this_iteration": len(seeds),
                    "ablation_variant": config.ablation_variant,
                })

            # Compute scalar summaries before PPO update so the episode objects
            # can be released immediately after update, reducing the lifetime of
            # their retained TokenBindingGraph trajectories.
            train_mean_makespan = mean(ep.makespan for ep in episodes)
            train_mean_return = mean(ep.episode_return for ep in episodes)
            metrics = trainer.update(episodes, seed=config.seed + iteration)
            del episodes

            record: Dict[str, Any] = {
                "protocol_version": config.protocol_version,
                "distribution": config.distribution,
                "training_size": config.training_size,
                "training_seed": config.seed,
                "iteration": iteration,
                "completed_training_episodes": (
                    iteration * config.training_instances_per_iteration
                ),
                "train_mean_makespan": train_mean_makespan,
                "train_mean_return": train_mean_return,
                **metrics,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                # These fields are kept for schema compatibility with Section
                # 5.2. They are not used as feasibility evidence in the paper;
                # independent checking is performed during formal test evaluation.
                "invalid_action_count": None,
                "infeasible_schedule_count": None,
                "iteration_seconds": time.perf_counter() - iteration_started,
            }

            improved = False
            validation_makespan: Optional[float] = None
            should_validate = (
                iteration % config.validate_every == 0
                or iteration == config.iterations
            )
            if should_validate:
                validation_started = time.perf_counter()
                validation = _evaluate_policy_streaming(
                    trainer, dataset.paths, config
                )
                validation_makespan = float(validation["mean_makespan"])
                record.update({
                    "validation_mean_makespan": validation["mean_makespan"],
                    "validation_std_makespan": pstdev(
                        row["makespan"] for row in validation["per_instance"]
                    ),
                    "validation_min_makespan": validation["min_makespan"],
                    "validation_max_makespan": validation["max_makespan"],
                    "validation_seconds": time.perf_counter() - validation_started,
                })
                improved = validation["mean_makespan"] < best_validation
                if improved:
                    best_validation = float(validation["mean_makespan"])
                    best_iteration = iteration
                record.update({
                    "best_validation_makespan": best_validation,
                    "best_iteration": best_iteration,
                    "is_best": improved,
                })
                history.append(record)
                rows = _validation_rows(iteration, validation, spt_result)
                _write_csv(
                    config.output_dir / "latest_validation_per_instance.csv",
                    rows,
                    VALIDATION_FIELDS,
                )
                if config.save_every_validation:
                    _write_csv(
                        config.output_dir
                        / "validation_snapshots"
                        / f"iteration_{iteration:04d}.csv",
                        rows,
                        VALIDATION_FIELDS,
                    )
                payload = _checkpoint_payload(
                    trainer,
                    schema,
                    normalizer,
                    config,
                    ppo_config,
                    iteration=iteration,
                    best_validation=best_validation,
                    best_iteration=best_iteration,
                    history=history,
                )
                if improved:
                    _atomic_torch_save(
                        payload, config.output_dir / "best_checkpoint.pt"
                    )
                    _write_csv(
                        config.output_dir / "best_validation_per_instance.csv",
                        rows,
                        VALIDATION_FIELDS,
                    )
            else:
                history.append(record)

            payload = _checkpoint_payload(
                trainer,
                schema,
                normalizer,
                config,
                ppo_config,
                iteration=iteration,
                best_validation=best_validation,
                best_iteration=best_iteration,
                history=history,
            )
            _atomic_torch_save(payload, checkpoint_path)
            archive_reasons = []
            if _periodic_checkpoint_due(config, iteration):
                archive_reasons.append("periodic")
            if (
                config.archive_validation_improvements
                and improved
                and iteration >= config.checkpoint_archive_start
            ):
                archive_reasons.append("validation_best")
            archived_path = _archive_checkpoint(
                payload,
                config,
                iteration=iteration,
                reasons=archive_reasons,
                validation_makespan=validation_makespan,
            )
            if archived_path is not None:
                print(
                    f"    archived checkpoint: {archived_path.name} "
                    f"({', '.join(archive_reasons)})",
                    flush=True,
                )
            _write_csv(
                config.output_dir / "training_history.csv", history, TRAINING_FIELDS
            )
            _write_csv(
                config.output_dir / "validation_history.csv",
                [row for row in history if "validation_mean_makespan" in row],
                TRAINING_FIELDS,
            )
            if iteration % config.print_every == 0:
                val_text = (
                    f" val={record['validation_mean_makespan']:.3f}"
                    if "validation_mean_makespan" in record else ""
                )
                print(
                    f"  iter={iteration:04d} "
                    f"train={record['train_mean_makespan']:.3f} "
                    f"loss={record['loss']:.4f} entropy={record['entropy']:.4f}"
                    f"{val_text} best={best_validation:.3f}@{best_iteration}",
                    flush=True,
                )
            _write_json(config.output_dir / "run_status.json", {
                "status": "running",
                "phase": "iteration_complete",
                "iteration": iteration,
                "completed_training_instances": (
                    iteration * config.training_instances_per_iteration
                ),
                "best_validation_makespan": best_validation,
                "best_iteration": best_iteration,
                "ablation_variant": config.ablation_variant,
            })
            _maybe_cleanup_memory(iteration, config.device)
    except KeyboardInterrupt:
        current_iteration = int(history[-1]["iteration"]) if history else 0
        payload = _checkpoint_payload(
            trainer,
            schema,
            normalizer,
            config,
            ppo_config,
            iteration=current_iteration,
            best_validation=best_validation,
            best_iteration=best_iteration,
            history=history,
        )
        _atomic_torch_save(
            payload, config.output_dir / "interrupted_checkpoint.pt"
        )
        _write_json(config.output_dir / "run_status.json", {
            "status": "interrupted",
            "phase": "interrupted",
            "iteration": current_iteration,
            "best_validation_makespan": best_validation,
            "best_iteration": best_iteration,
            "ablation_variant": config.ablation_variant,
        })
        raise

    result = {
        "status": "complete",
        "method": "BSG-RL-ablation",
        "ablation_variant": config.ablation_variant,
        "profile": config.profile,
        "seed": config.seed,
        "device": config.device,
        "distribution": config.distribution,
        "training_size": config.training_size,
        "iterations": config.iterations,
        "training_instances_per_iteration": config.training_instances_per_iteration,
        "training_instances_seen": (
            config.iterations * config.training_instances_per_iteration
        ),
        "validation_dataset": str(config.validation_dataset),
        "reserved_test_dataset": str(config.reserved_test_dataset),
        "validation_instances": len(dataset),
        "spt_mean_makespan": spt_result["mean_makespan"],
        "best_validation_makespan": best_validation,
        "best_iteration": best_iteration,
        "trainable_parameters": trainable_parameters,
        "elapsed_seconds_this_run": time.perf_counter() - run_started,
        "policy": {
            "hidden_dim": config.hidden_dim,
            "category_dim": config.category_dim,
            "reference_layers": config.reference_layers,
            "competition_layers": config.competition_layers,
            "dropout": config.dropout,
        },
        "ppo": asdict(ppo_config),
        "matched_full_control": {
            "section": "5.2",
            "seed": config.seed,
            "instance_seed_scheme": config.instance_seed_scheme,
        },
    }
    _write_json(config.output_dir / "training_result.json", result)
    _write_json(config.output_dir / "run_status.json", {
        "status": "complete",
        "phase": "complete",
        "iteration": config.iterations,
        "best_validation_makespan": best_validation,
        "best_iteration": best_iteration,
        "ablation_variant": config.ablation_variant,
    })
    print(f"Training complete: {config.output_dir}")
    return result
