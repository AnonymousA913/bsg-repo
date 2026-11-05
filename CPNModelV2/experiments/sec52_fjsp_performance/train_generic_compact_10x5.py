from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean, pstdev
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from CPNModelV2.core.fjsp_cpn_v2 import run_policy, select_spt_binding
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
    BSG_RL_SMALL_CATEGORY_DIM,
    BSG_RL_SMALL_COMPETITION_LAYERS,
    BSG_RL_SMALL_HIDDEN_DIM,
    BSG_RL_SMALL_REFERENCE_LAYERS,
    BSGRLSmallPolicyNetwork,
    GenericCompactPolicyNetwork,
    GenericCompactV7PolicyNetwork,
    MaskedMLPPolicyNetwork,
    PetriNetGCNPPOPolicyNetwork,
    PNCPPOPolicyNetwork,
    POLICY_AG_PPO,
    POLICY_BSG_RL_SMALL,
    POLICY_MASKED_MLP,
    POLICY_PETRI_NET_GCN_PPO,
    POLICY_PNC_PPO,
    CPNGraphEnv,
    PPOConfig,
    REWARD_REMAINING_WORK_POTENTIAL,
    SongSD1Config,
    SongSD1V5NetFactory,
    TokenBindingPPO,
    WangSD2Config,
    WangSD2V5NetFactory,
)


POLICY_GENERIC_COMPACT = "generic_compact_reference_binding_composition"
CHECKPOINT_FORMAT = "generic_compact_cpn_checkpoint_v1"
POLICY_GENERIC_COMPACT_V7 = "generic_compact_v7_role_semantic_c3"
CHECKPOINT_FORMAT_V7 = "generic_compact_v7_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_MASKED_MLP = "sec54_masked_mlp_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_PNC_PPO = "sec54_pnc_ppo_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_PETRI_NET_GCN_PPO = "sec54_petri_net_gcn_ppo_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_AG_PPO = "sec54_assignment_graph_ppo_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_BSG_RL_SMALL = "sec54_bsg_rl_small_cpn_checkpoint_v1"


def _checkpoint_format(policy_variant: str) -> str:
    if policy_variant == POLICY_GENERIC_COMPACT:
        return CHECKPOINT_FORMAT
    if policy_variant == POLICY_GENERIC_COMPACT_V7:
        return CHECKPOINT_FORMAT_V7
    if policy_variant == POLICY_MASKED_MLP:
        return CHECKPOINT_FORMAT_MASKED_MLP
    if policy_variant == POLICY_PNC_PPO:
        return CHECKPOINT_FORMAT_PNC_PPO
    if policy_variant == POLICY_PETRI_NET_GCN_PPO:
        return CHECKPOINT_FORMAT_PETRI_NET_GCN_PPO
    if policy_variant == POLICY_AG_PPO:
        return CHECKPOINT_FORMAT_AG_PPO
    if policy_variant == POLICY_BSG_RL_SMALL:
        return CHECKPOINT_FORMAT_BSG_RL_SMALL
    raise ValueError(f"Unsupported policy variant: {policy_variant}")


def _policy_class(policy_variant: str):
    if policy_variant == POLICY_GENERIC_COMPACT:
        return GenericCompactPolicyNetwork
    if policy_variant == POLICY_GENERIC_COMPACT_V7:
        return GenericCompactV7PolicyNetwork
    if policy_variant == POLICY_MASKED_MLP:
        return MaskedMLPPolicyNetwork
    if policy_variant == POLICY_PNC_PPO:
        return PNCPPOPolicyNetwork
    if policy_variant == POLICY_PETRI_NET_GCN_PPO:
        return PetriNetGCNPPOPolicyNetwork
    if policy_variant == POLICY_AG_PPO:
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPControlledAssignmentGraphPolicy,
        )

        return FJSPControlledAssignmentGraphPolicy
    if policy_variant == POLICY_BSG_RL_SMALL:
        return BSGRLSmallPolicyNetwork
    raise ValueError(f"Unsupported policy variant: {policy_variant}")


def _graph_builder(
    policy_variant: str,
    *,
    validate: bool,
    declare_all_participation_roles: bool = False,
):
    if policy_variant == POLICY_AG_PPO:
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPAssignmentGraphBuilder,
        )

        return FJSPAssignmentGraphBuilder(validate=validate)
    return FJSPV5GenericGraphBuilder(
        validate=validate,
        declare_all_participation_roles=declare_all_participation_roles,
    )


def _fit_representation(graphs: Sequence[Any], policy_variant: str):
    if policy_variant == POLICY_AG_PPO:
        from CPNModelV2.integration.bpm2024_assignment_graph import (
            AssignmentGraphFeatureSchema,
        )
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPAssignmentGraphNoOpNormalizer,
        )

        schema = AssignmentGraphFeatureSchema.fit(graphs)
        return schema, FJSPAssignmentGraphNoOpNormalizer.fit(graphs, schema)
    schema = GenericGraphFeatureSchema.fit(graphs)
    return schema, GenericNumericFeatureNormalizer.fit(graphs, schema)


def _restore_representation(checkpoint: Dict[str, Any], policy_variant: str):
    if policy_variant == POLICY_AG_PPO:
        from CPNModelV2.integration.bpm2024_assignment_graph import (
            AssignmentGraphFeatureSchema,
        )
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPAssignmentGraphNoOpNormalizer,
        )

        schema = AssignmentGraphFeatureSchema.from_dict(checkpoint["schema"])
        normalizer = FJSPAssignmentGraphNoOpNormalizer.from_dict(
            checkpoint["normalizer"], schema
        )
        return schema, normalizer
    schema = GenericGraphFeatureSchema.from_dict(checkpoint["schema"])
    normalizer = GenericNumericFeatureNormalizer.from_dict(
        checkpoint["normalizer"], schema
    )
    return schema, normalizer


def _tensorizer(schema: Any, normalizer: Any, policy_variant: str):
    if policy_variant == POLICY_AG_PPO:
        from CPNModelV2.integration.bpm2024_assignment_graph import (
            AssignmentGraphTensorizer,
        )

        return AssignmentGraphTensorizer(
            schema,
            normalizer=normalizer,
            validate_graphs=VALIDATE_INTERNAL_GRAPHS,
        )
    return GenericGraphTensorizer(
        schema,
        normalizer=normalizer,
        validate_graphs=VALIDATE_INTERNAL_GRAPHS,
    )


# =============================================================================
# USER CONFIGURATION: edit this section, then run this file directly.
# =============================================================================

RUN_PROFILE = "full"  # "pilot" for a short health check; "full" for training
POLICY_VARIANT = POLICY_GENERIC_COMPACT
SEED = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTO_RESUME = True

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
VALIDATION_DATASET = PROJECT_ROOT / "data" / "data_train_vali" / "SD1" / "10x5"
OUTPUT_ROOT = PACKAGE_ROOT / "results" / "ppo"
RUN_NAME = (
    f"generic_compact_sd1_10x5_h64_{RUN_PROFILE}_seed{SEED}"
)

# Formal Sections 5.2--5.3 training budget.
FULL_ITERATIONS = 1000
FULL_TRAINING_INSTANCES_PER_ITERATION = 20
FULL_VALIDATE_EVERY = 10
FULL_VALIDATION_INSTANCES = 20

# Quick integration run. It uses the same method with a smaller budget.
PILOT_ITERATIONS = 2
PILOT_TRAINING_INSTANCES_PER_ITERATION = 2
PILOT_VALIDATE_EVERY = 1
PILOT_VALIDATION_INSTANCES = 2

# Device batching controls peak memory without changing the comparison budget.
ROLLOUT_DEVICE_BATCH_SIZE = 4
PPO_DEVICE_MICROBATCH_SIZE = 32

# Feature calibration and validation batching.
CALIBRATION_INSTANCES = 8
EVALUATION_BATCH_SIZE = 5

# Binding-semantic encoder.
HIDDEN_DIM = 64
CATEGORY_DIM = 16
# These are architecture-specific counterparts, not identical implementations,
# of V1.2 TEMPLATE_LAYERS and FACTOR_ROUNDS.
REFERENCE_LAYERS = 2
# Optional conflict-aware refinement. Keep 0 for the lean core method; use 1
# only in its dedicated ablation experiment.
COMPETITION_LAYERS = 0
DROPOUT = 0.0

# PPO and reward.
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 1024
GAMMA = 1.0
GAE_LAMBDA = 0.98
CLIP_RATIO = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5
REWARD_MODE = REWARD_REMAINING_WORK_POTENTIAL
REWARD_SCALE = 20.0
POTENTIAL_COEFFICIENT = 0.1

# Safety and output cadence.
FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT = True
SAVE_EVERY_VALIDATION = True
PRINT_EVERY = 1
VALIDATE_INTERNAL_GRAPHS = False

# Candidate checkpoints for validation-only model selection. Periodic snapshots
# begin after the warm-up threshold; validation improvements use the same
# archive and never duplicate a periodic snapshot from the same iteration.
CHECKPOINT_ARCHIVE_START = 50
CHECKPOINT_ARCHIVE_INTERVAL = 20
ARCHIVE_VALIDATION_IMPROVEMENTS = True
CHECKPOINT_ARCHIVE_DIRECTORY = "checkpoint_archive"

# =============================================================================


@dataclass(frozen=True)
class TrainingConfig:
    profile: str
    policy_variant: str
    seed: int
    device: str
    output_dir: Path
    validation_dataset: Path
    iterations: int
    training_instances_per_iteration: int
    rollout_device_batch_size: int
    ppo_device_microbatch_size: int
    validate_every: int
    validation_instances: int
    evaluation_batch_size: int
    calibration_instances: int
    hidden_dim: int
    category_dim: int
    reference_layers: int
    competition_layers: int
    dropout: float
    reward_mode: str
    reward_scale: float
    potential_coefficient: float
    learning_rate: float
    update_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    auto_resume: bool
    save_every_validation: bool
    print_every: int
    checkpoint_archive_start: int
    checkpoint_archive_interval: int
    archive_validation_improvements: bool
    checkpoint_archive_directory: str
    distribution: str = "SD1"
    protocol_version: str = "development"
    instance_seed_scheme: str = "legacy_linear_v1"
    training_size: str = "10x5"
    num_jobs: int = 10
    num_machines: int = 5
    min_operations: int = 4
    max_operations: int = 6
    min_eligible_machines: int = 1
    max_eligible_machines: int = 5

    def validate(self) -> None:
        if self.profile not in {"pilot", "screen", "full"}:
            raise ValueError("RUN_PROFILE must be 'pilot', 'screen' or 'full'")
        if self.distribution not in {"SD1", "SD2"}:
            raise ValueError("distribution must be SD1 or SD2")
        if self.instance_seed_scheme not in {
            "legacy_linear_v1",
            "sha256_v1",
        }:
            raise ValueError("Unsupported instance_seed_scheme")
        expected_size = f"{self.num_jobs}x{self.num_machines}"
        if self.num_jobs <= 0 or self.num_machines <= 0:
            raise ValueError("Job and machine counts must be positive")
        if self.training_size != expected_size:
            raise ValueError(
                "training_size must match num_jobs and num_machines: "
                f"{self.training_size} != {expected_size}"
            )
        if not 1 <= self.min_operations <= self.max_operations:
            raise ValueError("Invalid operation-count range")
        if self.distribution == "SD2" and (
            self.min_operations != self.max_operations
            or self.min_operations != self.num_machines
        ):
            raise ValueError(
                "SD2 +mix uses exactly num_machines operations per job"
            )
        if not (
            1
            <= self.min_eligible_machines
            <= self.max_eligible_machines
            <= self.num_machines
        ):
            raise ValueError("Invalid eligible-machine range")
        if self.policy_variant not in {
            POLICY_GENERIC_COMPACT,
            POLICY_GENERIC_COMPACT_V7,
            POLICY_MASKED_MLP,
            POLICY_PNC_PPO,
            POLICY_PETRI_NET_GCN_PPO,
            POLICY_AG_PPO,
            POLICY_BSG_RL_SMALL,
        }:
            raise ValueError(f"Unsupported policy variant: {self.policy_variant}")
        if self.policy_variant == POLICY_BSG_RL_SMALL:
            observed = (
                self.hidden_dim,
                self.category_dim,
                self.reference_layers,
                self.competition_layers,
            )
            expected = (
                BSG_RL_SMALL_HIDDEN_DIM,
                BSG_RL_SMALL_CATEGORY_DIM,
                BSG_RL_SMALL_REFERENCE_LAYERS,
                BSG_RL_SMALL_COMPETITION_LAYERS,
            )
            if observed != expected:
                raise ValueError(
                    "BSG-RL-S requires the fixed capacity-control "
                    f"configuration {expected}, got {observed}"
                )
        if self.competition_layers != 0:
            raise ValueError("Generic Compact requires COMPETITION_LAYERS=0")
        positive = {
            "iterations": self.iterations,
            "training_instances_per_iteration": self.training_instances_per_iteration,
            "rollout_device_batch_size": self.rollout_device_batch_size,
            "ppo_device_microbatch_size": self.ppo_device_microbatch_size,
            "validate_every": self.validate_every,
            "validation_instances": self.validation_instances,
            "evaluation_batch_size": self.evaluation_batch_size,
            "calibration_instances": self.calibration_instances,
            "hidden_dim": self.hidden_dim,
            "category_dim": self.category_dim,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "print_every": self.print_every,
            "checkpoint_archive_interval": self.checkpoint_archive_interval,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Positive configuration required: {invalid}")
        if self.reference_layers < 0 or self.competition_layers < 0:
            raise ValueError("Encoder layer counts must be non-negative")
        if self.checkpoint_archive_start < 0:
            raise ValueError("checkpoint_archive_start must be non-negative")
        archive_dir = Path(self.checkpoint_archive_directory)
        if (
            not self.checkpoint_archive_directory
            or archive_dir.is_absolute()
            or archive_dir.name != self.checkpoint_archive_directory
        ):
            raise ValueError(
                "checkpoint_archive_directory must be one directory name"
            )
        if self.reward_scale <= 0 or self.potential_coefficient < 0:
            raise ValueError("Invalid reward configuration")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid PPO optimization configuration")
        if not 0.0 <= self.clip_ratio < 1.0:
            raise ValueError("CLIP_RATIO must be in [0, 1)")
        if not self.validation_dataset.is_dir():
            raise FileNotFoundError(self.validation_dataset)


def configured_run() -> TrainingConfig:
    if RUN_PROFILE == "full":
        iterations = FULL_ITERATIONS
        training_instances = FULL_TRAINING_INSTANCES_PER_ITERATION
        validate_every = FULL_VALIDATE_EVERY
        validation_instances = FULL_VALIDATION_INSTANCES
    elif RUN_PROFILE == "pilot":
        iterations = PILOT_ITERATIONS
        training_instances = PILOT_TRAINING_INSTANCES_PER_ITERATION
        validate_every = PILOT_VALIDATE_EVERY
        validation_instances = PILOT_VALIDATION_INSTANCES
    else:
        raise ValueError("RUN_PROFILE must be 'pilot' or 'full'")
    requested_device = DEVICE
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    return TrainingConfig(
        profile=RUN_PROFILE,
        policy_variant=POLICY_VARIANT,
        seed=SEED,
        device=requested_device,
        output_dir=(OUTPUT_ROOT / RUN_NAME).resolve(),
        validation_dataset=VALIDATION_DATASET.resolve(),
        iterations=iterations,
        training_instances_per_iteration=training_instances,
        rollout_device_batch_size=min(
            ROLLOUT_DEVICE_BATCH_SIZE, training_instances
        ),
        ppo_device_microbatch_size=PPO_DEVICE_MICROBATCH_SIZE,
        validate_every=validate_every,
        validation_instances=validation_instances,
        evaluation_batch_size=EVALUATION_BATCH_SIZE,
        calibration_instances=CALIBRATION_INSTANCES,
        hidden_dim=HIDDEN_DIM,
        category_dim=CATEGORY_DIM,
        reference_layers=REFERENCE_LAYERS,
        competition_layers=COMPETITION_LAYERS,
        dropout=DROPOUT,
        reward_mode=REWARD_MODE,
        reward_scale=REWARD_SCALE,
        potential_coefficient=POTENTIAL_COEFFICIENT,
        learning_rate=LEARNING_RATE,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_ratio=CLIP_RATIO,
        value_coefficient=VALUE_COEFFICIENT,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        max_grad_norm=MAX_GRAD_NORM,
        auto_resume=AUTO_RESUME,
        save_every_validation=SAVE_EVERY_VALIDATION,
        print_every=PRINT_EVERY,
        checkpoint_archive_start=CHECKPOINT_ARCHIVE_START,
        checkpoint_archive_interval=CHECKPOINT_ARCHIVE_INTERVAL,
        archive_validation_improvements=ARCHIVE_VALIDATION_IMPROVEMENTS,
        checkpoint_archive_directory=CHECKPOINT_ARCHIVE_DIRECTORY,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def _collect_calibration_graphs(
    factory: Any,
    seeds: Iterable[int],
    *,
    policy_variant: str = POLICY_GENERIC_COMPACT,
    declare_all_participation_roles: bool = False,
) -> list:
    builder = _graph_builder(
        policy_variant,
        validate=VALIDATE_INTERNAL_GRAPHS,
        declare_all_participation_roles=declare_all_participation_roles,
    )
    graphs = []
    for seed in seeds:
        rng = random.Random(seed + 1_000_003)
        net = factory(seed)
        net.advance_until_decision_or_terminal()
        while not net.is_terminal():
            graphs.append(builder.build(net))
            bindings = net.control_bindings()
            if not bindings:
                raise RuntimeError("Calibration V5 CPN has no enabled Binding")
            net.step(rng.choice(bindings).key)
        graphs.append(builder.build(net))
    if not graphs:
        raise RuntimeError("Calibration produced no V5 graphs")
    return graphs


def _evaluate_spt(paths: Sequence[Path]) -> Dict[str, Any]:
    rows = []
    for path in paths:
        net = FJSFileV5NetFactory(path)()
        run_policy(net, select_spt_binding)
        rows.append({"instance": path.name, "makespan": float(net.makespan())})
    values = [row["makespan"] for row in rows]
    return {
        "mean_makespan": mean(values),
        "min_makespan": min(values),
        "max_makespan": max(values),
        "per_instance": rows,
    }


def _evaluate_policy(
    trainer: TokenBindingPPO,
    paths: Sequence[Path],
    config: TrainingConfig,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(paths), config.evaluation_batch_size):
        chunk = paths[start : start + config.evaluation_batch_size]
        environments = [
            CPNGraphEnv(
                FJSFileV5NetFactory(path),
                graph_builder=_graph_builder(
                    config.policy_variant,
                    validate=VALIDATE_INTERNAL_GRAPHS,
                ),
                reward_scale=config.reward_scale,
                reward_mode=config.reward_mode,
                potential_coefficient=config.potential_coefficient,
            )
            for path in chunk
        ]
        episodes = trainer.collect_episodes_batched(
            environments, deterministic=True
        )
        rows.extend({
            "instance": path.name,
            "makespan": float(episode.makespan),
            "decisions": int(episode.decision_count),
        } for path, episode in zip(chunk, episodes))
    values = [row["makespan"] for row in rows]
    return {
        "mean_makespan": mean(values),
        "min_makespan": min(values),
        "max_makespan": max(values),
        "per_instance": rows,
    }


def _write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _periodic_checkpoint_due(
    config: TrainingConfig,
    iteration: int,
) -> bool:
    return (
        iteration > config.checkpoint_archive_start
        and iteration % config.checkpoint_archive_interval == 0
    )


def _archive_checkpoint(
    payload: Dict[str, Any],
    config: TrainingConfig,
    *,
    iteration: int,
    reasons: Sequence[str],
    validation_makespan: Optional[float],
) -> Optional[Path]:
    if not reasons:
        return None
    archive_dir = (
        config.output_dir / config.checkpoint_archive_directory
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    if "validation_best" in reasons:
        labels.append("best")
    if "periodic" in reasons:
        labels.append("periodic")
    labels.extend(
        reason for reason in reasons
        if reason not in {"validation_best", "periodic"}
    )
    checkpoint_path = archive_dir / (
        f"iteration_{iteration:04d}_{'_'.join(labels)}.pt"
    )
    _atomic_torch_save(payload, checkpoint_path)

    manifest_path = archive_dir / "manifest.json"
    manifest: Dict[str, Any] = {
        "selection_policy": (
            "Candidates only. Select using validation data before any reserved "
            "test results are inspected."
        ),
        "checkpoints": [],
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [
        entry for entry in manifest.get("checkpoints", [])
        if int(entry["iteration"]) != iteration
    ]
    entries.append({
        "iteration": iteration,
        "file": checkpoint_path.name,
        "reasons": sorted(set(reasons)),
        "validation_mean_makespan": validation_makespan,
        "best_validation_makespan": float(
            payload["best_validation_makespan"]
        ),
        "best_iteration": int(payload["best_iteration"]),
    })
    entries.sort(key=lambda entry: int(entry["iteration"]))
    manifest["checkpoints"] = entries
    _write_json(manifest_path, manifest)
    return checkpoint_path


TRAINING_INSTANCE_FIELDS = (
    "protocol_version",
    "distribution",
    "training_size",
    "training_seed",
    "iteration",
    "batch_index",
    "generation_seed",
    "instance_seed_scheme",
    "generator_version",
    "generator_config_sha256",
)


def _generator_config(config: TrainingConfig) -> Dict[str, Any]:
    values: Dict[str, Any] = {
        "distribution": config.distribution,
        "num_jobs": config.num_jobs,
        "num_machines": config.num_machines,
        "min_operations": config.min_operations,
        "max_operations": config.max_operations,
        "min_eligible_machines": config.min_eligible_machines,
        "max_eligible_machines": config.max_eligible_machines,
    }
    if config.distribution == "SD1":
        values.update({
            "min_processing_time": 1,
            "max_processing_time": 20,
            "processing_time_deviation": 0.2,
            "generator_version": "song_sd1_python_random_v1",
        })
    else:
        values.update({
            "operations_per_job": config.num_machines,
            "min_processing_time": 1,
            "max_processing_time": 99,
            "generator_version": WangSD2V5NetFactory.GENERATOR_VERSION,
        })
    return values


def _training_instance_seed(
    config: TrainingConfig,
    iteration: int,
    batch_index: int,
) -> int:
    if config.instance_seed_scheme == "legacy_linear_v1":
        return (
            config.seed * 1_000_000
            + (iteration - 1) * config.training_instances_per_iteration
            + batch_index
        )
    generator_version = _generator_config(config)["generator_version"]
    key = "|".join((
        "sec52",
        str(generator_version),
        config.distribution,
        config.training_size,
        str(config.seed),
        str(iteration),
        str(batch_index),
    ))
    return int.from_bytes(
        hashlib.sha256(key.encode("ascii")).digest()[:4],
        byteorder="big",
        signed=False,
    )


def _write_training_instance_manifest(config: TrainingConfig) -> None:
    generator_config = _generator_config(config)
    serialized = json.dumps(
        generator_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_sha256 = hashlib.sha256(serialized).hexdigest()
    rows = [
        {
            "protocol_version": config.protocol_version,
            "distribution": config.distribution,
            "training_size": config.training_size,
            "training_seed": config.seed,
            "iteration": iteration,
            "batch_index": batch_index,
            "generation_seed": _training_instance_seed(
                config, iteration, batch_index
            ),
            "instance_seed_scheme": config.instance_seed_scheme,
            "generator_version": generator_config["generator_version"],
            "generator_config_sha256": config_sha256,
        }
        for iteration in range(1, config.iterations + 1)
        for batch_index in range(config.training_instances_per_iteration)
    ]
    _write_csv(
        config.output_dir / "training_instance_manifest.csv",
        rows,
        TRAINING_INSTANCE_FIELDS,
    )


def _checkpoint_payload(
    trainer: TokenBindingPPO,
    schema: Any,
    normalizer: Any,
    config: TrainingConfig,
    ppo_config: PPOConfig,
    *,
    iteration: int,
    best_validation: float,
    best_iteration: int,
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "format": _checkpoint_format(config.policy_variant),
        "policy_variant": config.policy_variant,
        "model_state_dict": trainer.policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "schema": schema.to_dict(),
        "normalizer": normalizer.to_dict(),
        "training_config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "validation_dataset": str(config.validation_dataset),
        },
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


def _build_trainer(
    config: TrainingConfig,
    factory: Any,
) -> tuple[
    TokenBindingPPO,
    Any,
    Any,
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
            checkpoint_path, map_location=config.device, weights_only=False
        )
        if checkpoint.get("format") != _checkpoint_format(config.policy_variant):
            raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
        if checkpoint.get("policy_variant", POLICY_GENERIC_COMPACT) != config.policy_variant:
            raise ValueError("Checkpoint policy variant differs from configuration")

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
        calibration_seeds = range(
            config.seed * 10_000,
            config.seed * 10_000 + config.calibration_instances,
        )
        calibration_graphs = _collect_calibration_graphs(
            factory,
            calibration_seeds,
            policy_variant=config.policy_variant,
            declare_all_participation_roles=(
                config.policy_variant in {
                    POLICY_GENERIC_COMPACT_V7,
                    POLICY_PETRI_NET_GCN_PPO,
                }
            ),
        )
        schema, normalizer = _fit_representation(
            calibration_graphs, config.policy_variant
        )
    else:
        saved_config = checkpoint.get("training_config", {})
        resume_contract = {
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
            "training_instances_per_iteration": (
                config.training_instances_per_iteration
            ),
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
        legacy_defaults = {
            "distribution": "SD1",
            "protocol_version": "development",
            "instance_seed_scheme": "legacy_linear_v1",
            "training_size": "10x5",
            "num_jobs": 10,
            "num_machines": 5,
            "min_operations": 4,
            "max_operations": 6,
            "min_eligible_machines": 1,
            "max_eligible_machines": 5,
        }
        changed = {
            name: (saved_config.get(name, legacy_defaults.get(name)), current)
            for name, current in resume_contract.items()
            if saved_config.get(name, legacy_defaults.get(name)) != current
        }
        if changed:
            raise ValueError(
                "The model, PPO, or reward configuration differs from the checkpoint. "
                f"Use a new RUN_NAME or restore these values: {changed}"
            )
        schema, normalizer = _restore_representation(
            checkpoint, config.policy_variant
        )
        policy_config = dict(checkpoint["policy_config"])
        ppo_config = PPOConfig(**checkpoint["ppo_config"])
        start_iteration = int(checkpoint["iteration"])
        best_validation = float(checkpoint["best_validation_makespan"])
        best_iteration = int(checkpoint["best_iteration"])
        history = list(checkpoint.get("history", []))

    policy_class = _policy_class(config.policy_variant)
    policy = policy_class(schema, **policy_config)
    tensorizer = _tensorizer(schema, normalizer, config.policy_variant)
    trainer = TokenBindingPPO(
        policy, tensorizer, config=ppo_config, device=config.device
    )
    if checkpoint is not None:
        trainer.policy.load_state_dict(checkpoint["model_state_dict"])
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


TRAINING_FIELDS = (
    "protocol_version",
    "distribution",
    "training_size",
    "training_seed",
    "iteration",
    "completed_training_episodes",
    "train_mean_makespan",
    "train_mean_return",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "learning_rate",
    "invalid_action_count",
    "infeasible_schedule_count",
    "iteration_seconds",
    "validation_mean_makespan",
    "validation_std_makespan",
    "validation_min_makespan",
    "validation_max_makespan",
    "validation_seconds",
    "best_validation_makespan",
    "best_iteration",
    "is_best",
)

VALIDATION_FIELDS = (
    "iteration",
    "instance",
    "policy_makespan",
    "spt_makespan",
    "difference",
    "relative_gap_percent",
    "beats_spt",
    "decisions",
)


def _validation_rows(
    iteration: int,
    policy_result: Dict[str, Any],
    spt_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    spt = {
        row["instance"]: float(row["makespan"])
        for row in spt_result["per_instance"]
    }
    rows = []
    for row in policy_result["per_instance"]:
        policy_makespan = float(row["makespan"])
        spt_makespan = spt[row["instance"]]
        rows.append({
            "iteration": iteration,
            "instance": row["instance"],
            "policy_makespan": policy_makespan,
            "spt_makespan": spt_makespan,
            "difference": policy_makespan - spt_makespan,
            "relative_gap_percent": (
                100.0 * (policy_makespan - spt_makespan) / spt_makespan
            ),
            "beats_spt": policy_makespan < spt_makespan,
            "decisions": row["decisions"],
        })
    return rows


def run_training(config: TrainingConfig) -> Dict[str, Any]:
    config.validate()
    output_has_files = (
        config.output_dir.is_dir() and any(config.output_dir.iterdir())
    )
    if output_has_files and not config.auto_resume:
        raise FileExistsError(
            f"Output already contains a previous run: {config.output_dir}. "
            "Enable AUTO_RESUME or change RUN_NAME."
        )
    if (
        FAIL_IF_OUTPUT_EXISTS_WITHOUT_CHECKPOINT
        and output_has_files
        and not (config.output_dir / "last_checkpoint.pt").is_file()
    ):
        raise FileExistsError(
            f"Output is non-empty without a resumable checkpoint: "
            f"{config.output_dir}. Change RUN_NAME or clear it manually."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    _write_json(config.output_dir / "run_status.json", {
        "status": "running",
        "phase": "schema_or_checkpoint_initialization",
        "iteration": 0,
        "completed_training_instances": 0,
    })

    dataset = PublicFJSPDataset(
        config.validation_dataset,
        limit=config.validation_instances,
        expected_jobs=config.num_jobs,
        expected_machines=config.num_machines,
    )
    if config.distribution == "SD1":
        instance_config = SongSD1Config(
            num_jobs=config.num_jobs,
            num_machines=config.num_machines,
            min_operations=config.min_operations,
            max_operations=config.max_operations,
            min_eligible_machines=config.min_eligible_machines,
            max_eligible_machines=config.max_eligible_machines,
        )
        factory = SongSD1V5NetFactory(instance_config)
    else:
        instance_config = WangSD2Config(
            num_jobs=config.num_jobs,
            num_machines=config.num_machines,
            operations_per_job=config.num_machines,
            min_eligible_machines=config.min_eligible_machines,
            max_eligible_machines=config.max_eligible_machines,
        )
        factory = WangSD2V5NetFactory(instance_config)
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
    _write_json(
        config.output_dir / "run_config.json",
        {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "validation_dataset": str(config.validation_dataset),
            "ppo": asdict(ppo_config),
        },
    )

    spt_result = _evaluate_spt(dataset.paths)
    if start_iteration == 0 and not history:
        _write_json(config.output_dir / "run_status.json", {
            "status": "running",
            "phase": "initial_validation",
            "iteration": 0,
            "completed_training_instances": 0,
        })
        validation = _evaluate_policy(trainer, dataset.paths, config)
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
        _atomic_torch_save(payload, config.output_dir / "last_checkpoint.pt")

    # Create the record files before the first PPO iteration begins. This also
    # makes an iteration-0 checkpoint resume observable without duplicating its
    # initial validation row.
    _write_csv(
        config.output_dir / "training_history.csv",
        history,
        TRAINING_FIELDS,
    )
    _write_csv(
        config.output_dir / "validation_history.csv",
        [row for row in history if "validation_mean_makespan" in row],
        TRAINING_FIELDS,
    )

    _write_csv(
        config.output_dir / "spt_validation_per_instance.csv",
        [{
            "instance": row["instance"],
            "spt_makespan": row["makespan"],
        } for row in spt_result["per_instance"]],
        ("instance", "spt_makespan"),
    )
    environments = [
        CPNGraphEnv(
            factory,
            graph_builder=_graph_builder(
                config.policy_variant,
                validate=VALIDATE_INTERNAL_GRAPHS,
            ),
            reward_scale=config.reward_scale,
            reward_mode=config.reward_mode,
            potential_coefficient=config.potential_coefficient,
        )
        for _ in range(config.rollout_device_batch_size)
    ]

    print(
        f"Generic Compact CPN PPO on "
        f"{config.distribution} {config.training_size}"
    )
    print(f"  profile       : {config.profile}")
    print(f"  policy        : {config.policy_variant}")
    print(f"  device        : {config.device}")
    print(f"  iterations    : {start_iteration} -> {config.iterations}")
    print(
        "  train budget  : "
        f"{config.training_instances_per_iteration} instances/iteration"
    )
    print(f"  rollout batch : {config.rollout_device_batch_size}")
    print(f"  PPO microbatch: {config.ppo_device_microbatch_size}")
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.policy.parameters()
        if parameter.requires_grad
    )
    print(f"  parameters    : {trainable_parameters:,}")
    print(f"  validation    : {len(dataset)} public instances")
    print(f"  SPT mean      : {spt_result['mean_makespan']:.3f}")
    print(f"  output        : {config.output_dir}")

    run_started = time.perf_counter()
    try:
        for iteration in range(start_iteration + 1, config.iterations + 1):
            iteration_started = time.perf_counter()
            print(
                f"  iter={iteration:04d} rollout started "
                f"({config.training_instances_per_iteration} instances)",
                flush=True,
            )
            seeds = [
                _training_instance_seed(config, iteration, index)
                for index in range(config.training_instances_per_iteration)
            ]
            episodes = []
            for start in range(
                0, len(seeds), config.rollout_device_batch_size
            ):
                seed_chunk = seeds[
                    start : start + config.rollout_device_batch_size
                ]
                episodes.extend(trainer.collect_episodes_batched(
                    environments[:len(seed_chunk)], seeds=seed_chunk
                ))
                _write_json(config.output_dir / "run_status.json", {
                    "status": "running",
                    "phase": "rollout",
                    "iteration": iteration,
                    "completed_training_instances": len(episodes),
                    "training_instances_this_iteration": len(seeds),
                })
                print(
                    f"    rollout {len(episodes):02d}/{len(seeds):02d}",
                    flush=True,
                )
            _write_json(config.output_dir / "run_status.json", {
                "status": "running",
                "phase": "ppo_update",
                "iteration": iteration,
                "completed_training_instances": len(episodes),
                "training_instances_this_iteration": len(seeds),
            })
            print(f"    PPO update ({len(episodes)} episodes)", flush=True)
            metrics = trainer.update(episodes, seed=config.seed + iteration)
            record: Dict[str, Any] = {
                "protocol_version": config.protocol_version,
                "distribution": config.distribution,
                "training_size": config.training_size,
                "training_seed": config.seed,
                "iteration": iteration,
                "completed_training_episodes": (
                    iteration * config.training_instances_per_iteration
                ),
                "train_mean_makespan": mean(
                    episode.makespan for episode in episodes
                ),
                "train_mean_return": mean(
                    episode.episode_return for episode in episodes
                ),
                **metrics,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "invalid_action_count": 0,
                "infeasible_schedule_count": 0,
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
                validation = _evaluate_policy(trainer, dataset.paths, config)
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

            # Keep recovery current even when validation is infrequent.
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
            _atomic_torch_save(payload, config.output_dir / "last_checkpoint.pt")
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
                config.output_dir / "training_history.csv",
                history,
                TRAINING_FIELDS,
            )
            _write_csv(
                config.output_dir / "validation_history.csv",
                [row for row in history if "validation_mean_makespan" in row],
                TRAINING_FIELDS,
            )
            if iteration % config.print_every == 0:
                validation_text = (
                    f" val={record['validation_mean_makespan']:.3f}"
                    if "validation_mean_makespan" in record else ""
                )
                print(
                    f"  iter={iteration:04d} "
                    f"train={record['train_mean_makespan']:.3f} "
                    f"loss={record['loss']:.4f} "
                    f"entropy={record['entropy']:.4f}"
                    f"{validation_text} best={best_validation:.3f}@{best_iteration}",
                    flush=True,
                )
            _write_json(config.output_dir / "run_status.json", {
                "status": "running",
                "phase": "iteration_complete",
                "iteration": iteration,
                "completed_training_instances": (
                    iteration * config.training_instances_per_iteration
                ),
                "iteration_seconds": record["iteration_seconds"],
                "best_validation_makespan": best_validation,
                "best_iteration": best_iteration,
            })
    except KeyboardInterrupt:
        print(
            "Training interrupted; saving the current state before exit.",
            flush=True,
        )
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
        _atomic_torch_save(payload, config.output_dir / "interrupted_checkpoint.pt")
        _write_json(config.output_dir / "run_status.json", {
            "status": "interrupted",
            "phase": "interrupted",
            "iteration": current_iteration,
            "completed_training_instances": (
                current_iteration * config.training_instances_per_iteration
            ),
            "best_validation_makespan": best_validation,
            "best_iteration": best_iteration,
        })
        raise

    result = {
        "status": "complete",
        "method": config.policy_variant,
        "profile": config.profile,
        "seed": config.seed,
        "device": config.device,
        "iterations": config.iterations,
        "training_instances_per_iteration": (
            config.training_instances_per_iteration
        ),
        "rollout_device_batch_size": config.rollout_device_batch_size,
        "ppo_device_microbatch_size": config.ppo_device_microbatch_size,
        "training_instances_seen": (
            config.iterations * config.training_instances_per_iteration
        ),
        "validation_dataset": str(config.validation_dataset),
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
    }
    _write_json(config.output_dir / "training_result.json", result)
    _write_json(config.output_dir / "run_status.json", {
        "status": "complete",
        "phase": "complete",
        "iteration": config.iterations,
        "completed_training_instances": (
            config.iterations * config.training_instances_per_iteration
        ),
        "best_validation_makespan": best_validation,
        "best_iteration": best_iteration,
    })
    print(f"Training complete: {config.output_dir}")
    return result


def main() -> None:
    run_training(configured_run())


if __name__ == "__main__":
    main()
