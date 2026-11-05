from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
from statistics import mean
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from CPNModelV2.core.fjsp_cpn_v2 import OperationToken
from CPNModelV2.graph import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.integration.equivalence_core import CanonicalInstance
from CPNModelV2.integration.fjsp_v5_generic_adapter import (
    FJSPV5GenericGraphBuilder,
)
from CPNModelV2.integration.public_fjsp_dataset import (
    FJSFileV5NetFactory,
    PublicFJSPDataset,
)
from CPNModelV2.learning import (
    BSGRLSmallPolicyNetwork,
    CPNGraphEnv,
    GenericCompactPolicyNetwork,
    GenericCompactV7PolicyNetwork,
    MaskedMLPPolicyNetwork,
    PetriNetGCNPPOPolicyNetwork,
    PNCPPOPolicyNetwork,
    PPOConfig,
    POLICY_AG_PPO,
    POLICY_BSG_RL_SMALL,
    TokenBindingPPO,
)
from CPNModelV2.learning.generic_compact_ablation_policy import (
    GenericCompactAblationPolicyNetwork,
)


CHECKPOINT_FORMAT = "generic_compact_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_V7 = "generic_compact_v7_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_MASKED_MLP = "sec54_masked_mlp_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_PNC_PPO = "sec54_pnc_ppo_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_PETRI_NET_GCN_PPO = "sec54_petri_net_gcn_ppo_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_AG_PPO = "sec54_assignment_graph_ppo_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_BSG_RL_SMALL = "sec54_bsg_rl_small_cpn_checkpoint_v1"
CHECKPOINT_FORMAT_SEC55_ABLATION = (
    "generic_compact_sec55_ablation_checkpoint_v1"
)
SUPPORTED_CHECKPOINT_FORMATS = (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_V7,
    CHECKPOINT_FORMAT_MASKED_MLP,
    CHECKPOINT_FORMAT_PNC_PPO,
    CHECKPOINT_FORMAT_PETRI_NET_GCN_PPO,
    CHECKPOINT_FORMAT_AG_PPO,
    CHECKPOINT_FORMAT_BSG_RL_SMALL,
    CHECKPOINT_FORMAT_SEC55_ABLATION,
)
EVALUATION_PROTOCOL_VERSION = "sec52_evaluation_v1"


@dataclass(frozen=True)
class EvaluationTask:
    mode: str
    distribution: str
    size: str
    training_seed: int
    checkpoint: Path
    test_dataset: Path
    output_dir: Path
    checkpoint_label: str = ""
    test_limit: int = 100
    greedy_batch_size: int = 5
    sampling_rollouts: int = 100
    sampling_batch_size: int = 5
    device: str = "cpu"

    def validate(self) -> None:
        if self.mode not in {"pilot", "formal"}:
            raise ValueError(f"Unsupported training mode: {self.mode}")
        if self.distribution not in {"SD1", "SD2"}:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        if not re.fullmatch(r"\d+x\d+", self.size):
            raise ValueError(f"Invalid FJSP size: {self.size}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.test_dataset.is_dir():
            raise FileNotFoundError(self.test_dataset)
        for name, value in {
            "test_limit": self.test_limit,
            "greedy_batch_size": self.greedy_batch_size,
            "sampling_rollouts": self.sampling_rollouts,
            "sampling_batch_size": self.sampling_batch_size,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def task_id(self) -> str:
        if self.checkpoint_label:
            return (
                f"{self.distribution.lower()}_{self.size}_"
                f"seed{self.training_seed}_{self.checkpoint_label}"
            )
        base = (
            f"{self.mode}_{self.distribution.lower()}_"
            f"{self.size}_seed{self.training_seed}"
        )
        return base

    @property
    def dimensions(self) -> Tuple[int, int]:
        jobs, machines = self.size.split("x")
        return int(jobs), int(machines)


@dataclass(frozen=True)
class ScheduleAssignment:
    job_id: int
    operation_index: int
    machine_id: int
    start_time: float
    finish_time: float


@dataclass(frozen=True)
class RolloutOutcome:
    makespan: float
    decision_count: int
    assignments: Tuple[ScheduleAssignment, ...]


@dataclass
class TimingTotals:
    reset_and_initial_graph_s: float = 0.0
    tensorization_s: float = 0.0
    network_inference_s: float = 0.0
    action_selection_s: float = 0.0
    environment_step_and_graph_s: float = 0.0
    end_to_end_s: float = 0.0
    schedules: int = 0

    def add(self, other: "TimingTotals") -> None:
        for field in (
            "reset_and_initial_graph_s",
            "tensorization_s",
            "network_inference_s",
            "action_selection_s",
            "environment_step_and_graph_s",
            "end_to_end_s",
            "schedules",
        ):
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "reset_and_initial_graph_s": self.reset_and_initial_graph_s,
            "tensorization_s": self.tensorization_s,
            "network_inference_s": self.network_inference_s,
            "action_selection_s": self.action_selection_s,
            "environment_step_and_graph_s": self.environment_step_and_graph_s,
            "end_to_end_s": self.end_to_end_s,
            "schedules": self.schedules,
        }


def deterministic_evaluation_seed(
    task: EvaluationTask,
    instance_id: str,
    decoder: str,
    rollout_id: int,
) -> int:
    canonical = "|".join((
        EVALUATION_PROTOCOL_VERSION,
        task.distribution,
        task.size,
        str(task.training_seed),
        instance_id,
        decoder,
        str(rollout_id),
    ))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_evaluator(
    checkpoint_path: Path,
    *,
    device: str,
) -> Tuple[TokenBindingPPO, Dict[str, Any]]:
    checkpoint = _torch_load(checkpoint_path)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_path}: "
            f"{checkpoint.get('format')!r}"
        )
    if checkpoint_format == CHECKPOINT_FORMAT_AG_PPO:
        from CPNModelV2.integration.bpm2024_assignment_graph import (
            AssignmentGraphFeatureSchema,
            AssignmentGraphTensorizer,
        )
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPAssignmentGraphNoOpNormalizer,
            FJSPControlledAssignmentGraphPolicy,
        )

        schema = AssignmentGraphFeatureSchema.from_dict(checkpoint["schema"])
        normalizer = FJSPAssignmentGraphNoOpNormalizer.from_dict(
            checkpoint["normalizer"], schema
        )
        policy_class = FJSPControlledAssignmentGraphPolicy
        tensorizer = AssignmentGraphTensorizer(
            schema, normalizer=normalizer, validate_graphs=False
        )
    else:
        schema = GenericGraphFeatureSchema.from_dict(checkpoint["schema"])
        normalizer = GenericNumericFeatureNormalizer.from_dict(
            checkpoint["normalizer"], schema
        )
        policy_class = {
            CHECKPOINT_FORMAT: GenericCompactPolicyNetwork,
            CHECKPOINT_FORMAT_V7: GenericCompactV7PolicyNetwork,
            CHECKPOINT_FORMAT_MASKED_MLP: MaskedMLPPolicyNetwork,
            CHECKPOINT_FORMAT_PNC_PPO: PNCPPOPolicyNetwork,
            CHECKPOINT_FORMAT_PETRI_NET_GCN_PPO: PetriNetGCNPPOPolicyNetwork,
            CHECKPOINT_FORMAT_BSG_RL_SMALL: BSGRLSmallPolicyNetwork,
            CHECKPOINT_FORMAT_SEC55_ABLATION: (
                GenericCompactAblationPolicyNetwork
            ),
        }[checkpoint_format]
        tensorizer = GenericGraphTensorizer(
            schema, normalizer=normalizer, validate_graphs=False
        )
    policy_config = dict(checkpoint["policy_config"])
    if checkpoint_format == CHECKPOINT_FORMAT_SEC55_ABLATION:
        policy_config["ablation_variant"] = checkpoint["ablation_variant"]
    policy = policy_class(schema, **policy_config)
    trainer = TokenBindingPPO(
        policy,
        tensorizer,
        config=PPOConfig(**checkpoint["ppo_config"]),
        device=device,
    )
    trainer.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer.policy.eval()
    return trainer, checkpoint


def ordered_test_paths(task: EvaluationTask) -> List[Path]:
    jobs, machines = task.dimensions
    dataset = PublicFJSPDataset(
        task.test_dataset,
        expected_jobs=jobs,
        expected_machines=machines,
    )
    paths = list(dataset.paths)
    if task.distribution == "SD1":
        # This ordering matches the released Wang result arrays.
        paths.sort(key=lambda path: int(re.findall(r"\d+", path.name)[0]))
        paths.sort(key=lambda path: int(re.findall(r"\d+", path.name)[-1]))
    if len(paths) < task.test_limit:
        raise ValueError(
            f"{task.test_dataset} contains {len(paths)} instances, "
            f"but {task.test_limit} are required"
        )
    return paths[: task.test_limit]


def reference_makespans(task: EvaluationTask) -> np.ndarray:
    project_root = Path(__file__).resolve().parents[3]
    suffix = task.size if task.distribution == "SD1" else f"{task.size}+mix"
    path = (
        project_root
        / "or_solution"
        / task.distribution
        / f"solution_{suffix}.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(np.load(path), dtype=float)
    if values.ndim == 2:
        values = values[:, 0]
    if values.shape[0] < task.test_limit:
        raise ValueError(f"Reference vector is too short: {path}")
    return values[: task.test_limit]


def make_environment(
    path: Path,
    training: Mapping[str, Any],
) -> CPNGraphEnv:
    policy_variant = str(training.get("policy_variant", ""))
    if policy_variant == POLICY_AG_PPO:
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPAssignmentGraphBuilder,
        )

        graph_builder = FJSPAssignmentGraphBuilder(validate=False)
    else:
        graph_builder = FJSPV5GenericGraphBuilder(validate=False)
    return CPNGraphEnv(
        FJSFileV5NetFactory(path),
        graph_builder=graph_builder,
        reward_scale=float(training["reward_scale"]),
        reward_mode=str(training["reward_mode"]),
        potential_coefficient=float(training["potential_coefficient"]),
    )


def extract_assignments(env: CPNGraphEnv) -> Tuple[ScheduleAssignment, ...]:
    if env.net is None or not env.net.is_terminal():
        raise RuntimeError("Cannot extract a schedule from a non-terminal CPN")
    assignments: List[ScheduleAssignment] = []
    for place in env.net.places.values():
        for token in place.objects:
            if not isinstance(token, OperationToken):
                continue
            if (
                token.assigned_machine_id is None
                or token.start_time is None
                or token.finish_time is None
            ):
                raise RuntimeError(
                    f"Operation {token.op_uid} lacks a terminal assignment"
                )
            assignments.append(ScheduleAssignment(
                job_id=int(token.job_id) - 1,
                operation_index=int(token.op_index) - 1,
                machine_id=int(token.assigned_machine_id) - 1,
                start_time=float(token.start_time),
                finish_time=float(token.finish_time),
            ))
    return tuple(sorted(
        assignments,
        key=lambda item: (item.job_id, item.operation_index),
    ))


def validate_schedule(
    instance: CanonicalInstance,
    assignments: Sequence[ScheduleAssignment],
    reported_makespan: float,
    *,
    tolerance: float = 1e-7,
) -> List[str]:
    errors: List[str] = []
    expected = {
        (job_id, operation_index)
        for job_id, job in enumerate(instance.jobs)
        for operation_index in range(len(job))
    }
    grouped: Dict[Tuple[int, int], List[ScheduleAssignment]] = {}
    for assignment in assignments:
        grouped.setdefault(
            (assignment.job_id, assignment.operation_index), []
        ).append(assignment)
    actual = set(grouped)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    duplicates = sorted(key for key, values in grouped.items() if len(values) != 1)
    if missing:
        errors.append(f"missing_operations={missing}")
    if extra:
        errors.append(f"unknown_operations={extra}")
    if duplicates:
        errors.append(f"duplicate_operations={duplicates}")

    unique = {
        key: values[0]
        for key, values in grouped.items()
        if key in expected and len(values) == 1
    }
    for op_uid, assignment in unique.items():
        options = instance.options(op_uid)
        if assignment.machine_id not in options:
            errors.append(
                f"ineligible_machine={op_uid}->{assignment.machine_id}"
            )
            continue
        expected_duration = float(options[assignment.machine_id])
        actual_duration = assignment.finish_time - assignment.start_time
        if assignment.start_time < -tolerance:
            errors.append(f"negative_start={op_uid}:{assignment.start_time}")
        if abs(actual_duration - expected_duration) > tolerance:
            errors.append(
                f"duration_mismatch={op_uid}:{actual_duration}!={expected_duration}"
            )

    for job_id, job in enumerate(instance.jobs):
        for operation_index in range(1, len(job)):
            predecessor = unique.get((job_id, operation_index - 1))
            successor = unique.get((job_id, operation_index))
            if (
                predecessor is not None
                and successor is not None
                and predecessor.finish_time > successor.start_time + tolerance
            ):
                errors.append(
                    "precedence_violation="
                    f"{(job_id, operation_index - 1)}->"
                    f"{(job_id, operation_index)}"
                )

    by_machine: Dict[int, List[ScheduleAssignment]] = {}
    for assignment in unique.values():
        by_machine.setdefault(assignment.machine_id, []).append(assignment)
    for machine_id, machine_assignments in by_machine.items():
        ordered = sorted(
            machine_assignments,
            key=lambda item: (item.start_time, item.finish_time),
        )
        for previous, current in zip(ordered, ordered[1:]):
            if previous.finish_time > current.start_time + tolerance:
                errors.append(
                    "machine_overlap="
                    f"M{machine_id}:{(previous.job_id, previous.operation_index)}"
                    f"->{(current.job_id, current.operation_index)}"
                )

    if unique:
        computed_makespan = max(item.finish_time for item in unique.values())
        if abs(computed_makespan - reported_makespan) > tolerance:
            errors.append(
                f"makespan_mismatch={computed_makespan}!={reported_makespan}"
            )
    return errors


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def evaluate_batch(
    trainer: TokenBindingPPO,
    environments: Sequence[CPNGraphEnv],
    *,
    deterministic: bool,
    action_seeds: Optional[Sequence[int]] = None,
    max_decisions_per_episode: int = 100000,
) -> Tuple[List[RolloutOutcome], TimingTotals]:
    if not environments:
        raise ValueError("At least one environment is required")
    if not deterministic and (
        action_seeds is None or len(action_seeds) != len(environments)
    ):
        raise ValueError("Sampling requires one action seed per environment")

    started = time.perf_counter()
    t0 = time.perf_counter()
    observations = {
        index: env.reset() for index, env in enumerate(environments)
    }
    reset_seconds = time.perf_counter() - t0
    generators = (
        None
        if deterministic
        else [
            _generator(trainer.device, int(seed))
            for seed in action_seeds or ()
        ]
    )
    decision_counts = [0] * len(environments)
    final_makespans: List[Optional[float]] = [None] * len(environments)
    active = list(range(len(environments)))
    tensor_seconds = 0.0
    inference_seconds = 0.0
    selection_seconds = 0.0
    step_seconds = 0.0
    trainer.policy.eval()

    while active:
        active_observations = [observations[index] for index in active]
        if any(not observation.action_keys for observation in active_observations):
            raise RuntimeError("Active non-terminal graph has no Binding action")
        t0 = time.perf_counter()
        batch = trainer.tensorizer.tensorize(
            active_observations, device=trainer.device
        )
        tensor_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            output = trainer.policy(batch)
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        inference_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        selected_keys: List[Tuple[Any, ...]] = []
        for local_index, environment_index in enumerate(active):
            positions = torch.nonzero(
                batch.action_batch == local_index, as_tuple=False
            ).flatten()
            probabilities = output.probabilities[positions]
            if deterministic:
                local_choice = int(torch.argmax(probabilities).item())
            else:
                assert generators is not None
                local_choice = int(torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generators[environment_index],
                ).item())
            selected_keys.append(
                batch.action_keys[int(positions[local_choice])]
            )
        selection_seconds += time.perf_counter() - t0

        next_active: List[int] = []
        for environment_index, binding_key in zip(active, selected_keys):
            t0 = time.perf_counter()
            step = environments[environment_index].step(binding_key)
            step_seconds += time.perf_counter() - t0
            decision_counts[environment_index] += 1
            if decision_counts[environment_index] > max_decisions_per_episode:
                raise RuntimeError("Decision limit exceeded in evaluation")
            observations[environment_index] = step.observation
            if step.terminated:
                final_makespans[environment_index] = float(step.info["makespan"])
            else:
                next_active.append(environment_index)
        active = next_active

    outcomes = []
    for index, env in enumerate(environments):
        makespan = final_makespans[index]
        if makespan is None:
            raise RuntimeError("Evaluation ended without a terminal makespan")
        outcomes.append(RolloutOutcome(
            makespan=makespan,
            decision_count=decision_counts[index],
            assignments=extract_assignments(env),
        ))
    timing = TimingTotals(
        reset_and_initial_graph_s=reset_seconds,
        tensorization_s=tensor_seconds,
        network_inference_s=inference_seconds,
        action_selection_s=selection_seconds,
        environment_step_and_graph_s=step_seconds,
        end_to_end_s=time.perf_counter() - started,
        schedules=len(environments),
    )
    return outcomes, timing


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def checkpoint_audit(
    task: EvaluationTask,
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    training = checkpoint["training_config"]
    observed_distribution = str(training.get("distribution", "SD1"))
    observed_size = str(training.get("training_size", ""))
    observed_seed = int(training.get("seed", checkpoint.get("seed", -1)))
    expected = (task.distribution, task.size, task.training_seed)
    observed = (observed_distribution, observed_size, observed_seed)
    if observed != expected:
        raise ValueError(
            f"Checkpoint/task mismatch: expected={expected}, observed={observed}"
        )
    return {
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "best_validation_makespan": float(
            checkpoint["best_validation_makespan"]
        ),
        "policy_config": checkpoint["policy_config"],
        "training_protocol_version": training.get("protocol_version", "legacy"),
    }


def error_rows(
    task: EvaluationTask,
    instance_id: str,
    decoder: str,
    rollout_id: int,
    errors: Iterable[str],
) -> List[Dict[str, Any]]:
    return [{
        "distribution": task.distribution,
        "train_size": task.size,
        "test_size": task.size,
        "training_seed": task.training_seed,
        "instance_id": instance_id,
        "decoder": decoder,
        "rollout_id": rollout_id,
        "error": error,
    } for error in errors]
