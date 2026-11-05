from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from CPNModelV2.graph import GraphTensorizer, TokenBindingGraph
from CPNModelV2.learning.environment import CPNGraphEnv
from CPNModelV2.learning.token_binding_policy import (
    BindingPolicyNetwork,
    select_binding_actions,
)


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 64
    device_microbatch_size: Optional[int] = None
    max_grad_norm: float = 0.5

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_ratio <= 0 or self.learning_rate <= 0:
            raise ValueError("PPO clip ratio and learning rate must be positive")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("PPO update sizes must be positive")
        if (
            self.device_microbatch_size is not None
            and self.device_microbatch_size <= 0
        ):
            raise ValueError("device_microbatch_size must be positive when set")


@dataclass
class RolloutTransition:
    observation: TokenBindingGraph
    action_key: Tuple[Any, ...]
    old_log_probability: float
    old_value: float
    reward: float
    terminated: bool
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class EpisodeResult:
    transitions: List[RolloutTransition]
    episode_return: float
    makespan: float
    decision_count: int


@dataclass(frozen=True)
class EvaluationEpisodeResult:
    """Terminal statistics for an inference-only episode.

    Unlike :class:`EpisodeResult`, this object deliberately keeps no graph
    observations or PPO transition data.  Long validation episodes can
    therefore be evaluated without retaining their complete computation
    histories in host memory.
    """

    episode_return: float
    makespan: float
    decision_count: int


class TokenBindingPPO:
    """PPO trainer for variable-size Binding action sets."""

    def __init__(
        self,
        policy: BindingPolicyNetwork,
        tensorizer: GraphTensorizer,
        *,
        config: PPOConfig = PPOConfig(),
        device: str = "cpu",
    ) -> None:
        config.validate()
        self.policy = policy.to(device)
        self.tensorizer = tensorizer
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )

    def collect_episode(
        self,
        env: CPNGraphEnv,
        *,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> EpisodeResult:
        observation = env.reset(seed=seed)
        transitions: List[RolloutTransition] = []
        self.policy.eval()
        while True:
            if not observation.action_keys:
                if env.net is not None and env.net.is_terminal():
                    break
                raise RuntimeError("Non-terminal observation has no Binding action")
            batch = self.tensorizer.tensorize(
                observation, device=self.device
            )
            with torch.no_grad():
                output = self.policy(batch)
            selection = select_binding_actions(
                output, batch, deterministic=deterministic
            )[0]
            if selection is None:
                raise RuntimeError("Policy returned no action for non-terminal graph")
            step = env.step(selection.binding_key)
            transitions.append(RolloutTransition(
                observation=observation,
                action_key=selection.binding_key,
                old_log_probability=selection.log_probability,
                old_value=float(output.values[0].detach().cpu()),
                reward=step.reward,
                terminated=step.terminated,
            ))
            observation = step.observation
            if step.terminated:
                self._compute_advantages(transitions)
                return EpisodeResult(
                    transitions=transitions,
                    episode_return=float(step.info["episode_return"]),
                    makespan=float(step.info["makespan"]),
                    decision_count=int(step.info["decision_count"]),
                )
        raise RuntimeError("Terminal CPN produced no episode result")

    def collect_episodes_batched(
        self,
        environments: Sequence[CPNGraphEnv],
        *,
        seeds: Optional[Sequence[Optional[int]]] = None,
        deterministic: bool = False,
        max_decisions_per_episode: int = 100000,
    ) -> List[EpisodeResult]:
        """Synchronously batch policy inference across independent CPNs."""
        if not environments:
            raise ValueError("At least one environment is required")
        if max_decisions_per_episode <= 0:
            raise ValueError("max_decisions_per_episode must be positive")
        seed_values = list(seeds) if seeds is not None else [None] * len(environments)
        if len(seed_values) != len(environments):
            raise ValueError("Environment and seed counts differ")

        observations = {
            index: env.reset(seed=seed_values[index])
            for index, env in enumerate(environments)
        }
        transition_lists: List[List[RolloutTransition]] = [
            [] for _ in environments
        ]
        final_info: List[Optional[Dict[str, Any]]] = [None] * len(environments)
        active = list(range(len(environments)))
        self.policy.eval()
        while active:
            active_observations = [observations[index] for index in active]
            if any(not observation.action_keys for observation in active_observations):
                raise RuntimeError("Active non-terminal graph has no Binding action")
            batch = self.tensorizer.tensorize(
                active_observations, device=self.device
            )
            with torch.no_grad():
                output = self.policy(batch)
            selections = select_binding_actions(
                output, batch, deterministic=deterministic
            )
            next_active = []
            for local_index, environment_index in enumerate(active):
                selection = selections[local_index]
                if selection is None:
                    raise RuntimeError("Policy returned no action for active graph")
                observation = observations[environment_index]
                step = environments[environment_index].step(selection.binding_key)
                transition_lists[environment_index].append(RolloutTransition(
                    observation=observation,
                    action_key=selection.binding_key,
                    old_log_probability=selection.log_probability,
                    old_value=float(output.values[local_index].detach().cpu()),
                    reward=step.reward,
                    terminated=step.terminated,
                ))
                if len(transition_lists[environment_index]) > max_decisions_per_episode:
                    raise RuntimeError("Decision limit exceeded in batched rollout")
                observations[environment_index] = step.observation
                if step.terminated:
                    final_info[environment_index] = step.info
                else:
                    next_active.append(environment_index)
            active = next_active

        results = []
        for transitions, info in zip(transition_lists, final_info):
            if info is None:
                raise RuntimeError("Batched rollout ended without terminal info")
            self._compute_advantages(transitions)
            results.append(EpisodeResult(
                transitions=transitions,
                episode_return=float(info["episode_return"]),
                makespan=float(info["makespan"]),
                decision_count=int(info["decision_count"]),
            ))
        return results

    def evaluate_episodes_batched(
        self,
        environments: Sequence[CPNGraphEnv],
        *,
        seeds: Optional[Sequence[Optional[int]]] = None,
        deterministic: bool = True,
        max_decisions_per_episode: int = 100000,
    ) -> List[EvaluationEpisodeResult]:
        """Run batched policy inference without constructing PPO rollouts.

        Action selection and environment stepping intentionally mirror
        :meth:`collect_episodes_batched`.  The evaluation path only records
        terminal statistics, so it does not compute values, advantages, or
        returns and does not retain one graph object per decision.
        """
        if not environments:
            raise ValueError("At least one environment is required")
        if max_decisions_per_episode <= 0:
            raise ValueError("max_decisions_per_episode must be positive")
        seed_values = list(seeds) if seeds is not None else [None] * len(environments)
        if len(seed_values) != len(environments):
            raise ValueError("Environment and seed counts differ")

        observations = {
            index: env.reset(seed=seed_values[index])
            for index, env in enumerate(environments)
        }
        decision_counts = [0] * len(environments)
        final_info: List[Optional[Dict[str, Any]]] = [None] * len(environments)
        active = list(range(len(environments)))
        self.policy.eval()
        while active:
            active_observations = [observations[index] for index in active]
            if any(not observation.action_keys for observation in active_observations):
                raise RuntimeError("Active non-terminal graph has no Binding action")
            batch = self.tensorizer.tensorize(
                active_observations, device=self.device
            )
            with torch.no_grad():
                output = self.policy(batch)
            selections = select_binding_actions(
                output, batch, deterministic=deterministic
            )
            next_active = []
            for local_index, environment_index in enumerate(active):
                selection = selections[local_index]
                if selection is None:
                    raise RuntimeError("Policy returned no action for active graph")
                step = environments[environment_index].step(selection.binding_key)
                decision_counts[environment_index] += 1
                if decision_counts[environment_index] > max_decisions_per_episode:
                    raise RuntimeError("Decision limit exceeded in batched evaluation")
                observations[environment_index] = step.observation
                if step.terminated:
                    final_info[environment_index] = step.info
                else:
                    next_active.append(environment_index)
            active = next_active

        results = []
        for decision_count, info in zip(decision_counts, final_info):
            if info is None:
                raise RuntimeError("Batched evaluation ended without terminal info")
            results.append(EvaluationEpisodeResult(
                episode_return=float(info["episode_return"]),
                makespan=float(info["makespan"]),
                decision_count=decision_count,
            ))
        return results

    def update(
        self,
        episodes: Sequence[EpisodeResult],
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        transitions = [
            transition
            for episode in episodes
            for transition in episode.transitions
        ]
        if not transitions:
            raise ValueError("PPO update requires at least one transition")
        advantages = torch.tensor(
            [item.advantage for item in transitions], dtype=torch.float32
        )
        if len(transitions) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        returns = torch.tensor(
            [item.return_value for item in transitions], dtype=torch.float32
        )
        old_log_probabilities = torch.tensor(
            [item.old_log_probability for item in transitions],
            dtype=torch.float32,
        )

        order = list(range(len(transitions)))
        rng = random.Random(seed)
        metrics: Dict[str, List[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
        }
        self.policy.train()
        for _ in range(self.config.update_epochs):
            rng.shuffle(order)
            for start in range(0, len(order), self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                self.optimizer.zero_grad(set_to_none=True)
                logical_size = len(indices)
                microbatch_size = (
                    self.config.device_microbatch_size or logical_size
                )
                accumulated = {
                    "loss": 0.0,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "approx_kl": 0.0,
                    "clip_fraction": 0.0,
                }
                for micro_start in range(0, logical_size, microbatch_size):
                    micro_indices = indices[
                        micro_start : micro_start + microbatch_size
                    ]
                    weight = len(micro_indices) / logical_size
                    batch = self.tensorizer.tensorize(
                        [
                            transitions[index].observation
                            for index in micro_indices
                        ],
                        device=self.device,
                    )
                    output = self.policy(batch)
                    chosen_positions = _chosen_action_positions(
                        batch.action_keys,
                        batch.action_batch,
                        [
                            transitions[index].action_key
                            for index in micro_indices
                        ],
                    ).to(self.device)
                    new_log_probability = torch.log(
                        output.probabilities[chosen_positions].clamp_min(1e-12)
                    )
                    old_log = old_log_probabilities[micro_indices].to(self.device)
                    advantage = advantages[micro_indices].to(self.device)
                    return_value = returns[micro_indices].to(self.device)
                    ratio = torch.exp(new_log_probability - old_log)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    ) * advantage
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (
                        output.values - return_value
                    ).square().mean()
                    entropy = _mean_segment_entropy(
                        output.probabilities,
                        batch.action_batch,
                        batch.num_graphs,
                    )
                    loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy
                    )
                    (weight * loss).backward()

                    with torch.no_grad():
                        log_ratio = new_log_probability - old_log
                        approx_kl = (
                            (torch.exp(log_ratio) - 1.0) - log_ratio
                        ).mean()
                        clipped_mask = (
                            torch.abs(ratio - 1.0) > self.config.clip_ratio
                        ).float().mean()
                    micro_values = {
                        "loss": loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy": entropy,
                        "approx_kl": approx_kl,
                        "clip_fraction": clipped_mask,
                    }
                    for name, value in micro_values.items():
                        scalar = float(value.detach().cpu())
                        if not math.isfinite(scalar):
                            raise RuntimeError(
                                f"Non-finite PPO metric {name}={scalar}"
                            )
                        accumulated[name] += weight * scalar

                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                grad_norm_scalar = float(grad_norm.detach().cpu())
                if not math.isfinite(grad_norm_scalar):
                    raise RuntimeError(
                        f"Non-finite PPO metric grad_norm={grad_norm_scalar}"
                    )
                for name, value in accumulated.items():
                    metrics[name].append(value)
                metrics["grad_norm"].append(grad_norm_scalar)

        return {
            name: sum(values) / len(values) for name, values in metrics.items()
        }

    def _compute_advantages(
        self, transitions: List[RolloutTransition]
    ) -> None:
        gae = 0.0
        next_value = 0.0
        for transition in reversed(transitions):
            non_terminal = 0.0 if transition.terminated else 1.0
            delta = (
                transition.reward
                + self.config.gamma * next_value * non_terminal
                - transition.old_value
            )
            gae = delta + (
                self.config.gamma
                * self.config.gae_lambda
                * non_terminal
                * gae
            )
            transition.advantage = float(gae)
            transition.return_value = float(gae + transition.old_value)
            next_value = transition.old_value


def _chosen_action_positions(
    batch_action_keys: Sequence[Tuple[Any, ...]],
    action_batch: torch.Tensor,
    chosen_keys: Sequence[Tuple[Any, ...]],
) -> torch.Tensor:
    positions = []
    action_batch_cpu = action_batch.detach().cpu().tolist()
    for graph_index, chosen_key in enumerate(chosen_keys):
        matches = [
            action_index
            for action_index, (owner, action_key) in enumerate(
                zip(action_batch_cpu, batch_action_keys)
            )
            if owner == graph_index and action_key == chosen_key
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one chosen Binding in graph {graph_index}, got {matches}"
            )
        positions.append(matches[0])
    return torch.tensor(positions, dtype=torch.long)


def _mean_segment_entropy(
    probabilities: torch.Tensor,
    action_batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    entropies = []
    for graph_index in range(num_graphs):
        values = probabilities[action_batch == graph_index]
        if values.numel():
            entropies.append(-(values * torch.log(values.clamp_min(1e-12))).sum())
    if not entropies:
        return probabilities.new_tensor(0.0)
    return torch.stack(entropies).mean()
