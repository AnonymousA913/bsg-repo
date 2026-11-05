from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, Optional, Tuple

from CPNModelV2.core.fjsp_cpn_v2 import (
    EligibilityToken,
    JobToken,
    OperationToken,
    P_E_ACTIVE,
    P_O_COMPLETED,
    P_O_PROCESSING,
)
from CPNModelV2.core.fjsp_cpn_v4 import FJSPCPNV4
from CPNModelV2.core.fjsp_cpn_v5 import FJSPCPNV5
from CPNModelV2.graph import (
    TokenBindingGraph,
    TokenBindingGraphBuilder,
    V5TokenBindingGraphBuilder,
)


NetFactory = Callable[[Optional[int]], FJSPCPNV4]
REWARD_ELAPSED = "elapsed"
REWARD_ESTIMATED_MAKESPAN_DELTA = "estimated_makespan_delta"
REWARD_REMAINING_WORK_POTENTIAL = "remaining_work_potential"
REWARD_MODES = (
    REWARD_ELAPSED,
    REWARD_ESTIMATED_MAKESPAN_DELTA,
    REWARD_REMAINING_WORK_POTENTIAL,
)


@dataclass
class CPNGraphStep:
    observation: TokenBindingGraph
    reward: float
    terminated: bool
    info: Dict[str, Any]


class CPNGraphEnv:
    """Decision-epoch RL wrapper around the executable CPN."""

    def __init__(
        self,
        net_factory: NetFactory,
        *,
        graph_builder: Optional[Any] = None,
        reward_scale: float = 1.0,
        reward_mode: str = REWARD_ELAPSED,
        potential_coefficient: float = 0.1,
    ) -> None:
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward_mode={reward_mode!r}")
        if potential_coefficient < 0:
            raise ValueError("potential_coefficient must be non-negative")
        self.net_factory = net_factory
        self.graph_builder = graph_builder
        self.reward_scale = float(reward_scale)
        self.reward_mode = reward_mode
        self.potential_coefficient = float(potential_coefficient)
        self.net: Optional[FJSPCPNV4] = None
        self.observation: Optional[TokenBindingGraph] = None
        self.decision_count = 0
        self.episode_return = 0.0
        self._reward_clock = 0.0
        self._initial_makespan_estimate = 0.0
        self._makespan_estimate = 0.0
        self._initial_remaining_work = 0.0
        self._remaining_work = 0.0

    def reset(self, *, seed: Optional[int] = None) -> TokenBindingGraph:
        self.net = self.net_factory(seed)
        if self.graph_builder is None:
            self.graph_builder = (
                V5TokenBindingGraphBuilder()
                if isinstance(self.net, FJSPCPNV5)
                else TokenBindingGraphBuilder()
            )
        self.net.advance_until_decision_or_terminal()
        self.decision_count = 0
        self.episode_return = 0.0
        self._reward_clock = 0.0
        self._initial_makespan_estimate = estimate_cpn_makespan(self.net)
        self._makespan_estimate = self._initial_makespan_estimate
        self._initial_remaining_work = estimate_cpn_remaining_work(self.net)
        self._remaining_work = self._initial_remaining_work
        self.observation = self.graph_builder.build(self.net)
        return self.observation

    def step(self, binding_key: Tuple[Any, ...]) -> CPNGraphStep:
        if self.net is None or self.observation is None:
            raise RuntimeError("Environment must be reset before step")
        if self.net.is_terminal():
            raise RuntimeError("Cannot step a terminal CPN")
        enabled = {binding.key for binding in self.net.control_bindings()}
        if binding_key not in enabled:
            raise ValueError(
                f"Binding is not enabled at t={self.net.now}: {binding_key}"
            )

        estimate_before = self._makespan_estimate
        remaining_work_before = self._remaining_work
        self.net.step(binding_key)
        elapsed = float(self.net.now) - self._reward_clock
        if elapsed < -1e-9:
            raise RuntimeError("CPN time moved backwards")
        estimate_after = estimate_cpn_makespan(self.net)
        remaining_work_after = estimate_cpn_remaining_work(self.net)
        if self.reward_mode == REWARD_ELAPSED:
            reward = -max(0.0, elapsed) / self.reward_scale
        elif self.reward_mode == REWARD_ESTIMATED_MAKESPAN_DELTA:
            reward = (estimate_before - estimate_after) / self.reward_scale
        else:
            reward = (
                -max(0.0, elapsed)
                + self.potential_coefficient
                * (remaining_work_before - remaining_work_after)
            ) / self.reward_scale
        self._reward_clock = float(self.net.now)
        self._makespan_estimate = estimate_after
        self._remaining_work = remaining_work_after
        self.decision_count += 1
        self.episode_return += reward
        terminated = self.net.is_terminal()
        self.observation = self.graph_builder.build(self.net)
        info: Dict[str, Any] = {
            "time": float(self.net.now),
            "decision_count": self.decision_count,
            "episode_return": self.episode_return,
            "enabled_actions": len(self.observation.action_keys),
            "reward_mode": self.reward_mode,
            "potential_coefficient": self.potential_coefficient,
            "estimated_makespan": estimate_after,
            "estimated_remaining_work": remaining_work_after,
        }
        if terminated:
            makespan = float(self.net.makespan())
            if self.reward_mode == REWARD_ELAPSED:
                expected_return = -makespan / self.reward_scale
            elif self.reward_mode == REWARD_ESTIMATED_MAKESPAN_DELTA:
                expected_return = (
                    self._initial_makespan_estimate - makespan
                ) / self.reward_scale
            else:
                expected_return = (
                    self.potential_coefficient * self._initial_remaining_work
                    - makespan
                ) / self.reward_scale
            info.update({
                "makespan": makespan,
                "initial_estimated_makespan": self._initial_makespan_estimate,
                "initial_estimated_remaining_work": self._initial_remaining_work,
                "expected_episode_return": expected_return,
                "return_matches_objective": abs(
                    self.episode_return - expected_return
                ) <= 1e-7,
                "return_matches_makespan": abs(
                    self.episode_return + makespan / self.reward_scale
                ) <= 1e-7,
            })
        return CPNGraphStep(self.observation, reward, terminated, info)


def estimate_cpn_makespan(net: FJSPCPNV4) -> float:
    """Estimate final makespan from the current marking without future leakage."""

    if net.is_terminal():
        return float(net.makespan())

    operations: Dict[int, list[OperationToken]] = defaultdict(list)
    jobs: Dict[int, JobToken] = {}
    active_times: Dict[Tuple[int, int], list[float]] = defaultdict(list)
    known_times: Dict[Tuple[int, int], list[float]] = defaultdict(list)
    for place_name, place in net.places.items():
        for token in place.objects:
            if isinstance(token, OperationToken):
                operations[token.job_id].append(token)
            elif isinstance(token, JobToken):
                jobs[token.job_id] = token
            elif isinstance(token, EligibilityToken):
                processing_time = float(token.processing_time)
                known_times[token.op_uid].append(processing_time)
                if place_name == P_E_ACTIVE:
                    active_times[token.op_uid].append(processing_time)

    projected_completions = [float(net.now)]
    for job_id, job_operations in operations.items():
        job = jobs.get(job_id)
        cursor = max(0.0, float(job.release_time) if job is not None else 0.0)
        for operation in sorted(job_operations, key=lambda item: item.op_index):
            if operation.finish_time is not None:
                cursor = max(cursor, float(operation.finish_time))
                continue
            processing_times = (
                active_times.get(operation.op_uid)
                or known_times.get(operation.op_uid)
                or [0.0]
            )
            if operation.ready_time is not None:
                cursor = max(cursor, float(operation.ready_time), float(net.now))
            cursor += mean(processing_times)
        if job is not None and job.completion_time is not None:
            cursor = max(cursor, float(job.completion_time))
        projected_completions.append(cursor)

    return max(projected_completions)


def estimate_cpn_remaining_work(net: FJSPCPNV4) -> float:
    """Potential over unfinished work; it is exactly zero at termination."""

    active_times: Dict[Tuple[int, int], list[float]] = defaultdict(list)
    known_times: Dict[Tuple[int, int], list[float]] = defaultdict(list)
    operations: list[OperationToken] = []
    for place_name, place in net.places.items():
        for token in place.objects:
            if isinstance(token, OperationToken):
                operations.append(token)
            elif isinstance(token, EligibilityToken):
                processing_time = float(token.processing_time)
                known_times[token.op_uid].append(processing_time)
                if place_name == P_E_ACTIVE:
                    active_times[token.op_uid].append(processing_time)

    remaining = 0.0
    for operation in operations:
        if operation.current_place == P_O_COMPLETED:
            continue
        if (
            operation.current_place == P_O_PROCESSING
            and operation.finish_time is not None
        ):
            remaining += max(0.0, float(operation.finish_time) - float(net.now))
            continue
        processing_times = (
            active_times.get(operation.op_uid)
            or known_times.get(operation.op_uid)
            or [0.0]
        )
        remaining += mean(processing_times)
    return remaining
