from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence

import numpy as np

from CPNModelV2.core.fjsp_cpn_v1 import FJSPInstance
from CPNModelV2.core.fjsp_cpn_v4 import FJSPCPNV4
from CPNModelV2.core.fjsp_cpn_v5 import FJSPCPNV5
from CPNModelV2.graph import TokenBindingGraph, TokenBindingGraphBuilder


@dataclass(frozen=True)
class RandomFJSPConfig:
    num_jobs: int = 2
    num_machines: int = 2
    min_operations: int = 2
    max_operations: int = 2
    min_eligible_machines: int = 1
    max_eligible_machines: int = 2
    min_processing_time: int = 1
    max_processing_time: int = 9

    def validate(self) -> None:
        if self.num_jobs <= 0 or self.num_machines <= 0:
            raise ValueError("FJSP job and machine counts must be positive")
        if not 1 <= self.min_operations <= self.max_operations:
            raise ValueError("Invalid operation-count range")
        if not 1 <= self.min_eligible_machines <= self.max_eligible_machines:
            raise ValueError("Invalid eligible-machine range")
        if self.max_eligible_machines > self.num_machines:
            raise ValueError("Eligible-machine count exceeds machine count")
        if not 1 <= self.min_processing_time <= self.max_processing_time:
            raise ValueError("Invalid processing-time range")


class RandomFJSPNetFactory:
    """Deterministic-by-seed generator for static FJSP CPN episodes."""

    def __init__(self, config: RandomFJSPConfig) -> None:
        config.validate()
        self.config = config

    def __call__(self, seed: Optional[int] = None) -> FJSPCPNV4:
        rng = random.Random(0 if seed is None else seed)
        machine_ids = list(range(1, self.config.num_machines + 1))
        jobs: List[List[dict[int, float]]] = []
        for _ in range(self.config.num_jobs):
            operation_count = rng.randint(
                self.config.min_operations, self.config.max_operations
            )
            job = []
            for _ in range(operation_count):
                eligible_count = rng.randint(
                    self.config.min_eligible_machines,
                    self.config.max_eligible_machines,
                )
                eligible = rng.sample(machine_ids, eligible_count)
                job.append({
                    machine_id: float(rng.randint(
                        self.config.min_processing_time,
                        self.config.max_processing_time,
                    ))
                    for machine_id in eligible
                })
            jobs.append(job)
        instance = FJSPInstance.from_nested(jobs, machine_ids=machine_ids)
        return FJSPCPNV4(instance)


@dataclass(frozen=True)
class SongSD1Config:
    """Instance distribution used by Song and Wang for SD1 training."""

    num_jobs: int = 10
    num_machines: int = 5
    min_operations: int = 4
    max_operations: int = 6
    min_eligible_machines: int = 1
    max_eligible_machines: int = 5
    min_processing_time: int = 1
    max_processing_time: int = 20
    processing_time_deviation: float = 0.2

    def validate(self) -> None:
        if self.num_jobs <= 0 or self.num_machines <= 0:
            raise ValueError("SD1 job and machine counts must be positive")
        if not 1 <= self.min_operations <= self.max_operations:
            raise ValueError("Invalid SD1 operation-count range")
        if not 1 <= self.min_eligible_machines <= self.max_eligible_machines:
            raise ValueError("Invalid SD1 eligible-machine range")
        if self.max_eligible_machines > self.num_machines:
            raise ValueError("SD1 eligible-machine count exceeds machine count")
        if not 1 <= self.min_processing_time <= self.max_processing_time:
            raise ValueError("Invalid SD1 processing-time range")
        if not 0.0 <= self.processing_time_deviation < 1.0:
            raise ValueError("SD1 processing-time deviation must be in [0, 1)")


class SongSD1NetFactory:
    """Deterministic CPN generator matching the public SD1 distribution."""

    def __init__(self, config: SongSD1Config = SongSD1Config()) -> None:
        config.validate()
        self.config = config

    def __call__(
        self,
        seed: Optional[int] = None,
        *,
        operation_counts: Optional[Sequence[int]] = None,
    ) -> FJSPCPNV4:
        return FJSPCPNV4(self.generate_instance(
            seed, operation_counts=operation_counts
        ))

    def generate_instance(
        self,
        seed: Optional[int] = None,
        *,
        operation_counts: Optional[Sequence[int]] = None,
    ) -> FJSPInstance:
        rng = random.Random(0 if seed is None else seed)
        config = self.config
        if operation_counts is None:
            counts = [
                rng.randint(config.min_operations, config.max_operations)
                for _ in range(config.num_jobs)
            ]
        else:
            counts = [int(value) for value in operation_counts]
            if len(counts) != config.num_jobs:
                raise ValueError("SD1 operation-count vector has the wrong job count")
            if any(
                value < config.min_operations or value > config.max_operations
                for value in counts
            ):
                raise ValueError("SD1 operation count lies outside the configured range")

        total_operations = sum(counts)
        eligible_counts = [
            rng.randint(
                config.min_eligible_machines,
                config.max_eligible_machines,
            )
            for _ in range(total_operations)
        ]
        machine_ids = list(range(1, config.num_machines + 1))
        eligible_machines = [
            sorted(rng.sample(machine_ids, count)) for count in eligible_counts
        ]
        processing_means = [
            rng.randint(config.min_processing_time, config.max_processing_time)
            for _ in range(total_operations)
        ]

        flat_operations: List[dict[int, float]] = []
        for machines, processing_mean in zip(
            eligible_machines, processing_means
        ):
            lower = max(
                config.min_processing_time,
                round(processing_mean * (1.0 - config.processing_time_deviation)),
            )
            upper = min(
                config.max_processing_time,
                round(processing_mean * (1.0 + config.processing_time_deviation)),
            )
            flat_operations.append({
                machine_id: float(rng.randint(lower, upper))
                for machine_id in machines
            })

        jobs: List[List[dict[int, float]]] = []
        cursor = 0
        for count in counts:
            jobs.append(flat_operations[cursor : cursor + count])
            cursor += count
        return FJSPInstance.from_nested(jobs, machine_ids=machine_ids)


class SongSD1V5NetFactory(SongSD1NetFactory):
    """Generate matched SD1 instances as executable V5 CPNs."""

    def __call__(
        self,
        seed: Optional[int] = None,
        *,
        operation_counts: Optional[Sequence[int]] = None,
    ) -> FJSPCPNV5:
        return FJSPCPNV5(self.generate_instance(
            seed, operation_counts=operation_counts
        ))


@dataclass(frozen=True)
class WangSD2Config:
    """Configuration for Wang's SD2 +mix instance distribution."""

    num_jobs: int = 10
    num_machines: int = 5
    operations_per_job: Optional[int] = None
    min_eligible_machines: int = 1
    max_eligible_machines: Optional[int] = None
    min_processing_time: int = 1
    max_processing_time: int = 99

    def validate(self) -> None:
        if self.num_jobs <= 0 or self.num_machines <= 0:
            raise ValueError("SD2 job and machine counts must be positive")
        operations = (
            self.num_machines
            if self.operations_per_job is None
            else self.operations_per_job
        )
        if operations <= 0:
            raise ValueError("SD2 operations_per_job must be positive")
        max_eligible = (
            self.num_machines
            if self.max_eligible_machines is None
            else self.max_eligible_machines
        )
        if not (
            1
            <= self.min_eligible_machines
            <= max_eligible
            <= self.num_machines
        ):
            raise ValueError("Invalid SD2 eligible-machine range")
        if not 1 <= self.min_processing_time <= self.max_processing_time:
            raise ValueError("Invalid SD2 processing-time range")


class WangSD2V5NetFactory:
    """Generate deterministic executable V5 CPNs under Wang's SD2 mix rule."""

    GENERATOR_VERSION = "wang_sd2_mix_randomstate_v1"

    def __init__(self, config: WangSD2Config = WangSD2Config()) -> None:
        config.validate()
        self.config = config

    def __call__(self, seed: Optional[int] = None) -> FJSPCPNV5:
        return FJSPCPNV5(self.generate_instance(seed))

    def generate_instance(self, seed: Optional[int] = None) -> FJSPInstance:
        config = self.config
        rng = np.random.RandomState(0 if seed is None else seed)
        operations_per_job = (
            config.num_machines
            if config.operations_per_job is None
            else config.operations_per_job
        )
        max_eligible = (
            config.num_machines
            if config.max_eligible_machines is None
            else config.max_eligible_machines
        )
        total_operations = config.num_jobs * operations_per_job
        eligible_counts = rng.randint(
            low=config.min_eligible_machines,
            high=max_eligible + 1,
            size=total_operations,
        )
        processing_times = rng.randint(
            low=config.min_processing_time,
            high=config.max_processing_time + 1,
            size=(total_operations, config.num_machines),
        )

        machine_ids = tuple(range(1, config.num_machines + 1))
        flat_operations: List[dict[int, float]] = []
        for row_index, eligible_count in enumerate(eligible_counts):
            eligible_count = int(eligible_count)
            if eligible_count < config.num_machines:
                disabled = rng.choice(
                    np.arange(config.num_machines),
                    config.num_machines - eligible_count,
                    replace=False,
                )
                processing_times[row_index, disabled] = 0
            flat_operations.append({
                machine_id: float(processing_times[row_index, machine_id - 1])
                for machine_id in machine_ids
                if processing_times[row_index, machine_id - 1] > 0
            })

        jobs = [
            flat_operations[
                job_index * operations_per_job:
                (job_index + 1) * operations_per_job
            ]
            for job_index in range(config.num_jobs)
        ]
        return FJSPInstance.from_nested(jobs, machine_ids=machine_ids)


def collect_random_calibration_graphs(
    factory: RandomFJSPNetFactory,
    seeds: Iterable[int],
    *,
    include_terminal: bool = True,
) -> List[TokenBindingGraph]:
    """Collect frozen-schema and numeric-statistics graphs without a policy."""
    builder = TokenBindingGraphBuilder()
    graphs: List[TokenBindingGraph] = []
    for seed in seeds:
        rng = random.Random(seed + 1000003)
        net = factory(seed)
        net.advance_until_decision_or_terminal()
        while not net.is_terminal():
            graphs.append(builder.build(net))
            bindings = net.control_bindings()
            if not bindings:
                raise RuntimeError("Calibration CPN has no enabled Binding")
            net.step(rng.choice(bindings).key)
        if include_terminal:
            graphs.append(builder.build(net))
    if not graphs:
        raise ValueError("Calibration requires at least one graph")
    return graphs
