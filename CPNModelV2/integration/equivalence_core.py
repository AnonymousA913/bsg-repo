from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from CPNModelV2.core.fjsp_cpn_v1 import (
    FJSPCPN,
    FJSPInstance,
    P_O_COMPLETED,
    P_O_PROCESSING,
    P_O_READY,
    P_O_WAITING,
)


EPS = 1e-8
OperationUID = Tuple[int, int]


@dataclass(frozen=True, order=True)
class CanonicalAction:
    """Zero-based job, operation, and machine identifiers."""

    job_id: int
    op_index: int
    machine_id: int

    @property
    def op_uid(self) -> OperationUID:
        return self.job_id, self.op_index

    def label(self) -> str:
        return f"O({self.job_id + 1},{self.op_index + 1})->M{self.machine_id + 1}"


@dataclass(frozen=True)
class Assignment:
    machine_id: int
    start_time: float
    finish_time: float


@dataclass(frozen=True)
class CanonicalInstance:
    """Structured, zero-based representation of a static FJSP instance."""

    jobs: Tuple[Tuple[Tuple[Tuple[int, float], ...], ...], ...]
    num_machines: int
    source: str = "memory"

    @classmethod
    def from_nested(
        cls,
        jobs: Sequence[Sequence[Mapping[int, float]]],
        *,
        num_machines: Optional[int] = None,
        source: str = "memory",
    ) -> "CanonicalInstance":
        if not jobs:
            raise ValueError("At least one job is required")
        normalized_jobs = []
        inferred = set()
        for job_id, job in enumerate(jobs):
            if not job:
                raise ValueError(f"Job {job_id} has no operations")
            normalized_operations = []
            for op_index, options in enumerate(job):
                if not options:
                    raise ValueError(f"Operation {(job_id, op_index)} has no eligible machine")
                normalized = []
                for machine_id, processing_time in sorted(options.items()):
                    machine_id = int(machine_id)
                    processing_time = float(processing_time)
                    if machine_id < 0 or processing_time <= 0:
                        raise ValueError(f"Invalid option for operation {(job_id, op_index)}")
                    normalized.append((machine_id, processing_time))
                    inferred.add(machine_id)
                normalized_operations.append(tuple(normalized))
            normalized_jobs.append(tuple(normalized_operations))

        inferred_count = max(inferred) + 1
        machine_count = int(num_machines or inferred_count)
        if machine_count < inferred_count:
            raise ValueError("num_machines omits an eligible machine")
        return cls(tuple(normalized_jobs), machine_count, source)

    @classmethod
    def from_fjs(cls, path: Path | str) -> "CanonicalInstance":
        path = Path(path)
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            raise ValueError(f"Empty FJSP file: {path}")
        header = lines[0].split()
        if len(header) < 2:
            raise ValueError(f"Invalid FJSP header: {lines[0]}")
        num_jobs, num_machines = int(header[0]), int(header[1])
        if len(lines) - 1 < num_jobs:
            raise ValueError(f"Expected {num_jobs} job lines, found {len(lines) - 1}")

        jobs = []
        for job_id, line in enumerate(lines[1 : num_jobs + 1]):
            values = [int(value) for value in line.split()]
            if not values:
                raise ValueError(f"Missing data for job {job_id}")
            num_operations = values[0]
            cursor = 1
            operations = []
            for op_index in range(num_operations):
                if cursor >= len(values):
                    raise ValueError(f"Truncated operation {(job_id, op_index)}")
                num_options = values[cursor]
                cursor += 1
                options: Dict[int, float] = {}
                for _ in range(num_options):
                    if cursor + 1 >= len(values):
                        raise ValueError(f"Truncated machine option for {(job_id, op_index)}")
                    machine_id = values[cursor] - 1
                    processing_time = float(values[cursor + 1])
                    cursor += 2
                    if machine_id in options:
                        raise ValueError(f"Duplicate machine for {(job_id, op_index)}")
                    options[machine_id] = processing_time
                operations.append(options)
            if cursor != len(values):
                raise ValueError(f"Unexpected trailing values in job {job_id}: {values[cursor:]}")
            jobs.append(operations)
        return cls.from_nested(jobs, num_machines=num_machines, source=str(path.resolve()))

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def total_operations(self) -> int:
        return sum(len(job) for job in self.jobs)

    @property
    def job_lengths(self) -> Tuple[int, ...]:
        return tuple(len(job) for job in self.jobs)

    def options(self, op_uid: OperationUID) -> Dict[int, float]:
        job_id, op_index = op_uid
        return dict(self.jobs[job_id][op_index])

    def processing_time(self, action: CanonicalAction) -> float:
        options = self.options(action.op_uid)
        if action.machine_id not in options:
            raise ValueError(f"Ineligible action: {action.label()}")
        return float(options[action.machine_id])

    def global_operation_id(self, op_uid: OperationUID) -> int:
        job_id, op_index = op_uid
        return sum(self.job_lengths[:job_id]) + op_index

    def operation_uid(self, global_operation_id: int) -> OperationUID:
        cursor = 0
        for job_id, length in enumerate(self.job_lengths):
            if global_operation_id < cursor + length:
                return job_id, global_operation_id - cursor
            cursor += length
        raise IndexError(f"Unknown global operation id: {global_operation_id}")

    def to_fjs_text(self) -> str:
        relation_count = sum(len(operation) for job in self.jobs for operation in job)
        average = relation_count / self.total_operations
        lines = [f"{self.num_jobs} {self.num_machines} {average:.6f}"]
        for job in self.jobs:
            values = [str(len(job))]
            for operation in job:
                values.append(str(len(operation)))
                for machine_id, processing_time in operation:
                    values.extend((str(machine_id + 1), _format_number(processing_time)))
            lines.append(" ".join(values))
        return "\n".join(lines) + "\n"

    def to_cpn_instance(self) -> FJSPInstance:
        jobs = []
        for job in self.jobs:
            jobs.append([
                {machine_id + 1: processing_time for machine_id, processing_time in operation}
                for operation in job
            ])
        return FJSPInstance.from_nested(jobs, machine_ids=range(1, self.num_machines + 1))


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


@dataclass(frozen=True)
class SchedulerSnapshot:
    time: float
    candidates: frozenset[CanonicalAction]
    assignments: Mapping[OperationUID, Assignment]
    completed: frozenset[OperationUID]
    terminal: bool


# 统一动作接口，负责把CPN内部的Binding动作转为FJSP适用的动作
# 比如 CPN Binding: (T_StartOperation, O11, M2, E_O11_M2)
# 转为：CanonicalAction(job=0, operation=0, machine=1)，这样FJSP就可以根据CPN的Binding进行操作了
# 反过来，当Song策略输出这个动作时，CPNAdapter.step()会在当前CPN中找到唯一匹配的Binding并激发
class CPNAdapter:
    name = "cpn"

    def __init__(self, instance: CanonicalInstance) -> None:
        self.instance = instance
        self.net = FJSPCPN(instance.to_cpn_instance())
        self.net.advance_until_decision_or_terminal()

    def candidates(self) -> set[CanonicalAction]:
        actions = set()
        for binding in self.net.control_bindings():
            operation = binding.payload["operation"]
            machine = binding.payload["machine"]
            actions.add(CanonicalAction(
                operation.job_id - 1,
                operation.op_index - 1,
                machine.machine_id - 1,
            ))
        return actions

    def assignments(self) -> Dict[OperationUID, Assignment]:
        result = {}
        places = (P_O_WAITING, P_O_READY, P_O_PROCESSING, P_O_COMPLETED)
        for place_name in places:
            for operation in self.net.places[place_name].objects:
                if operation.assigned_machine_id is None:
                    continue
                result[(operation.job_id - 1, operation.op_index - 1)] = Assignment(
                    operation.assigned_machine_id - 1,
                    float(operation.start_time),
                    float(operation.finish_time),
                )
        return result

    def current_time(self) -> float:
        return float(self.net.now)

    def terminal(self) -> bool:
        return self.net.is_terminal()

    def step(self, action: CanonicalAction) -> None:
        matches = []
        for binding in self.net.control_bindings():
            operation = binding.payload["operation"]
            machine = binding.payload["machine"]
            candidate = CanonicalAction(
                operation.job_id - 1,
                operation.op_index - 1,
                machine.machine_id - 1,
            )
            if candidate == action:
                matches.append(binding)
        if len(matches) != 1:
            raise ValueError(f"Expected one CPN binding for {action.label()}, found {len(matches)}")
        self.net.step(matches[0].key)

    def makespan(self) -> float:
        return self.net.makespan()

    def close(self) -> None:
        return None


def snapshot(adapter: Any) -> SchedulerSnapshot:
    current_time = adapter.current_time()
    assignments = adapter.assignments()
    completed = frozenset(
        op_uid
        for op_uid, assignment in assignments.items()
        if assignment.finish_time <= current_time + EPS
    )
    return SchedulerSnapshot(
        time=current_time,
        candidates=frozenset(adapter.candidates()),
        assignments=dict(assignments),
        completed=completed,
        terminal=adapter.terminal(),
    )


@dataclass
class EquivalenceResult:
    instance: str
    baseline: str
    policy: str
    seed: int
    passed: bool
    decisions: int
    operations: int
    cpn_makespan: Optional[float] = None
    baseline_makespan: Optional[float] = None
    failure_step: Optional[int] = None
    failure_category: Optional[str] = None
    failure_detail: Optional[str] = None
    trace: list[str] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "baseline": self.baseline,
            "policy": self.policy,
            "seed": self.seed,
            "passed": self.passed,
            "decisions": self.decisions,
            "operations": self.operations,
            "cpn_makespan": self.cpn_makespan,
            "baseline_makespan": self.baseline_makespan,
            "failure_step": self.failure_step,
            "failure_category": self.failure_category,
            "failure_detail": self.failure_detail,
        }


def choose_action(
    instance: CanonicalInstance,
    candidates: Iterable[CanonicalAction],
    policy: str,
    rng: random.Random,
) -> CanonicalAction:
    candidates = sorted(candidates)
    if not candidates:
        raise ValueError("No candidate action")
    if policy == "lexicographic":
        return candidates[0]
    if policy == "spt":
        return min(candidates, key=lambda action: (instance.processing_time(action), action))
    if policy == "lpt":
        return min(candidates, key=lambda action: (-instance.processing_time(action), action))
    if policy == "random":
        return rng.choice(candidates)
    raise ValueError(f"Unknown policy: {policy}")


def compare_snapshots(
    left: SchedulerSnapshot,
    right: SchedulerSnapshot,
) -> Optional[Tuple[str, str]]:
    if abs(left.time - right.time) > EPS:
        return "time", f"CPN={left.time}, baseline={right.time}"
    if left.terminal != right.terminal:
        return "terminal", f"CPN={left.terminal}, baseline={right.terminal}"
    if left.candidates != right.candidates:
        only_cpn = sorted(action.label() for action in left.candidates - right.candidates)
        only_baseline = sorted(action.label() for action in right.candidates - left.candidates)
        return "candidates", f"only_cpn={only_cpn}; only_baseline={only_baseline}"
    if set(left.assignments) != set(right.assignments):
        return (
            "scheduled_operations",
            f"CPN={sorted(left.assignments)}; baseline={sorted(right.assignments)}",
        )
    for op_uid in sorted(left.assignments):
        cpn_assignment = left.assignments[op_uid]
        baseline_assignment = right.assignments[op_uid]
        if cpn_assignment.machine_id != baseline_assignment.machine_id:
            return "machine_assignment", f"operation={op_uid}; CPN={cpn_assignment}; baseline={baseline_assignment}"
        if abs(cpn_assignment.start_time - baseline_assignment.start_time) > EPS:
            return "start_time", f"operation={op_uid}; CPN={cpn_assignment}; baseline={baseline_assignment}"
        if abs(cpn_assignment.finish_time - baseline_assignment.finish_time) > EPS:
            return "finish_time", f"operation={op_uid}; CPN={cpn_assignment}; baseline={baseline_assignment}"
    if left.completed != right.completed:
        return "completed_operations", f"CPN={sorted(left.completed)}; baseline={sorted(right.completed)}"
    return None


def run_lockstep(
    instance: CanonicalInstance,
    baseline_factory: Callable[[CanonicalInstance], Any],
    *,
    policy: str,
    seed: int,
    cpn_factory: Callable[[CanonicalInstance], Any] = CPNAdapter,
) -> EquivalenceResult:
    cpn = cpn_factory(instance)
    baseline = baseline_factory(instance)
    result = EquivalenceResult(
        instance=instance.source,
        baseline=baseline.name,
        policy=policy,
        seed=seed,
        passed=False,
        decisions=0,
        operations=instance.total_operations,
    )
    rng = random.Random(seed)
    try:
        for decision in range(instance.total_operations + 1):
            cpn_state = snapshot(cpn)
            baseline_state = snapshot(baseline)
            difference = compare_snapshots(cpn_state, baseline_state)
            if difference is not None:
                result.failure_step = decision
                result.failure_category, result.failure_detail = difference
                return result
            if cpn_state.terminal:
                result.cpn_makespan = cpn.makespan()
                result.baseline_makespan = baseline.makespan()
                if abs(result.cpn_makespan - result.baseline_makespan) > EPS:
                    result.failure_step = decision
                    result.failure_category = "makespan"
                    result.failure_detail = (
                        f"CPN={result.cpn_makespan}, baseline={result.baseline_makespan}"
                    )
                    return result
                result.passed = True
                return result

            action = choose_action(instance, cpn_state.candidates, policy, rng)
            result.trace.append(action.label())
            cpn.step(action)
            baseline.step(action)
            result.decisions += 1

        result.failure_step = result.decisions
        result.failure_category = "decision_limit"
        result.failure_detail = "The environments did not terminate after every operation was dispatched"
        return result
    except Exception as error:  # Preserve a reproducible diagnostic row for batch runs.
        result.failure_step = result.decisions
        result.failure_category = "exception"
        result.failure_detail = f"{type(error).__name__}: {error}"
        return result
    finally:
        cpn.close()
        baseline.close()
