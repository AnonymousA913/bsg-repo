from __future__ import annotations

"""Controlled relational-complexity extensions of the eight GymPN scenarios.

The source-faithful GymPN scenarios remain in :mod:`gympn_scenarios`.  This
module adds the composite-complexity environments used by Section 5.6 without
changing the published P0 models.  Every extended scenario uses the same
semantic object vocabulary (flow, activity, pool, resource, eligibility, and
work Tokens), so complexity changes do not also change the observation
contract.
"""

from dataclasses import dataclass
from functools import partial
import inspect
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from gympn.simulator import GymProblem
from simpn.simulator import SimToken

from CPNModelV2.integration.gympn_scenarios import (
    PAPER_SCENARIOS,
    PAPER_SCENARIO_NAMES,
    build_paper_example_problem,
)


# Operational identifiers deliberately retain C1 and C2 so completed runs can
# be reused.  The paper presents PAPER_COMPLEXITY_LEVELS as L1--L5.  C3 is the
# retired extreme stress point and is excluded from the main experiment.
PAPER_COMPLEXITY_LEVELS: Tuple[str, ...] = (
    "C1",
    "C1_5",
    "C2",
    "C2_25",
    "C2_5",
)
PAPER_COMPLEXITY_LABELS: Dict[str, str] = {
    level: f"L{index}"
    for index, level in enumerate(PAPER_COMPLEXITY_LEVELS, start=1)
}
LEGACY_STRESS_LEVELS: Tuple[str, ...] = ("C3",)
COMPLEXITY_LEVEL_NAMES: Tuple[str, ...] = (
    "P0",
    *PAPER_COMPLEXITY_LEVELS,
    *LEGACY_STRESS_LEVELS,
)


@dataclass(frozen=True)
class GymPNComplexityConfig:
    """One predeclared point on the Section 5.6 complexity staircase."""

    level: str
    flow_size: int
    case_streams: int
    case_classes: int
    resources_per_pool: int
    eligibility_per_activity: int
    processing_time_max: int
    target_offered_load: float

    def validate(self) -> None:
        if self.level not in COMPLEXITY_LEVEL_NAMES[1:]:
            choices = ", ".join(COMPLEXITY_LEVEL_NAMES[1:])
            raise ValueError(
                f"Extended level must be one of {choices}, got {self.level!r}"
            )
        integer_fields = (
            self.flow_size,
            self.case_streams,
            self.case_classes,
            self.resources_per_pool,
            self.eligibility_per_activity,
            self.processing_time_max,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("All integer complexity parameters must be positive")
        if self.eligibility_per_activity > self.resources_per_pool:
            raise ValueError("Eligibility count cannot exceed resources per pool")
        if not 0.0 < self.target_offered_load <= 2.0:
            raise ValueError("Target offered load must lie in (0, 2]")


COMPLEXITY_CONFIGS: Dict[str, GymPNComplexityConfig] = {
    "C1": GymPNComplexityConfig(
        level="C1",
        flow_size=3,
        case_streams=2,
        case_classes=2,
        resources_per_pool=4,
        eligibility_per_activity=3,
        processing_time_max=4,
        target_offered_load=0.85,
    ),
    "C1_5": GymPNComplexityConfig(
        level="C1_5",
        flow_size=3,
        case_streams=3,
        case_classes=3,
        resources_per_pool=4,
        eligibility_per_activity=3,
        processing_time_max=6,
        target_offered_load=0.95,
    ),
    "C2": GymPNComplexityConfig(
        level="C2",
        flow_size=4,
        case_streams=4,
        case_classes=4,
        resources_per_pool=5,
        eligibility_per_activity=3,
        processing_time_max=8,
        target_offered_load=1.05,
    ),
    "C2_25": GymPNComplexityConfig(
        level="C2_25",
        flow_size=5,
        case_streams=4,
        case_classes=4,
        resources_per_pool=5,
        eligibility_per_activity=3,
        processing_time_max=10,
        target_offered_load=1.10,
    ),
    "C2_5": GymPNComplexityConfig(
        level="C2_5",
        flow_size=5,
        case_streams=5,
        case_classes=5,
        resources_per_pool=5,
        eligibility_per_activity=3,
        processing_time_max=12,
        target_offered_load=1.15,
    ),
    # Retained only so historical C3 checkpoints remain loadable.  New formal
    # runs should use PAPER_COMPLEXITY_LEVELS unless explicitly stress testing.
    "C3": GymPNComplexityConfig(
        level="C3",
        flow_size=6,
        case_streams=6,
        case_classes=6,
        resources_per_pool=6,
        eligibility_per_activity=3,
        processing_time_max=12,
        target_offered_load=1.20,
    ),
}


NetFactory = Callable[[Optional[int]], GymProblem]


def complexity_scenario_factory(
    scenario_name: str,
    complexity_level: str,
) -> NetFactory:
    """Return a seed-compatible GymPN scenario factory."""

    _validate_scenario_name(scenario_name)
    level = _normalize_level(complexity_level)
    return partial(
        build_complexity_scenario_problem,
        scenario_name,
        level,
    )


def build_complexity_scenario_problem(
    scenario_name: str,
    complexity_level: str,
    seed: Optional[int] = None,
) -> GymProblem:
    """Build one of eight workflows at a predeclared complexity level.

    P0 delegates to the source-faithful builder.  Extended levels use the
    common relational schema below.  Instance seeds rotate balanced resource
    ranks; all realized processing and quality facts remain visible to every
    policy.
    """

    _validate_scenario_name(scenario_name)
    level = _normalize_level(complexity_level)
    if level == "P0":
        problem = build_paper_example_problem(scenario_name, seed=seed)
        problem.gympn_complexity_level = "P0"
        problem.gympn_complexity_family = "gympn_paper_original"
        problem.gympn_semantic_references = False
        return problem

    config = COMPLEXITY_CONFIGS[level]
    config.validate()
    scenario = PAPER_SCENARIOS[scenario_name]
    if scenario.control_flow == "sequence":
        return _build_sequence(scenario_name, scenario.resource_layout, config, seed)
    if scenario.control_flow == "parallel":
        return _build_parallel(scenario_name, scenario.resource_layout, config, seed)
    if scenario.control_flow == "loop":
        return _build_loop(scenario_name, scenario.resource_layout, config, seed)
    if scenario.control_flow == "exclusive_choice":
        return _build_exclusive_choice(
            scenario_name,
            scenario.resource_layout,
            config,
            seed,
        )
    raise AssertionError(f"Unhandled GymPN control flow: {scenario.control_flow}")


def _new_problem(
    scenario_name: str,
    resource_layout: str,
    config: GymPNComplexityConfig,
) -> GymProblem:
    problem = GymProblem(debugging=False, causal_rl=False)
    problem._debugging = False
    problem.gympn_scenario_name = scenario_name
    problem.gympn_case_multiplier = config.case_streams
    problem.gympn_complexity_level = config.level
    problem.gympn_complexity_family = "relational_workflow_v1"
    problem.gympn_semantic_references = True
    problem.gympn_flow_size = config.flow_size
    problem.gympn_case_streams = config.case_streams
    problem.gympn_case_classes = config.case_classes
    problem.gympn_resources_per_pool = config.resources_per_pool
    problem.gympn_eligibility_per_activity = config.eligibility_per_activity
    problem.gympn_processing_time_max = config.processing_time_max
    problem.gympn_target_offered_load = config.target_offered_load
    problem.gympn_resource_layout = resource_layout
    control_flow = PAPER_SCENARIOS[scenario_name].control_flow
    problem.gympn_declared_token_samples = _declared_token_samples(
        control_flow,
        config,
    )
    # The generic GymPN adapter falls back to positional input roles for
    # source models.  Extended scenarios declare the executable meaning of
    # each start-transition input so role-wise Binding composition does not
    # have to infer semantics from an arbitrary input index.
    problem.gympn_participation_roles = {}
    return problem


def _declared_token_samples(
    control_flow: str,
    config: GymPNComplexityConfig,
) -> Tuple[Tuple[str, Mapping[str, Any]], ...]:
    """Declare every Token shape that may appear at a decision epoch.

    Some shapes, especially the arrival-stream and busy-assignment Tokens,
    occur only under particular event/action timing.  Calibration must not
    depend on a small random rollout happening to expose them.
    """

    work = _make_work(0, 0, 0)
    resource = {
        "object_id": _resource_id(0, 0),
        "resource_id": "0",
        "pool_ref": _pool_id(0),
    }
    eligibility = {
        "object_id": "eligibility:0:0:0:0",
        "activity_ref": _activity_id(0),
        "resource_ref": _resource_id(0, 0),
        "case_class": "0",
        "processing_time": float(_ranked_processing_time(0, config)),
        "quality": 0.0,
    }
    return (
        ("flow_catalog", {
            "object_id": "flow:root",
            "flow_kind": control_flow,
        }),
        ("activity_catalog", {
            "object_id": _activity_id(0),
            "activity_index": "0",
            "flow_ref": "flow:root",
            "next_ref": _activity_id(1 if config.flow_size > 1 else 0),
        }),
        ("pool_catalog", {
            "object_id": _pool_id(0),
            "pool_index": "0",
        }),
        ("resources_0", resource),
        ("eligibility_catalog", eligibility),
        ("arrival", {"stream_id": 0, "next_case_id": 0}),
        ("waiting_0", work),
        ("busy_0", {
            "work": work,
            "resource": resource,
            "eligibility": eligibility,
        }),
    )


def _install_semantic_catalog(
    problem: GymProblem,
    *,
    control_flow: str,
    resource_layout: str,
    config: GymPNComplexityConfig,
    seed: Optional[int],
) -> Tuple[Sequence[Any], Any, float]:
    """Add shared flow/activity/resource/eligibility Token objects."""

    flow_catalog = problem.add_var(
        "flow_catalog",
        var_attributes=["object_id", "flow_kind"],
    )
    flow_catalog.put({"object_id": "flow:root", "flow_kind": control_flow})

    activity_catalog = problem.add_var(
        "activity_catalog",
        var_attributes=["object_id", "activity_index", "flow_ref", "next_ref"],
    )
    for activity_index in range(config.flow_size):
        if control_flow == "loop":
            next_index: Optional[int] = (activity_index + 1) % config.flow_size
        elif control_flow == "sequence" and activity_index + 1 < config.flow_size:
            next_index = activity_index + 1
        else:
            next_index = None
        activity_catalog.put({
            "object_id": _activity_id(activity_index),
            "activity_index": str(activity_index),
            "flow_ref": "flow:root",
            "next_ref": (
                _activity_id(next_index) if next_index is not None else ""
            ),
        })

    pool_count = 1 if resource_layout == "joint" else config.flow_size
    pool_catalog = problem.add_var(
        "pool_catalog",
        var_attributes=["object_id", "pool_index"],
    )
    resource_places = []
    for pool_index in range(pool_count):
        pool_id = _pool_id(pool_index)
        pool_catalog.put({"object_id": pool_id, "pool_index": str(pool_index)})
        place = problem.add_var(
            f"resources_{pool_index}",
            var_attributes=["object_id", "resource_id", "pool_ref"],
        )
        for local_resource in range(config.resources_per_pool):
            resource_id = _resource_id(pool_index, local_resource)
            place.put({
                "object_id": resource_id,
                "resource_id": str(local_resource),
                "pool_ref": pool_id,
            })
        resource_places.append(place)

    eligibility_catalog = problem.add_var(
        "eligibility_catalog",
        var_attributes=[
            "object_id",
            "activity_ref",
            "resource_ref",
            "case_class",
            "processing_time",
            "quality",
        ],
    )
    seed_offset = int(seed or 0) % config.resources_per_pool
    for activity_index in range(config.flow_size):
        pool_index = 0 if resource_layout == "joint" else activity_index
        for case_class in range(config.case_classes):
            fastest = (
                activity_index * config.case_classes + case_class + seed_offset
            ) % config.resources_per_pool
            eligible_resources = [
                (fastest + rank) % config.resources_per_pool
                for rank in range(config.eligibility_per_activity)
            ]
            for rank, local_resource in enumerate(eligible_resources):
                processing_time = _ranked_processing_time(rank, config)
                resource_id = _resource_id(pool_index, local_resource)
                eligibility_catalog.put({
                    "object_id": (
                        f"eligibility:{activity_index}:{case_class}:"
                        f"{pool_index}:{local_resource}"
                    ),
                    "activity_ref": _activity_id(activity_index),
                    "resource_ref": resource_id,
                    "case_class": str(case_class),
                    "processing_time": float(processing_time),
                    # Faster resources deliberately have lower quality.  Only
                    # the loop quality gate consumes this field.
                    "quality": float(rank),
                })

    arrival_interval = _arrival_interval(
        control_flow,
        resource_layout,
        config,
    )
    problem.gympn_arrival_interval = arrival_interval
    return resource_places, eligibility_catalog, arrival_interval


def _build_sequence(
    scenario_name: str,
    resource_layout: str,
    config: GymPNComplexityConfig,
    seed: Optional[int],
) -> GymProblem:
    problem = _new_problem(scenario_name, resource_layout, config)
    resources, eligibility, arrival_interval = _install_semantic_catalog(
        problem,
        control_flow="sequence",
        resource_layout=resource_layout,
        config=config,
        seed=seed,
    )
    arrival = _add_arrival_place(problem, config)
    waiting, busy = _add_work_places(problem, config.flow_size)
    _seed_arrival_streams(arrival, config)

    def arrive(stream: Mapping[str, Any]):
        next_stream, work = _next_arrival(stream, config, arrival_interval, 0)
        return [next_stream, work]

    problem.add_event([arrival], [arrival, waiting[0]], arrive, name="arrive")
    for activity_index in range(config.flow_size):
        resource_place = resources[
            0 if resource_layout == "joint" else activity_index
        ]
        _add_start_action(
            problem,
            activity_index,
            waiting[activity_index],
            resource_place,
            eligibility,
            busy[activity_index],
        )
        if activity_index + 1 < config.flow_size:
            problem.add_event(
                [busy[activity_index]],
                [resource_place, waiting[activity_index + 1]],
                _SequenceCompletion(activity_index + 1),
                name=f"done_{activity_index}",
            )
        else:
            problem.add_event(
                [busy[activity_index]],
                [resource_place],
                _FinalCompletion(),
                name=f"done_{activity_index}",
                reward_function=lambda assignment: 1,
            )
    return problem


def _build_parallel(
    scenario_name: str,
    resource_layout: str,
    config: GymPNComplexityConfig,
    seed: Optional[int],
) -> GymProblem:
    problem = _new_problem(scenario_name, resource_layout, config)
    resources, eligibility, arrival_interval = _install_semantic_catalog(
        problem,
        control_flow="parallel",
        resource_layout=resource_layout,
        config=config,
        seed=seed,
    )
    arrival = _add_arrival_place(problem, config)
    waiting, busy = _add_work_places(problem, config.flow_size)
    completed = [
        problem.add_var(
            f"completed_{i}",
            var_attributes=_work_attributes(),
        )
        for i in range(config.flow_size)
    ]
    _seed_arrival_streams(arrival, config)

    def arrive(stream: Mapping[str, Any]):
        case_id, case_class, next_stream = _advance_stream(
            stream,
            config,
            arrival_interval,
        )
        works = [
            SimToken(_make_work(case_id, case_class, i))
            for i in range(config.flow_size)
        ]
        return [next_stream, *works]

    problem.add_event(
        [arrival],
        [arrival, *waiting],
        arrive,
        name="arrive",
    )
    for activity_index in range(config.flow_size):
        resource_place = resources[
            0 if resource_layout == "joint" else activity_index
        ]
        _add_start_action(
            problem,
            activity_index,
            waiting[activity_index],
            resource_place,
            eligibility,
            busy[activity_index],
        )
        problem.add_event(
            [busy[activity_index]],
            [resource_place, completed[activity_index]],
            _ParallelCompletion(),
            name=f"done_{activity_index}",
        )

    join = _FixedArityCallable(
        config.flow_size,
        lambda *works: [],
        name="join_parallel",
    )
    same_case = _FixedArityCallable(
        config.flow_size,
        lambda *works: len({int(work["case_id"]) for work in works}) == 1,
        name="same_case",
    )
    reward = _FixedArityCallable(
        config.flow_size,
        lambda *works: 1,
        name="complete_case_reward",
    )
    problem.add_event(
        completed,
        [],
        join,
        name="join",
        guard=same_case,
        reward_function=reward,
    )
    return problem


def _build_loop(
    scenario_name: str,
    resource_layout: str,
    config: GymPNComplexityConfig,
    seed: Optional[int],
) -> GymProblem:
    problem = _new_problem(scenario_name, resource_layout, config)
    resources, eligibility, arrival_interval = _install_semantic_catalog(
        problem,
        control_flow="loop",
        resource_layout=resource_layout,
        config=config,
        seed=seed,
    )
    arrival = _add_arrival_place(problem, config)
    waiting, busy = _add_work_places(problem, config.flow_size)
    completed = problem.add_var("completed", var_attributes=_work_attributes())
    _seed_arrival_streams(arrival, config)

    def arrive(stream: Mapping[str, Any]):
        next_stream, work = _next_arrival(stream, config, arrival_interval, 0)
        return [next_stream, work]

    problem.add_event([arrival], [arrival, waiting[0]], arrive, name="arrive")
    for activity_index in range(config.flow_size):
        resource_place = resources[
            0 if resource_layout == "joint" else activity_index
        ]
        _add_start_action(
            problem,
            activity_index,
            waiting[activity_index],
            resource_place,
            eligibility,
            busy[activity_index],
        )
        if activity_index + 1 < config.flow_size:
            problem.add_event(
                [busy[activity_index]],
                [resource_place, waiting[activity_index + 1]],
                _SequenceCompletion(activity_index + 1),
                name=f"done_{activity_index}",
            )
        else:
            problem.add_event(
                [busy[activity_index]],
                [resource_place, waiting[0], completed],
                _LoopQualityCompletion(config.eligibility_per_activity),
                name=f"done_{activity_index}",
            )
    problem.add_event(
        [completed],
        [],
        lambda work: [],
        name="complete_case",
        reward_function=lambda work: 1,
    )
    return problem


def _build_exclusive_choice(
    scenario_name: str,
    resource_layout: str,
    config: GymPNComplexityConfig,
    seed: Optional[int],
) -> GymProblem:
    problem = _new_problem(scenario_name, resource_layout, config)
    resources, eligibility, arrival_interval = _install_semantic_catalog(
        problem,
        control_flow="exclusive_choice",
        resource_layout=resource_layout,
        config=config,
        seed=seed,
    )
    arrival = _add_arrival_place(problem, config)
    waiting, busy = _add_work_places(problem, config.flow_size)
    _seed_arrival_streams(arrival, config)

    def arrive(stream: Mapping[str, Any]):
        case_id, case_class, next_stream = _advance_stream(
            stream,
            config,
            arrival_interval,
        )
        branch = case_class % config.flow_size
        outputs = [None] * config.flow_size
        outputs[branch] = SimToken(_make_work(case_id, case_class, branch))
        return [next_stream, *outputs]

    problem.add_event(
        [arrival],
        [arrival, *waiting],
        arrive,
        name="arrive",
    )
    for activity_index in range(config.flow_size):
        resource_place = resources[
            0 if resource_layout == "joint" else activity_index
        ]
        _add_start_action(
            problem,
            activity_index,
            waiting[activity_index],
            resource_place,
            eligibility,
            busy[activity_index],
        )
        problem.add_event(
            [busy[activity_index]],
            [resource_place],
            _FinalCompletion(),
            name=f"done_{activity_index}",
            reward_function=lambda assignment: 1,
        )
    return problem


def _add_start_action(
    problem: GymProblem,
    activity_index: int,
    waiting: Any,
    resources: Any,
    eligibility: Any,
    busy: Any,
) -> None:
    def eligible(
        work: Mapping[str, Any],
        resource: Mapping[str, Any],
        relation: Mapping[str, Any],
    ) -> bool:
        return (
            str(work["activity_ref"]) == str(relation["activity_ref"])
            and str(work["case_class"]) == str(relation["case_class"])
            and str(resource["object_id"]) == str(relation["resource_ref"])
        )

    def start(
        work: Mapping[str, Any],
        resource: Mapping[str, Any],
        relation: Mapping[str, Any],
    ):
        assignment = {
            "work": dict(work),
            "resource": dict(resource),
            "eligibility": dict(relation),
        }
        return [
            SimToken(assignment, delay=float(relation["processing_time"])),
            SimToken(dict(relation)),
        ]

    transition_name = f"start_{activity_index}"
    problem.add_action(
        [waiting, resources, eligibility],
        [busy, eligibility],
        start,
        name=transition_name,
        guard=eligible,
    )
    problem.gympn_participation_roles[transition_name] = (
        "consume::work",
        "consume::resource",
        "read::eligibility",
    )


class _SequenceCompletion:
    def __init__(self, next_activity: int) -> None:
        self.next_activity = int(next_activity)
        self.__name__ = f"complete_to_{next_activity}"

    def __call__(self, assignment: Mapping[str, Any]):
        work = dict(assignment["work"])
        work["object_id"] = _work_id(int(work["case_id"]), self.next_activity)
        work["activity_ref"] = _activity_id(self.next_activity)
        return [
            SimToken(dict(assignment["resource"])),
            SimToken(work),
        ]


class _ParallelCompletion:
    __name__ = "complete_parallel_branch"

    def __call__(self, assignment: Mapping[str, Any]):
        return [
            SimToken(dict(assignment["resource"])),
            SimToken(dict(assignment["work"])),
        ]


class _FinalCompletion:
    __name__ = "complete_final_activity"

    def __call__(self, assignment: Mapping[str, Any]):
        return [SimToken(dict(assignment["resource"]))]


class _LoopQualityCompletion:
    __name__ = "complete_loop_quality_gate"

    def __init__(self, eligibility_count: int) -> None:
        self.eligibility_count = int(eligibility_count)

    def __call__(self, assignment: Mapping[str, Any]):
        work = dict(assignment["work"])
        relation = assignment["eligibility"]
        threshold_span = max(1, self.eligibility_count - 1)
        required_quality = 1 + int(work["case_class"]) % threshold_span
        passed = float(relation["quality"]) >= float(required_quality)
        resource_token = SimToken(dict(assignment["resource"]))
        if passed:
            return [resource_token, None, SimToken(work)]
        work["rework_count"] = int(work.get("rework_count", 0)) + 1
        work["object_id"] = _work_id(int(work["case_id"]), 0)
        work["activity_ref"] = _activity_id(0)
        return [resource_token, SimToken(work), None]


class _FixedArityCallable:
    """Callable with an inspectable positional signature of runtime arity."""

    def __init__(
        self,
        arity: int,
        function: Callable[..., Any],
        *,
        name: str,
    ) -> None:
        self.arity = int(arity)
        self.function = function
        self.__name__ = str(name)
        self.__signature__ = inspect.Signature([
            inspect.Parameter(f"value_{index}", inspect.Parameter.POSITIONAL_ONLY)
            for index in range(self.arity)
        ])

    def __call__(self, *values: Any) -> Any:
        if len(values) != self.arity:
            raise TypeError(f"Expected {self.arity} values, got {len(values)}")
        return self.function(*values)


def _add_arrival_place(
    problem: GymProblem,
    config: GymPNComplexityConfig,
) -> Any:
    return problem.add_var(
        "arrival",
        var_attributes=["stream_id", "next_case_id"],
    )


def _add_work_places(
    problem: GymProblem,
    flow_size: int,
) -> Tuple[Sequence[Any], Sequence[Any]]:
    waiting = [
        problem.add_var(f"waiting_{i}", var_attributes=_work_attributes())
        for i in range(flow_size)
    ]
    busy = [
        problem.add_var(
            f"busy_{i}",
            var_attributes=["work", "resource", "eligibility"],
        )
        for i in range(flow_size)
    ]
    return waiting, busy


def _work_attributes() -> Sequence[str]:
    return (
        "object_id",
        "case_id",
        "case_class",
        "activity_ref",
        "rework_count",
    )


def _seed_arrival_streams(arrival: Any, config: GymPNComplexityConfig) -> None:
    for stream_id in range(config.case_streams):
        arrival.put({
            "stream_id": stream_id,
            "next_case_id": stream_id,
        })


def _advance_stream(
    stream: Mapping[str, Any],
    config: GymPNComplexityConfig,
    arrival_interval: float,
) -> Tuple[int, int, SimToken]:
    case_id = int(stream["next_case_id"])
    # Spread simultaneous streams over the case classes and rotate the class
    # assigned to each stream on later arrivals.  Using only
    # ``case_id // case_streams`` would synchronize every stream onto the same
    # class, causing exclusive-choice cases to arrive in one-branch bursts and
    # leaving the other disjoint resource pools idle.
    arrival_index = case_id // config.case_streams
    case_class = (
        int(stream["stream_id"]) + arrival_index
    ) % config.case_classes
    next_stream = {
        "stream_id": int(stream["stream_id"]),
        "next_case_id": case_id + config.case_streams,
    }
    return (
        case_id,
        case_class,
        SimToken(next_stream, delay=float(arrival_interval)),
    )


def _next_arrival(
    stream: Mapping[str, Any],
    config: GymPNComplexityConfig,
    arrival_interval: float,
    activity_index: int,
) -> Tuple[SimToken, SimToken]:
    case_id, case_class, next_stream = _advance_stream(
        stream,
        config,
        arrival_interval,
    )
    return (
        next_stream,
        SimToken(_make_work(case_id, case_class, activity_index)),
    )


def _make_work(case_id: int, case_class: int, activity_index: int) -> Dict[str, Any]:
    return {
        "object_id": _work_id(case_id, activity_index),
        "case_id": int(case_id),
        "case_class": str(case_class),
        "activity_ref": _activity_id(activity_index),
        "rework_count": 0,
    }


def _arrival_interval(
    control_flow: str,
    resource_layout: str,
    config: GymPNComplexityConfig,
) -> float:
    """Calibrate a controlled offered-load staircase.

    The reference workload is the expected demand of a policy that chooses
    uniformly among eligible resources.  The main L1--L5 ladder increases
    offered load gradually; legacy C3 remains the retired extreme point.  A
    policy that exploits the visible eligibility and quality relations can
    reduce its effective load, while indiscriminate choices accumulate a
    bounded but informative queue over the finite episode horizon.
    """

    eligible_times = [
        _ranked_processing_time(rank, config)
        for rank in range(config.eligibility_per_activity)
    ]
    mean_processing = sum(eligible_times) / float(len(eligible_times))
    expected_visits = 1.0
    if control_flow == "loop":
        threshold_span = max(1, config.eligibility_per_activity - 1)
        visits = []
        for case_class in range(config.case_classes):
            required_rank = 1 + case_class % threshold_span
            passing_choices = config.eligibility_per_activity - required_rank
            pass_probability = (
                passing_choices / float(config.eligibility_per_activity)
            )
            visits.append(1.0 / pass_probability)
        expected_visits = sum(visits) / float(len(visits))

    if control_flow == "exclusive_choice":
        activities_per_visit = 1.0
    else:
        activities_per_visit = float(config.flow_size)

    if resource_layout == "joint":
        capacity = float(config.resources_per_pool)
        workload = activities_per_visit * mean_processing * expected_visits
    elif control_flow == "exclusive_choice":
        # Arrivals are balanced over the exclusive branches, so the combined
        # capacity of their disjoint pools is available to the full stream.
        capacity = float(config.flow_size * config.resources_per_pool)
        workload = mean_processing
    else:
        # Each case visits every separate pool once per pass.  The bottleneck
        # pool, rather than total system capacity, determines the arrival rate.
        capacity = float(config.resources_per_pool)
        workload = mean_processing * expected_visits

    arrival_rate = config.target_offered_load * capacity / workload
    interval = float(config.case_streams) / max(arrival_rate, 1e-9)
    return max(0.05, interval)


def _ranked_processing_time(
    rank: int,
    config: GymPNComplexityConfig,
) -> int:
    if config.eligibility_per_activity == 1:
        return 1
    fraction = rank / float(config.eligibility_per_activity - 1)
    return 1 + int(round(fraction * (config.processing_time_max - 1)))


def _activity_id(activity_index: int) -> str:
    return f"activity:{int(activity_index)}"


def _pool_id(pool_index: int) -> str:
    return f"pool:{int(pool_index)}"


def _resource_id(pool_index: int, local_resource: int) -> str:
    return f"resource:{int(pool_index)}:{int(local_resource)}"


def _work_id(case_id: int, activity_index: int) -> str:
    return f"work:{int(case_id)}:{int(activity_index)}"


def _normalize_level(value: str) -> str:
    result = str(value).upper()
    if result not in COMPLEXITY_LEVEL_NAMES:
        choices = ", ".join(COMPLEXITY_LEVEL_NAMES)
        raise ValueError(f"Unknown GymPN complexity level {value!r}; choose {choices}")
    return result


def _validate_scenario_name(value: str) -> None:
    if value not in PAPER_SCENARIO_NAMES:
        choices = ", ".join(PAPER_SCENARIO_NAMES)
        raise ValueError(f"Unknown GymPN scenario {value!r}; choose {choices}")


__all__ = [
    "COMPLEXITY_CONFIGS",
    "COMPLEXITY_LEVEL_NAMES",
    "GymPNComplexityConfig",
    "LEGACY_STRESS_LEVELS",
    "PAPER_COMPLEXITY_LABELS",
    "PAPER_COMPLEXITY_LEVELS",
    "build_complexity_scenario_problem",
    "complexity_scenario_factory",
]
