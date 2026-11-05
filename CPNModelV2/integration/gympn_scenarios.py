from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import operator
from typing import Callable, Dict, Optional, Tuple

from gympn.simulator import GymProblem
from simpn.simulator import SimToken


PAPER_SCENARIO_NAMES: Tuple[str, ...] = (
    "sequence_joint",
    "sequence_disjoint",
    "parallel_joint",
    "parallel_disjoint",
    "loop_joint",
    "loop_disjoint",
    "exclusive_choice_joint",
    "exclusive_choice_disjoint",
)


@dataclass(frozen=True)
class GymPNPaperScenario:
    name: str
    control_flow: str
    resource_layout: str


PAPER_SCENARIOS: Dict[str, GymPNPaperScenario] = {
    name: GymPNPaperScenario(
        name=name,
        control_flow=name.rsplit("_", 1)[0],
        resource_layout=name.rsplit("_", 1)[1],
    )
    for name in PAPER_SCENARIO_NAMES
}


def build_minimal_task_assignment_problem(
    seed: Optional[int] = None,
) -> GymProblem:
    """GymPN's minimal heterogeneous task-resource assignment scenario."""

    del seed
    problem = _new_problem("minimal_task_assignment")
    arrival = problem.add_var("arrival", var_attributes=["task_type"])
    waiting = problem.add_var("waiting", var_attributes=["task_type"])
    busy = problem.add_var(
        "busy", var_attributes=["task_type", "resource_id"]
    )
    arrival.put({"task_type": 0})
    arrival.put({"task_type": 1})

    employee = problem.add_var(
        "employee", var_attributes=["code_employee"]
    )
    employee.put({"code_employee": 0})
    employee.put({"code_employee": 1})

    def arrive(task):
        return [SimToken(task, delay=1), SimToken(task)]

    problem.add_event([arrival], [arrival, waiting], arrive, name="arrive")
    problem.add_action(
        [waiting, employee],
        [busy],
        behavior=_start_processing,
        name="start",
    )

    def complete(assignment):
        return [SimToken(assignment[1])]

    problem.add_event(
        [busy],
        [employee],
        behavior=complete,
        name="complete",
        reward_function=lambda assignment: 1,
    )
    return problem


def build_paper_example_problem(
    scenario_name: str,
    seed: Optional[int] = None,
    *,
    case_multiplier: int = 1,
) -> GymProblem:
    """Build one of the eight workflow/resource scenarios used by GymPN."""

    del seed
    case_multiplier = _validate_case_multiplier(case_multiplier)
    try:
        scenario = PAPER_SCENARIOS[scenario_name]
    except KeyError as exc:
        choices = ", ".join(PAPER_SCENARIO_NAMES)
        raise ValueError(
            f"Unknown GymPN paper scenario {scenario_name!r}; choose {choices}"
        ) from exc
    if scenario.control_flow == "sequence":
        return _build_sequence(scenario, case_multiplier)
    if scenario.control_flow == "parallel":
        return _build_parallel(scenario, case_multiplier)
    if scenario.control_flow == "loop":
        return _build_loop(scenario, case_multiplier)
    if scenario.control_flow == "exclusive_choice":
        return _build_exclusive_choice(scenario, case_multiplier)
    raise AssertionError(f"Unhandled GymPN scenario: {scenario}")


def paper_scenario_factory(
    scenario_name: str,
    *,
    case_multiplier: int = 1,
) -> Callable[[Optional[int]], GymProblem]:
    """Return the seed-compatible factory expected by GymPNDecisionEnv."""

    if scenario_name not in PAPER_SCENARIOS:
        raise ValueError(f"Unknown GymPN paper scenario: {scenario_name}")
    case_multiplier = _validate_case_multiplier(case_multiplier)
    return partial(
        build_paper_example_problem,
        scenario_name,
        case_multiplier=case_multiplier,
    )


def _validate_case_multiplier(value: int) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("case_multiplier must be an integer") from exc
    if result < 1:
        raise ValueError("case_multiplier must be at least 1")
    return result


def _new_problem(
    scenario_name: str,
    case_multiplier: int = 1,
) -> GymProblem:
    problem = GymProblem(debugging=False, causal_rl=False)
    problem._debugging = False
    problem.gympn_scenario_name = scenario_name
    problem.gympn_case_multiplier = int(case_multiplier)
    return problem


def _add_resources(problem: GymProblem, place_name: str):
    place = problem.add_var(place_name, var_attributes=["code_employee"])
    for resource_id in range(3):
        place.put({"code_employee": resource_id})
    return place


def _start_processing(task, resource):
    task_type = int(task["task_type"])
    resource_type = int(resource["code_employee"])
    if task_type == resource_type:
        delay = 1
    elif resource_type in {0, 1}:
        delay = 2
    else:
        delay = 3
    return [SimToken((task, resource), delay=delay)]


def _build_sequence(
    scenario: GymPNPaperScenario,
    case_multiplier: int = 1,
) -> GymProblem:
    problem = _new_problem(scenario.name, case_multiplier)
    arrival = problem.add_var("arrival", var_attributes=["task_type"])
    waiting1 = problem.add_var("waiting1", var_attributes=["task_type"])
    busy1 = problem.add_var(
        "busy1", var_attributes=["task_type", "code_employee"]
    )
    waiting2 = problem.add_var("waiting2", var_attributes=["task_type"])
    busy2 = problem.add_var(
        "busy2", var_attributes=["task_type", "code_employee"]
    )
    for _stream_id in range(case_multiplier):
        arrival.put({"task_type": 0})
    employee1 = _add_resources(problem, "employee1")
    employee2 = (
        employee1
        if scenario.resource_layout == "joint"
        else _add_resources(problem, "employee2")
    )

    def arrive(task):
        return [SimToken(task, delay=1), SimToken(task)]

    def complete1(assignment):
        return [SimToken(assignment[-1]), SimToken({"task_type": 1})]

    def complete2(assignment):
        return [SimToken(assignment[-1])]

    problem.add_event([arrival], [arrival, waiting1], arrive, name="arrive")
    problem.add_action(
        [waiting1, employee1], [busy1], _start_processing, name="start1"
    )
    problem.add_event(
        [busy1], [employee1, waiting2], complete1, name="done1"
    )
    problem.add_action(
        [waiting2, employee2], [busy2], _start_processing, name="start2"
    )
    problem.add_event(
        [busy2],
        [employee2],
        complete2,
        name="done2",
        reward_function=lambda assignment: 1,
    )
    return problem


def _build_parallel(
    scenario: GymPNPaperScenario,
    case_multiplier: int = 1,
) -> GymProblem:
    problem = _new_problem(scenario.name, case_multiplier)
    case_attributes = ["task_type", "case_id"]
    arrival = problem.add_var("arrival", var_attributes=case_attributes)
    waiting1 = problem.add_var("waiting1", var_attributes=case_attributes)
    busy1 = problem.add_var(
        "busy1", var_attributes=["task_type", "code_employee", "case_id"]
    )
    waiting2 = problem.add_var("waiting2", var_attributes=case_attributes)
    busy2 = problem.add_var(
        "busy2", var_attributes=["task_type", "code_employee", "case_id"]
    )
    completed1 = problem.add_var("completed1", var_attributes=case_attributes)
    completed2 = problem.add_var("completed2", var_attributes=case_attributes)
    for stream_id in range(case_multiplier):
        arrival.put({"task_type": 0, "case_id": stream_id})
        arrival.put({"task_type": 1, "case_id": stream_id})
    employee1 = _add_resources(problem, "employee1")
    employee2 = (
        employee1
        if scenario.resource_layout == "joint"
        else _add_resources(problem, "employee2")
    )

    def arrive(task):
        # Each stream uses a disjoint case-id sequence so that parallel
        # branches still synchronize with the matching case only.
        task["case_id"] += case_multiplier
        if task["task_type"] == 0:
            return [SimToken(task, delay=1), SimToken(task), None]
        return [SimToken(task, delay=1), None, SimToken(task)]

    def complete(assignment):
        return [SimToken(assignment[-1]), SimToken(assignment[0])]

    def same_case(first, second):
        return first["case_id"] == second["case_id"]

    problem.add_event(
        [arrival], [arrival, waiting1, waiting2], arrive, name="arrive"
    )
    problem.add_action(
        [waiting1, employee1], [busy1], _start_processing, name="start1"
    )
    problem.add_event(
        [busy1], [employee1, completed1], complete, name="done1"
    )
    problem.add_action(
        [waiting2, employee2], [busy2], _start_processing, name="start2"
    )
    problem.add_event(
        [busy2], [employee2, completed2], complete, name="done2"
    )
    problem.add_event(
        [completed1, completed2],
        [],
        behavior=lambda first, second: [],
        name="doneFinal",
        reward_function=lambda first, second: 1,
        guard=same_case,
    )
    return problem


def _build_loop(
    scenario: GymPNPaperScenario,
    case_multiplier: int = 1,
) -> GymProblem:
    problem = _new_problem(scenario.name, case_multiplier)
    arrival = problem.add_var("arrival", var_attributes=["task_type"])
    waiting1 = problem.add_var("waiting1", var_attributes=["task_type"])
    busy1 = problem.add_var(
        "busy1", var_attributes=["task_type", "code_employee"]
    )
    waiting2 = problem.add_var("waiting2", var_attributes=["task_type"])
    busy2 = problem.add_var(
        "busy2", var_attributes=["task_type", "code_employee"]
    )
    completed = problem.add_var("completed", var_attributes=["task_type"])
    for _stream_id in range(case_multiplier):
        arrival.put({"task_type": 0})
    employee1 = _add_resources(problem, "employee1")
    employee2 = (
        employee1
        if scenario.resource_layout == "joint"
        else _add_resources(problem, "employee2")
    )

    def arrive(task):
        return [SimToken(task, delay=1), SimToken(task)]

    def complete1(assignment):
        return [SimToken(assignment[-1]), SimToken({"task_type": 1})]

    def complete2(assignment):
        resource = assignment[-1]
        matched = assignment[0]["task_type"] == resource["code_employee"]
        if matched:
            return [SimToken(resource), None, SimToken(assignment[0])]
        # The paper source resets only the disjoint variant to type 0.
        retry = (
            {"task_type": 0}
            if scenario.resource_layout == "disjoint"
            else assignment[0]
        )
        return [SimToken(resource), SimToken(retry), None]

    problem.add_event([arrival], [arrival, waiting1], arrive, name="arrive")
    problem.add_action(
        [waiting1, employee1], [busy1], _start_processing, name="start1"
    )
    problem.add_event(
        [busy1], [employee1, waiting2], complete1, name="done1"
    )
    problem.add_action(
        [waiting2, employee2], [busy2], _start_processing, name="start2"
    )
    problem.add_event(
        [busy2],
        [employee2, waiting1, completed],
        complete2,
        name="done2",
    )
    problem.add_event(
        [completed],
        [],
        behavior=lambda task: [],
        name="complete_case",
        reward_function=lambda task: 1,
    )
    return problem


def _build_exclusive_choice(
    scenario: GymPNPaperScenario,
    case_multiplier: int = 1,
) -> GymProblem:
    problem = _new_problem(scenario.name, case_multiplier)
    arrival = problem.add_var("arrival", var_attributes=["task_type"])
    waiting1 = problem.add_var("waiting1", var_attributes=["task_type"])
    busy1 = problem.add_var(
        "busy1", var_attributes=["task_type", "code_employee"]
    )
    waiting2 = problem.add_var("waiting2", var_attributes=["task_type"])
    busy2 = problem.add_var(
        "busy2", var_attributes=["task_type", "code_employee"]
    )
    for _stream_id in range(case_multiplier):
        arrival.put({"task_type": 0})
        arrival.put({"task_type": 1})
    employee1 = _add_resources(
        problem,
        "employee" if scenario.resource_layout == "joint" else "employee1",
    )
    employee2 = (
        employee1
        if scenario.resource_layout == "joint"
        else _add_resources(problem, "employee2")
    )

    def arrive(task):
        if task["task_type"] == 0:
            return [SimToken(task, delay=1), SimToken(task), None]
        return [SimToken(task, delay=1), None, SimToken(task)]

    def complete(assignment):
        return [SimToken(assignment[-1])]

    problem.add_event(
        [arrival], [arrival, waiting1, waiting2], arrive, name="arrive"
    )
    problem.add_action(
        [waiting1, employee1], [busy1], _start_processing, name="start1"
    )
    problem.add_event(
        [busy1],
        [employee1],
        complete,
        name="done1",
        reward_function=lambda assignment: 1,
    )
    problem.add_action(
        [waiting2, employee2], [busy2], _start_processing, name="start2"
    )
    problem.add_event(
        [busy2],
        [employee2],
        complete,
        name="done2",
        reward_function=lambda assignment: 1,
    )
    return problem
