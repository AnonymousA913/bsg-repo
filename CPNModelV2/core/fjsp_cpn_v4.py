from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from CPNModelV2.core.fjsp_cpn_v1 import FJSPInstance
from CPNModelV2.core.fjsp_cpn_v2 import (
    Binding,
    EligibilityToken,
    JobToken,
    MachineToken,
    OperationToken,
    P_E_ACTIVE,
    P_J_ACTIVE,
    P_M_AVAILABLE,
    P_M_BUSY,
    P_O_COMPLETED,
    P_O_PROCESSING,
    P_O_READY,
    P_O_WAITING,
    P_R_ACTIVE,
    PrecedenceToken,
    Transition,
)
from CPNModelV2.core.fjsp_cpn_v3 import (
    EVENT_ELIGIBILITY_ADD,
    EVENT_JOB_ADD,
    EVENT_JOB_RELEASE,
    EVENT_MACHINE_ADD,
    EVENT_PRECEDENCE_ADD,
    EventToken,
    FJSPCPNV3,
    P_E_INACTIVE,
    P_EVT_APPLIED,
    P_EVT_PENDING,
    P_J_COMPLETED,
    P_J_UNRELEASED,
    P_M_OFFLINE,
    P_O_INTERRUPTED,
    P_R_INACTIVE,
    _apply_event,
    _due_events,
)

# V4版本：加入更加复杂的异常事件，尤其是新增关系结构，且是在线实时新增
# 1. 加入了在线新增Job、Machine、合法边和加工顺序

T_ADD_JOB = "T_AddJob"
T_ADD_MACHINE = "T_AddMachine"
T_ADD_ELIGIBILITY = "T_AddEligibility"
T_ADD_PRECEDENCE = "T_AddPrecedence"

OPERATION_PLACES = (
    P_O_WAITING,
    P_O_READY,
    P_O_PROCESSING,
    P_O_INTERRUPTED,
    P_O_COMPLETED,
)
MACHINE_PLACES = (P_M_AVAILABLE, P_M_BUSY, P_M_OFFLINE)
JOB_PLACES = (P_J_UNRELEASED, P_J_ACTIVE, P_J_COMPLETED)


class FJSPCPNV4(FJSPCPNV3):
    """CPN V3 plus online creation of jobs, machines, and relation tokens."""

    def __init__(
        self,
        instance: FJSPInstance,
        *,
        job_release_times: Optional[Mapping[int, float]] = None,
        events: Sequence[EventToken] = (),
    ) -> None:
        self.total_jobs = len(instance.jobs)
        self.total_machines = len(instance.machine_ids)
        super().__init__(
            instance,
            job_release_times=job_release_times,
            events=events,
        )

    def _build_transitions(self) -> None:
        super()._build_transitions()
        for name, bind_fn, apply_fn in (
            (T_ADD_JOB, _bind_add_job, _apply_add_job),
            (T_ADD_MACHINE, _bind_add_machine, _apply_add_machine),
            (T_ADD_ELIGIBILITY, _bind_add_eligibility, _apply_add_eligibility),
            (T_ADD_PRECEDENCE, _bind_add_precedence, _apply_add_precedence),
        ):
            self.transitions[name] = Transition(
                name,
                controlled=False,
                bind_fn=bind_fn,
                apply_fn=apply_fn,
                priority=25,
            )

    def _build_arcs(self) -> None:
        super()._build_arcs()

        def p2t(place: str, transition: str, arc_type: str) -> None:
            self.arcs.append({
                "src_type": "place",
                "src": place,
                "dst_type": "transition",
                "dst": transition,
                "arc_type": arc_type,
            })

        def t2p(transition: str, place: str, arc_type: str) -> None:
            self.arcs.append({
                "src_type": "transition",
                "src": transition,
                "dst_type": "place",
                "dst": place,
                "arc_type": arc_type,
            })

        for transition in (
            T_ADD_JOB,
            T_ADD_MACHINE,
            T_ADD_ELIGIBILITY,
            T_ADD_PRECEDENCE,
        ):
            p2t(P_EVT_PENDING, transition, "consume_due_event")
            t2p(transition, P_EVT_APPLIED, "produce_applied_event")

        for place in MACHINE_PLACES:
            p2t(place, T_ADD_JOB, "read_referenced_machines")
        t2p(T_ADD_JOB, P_J_ACTIVE, "produce_job")
        t2p(T_ADD_JOB, P_O_WAITING, "produce_operations")
        t2p(T_ADD_JOB, P_E_ACTIVE, "produce_eligibility")
        t2p(T_ADD_JOB, P_R_ACTIVE, "produce_precedence")

        for place in OPERATION_PLACES:
            p2t(place, T_ADD_MACHINE, "read_referenced_operations")
        t2p(T_ADD_MACHINE, P_M_AVAILABLE, "produce_machine")
        t2p(T_ADD_MACHINE, P_E_ACTIVE, "produce_eligibility")

        for place in OPERATION_PLACES:
            p2t(place, T_ADD_ELIGIBILITY, "read_operation")
            p2t(place, T_ADD_PRECEDENCE, "read_operations")
        for place in MACHINE_PLACES:
            p2t(place, T_ADD_ELIGIBILITY, "read_machine")
        t2p(T_ADD_ELIGIBILITY, P_E_ACTIVE, "produce_eligibility")
        t2p(T_ADD_PRECEDENCE, P_R_ACTIVE, "produce_precedence")

    def is_terminal(self) -> bool:
        future_work = any(
            event.event_type in {EVENT_JOB_ADD, EVENT_JOB_RELEASE}
            for event in self.places[P_EVT_PENDING].objects
        )
        return not future_work and super().is_terminal()

    def check_invariants(self) -> Dict[str, int]:
        counts = super().check_invariants()
        operations = {
            operation.op_uid
            for place in OPERATION_PLACES
            for operation in self.places[place].objects
        }
        machines = {
            machine.machine_id
            for place in MACHINE_PLACES
            for machine in self.places[place].objects
        }
        jobs = {
            job.job_id
            for place in JOB_PLACES
            for job in self.places[place].objects
        }
        errors = []
        for relation in (
            list(self.places[P_E_ACTIVE].objects)
            + list(self.places[P_E_INACTIVE].objects)
        ):
            if relation.op_uid not in operations:
                errors.append(f"Eligibility {relation.relation_id} has unknown operation")
            if relation.machine_id not in machines:
                errors.append(f"Eligibility {relation.relation_id} has unknown machine")
        for relation in (
            list(self.places[P_R_ACTIVE].objects)
            + list(self.places[P_R_INACTIVE].objects)
        ):
            if relation.predecessor_uid not in operations:
                errors.append(f"Precedence {relation.relation_id} has unknown predecessor")
            if relation.successor_uid not in operations:
                errors.append(f"Precedence {relation.relation_id} has unknown successor")
        for operation_uid in operations:
            if operation_uid[0] not in jobs:
                errors.append(f"Operation {operation_uid} has unknown job")
        if errors:
            raise AssertionError("\n".join(errors))
        return counts


def _tokens_with_places(net: FJSPCPNV4, places: Iterable[str]) -> List[Tuple[str, Any]]:
    return [
        (place, token)
        for place in places
        for token in net.places[place].objects
    ]


def _find_token_with_place(
    net: FJSPCPNV4,
    places: Iterable[str],
    attribute: str,
    value: Any,
) -> Optional[Tuple[str, Any]]:
    return next(
        (
            (place, token)
            for place, token in _tokens_with_places(net, places)
            if getattr(token, attribute) == value
        ),
        None,
    )


def _relation_ids(net: FJSPCPNV4) -> set[str]:
    return {
        relation.relation_id
        for place in (P_E_ACTIVE, P_E_INACTIVE, P_R_ACTIVE, P_R_INACTIVE)
        for relation in net.places[place].objects
    }


def _group_read_objects(items: Iterable[Tuple[str, Any]]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for place, token in items:
        grouped.setdefault(place, []).append(token)
    return grouped


def _new_event_binding(
    transition: str,
    event: EventToken,
    prepared: Dict[str, Any],
    read_items: Iterable[Tuple[str, Any]] = (),
) -> Binding:
    return Binding(
        transition_name=transition,
        key=(transition, event.event_id),
        consume_objects={P_EVT_PENDING: [event]},
        read_objects=_group_read_objects(read_items),
        payload={"event": event, **prepared},
    )


def _parse_operation_uid(value: Any) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("op_uid must contain [job_id, op_index]")
    result = int(value[0]), int(value[1])
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("Operation identifiers must be positive")
    return result


def _is_acyclic(edges: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    indegree: Dict[Tuple[int, int], int] = {}
    for predecessor, successor in edges:
        adjacency.setdefault(predecessor, []).append(successor)
        indegree.setdefault(predecessor, 0)
        indegree[successor] = indegree.get(successor, 0) + 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for successor in adjacency.get(node, []):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    return visited == len(indegree)


def _prepare_add_job(net: FJSPCPNV4, event: EventToken) -> Dict[str, Any]:
    job_id = int(event.target_id)
    if _find_token_with_place(net, JOB_PLACES, "job_id", job_id) is not None:
        raise ValueError(f"job_id already exists: {job_id}")
    payload = event.payload or {}
    raw_operations = payload.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("JOB_ADD payload requires a non-empty operations list")

    machine_tokens = {
        machine.machine_id: (place, machine)
        for place, machine in _tokens_with_places(net, MACHINE_PLACES)
    }
    operations: List[OperationToken] = []
    eligibility: List[EligibilityToken] = []
    read_machines: Dict[int, Tuple[str, MachineToken]] = {}
    existing_relation_ids = _relation_ids(net)
    for zero_index, operation_spec in enumerate(raw_operations):
        options = (
            operation_spec.get("options")
            if isinstance(operation_spec, dict) and "options" in operation_spec
            else operation_spec
        )
        if not isinstance(options, dict) or not options:
            raise ValueError(f"Operation {zero_index + 1} requires machine options")
        op_uid = (job_id, zero_index + 1)
        operations.append(OperationToken(
            op_uid=op_uid,
            job_id=job_id,
            op_index=zero_index + 1,
            num_operations=len(raw_operations),
        ))
        for raw_machine_id, raw_time in options.items():
            machine_id = int(raw_machine_id)
            processing_time = float(raw_time)
            if machine_id not in machine_tokens:
                raise ValueError(f"Unknown machine_id={machine_id}")
            if processing_time <= 0:
                raise ValueError("Processing time must be positive")
            relation_id = f"E_O{job_id}_{zero_index + 1}_M{machine_id}"
            if relation_id in existing_relation_ids:
                raise ValueError(f"Relation already exists: {relation_id}")
            existing_relation_ids.add(relation_id)
            eligibility.append(EligibilityToken(
                relation_id=relation_id,
                op_uid=op_uid,
                machine_id=machine_id,
                processing_time=processing_time,
            ))
            read_machines[machine_id] = machine_tokens[machine_id]

    raw_precedence = payload.get("precedence")
    if raw_precedence is None:
        raw_precedence = [
            [index, index + 1] for index in range(1, len(operations))
        ]
    precedence: List[PrecedenceToken] = []
    edges = []
    seen_edges = set()
    for raw_edge in raw_precedence:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            raise ValueError("Each precedence edge must contain [predecessor, successor]")
        predecessor_index, successor_index = int(raw_edge[0]), int(raw_edge[1])
        if not (1 <= predecessor_index <= len(operations)):
            raise ValueError("Predecessor index is outside the new job")
        if not (1 <= successor_index <= len(operations)):
            raise ValueError("Successor index is outside the new job")
        edge = ((job_id, predecessor_index), (job_id, successor_index))
        if edge in seen_edges or edge[0] == edge[1]:
            raise ValueError("Duplicate or self precedence edge")
        seen_edges.add(edge)
        edges.append(edge)
        relation_id = (
            f"R_O{job_id}_{predecessor_index}_O{job_id}_{successor_index}"
        )
        if relation_id in existing_relation_ids:
            raise ValueError(f"Relation already exists: {relation_id}")
        existing_relation_ids.add(relation_id)
        precedence.append(PrecedenceToken(relation_id, edge[0], edge[1]))
    if not _is_acyclic(edges):
        raise ValueError("New job precedence graph contains a cycle")

    job = JobToken(
        job_id=job_id,
        num_operations=len(operations),
        release_time=float(event.event_time),
        due_date=(
            None if payload.get("due_date") is None else float(payload["due_date"])
        ),
        priority=float(payload.get("priority", 1.0)),
    )
    return {
        "job": job,
        "operations": operations,
        "eligibility": eligibility,
        "precedence": precedence,
        "read_items": list(read_machines.values()),
    }


def _bind_add_job(net: FJSPCPNV4, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_JOB_ADD, now):
        try:
            prepared = _prepare_add_job(net, event)
        except (TypeError, ValueError):
            continue
        read_items = prepared.pop("read_items")
        bindings.append(_new_event_binding(T_ADD_JOB, event, prepared, read_items))
    return bindings


def _apply_add_job(net: FJSPCPNV4, binding: Binding, now: float) -> None:
    job: JobToken = binding.payload["job"]
    operations: List[OperationToken] = binding.payload["operations"]
    eligibility: List[EligibilityToken] = binding.payload["eligibility"]
    precedence: List[PrecedenceToken] = binding.payload["precedence"]
    event: EventToken = binding.payload["event"]
    net.places[P_J_ACTIVE].add(job)
    for operation in operations:
        net.places[P_O_WAITING].add(operation)
    for relation in eligibility:
        net.places[P_E_ACTIVE].add(relation)
    for relation in precedence:
        net.places[P_R_ACTIVE].add(relation)
    net.total_jobs += 1
    net.total_operations += len(operations)
    net.initial_eligibility_count += len(eligibility)
    net.initial_precedence_count += len(precedence)
    _apply_event(net, event, now, f"added job {job.job_id}")


def _prepare_add_machine(net: FJSPCPNV4, event: EventToken) -> Dict[str, Any]:
    machine_id = int(event.target_id)
    if _find_token_with_place(net, MACHINE_PLACES, "machine_id", machine_id) is not None:
        raise ValueError(f"machine_id already exists: {machine_id}")
    machine = MachineToken(machine_id=machine_id, available_at=float(event.event_time))
    payload = event.payload or {}
    raw_edges = payload.get("eligibility", [])
    eligibility = []
    read_items = []
    existing_relation_ids = _relation_ids(net)
    for raw_edge in raw_edges:
        op_uid = _parse_operation_uid(raw_edge.get("op_uid"))
        operation_item = _find_token_with_place(net, OPERATION_PLACES, "op_uid", op_uid)
        if operation_item is None:
            raise ValueError(f"Unknown operation {op_uid}")
        processing_time = float(raw_edge.get("processing_time"))
        if processing_time <= 0:
            raise ValueError("Processing time must be positive")
        relation_id = str(
            raw_edge.get("relation_id", f"E_O{op_uid[0]}_{op_uid[1]}_M{machine_id}")
        )
        if relation_id in existing_relation_ids:
            raise ValueError(f"Relation already exists: {relation_id}")
        existing_relation_ids.add(relation_id)
        eligibility.append(EligibilityToken(
            relation_id=relation_id,
            op_uid=op_uid,
            machine_id=machine_id,
            processing_time=processing_time,
        ))
        read_items.append(operation_item)
    return {"machine": machine, "eligibility": eligibility, "read_items": read_items}


def _bind_add_machine(net: FJSPCPNV4, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_MACHINE_ADD, now):
        try:
            prepared = _prepare_add_machine(net, event)
        except (TypeError, ValueError):
            continue
        read_items = prepared.pop("read_items")
        bindings.append(_new_event_binding(T_ADD_MACHINE, event, prepared, read_items))
    return bindings


def _apply_add_machine(net: FJSPCPNV4, binding: Binding, now: float) -> None:
    machine: MachineToken = binding.payload["machine"]
    eligibility: List[EligibilityToken] = binding.payload["eligibility"]
    event: EventToken = binding.payload["event"]
    net.places[P_M_AVAILABLE].add(machine)
    for relation in eligibility:
        net.places[P_E_ACTIVE].add(relation)
    net.total_machines += 1
    net.initial_eligibility_count += len(eligibility)
    _apply_event(net, event, now, f"added machine {machine.machine_id}")


def _prepare_add_eligibility(net: FJSPCPNV4, event: EventToken) -> Dict[str, Any]:
    payload = event.payload or {}
    op_uid = _parse_operation_uid(payload.get("op_uid"))
    machine_id = int(payload.get("machine_id"))
    processing_time = float(payload.get("processing_time"))
    operation_item = _find_token_with_place(net, OPERATION_PLACES, "op_uid", op_uid)
    machine_item = _find_token_with_place(net, MACHINE_PLACES, "machine_id", machine_id)
    if operation_item is None or machine_item is None:
        raise ValueError("Eligibility endpoint does not exist")
    if processing_time <= 0:
        raise ValueError("Processing time must be positive")
    relation_id = str(event.target_id)
    if relation_id in _relation_ids(net):
        raise ValueError(f"Relation already exists: {relation_id}")
    relation = EligibilityToken(relation_id, op_uid, machine_id, processing_time)
    return {"relation": relation, "read_items": [operation_item, machine_item]}


def _bind_add_eligibility(net: FJSPCPNV4, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_ELIGIBILITY_ADD, now):
        try:
            prepared = _prepare_add_eligibility(net, event)
        except (TypeError, ValueError):
            continue
        read_items = prepared.pop("read_items")
        bindings.append(_new_event_binding(
            T_ADD_ELIGIBILITY, event, prepared, read_items
        ))
    return bindings


def _apply_add_eligibility(net: FJSPCPNV4, binding: Binding, now: float) -> None:
    relation: EligibilityToken = binding.payload["relation"]
    event: EventToken = binding.payload["event"]
    net.places[P_E_ACTIVE].add(relation)
    net.initial_eligibility_count += 1
    _apply_event(net, event, now, f"added relation {relation.relation_id}")


def _prepare_add_precedence(net: FJSPCPNV4, event: EventToken) -> Dict[str, Any]:
    payload = event.payload or {}
    predecessor_uid = _parse_operation_uid(payload.get("predecessor_uid"))
    successor_uid = _parse_operation_uid(payload.get("successor_uid"))
    predecessor_item = _find_token_with_place(
        net, OPERATION_PLACES, "op_uid", predecessor_uid
    )
    successor_item = _find_token_with_place(
        net, OPERATION_PLACES, "op_uid", successor_uid
    )
    if predecessor_item is None or successor_item is None:
        raise ValueError("Precedence endpoint does not exist")
    if successor_item[0] != P_O_WAITING:
        raise ValueError("Precedence successor must still be waiting")
    relation_id = str(event.target_id)
    if relation_id in _relation_ids(net):
        raise ValueError(f"Relation already exists: {relation_id}")
    relation = PrecedenceToken(relation_id, predecessor_uid, successor_uid)
    edges = [
        (current.predecessor_uid, current.successor_uid)
        for current in net.places[P_R_ACTIVE].objects
    ]
    edges.append((predecessor_uid, successor_uid))
    if not _is_acyclic(edges):
        raise ValueError("New precedence relation creates a cycle")
    return {
        "relation": relation,
        "read_items": [predecessor_item, successor_item],
    }


def _bind_add_precedence(net: FJSPCPNV4, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_PRECEDENCE_ADD, now):
        try:
            prepared = _prepare_add_precedence(net, event)
        except (TypeError, ValueError):
            continue
        read_items = prepared.pop("read_items")
        bindings.append(_new_event_binding(
            T_ADD_PRECEDENCE, event, prepared, read_items
        ))
    return bindings


def _apply_add_precedence(net: FJSPCPNV4, binding: Binding, now: float) -> None:
    relation: PrecedenceToken = binding.payload["relation"]
    event: EventToken = binding.payload["event"]
    net.places[P_R_ACTIVE].add(relation)
    net.initial_precedence_count += 1
    _apply_event(net, event, now, f"added relation {relation.relation_id}")
