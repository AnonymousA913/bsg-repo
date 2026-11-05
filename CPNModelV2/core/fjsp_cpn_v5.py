from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from CPNModelV2.core.fjsp_cpn_v1 import FJSPInstance
from CPNModelV2.core.fjsp_cpn_v2 import (
    EPS,
    Binding,
    EligibilityToken,
    MachineToken,
    OperationToken,
    P_E_ACTIVE,
    P_M_AVAILABLE,
    P_M_BUSY,
    P_O_COMPLETED,
    P_O_PROCESSING,
    P_O_READY,
    P_O_WAITING,
    P_R_ACTIVE,
    Place,
    PrecedenceToken,
    T_ENABLE,
    T_FINISH,
    T_START,
    Transition,
    _apply_enable,
    _apply_finish,
    _apply_start,
    _bind_finish,
    _bind_start,
)
from CPNModelV2.core.fjsp_cpn_v3 import (
    EVENT_ELIGIBILITY_ACTIVATE,
    EVENT_ELIGIBILITY_ADD,
    EVENT_ELIGIBILITY_DEACTIVATE,
    EVENT_JOB_ADD,
    EVENT_JOB_RELEASE,
    EVENT_MACHINE_ADD,
    EVENT_MACHINE_FAILURE,
    EVENT_MACHINE_RECOVERY,
    EVENT_PRECEDENCE_ACTIVATE,
    EVENT_PRECEDENCE_ADD,
    EVENT_PRECEDENCE_DEACTIVATE,
    EventToken,
    P_E_INACTIVE,
    P_EVT_APPLIED,
    P_EVT_PENDING,
    P_M_OFFLINE,
    P_O_INTERRUPTED,
    P_R_INACTIVE,
    T_ACTIVATE_ELIGIBILITY,
    T_ACTIVATE_PRECEDENCE,
    T_DEACTIVATE_ELIGIBILITY,
    T_DEACTIVATE_PRECEDENCE,
    T_FAIL_BUSY_MACHINE,
    T_FAIL_IDLE_MACHINE,
    T_IGNORE_EVENT,
    T_RECOVER_MACHINE,
    T_RELEASE_JOB,
    T_REQUEUE_OPERATION,
    _apply_activate_eligibility,
    _apply_activate_precedence,
    _apply_deactivate_eligibility,
    _apply_deactivate_precedence,
    _apply_event,
    _apply_fail_busy_machine,
    _apply_fail_idle_machine,
    _apply_ignore_event,
    _apply_recover_machine,
    _apply_requeue_operation,
    _bind_activate_eligibility,
    _bind_activate_precedence,
    _bind_deactivate_eligibility,
    _bind_deactivate_precedence,
    _bind_fail_busy_machine,
    _bind_fail_idle_machine,
    _bind_recover_machine,
    _due_events,
    _event_binding,
    _find_by_attribute,
    _find_operation,
    _would_create_cycle,
)
from CPNModelV2.core.fjsp_cpn_v4 import (
    FJSPCPNV4,
    T_ADD_ELIGIBILITY,
    T_ADD_JOB,
    T_ADD_MACHINE,
    T_ADD_PRECEDENCE,
    _apply_add_eligibility,
    _apply_add_machine,
    _apply_add_precedence,
    _bind_add_eligibility,
    _bind_add_machine,
    _bind_add_precedence,
    _find_token_with_place,
    _is_acyclic,
    _new_event_binding,
    _relation_ids,
    _tokens_with_places,
)


OPERATION_PLACES = (
    P_O_WAITING,
    P_O_READY,
    P_O_PROCESSING,
    P_O_INTERRUPTED,
    P_O_COMPLETED,
)
MACHINE_PLACES = (P_M_AVAILABLE, P_M_BUSY, P_M_OFFLINE)
RELATION_PLACES = (P_E_ACTIVE, P_E_INACTIVE, P_R_ACTIVE, P_R_INACTIVE)
EVENT_PLACES = (P_EVT_PENDING, P_EVT_APPLIED)
POLICY_TOKEN_KINDS = frozenset(
    {"operation", "machine", "eligibility", "precedence"}
)


class FJSPCPNV5(FJSPCPNV4):
    """Job-token-free executable CPN for dynamic FJSP decisions.

    Job metadata is carried by operation tokens. Event tokens remain part of the
    execution semantics but are excluded from the policy-facing marking.
    """

    def __init__(
        self,
        instance: FJSPInstance,
        *,
        job_release_times: Optional[Mapping[int, float]] = None,
        events: Sequence[EventToken] = (),
    ) -> None:
        super().__init__(
            instance,
            job_release_times=job_release_times,
            events=events,
        )

    def _build_places(self) -> None:
        for name in OPERATION_PLACES:
            self.places[name] = Place(name, "operation")
        for name in MACHINE_PLACES:
            self.places[name] = Place(name, "machine")
        self.places[P_E_ACTIVE] = Place(P_E_ACTIVE, "eligibility")
        self.places[P_E_INACTIVE] = Place(P_E_INACTIVE, "eligibility")
        self.places[P_R_ACTIVE] = Place(P_R_ACTIVE, "precedence")
        self.places[P_R_INACTIVE] = Place(P_R_INACTIVE, "precedence")
        self.places[P_EVT_PENDING] = Place(P_EVT_PENDING, "event")
        self.places[P_EVT_APPLIED] = Place(P_EVT_APPLIED, "event")

    def _build_transitions(self) -> None:
        self.transitions[T_ENABLE] = Transition(
            T_ENABLE,
            controlled=False,
            bind_fn=_bind_enable_v5,
            apply_fn=_apply_enable,
            priority=20,
        )
        self.transitions[T_START] = Transition(
            T_START,
            controlled=True,
            bind_fn=_bind_start,
            apply_fn=_apply_start,
        )
        self.transitions[T_FINISH] = Transition(
            T_FINISH,
            controlled=False,
            bind_fn=_bind_finish,
            apply_fn=_apply_finish,
            priority=30,
        )

        event_priority = 25
        transition_specs = (
            (T_RELEASE_JOB, _bind_release_operations, _apply_release_operations, event_priority),
            (T_FAIL_IDLE_MACHINE, _bind_fail_idle_machine, _apply_fail_idle_machine, event_priority),
            (T_FAIL_BUSY_MACHINE, _bind_fail_busy_machine, _apply_fail_busy_machine, event_priority),
            (T_RECOVER_MACHINE, _bind_recover_machine, _apply_recover_machine, event_priority),
            (T_REQUEUE_OPERATION, _bind_requeue_operation_v5, _apply_requeue_operation, 24),
            (T_DEACTIVATE_ELIGIBILITY, _bind_deactivate_eligibility, _apply_deactivate_eligibility, event_priority),
            (T_ACTIVATE_ELIGIBILITY, _bind_activate_eligibility, _apply_activate_eligibility, event_priority),
            (T_DEACTIVATE_PRECEDENCE, _bind_deactivate_precedence, _apply_deactivate_precedence, event_priority),
            (T_ACTIVATE_PRECEDENCE, _bind_activate_precedence, _apply_activate_precedence, event_priority),
            (T_ADD_JOB, _bind_add_job_v5, _apply_add_job_v5, event_priority),
            (T_ADD_MACHINE, _bind_add_machine, _apply_add_machine, event_priority),
            (T_ADD_ELIGIBILITY, _bind_add_eligibility, _apply_add_eligibility, event_priority),
            (T_ADD_PRECEDENCE, _bind_add_precedence, _apply_add_precedence, event_priority),
            (T_IGNORE_EVENT, _bind_ignore_event_v5, _apply_ignore_event, 1),
        )
        for name, bind_fn, apply_fn, priority in transition_specs:
            self.transitions[name] = Transition(
                name,
                controlled=False,
                bind_fn=bind_fn,
                apply_fn=apply_fn,
                priority=priority,
            )

    def _build_arcs(self) -> None:
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

        p2t(P_O_WAITING, T_ENABLE, "consume_operation")
        p2t(P_R_ACTIVE, T_ENABLE, "read_precedence")
        p2t(P_O_COMPLETED, T_ENABLE, "read_completed_predecessors")
        t2p(T_ENABLE, P_O_READY, "produce_ready")

        p2t(P_O_READY, T_START, "consume_operation")
        p2t(P_M_AVAILABLE, T_START, "consume_machine")
        p2t(P_E_ACTIVE, T_START, "read_eligibility")
        t2p(T_START, P_O_PROCESSING, "produce_operation")
        t2p(T_START, P_M_BUSY, "produce_machine")

        p2t(P_O_PROCESSING, T_FINISH, "consume_operation")
        p2t(P_M_BUSY, T_FINISH, "consume_machine")
        t2p(T_FINISH, P_O_COMPLETED, "produce_completed")
        t2p(T_FINISH, P_M_AVAILABLE, "release_machine")

        event_transitions = (
            T_RELEASE_JOB,
            T_FAIL_IDLE_MACHINE,
            T_FAIL_BUSY_MACHINE,
            T_RECOVER_MACHINE,
            T_DEACTIVATE_ELIGIBILITY,
            T_ACTIVATE_ELIGIBILITY,
            T_DEACTIVATE_PRECEDENCE,
            T_ACTIVATE_PRECEDENCE,
            T_ADD_JOB,
            T_ADD_MACHINE,
            T_ADD_ELIGIBILITY,
            T_ADD_PRECEDENCE,
            T_IGNORE_EVENT,
        )
        for transition in event_transitions:
            p2t(P_EVT_PENDING, transition, "consume_due_event")
            t2p(transition, P_EVT_APPLIED, "produce_applied_event")

        p2t(P_O_WAITING, T_RELEASE_JOB, "read_released_operations")

        p2t(P_M_AVAILABLE, T_FAIL_IDLE_MACHINE, "consume_machine")
        t2p(T_FAIL_IDLE_MACHINE, P_M_OFFLINE, "produce_offline_machine")
        p2t(P_M_BUSY, T_FAIL_BUSY_MACHINE, "consume_machine")
        p2t(P_O_PROCESSING, T_FAIL_BUSY_MACHINE, "consume_operation")
        t2p(T_FAIL_BUSY_MACHINE, P_M_OFFLINE, "produce_offline_machine")
        t2p(T_FAIL_BUSY_MACHINE, P_O_INTERRUPTED, "produce_interrupted_operation")

        p2t(P_O_INTERRUPTED, T_REQUEUE_OPERATION, "consume_operation")
        p2t(P_R_ACTIVE, T_REQUEUE_OPERATION, "read_precedence")
        p2t(P_O_COMPLETED, T_REQUEUE_OPERATION, "read_completed_predecessors")
        t2p(T_REQUEUE_OPERATION, P_O_READY, "produce_ready_operation")

        p2t(P_M_OFFLINE, T_RECOVER_MACHINE, "consume_machine")
        t2p(T_RECOVER_MACHINE, P_M_AVAILABLE, "produce_available_machine")

        p2t(P_E_ACTIVE, T_DEACTIVATE_ELIGIBILITY, "consume_relation")
        t2p(T_DEACTIVATE_ELIGIBILITY, P_E_INACTIVE, "produce_relation")
        p2t(P_E_INACTIVE, T_ACTIVATE_ELIGIBILITY, "consume_relation")
        t2p(T_ACTIVATE_ELIGIBILITY, P_E_ACTIVE, "produce_relation")
        p2t(P_R_ACTIVE, T_DEACTIVATE_PRECEDENCE, "consume_relation")
        t2p(T_DEACTIVATE_PRECEDENCE, P_R_INACTIVE, "produce_relation")
        p2t(P_R_INACTIVE, T_ACTIVATE_PRECEDENCE, "consume_relation")
        t2p(T_ACTIVATE_PRECEDENCE, P_R_ACTIVE, "produce_relation")

        for place in MACHINE_PLACES:
            p2t(place, T_ADD_JOB, "read_referenced_machines")
        t2p(T_ADD_JOB, P_O_WAITING, "produce_operations")
        t2p(T_ADD_JOB, P_E_ACTIVE, "produce_eligibility")
        t2p(T_ADD_JOB, P_R_ACTIVE, "produce_precedence")

        for place in OPERATION_PLACES:
            p2t(place, T_ADD_MACHINE, "read_referenced_operations")
            p2t(place, T_ADD_ELIGIBILITY, "read_operation")
            p2t(place, T_ADD_PRECEDENCE, "read_operations")
        for place in MACHINE_PLACES:
            p2t(place, T_ADD_ELIGIBILITY, "read_machine")
        t2p(T_ADD_MACHINE, P_M_AVAILABLE, "produce_machine")
        t2p(T_ADD_MACHINE, P_E_ACTIVE, "produce_eligibility")
        t2p(T_ADD_ELIGIBILITY, P_E_ACTIVE, "produce_eligibility")
        t2p(T_ADD_PRECEDENCE, P_R_ACTIVE, "produce_precedence")

    def _populate_marking(self) -> None:
        generated_events: List[EventToken] = []
        for job_id, job in enumerate(self.instance.jobs, start=1):
            release_time = float(self._job_release_times.get(job_id, 0.0))
            if release_time < 0:
                raise ValueError("Job release time must be non-negative")
            if release_time > EPS:
                generated_events.append(EventToken(
                    event_id=f"release_job_{job_id}",
                    event_type=EVENT_JOB_RELEASE,
                    event_time=release_time,
                    target_id=job_id,
                ))
            for zero_index, options in enumerate(job):
                op_index = zero_index + 1
                op_uid = (job_id, op_index)
                self.places[P_O_WAITING].add(OperationToken(
                    op_uid=op_uid,
                    job_id=job_id,
                    op_index=op_index,
                    num_operations=len(job),
                    release_time=release_time,
                ))
                for machine_id, processing_time in options.items():
                    self.places[P_E_ACTIVE].add(EligibilityToken(
                        relation_id=f"E_O{job_id}_{op_index}_M{machine_id}",
                        op_uid=op_uid,
                        machine_id=int(machine_id),
                        processing_time=float(processing_time),
                    ))
                if op_index > 1:
                    self.places[P_R_ACTIVE].add(PrecedenceToken(
                        relation_id=(
                            f"R_O{job_id}_{op_index - 1}_O{job_id}_{op_index}"
                        ),
                        predecessor_uid=(job_id, op_index - 1),
                        successor_uid=op_uid,
                    ))
        for machine_id in self.instance.machine_ids:
            self.places[P_M_AVAILABLE].add(MachineToken(machine_id=int(machine_id)))

        events = generated_events + self._initial_events
        event_ids = [event.event_id for event in events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Duplicate event_id in initial event schedule")
        for event in events:
            self.places[P_EVT_PENDING].add(event)
        self.initial_event_count = len(events)

    def policy_marking(self) -> Dict[str, Tuple[Any, ...]]:
        """Return the four semantic token families visible to the policy."""
        return {
            name: tuple(place.objects)
            for name, place in self.places.items()
            if place.token_kind in POLICY_TOKEN_KINDS
        }

    def event_marking(self) -> Dict[str, Tuple[EventToken, ...]]:
        """Return environment-only event tokens for debugging and replay."""
        return {
            name: tuple(self.places[name].objects)
            for name in EVENT_PLACES
        }

    def is_terminal(self) -> bool:
        future_job_addition = any(
            event.event_type == EVENT_JOB_ADD
            for event in self.places[P_EVT_PENDING].objects
        )
        return (
            not future_job_addition
            and self.places[P_O_COMPLETED].tokens == self.total_operations
        )

    def check_invariants(self) -> Dict[str, int]:
        operations = [
            token for name in OPERATION_PLACES for token in self.places[name].objects
        ]
        machines = [
            token for name in MACHINE_PLACES for token in self.places[name].objects
        ]
        eligibility = (
            list(self.places[P_E_ACTIVE].objects)
            + list(self.places[P_E_INACTIVE].objects)
        )
        precedence = (
            list(self.places[P_R_ACTIVE].objects)
            + list(self.places[P_R_INACTIVE].objects)
        )
        events = (
            list(self.places[P_EVT_PENDING].objects)
            + list(self.places[P_EVT_APPLIED].objects)
        )
        errors: List[str] = []

        if len(operations) != self.total_operations:
            errors.append("Operation conservation violated")
        if len({token.op_uid for token in operations}) != len(operations):
            errors.append("Duplicate operation token")
        if len(machines) != self.total_machines:
            errors.append("Machine conservation violated")
        if len({token.machine_id for token in machines}) != len(machines):
            errors.append("Duplicate machine token")
        if len(eligibility) != self.initial_eligibility_count:
            errors.append("Eligibility conservation violated")
        if len(precedence) != self.initial_precedence_count:
            errors.append("Precedence conservation violated")
        if len(events) != self.initial_event_count:
            errors.append("Event conservation violated")
        if len({token.event_id for token in events}) != len(events):
            errors.append("Duplicate event token")

        operation_ids = {token.op_uid for token in operations}
        machine_ids = {token.machine_id for token in machines}
        jobs: Dict[int, List[OperationToken]] = {}
        for operation in operations:
            jobs.setdefault(operation.job_id, []).append(operation)
        if len(jobs) != self.total_jobs:
            errors.append("Operation-derived job count is inconsistent")
        for job_id, job_operations in jobs.items():
            expected = {operation.num_operations for operation in job_operations}
            indices = {operation.op_index for operation in job_operations}
            if len(expected) != 1 or len(job_operations) != next(iter(expected)):
                errors.append(f"Job {job_id} has inconsistent operation count")
            elif indices != set(range(1, next(iter(expected)) + 1)):
                errors.append(f"Job {job_id} has non-contiguous operation indices")
            metadata = {
                (operation.release_time, operation.due_date, operation.priority)
                for operation in job_operations
            }
            if len(metadata) != 1:
                errors.append(f"Job {job_id} metadata differs across operations")

        for relation in eligibility:
            if relation.op_uid not in operation_ids:
                errors.append(f"Eligibility {relation.relation_id} has unknown operation")
            if relation.machine_id not in machine_ids:
                errors.append(f"Eligibility {relation.relation_id} has unknown machine")
        for relation in precedence:
            if relation.predecessor_uid not in operation_ids:
                errors.append(f"Precedence {relation.relation_id} has unknown predecessor")
            if relation.successor_uid not in operation_ids:
                errors.append(f"Precedence {relation.relation_id} has unknown successor")

        completed = {
            token.op_uid: token for token in self.places[P_O_COMPLETED].objects
        }
        incoming: Dict[Tuple[int, int], List[PrecedenceToken]] = {}
        for relation in self.places[P_R_ACTIVE].objects:
            incoming.setdefault(relation.successor_uid, []).append(relation)
        for operation in (
            self.places[P_O_READY].objects + self.places[P_O_PROCESSING].objects
        ):
            if operation.release_time > self.now + EPS:
                errors.append(f"Operation {operation.op_uid} started before release")
            if any(
                relation.predecessor_uid not in completed
                for relation in incoming.get(operation.op_uid, [])
            ):
                errors.append(f"Precedence violation for {operation.op_uid}")

        processing = self.places[P_O_PROCESSING].objects
        busy = self.places[P_M_BUSY].objects
        for operation in processing:
            matching = [
                machine for machine in busy
                if machine.machine_id == operation.assigned_machine_id
                and machine.bound_op_uid == operation.op_uid
            ]
            if len(matching) != 1:
                errors.append(f"Processing operation {operation.op_uid} lacks machine")
        for machine in busy:
            matching = [
                operation for operation in processing
                if operation.op_uid == machine.bound_op_uid
                and operation.assigned_machine_id == machine.machine_id
            ]
            if len(matching) != 1:
                errors.append(f"Busy machine {machine.machine_id} lacks operation")
        for operation in self.places[P_O_INTERRUPTED].objects:
            if operation.assigned_machine_id is not None:
                errors.append(f"Interrupted operation {operation.op_uid} retains assignment")
        for machine in self.places[P_M_OFFLINE].objects:
            if machine.bound_op_uid is not None:
                errors.append(f"Offline machine {machine.machine_id} retains operation")

        for place_name in OPERATION_PLACES + MACHINE_PLACES + RELATION_PLACES + EVENT_PLACES:
            if any(
                token.current_place != place_name
                for token in self.places[place_name].objects
            ):
                errors.append(f"Token current_place mismatch in {place_name}")

        if errors:
            raise AssertionError("\n".join(errors))
        return {
            "operations": len(operations),
            "machines": len(machines),
            "jobs": len(jobs),
            "eligibility": len(eligibility),
            "precedence": len(precedence),
            "events": len(events),
        }


def _bind_enable_v5(net: FJSPCPNV5, now: float) -> List[Binding]:
    completed = {
        token.op_uid: token for token in net.places[P_O_COMPLETED].objects
    }
    incoming: Dict[Tuple[int, int], List[PrecedenceToken]] = {}
    for relation in net.places[P_R_ACTIVE].objects:
        incoming.setdefault(relation.successor_uid, []).append(relation)

    bindings = []
    for operation in net.places[P_O_WAITING].objects:
        if operation.release_time > now + EPS:
            continue
        relations = incoming.get(operation.op_uid, [])
        if any(relation.predecessor_uid not in completed for relation in relations):
            continue
        predecessors = [completed[relation.predecessor_uid] for relation in relations]
        bindings.append(Binding(
            transition_name=T_ENABLE,
            key=(T_ENABLE, operation.op_uid),
            consume_objects={P_O_WAITING: [operation]},
            read_objects={
                P_R_ACTIVE: list(relations),
                P_O_COMPLETED: predecessors,
            },
            payload={
                "operation": operation,
                "precedence": relations,
                "predecessors": predecessors,
            },
        ))
    return bindings


def _bind_release_operations(net: FJSPCPNV5, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_JOB_RELEASE, now):
        operations = [
            operation for operation in net.places[P_O_WAITING].objects
            if operation.job_id == int(event.target_id)
        ]
        if not operations:
            continue
        bindings.append(_event_binding(
            T_RELEASE_JOB,
            event,
            payload={"operations": operations},
        ))
        bindings[-1].read_objects = {P_O_WAITING: operations}
    return bindings


def _apply_release_operations(
    net: FJSPCPNV5, binding: Binding, now: float
) -> None:
    event: EventToken = binding.payload["event"]
    operations: List[OperationToken] = binding.payload["operations"]
    _apply_event(
        net,
        event,
        now,
        f"released operation group for job {operations[0].job_id}",
    )


def _bind_requeue_operation_v5(net: FJSPCPNV5, now: float) -> List[Binding]:
    del now
    completed = {
        token.op_uid: token for token in net.places[P_O_COMPLETED].objects
    }
    bindings = []
    for operation in net.places[P_O_INTERRUPTED].objects:
        relations = [
            relation for relation in net.places[P_R_ACTIVE].objects
            if relation.successor_uid == operation.op_uid
        ]
        if any(relation.predecessor_uid not in completed for relation in relations):
            continue
        predecessors = [completed[relation.predecessor_uid] for relation in relations]
        bindings.append(Binding(
            transition_name=T_REQUEUE_OPERATION,
            key=(T_REQUEUE_OPERATION, operation.op_uid),
            consume_objects={P_O_INTERRUPTED: [operation]},
            read_objects={
                P_R_ACTIVE: relations,
                P_O_COMPLETED: predecessors,
            },
            payload={"operation": operation},
        ))
    return bindings


def _prepare_add_job_v5(net: FJSPCPNV5, event: EventToken) -> Dict[str, Any]:
    job_id = int(event.target_id)
    existing_jobs = {
        operation.job_id
        for _, operation in _tokens_with_places(net, OPERATION_PLACES)
    }
    if job_id in existing_jobs:
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
    due_date = None if payload.get("due_date") is None else float(payload["due_date"])
    priority = float(payload.get("priority", 1.0))
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
            release_time=float(event.event_time),
            due_date=due_date,
            priority=priority,
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
        raw_precedence = [[index, index + 1] for index in range(1, len(operations))]
    precedence: List[PrecedenceToken] = []
    edges = []
    seen_edges = set()
    for raw_edge in raw_precedence:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            raise ValueError("Each precedence edge must contain two indices")
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
        relation_id = f"R_O{job_id}_{predecessor_index}_O{job_id}_{successor_index}"
        if relation_id in existing_relation_ids:
            raise ValueError(f"Relation already exists: {relation_id}")
        existing_relation_ids.add(relation_id)
        precedence.append(PrecedenceToken(relation_id, edge[0], edge[1]))
    if not _is_acyclic(edges):
        raise ValueError("New job precedence graph contains a cycle")
    return {
        "operations": operations,
        "eligibility": eligibility,
        "precedence": precedence,
        "read_items": list(read_machines.values()),
    }


def _bind_add_job_v5(net: FJSPCPNV5, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_JOB_ADD, now):
        try:
            prepared = _prepare_add_job_v5(net, event)
        except (TypeError, ValueError):
            continue
        read_items = prepared.pop("read_items")
        bindings.append(_new_event_binding(T_ADD_JOB, event, prepared, read_items))
    return bindings


def _apply_add_job_v5(net: FJSPCPNV5, binding: Binding, now: float) -> None:
    operations: List[OperationToken] = binding.payload["operations"]
    eligibility: List[EligibilityToken] = binding.payload["eligibility"]
    precedence: List[PrecedenceToken] = binding.payload["precedence"]
    event: EventToken = binding.payload["event"]
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
    _apply_event(net, event, now, f"added operation group for job {operations[0].job_id}")


def _event_is_applicable_v5(net: FJSPCPNV5, event: EventToken) -> bool:
    target = event.target_id
    if event.event_type == EVENT_JOB_RELEASE:
        return any(
            operation.job_id == int(target)
            for operation in net.places[P_O_WAITING].objects
        )
    if event.event_type == EVENT_MACHINE_FAILURE:
        return any(
            _find_by_attribute(net.places[place].objects, "machine_id", int(target))
            is not None
            for place in (P_M_AVAILABLE, P_M_BUSY)
        )
    if event.event_type == EVENT_MACHINE_RECOVERY:
        return (
            _find_by_attribute(net.places[P_M_OFFLINE].objects, "machine_id", int(target))
            is not None
        )
    relation_specs = {
        EVENT_ELIGIBILITY_DEACTIVATE: (P_E_ACTIVE, False),
        EVENT_ELIGIBILITY_ACTIVATE: (P_E_INACTIVE, False),
        EVENT_PRECEDENCE_DEACTIVATE: (P_R_ACTIVE, True),
        EVENT_PRECEDENCE_ACTIVATE: (P_R_INACTIVE, True),
    }
    if event.event_type in relation_specs:
        source_place, is_precedence = relation_specs[event.event_type]
        relation = _find_by_attribute(
            net.places[source_place].objects, "relation_id", str(target)
        )
        if relation is None:
            return False
        if is_precedence:
            successor = _find_operation(net, relation.successor_uid)
            if successor is None or successor.current_place != P_O_WAITING:
                return False
            if event.event_type == EVENT_PRECEDENCE_ACTIVATE:
                return not _would_create_cycle(net, relation)
        return True
    if event.event_type in {
        EVENT_JOB_ADD,
        EVENT_MACHINE_ADD,
        EVENT_ELIGIBILITY_ADD,
        EVENT_PRECEDENCE_ADD,
    }:
        return False
    return False


def _bind_ignore_event_v5(net: FJSPCPNV5, now: float) -> List[Binding]:
    bindings = []
    for event in net.places[P_EVT_PENDING].objects:
        if event.event_time <= now + EPS and not _event_is_applicable_v5(net, event):
            bindings.append(_event_binding(T_IGNORE_EVENT, event))
    return bindings
