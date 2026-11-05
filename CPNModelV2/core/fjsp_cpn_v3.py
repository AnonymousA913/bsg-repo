from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from CPNModelV2.core.fjsp_cpn_v1 import FJSPInstance
from CPNModelV2.core.fjsp_cpn_v2 import (
    EPS,
    Binding,
    EligibilityToken,
    FJSPCPNV2,
    JobToken,
    MachineToken,
    OperationToken,
    P_E_ACTIVE,
    P_J_ACTIVE,
    P_J_COMPLETED,
    P_M_AVAILABLE,
    P_M_BUSY,
    P_O_COMPLETED,
    P_O_PROCESSING,
    P_O_READY,
    P_O_WAITING,
    P_R_ACTIVE,
    Place,
    PrecedenceToken,
    Transition,
)

# V3版本：加入异常事件的语义
# 1. 加入EventToken，即事件Token，使得FJSP中加入了故障、恢复、关系激活/失活等异常事件类型，不过这个还不是所有的事件，V4版本将加入Song等方法难以解决的异常事件类型

P_O_INTERRUPTED = "P_O_Interrupted"
P_M_OFFLINE = "P_M_Offline"
P_J_UNRELEASED = "P_J_Unreleased"
P_E_INACTIVE = "P_E_Inactive"
P_R_INACTIVE = "P_R_Inactive"
P_EVT_PENDING = "P_EVT_Pending"
P_EVT_APPLIED = "P_EVT_Applied"

T_RELEASE_JOB = "T_ReleaseJob"
T_FAIL_IDLE_MACHINE = "T_FailIdleMachine"
T_FAIL_BUSY_MACHINE = "T_FailBusyMachine"
T_RECOVER_MACHINE = "T_RecoverMachine"
T_REQUEUE_OPERATION = "T_RequeueOperation"
T_DEACTIVATE_ELIGIBILITY = "T_DeactivateEligibility"
T_ACTIVATE_ELIGIBILITY = "T_ActivateEligibility"
T_DEACTIVATE_PRECEDENCE = "T_DeactivatePrecedence"
T_ACTIVATE_PRECEDENCE = "T_ActivatePrecedence"
T_IGNORE_EVENT = "T_IgnoreInapplicableEvent"

EVENT_JOB_RELEASE = "JOB_RELEASE"
EVENT_MACHINE_FAILURE = "MACHINE_FAILURE"
EVENT_MACHINE_RECOVERY = "MACHINE_RECOVERY"
EVENT_ELIGIBILITY_DEACTIVATE = "ELIGIBILITY_DEACTIVATE"
EVENT_ELIGIBILITY_ACTIVATE = "ELIGIBILITY_ACTIVATE"
EVENT_PRECEDENCE_DEACTIVATE = "PRECEDENCE_DEACTIVATE"
EVENT_PRECEDENCE_ACTIVATE = "PRECEDENCE_ACTIVATE"
EVENT_JOB_ADD = "JOB_ADD"
EVENT_MACHINE_ADD = "MACHINE_ADD"
EVENT_ELIGIBILITY_ADD = "ELIGIBILITY_ADD"
EVENT_PRECEDENCE_ADD = "PRECEDENCE_ADD"

EVENT_TYPES = {
    EVENT_JOB_RELEASE,
    EVENT_MACHINE_FAILURE,
    EVENT_MACHINE_RECOVERY,
    EVENT_ELIGIBILITY_DEACTIVATE,
    EVENT_ELIGIBILITY_ACTIVATE,
    EVENT_PRECEDENCE_DEACTIVATE,
    EVENT_PRECEDENCE_ACTIVATE,
    EVENT_JOB_ADD,
    EVENT_MACHINE_ADD,
    EVENT_ELIGIBILITY_ADD,
    EVENT_PRECEDENCE_ADD,
}


@dataclass
class EventToken:
    event_id: str
    event_type: str
    event_time: float
    target_id: Any
    payload: Optional[Dict[str, Any]] = None
    reveal_time: float = 0.0
    applied_time: Optional[float] = None
    outcome: Optional[str] = None
    current_place: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event_type={self.event_type}")
        if self.event_time < 0:
            raise ValueError("event_time must be non-negative")
        if self.reveal_time > self.event_time:
            raise ValueError("reveal_time cannot be later than event_time")


class FJSPCPNV3(FJSPCPNV2):
    """Dynamic FJSP CPN with event-driven failures and relation changes."""

    def __init__(
        self,
        instance: FJSPInstance,
        *,
        job_release_times: Optional[Mapping[int, float]] = None,
        events: Sequence[EventToken] = (),
    ) -> None:
        self._job_release_times = {
            int(job_id): float(release_time)
            for job_id, release_time in (job_release_times or {}).items()
        }
        self._initial_events = list(events)
        self.initial_event_count = 0
        super().__init__(instance)

    def _build_places(self) -> None:
        super()._build_places()
        self.places[P_O_INTERRUPTED] = Place(P_O_INTERRUPTED, "operation")
        self.places[P_M_OFFLINE] = Place(P_M_OFFLINE, "machine")
        self.places[P_J_UNRELEASED] = Place(P_J_UNRELEASED, "job")
        self.places[P_E_INACTIVE] = Place(P_E_INACTIVE, "eligibility")
        self.places[P_R_INACTIVE] = Place(P_R_INACTIVE, "precedence")
        self.places[P_EVT_PENDING] = Place(P_EVT_PENDING, "event")
        self.places[P_EVT_APPLIED] = Place(P_EVT_APPLIED, "event")

    def _build_transitions(self) -> None:
        super()._build_transitions()
        event_priority = 25
        self.transitions[T_RELEASE_JOB] = Transition(
            T_RELEASE_JOB,
            controlled=False,
            bind_fn=_bind_release_job,
            apply_fn=_apply_release_job,
            priority=event_priority,
        )
        self.transitions[T_FAIL_IDLE_MACHINE] = Transition(
            T_FAIL_IDLE_MACHINE,
            controlled=False,
            bind_fn=_bind_fail_idle_machine,
            apply_fn=_apply_fail_idle_machine,
            priority=event_priority,
        )
        self.transitions[T_FAIL_BUSY_MACHINE] = Transition(
            T_FAIL_BUSY_MACHINE,
            controlled=False,
            bind_fn=_bind_fail_busy_machine,
            apply_fn=_apply_fail_busy_machine,
            priority=event_priority,
        )
        self.transitions[T_RECOVER_MACHINE] = Transition(
            T_RECOVER_MACHINE,
            controlled=False,
            bind_fn=_bind_recover_machine,
            apply_fn=_apply_recover_machine,
            priority=event_priority,
        )
        self.transitions[T_REQUEUE_OPERATION] = Transition(
            T_REQUEUE_OPERATION,
            controlled=False,
            bind_fn=_bind_requeue_operation,
            apply_fn=_apply_requeue_operation,
            priority=24,
        )
        self.transitions[T_DEACTIVATE_ELIGIBILITY] = Transition(
            T_DEACTIVATE_ELIGIBILITY,
            controlled=False,
            bind_fn=_bind_deactivate_eligibility,
            apply_fn=_apply_deactivate_eligibility,
            priority=event_priority,
        )
        self.transitions[T_ACTIVATE_ELIGIBILITY] = Transition(
            T_ACTIVATE_ELIGIBILITY,
            controlled=False,
            bind_fn=_bind_activate_eligibility,
            apply_fn=_apply_activate_eligibility,
            priority=event_priority,
        )
        self.transitions[T_DEACTIVATE_PRECEDENCE] = Transition(
            T_DEACTIVATE_PRECEDENCE,
            controlled=False,
            bind_fn=_bind_deactivate_precedence,
            apply_fn=_apply_deactivate_precedence,
            priority=event_priority,
        )
        self.transitions[T_ACTIVATE_PRECEDENCE] = Transition(
            T_ACTIVATE_PRECEDENCE,
            controlled=False,
            bind_fn=_bind_activate_precedence,
            apply_fn=_apply_activate_precedence,
            priority=event_priority,
        )
        self.transitions[T_IGNORE_EVENT] = Transition(
            T_IGNORE_EVENT,
            controlled=False,
            bind_fn=_bind_ignore_event,
            apply_fn=_apply_ignore_event,
            priority=1,
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

        event_transitions = (
            T_RELEASE_JOB,
            T_FAIL_IDLE_MACHINE,
            T_FAIL_BUSY_MACHINE,
            T_RECOVER_MACHINE,
            T_DEACTIVATE_ELIGIBILITY,
            T_ACTIVATE_ELIGIBILITY,
            T_DEACTIVATE_PRECEDENCE,
            T_ACTIVATE_PRECEDENCE,
            T_IGNORE_EVENT,
        )
        for transition in event_transitions:
            p2t(P_EVT_PENDING, transition, "consume_due_event")
            t2p(transition, P_EVT_APPLIED, "produce_applied_event")

        p2t(P_J_UNRELEASED, T_RELEASE_JOB, "consume_job")
        t2p(T_RELEASE_JOB, P_J_ACTIVE, "produce_active_job")

        p2t(P_M_AVAILABLE, T_FAIL_IDLE_MACHINE, "consume_machine")
        t2p(T_FAIL_IDLE_MACHINE, P_M_OFFLINE, "produce_offline_machine")

        p2t(P_M_BUSY, T_FAIL_BUSY_MACHINE, "consume_machine")
        p2t(P_O_PROCESSING, T_FAIL_BUSY_MACHINE, "consume_operation")
        t2p(T_FAIL_BUSY_MACHINE, P_M_OFFLINE, "produce_offline_machine")
        t2p(T_FAIL_BUSY_MACHINE, P_O_INTERRUPTED, "produce_interrupted_operation")

        p2t(P_O_INTERRUPTED, T_REQUEUE_OPERATION, "consume_operation")
        p2t(P_J_ACTIVE, T_REQUEUE_OPERATION, "read_job")
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

    def _populate_marking(self) -> None:
        super()._populate_marking()
        known_jobs = {job.job_id: job for job in self.places[P_J_ACTIVE].objects}
        generated_events: List[EventToken] = []
        for job_id, release_time in self._job_release_times.items():
            if job_id not in known_jobs:
                raise ValueError(f"Unknown job_id in release schedule: {job_id}")
            if release_time < 0:
                raise ValueError("Job release time must be non-negative")
            job = known_jobs[job_id]
            job.release_time = release_time
            if release_time > EPS:
                self.places[P_J_ACTIVE].remove_identity(job)
                self.places[P_J_UNRELEASED].add(job)
                generated_events.append(EventToken(
                    event_id=f"release_job_{job_id}",
                    event_type=EVENT_JOB_RELEASE,
                    event_time=release_time,
                    target_id=job_id,
                ))

        events = generated_events + self._initial_events
        event_ids = [event.event_id for event in events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Duplicate event_id in initial event schedule")
        for event in events:
            self.places[P_EVT_PENDING].add(event)
        self.initial_event_count = len(events)

    def schedule_event(self, event: EventToken) -> None:
        if event.event_time < self.now - EPS:
            raise ValueError("Cannot schedule an event in the past")
        existing = {
            token.event_id
            for place_name in (P_EVT_PENDING, P_EVT_APPLIED)
            for token in self.places[place_name].objects
        }
        if event.event_id in existing:
            raise ValueError(f"Duplicate event_id={event.event_id}")
        self.places[P_EVT_PENDING].add(event)
        self.initial_event_count += 1
        self.check_invariants()

    def next_external_event_time(self) -> Optional[float]:
        times = [
            float(event.event_time)
            for event in self.places[P_EVT_PENDING].objects
            if event.event_time > self.now + EPS
        ]
        return min(times) if times else None

    def advance_until_decision_or_terminal(self) -> None:
        while True:
            self.trigger_all_auto()
            if self.is_terminal() or self.control_bindings():
                return
            candidates = [
                time
                for time in (self.next_completion_time(), self.next_external_event_time())
                if time is not None
            ]
            if not candidates:
                raise RuntimeError(
                    "Dead end: unfinished operations, no binding, completion, or event"
                )
            self.now = min(candidates)

    def is_terminal(self) -> bool:
        expected_jobs = getattr(self, "total_jobs", len(self.instance.jobs))
        return (
            self.places[P_O_COMPLETED].tokens == self.total_operations
            and self.places[P_J_COMPLETED].tokens == expected_jobs
        )

    def check_invariants(self) -> Dict[str, int]:
        operation_places = (
            P_O_WAITING,
            P_O_READY,
            P_O_PROCESSING,
            P_O_INTERRUPTED,
            P_O_COMPLETED,
        )
        machine_places = (P_M_AVAILABLE, P_M_BUSY, P_M_OFFLINE)
        job_places = (P_J_UNRELEASED, P_J_ACTIVE, P_J_COMPLETED)
        operations = [token for name in operation_places for token in self.places[name].objects]
        machines = [token for name in machine_places for token in self.places[name].objects]
        jobs = [token for name in job_places for token in self.places[name].objects]
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
        expected_machines = getattr(
            self, "total_machines", len(self.instance.machine_ids)
        )
        expected_jobs = getattr(self, "total_jobs", len(self.instance.jobs))
        if len(machines) != expected_machines:
            errors.append("Machine conservation violated")
        if len({token.machine_id for token in machines}) != len(machines):
            errors.append("Duplicate machine token")
        if len(jobs) != expected_jobs:
            errors.append("Job conservation violated")
        if len({token.job_id for token in jobs}) != len(jobs):
            errors.append("Duplicate job token")
        if len(eligibility) != self.initial_eligibility_count:
            errors.append("Eligibility conservation violated")
        if len(precedence) != self.initial_precedence_count:
            errors.append("Precedence conservation violated")
        if len(events) != self.initial_event_count:
            errors.append("Event conservation violated")
        if len({token.event_id for token in events}) != len(events):
            errors.append("Duplicate event token")

        completed_tokens = {
            token.op_uid: token for token in self.places[P_O_COMPLETED].objects
        }
        active_jobs = {token.job_id for token in self.places[P_J_ACTIVE].objects}
        incoming: Dict[Tuple[int, int], List[PrecedenceToken]] = {}
        for relation in self.places[P_R_ACTIVE].objects:
            incoming.setdefault(relation.successor_uid, []).append(relation)
        for operation in (
            self.places[P_O_READY].objects + self.places[P_O_PROCESSING].objects
        ):
            if operation.job_id not in active_jobs:
                errors.append(f"Operation {operation.op_uid} belongs to an inactive job")
            if any(
                relation.predecessor_uid not in completed_tokens
                for relation in incoming.get(operation.op_uid, [])
            ):
                errors.append(f"Precedence violation for {operation.op_uid}")

        processing = self.places[P_O_PROCESSING].objects
        busy = self.places[P_M_BUSY].objects
        for operation in processing:
            matches = [
                machine for machine in busy
                if machine.machine_id == operation.assigned_machine_id
                and machine.bound_op_uid == operation.op_uid
            ]
            if len(matches) != 1:
                errors.append(f"Processing operation {operation.op_uid} lacks matching machine")
        for machine in busy:
            matches = [
                operation for operation in processing
                if operation.op_uid == machine.bound_op_uid
                and operation.assigned_machine_id == machine.machine_id
            ]
            if len(matches) != 1:
                errors.append(f"Busy machine {machine.machine_id} lacks matching operation")
        for operation in self.places[P_O_INTERRUPTED].objects:
            if operation.assigned_machine_id is not None:
                errors.append(f"Interrupted operation {operation.op_uid} retains assignment")
        for machine in self.places[P_M_OFFLINE].objects:
            if machine.bound_op_uid is not None:
                errors.append(f"Offline machine {machine.machine_id} retains operation")

        completed_uids = set(completed_tokens)
        for job in self.places[P_J_COMPLETED].objects:
            completed_count = sum(uid[0] == job.job_id for uid in completed_uids)
            if completed_count != job.num_operations:
                errors.append(f"Completed job {job.job_id} has unfinished operations")

        for place_name in operation_places + machine_places + job_places + (
            P_E_ACTIVE,
            P_E_INACTIVE,
            P_R_ACTIVE,
            P_R_INACTIVE,
            P_EVT_PENDING,
            P_EVT_APPLIED,
        ):
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


def _due_events(net: FJSPCPNV3, event_type: str, now: float) -> List[EventToken]:
    return [
        event for event in net.places[P_EVT_PENDING].objects
        if event.event_type == event_type and event.event_time <= now + EPS
    ]


def _find_by_attribute(
    tokens: Iterable[Any], attribute: str, value: Any
) -> Optional[Any]:
    return next(
        (token for token in tokens if getattr(token, attribute) == value),
        None,
    )


def _event_binding(
    transition_name: str,
    event: EventToken,
    *,
    consume_objects: Optional[Dict[str, List[Any]]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Binding:
    consumed = {P_EVT_PENDING: [event]}
    for place_name, tokens in (consume_objects or {}).items():
        consumed.setdefault(place_name, []).extend(tokens)
    values = {"event": event}
    values.update(payload or {})
    return Binding(
        transition_name=transition_name,
        key=(transition_name, event.event_id),
        consume_objects=consumed,
        payload=values,
    )


def _apply_event(
    net: FJSPCPNV3, event: EventToken, now: float, outcome: str
) -> None:
    event.applied_time = float(now)
    event.outcome = outcome
    net.places[P_EVT_APPLIED].add(event)
    net.event_log.append({
        "event": "external_event",
        "event_id": event.event_id,
        "event_type": event.event_type,
        "time": float(now),
        "target_id": event.target_id,
        "outcome": outcome,
    })


def _bind_release_job(net: FJSPCPNV3, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_JOB_RELEASE, now):
        job = _find_by_attribute(
            net.places[P_J_UNRELEASED].objects, "job_id", int(event.target_id)
        )
        if job is not None:
            bindings.append(_event_binding(
                T_RELEASE_JOB,
                event,
                consume_objects={P_J_UNRELEASED: [job]},
                payload={"job": job},
            ))
    return bindings


def _apply_release_job(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    job: JobToken = binding.payload["job"]
    event: EventToken = binding.payload["event"]
    net.places[P_J_ACTIVE].add(job)
    _apply_event(net, event, now, f"released job {job.job_id}")


def _bind_fail_idle_machine(net: FJSPCPNV3, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_MACHINE_FAILURE, now):
        machine = _find_by_attribute(
            net.places[P_M_AVAILABLE].objects, "machine_id", int(event.target_id)
        )
        if machine is not None:
            bindings.append(_event_binding(
                T_FAIL_IDLE_MACHINE,
                event,
                consume_objects={P_M_AVAILABLE: [machine]},
                payload={"machine": machine},
            ))
    return bindings


def _apply_fail_idle_machine(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    machine: MachineToken = binding.payload["machine"]
    event: EventToken = binding.payload["event"]
    machine.failure_count += 1
    machine.available_at = float("inf")
    net.places[P_M_OFFLINE].add(machine)
    _apply_event(net, event, now, f"failed idle machine {machine.machine_id}")


def _bind_fail_busy_machine(net: FJSPCPNV3, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_MACHINE_FAILURE, now):
        machine = _find_by_attribute(
            net.places[P_M_BUSY].objects, "machine_id", int(event.target_id)
        )
        if machine is None:
            continue
        operation = _find_by_attribute(
            net.places[P_O_PROCESSING].objects, "op_uid", machine.bound_op_uid
        )
        if operation is not None:
            bindings.append(_event_binding(
                T_FAIL_BUSY_MACHINE,
                event,
                consume_objects={P_M_BUSY: [machine], P_O_PROCESSING: [operation]},
                payload={"machine": machine, "operation": operation},
            ))
    return bindings


def _apply_fail_busy_machine(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    machine: MachineToken = binding.payload["machine"]
    operation: OperationToken = binding.payload["operation"]
    event: EventToken = binding.payload["event"]
    operation.failed_attempts.append({
        "attempt": operation.attempt_count,
        "machine_id": machine.machine_id,
        "start_time": operation.start_time,
        "failure_time": float(now),
        "planned_finish_time": operation.finish_time,
        "policy": "restart",
    })
    operation.assigned_machine_id = None
    operation.start_time = None
    operation.finish_time = None
    operation.ready_time = float(now)
    machine.bound_op_uid = None
    machine.available_at = float("inf")
    machine.failure_count += 1
    net.places[P_O_INTERRUPTED].add(operation)
    net.places[P_M_OFFLINE].add(machine)
    _apply_event(
        net,
        event,
        now,
        f"failed busy machine {machine.machine_id}; interrupted {operation.op_uid}",
    )


def _active_predecessor_context(
    net: FJSPCPNV3, operation: OperationToken
) -> Optional[Tuple[JobToken, List[PrecedenceToken], List[OperationToken]]]:
    job = _find_by_attribute(
        net.places[P_J_ACTIVE].objects, "job_id", operation.job_id
    )
    if job is None:
        return None
    relations = [
        relation for relation in net.places[P_R_ACTIVE].objects
        if relation.successor_uid == operation.op_uid
    ]
    completed = {
        token.op_uid: token for token in net.places[P_O_COMPLETED].objects
    }
    if any(relation.predecessor_uid not in completed for relation in relations):
        return None
    predecessors = [completed[relation.predecessor_uid] for relation in relations]
    return job, relations, predecessors


def _bind_requeue_operation(net: FJSPCPNV3, now: float) -> List[Binding]:
    del now
    bindings = []
    for operation in net.places[P_O_INTERRUPTED].objects:
        context = _active_predecessor_context(net, operation)
        if context is None:
            continue
        job, relations, predecessors = context
        bindings.append(Binding(
            transition_name=T_REQUEUE_OPERATION,
            key=(T_REQUEUE_OPERATION, operation.op_uid),
            consume_objects={P_O_INTERRUPTED: [operation]},
            read_objects={
                P_J_ACTIVE: [job],
                P_R_ACTIVE: relations,
                P_O_COMPLETED: predecessors,
            },
            payload={"operation": operation},
        ))
    return bindings


def _apply_requeue_operation(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    operation: OperationToken = binding.payload["operation"]
    operation.ready_time = float(now)
    net.places[P_O_READY].add(operation)
    net.event_log.append({
        "event": "requeue",
        "transition": T_REQUEUE_OPERATION,
        "time": float(now),
        "op_uid": operation.op_uid,
        "attempt_count": operation.attempt_count,
    })


def _bind_recover_machine(net: FJSPCPNV3, now: float) -> List[Binding]:
    bindings = []
    for event in _due_events(net, EVENT_MACHINE_RECOVERY, now):
        machine = _find_by_attribute(
            net.places[P_M_OFFLINE].objects, "machine_id", int(event.target_id)
        )
        if machine is not None:
            bindings.append(_event_binding(
                T_RECOVER_MACHINE,
                event,
                consume_objects={P_M_OFFLINE: [machine]},
                payload={"machine": machine},
            ))
    return bindings


def _apply_recover_machine(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    machine: MachineToken = binding.payload["machine"]
    event: EventToken = binding.payload["event"]
    machine.available_at = float(now)
    net.places[P_M_AVAILABLE].add(machine)
    _apply_event(net, event, now, f"recovered machine {machine.machine_id}")


def _bind_relation_event(
    net: FJSPCPNV3,
    now: float,
    *,
    transition_name: str,
    event_type: str,
    source_place: str,
    target_place: str,
) -> List[Binding]:
    bindings = []
    for event in _due_events(net, event_type, now):
        relation = _find_by_attribute(
            net.places[source_place].objects, "relation_id", str(event.target_id)
        )
        if relation is None:
            continue
        if isinstance(relation, PrecedenceToken):
            successor = _find_operation(net, relation.successor_uid)
            if successor is None or successor.current_place != P_O_WAITING:
                continue
            if target_place == P_R_ACTIVE and _would_create_cycle(net, relation):
                continue
        bindings.append(_event_binding(
            transition_name,
            event,
            consume_objects={source_place: [relation]},
            payload={
                "relation": relation,
                "source_place": source_place,
                "target_place": target_place,
            },
        ))
    return bindings


def _apply_relation_event(
    net: FJSPCPNV3, binding: Binding, now: float, verb: str
) -> None:
    relation = binding.payload["relation"]
    event: EventToken = binding.payload["event"]
    target_place = binding.payload["target_place"]
    net.places[target_place].add(relation)
    _apply_event(net, event, now, f"{verb} relation {relation.relation_id}")


def _bind_deactivate_eligibility(net: FJSPCPNV3, now: float) -> List[Binding]:
    return _bind_relation_event(
        net,
        now,
        transition_name=T_DEACTIVATE_ELIGIBILITY,
        event_type=EVENT_ELIGIBILITY_DEACTIVATE,
        source_place=P_E_ACTIVE,
        target_place=P_E_INACTIVE,
    )


def _apply_deactivate_eligibility(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    _apply_relation_event(net, binding, now, "deactivated")


def _bind_activate_eligibility(net: FJSPCPNV3, now: float) -> List[Binding]:
    return _bind_relation_event(
        net,
        now,
        transition_name=T_ACTIVATE_ELIGIBILITY,
        event_type=EVENT_ELIGIBILITY_ACTIVATE,
        source_place=P_E_INACTIVE,
        target_place=P_E_ACTIVE,
    )


def _apply_activate_eligibility(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    _apply_relation_event(net, binding, now, "activated")


def _bind_deactivate_precedence(net: FJSPCPNV3, now: float) -> List[Binding]:
    return _bind_relation_event(
        net,
        now,
        transition_name=T_DEACTIVATE_PRECEDENCE,
        event_type=EVENT_PRECEDENCE_DEACTIVATE,
        source_place=P_R_ACTIVE,
        target_place=P_R_INACTIVE,
    )


def _apply_deactivate_precedence(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    _apply_relation_event(net, binding, now, "deactivated")


def _bind_activate_precedence(net: FJSPCPNV3, now: float) -> List[Binding]:
    return _bind_relation_event(
        net,
        now,
        transition_name=T_ACTIVATE_PRECEDENCE,
        event_type=EVENT_PRECEDENCE_ACTIVATE,
        source_place=P_R_INACTIVE,
        target_place=P_R_ACTIVE,
    )


def _apply_activate_precedence(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    _apply_relation_event(net, binding, now, "activated")


def _find_operation(
    net: FJSPCPNV3, op_uid: Tuple[int, int]
) -> Optional[OperationToken]:
    for place_name in (
        P_O_WAITING,
        P_O_READY,
        P_O_PROCESSING,
        P_O_INTERRUPTED,
        P_O_COMPLETED,
    ):
        operation = _find_by_attribute(net.places[place_name].objects, "op_uid", op_uid)
        if operation is not None:
            return operation
    return None


def _would_create_cycle(net: FJSPCPNV3, candidate: PrecedenceToken) -> bool:
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for relation in list(net.places[P_R_ACTIVE].objects) + [candidate]:
        adjacency.setdefault(relation.predecessor_uid, []).append(relation.successor_uid)
    target = candidate.predecessor_uid
    stack = [candidate.successor_uid]
    visited = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, []))
    return False


def _event_is_applicable(net: FJSPCPNV3, event: EventToken) -> bool:
    target = event.target_id
    if event.event_type == EVENT_JOB_RELEASE:
        return _find_by_attribute(net.places[P_J_UNRELEASED].objects, "job_id", int(target)) is not None
    if event.event_type == EVENT_MACHINE_FAILURE:
        return any(
            _find_by_attribute(net.places[place].objects, "machine_id", int(target)) is not None
            for place in (P_M_AVAILABLE, P_M_BUSY)
        )
    if event.event_type == EVENT_MACHINE_RECOVERY:
        return _find_by_attribute(net.places[P_M_OFFLINE].objects, "machine_id", int(target)) is not None
    relation_specs = {
        EVENT_ELIGIBILITY_DEACTIVATE: (P_E_ACTIVE, False),
        EVENT_ELIGIBILITY_ACTIVATE: (P_E_INACTIVE, False),
        EVENT_PRECEDENCE_DEACTIVATE: (P_R_ACTIVE, True),
        EVENT_PRECEDENCE_ACTIVATE: (P_R_INACTIVE, True),
    }
    if event.event_type not in relation_specs:
        return False
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


def _bind_ignore_event(net: FJSPCPNV3, now: float) -> List[Binding]:
    bindings = []
    for event in net.places[P_EVT_PENDING].objects:
        if event.event_time <= now + EPS and not _event_is_applicable(net, event):
            bindings.append(_event_binding(T_IGNORE_EVENT, event))
    return bindings


def _apply_ignore_event(net: FJSPCPNV3, binding: Binding, now: float) -> None:
    event: EventToken = binding.payload["event"]
    _apply_event(net, event, now, "ignored because target state or guard did not match")
