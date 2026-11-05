from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

EPS = 1e-9

P_O_WAITING = "P_O_Waiting"
P_O_READY = "P_O_Ready"
P_O_PROCESSING = "P_O_Processing"
P_O_COMPLETED = "P_O_Completed"
P_M_AVAILABLE = "P_M_Available"
P_M_BUSY = "P_M_Busy"
P_E_ACTIVE = "P_E_Active"

T_START = "T_StartOperation"
T_FINISH_INTERMEDIATE = "T_FinishIntermediate"
T_FINISH_FINAL = "T_FinishFinal"


@dataclass
class OperationToken:
    """One unique colored token for one concrete operation O_ij."""

    op_uid: Tuple[int, int]
    job_id: int
    op_index: int
    num_operations: int
    predecessor_uid: Optional[Tuple[int, int]]
    successor_uid: Optional[Tuple[int, int]]
    ready_time: Optional[float] = None
    assigned_machine_id: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    current_place: Optional[str] = None


@dataclass
class MachineToken:
    """One conserved token for one machine."""

    machine_id: int
    bound_op_uid: Optional[Tuple[int, int]] = None
    available_at: float = 0.0
    current_place: Optional[str] = None


@dataclass
class EligibilityToken:
    """Read-only relation token: operation-machine compatibility and duration."""

    relation_id: str
    op_uid: Tuple[int, int]
    machine_id: int
    processing_time: float
    current_place: Optional[str] = None

    def __post_init__(self) -> None:
        if self.processing_time <= 0:
            raise ValueError(f"{self.relation_id}: processing_time must be positive")


class Place:
    def __init__(self, name: str, token_kind: str) -> None:
        if token_kind not in {"operation", "machine", "eligibility"}:
            raise ValueError(f"Unsupported token_kind={token_kind}")
        self.name = name
        self.token_kind = token_kind
        self._tokens: List[Any] = []

    @property
    def objects(self) -> List[Any]:
        return self._tokens

    @property
    def tokens(self) -> int:
        return len(self._tokens)

    def add(self, token: Any) -> None:
        if token is None:
            raise ValueError(f"{self.name} cannot receive None")
        if getattr(token, "current_place", None) is not None:
            raise RuntimeError(
                f"Unique token already belongs to {token.current_place}: {token!r}"
            )
        token.current_place = self.name
        self._tokens.append(token)

    def remove_identity(self, token: Any) -> Any:
        for index, current in enumerate(self._tokens):
            if current is token:
                removed = self._tokens.pop(index)
                removed.current_place = None
                return removed
        raise RuntimeError(f"Token not found in {self.name}: {token!r}")

    def find_first(self, predicate: Callable[[Any], bool]) -> Optional[Any]:
        return next((x for x in self._tokens if predicate(x)), None)

    def __repr__(self) -> str:
        return f"Place({self.name}, kind={self.token_kind}, tokens={self.tokens})"


@dataclass
class Binding:
    transition_name: str
    key: Tuple[Any, ...]
    consume_objects: Dict[str, List[Any]] = field(default_factory=dict)
    read_objects: Dict[str, List[Any]] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


class Transition:
    def __init__(
        self,
        name: str,
        *,
        controlled: bool,
        bind_fn: Callable[["FJSPCPN", float], List[Binding]],
        apply_fn: Callable[["FJSPCPN", Binding, float], None],
        priority: int = 0,
    ) -> None:
        self.name = name
        self.controlled = controlled
        self.bind_fn = bind_fn
        self.apply_fn = apply_fn
        self.priority = priority

    def enabled_bindings(self, net: "FJSPCPN", now: float) -> List[Binding]:
        return self.bind_fn(net, now)

    def fire(self, net: "FJSPCPN", binding: Binding, now: float) -> None:
        if binding.transition_name != self.name:
            raise ValueError(f"Binding belongs to {binding.transition_name}, not {self.name}")

        # Revalidate both consumed and read tokens; stale action objects are rejected.
        for place_name, objects in binding.consume_objects.items():
            place = net.places[place_name]
            for token in objects:
                if not any(x is token for x in place.objects):
                    raise RuntimeError(f"[{self.name}] stale consume token in {place_name}")
        for place_name, objects in binding.read_objects.items():
            place = net.places[place_name]
            for token in objects:
                if not any(x is token for x in place.objects):
                    raise RuntimeError(f"[{self.name}] stale read token in {place_name}")

        for place_name, objects in binding.consume_objects.items():
            for token in objects:
                net.places[place_name].remove_identity(token)

        self.apply_fn(net, binding, now)


@dataclass(frozen=True)
class FJSPInstance:
    """jobs[j][o] is {machine_id: processing_time}."""

    jobs: Tuple[Tuple[Mapping[int, float], ...], ...]
    machine_ids: Tuple[int, ...]

    @classmethod
    def from_nested(
        cls,
        jobs: Sequence[Sequence[Mapping[int, float]]],
        machine_ids: Optional[Sequence[int]] = None,
    ) -> "FJSPInstance":
        if not jobs:
            raise ValueError("At least one job is required")

        normalized_jobs: List[Tuple[Mapping[int, float], ...]] = []
        inferred: set[int] = set()
        for job_id, job in enumerate(jobs, start=1):
            if not job:
                raise ValueError(f"Job {job_id} has no operations")
            normalized_ops: List[Mapping[int, float]] = []
            for op_index, options in enumerate(job, start=1):
                if not options:
                    raise ValueError(f"Operation {(job_id, op_index)} has no machine")
                normalized = {int(m): float(p) for m, p in options.items()}
                if any(p <= 0 for p in normalized.values()):
                    raise ValueError(f"Operation {(job_id, op_index)} has invalid time")
                inferred.update(normalized)
                normalized_ops.append(normalized)
            normalized_jobs.append(tuple(normalized_ops))

        mids = tuple(sorted(set(machine_ids or inferred)))
        if not inferred.issubset(set(mids)):
            raise ValueError("machine_ids omits an eligible machine")
        return cls(tuple(normalized_jobs), mids)


class FJSPCPN:
    """Compact executable CPN for static, non-preemptive FJSP V1."""

    def __init__(self, instance: FJSPInstance) -> None:
        self.instance = instance
        self.now = 0.0
        self.places: Dict[str, Place] = {}
        self.transitions: Dict[str, Transition] = {}
        self.arcs: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.total_operations = sum(len(job) for job in instance.jobs)
        self.initial_eligibility_count = sum(
            len(options) for job in instance.jobs for options in job
        )

        self._build_places()
        self._build_transitions()
        self._build_arcs()
        self._populate_marking()
        self.check_invariants()

    def _build_places(self) -> None:
        for name in (P_O_WAITING, P_O_READY, P_O_PROCESSING, P_O_COMPLETED):
            self.places[name] = Place(name, "operation")
        for name in (P_M_AVAILABLE, P_M_BUSY):
            self.places[name] = Place(name, "machine")
        self.places[P_E_ACTIVE] = Place(P_E_ACTIVE, "eligibility")

    def _build_transitions(self) -> None:
        self.transitions[T_START] = Transition(
            T_START, controlled=True, bind_fn=_bind_start, apply_fn=_apply_start
        )
        self.transitions[T_FINISH_INTERMEDIATE] = Transition(
            T_FINISH_INTERMEDIATE,
            controlled=False,
            bind_fn=_bind_finish_intermediate,
            apply_fn=_apply_finish_intermediate,
            priority=20,
        )
        self.transitions[T_FINISH_FINAL] = Transition(
            T_FINISH_FINAL,
            controlled=False,
            bind_fn=_bind_finish_final,
            apply_fn=_apply_finish_final,
            priority=10,
        )

    def _build_arcs(self) -> None:
        def p2t(place: str, transition: str, arc_type: str) -> None:
            self.arcs.append({
                "src_type": "place", "src": place,
                "dst_type": "transition", "dst": transition,
                "arc_type": arc_type,
            })

        def t2p(transition: str, place: str, arc_type: str) -> None:
            self.arcs.append({
                "src_type": "transition", "src": transition,
                "dst_type": "place", "dst": place,
                "arc_type": arc_type,
            })

        p2t(P_O_READY, T_START, "consume_operation")
        p2t(P_M_AVAILABLE, T_START, "consume_machine")
        p2t(P_E_ACTIVE, T_START, "read_eligibility")
        t2p(T_START, P_O_PROCESSING, "produce_operation")
        t2p(T_START, P_M_BUSY, "produce_machine")

        p2t(P_O_PROCESSING, T_FINISH_INTERMEDIATE, "consume_operation")
        p2t(P_M_BUSY, T_FINISH_INTERMEDIATE, "consume_machine")
        p2t(P_O_WAITING, T_FINISH_INTERMEDIATE, "consume_successor")
        t2p(T_FINISH_INTERMEDIATE, P_O_COMPLETED, "produce_completed")
        t2p(T_FINISH_INTERMEDIATE, P_M_AVAILABLE, "release_machine")
        t2p(T_FINISH_INTERMEDIATE, P_O_READY, "release_successor")

        p2t(P_O_PROCESSING, T_FINISH_FINAL, "consume_operation")
        p2t(P_M_BUSY, T_FINISH_FINAL, "consume_machine")
        t2p(T_FINISH_FINAL, P_O_COMPLETED, "produce_completed")
        t2p(T_FINISH_FINAL, P_M_AVAILABLE, "release_machine")

    def _populate_marking(self) -> None:
        for job_id, job in enumerate(self.instance.jobs, start=1):
            n_ops = len(job)
            for zero_index, options in enumerate(job):
                op_index = zero_index + 1
                uid = (job_id, op_index)
                op = OperationToken(
                    op_uid=uid,
                    job_id=job_id,
                    op_index=op_index,
                    num_operations=n_ops,
                    predecessor_uid=None if op_index == 1 else (job_id, op_index - 1),
                    successor_uid=None if op_index == n_ops else (job_id, op_index + 1),
                    ready_time=0.0 if op_index == 1 else None,
                )
                self.places[P_O_READY if op_index == 1 else P_O_WAITING].add(op)

                for machine_id, processing_time in options.items():
                    self.places[P_E_ACTIVE].add(EligibilityToken(
                        relation_id=f"E_O{job_id}_{op_index}_M{machine_id}",
                        op_uid=uid,
                        machine_id=int(machine_id),
                        processing_time=float(processing_time),
                    ))

        for machine_id in self.instance.machine_ids:
            self.places[P_M_AVAILABLE].add(MachineToken(machine_id=int(machine_id)))

    # ---------------- public execution interface ----------------

    def control_bindings(self) -> List[Binding]:
        return sorted(
            self.transitions[T_START].enabled_bindings(self, self.now),
            key=lambda binding: binding.key,
        )

    def fire_control(self, binding_key: Tuple[Any, ...]) -> None:
        current = {binding.key: binding for binding in self.control_bindings()}
        if binding_key not in current:
            raise ValueError(f"Binding is not enabled at t={self.now}: {binding_key}")
        self.transitions[T_START].fire(self, current[binding_key], self.now)
        self.check_invariants()

    def trigger_all_auto(self, max_firings: int = 10000) -> int:
        autos = sorted(
            (t for t in self.transitions.values() if not t.controlled),
            key=lambda t: t.priority,
            reverse=True,
        )
        count = 0
        while count < max_firings:
            fired = False
            for transition in autos:
                bindings = sorted(
                    transition.enabled_bindings(self, self.now), key=lambda b: b.key
                )
                if not bindings:
                    continue
                transition.fire(self, bindings[0], self.now)
                self.check_invariants()
                count += 1
                fired = True
                break  # Bindings must be recomputed after every firing.
            if not fired:
                return count
        raise RuntimeError("Automatic firing limit exceeded")

    def next_completion_time(self) -> Optional[float]:
        times = [
            float(op.finish_time)
            for op in self.places[P_O_PROCESSING].objects
            if op.finish_time is not None and op.finish_time > self.now + EPS
        ]
        return min(times) if times else None

    def advance_until_decision_or_terminal(self) -> None:
        while True:
            self.trigger_all_auto()
            if self.is_terminal() or self.control_bindings():
                return
            next_time = self.next_completion_time()
            if next_time is None:
                raise RuntimeError(
                    "Dead end: unfinished operations, no control binding, and no completion"
                )
            self.now = next_time

    def step(self, binding_key: Tuple[Any, ...]) -> None:
        self.fire_control(binding_key)
        self.advance_until_decision_or_terminal()

    def is_terminal(self) -> bool:
        return self.places[P_O_COMPLETED].tokens == self.total_operations

    def makespan(self) -> float:
        if not self.is_terminal():
            raise RuntimeError("Makespan requested before terminal state")
        return float(self.now)

    def state_counts(self) -> Dict[str, int]:
        return {name: place.tokens for name, place in self.places.items()}

    def check_invariants(self) -> Dict[str, int]:
        operation_places = (P_O_WAITING, P_O_READY, P_O_PROCESSING, P_O_COMPLETED)
        machine_places = (P_M_AVAILABLE, P_M_BUSY)
        operations = [x for p in operation_places for x in self.places[p].objects]
        machines = [x for p in machine_places for x in self.places[p].objects]
        eligibility = list(self.places[P_E_ACTIVE].objects)
        errors: List[str] = []

        if len(operations) != self.total_operations:
            errors.append("Operation conservation violated")
        if len({o.op_uid for o in operations}) != len(operations):
            errors.append("Duplicate operation token")
        if len(machines) != len(self.instance.machine_ids):
            errors.append("Machine conservation violated")
        if len({m.machine_id for m in machines}) != len(machines):
            errors.append("Duplicate machine token")
        if len(eligibility) != self.initial_eligibility_count:
            errors.append("Eligibility conservation violated")
        if len({e.relation_id for e in eligibility}) != len(eligibility):
            errors.append("Duplicate eligibility relation")

        completed = {o.op_uid for o in self.places[P_O_COMPLETED].objects}
        for op in self.places[P_O_READY].objects + self.places[P_O_PROCESSING].objects:
            if op.predecessor_uid is not None and op.predecessor_uid not in completed:
                errors.append(f"Precedence violation for {op.op_uid}")

        processing = self.places[P_O_PROCESSING].objects
        busy = self.places[P_M_BUSY].objects
        for op in processing:
            matched = [
                m for m in busy
                if m.machine_id == op.assigned_machine_id and m.bound_op_uid == op.op_uid
            ]
            if len(matched) != 1:
                errors.append(f"Processing operation {op.op_uid} lacks matching machine")
        for machine in busy:
            matched = [
                op for op in processing
                if op.op_uid == machine.bound_op_uid
                and op.assigned_machine_id == machine.machine_id
            ]
            if len(matched) != 1:
                errors.append(f"Busy machine {machine.machine_id} lacks matching operation")

        for place_name in operation_places:
            if any(op.current_place != place_name for op in self.places[place_name].objects):
                errors.append(f"Operation current_place mismatch in {place_name}")
        for place_name in machine_places:
            if any(m.current_place != place_name for m in self.places[place_name].objects):
                errors.append(f"Machine current_place mismatch in {place_name}")
        if any(e.current_place != P_E_ACTIVE for e in eligibility):
            errors.append("Eligibility moved despite read-only semantics")

        if errors:
            raise AssertionError("\n".join(errors))
        return {
            "operations": len(operations),
            "machines": len(machines),
            "eligibility": len(eligibility),
        }


# ---------------- transition binding / update functions ----------------

def _bind_start(net: FJSPCPN, now: float) -> List[Binding]:
    del now
    ready = {op.op_uid: op for op in net.places[P_O_READY].objects}
    available = {m.machine_id: m for m in net.places[P_M_AVAILABLE].objects}
    bindings: List[Binding] = []

    # Iterate relation tokens: O(|E_active|), not O(|O||M||E|).
    for relation in net.places[P_E_ACTIVE].objects:
        operation = ready.get(relation.op_uid)
        machine = available.get(relation.machine_id)
        if operation is None or machine is None:
            continue
        key = (T_START, operation.op_uid, machine.machine_id, relation.relation_id)
        bindings.append(Binding(
            transition_name=T_START,
            key=key,
            consume_objects={P_O_READY: [operation], P_M_AVAILABLE: [machine]},
            read_objects={P_E_ACTIVE: [relation]},
            payload={
                "operation": operation,
                "machine": machine,
                "eligibility": relation,
                "processing_time": relation.processing_time,
            },
        ))
    return bindings


def _apply_start(net: FJSPCPN, binding: Binding, now: float) -> None:
    op: OperationToken = binding.payload["operation"]
    machine: MachineToken = binding.payload["machine"]
    relation: EligibilityToken = binding.payload["eligibility"]

    op.assigned_machine_id = machine.machine_id
    op.start_time = float(now)
    op.finish_time = float(now) + relation.processing_time
    machine.bound_op_uid = op.op_uid
    machine.available_at = op.finish_time

    net.places[P_O_PROCESSING].add(op)
    net.places[P_M_BUSY].add(machine)
    net.event_log.append({
        "event": "start",
        "transition": T_START,
        "time": float(now),
        "op_uid": op.op_uid,
        "job_id": op.job_id,
        "op_index": op.op_index,
        "machine_id": machine.machine_id,
        "relation_id": relation.relation_id,
        "processing_time": relation.processing_time,
        "finish_time": op.finish_time,
        "binding_key": binding.key,
    })


def _matching_busy_machine(net: FJSPCPN, op: OperationToken) -> Optional[MachineToken]:
    return net.places[P_M_BUSY].find_first(
        lambda m: m.machine_id == op.assigned_machine_id and m.bound_op_uid == op.op_uid
    )


def _bind_finish_intermediate(net: FJSPCPN, now: float) -> List[Binding]:
    bindings: List[Binding] = []
    for op in net.places[P_O_PROCESSING].objects:
        if op.finish_time is None or op.finish_time > now + EPS or op.successor_uid is None:
            continue
        machine = _matching_busy_machine(net, op)
        successor = net.places[P_O_WAITING].find_first(
            lambda candidate, uid=op.successor_uid: candidate.op_uid == uid
        )
        if machine is None or successor is None:
            continue
        bindings.append(Binding(
            transition_name=T_FINISH_INTERMEDIATE,
            key=(T_FINISH_INTERMEDIATE, op.op_uid, machine.machine_id, successor.op_uid),
            consume_objects={
                P_O_PROCESSING: [op],
                P_M_BUSY: [machine],
                P_O_WAITING: [successor],
            },
            payload={"operation": op, "machine": machine, "successor": successor},
        ))
    return bindings


def _apply_finish_intermediate(net: FJSPCPN, binding: Binding, now: float) -> None:
    op: OperationToken = binding.payload["operation"]
    machine: MachineToken = binding.payload["machine"]
    successor: OperationToken = binding.payload["successor"]

    machine.bound_op_uid = None
    machine.available_at = float(now)
    successor.ready_time = float(now)
    net.places[P_O_COMPLETED].add(op)
    net.places[P_M_AVAILABLE].add(machine)
    net.places[P_O_READY].add(successor)
    net.event_log.append({
        "event": "finish",
        "transition": T_FINISH_INTERMEDIATE,
        "time": float(now),
        "op_uid": op.op_uid,
        "job_id": op.job_id,
        "op_index": op.op_index,
        "machine_id": op.assigned_machine_id,
        "successor_uid": successor.op_uid,
    })


def _bind_finish_final(net: FJSPCPN, now: float) -> List[Binding]:
    bindings: List[Binding] = []
    for op in net.places[P_O_PROCESSING].objects:
        if op.finish_time is None or op.finish_time > now + EPS or op.successor_uid is not None:
            continue
        machine = _matching_busy_machine(net, op)
        if machine is None:
            continue
        bindings.append(Binding(
            transition_name=T_FINISH_FINAL,
            key=(T_FINISH_FINAL, op.op_uid, machine.machine_id),
            consume_objects={P_O_PROCESSING: [op], P_M_BUSY: [machine]},
            payload={"operation": op, "machine": machine},
        ))
    return bindings


def _apply_finish_final(net: FJSPCPN, binding: Binding, now: float) -> None:
    op: OperationToken = binding.payload["operation"]
    machine: MachineToken = binding.payload["machine"]
    machine.bound_op_uid = None
    machine.available_at = float(now)
    net.places[P_O_COMPLETED].add(op)
    net.places[P_M_AVAILABLE].add(machine)
    net.event_log.append({
        "event": "finish",
        "transition": T_FINISH_FINAL,
        "time": float(now),
        "op_uid": op.op_uid,
        "job_id": op.job_id,
        "op_index": op.op_index,
        "machine_id": op.assigned_machine_id,
        "successor_uid": None,
    })


def select_spt_binding(bindings: Sequence[Binding]) -> Binding:
    """Deterministic shortest-processing-time sanity policy."""
    if not bindings:
        raise ValueError("No candidate binding")
    return min(
        bindings,
        key=lambda b: (
            float(b.payload["processing_time"]),
            b.payload["operation"].op_uid,
            b.payload["machine"].machine_id,
            b.payload["eligibility"].relation_id,
        ),
    )


def run_policy(
    net: FJSPCPN,
    selector: Callable[[Sequence[Binding]], Binding] = select_spt_binding,
    *,
    max_decisions: int = 100000,
) -> FJSPCPN:
    net.advance_until_decision_or_terminal()
    decisions = 0
    while not net.is_terminal():
        bindings = net.control_bindings()
        if not bindings:
            net.advance_until_decision_or_terminal()
            continue
        net.step(selector(bindings).key)
        decisions += 1
        if decisions > max_decisions:
            raise RuntimeError("Decision limit exceeded")
    return net
