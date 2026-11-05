from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from CPNModelV2.core.fjsp_cpn_v1 import FJSPInstance

# V2版本：基础模型版本
# 1. 定义了模型在不加入异常事件时的基本Token颜色：工序Token、机器Token、工作Token、合法边Token、加工顺序Token
# 2. 定义了核心对象：库所、变迁、绑定（Binding）

EPS = 1e-9

P_O_WAITING = "P_O_Waiting"
P_O_READY = "P_O_Ready"
P_O_PROCESSING = "P_O_Processing"
P_O_COMPLETED = "P_O_Completed"
P_M_AVAILABLE = "P_M_Available"
P_M_BUSY = "P_M_Busy"
P_J_ACTIVE = "P_J_Active"
P_J_COMPLETED = "P_J_Completed"
P_E_ACTIVE = "P_E_Active"
P_R_ACTIVE = "P_R_Active"

T_ENABLE = "T_EnableOperation"
T_START = "T_StartOperation"
T_FINISH = "T_FinishOperation"
T_COMPLETE_JOB = "T_CompleteJob"


# 工序Token
@dataclass
class OperationToken:
    """One token for one operation; precedence is stored in relation tokens."""

    op_uid: Tuple[int, int] # 身份
    job_id: int     # Job的id
    op_index: int   # 工序位置
    num_operations: int
    release_time: float = 0.0
    due_date: Optional[float] = None
    priority: float = 1.0
    ready_time: Optional[float] = None
    assigned_machine_id: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    attempt_count: int = 0
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    current_place: Optional[str] = None


@dataclass
class MachineToken:
    machine_id: int
    bound_op_uid: Optional[Tuple[int, int]] = None
    available_at: float = 0.0
    failure_count: int = 0
    current_place: Optional[str] = None


@dataclass
class JobToken:
    job_id: int
    num_operations: int
    release_time: float = 0.0
    due_date: Optional[float] = None
    priority: float = 1.0
    completion_time: Optional[float] = None
    current_place: Optional[str] = None


@dataclass
class EligibilityToken:
    relation_id: str
    op_uid: Tuple[int, int]
    machine_id: int
    processing_time: float
    current_place: Optional[str] = None

    def __post_init__(self) -> None:
        if self.processing_time <= 0:
            raise ValueError(f"{self.relation_id}: processing_time must be positive")


@dataclass
class PrecedenceToken:
    relation_id: str

    # 此处的设计比较重要，为了便于流程关系成为Marking中的对象，而不是程序结构的一部分，本属于Operation的predecessor/successor被挪到工序关系Token的属性中
    # 比如O11 → O12，他不是写成O12.predecessor = O11，而是在此处变为：predecessor_uid = O11, successor_uid = O12
    # 同理，Eligibility的Token也是一样的，工序id、机器id和执行时间被放置在一块，成为了一个Token，这就是为什么后面V3和V4版本中可以让关系本身发生激活、失效或新增
    predecessor_uid: Tuple[int, int]
    successor_uid: Tuple[int, int]

    current_place: Optional[str] = None

    def __post_init__(self) -> None:
        if self.predecessor_uid == self.successor_uid:
            raise ValueError(f"{self.relation_id}: self precedence is invalid")


# 库所类：标准的有色Token容器，没什么好讲的
class Place:
    def __init__(self, name: str, token_kind: str) -> None:
        # 此处定义了CPN中的6类Token，实际上FJSP用CPN表示，一类Token基本就足够了，只不过我们将网络结构固定后，需要把机器信息从结构变为Token，顺带着合法边也被拉进来了。
        # 如果还需要考虑异常事件，那么我们就得加入更多变化项，用于模拟各类结构的改变，于是Job被纳入Token以便于新增Job、工序顺序也被纳入Token，以便于修改加工顺序，
        # Event作为异常事件信息，作为一个记录器，也可以适当进入后续的图观测中，不过不能让网络一开始就知道所有的异常事件信息。
        if token_kind not in {
            "operation",
            "machine",
            "job",
            "eligibility",
            "precedence",
            "event",
        }:
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

    # 这里有一处CPN的重要定义，他意味着一枚实体token在任意时刻只能属于一个Place，即：实体Token在CPN中具备唯一性，无法被复制或被消灭，这是Token对象化的一个常见做法
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
        return next((token for token in self._tokens if predicate(token)), None)


# 绑定：该部分代码的核心机制，可以理解为一次具体的Transition激发实例，在析取图+FJSP中，它实际上就是控制FJSP的动作定义
# 常规的Petri中，一次动作就是激发一个变迁，但是CPN中，一次真正的动作是需要考虑到上游库所的token对象的。
# 因此，此处真正的某一次动作是一个三元组：b=(O12, M3, E12,3)，即：激发O12工序，采用M3机器，相关连边合法。这个语义也可以被描述为：
# transition_name:
#     T_StartOperation
#
# consume_objects:
#     P_O_Ready      : O12
#     P_M_Available  : M3
#
# read_objects:
#     P_E_Active     : E(O12,M3)
#
# payload:
#     operation
#     machine
#     eligibility
#     processing_time
# 因此，需要记住：CPN的动作不是Transition，而是Enabled Binding，这是核心，是继承A-E PN和GymPN对于Binding的发现与贡献而得到的理论支撑
class ArcMode(str, Enum):
    CONSUME = "consume"
    READ = "read"


@dataclass(frozen=True)
class BindingParticipant:
    """One concrete colored object participating in an enabled binding."""

    place_name: str
    token: Any
    arc_mode: ArcMode

    @property
    def token_type(self) -> str:
        return type(self.token).__name__


@dataclass
class Binding:
    transition_name: str
    key: Tuple[Any, ...]
    consume_objects: Dict[str, List[Any]] = field(default_factory=dict)
    read_objects: Dict[str, List[Any]] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    def participants(self) -> Tuple[BindingParticipant, ...]:
        """Expose arc roles without duplicating them in each binding payload."""
        result: List[BindingParticipant] = []
        for arc_mode, grouped_objects in (
            (ArcMode.CONSUME, self.consume_objects),
            (ArcMode.READ, self.read_objects),
        ):
            for place_name, tokens in grouped_objects.items():
                result.extend(
                    BindingParticipant(place_name, token, arc_mode)
                    for token in tokens
                )
        return tuple(result)


class Transition:
    def __init__(
        self,
        name: str,
        *,
        controlled: bool,
        bind_fn: Callable[["FJSPCPNV2", float], List[Binding]],
        apply_fn: Callable[["FJSPCPNV2", Binding, float], None],
        priority: int = 0,
    ) -> None:
        self.name = name
        self.controlled = controlled
        self.bind_fn = bind_fn
        self.apply_fn = apply_fn
        self.priority = priority

    # 此处有一个重点，我们的代码中没有显示的Guard类
    # 理论上，CPN的Guard是需要单独设计的，并且他的函数值为True后，Binding才可以激发。我们的代码没有Guard类，而是将Guard逻辑放进了_bind_xxx()的众多函数中
    # 换句话说，一个Binding如果能够被_bind_xxx()生成，那么就意味着Gt(b, M, tho) = True，没有生成，那么就意味着Guard不成立
    # 因此，_bind_xxx()同时承担了Binding枚举器 + Guard求解器
    def enabled_bindings(self, net: "FJSPCPNV2", now: float) -> List[Binding]:
        return self.bind_fn(net, now)

    # 此处的激发函数还会重新验证对象身份和所在库所，防止出现旧绑定被激发
    def fire(self, net: "FJSPCPNV2", binding: Binding, now: float) -> None:
        if binding.transition_name != self.name:
            raise ValueError(f"Binding belongs to {binding.transition_name}, not {self.name}")
        for place_name, objects in binding.consume_objects.items():
            for token in objects:
                if not any(current is token for current in net.places[place_name].objects):
                    raise RuntimeError(f"[{self.name}] stale consume token in {place_name}")
        for place_name, objects in binding.read_objects.items():
            for token in objects:
                if not any(current is token for current in net.places[place_name].objects):
                    raise RuntimeError(f"[{self.name}] stale read token in {place_name}")
        for place_name, objects in binding.consume_objects.items():
            for token in objects:
                net.places[place_name].remove_identity(token)
        self.apply_fn(net, binding, now)


class FJSPCPNV2:
    """Static FJSP CPN with tokenized jobs and precedence relations."""

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
        self.initial_precedence_count = sum(max(0, len(job) - 1) for job in instance.jobs)

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
        for name in (P_J_ACTIVE, P_J_COMPLETED):
            self.places[name] = Place(name, "job")
        self.places[P_E_ACTIVE] = Place(P_E_ACTIVE, "eligibility")
        self.places[P_R_ACTIVE] = Place(P_R_ACTIVE, "precedence")

    # 注意此处的变迁优先级设计，Finish是最高优先级，其次是使能，再次是完成Job，后续加入的异常事件大多为25或24，总体来讲，其顺序如下：
    # Finish(30) → ExternalEvent(25) → Requeue(24) → Enable(20) → CompleteJob(10).
    # 他非常重要，因为他定义了同时事件的离散事件语义，也就是激发优先级
    # 例如工序计划在t=10完成，机器也计划在t=10故障，那么Finish优先级30高于Failure的25，所以：工序先完成，再处理机器故障。
    def _build_transitions(self) -> None:
        self.transitions[T_ENABLE] = Transition(
            T_ENABLE,
            controlled=False,
            bind_fn=_bind_enable,
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
        self.transitions[T_COMPLETE_JOB] = Transition(
            T_COMPLETE_JOB,
            controlled=False,
            bind_fn=_bind_complete_job,
            apply_fn=_apply_complete_job,
            priority=10,
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
        p2t(P_J_ACTIVE, T_ENABLE, "read_job")
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

        p2t(P_J_ACTIVE, T_COMPLETE_JOB, "consume_job")
        p2t(P_O_COMPLETED, T_COMPLETE_JOB, "read_completed_operations")
        t2p(T_COMPLETE_JOB, P_J_COMPLETED, "produce_completed_job")

    def _populate_marking(self) -> None:
        for job_id, job in enumerate(self.instance.jobs, start=1):
            self.places[P_J_ACTIVE].add(
                JobToken(job_id=job_id, num_operations=len(job))
            )
            for zero_index, options in enumerate(job):
                op_index = zero_index + 1
                op_uid = (job_id, op_index)
                self.places[P_O_WAITING].add(OperationToken(
                    op_uid=op_uid,
                    job_id=job_id,
                    op_index=op_index,
                    num_operations=len(job),
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
                        relation_id=f"R_O{job_id}_{op_index - 1}_O{job_id}_{op_index}",
                        predecessor_uid=(job_id, op_index - 1),
                        successor_uid=op_uid,
                    ))
        for machine_id in self.instance.machine_ids:
            self.places[P_M_AVAILABLE].add(MachineToken(machine_id=int(machine_id)))

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

    # 这是一个非常关键的函数
    def trigger_all_auto(self, max_firings: int = 10000) -> int:
        # 找到所有自动激发的变迁，并将其按照优先级排序
        automatic = sorted(
            (transition for transition in self.transitions.values() if not transition.controlled),
            key=lambda transition: transition.priority,
            reverse=True,
        )
        # 寻找当前的可激发Binding，并且激发一个，随后重新计算全部Binding，直到当前时刻再无自动变迁可激发
        count = 0
        while count < max_firings:
            for transition in automatic:
                bindings = sorted(
                    transition.enabled_bindings(self, self.now),
                    key=lambda binding: binding.key,
                )
                if not bindings:
                    continue
                transition.fire(self, bindings[0], self.now)
                self.check_invariants()
                count += 1
                break
            else:
                return count
        raise RuntimeError("Automatic firing limit exceeded")

    def next_completion_time(self) -> Optional[float]:
        times = [
            float(operation.finish_time)
            for operation in self.places[P_O_PROCESSING].objects
            if operation.finish_time is not None and operation.finish_time > self.now + EPS
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
        return (
            self.places[P_O_COMPLETED].tokens == self.total_operations
            and self.places[P_J_COMPLETED].tokens == len(self.instance.jobs)
        )

    def makespan(self) -> float:
        if not self.is_terminal():
            raise RuntimeError("Makespan requested before terminal state")
        return float(self.now)

    def state_counts(self) -> Dict[str, int]:
        return {name: place.tokens for name, place in self.places.items()}

    def check_invariants(self) -> Dict[str, int]:
        operation_places = (P_O_WAITING, P_O_READY, P_O_PROCESSING, P_O_COMPLETED)
        machine_places = (P_M_AVAILABLE, P_M_BUSY)
        job_places = (P_J_ACTIVE, P_J_COMPLETED)
        operations = [token for name in operation_places for token in self.places[name].objects]
        machines = [token for name in machine_places for token in self.places[name].objects]
        jobs = [token for name in job_places for token in self.places[name].objects]
        eligibility = list(self.places[P_E_ACTIVE].objects)
        precedence = list(self.places[P_R_ACTIVE].objects)
        errors: List[str] = []

        if len(operations) != self.total_operations:
            errors.append("Operation conservation violated")
        if len({token.op_uid for token in operations}) != len(operations):
            errors.append("Duplicate operation token")
        if len(machines) != len(self.instance.machine_ids):
            errors.append("Machine conservation violated")
        if len({token.machine_id for token in machines}) != len(machines):
            errors.append("Duplicate machine token")
        if len(jobs) != len(self.instance.jobs):
            errors.append("Job conservation violated")
        if len({token.job_id for token in jobs}) != len(jobs):
            errors.append("Duplicate job token")
        if len(eligibility) != self.initial_eligibility_count:
            errors.append("Eligibility conservation violated")
        if len(precedence) != self.initial_precedence_count:
            errors.append("Precedence conservation violated")
        if len({token.relation_id for token in precedence}) != len(precedence):
            errors.append("Duplicate precedence relation")

        completed = {token.op_uid for token in self.places[P_O_COMPLETED].objects}
        active_jobs = {token.job_id for token in self.places[P_J_ACTIVE].objects}
        incoming: Dict[Tuple[int, int], List[PrecedenceToken]] = {}
        for relation in precedence:
            incoming.setdefault(relation.successor_uid, []).append(relation)
        for operation in (
            self.places[P_O_READY].objects + self.places[P_O_PROCESSING].objects
        ):
            if operation.job_id not in active_jobs:
                errors.append(f"Operation {operation.op_uid} belongs to an inactive job")
            if any(
                relation.predecessor_uid not in completed
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

        for job in self.places[P_J_COMPLETED].objects:
            completed_count = sum(uid[0] == job.job_id for uid in completed)
            if completed_count != job.num_operations:
                errors.append(f"Completed job {job.job_id} has unfinished operations")

        for place_name in operation_places:
            if any(
                token.current_place != place_name
                for token in self.places[place_name].objects
            ):
                errors.append(f"Operation current_place mismatch in {place_name}")
        for place_name in machine_places:
            if any(
                token.current_place != place_name
                for token in self.places[place_name].objects
            ):
                errors.append(f"Machine current_place mismatch in {place_name}")
        for place_name in job_places:
            if any(
                token.current_place != place_name
                for token in self.places[place_name].objects
            ):
                errors.append(f"Job current_place mismatch in {place_name}")
        if any(token.current_place != P_E_ACTIVE for token in eligibility):
            errors.append("Eligibility moved despite read-only semantics")
        if any(token.current_place != P_R_ACTIVE for token in precedence):
            errors.append("Precedence moved despite read-only semantics")

        if errors:
            raise AssertionError("\n".join(errors))
        return {
            "operations": len(operations),
            "machines": len(machines),
            "jobs": len(jobs),
            "eligibility": len(eligibility),
            "precedence": len(precedence),
        }


def _bind_enable(net: FJSPCPNV2, now: float) -> List[Binding]:
    completed = {
        token.op_uid: token for token in net.places[P_O_COMPLETED].objects
    }
    active_jobs = {
        token.job_id: token for token in net.places[P_J_ACTIVE].objects
        if token.release_time <= now + EPS
    }
    incoming: Dict[Tuple[int, int], List[PrecedenceToken]] = {}
    for relation in net.places[P_R_ACTIVE].objects:
        incoming.setdefault(relation.successor_uid, []).append(relation)

    bindings = []
    for operation in net.places[P_O_WAITING].objects:
        job = active_jobs.get(operation.job_id)
        if job is None:
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
                P_J_ACTIVE: [job],
                P_R_ACTIVE: list(relations),
                P_O_COMPLETED: predecessors,
            },
            payload={
                "operation": operation,
                "job": job,
                "precedence": relations,
                "predecessors": predecessors,
            },
        ))
    return bindings


def _apply_enable(net: FJSPCPNV2, binding: Binding, now: float) -> None:
    operation: OperationToken = binding.payload["operation"]
    operation.ready_time = float(now)
    net.places[P_O_READY].add(operation)
    net.event_log.append({
        "event": "enable",
        "transition": T_ENABLE,
        "time": float(now),
        "op_uid": operation.op_uid,
        "precedence": [
            relation.relation_id for relation in binding.payload["precedence"]
        ],
    })


def _bind_start(net: FJSPCPNV2, now: float) -> List[Binding]:
    del now
    ready = {token.op_uid: token for token in net.places[P_O_READY].objects}
    available = {
        token.machine_id: token for token in net.places[P_M_AVAILABLE].objects
    }
    bindings = []
    for relation in net.places[P_E_ACTIVE].objects:
        operation = ready.get(relation.op_uid)
        machine = available.get(relation.machine_id)
        if operation is None or machine is None:
            continue
        bindings.append(Binding(
            transition_name=T_START,
            key=(T_START, operation.op_uid, machine.machine_id, relation.relation_id),
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


def _apply_start(net: FJSPCPNV2, binding: Binding, now: float) -> None:
    operation: OperationToken = binding.payload["operation"]
    machine: MachineToken = binding.payload["machine"]
    relation: EligibilityToken = binding.payload["eligibility"]
    operation.assigned_machine_id = machine.machine_id
    operation.start_time = float(now)
    operation.finish_time = float(now) + relation.processing_time
    operation.attempt_count += 1
    machine.bound_op_uid = operation.op_uid
    machine.available_at = operation.finish_time
    net.places[P_O_PROCESSING].add(operation)
    net.places[P_M_BUSY].add(machine)
    net.event_log.append({
        "event": "start",
        "transition": T_START,
        "time": float(now),
        "op_uid": operation.op_uid,
        "machine_id": machine.machine_id,
        "processing_time": relation.processing_time,
        "finish_time": operation.finish_time,
        "binding_key": binding.key,
    })


def _matching_busy_machine(
    net: FJSPCPNV2, operation: OperationToken
) -> Optional[MachineToken]:
    return net.places[P_M_BUSY].find_first(
        lambda machine: machine.machine_id == operation.assigned_machine_id
        and machine.bound_op_uid == operation.op_uid
    )


def _bind_finish(net: FJSPCPNV2, now: float) -> List[Binding]:
    bindings = []
    for operation in net.places[P_O_PROCESSING].objects:
        if operation.finish_time is None or operation.finish_time > now + EPS:
            continue
        machine = _matching_busy_machine(net, operation)
        if machine is None:
            continue
        bindings.append(Binding(
            transition_name=T_FINISH,
            key=(T_FINISH, operation.op_uid, machine.machine_id),
            consume_objects={P_O_PROCESSING: [operation], P_M_BUSY: [machine]},
            payload={"operation": operation, "machine": machine},
        ))
    return bindings


def _apply_finish(net: FJSPCPNV2, binding: Binding, now: float) -> None:
    operation: OperationToken = binding.payload["operation"]
    machine: MachineToken = binding.payload["machine"]
    machine.bound_op_uid = None
    machine.available_at = float(now)
    net.places[P_O_COMPLETED].add(operation)
    net.places[P_M_AVAILABLE].add(machine)
    net.event_log.append({
        "event": "finish",
        "transition": T_FINISH,
        "time": float(now),
        "op_uid": operation.op_uid,
        "machine_id": operation.assigned_machine_id,
    })


def _bind_complete_job(net: FJSPCPNV2, now: float) -> List[Binding]:
    del now
    completed_by_job: Dict[int, List[OperationToken]] = {}
    for operation in net.places[P_O_COMPLETED].objects:
        completed_by_job.setdefault(operation.job_id, []).append(operation)
    bindings = []
    for job in net.places[P_J_ACTIVE].objects:
        operations = completed_by_job.get(job.job_id, [])
        if len(operations) != job.num_operations:
            continue
        bindings.append(Binding(
            transition_name=T_COMPLETE_JOB,
            key=(T_COMPLETE_JOB, job.job_id),
            consume_objects={P_J_ACTIVE: [job]},
            read_objects={P_O_COMPLETED: list(operations)},
            payload={"job": job, "operations": operations},
        ))
    return bindings


def _apply_complete_job(net: FJSPCPNV2, binding: Binding, now: float) -> None:
    job: JobToken = binding.payload["job"]
    job.completion_time = float(now)
    net.places[P_J_COMPLETED].add(job)
    net.event_log.append({
        "event": "complete_job",
        "transition": T_COMPLETE_JOB,
        "time": float(now),
        "job_id": job.job_id,
    })


def select_spt_binding(bindings: Sequence[Binding]) -> Binding:
    if not bindings:
        raise ValueError("No candidate binding")
    return min(
        bindings,
        key=lambda binding: (
            float(binding.payload["processing_time"]),
            binding.payload["operation"].op_uid,
            binding.payload["machine"].machine_id,
        ),
    )


def run_policy(
    net: FJSPCPNV2,
    selector: Callable[[Sequence[Binding]], Binding] = select_spt_binding,
    *,
    max_decisions: int = 100000,
) -> FJSPCPNV2:
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
