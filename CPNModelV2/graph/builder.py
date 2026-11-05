from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

from CPNModelV2.core.fjsp_cpn_v2 import (
    Binding,
    EligibilityToken,
    FJSPCPNV2,
    JobToken,
    MachineToken,
    OperationToken,
    PrecedenceToken,
)
from CPNModelV2.core.fjsp_cpn_v3 import EventToken, P_EVT_APPLIED, P_EVT_PENDING
from CPNModelV2.graph.schema import GraphEdge, GraphNode, TokenBindingGraph

# 核心代码
class TokenBindingGraphBuilder:
    """Build a dynamic semantic graph directly from a CPN marking."""

    def __init__(
        self,
        *,
        include_revealed_pending_events: bool = True,
        include_applied_events: bool = False,
    ) -> None:
        self.include_revealed_pending_events = include_revealed_pending_events
        self.include_applied_events = include_applied_events

    def build(self, net: FJSPCPNV2) -> TokenBindingGraph:
        graph = TokenBindingGraph(
            time=float(net.now),
            metadata={
                "observation": "dynamic_token_binding_graph_v1",
                "cpn_class": type(net).__name__,
                "applied_events_included": self.include_applied_events,
            },
        )
        token_nodes: Dict[int, str] = {}
        lookups: Dict[str, Dict[Any, str]] = {
            "operation": {},
            "machine": {},
            "job": {},
            "eligibility": {},
            "precedence": {},
        }

        # 整个代码中最重要的几个函数就在这里了，也就是下面的五行代码
        # 它实际上以及揭示了目前图的四层结构：Structure + Token objects + Token relations + Binding actions
        # 第一层：把Petri网的骨架原样放进图
        self._add_structure(net, graph)
        # 第二层：Marking中的每一枚Token都是独立节点
        self._add_tokens(net, graph, token_nodes, lookups)
        # 第三层：把Token颜色里的引用关系拓扑化，这是最值得关注的部分
        # 它遍历所有Token节点，然后根据Token类型读取identity字段，创建语义边
        self._add_token_semantics(graph, lookups)
        # 第四层：Binding节点，也是最关键的一层
        self._add_bindings(net, graph, token_nodes)
        graph.validate()
        return graph

    # 第一层：把Petri网的骨架原样放进图
    def _add_structure(self, net: FJSPCPNV2, graph: TokenBindingGraph) -> None:
        # 首先就是让所有的Place成为节点
        for place_name in sorted(net.places):
            place = net.places[place_name]
            graph.add_node(GraphNode(
                node_id=_place_node_id(place_name),
                node_type="place",  # 图节点类型为库所
                # 节点进入的时候只携带了2个信息：place_name和token_kind，他们是库所的身份信息
                categorical={
                    "place_name": place_name,
                    "token_kind": place.token_kind,
                },
                identity={"place_name": place_name},
            ))
        # 然后就是让所有的Transition也成为图中的节点
        for transition_name in sorted(net.transitions):
            transition = net.transitions[transition_name]
            graph.add_node(GraphNode(
                node_id=_transition_node_id(transition_name),
                node_type="transition",
                # 变迁进入的时候携带了3个信息
                numeric={
                    "controlled": float(transition.controlled), # 是否可控
                    "priority": float(transition.priority),     # 优先级
                },
                categorical={"transition_name": transition_name},   # 变迁名称
                identity={"transition_name": transition_name},
            ))
        # 弧信息也被完整保留
        for arc in net.arcs:
            source = (
                _place_node_id(arc["src"])
                if arc["src_type"] == "place"
                else _transition_node_id(arc["src"])
            )
            target = (
                _place_node_id(arc["dst"])
                if arc["dst_type"] == "place"
                else _transition_node_id(arc["dst"])
            )
            # 不过弧的信息只有1条，那就是弧类型，比如原CPN中如果是：
            # P_O_Ready
            #     --consume_operation-->
            # T_StartOperation
            # 就成为图里的：
            # place:P_O_Ready
            #     --arc:consume_operation-->
            # transition:T_StartOperation
            graph.add_edge(GraphEdge(
                source=source,
                target=target,
                edge_type=f"arc:{arc['arc_type']}",
            ))

    # 第二层：Marking中的每一枚Token都是独立节点
    # 这里也是当前方法与PN-GCN区别最大的地方之一。
    def _add_tokens(
        self,
        net: FJSPCPNV2,
        graph: TokenBindingGraph,
        token_nodes: Dict[int, str],
        lookups: Dict[str, Dict[Any, str]],
    ) -> None:
        visible_tokens = []
        for place_name in sorted(net.places):
            for token in net.places[place_name].objects:
                if not self._event_is_visible(token, place_name, net.now):
                    continue
                visible_tokens.append((place_name, token))
        visible_tokens.sort(key=lambda item: _token_sort_key(item[1]))

        # 代码遍历所有的库所，对每一枚Token调用_token_node()
        # 所以：
        # P_O_Ready
        # 里面如果有：
        # O11
        # O21
        # O31
        # 不是：
        # P_O_Ready feature = [token_count=3,...]
        # 而是真的有三个节点：
        # operation:1:1
        # operation:2:1
        # operation:3:1
        # 而且每一个节点还会通过token_located_in_place，连接到place: P_O_Ready
        # 这样Token就保持了个体身份
        for place_name, token in visible_tokens:
            node = _token_node(token, place_name, net.now)
            graph.add_node(node)
            token_nodes[id(token)] = node.node_id
            _register_lookup(lookups, token, node.node_id)
            # 这个设计有一个很重要的冗余
            # 以Operation为例，代码同时保存了：categorical["current_place"] = place_name，但又建立Operation -> Place的token_located_in_place边
            # 也就是说：token状态现在有2种表达：1）categorical feature; 2)explicit Place edge
            # 这也不是错误，但是他也给我们以后提供了一个很好的消融实验，因为我们可以去掉一个，去看看他们效果的差别，用于证明显式Place节点可能没有提供额外价值
            graph.add_edge(GraphEdge(
                source=node.node_id,
                target=_place_node_id(place_name),
                edge_type="token_located_in_place",
            ))

    # Eventtoken这里做了一个非常重要的可见性过滤，确保智能体无法观测到全局的EventToken，而是到了时间才可以看到该事件的Token，避免了未来信息泄露
    def _event_is_visible(self, token: Any, place_name: str, now: float) -> bool:
        if not isinstance(token, EventToken):
            return True
        if place_name == P_EVT_APPLIED:
            return self.include_applied_events
        if place_name == P_EVT_PENDING:
            return (
                self.include_revealed_pending_events
                and token.reveal_time <= now + 1e-9
            )
        return True

    # 第三层：把Token颜色里的引用关系拓扑化，这是最值得关注的部分
    # 它遍历所有Token节点，然后根据Token类型读取identity字段，创建语义边
    def _add_token_semantics(
        self,
        graph: TokenBindingGraph,
        lookups: Dict[str, Dict[Any, str]],
    ) -> None:
        for node in tuple(graph.nodes.values()):
            identity = node.identity


            if node.node_type == "operation":
                # Operation与Job：根据Operation.job_id
                # 生成：Job --job_contains_operation--> Operation
                # 所以：
                # J1
                #  ├── O11
                #  ├── O12
                #  └── O13
                # 这就使得job_id变为了图关系
                _add_reference_edge(
                    graph,
                    lookups["job"].get(identity["job_id"]),
                    node.node_id,
                    "job_contains_operation",
                )

                # Operation与Machine的当前执行绑定
                # 如果：operation.assigned_machine_id = 3
                # 就产生：
                # Operation
                #     --operation_assigned_to_machine-->
                # Machine
                # 同时，如果：
                # machine.bound_op_uid = O12
                # 还会生成反向语义：
                # Machine
                #     --machine_bound_to_operation-->
                # Operation
                # 于是：
                # O12  ──assigned-to──▶ M3
                # O12  ◀──bound-to──── M3
                # 它们不是普通“reverse edge”，而是两个有各自语义名称的关系。
                assigned = identity.get("assigned_machine_id")
                if assigned is not None:
                    _add_reference_edge(
                        graph,
                        node.node_id,
                        lookups["machine"].get(assigned),
                        "operation_assigned_to_machine",
                    )

            elif node.node_type == "machine":
                bound = identity.get("bound_op_uid")
                if bound is not None:
                    _add_reference_edge(
                        graph,
                        node.node_id,
                        lookups["operation"].get(tuple(bound)),
                        "machine_bound_to_operation",
                    )

            # EligibilityToken真正变成了关系对象，一个E12,3会生成：
            # Eligibility
            #     --eligibility_for_operation-->
            # O12
            # Eligibility
            #     --eligibility_for_machine-->
            # M3
            # 因此它现在不是Song那种直接： O12 --p=5--> M3
            # 而是：E12,3 分别连接O12和M3
            # 这就是关系节点化，其优势是Eligibility自身还拥有Active/Inactive状态、processing time，将来可能拥有可靠性、版本、能力等级等属性
            elif node.node_type == "eligibility":
                _add_reference_edge(
                    graph,
                    node.node_id,
                    lookups["operation"].get(tuple(identity["op_uid"])),
                    "eligibility_for_operation",
                )
                _add_reference_edge(
                    graph,
                    node.node_id,
                    lookups["machine"].get(identity["machine_id"]),
                    "eligibility_for_machine",
                )
            # Precedence也是同样的关系节点化
            # 比如R12变为O11和O12，这和直接进行O11 -> O12相比，多了一枚关系对象，对于静态FJSP可能显得复杂，但是对于R12这种会被Inactive的动态关系，就很有意义了。
            elif node.node_type == "precedence":
                _add_reference_edge(
                    graph,
                    node.node_id,
                    lookups["operation"].get(tuple(identity["predecessor_uid"])),
                    "precedence_from_operation",
                )
                _add_reference_edge(
                    graph,
                    node.node_id,
                    lookups["operation"].get(tuple(identity["successor_uid"])),
                    "precedence_to_operation",
                )

            # EventToken同样被关系化，比如：
            # MACHINE_FAILURE
            # target_id = 3
            # 会得到：Event --event_targets_machine--> M3
            # Eligibility事件、Precedence事件和Job事件分别连向对应对象。
            elif node.node_type == "event":
                self._add_event_references(graph, node, lookups)

    def _add_event_references(
        self,
        graph: TokenBindingGraph,
        node: GraphNode,
        lookups: Dict[str, Dict[Any, str]],
    ) -> None:
        event_type = node.categorical["event_type"]
        target = node.identity.get("target_id")
        if event_type in {"MACHINE_FAILURE", "MACHINE_RECOVERY", "MACHINE_ADD"}:
            _add_reference_edge(
                graph, node.node_id, lookups["machine"].get(_as_int(target)),
                "event_targets_machine",
            )
        elif event_type in {"JOB_RELEASE", "JOB_ADD"}:
            _add_reference_edge(
                graph, node.node_id, lookups["job"].get(_as_int(target)),
                "event_targets_job",
            )
        elif event_type in {"ELIGIBILITY_ACTIVATE", "ELIGIBILITY_DEACTIVATE"}:
            _add_reference_edge(
                graph, node.node_id, lookups["eligibility"].get(str(target)),
                "event_targets_eligibility",
            )
        elif event_type in {"PRECEDENCE_ACTIVATE", "PRECEDENCE_DEACTIVATE"}:
            _add_reference_edge(
                graph, node.node_id, lookups["precedence"].get(str(target)),
                "event_targets_precedence",
            )

        payload = node.identity.get("payload") or {}
        op_uid = payload.get("op_uid")
        if isinstance(op_uid, (list, tuple)) and len(op_uid) == 2:
            _add_reference_edge(
                graph, node.node_id,
                lookups["operation"].get(tuple(int(item) for item in op_uid)),
                "event_references_operation",
            )
        machine_id = payload.get("machine_id")
        if machine_id is not None:
            _add_reference_edge(
                graph, node.node_id, lookups["machine"].get(_as_int(machine_id)),
                "event_references_machine",
            )

    # 第四层：Binding节点，也是最关键的一层
    def _add_bindings(
        self,
        net: FJSPCPNV2,
        graph: TokenBindingGraph,
        token_nodes: Dict[int, str],
    ) -> None:
        bindings = []
        for transition in net.transitions.values():
            # 注意，Binding仅仅枚举目前受控变迁的，也就是Enabled Bindings
            # 所以，图里面的Binding节点不是所有Petri firing，而是RL当前真正可以选择的动作
            if transition.controlled:
                # 注意，Binding首先连接他所属的变迁，比如：
                # binding:000001
                #     --binding_instantiates_transition-->
                # T_StartOperation
                bindings.extend(transition.enabled_bindings(net, net.now))
        bindings.sort(key=lambda binding: repr(binding.key))

        for index, binding in enumerate(bindings):
            node_id = f"binding:{index:06d}"
            graph.add_node(_binding_node(node_id, binding))
            graph.action_keys[node_id] = binding.key
            graph.add_edge(GraphEdge(
                source=node_id,
                target=_transition_node_id(binding.transition_name),
                edge_type="binding_instantiates_transition",
            ))
            # 然后参与Token按照CPN角色连接Binding，这是这一套图目前最漂亮的一部分
            # _add_binding_participants()根据consume_objects和read_objects生成不同边，对于b = (O11, M2, E11,2)，当前就是：
            # O11 ──binding_consumes_operation──▶ B1
            # M2  ──binding_consumes_machine────▶ B1
            # E12 ──binding_reads_eligibility───▶ B1
            self._add_binding_participants(
                graph, node_id, binding.consume_objects, token_nodes, "consumes"
            )
            self._add_binding_participants(
                graph, node_id, binding.read_objects, token_nodes, "reads"
            )

    @staticmethod
    def _add_binding_participants(
        graph: TokenBindingGraph,
        binding_node_id: str,
        objects_by_place: Dict[str, list[Any]],
        token_nodes: Dict[int, str],
        role: str,
    ) -> None:
        for place_name, objects in sorted(objects_by_place.items()):
            for token in objects:
                token_node_id = token_nodes.get(id(token))
                if token_node_id is None:
                    raise ValueError(
                        f"Binding participant is hidden or absent from graph: {token!r}"
                    )
                token_type = graph.nodes[token_node_id].node_type
                graph.add_edge(GraphEdge(
                    source=token_node_id,
                    target=binding_node_id,
                    edge_type=f"binding_{role}_{token_type}",
                    categorical={"place_name": place_name},
                ))

# 这个函数是整个特征工程的入口，需要逐类Token的看
def _token_node(token: Any, place_name: str, now: float) -> GraphNode:
    categorical = {"current_place": place_name}
    numeric: Dict[str, float] = {}
    identity: Dict[str, Any]

    # OperationToken的特征选择
    if isinstance(token, OperationToken):
        node_type = "operation"
        # 1. 数值特征
        numeric.update({
            "op_index": float(token.op_index),                          # 1.1 工序索引
            "num_operations": float(token.num_operations),              # 1.2 工序数量
            "attempt_count": float(token.attempt_count),                # 1.3 尝试次数
            "failed_attempt_count": float(len(token.failed_attempts)),  # 1.4 失败尝试次数
        })
        _add_optional_time(numeric, "ready", token.ready_time, now)     # 1.5 准备时间
        _add_optional_time(numeric, "start", token.start_time, now)     # 1.6 开始时间
        _add_optional_time(numeric, "finish", token.finish_time, now)   # 1.7 完成时间

        # 2. 身份信息
        identity = {
            "op_uid": token.op_uid,
            "job_id": token.job_id,
            "assigned_machine_id": token.assigned_machine_id,
        }

    # MachineToken的特征选择
    elif isinstance(token, MachineToken):
        node_type = "machine"
        numeric.update({
            "failure_count": float(token.failure_count),
            "has_bound_operation": float(token.bound_op_uid is not None),
        })
        _add_optional_time(numeric, "available", token.available_at, now)
        identity = {
            "machine_id": token.machine_id,
            "bound_op_uid": token.bound_op_uid, # 此处需要注意bound_up_uid没有直接数值化，稍后他会被转为一条图边
        }

    # JobToken的特征选择
    elif isinstance(token, JobToken):
        node_type = "job"
        numeric.update({
            "num_operations": float(token.num_operations),
            "priority": float(token.priority),
        })
        _add_optional_time(numeric, "release", token.release_time, now)
        _add_optional_time(numeric, "due", token.due_date, now)
        _add_optional_time(numeric, "completion", token.completion_time, now)
        identity = {"job_id": token.job_id}

    elif isinstance(token, EligibilityToken):
        node_type = "eligibility"
        numeric["processing_time"] = float(token.processing_time)
        identity = {
            "relation_id": token.relation_id,
            "op_uid": token.op_uid,
            "machine_id": token.machine_id,
        }
    elif isinstance(token, PrecedenceToken):
        node_type = "precedence"
        identity = {
            "relation_id": token.relation_id,
            "predecessor_uid": token.predecessor_uid,
            "successor_uid": token.successor_uid,
        }
    elif isinstance(token, EventToken):
        node_type = "event"
        categorical["event_type"] = token.event_type
        categorical["event_status"] = (
            "applied" if place_name == P_EVT_APPLIED else "pending"
        )
        _add_optional_time(numeric, "event", token.event_time, now)
        _add_optional_time(numeric, "reveal", token.reveal_time, now)
        _add_optional_time(numeric, "applied", token.applied_time, now)
        _add_event_payload_counts(numeric, token.payload)
        identity = {
            "event_id": token.event_id,
            "target_id": token.target_id,
            "payload": token.payload,
        }
    else:
        raise TypeError(f"Unsupported Token class: {type(token).__name__}")

    return GraphNode(
        node_id=_token_node_id(token),
        node_type=node_type,
        numeric=numeric,
        categorical=categorical,
        identity=identity,
    )


def _binding_node(node_id: str, binding: Binding) -> GraphNode:
    processing_time = binding.payload.get("processing_time")
    numeric = {
        "consume_count": float(sum(map(len, binding.consume_objects.values()))),
        "read_count": float(sum(map(len, binding.read_objects.values()))),
    }
    if processing_time is not None and math.isfinite(float(processing_time)):
        numeric["processing_time"] = float(processing_time)
        numeric["processing_time_known"] = 1.0
    else:
        numeric["processing_time"] = 0.0
        numeric["processing_time_known"] = 0.0
    return GraphNode(
        node_id=node_id,
        node_type="binding",
        numeric=numeric,
        categorical={"transition_name": binding.transition_name},
        identity={"binding_key": binding.key},
    )


def _register_lookup(
    lookups: Dict[str, Dict[Any, str]], token: Any, node_id: str
) -> None:
    if isinstance(token, OperationToken):
        lookups["operation"][token.op_uid] = node_id
    elif isinstance(token, MachineToken):
        lookups["machine"][token.machine_id] = node_id
    elif isinstance(token, JobToken):
        lookups["job"][token.job_id] = node_id
    elif isinstance(token, EligibilityToken):
        lookups["eligibility"][token.relation_id] = node_id
    elif isinstance(token, PrecedenceToken):
        lookups["precedence"][token.relation_id] = node_id


def _add_reference_edge(
    graph: TokenBindingGraph,
    source: Optional[str],
    target: Optional[str],
    edge_type: str,
) -> None:
    if source is not None and target is not None:
        graph.add_edge(GraphEdge(source=source, target=target, edge_type=edge_type))


def _add_optional_time(
    features: Dict[str, float], name: str, value: Optional[float], now: float
) -> None:
    known = value is not None and math.isfinite(float(value))
    features[f"{name}_time_delta"] = float(value) - float(now) if known else 0.0
    features[f"{name}_time_known"] = float(known)


def _add_event_payload_counts(
    features: Dict[str, float], payload: Optional[Dict[str, Any]]
) -> None:
    values = payload or {}
    operations = values.get("operations") or []
    eligibility = values.get("eligibility") or []
    precedence = values.get("precedence") or []
    features["payload_operation_count"] = float(len(operations))
    features["payload_eligibility_count"] = float(
        len(eligibility)
        + sum(len(options) for options in operations if isinstance(options, dict))
    )
    features["payload_precedence_count"] = float(len(precedence))


def _token_node_id(token: Any) -> str:
    if isinstance(token, OperationToken):
        return f"operation:{token.op_uid[0]}:{token.op_uid[1]}"
    if isinstance(token, MachineToken):
        return f"machine:{token.machine_id}"
    if isinstance(token, JobToken):
        return f"job:{token.job_id}"
    if isinstance(token, EligibilityToken):
        return f"eligibility:{token.relation_id}"
    if isinstance(token, PrecedenceToken):
        return f"precedence:{token.relation_id}"
    if isinstance(token, EventToken):
        return f"event:{token.event_id}"
    raise TypeError(f"Unsupported Token class: {type(token).__name__}")


def _token_sort_key(token: Any) -> Tuple[str, str]:
    return type(token).__name__, _token_node_id(token)


def _place_node_id(place_name: str) -> str:
    return f"place:{place_name}"


def _transition_node_id(transition_name: str) -> str:
    return f"transition:{transition_name}"


def _as_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value
