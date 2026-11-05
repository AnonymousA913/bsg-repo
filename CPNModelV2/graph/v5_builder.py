from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from CPNModelV2.core.fjsp_cpn_v2 import (
    Binding,
    EligibilityToken,
    MachineToken,
    OperationToken,
    PrecedenceToken,
)
from CPNModelV2.core.fjsp_cpn_v5 import FJSPCPNV5
from CPNModelV2.graph.schema import GraphEdge, GraphNode, TokenBindingGraph


POLICY_NODE_TYPES: Tuple[str, ...] = (
    "operation",
    "machine",
    "eligibility",
    "precedence",
    "binding",
)

REFERENCE_RELATIONS: Tuple[str, ...] = (
    "eligibility_for_operation",
    "eligibility_for_machine",
    "precedence_from_operation",
    "precedence_to_operation",
    "operation_assigned_to_machine",
    "machine_bound_to_operation",
)

BINDING_ROLE_RELATIONS: Tuple[str, ...] = (
    "binding_consumes_operation",
    "binding_consumes_machine",
    "binding_reads_eligibility",
    "binding_reads_precedence",
)


class V5TokenBindingGraphBuilder:
    """Lift a V5 policy marking into a field-decoupled Token-Binding graph.

    Numeric and categorical fields are learnable. Object identifiers remain in
    ``identity`` and are used only to resolve semantic edges and executable
    Binding keys. Event tokens and the static Place-Transition skeleton stay in
    the CPN execution layer rather than becoming policy nodes.
    """

    def __init__(self, *, validate: bool = True) -> None:
        self.validate = bool(validate)

    def build(self, net: FJSPCPNV5) -> TokenBindingGraph:
        if not isinstance(net, FJSPCPNV5):
            raise TypeError(
                "V5TokenBindingGraphBuilder requires an FJSPCPNV5 instance"
            )
        graph = TokenBindingGraph(
            time=float(net.now),
            metadata={
                "observation": "v5_field_decoupled_token_binding_graph",
                "cpn_class": type(net).__name__,
                "policy_node_types": POLICY_NODE_TYPES,
                "identity_fields_are_learnable": False,
                "event_tokens_visible": False,
                "static_pt_nodes_materialized": False,
            },
        )
        token_nodes: Dict[int, str] = {}
        lookups: Dict[str, Dict[Any, str]] = {
            "operation": {},
            "machine": {},
            "eligibility": {},
            "precedence": {},
        }
        visible = self._policy_tokens(net)
        for place_name, token in visible:
            node = self._token_node(token, place_name, net.now)
            graph.add_node(node)
            token_nodes[id(token)] = node.node_id
            self._register_lookup(lookups, token, node.node_id)

        self._add_reference_relations(graph, lookups)
        self._add_bindings(net, graph, token_nodes)
        if self.validate:
            graph.validate()
        return graph

    @staticmethod
    def _policy_tokens(net: FJSPCPNV5) -> List[Tuple[str, Any]]:
        visible = [
            (place_name, token)
            for place_name, tokens in net.policy_marking().items()
            for token in tokens
        ]
        visible.sort(key=lambda item: _token_sort_key(item[1]))
        return visible

    @staticmethod
    def _token_node(token: Any, place_name: str, now: float) -> GraphNode:
        numeric: Dict[str, float] = {}
        categorical = {"current_place": place_name}

        if isinstance(token, OperationToken):
            denominator = max(1, int(token.num_operations) - 1)
            relative_position = (
                0.0
                if token.num_operations <= 1
                else (float(token.op_index) - 1.0) / float(denominator)
            )
            numeric.update({
                "relative_position": relative_position,
                "num_operations": float(token.num_operations),
                "priority": float(token.priority),
            })
            _add_optional_time(numeric, "release", token.release_time, now)
            _add_optional_time(numeric, "due", token.due_date, now)
            _add_optional_time(numeric, "ready", token.ready_time, now)
            _add_optional_time(numeric, "start", token.start_time, now)
            _add_optional_time(numeric, "finish", token.finish_time, now)
            return GraphNode(
                node_id=_operation_node_id(token.op_uid),
                node_type="operation",
                numeric=numeric,
                categorical=categorical,
                identity={
                    "op_uid": token.op_uid,
                    "job_id": token.job_id,
                    "op_index": token.op_index,
                    "assigned_machine_id": token.assigned_machine_id,
                },
            )

        if isinstance(token, MachineToken):
            numeric["is_busy"] = float(token.bound_op_uid is not None)
            _add_optional_time(numeric, "available", token.available_at, now)
            return GraphNode(
                node_id=f"machine:{token.machine_id}",
                node_type="machine",
                numeric=numeric,
                categorical=categorical,
                identity={
                    "machine_id": token.machine_id,
                    "bound_op_uid": token.bound_op_uid,
                },
            )

        if isinstance(token, EligibilityToken):
            return GraphNode(
                node_id=f"eligibility:{token.relation_id}",
                node_type="eligibility",
                numeric={"processing_time": float(token.processing_time)},
                categorical=categorical,
                identity={
                    "relation_id": token.relation_id,
                    "op_uid": token.op_uid,
                    "machine_id": token.machine_id,
                },
            )

        if isinstance(token, PrecedenceToken):
            return GraphNode(
                node_id=f"precedence:{token.relation_id}",
                node_type="precedence",
                numeric={},
                categorical=categorical,
                identity={
                    "relation_id": token.relation_id,
                    "predecessor_uid": token.predecessor_uid,
                    "successor_uid": token.successor_uid,
                },
            )

        raise TypeError(f"Unsupported V5 policy token: {type(token).__name__}")

    @staticmethod
    def _register_lookup(
        lookups: Dict[str, Dict[Any, str]], token: Any, node_id: str
    ) -> None:
        if isinstance(token, OperationToken):
            lookups["operation"][token.op_uid] = node_id
        elif isinstance(token, MachineToken):
            lookups["machine"][token.machine_id] = node_id
        elif isinstance(token, EligibilityToken):
            lookups["eligibility"][token.relation_id] = node_id
        elif isinstance(token, PrecedenceToken):
            lookups["precedence"][token.relation_id] = node_id

    @staticmethod
    def _add_reference_relations(
        graph: TokenBindingGraph,
        lookups: Mapping[str, Mapping[Any, str]],
    ) -> None:
        for node in tuple(graph.nodes.values()):
            identity = node.identity
            if node.node_type == "operation":
                machine_id = identity.get("assigned_machine_id")
                if machine_id is not None:
                    _add_edge_if_resolved(
                        graph,
                        node.node_id,
                        lookups["machine"].get(machine_id),
                        "operation_assigned_to_machine",
                    )
            elif node.node_type == "machine":
                op_uid = identity.get("bound_op_uid")
                if op_uid is not None:
                    _add_edge_if_resolved(
                        graph,
                        node.node_id,
                        lookups["operation"].get(tuple(op_uid)),
                        "machine_bound_to_operation",
                    )
            elif node.node_type == "eligibility":
                _add_edge_if_resolved(
                    graph,
                    node.node_id,
                    lookups["operation"].get(tuple(identity["op_uid"])),
                    "eligibility_for_operation",
                )
                _add_edge_if_resolved(
                    graph,
                    node.node_id,
                    lookups["machine"].get(identity["machine_id"]),
                    "eligibility_for_machine",
                )
            elif node.node_type == "precedence":
                _add_edge_if_resolved(
                    graph,
                    node.node_id,
                    lookups["operation"].get(
                        tuple(identity["predecessor_uid"])
                    ),
                    "precedence_from_operation",
                )
                _add_edge_if_resolved(
                    graph,
                    node.node_id,
                    lookups["operation"].get(
                        tuple(identity["successor_uid"])
                    ),
                    "precedence_to_operation",
                )

    @staticmethod
    def _add_bindings(
        net: FJSPCPNV5,
        graph: TokenBindingGraph,
        token_nodes: Mapping[int, str],
    ) -> None:
        bindings = sorted(net.control_bindings(), key=lambda item: repr(item.key))
        for index, binding in enumerate(bindings):
            node_id = f"binding:{index:06d}"
            graph.add_node(GraphNode(
                node_id=node_id,
                node_type="binding",
                numeric={},
                categorical={"transition_name": binding.transition_name},
                identity={"binding_key": binding.key},
            ))
            graph.action_keys[node_id] = binding.key
            V5TokenBindingGraphBuilder._add_binding_participants(
                graph, node_id, binding, token_nodes
            )

    @staticmethod
    def _add_binding_participants(
        graph: TokenBindingGraph,
        binding_node_id: str,
        binding: Binding,
        token_nodes: Mapping[int, str],
    ) -> None:
        for participant in binding.participants():
            token_node_id = token_nodes.get(id(participant.token))
            if token_node_id is None:
                raise ValueError(
                    "Enabled Binding participant is absent from the V5 policy "
                    f"marking: {participant.token!r}"
                )
            token_type = graph.nodes[token_node_id].node_type
            edge_type = (
                f"binding_{participant.arc_mode.value}s_{token_type}"
            )
            graph.add_edge(GraphEdge(
                source=token_node_id,
                target=binding_node_id,
                edge_type=edge_type,
                categorical={"place_name": participant.place_name},
            ))


def _operation_node_id(op_uid: Iterable[int]) -> str:
    job_id, op_index = tuple(op_uid)
    return f"operation:{int(job_id)}:{int(op_index)}"


def _add_optional_time(
    numeric: Dict[str, float],
    name: str,
    value: Any,
    now: float,
) -> None:
    known = value is not None and math.isfinite(float(value))
    numeric[f"{name}_time_delta"] = (
        float(value) - float(now) if known else 0.0
    )
    numeric[f"{name}_time_known"] = float(known)


def _add_edge_if_resolved(
    graph: TokenBindingGraph,
    source: str | None,
    target: str | None,
    edge_type: str,
) -> None:
    if source is not None and target is not None:
        graph.add_edge(GraphEdge(source=source, target=target, edge_type=edge_type))


def _token_sort_key(token: Any) -> Tuple[Any, ...]:
    if isinstance(token, OperationToken):
        return (0, *token.op_uid)
    if isinstance(token, MachineToken):
        return (1, token.machine_id)
    if isinstance(token, EligibilityToken):
        return (2, *token.op_uid, token.machine_id, token.relation_id)
    if isinstance(token, PrecedenceToken):
        return (3, *token.predecessor_uid, *token.successor_uid, token.relation_id)
    return (99, type(token).__name__, repr(token))
