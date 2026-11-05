from __future__ import annotations

from typing import Any

from CPNModelV2.core.fjsp_cpn_v5 import FJSPCPNV5
from CPNModelV2.graph.generic_tensorizer import (
    participation_relation,
    reference_relation,
)
from CPNModelV2.graph.schema import GraphEdge, GraphNode, TokenBindingGraph
from CPNModelV2.graph.v5_builder import (
    BINDING_ROLE_RELATIONS,
    V5TokenBindingGraphBuilder,
)
from CPNModelV2.graph.v5_tensorizer import BINARY_NUMERIC_FIELDS


class FJSPV5GenericGraphBuilder:
    """Adapt the executable FJSP CPN to the generic Token-Binding contract.

    This module owns all FJSP names. The generic tensorizer and policy only see
    runtime token types, feature fields, optional reference relations and
    Token-to-Binding participation roles discovered from its output.
    """

    def __init__(
        self,
        *,
        validate: bool = True,
        declare_all_participation_roles: bool = False,
    ) -> None:
        self.validate = bool(validate)
        self.declare_all_participation_roles = bool(
            declare_all_participation_roles
        )
        self.source_builder = V5TokenBindingGraphBuilder(validate=validate)

    def build(self, net: FJSPCPNV5) -> TokenBindingGraph:
        source = self.source_builder.build(net)
        binary_by_type = {
            node_type: sorted(
                field
                for field in BINARY_NUMERIC_FIELDS
                if any(
                    node.node_type == node_type and field in node.numeric
                    for node in source.nodes.values()
                )
            )
            for node_type in {
                node.node_type for node in source.nodes.values()
            }
        }
        metadata = {
            **source.metadata,
            "observation": "generic_token_binding_graph",
            "adapter": type(self).__name__,
            "binary_numeric_fields": binary_by_type,
            "reference_relations_optional": True,
            "token_types_runtime_declared": True,
        }
        if self.declare_all_participation_roles:
            metadata["participation_roles"] = [
                participation_relation(role)
                for role in BINDING_ROLE_RELATIONS
            ]
        graph = TokenBindingGraph(
            time=source.time,
            metadata=metadata,
        )
        for node in source.nodes.values():
            graph.add_node(GraphNode(
                node_id=node.node_id,
                node_type=node.node_type,
                numeric=dict(node.numeric),
                categorical=dict(node.categorical),
                identity=dict(node.identity),
            ))
        for edge in source.edges:
            relation = (
                participation_relation(edge.edge_type)
                if edge.edge_type.startswith("binding_")
                else reference_relation(edge.edge_type)
            )
            graph.add_edge(GraphEdge(
                source=edge.source,
                target=edge.target,
                edge_type=relation,
                numeric=dict(edge.numeric),
                categorical=dict(edge.categorical),
            ))
        graph.action_keys.update({
            node_id: _tuple_key(key)
            for node_id, key in source.action_keys.items()
        })
        if self.validate:
            graph.validate()
        return graph


def _tuple_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_tuple_key(item) for item in value)
    if isinstance(value, list):
        return tuple(_tuple_key(item) for item in value)
    return value
