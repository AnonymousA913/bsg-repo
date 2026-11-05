from __future__ import annotations

import copy
from typing import Dict, Iterable, Mapping, Protocol, Tuple

from CPNModelV2.core.fjsp_cpn_v2 import (
    Binding,
    P_M_BUSY,
    P_O_PROCESSING,
    T_START,
)
from CPNModelV2.core.fjsp_cpn_v5 import FJSPCPNV5
from CPNModelV2.graph.binding_effect import (
    BindingEffect,
    BindingEffectGraph,
    TokenEffect,
)
from CPNModelV2.graph.generic_tensorizer import (
    BINDING_NODE_TYPE,
    PARTICIPATION_PREFIX,
    REFERENCE_PREFIX,
)
from CPNModelV2.graph.schema import GraphEdge, GraphNode, TokenBindingGraph
from CPNModelV2.integration.fjsp_v5_generic_adapter import (
    FJSPV5GenericGraphBuilder,
)


class BindingEffectProvider(Protocol):
    def effects(
        self,
        net: FJSPCPNV5,
        graph: TokenBindingGraph,
        graph_builder: FJSPV5GenericGraphBuilder,
    ) -> Mapping[str, BindingEffect]: ...


class FJSPV5ImmediateEffectProvider:
    """Fast exact projection of the V5 controlled ``T_START`` firing.

    The provider contains simulator knowledge, while the graph schema,
    tensorizer and policy remain independent of FJSP names. Its output is
    checked against ``CloneFireEffectProvider`` in the test suite.
    """

    def effects(
        self,
        net: FJSPCPNV5,
        graph: TokenBindingGraph,
        graph_builder: FJSPV5GenericGraphBuilder,
    ) -> Mapping[str, BindingEffect]:
        del graph_builder
        bindings = {binding.key: binding for binding in net.control_bindings()}
        result: Dict[str, BindingEffect] = {}
        for binding_node_id, binding_key in graph.action_keys.items():
            binding = bindings.get(binding_key)
            if binding is None:
                raise RuntimeError(f"Enabled Binding disappeared: {binding_key}")
            if binding.transition_name != T_START:
                raise ValueError(
                    "The fast V5 provider only supports the controlled T_START "
                    f"transition, got {binding.transition_name}"
                )
            token_effects = tuple(
                self._token_effect(graph, edge, binding)
                for edge in graph.edges
                if edge.target == binding_node_id
                and edge.edge_type.startswith(PARTICIPATION_PREFIX)
            )
            result[binding_node_id] = BindingEffect(
                binding_node_id=binding_node_id,
                token_effects=token_effects,
            )
        return result

    @staticmethod
    def _token_effect(
        graph: TokenBindingGraph,
        edge: GraphEdge,
        binding: Binding,
    ) -> TokenEffect:
        node = graph.nodes[edge.source]
        arc_mode = _arc_mode(edge.edge_type)
        before_place = node.categorical.get("current_place", "<MISSING>")
        after_place = before_place
        after_numeric = dict(node.numeric)
        reference_changes = 0

        if arc_mode == "consume" and node.node_type == "operation":
            processing_time = float(binding.payload["processing_time"])
            after_place = P_O_PROCESSING
            after_numeric.update({
                "start_time_delta": 0.0,
                "start_time_known": 1.0,
                "finish_time_delta": processing_time,
                "finish_time_known": 1.0,
            })
            reference_changes = 1
        elif arc_mode == "consume" and node.node_type == "machine":
            processing_time = float(binding.payload["processing_time"])
            after_place = P_M_BUSY
            after_numeric.update({
                "is_busy": 1.0,
                "available_time_delta": processing_time,
                "available_time_known": 1.0,
            })
            reference_changes = 1

        return TokenEffect(
            token_node_id=node.node_id,
            participation_role=edge.edge_type,
            arc_mode=arc_mode,
            before_place=before_place,
            after_place=after_place,
            numeric_delta={
                name: float(after_numeric.get(name, 0.0)) - float(value)
                for name, value in node.numeric.items()
            },
            reference_changes=reference_changes,
            survives=True,
        )


class CloneFireEffectProvider:
    """Reference implementation that obtains effects by executable preview."""

    def effects(
        self,
        net: FJSPCPNV5,
        graph: TokenBindingGraph,
        graph_builder: FJSPV5GenericGraphBuilder,
    ) -> Mapping[str, BindingEffect]:
        result: Dict[str, BindingEffect] = {}
        before_token_ids = _token_node_ids(graph)
        for binding_node_id, binding_key in graph.action_keys.items():
            preview = copy.deepcopy(net)
            preview.fire_control(binding_key)
            after = graph_builder.build(preview)
            after_token_ids = _token_node_ids(after)
            token_effects = []
            for edge in graph.edges:
                if (
                    edge.target != binding_node_id
                    or not edge.edge_type.startswith(PARTICIPATION_PREFIX)
                ):
                    continue
                before_node = graph.nodes[edge.source]
                after_node = after.nodes.get(edge.source)
                token_effects.append(_compare_token(
                    graph,
                    after,
                    before_node,
                    after_node,
                    edge.edge_type,
                ))
            result[binding_node_id] = BindingEffect(
                binding_node_id=binding_node_id,
                token_effects=tuple(token_effects),
                created_tokens=len(after_token_ids - before_token_ids),
                removed_tokens=len(before_token_ids - after_token_ids),
            )
        return result


class FJSPV5BindingEffectGraphBuilder:
    """Build the V6 executable Token-Binding-Effect observation."""

    def __init__(
        self,
        *,
        effect_provider: BindingEffectProvider | None = None,
        validate: bool = True,
    ) -> None:
        self.validate = bool(validate)
        self.base_builder = FJSPV5GenericGraphBuilder(validate=validate)
        self.effect_provider = effect_provider or FJSPV5ImmediateEffectProvider()

    def build(self, net: FJSPCPNV5) -> BindingEffectGraph:
        base = self.base_builder.build(net)
        effects = self.effect_provider.effects(net, base, self.base_builder)
        graph = BindingEffectGraph(
            time=base.time,
            nodes={
                key: GraphNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    numeric=dict(node.numeric),
                    categorical=dict(node.categorical),
                    identity=dict(node.identity),
                )
                for key, node in base.nodes.items()
            },
            edges=[
                GraphEdge(
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    numeric=dict(edge.numeric),
                    categorical=dict(edge.categorical),
                )
                for edge in base.edges
            ],
            action_keys=dict(base.action_keys),
            metadata={
                **base.metadata,
                "observation": "binding_effect_graph",
                "effect_semantics": "immediate_controlled_firing",
                "effect_provider": type(self.effect_provider).__name__,
            },
            binding_effects=dict(effects),
        )
        if self.validate:
            graph.validate()
        return graph


def _arc_mode(role: str) -> str:
    if "_consumes_" in role:
        return "consume"
    if "_reads_" in role:
        return "read"
    raise ValueError(f"Cannot infer CPN arc mode from role: {role}")


def _token_node_ids(graph: TokenBindingGraph) -> set[str]:
    return {
        node.node_id
        for node in graph.nodes.values()
        if node.node_type != BINDING_NODE_TYPE
    }


def _reference_signature(
    graph: TokenBindingGraph,
    node_id: str,
) -> set[Tuple[str, str]]:
    return {
        (edge.edge_type, edge.target)
        for edge in graph.edges
        if edge.source == node_id and edge.edge_type.startswith(REFERENCE_PREFIX)
    }


def _compare_token(
    before_graph: TokenBindingGraph,
    after_graph: TokenBindingGraph,
    before: GraphNode,
    after: GraphNode | None,
    role: str,
) -> TokenEffect:
    before_place = before.categorical.get("current_place", "<MISSING>")
    if after is None:
        return TokenEffect(
            token_node_id=before.node_id,
            participation_role=role,
            arc_mode=_arc_mode(role),
            before_place=before_place,
            after_place="<REMOVED>",
            numeric_delta={name: -float(value) for name, value in before.numeric.items()},
            reference_changes=len(_reference_signature(before_graph, before.node_id)),
            survives=False,
        )
    return TokenEffect(
        token_node_id=before.node_id,
        participation_role=role,
        arc_mode=_arc_mode(role),
        before_place=before_place,
        after_place=after.categorical.get("current_place", "<MISSING>"),
        numeric_delta={
            name: float(after.numeric.get(name, 0.0)) - float(value)
            for name, value in before.numeric.items()
        },
        reference_changes=len(
            _reference_signature(before_graph, before.node_id)
            ^ _reference_signature(after_graph, after.node_id)
        ),
        survives=True,
    )
