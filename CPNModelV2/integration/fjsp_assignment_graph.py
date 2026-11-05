from __future__ import annotations

"""FJSP Assignment Graph adapter for the Section 5.4 AG-PPO baseline.

The policy, schema and tensorizer are shared with the Section 5.6 AG-PPO
baseline.  This adapter only translates the FJSP decision marking into the
expanded token--action incidence graph expected by that implementation.  It
intentionally excludes BSG reference edges and role-aware Binding semantics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from torch_geometric.data import HeteroData

from CPNModelV2.core.fjsp_cpn_v5 import (
    P_E_ACTIVE,
    P_E_INACTIVE,
    P_M_AVAILABLE,
    P_M_BUSY,
    P_M_OFFLINE,
    P_O_COMPLETED,
    P_O_INTERRUPTED,
    P_O_PROCESSING,
    P_O_READY,
    P_O_WAITING,
    P_R_ACTIVE,
    P_R_INACTIVE,
)
from CPNModelV2.graph.schema import GraphNode, TokenBindingGraph
from CPNModelV2.graph.v5_builder import V5TokenBindingGraphBuilder
from CPNModelV2.integration.bpm2024_assignment_graph import (
    AssignmentGraphFeatureSchema,
    ControlledAssignmentGraphPolicy,
)


TOKEN_NODE_TYPES: Tuple[str, ...] = (
    "operation",
    "machine",
    "eligibility",
    "precedence",
)
ACTION_NODE_TYPE = "a_transition"

NUMERIC_FIELDS = {
    "operation": (
        "relative_position",
        "num_operations",
        "priority",
        "release_time_delta",
        "release_time_known",
        "due_time_delta",
        "due_time_known",
        "ready_time_delta",
        "ready_time_known",
        "start_time_delta",
        "start_time_known",
        "finish_time_delta",
        "finish_time_known",
    ),
    "machine": (
        "is_busy",
        "available_time_delta",
        "available_time_known",
    ),
    "eligibility": ("processing_time",),
    "precedence": (),
}

PLACE_VALUES = {
    "operation": (
        P_O_WAITING,
        P_O_READY,
        P_O_PROCESSING,
        P_O_INTERRUPTED,
        P_O_COMPLETED,
    ),
    "machine": (P_M_AVAILABLE, P_M_BUSY, P_M_OFFLINE),
    "eligibility": (P_E_ACTIVE, P_E_INACTIVE),
    "precedence": (P_R_ACTIVE, P_R_INACTIVE),
}

EDGE_TYPES: Tuple[Tuple[str, str, str], ...] = tuple(
    edge_type
    for token_type in TOKEN_NODE_TYPES
    for edge_type in (
        (token_type, "edge_to_a_transition", ACTION_NODE_TYPE),
        (ACTION_NODE_TYPE, f"edge_to_{token_type}", token_type),
    )
) + ((ACTION_NODE_TYPE, "self_loop", ACTION_NODE_TYPE),)


class FJSPAssignmentGraphBuilder:
    """Build a fixed-metadata Assignment Graph from an FJSP V5 marking."""

    def __init__(self, *, validate: bool = True) -> None:
        self.validate = bool(validate)
        self.source_builder = V5TokenBindingGraphBuilder(validate=validate)

    def build(self, net: Any) -> TokenBindingGraph:
        source = self.source_builder.build(net)
        assignment_graph = HeteroData()
        local_index: Dict[str, int] = {}

        for node_type in TOKEN_NODE_TYPES:
            nodes = source.nodes_of_type(node_type)
            rows = [_node_features(node, node_type) for node in nodes]
            width = len(NUMERIC_FIELDS[node_type]) + len(PLACE_VALUES[node_type])
            assignment_graph[node_type].x = (
                torch.tensor(rows, dtype=torch.float32)
                if rows
                else torch.empty((0, width), dtype=torch.float32)
            )
            local_index.update({node.node_id: index for index, node in enumerate(nodes)})

        action_nodes = source.nodes_of_type("binding")
        # FJSP exposes one controlled transition (operation start).  A constant
        # feature mirrors the structural transition placeholder used by the
        # Section 5.6 AG implementation without adding problem-specific clues.
        assignment_graph[ACTION_NODE_TYPE].x = torch.ones(
            (len(action_nodes), 1), dtype=torch.float32
        )
        action_index = {
            node.node_id: index for index, node in enumerate(action_nodes)
        }

        edge_columns: Dict[Tuple[str, str, str], list[Tuple[int, int]]] = {
            edge_type: [] for edge_type in EDGE_TYPES
        }
        for edge in source.edges:
            target = source.nodes[edge.target]
            source_node = source.nodes[edge.source]
            if target.node_type != "binding" or source_node.node_type not in TOKEN_NODE_TYPES:
                continue
            token_type = source_node.node_type
            token = local_index[edge.source]
            action = action_index[edge.target]
            edge_columns[(token_type, "edge_to_a_transition", ACTION_NODE_TYPE)].append(
                (token, action)
            )
            edge_columns[(ACTION_NODE_TYPE, f"edge_to_{token_type}", token_type)].append(
                (action, token)
            )
        edge_columns[(ACTION_NODE_TYPE, "self_loop", ACTION_NODE_TYPE)] = [
            (index, index) for index in range(len(action_nodes))
        ]
        for edge_type in EDGE_TYPES:
            columns = edge_columns[edge_type]
            assignment_graph[edge_type].edge_index = (
                torch.tensor(columns, dtype=torch.long).t().contiguous()
                if columns
                else torch.empty((2, 0), dtype=torch.long)
            )

        graph = TokenBindingGraph(
            time=source.time,
            metadata={
                **source.metadata,
                "observation": "fjsp_assignment_graph",
                "adapter": type(self).__name__,
                "representation": "token_action_incidence_only",
                "reference_edges_visible": False,
                "binding_role_labels_visible": False,
            },
        )
        for action_node in action_nodes:
            graph.add_node(GraphNode(
                node_id=action_node.node_id,
                node_type="binding",
                identity=dict(action_node.identity),
            ))
            graph.action_keys[action_node.node_id] = source.action_keys[
                action_node.node_id
            ]
        graph.assignment_graph = assignment_graph
        graph.assignment_action_indices = tuple(range(len(action_nodes)))
        if self.validate:
            graph.validate()
        return graph


@dataclass(frozen=True)
class FJSPAssignmentGraphNoOpNormalizer:
    """Compatibility object; standardization is stored in the AG schema."""

    format: str = "fjsp_assignment_graph_schema_normalization_v1"

    @classmethod
    def fit(
        cls,
        _graphs: Iterable[TokenBindingGraph],
        _schema: AssignmentGraphFeatureSchema,
    ) -> "FJSPAssignmentGraphNoOpNormalizer":
        return cls()

    def to_dict(self) -> Dict[str, str]:
        return {"format": self.format}

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        _schema: AssignmentGraphFeatureSchema,
    ) -> "FJSPAssignmentGraphNoOpNormalizer":
        observed = str(values.get("format", ""))
        if observed != cls().format:
            raise ValueError(
                f"Unsupported FJSP Assignment Graph normalizer: {observed}"
            )
        return cls()


class FJSPControlledAssignmentGraphPolicy(ControlledAssignmentGraphPolicy):
    """Section 5.6 HAN policy accepting the common FJSP runner arguments."""

    def __init__(
        self,
        schema: AssignmentGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 0,
        competition_layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        # Retain common protocol fields in checkpoints even though HAN does not
        # use the generic-graph category and propagation-depth parameters.
        self.category_dim = int(category_dim)
        self.reference_layers = int(reference_layers)
        self.competition_layers = int(competition_layers)
        super().__init__(schema, hidden_dim=hidden_dim, dropout=dropout)


def _node_features(node: GraphNode, node_type: str) -> list[float]:
    place = node.categorical.get("current_place")
    allowed_places = PLACE_VALUES[node_type]
    if place not in allowed_places:
        raise ValueError(
            f"Unsupported FJSP place for {node_type}: {place!r}"
        )
    numeric = [float(node.numeric[field]) for field in NUMERIC_FIELDS[node_type]]
    one_hot = [float(place == value) for value in allowed_places]
    return numeric + one_hot
