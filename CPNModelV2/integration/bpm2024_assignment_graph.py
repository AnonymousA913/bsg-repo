from __future__ import annotations

"""Controlled Assignment Graph observation, tensorizer and actor-critic."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import torch
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HANConv

from CPNModelV2.graph.schema import GraphNode, TokenBindingGraph
from CPNModelV2.learning.token_binding_policy import (
    BindingPolicyOutput,
    segmented_softmax,
)


# Kept local so the reusable Assignment Graph encoder does not import the
# BPM2024 executor and its optional Stable-Baselines dependency at import time.
BPM2024_ASSIGNMENT_ACTION_KEY = "bpm2024_assignment_binding"


Metadata = Tuple[Tuple[str, ...], Tuple[Tuple[str, str, str], ...]]


class BPM2024AssignmentGraphBuilder:
    """Preserve the paper's expanded heterogeneous Assignment Graph."""

    def __init__(self, *, validate: bool = True) -> None:
        self.validate = bool(validate)

    def build(
        self,
        problem: Any,
        bindings: Iterable[Tuple[Any, ...]],
        *,
        problem_name: str,
        decision_id: int,
    ) -> TokenBindingGraph:
        source = problem.get_observation()
        assignment_graph = source["graph"]
        source_actions = source["actions_dict"]
        current_bindings = list(bindings)
        graph = TokenBindingGraph(
            time=float(problem.clock),
            metadata={
                "observation": "bpm2024_assignment_graph",
                "adapter": type(self).__name__,
                "cpn_family": "attributed_action_evolution_petri_net",
                "problem_name": problem_name,
            },
        )
        enabled_action_indices: List[int] = []
        if not current_bindings:
            graph.assignment_graph = assignment_graph
            graph.assignment_action_indices = ()
            if self.validate:
                graph.validate()
            return graph
        for action_node_index, source_binding in source_actions.items():
            matches = [
                index
                for index, current in enumerate(current_bindings)
                if current == source_binding
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Assignment Graph action does not map to exactly one "
                    f"executor Binding: node={action_node_index}, matches={matches}"
                )
            binding_index = matches[0]
            transition = current_bindings[binding_index][0][0][0].dst
            node_id = f"binding:{int(action_node_index):06d}"
            graph.add_node(GraphNode(
                node_id=node_id,
                node_type="binding",
                identity={
                    "decision_id": int(decision_id),
                    "binding_index": int(binding_index),
                    "assignment_action_index": int(action_node_index),
                },
            ))
            graph.action_keys[node_id] = (
                BPM2024_ASSIGNMENT_ACTION_KEY,
                int(decision_id),
                int(binding_index),
                str(transition._id),
                int(action_node_index),
            )
            enabled_action_indices.append(int(action_node_index))

        # These runtime-only attributes are consumed by the AG tensorizer. The
        # common PPO still sees the standard action_keys contract.
        graph.assignment_graph = assignment_graph
        graph.assignment_action_indices = tuple(enabled_action_indices)
        if self.validate:
            graph.validate()
        return graph


@dataclass(frozen=True)
class AssignmentGraphFeatureSchema:
    node_types: Tuple[str, ...]
    edge_types: Tuple[Tuple[str, str, str], ...]
    numeric_means: Mapping[str, Tuple[float, ...]]
    numeric_scales: Mapping[str, Tuple[float, ...]]

    @classmethod
    def fit(
        cls, graphs: Iterable[TokenBindingGraph]
    ) -> "AssignmentGraphFeatureSchema":
        graph_list = list(graphs)
        if not graph_list:
            raise ValueError("Assignment Graph schema fitting needs observations")
        metadata = [_metadata(graph.assignment_graph) for graph in graph_list]
        first = metadata[0]
        if any(value != first for value in metadata[1:]):
            raise ValueError("Assignment Graph metadata changed during calibration")
        rows: Dict[str, List[torch.Tensor]] = {
            node_type: [] for node_type in first[0]
        }
        for graph in graph_list:
            source = graph.assignment_graph
            for node_type in first[0]:
                # Transition features are binary masks or empty placeholders;
                # preserve them exactly as in the published representation.
                if node_type in {"a_transition", "e_transition"}:
                    continue
                values = source[node_type].x
                if values.ndim == 2 and values.shape[0] and values.shape[1]:
                    rows[node_type].append(values.detach().float().cpu())
        means: Dict[str, Tuple[float, ...]] = {}
        scales: Dict[str, Tuple[float, ...]] = {}
        for node_type in first[0]:
            if not rows[node_type]:
                means[node_type] = ()
                scales[node_type] = ()
                continue
            values = torch.cat(rows[node_type], dim=0)
            mean = values.mean(dim=0)
            scale = values.std(dim=0, unbiased=False)
            scale = torch.where(scale >= 1e-6, scale, torch.ones_like(scale))
            means[node_type] = tuple(float(value) for value in mean)
            scales[node_type] = tuple(float(value) for value in scale)
        return cls(first[0], first[1], means, scales)

    @property
    def metadata(self) -> Metadata:
        return self.node_types, self.edge_types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_types": list(self.node_types),
            "edge_types": [list(edge_type) for edge_type in self.edge_types],
            "numeric_means": {
                key: list(value) for key, value in self.numeric_means.items()
            },
            "numeric_scales": {
                key: list(value) for key, value in self.numeric_scales.items()
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AssignmentGraphFeatureSchema":
        node_types = tuple(values["node_types"])
        return cls(
            node_types=node_types,
            edge_types=tuple(tuple(value) for value in values["edge_types"]),
            numeric_means={
                node_type: tuple(
                    float(value)
                    for value in values.get("numeric_means", {}).get(node_type, ())
                )
                for node_type in node_types
            },
            numeric_scales={
                node_type: tuple(
                    float(value)
                    for value in values.get("numeric_scales", {}).get(node_type, ())
                )
                for node_type in node_types
            },
        )


@dataclass
class AssignmentGraphBatch:
    num_graphs: int
    graph: Batch
    action_node_index: torch.Tensor
    action_batch: torch.Tensor
    action_node_ids: List[str]
    action_keys: List[Tuple[Any, ...]]
    schema: AssignmentGraphFeatureSchema

    def to(self, device: Union[str, torch.device]) -> "AssignmentGraphBatch":
        target = torch.device(device)
        return AssignmentGraphBatch(
            num_graphs=self.num_graphs,
            graph=self.graph.to(target),
            action_node_index=self.action_node_index.to(target),
            action_batch=self.action_batch.to(target),
            action_node_ids=list(self.action_node_ids),
            action_keys=list(self.action_keys),
            schema=self.schema,
        )


class AssignmentGraphTensorizer:
    def __init__(
        self,
        schema: AssignmentGraphFeatureSchema,
        *,
        normalizer: Any = None,
        validate_graphs: bool = True,
    ) -> None:
        self.schema = schema
        # AssignmentGraphFeatureSchema already stores its training-only
        # standardization statistics.  ``normalizer`` is accepted so this
        # tensorizer can use the same PPO construction path as BSG_RL without
        # applying a second normalization.
        self.normalizer = normalizer
        self.validate_graphs = bool(validate_graphs)

    def tensorize(
        self,
        graphs: Union[TokenBindingGraph, Sequence[TokenBindingGraph]],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> AssignmentGraphBatch:
        graph_list = [graphs] if isinstance(graphs, TokenBindingGraph) else list(graphs)
        if not graph_list:
            raise ValueError("At least one Assignment Graph is required")
        pyg_graphs = []
        action_node_indices: List[int] = []
        action_batches: List[int] = []
        action_node_ids: List[str] = []
        action_keys: List[Tuple[Any, ...]] = []
        action_offset = 0
        for graph_index, graph in enumerate(graph_list):
            if self.validate_graphs:
                graph.validate()
                if _metadata(graph.assignment_graph) != self.schema.metadata:
                    raise ValueError("Assignment Graph does not match fitted metadata")
            pyg_graphs.append(_normalize_graph(graph.assignment_graph, self.schema))
            local_indices = tuple(graph.assignment_action_indices)
            if len(local_indices) != len(graph.action_keys):
                raise ValueError("Assignment Graph action index count differs")
            action_node_indices.extend(
                action_offset + int(index) for index in local_indices
            )
            action_batches.extend([graph_index] * len(local_indices))
            action_node_ids.extend(graph.action_keys)
            action_keys.extend(graph.action_keys.values())
            action_offset += int(graph.assignment_graph["a_transition"].num_nodes)

        target = torch.device(device)
        batch = AssignmentGraphBatch(
            num_graphs=len(graph_list),
            graph=Batch.from_data_list(pyg_graphs).to(target),
            action_node_index=torch.tensor(
                action_node_indices, dtype=torch.long, device=target
            ),
            action_batch=torch.tensor(
                action_batches, dtype=torch.long, device=target
            ),
            action_node_ids=action_node_ids,
            action_keys=action_keys,
            schema=self.schema,
        )
        if len(batch.action_keys) != batch.action_node_index.numel():
            raise ValueError("Invalid Assignment Graph action batch")
        return batch


class ControlledAssignmentGraphPolicy(nn.Module):
    """Paper-style HAN actor with a shared-PPO compatible value head."""

    def __init__(
        self,
        schema: AssignmentGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.encoder = HANConv(
            -1,
            hidden_dim,
            heads=1,
            dropout=dropout,
            metadata=schema.metadata,
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: AssignmentGraphBatch) -> BindingPolicyOutput:
        hidden = self.encoder(batch.graph.x_dict, batch.graph.edge_index_dict)
        action_hidden = hidden["a_transition"][batch.action_node_index]
        logits = self.actor(action_hidden).squeeze(-1)
        probabilities = segmented_softmax(
            logits, batch.action_batch, batch.num_graphs
        )
        pooled = logits.new_zeros((batch.num_graphs, self.hidden_dim))
        for node_type in self.schema.node_types:
            values = hidden.get(node_type)
            if values is None or values.numel() == 0:
                continue
            owner = batch.graph[node_type].batch
            pooled.index_add_(0, owner, values)
        values = self.critic(pooled).squeeze(-1)
        return BindingPolicyOutput(logits, probabilities, values)


def _metadata(graph: HeteroData) -> Metadata:
    node_types, edge_types = graph.metadata()
    return tuple(node_types), tuple(tuple(value) for value in edge_types)


def _normalize_graph(
    graph: HeteroData,
    schema: AssignmentGraphFeatureSchema,
) -> HeteroData:
    normalized = graph.clone()
    for node_type in schema.node_types:
        values = normalized[node_type].x
        if values.ndim == 2 and values.shape[1] == 0:
            # BPM2024 encodes evolution transitions with torch.tensor([]).
            # PyG consequently creates Linear(0, hidden), whose bias can retain
            # uninitialized memory and explode after a few optimizer steps.
            # A constant structural feature carries no privileged state while
            # giving the projection a well-defined input dimension.
            normalized[node_type].x = values.new_ones((values.shape[0], 1))
            values = normalized[node_type].x
        means = schema.numeric_means.get(node_type, ())
        scales = schema.numeric_scales.get(node_type, ())
        if not means:
            continue
        if values.ndim != 2 or values.shape[1] != len(means):
            raise ValueError(
                f"Assignment Graph feature width changed for {node_type}: "
                f"expected {len(means)}, got {tuple(values.shape)}"
            )
        mean = values.new_tensor(means).unsqueeze(0)
        scale = values.new_tensor(scales).unsqueeze(0)
        normalized[node_type].x = (values - mean) / scale
    return normalized
