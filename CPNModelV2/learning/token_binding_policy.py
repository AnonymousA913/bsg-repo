from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from CPNModelV2.graph.tensorizer import (
    CATEGORICAL_FIELDS,
    NODE_TYPES,
    NUMERIC_FIELDS,
    GraphFeatureSchema,
    TokenBindingBatch,
)


@dataclass
class BindingPolicyOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    values: torch.Tensor


@dataclass
class BindingActionSelection:
    graph_index: int
    action_index: int
    binding_node_id: str
    binding_key: Tuple[Any, ...]
    probability: float
    log_probability: float


class BindingPolicyNetwork(nn.Module):
    """Minimal relation-aware actor-critic over dynamic Binding actions."""

    def __init__(
        self,
        schema: GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        message_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or category_dim <= 0 or message_layers < 0:
            raise ValueError("Invalid Token-Binding policy dimensions")
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.category_dim = category_dim
        self.category_embedding = nn.Embedding(
            len(schema.category_values), category_dim
        )
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.numeric_projection = nn.ModuleDict({
            node_type: nn.Linear(len(fields), hidden_dim)
            for node_type, fields in NUMERIC_FIELDS.items()
            if fields
        })
        self.categorical_projection = nn.ModuleDict({
            node_type: nn.Linear(
                len(fields) * category_dim, hidden_dim
            )
            for node_type, fields in CATEGORICAL_FIELDS.items()
            if fields
        })
        self.input_norm = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES
        })
        self.message_passing = nn.ModuleList([
            RelationalMessageLayer(
                hidden_dim,
                len(schema.relation_values),
                dropout=dropout,
            )
            for _ in range(message_layers)
        ])
        pooled_dim = len(NODE_TYPES) * hidden_dim
        self.context_projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: TokenBindingBatch) -> BindingPolicyOutput:
        hidden = self._encode_nodes(batch)
        for layer in self.message_passing:
            hidden = layer(hidden, batch)

        pooled = self._pool_by_type(hidden, batch)
        context = self.context_projection(pooled)
        binding_hidden = hidden["binding"][batch.action_binding_index]
        if binding_hidden.shape[0]:
            actor_input = torch.cat(
                [binding_hidden, context[batch.action_batch]], dim=-1
            )
            logits = self.actor(actor_input).squeeze(-1)
        else:
            logits = context.new_empty((0,))
        probabilities = segmented_softmax(
            logits, batch.action_batch, batch.num_graphs
        )
        values = self.critic(pooled).squeeze(-1)
        return BindingPolicyOutput(logits, probabilities, values)

    def _encode_nodes(
        self, batch: TokenBindingBatch
    ) -> Dict[str, torch.Tensor]:
        hidden: Dict[str, torch.Tensor] = {}
        for type_index, node_type in enumerate(NODE_TYPES):
            count = batch.node_numeric[node_type].shape[0]
            base = self.type_embedding.weight[type_index].unsqueeze(0).expand(
                count, -1
            )
            if node_type in self.numeric_projection:
                base = base + self.numeric_projection[node_type](
                    batch.node_numeric[node_type]
                )
            if node_type in self.categorical_projection:
                categorical = self.category_embedding(
                    batch.node_categorical[node_type]
                ).reshape(
                    count,
                    len(CATEGORICAL_FIELDS[node_type]) * self.category_dim,
                )
                base = base + self.categorical_projection[node_type](categorical)
            hidden[node_type] = self.input_norm[node_type](F.relu(base))
        return hidden

    @staticmethod
    def _pool_by_type(
        hidden: Dict[str, torch.Tensor], batch: TokenBindingBatch
    ) -> torch.Tensor:
        pooled_parts = []
        for node_type in NODE_TYPES:
            values = hidden[node_type]
            pooled = values.new_zeros((batch.num_graphs, values.shape[-1]))
            counts = values.new_zeros((batch.num_graphs, 1))
            if values.shape[0]:
                graph_index = batch.node_batch[node_type]
                pooled.index_add_(0, graph_index, values)
                counts.index_add_(
                    0,
                    graph_index,
                    values.new_ones((values.shape[0], 1)),
                )
            pooled_parts.append(pooled / counts.clamp_min(1.0))
        return torch.cat(pooled_parts, dim=-1)


class RelationalMessageLayer(nn.Module):
    """Shared message transform conditioned on a semantic relation embedding."""

    def __init__(
        self,
        hidden_dim: int,
        relation_count: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.source_projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES
        })

    def forward(
        self,
        hidden: Dict[str, torch.Tensor],
        batch: TokenBindingBatch,
    ) -> Dict[str, torch.Tensor]:
        sums = {
            node_type: values.new_zeros(values.shape)
            for node_type, values in hidden.items()
        }
        counts = {
            node_type: values.new_zeros((values.shape[0], 1))
            for node_type, values in hidden.items()
        }
        for canonical_type, edge_index in batch.edge_index.items():
            source_type, _, target_type = canonical_type
            if edge_index.shape[1] == 0:
                continue
            source_index, target_index = edge_index
            source = hidden[source_type][source_index]
            relation_id = batch.edge_relation_ids[canonical_type]
            relation = self.relation_embedding.weight[relation_id].unsqueeze(0)
            message = torch.tanh(self.source_projection(source) + relation)
            sums[target_type].index_add_(0, target_index, message)
            counts[target_type].index_add_(
                0,
                target_index,
                message.new_ones((message.shape[0], 1)),
            )

        updated: Dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            current = hidden[node_type]
            if current.shape[0] == 0:
                updated[node_type] = current
                continue
            aggregate = sums[node_type] / counts[node_type].clamp_min(1.0)
            proposal = self.norm[node_type](
                current + self.update(torch.cat([current, aggregate], dim=-1))
            )
            updated[node_type] = torch.where(
                counts[node_type] > 0,
                proposal,
                current,
            )
        return updated


def segmented_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    probabilities = torch.empty_like(logits)
    for graph_index in range(num_segments):
        positions = torch.nonzero(
            segment_ids == graph_index, as_tuple=False
        ).flatten()
        if positions.numel():
            probabilities[positions] = torch.softmax(logits[positions], dim=0)
    return probabilities


def select_binding_actions(
    output: BindingPolicyOutput,
    batch: TokenBindingBatch,
    *,
    deterministic: bool = False,
    generator: Optional[torch.Generator] = None,
) -> List[Optional[BindingActionSelection]]:
    selections: List[Optional[BindingActionSelection]] = []
    for graph_index in range(batch.num_graphs):
        positions = torch.nonzero(
            batch.action_batch == graph_index, as_tuple=False
        ).flatten()
        if not positions.numel():
            selections.append(None)
            continue
        local_probabilities = output.probabilities[positions]
        if deterministic:
            local_choice = int(torch.argmax(local_probabilities))
        else:
            local_choice = int(torch.multinomial(
                local_probabilities,
                num_samples=1,
                generator=generator,
            ).item())
        action_index = int(positions[local_choice])
        probability = output.probabilities[action_index].clamp_min(1e-12)
        selections.append(BindingActionSelection(
            graph_index=graph_index,
            action_index=action_index,
            binding_node_id=batch.action_node_ids[action_index],
            binding_key=batch.action_keys[action_index],
            probability=float(probability.detach().cpu()),
            log_probability=float(torch.log(probability).detach().cpu()),
        ))
    return selections
