from __future__ import annotations

import math
from typing import Dict, List, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from CPNModelV2.graph.generic_tensorizer import (
    BINDING_NODE_TYPE,
    GenericGraphFeatureSchema,
    GenericTokenBindingBatch,
)
from CPNModelV2.learning.token_binding_policy import (
    BindingPolicyOutput,
)


class GenericCompactPolicyNetwork(nn.Module):
    """Schema-driven Compact V5 policy for executable CPN bindings.

    The operator is domain independent: Token types, fields, reference
    relations and Binding roles are all supplied by the fitted graph schema.
    Empty reference sets and a single Token type are valid first-class cases.
    """

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or category_dim <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if reference_layers < 0 or competition_layers < 0:
            raise ValueError("Layer counts must be non-negative")
        if competition_layers != 0:
            raise ValueError(
                "Generic Compact V5 keeps competition_layers=0; "
                "candidate interaction belongs to the subsequent V2 method"
            )
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.category_dim = category_dim

        self.category_embedding = nn.Embedding(
            len(schema.category_values), category_dim
        )
        self.numeric_projection = nn.ModuleList([
            nn.Linear(len(schema.numeric_fields[node_type]), hidden_dim)
            if schema.numeric_fields[node_type] else nn.Identity()
            for node_type in schema.node_types
        ])
        self.categorical_projection = nn.ModuleList([
            nn.Linear(
                len(schema.categorical_fields[node_type]) * category_dim,
                hidden_dim,
            )
            if schema.categorical_fields[node_type] else nn.Identity()
            for node_type in schema.node_types
        ])
        self.input_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in schema.node_types
        ])

        self.reference_layers = nn.ModuleList([
            GenericReferenceAttentionLayer(
                schema,
                hidden_dim,
                relation_count=len(schema.relation_values),
                dropout=dropout,
            )
            for _ in range(reference_layers)
        ])
        self.binding_composer = self._make_binding_composer(
            schema, hidden_dim, dropout=dropout
        )
        self.poolers = nn.ModuleList([
            GenericAttentionPool(hidden_dim) for _ in schema.node_types
        ])
        pooled_dim = len(schema.node_types) * hidden_dim
        self.context_projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
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

    def _make_binding_composer(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        *,
        dropout: float,
    ) -> nn.Module:
        return GenericRoleConditionedBindingComposer(
            schema, hidden_dim, dropout=dropout
        )

    def forward(self, batch: GenericTokenBindingBatch) -> BindingPolicyOutput:
        hidden = self._encode_nodes(batch)
        for layer in self.reference_layers:
            hidden = layer(hidden, batch)
        hidden[BINDING_NODE_TYPE] = self.binding_composer(hidden, batch)

        pooled = torch.cat([
            self.poolers[index](
                hidden[node_type],
                batch.node_batch[node_type],
                batch.num_graphs,
            )
            for index, node_type in enumerate(self.schema.node_types)
        ], dim=-1)
        context = self.context_projection(pooled)
        binding_hidden = hidden[BINDING_NODE_TYPE][batch.action_binding_index]
        if binding_hidden.shape[0]:
            logits = self.actor(torch.cat([
                binding_hidden,
                context[batch.action_batch],
            ], dim=-1)).squeeze(-1)
        else:
            logits = context.new_empty((0,))
        probabilities = _segmented_softmax(
            logits, batch.action_batch, batch.num_graphs
        )
        values = self.critic(pooled).squeeze(-1)
        return BindingPolicyOutput(logits, probabilities, values)


    def _encode_nodes(
        self,
        batch: GenericTokenBindingBatch,
    ) -> Dict[str, torch.Tensor]:
        hidden: Dict[str, torch.Tensor] = {}
        for index, node_type in enumerate(self.schema.node_types):
            count = batch.node_numeric[node_type].shape[0]
            base = batch.graph_times.new_zeros((count, self.hidden_dim))
            if self.schema.numeric_fields[node_type]:
                base = base + self.numeric_projection[index](
                    batch.node_numeric[node_type]
                )
            if self.schema.categorical_fields[node_type]:
                categories = self.category_embedding(
                    batch.node_categorical[node_type]
                ).reshape(
                    count,
                    len(self.schema.categorical_fields[node_type])
                    * self.category_dim,
                )
                base = base + self.categorical_projection[index](categories)
            hidden[node_type] = self.input_norm[index](F.relu(base))
        return hidden


BSG_RL_SMALL_HIDDEN_DIM = 16
BSG_RL_SMALL_CATEGORY_DIM = 16
BSG_RL_SMALL_REFERENCE_LAYERS = 2
BSG_RL_SMALL_COMPETITION_LAYERS = 0


class BSGRLSmallPolicyNetwork(GenericCompactPolicyNetwork):
    """Capacity-matched BSG-RL used by the Section 5.4 control.

    The semantic operators are identical to the full BSG-RL policy.  Only the
    latent width is reduced, keeping reference propagation depth and
    role-conditioned Binding composition unchanged.  The fixed configuration
    prevents a checkpoint labelled BSG-RL-S from silently using another
    capacity budget.
    """

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        *,
        hidden_dim: int = BSG_RL_SMALL_HIDDEN_DIM,
        category_dim: int = BSG_RL_SMALL_CATEGORY_DIM,
        reference_layers: int = BSG_RL_SMALL_REFERENCE_LAYERS,
        competition_layers: int = BSG_RL_SMALL_COMPETITION_LAYERS,
        dropout: float = 0.0,
    ) -> None:
        observed = (
            int(hidden_dim),
            int(category_dim),
            int(reference_layers),
            int(competition_layers),
        )
        expected = (
            BSG_RL_SMALL_HIDDEN_DIM,
            BSG_RL_SMALL_CATEGORY_DIM,
            BSG_RL_SMALL_REFERENCE_LAYERS,
            BSG_RL_SMALL_COMPETITION_LAYERS,
        )
        if observed != expected:
            raise ValueError(
                "BSG-RL-S uses the fixed capacity-control configuration "
                f"hidden/category/reference/competition={expected}, got "
                f"{observed}"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=competition_layers,
            dropout=dropout,
        )


class GenericReferenceAttentionLayer(nn.Module):
    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        relation_count: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_key = nn.Embedding(relation_count, hidden_dim)
        self.relation_value = nn.Embedding(relation_count, hidden_dim)
        self.direction_key = nn.Embedding(2, hidden_dim)
        self.direction_value = nn.Embedding(2, hidden_dim)
        self.update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in schema.token_types
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in schema.token_types
        ])

    def forward(
        self,
        hidden: Dict[str, torch.Tensor],
        batch: GenericTokenBindingBatch,
    ) -> Dict[str, torch.Tensor]:
        incoming: Dict[
            str,
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ] = {
            node_type: []
            for node_type in self.schema.token_types
        }
        for canonical, edge_index in batch.reference_edge_index.items():
            source_type, _relation, target_type = canonical
            source_index, target_index = edge_index
            relation_id = batch.relation_ids[canonical]
            self._collect_direction(
                incoming[target_type],
                hidden[source_type],
                hidden[target_type],
                source_index,
                target_index,
                relation_id,
                0,
            )
            self._collect_direction(
                incoming[source_type],
                hidden[target_type],
                hidden[source_type],
                target_index,
                source_index,
                relation_id,
                1,
            )

        updated = dict(hidden)
        for index, node_type in enumerate(self.schema.token_types):
            current = hidden[node_type]
            if not incoming[node_type] or current.shape[0] == 0:
                continue
            targets = torch.cat([item[0] for item in incoming[node_type]])
            scores = torch.cat([item[1] for item in incoming[node_type]])
            messages = torch.cat([item[2] for item in incoming[node_type]])
            weights = _segmented_softmax(scores, targets, current.shape[0])
            aggregate = current.new_zeros(current.shape)
            aggregate.index_add_(0, targets, weights.unsqueeze(-1) * messages)
            proposal = self.update[index](torch.cat([
                current, aggregate
            ], dim=-1))
            updated[node_type] = self.norm[index](current + proposal)
        return updated

    def _collect_direction(
        self,
        destination: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        source: torch.Tensor,
        target: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        relation_id: int,
        direction: int,
    ) -> None:
        if source_index.numel() == 0:
            return
        relation_key = self.relation_key.weight[relation_id]
        relation_value = self.relation_value.weight[relation_id]
        direction_key = self.direction_key.weight[direction]
        direction_value = self.direction_value.weight[direction]
        score = (
            self.query(target[target_index])
            * (self.key(source[source_index]) + relation_key + direction_key)
        ).sum(-1) / math.sqrt(self.hidden_dim)
        message = (
            self.value(source[source_index]) + relation_value + direction_value
        )
        destination.append((target_index, score, message))


class GenericRoleConditionedBindingComposer(nn.Module):
    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.role_to_index = {
            role: index for index, role in enumerate(schema.participation_roles)
        }
        self.fusion = nn.Sequential(
            nn.Linear(
                (1 + len(schema.participation_roles)) * hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: GenericTokenBindingBatch,
    ) -> torch.Tensor:
        binding = hidden[BINDING_NODE_TYPE]
        channels = [binding.new_zeros(binding.shape) for _ in self.role_to_index]
        counts = [
            binding.new_zeros((binding.shape[0], 1)) for _ in self.role_to_index
        ]
        for canonical, edge_index in batch.participation_edge_index.items():
            source_type, role, target_type = canonical
            if target_type != BINDING_NODE_TYPE:
                continue
            role_index = self.role_to_index[role]
            token_index, binding_index = edge_index
            channels[role_index].index_add_(
                0, binding_index, hidden[source_type][token_index]
            )
            counts[role_index].index_add_(
                0,
                binding_index,
                binding.new_ones((binding_index.shape[0], 1)),
            )
        normalized = [
            channel / count.clamp_min(1.0)
            for channel, count in zip(channels, counts)
        ]
        proposal = self.fusion(torch.cat([binding, *normalized], dim=-1))
        return self.norm(binding + proposal)


class GenericAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if values.shape[0] == 0:
            return values.new_zeros((num_graphs, values.shape[-1]))
        weights = _segmented_softmax(
            self.score(values).squeeze(-1), graph_index, num_graphs
        )
        pooled = values.new_zeros((num_graphs, values.shape[-1]))
        pooled.index_add_(0, graph_index, weights.unsqueeze(-1) * values)
        return pooled


def _segmented_softmax(
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Vectorized softmax over variable graph, Token or Binding groups."""
    if scores.numel() == 0:
        return scores
    maxima = scores.new_full((num_segments,), -torch.inf)
    maxima.scatter_reduce_(
        0, segment_ids, scores, reduce="amax", include_self=True
    )
    exponentials = torch.exp(scores - maxima[segment_ids])
    denominators = scores.new_zeros((num_segments,))
    denominators.scatter_add_(0, segment_ids, exponentials)
    return exponentials / denominators[segment_ids].clamp_min(1e-12)
