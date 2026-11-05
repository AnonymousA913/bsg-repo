from __future__ import annotations

import math
from typing import Dict, List, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from CPNModelV2.graph.binding_effect import (
    EFFECT_FIXED_FIELDS,
    EFFECT_SUMMARY_FIELDS,
    BindingEffectBatch,
)
from CPNModelV2.graph.generic_tensorizer import (
    BINDING_NODE_TYPE,
    GenericGraphFeatureSchema,
)
from CPNModelV2.learning.generic_compact_policy import (
    GenericAttentionPool,
    GenericReferenceAttentionLayer,
    GenericRoleConditionedBindingComposer,
    _segmented_softmax,
)
from CPNModelV2.learning.token_binding_policy import BindingPolicyOutput


class BindingEffectPolicyNetwork(nn.Module):
    """V6 policy over marking-instantiated Binding effects.

    Reference propagation contextualizes Tokens. Enabled Bindings are then
    composed from their participants and immediate firing effects. Repeated
    Binding-to-Token and Token-to-Binding updates let mutually exclusive
    candidates reason through their shared colored objects without introducing
    an FJSP-specific conflict graph.
    """

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        factor_rounds: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or category_dim <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if reference_layers < 0 or factor_rounds < 0:
            raise ValueError("Layer counts must be non-negative")
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

        self.initial_binding_composer = GenericRoleConditionedBindingComposer(
            schema, hidden_dim, dropout=dropout
        )
        self.initial_effect_composer = GenericBindingEffectComposer(
            schema,
            hidden_dim,
            category_dim,
            dropout=dropout,
        )
        self.factor_layers = nn.ModuleList([
            BidirectionalTokenBindingFactorLayer(
                schema,
                hidden_dim,
                category_dim,
                dropout=dropout,
            )
            for _ in range(factor_rounds)
        ])

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

    def forward(self, batch: BindingEffectBatch) -> BindingPolicyOutput:
        hidden = self._encode_nodes(batch)
        for layer in self.reference_layers:
            hidden = layer(hidden, batch)
        hidden[BINDING_NODE_TYPE] = self.initial_binding_composer(hidden, batch)
        hidden[BINDING_NODE_TYPE] = self.initial_effect_composer(
            hidden,
            batch,
            self.category_embedding,
        )
        for layer in self.factor_layers:
            hidden = layer(hidden, batch, self.category_embedding)

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
        batch: BindingEffectBatch,
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


class GenericBindingEffectComposer(nn.Module):
    """Compose one implicit Effect factor for every enabled Binding."""

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        category_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.role_to_index = {
            role: index for index, role in enumerate(schema.participation_roles)
        }
        self.role_embedding = nn.Embedding(
            max(1, len(schema.participation_roles)), hidden_dim
        )
        self.edge_projection = nn.ModuleList([
            nn.Linear(
                len(EFFECT_FIXED_FIELDS) + len(schema.numeric_fields[node_type]),
                hidden_dim,
            )
            for node_type in schema.token_types
        ])
        self.place_projection = nn.Linear(2 * category_dim, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.summary_projection = nn.Sequential(
            nn.Linear(len(EFFECT_SUMMARY_FIELDS), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: BindingEffectBatch,
        category_embedding: nn.Embedding,
    ) -> torch.Tensor:
        binding = hidden[BINDING_NODE_TYPE]
        targets: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        messages: List[torch.Tensor] = []
        type_to_index = {
            node_type: index for index, node_type in enumerate(self.schema.token_types)
        }
        for canonical, edge_index in batch.effect_edge_index.items():
            source_type, role, _target_type = canonical
            token_index, binding_index = edge_index
            if token_index.numel() == 0:
                continue
            numeric_context = self.edge_projection[type_to_index[source_type]](
                batch.effect_edge_numeric[canonical]
            )
            places = category_embedding(
                batch.effect_place_categorical[canonical]
            ).reshape(token_index.shape[0], -1)
            role_index = self.role_to_index.get(role, 0)
            effect_context = (
                numeric_context
                + self.place_projection(places)
                + self.role_embedding.weight[role_index]
            )
            message = self.message(torch.cat([
                hidden[source_type][token_index],
                effect_context,
            ], dim=-1))
            score = (
                self.query(binding[binding_index]) * self.key(message)
            ).sum(-1) / math.sqrt(self.hidden_dim)
            targets.append(binding_index)
            scores.append(score)
            messages.append(message)

        aggregate = binding.new_zeros(binding.shape)
        if targets:
            target_index = torch.cat(targets)
            all_scores = torch.cat(scores)
            all_messages = torch.cat(messages)
            weights = _segmented_softmax(
                all_scores, target_index, binding.shape[0]
            )
            aggregate.index_add_(
                0, target_index, weights.unsqueeze(-1) * all_messages
            )
        summary = self.summary_projection(batch.effect_summary)
        proposal = self.fusion(torch.cat([
            binding,
            aggregate,
            summary,
        ], dim=-1))
        return self.norm(binding + proposal)


class BidirectionalTokenBindingFactorLayer(nn.Module):
    """One Binding-to-Token feedback and Token-to-Binding recomposition round."""

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        category_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.role_to_index = {
            role: index for index, role in enumerate(schema.participation_roles)
        }
        self.role_embedding = nn.Embedding(
            max(1, len(schema.participation_roles)), hidden_dim
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.token_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in schema.token_types
        ])
        self.token_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in schema.token_types
        ])
        self.binding_composer = GenericRoleConditionedBindingComposer(
            schema, hidden_dim, dropout=dropout
        )
        self.effect_composer = GenericBindingEffectComposer(
            schema,
            hidden_dim,
            category_dim,
            dropout=dropout,
        )

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: BindingEffectBatch,
        category_embedding: nn.Embedding,
    ) -> Dict[str, torch.Tensor]:
        binding = hidden[BINDING_NODE_TYPE]
        incoming: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            node_type: [] for node_type in self.schema.token_types
        }
        for canonical, edge_index in batch.participation_edge_index.items():
            source_type, role, target_type = canonical
            if target_type != BINDING_NODE_TYPE:
                continue
            token_index, binding_index = edge_index
            if token_index.numel() == 0:
                continue
            role_index = self.role_to_index.get(role, 0)
            candidate = binding[binding_index] + self.role_embedding.weight[role_index]
            score = (
                self.query(hidden[source_type][token_index])
                * self.key(candidate)
            ).sum(-1) / math.sqrt(self.hidden_dim)
            message = self.value(candidate)
            incoming[source_type].append((token_index, score, message))

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
            proposal = self.token_updates[index](torch.cat([
                current,
                aggregate,
            ], dim=-1))
            updated[node_type] = self.token_norms[index](current + proposal)

        updated[BINDING_NODE_TYPE] = self.binding_composer(updated, batch)
        updated[BINDING_NODE_TYPE] = self.effect_composer(
            updated,
            batch,
            category_embedding,
        )
        return updated
