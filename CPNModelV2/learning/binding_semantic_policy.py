from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from CPNModelV2.graph.v5_tensorizer import (
    V5_CATEGORICAL_FIELDS,
    V5_NODE_TYPES,
    V5_NUMERIC_FIELDS,
    V5GraphFeatureSchema,
    V5TokenBindingBatch,
)
from CPNModelV2.learning.token_binding_policy import (
    BindingPolicyOutput,
    segmented_softmax,
)


BINDING_ROLE_SLOTS: Tuple[str, ...] = (
    "binding_consumes_operation",
    "binding_consumes_machine",
    "binding_reads_eligibility",
    "binding_reads_precedence",
)

TOKEN_NODE_TYPES: Tuple[str, ...] = (
    "operation",
    "machine",
    "eligibility",
    "precedence",
)

TOKEN_BINDING_INTERACTION_MODES: Tuple[str, ...] = (
    "consume",
    "read",
    "other",
)


def binding_participation_mode(relation: str) -> str:
    """Map a Token--Binding relation to its CPN participation semantics."""
    if relation.startswith("binding_consumes_"):
        return "consume"
    if relation.startswith("binding_reads_"):
        return "read"
    return "other"


class BindingSemanticPolicyNetwork(nn.Module):
    """Binding-centric actor-critic for field-decoupled V5 CPN graphs.

    The encoder follows three explicit semantic stages:

    1. typed Token attributes and Place-state categories are encoded;
    2. identity references are used only through resolved semantic edges;
    3. role-conditioned Bindings interact through shared Token instances.
    """

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or category_dim <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if reference_layers < 0 or competition_layers < 0:
            raise ValueError("Layer counts must be non-negative")
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.category_dim = category_dim

        self.type_embedding = nn.Embedding(len(V5_NODE_TYPES), hidden_dim)
        self.category_embedding = nn.Embedding(
            len(schema.category_values), category_dim
        )
        self.numeric_projection = nn.ModuleDict({
            node_type: nn.Linear(len(fields), hidden_dim)
            for node_type, fields in V5_NUMERIC_FIELDS.items()
            if fields
        })
        self.categorical_projection = nn.ModuleDict({
            node_type: nn.Linear(len(fields) * category_dim, hidden_dim)
            for node_type, fields in V5_CATEGORICAL_FIELDS.items()
            if fields
        })
        self.input_norm = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in V5_NODE_TYPES
        })

        self.reference_layers = nn.ModuleList([
            ReferenceSemanticAttentionLayer(
                hidden_dim,
                relation_count=len(schema.relation_values),
                dropout=dropout,
            )
            for _ in range(reference_layers)
        ])
        self.binding_composer = RoleConditionedBindingComposer(
            hidden_dim,
            relation_count=len(schema.relation_values),
            dropout=dropout,
        )
        self.competition_layers = nn.ModuleList([
            SharedTokenCompetitionLayer(hidden_dim, dropout=dropout)
            for _ in range(competition_layers)
        ])

        self.poolers = nn.ModuleDict({
            node_type: SemanticAttentionPool(hidden_dim)
            for node_type in V5_NODE_TYPES
        })
        pooled_dim = len(V5_NODE_TYPES) * hidden_dim
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

    def forward(self, batch: V5TokenBindingBatch) -> BindingPolicyOutput:
        hidden = self._encode_nodes(batch)
        for layer in self.reference_layers:
            hidden = layer(hidden, batch)

        hidden["binding"] = self.binding_composer(hidden, batch)
        for layer in self.competition_layers:
            hidden["binding"] = layer(hidden, batch)

        pooled = self._pool(hidden, batch)
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
        self, batch: V5TokenBindingBatch
    ) -> Dict[str, torch.Tensor]:
        hidden: Dict[str, torch.Tensor] = {}
        for type_index, node_type in enumerate(V5_NODE_TYPES):
            count = batch.node_numeric[node_type].shape[0]
            base = self.type_embedding.weight[type_index].unsqueeze(0).expand(
                count, -1
            )
            if node_type in self.numeric_projection:
                base = base + self.numeric_projection[node_type](
                    batch.node_numeric[node_type]
                )
            if node_type in self.categorical_projection:
                categories = self.category_embedding(
                    batch.node_categorical[node_type]
                ).reshape(
                    count,
                    len(V5_CATEGORICAL_FIELDS[node_type]) * self.category_dim,
                )
                base = base + self.categorical_projection[node_type](categories)
            hidden[node_type] = self.input_norm[node_type](F.relu(base))
        return hidden

    def _pool(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: V5TokenBindingBatch,
    ) -> torch.Tensor:
        return torch.cat([
            self.poolers[node_type](
                hidden[node_type],
                batch.node_batch[node_type],
                batch.num_graphs,
            )
            for node_type in V5_NODE_TYPES
        ], dim=-1)


class ReferenceSemanticAttentionLayer(nn.Module):
    """Propagate Token attributes along topology resolved from identity fields."""

    def __init__(
        self,
        hidden_dim: int,
        relation_count: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_key = nn.Embedding(relation_count, hidden_dim)
        self.relation_value = nn.Embedding(relation_count, hidden_dim)
        self.direction_key = nn.Embedding(2, hidden_dim)
        self.direction_value = nn.Embedding(2, hidden_dim)
        self.update = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for node_type in TOKEN_NODE_TYPES
        })
        self.norm = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in TOKEN_NODE_TYPES
        })

    def forward(
        self,
        hidden: Dict[str, torch.Tensor],
        batch: V5TokenBindingBatch,
    ) -> Dict[str, torch.Tensor]:
        incoming: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            node_type: [] for node_type in TOKEN_NODE_TYPES
        }
        for canonical, edge_index in batch.reference_edge_index.items():
            source_type, _, target_type = canonical
            relation_id = batch.relation_ids[canonical]
            self._collect_direction(
                incoming[target_type],
                hidden[source_type],
                hidden[target_type],
                edge_index[0],
                edge_index[1],
                relation_id,
                direction=0,
            )
            self._collect_direction(
                incoming[source_type],
                hidden[target_type],
                hidden[source_type],
                edge_index[1],
                edge_index[0],
                relation_id,
                direction=1,
            )

        updated = dict(hidden)
        for node_type in TOKEN_NODE_TYPES:
            current = hidden[node_type]
            if not incoming[node_type] or current.shape[0] == 0:
                continue
            targets = torch.cat([item[0] for item in incoming[node_type]])
            scores = torch.cat([item[1] for item in incoming[node_type]])
            messages = torch.cat([item[2] for item in incoming[node_type]])
            weights = _segmented_edge_softmax(scores, targets, current.shape[0])
            aggregate = current.new_zeros(current.shape)
            aggregate.index_add_(0, targets, weights.unsqueeze(-1) * messages)
            proposal = self.update[node_type](
                torch.cat([current, aggregate], dim=-1)
            )
            updated[node_type] = self.norm[node_type](current + proposal)
        return updated

    def _collect_direction(
        self,
        destination: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        relation_id: int,
        *,
        direction: int,
    ) -> None:
        relation_key = self.relation_key.weight[relation_id]
        relation_value = self.relation_value.weight[relation_id]
        direction_key = self.direction_key.weight[direction]
        direction_value = self.direction_value.weight[direction]
        source = source_hidden[source_index]
        target = target_hidden[target_index]
        key = self.key(source) + relation_key + direction_key
        query = self.query(target)
        score = (query * key).sum(-1) / math.sqrt(self.hidden_dim)
        message = self.value(source) + relation_value + direction_value
        destination.append((target_index, score, message))


class ReferenceFreeTokenUpdateLayer(nn.Module):
    """Capacity control that updates each Token without cross-Token messages."""

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for node_type in TOKEN_NODE_TYPES
        })
        self.norm = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in TOKEN_NODE_TYPES
        })

    def forward(
        self,
        hidden: Dict[str, torch.Tensor],
        batch: V5TokenBindingBatch,
    ) -> Dict[str, torch.Tensor]:
        del batch
        updated = dict(hidden)
        for node_type in TOKEN_NODE_TYPES:
            current = hidden[node_type]
            if current.shape[0] == 0:
                continue
            gate = torch.sigmoid(
                (self.query(current) * self.key(current)).sum(-1, keepdim=True)
                / math.sqrt(self.hidden_dim)
            )
            aggregate = gate * self.value(current)
            proposal = self.update[node_type](
                torch.cat([current, aggregate], dim=-1)
            )
            updated[node_type] = self.norm[node_type](current + proposal)
        return updated


class RoleConditionedBindingComposer(nn.Module):
    """Compose one Binding from participant Tokens without copying attributes."""

    def __init__(
        self,
        hidden_dim: int,
        relation_count: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.role_projection = nn.ModuleDict({
            role: nn.Linear(hidden_dim, hidden_dim)
            for role in BINDING_ROLE_SLOTS
        })
        self.role_embedding = nn.Embedding(relation_count, hidden_dim)
        self.role_gate = nn.ModuleDict({
            role: nn.Linear(2 * hidden_dim, 1)
            for role in BINDING_ROLE_SLOTS
        })
        self.fusion = nn.Sequential(
            nn.Linear(
                (1 + len(BINDING_ROLE_SLOTS)) * hidden_dim,
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
        batch: V5TokenBindingBatch,
    ) -> torch.Tensor:
        binding = hidden["binding"]
        role_summaries = {
            role: binding.new_zeros(binding.shape)
            for role in BINDING_ROLE_SLOTS
        }
        role_weights = {
            role: binding.new_zeros((binding.shape[0], 1))
            for role in BINDING_ROLE_SLOTS
        }
        for canonical, edge_index in batch.binding_role_edge_index.items():
            source_type, relation, target_type = canonical
            if target_type != "binding" or relation not in role_summaries:
                continue
            source_index, binding_index = edge_index
            source = hidden[source_type][source_index]
            target = binding[binding_index]
            relation_context = self.role_embedding.weight[
                batch.relation_ids[canonical]
            ].unsqueeze(0)
            transformed = self.role_projection[relation](source) + relation_context
            gate = torch.sigmoid(
                self.role_gate[relation](torch.cat([transformed, target], dim=-1))
            )
            role_summaries[relation].index_add_(
                0, binding_index, gate * transformed
            )
            role_weights[relation].index_add_(0, binding_index, gate)

        normalized = [
            role_summaries[role] / role_weights[role].clamp_min(1e-8)
            for role in BINDING_ROLE_SLOTS
        ]
        proposal = self.fusion(torch.cat([binding, *normalized], dim=-1))
        return self.norm(binding + proposal)


class RoleAgnosticBindingComposer(nn.Module):
    """Compose a Binding without distinguishing participation-arc roles.

    Token types retain their own encoders, but all Token--Binding participation
    edges share one projection and one gate. This forms a controlled ablation
    of CPN arc-role semantics while preserving the same participant set.
    """

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.participant_projection = nn.Linear(hidden_dim, hidden_dim)
        self.participant_gate = nn.Linear(2 * hidden_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, 3 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: V5TokenBindingBatch,
    ) -> torch.Tensor:
        binding = hidden["binding"]
        participant_sum = binding.new_zeros(binding.shape)
        participant_weight = binding.new_zeros((binding.shape[0], 1))

        for canonical, edge_index in batch.binding_role_edge_index.items():
            source_type, _, target_type = canonical
            if target_type != "binding" or edge_index.shape[1] == 0:
                continue
            source_index, binding_index = edge_index
            source = self.participant_projection(
                hidden[source_type][source_index]
            )
            target = binding[binding_index]
            gate = torch.sigmoid(
                self.participant_gate(torch.cat([source, target], dim=-1))
            )
            participant_sum.index_add_(0, binding_index, gate * source)
            participant_weight.index_add_(0, binding_index, gate)

        participant = participant_sum / participant_weight.clamp_min(1e-8)
        proposal = self.fusion(torch.cat([binding, participant], dim=-1))
        return self.norm(binding + proposal)


class SharedTokenCompetitionLayer(nn.Module):
    """Pass Binding messages through shared Token instances.

    For each participant Token, a Binding attends only to *other* Bindings that
    use that same concrete object. Consequently, a shared Operation models
    alternative resource assignments, while a shared Machine models resource
    competition. A participant used by only one Binding contributes no
    competition message.
    """

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.ModuleDict({
            role: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for role in BINDING_ROLE_SLOTS
        })
        self.key = nn.ModuleDict({
            role: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for role in BINDING_ROLE_SLOTS
        })
        self.value = nn.ModuleDict({
            role: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for role in BINDING_ROLE_SLOTS
        })
        self.token_context = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for node_type in TOKEN_NODE_TYPES
        })
        self.fusion = nn.Sequential(
            nn.Linear(
                (1 + len(BINDING_ROLE_SLOTS)) * hidden_dim,
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
        batch: V5TokenBindingBatch,
    ) -> torch.Tensor:
        binding = hidden["binding"]
        channels = {
            role: binding.new_zeros(binding.shape)
            for role in BINDING_ROLE_SLOTS
        }
        channel_counts = {
            role: binding.new_zeros((binding.shape[0], 1))
            for role in BINDING_ROLE_SLOTS
        }

        for canonical, edge_index in batch.binding_role_edge_index.items():
            token_type, relation, target_type = canonical
            if target_type != "binding" or relation not in channels:
                continue
            token_index, binding_index = edge_index
            for token_position in torch.unique(token_index).tolist():
                incident = torch.nonzero(
                    token_index == token_position, as_tuple=False
                ).flatten()
                if incident.numel() < 2:
                    continue
                incident_bindings = binding_index[incident]
                token_state = self.token_context[token_type](
                    hidden[token_type][token_position]
                )
                for receiver_position in range(incident.numel()):
                    receiver = incident_bindings[receiver_position]
                    others_mask = torch.arange(
                        incident.numel(), device=incident.device
                    ) != receiver_position
                    others = incident_bindings[others_mask]
                    query = self.query[relation](binding[receiver]) + token_state
                    keys = self.key[relation](binding[others]) + token_state
                    scores = (keys * query.unsqueeze(0)).sum(-1) / math.sqrt(
                        self.hidden_dim
                    )
                    attention = torch.softmax(scores, dim=0)
                    message = torch.sum(
                        attention.unsqueeze(-1)
                        * self.value[relation](binding[others]),
                        dim=0,
                    )
                    channels[relation][receiver] = (
                        channels[relation][receiver] + message
                    )
                    channel_counts[relation][receiver] = (
                        channel_counts[relation][receiver] + 1.0
                    )

        normalized = [
            channels[role] / channel_counts[role].clamp_min(1.0)
            for role in BINDING_ROLE_SLOTS
        ]
        proposal = self.fusion(torch.cat([binding, *normalized], dim=-1))
        has_competition = sum(channel_counts.values()) > 0
        updated = self.norm(binding + proposal)
        return torch.where(has_competition, updated, binding)


class VectorizedSharedTokenInteractionLayer(nn.Module):
    """Exchange Binding messages through shared concrete Token instances.

    The operator is independent of application-level Token names. For every
    Token--Binding participation relation, it forms groups of Bindings incident
    to the same concrete Token. Each Binding receives the mean representation
    of its peers, excluding itself, together with the shared Token state and the
    participation-relation embedding. The implementation is linear in the
    number of participation edges and contains no per-Token Python loop.
    """

    def __init__(
        self,
        hidden_dim: int,
        relation_count: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
        self.interaction = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: V5TokenBindingBatch,
    ) -> torch.Tensor:
        binding = hidden["binding"]
        message_sum = binding.new_zeros(binding.shape)
        gate_sum = binding.new_zeros((binding.shape[0], 1))

        for canonical, edge_index in batch.binding_role_edge_index.items():
            token_type, _relation, target_type = canonical
            if target_type != "binding" or edge_index.shape[1] == 0:
                continue
            token_index, binding_index = edge_index
            token_count = hidden[token_type].shape[0]

            group_sum = binding.new_zeros((token_count, self.hidden_dim))
            group_sum.index_add_(0, token_index, binding[binding_index])
            group_count = binding.new_zeros((token_count, 1))
            group_count.index_add_(
                0,
                token_index,
                binding.new_ones((token_index.shape[0], 1)),
            )

            peer_count = group_count[token_index] - 1.0
            shared_mask = peer_count.squeeze(-1) > 0
            if not torch.any(shared_mask):
                continue

            receiver_index = binding_index[shared_mask]
            receiver = binding[receiver_index]
            peer_mean = (
                group_sum[token_index[shared_mask]] - receiver
            ) / peer_count[shared_mask]
            token_state = hidden[token_type][token_index[shared_mask]]
            relation_state = self.relation_embedding.weight[
                batch.relation_ids[canonical]
            ].unsqueeze(0).expand(receiver.shape[0], -1)
            relative = receiver - peer_mean
            interaction = self.interaction(torch.cat(
                [receiver, peer_mean, relative, token_state, relation_state],
                dim=-1,
            ))
            gate = self.gate(torch.cat(
                [receiver, peer_mean, token_state, relation_state], dim=-1
            ))
            message_sum.index_add_(0, receiver_index, gate * interaction)
            gate_sum.index_add_(0, receiver_index, gate)

        has_interaction = gate_sum.squeeze(-1) > 0
        peer_context = message_sum / gate_sum.clamp_min(1e-8)
        proposal = self.fusion(torch.cat([binding, peer_context], dim=-1))
        updated = self.norm(binding + proposal)
        return torch.where(has_interaction.unsqueeze(-1), updated, binding)


class RoleAwareTokenBindingAttentionLayer(nn.Module):
    """Attend over Bindings that participate through the same concrete Token.

    A Token-conditioned segmented attention preserves the most relevant group
    member instead of reducing all incident Bindings to an unweighted mean.
    Participation modes are kept in separate channels, and logarithmic group
    cardinality retains scale information lost by normalized aggregation. The
    zero-initialized gated residual starts as an identity mapping so interaction
    is adopted only when it improves the policy objective.
    """

    def __init__(
        self,
        hidden_dim: int,
        relation_count: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
        self.token_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.token_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relative_context = nn.Linear(hidden_dim, hidden_dim, bias=False)

        fusion_width = (
            (1 + len(TOKEN_BINDING_INTERACTION_MODES)) * hidden_dim
            + len(TOKEN_BINDING_INTERACTION_MODES)
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_gate = nn.Linear(fusion_width, 1)

        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.constant_(self.update_gate.bias, -2.0)

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: V5TokenBindingBatch,
    ) -> torch.Tensor:
        binding = hidden["binding"]
        channel_sum = {
            mode: binding.new_zeros(binding.shape)
            for mode in TOKEN_BINDING_INTERACTION_MODES
        }
        channel_count = {
            mode: binding.new_zeros((binding.shape[0], 1))
            for mode in TOKEN_BINDING_INTERACTION_MODES
        }
        cardinality_sum = {
            mode: binding.new_zeros((binding.shape[0], 1))
            for mode in TOKEN_BINDING_INTERACTION_MODES
        }

        for canonical, edge_index in batch.binding_role_edge_index.items():
            token_type, relation, target_type = canonical
            if target_type != "binding" or edge_index.shape[1] == 0:
                continue
            token_index, binding_index = edge_index
            token_count = hidden[token_type].shape[0]
            group_count = binding.new_zeros((token_count, 1))
            group_count.index_add_(
                0,
                token_index,
                binding.new_ones((token_index.shape[0], 1)),
            )
            peer_count = group_count[token_index] - 1.0
            shared_mask = peer_count.squeeze(-1) > 0
            if not torch.any(shared_mask):
                continue

            relation_state = self.relation_embedding.weight[
                batch.relation_ids[canonical]
            ]
            token_state = hidden[token_type][token_index]
            query = self.token_query(token_state) + relation_state.unsqueeze(0)
            score = (query * binding[binding_index]).sum(-1) / math.sqrt(
                self.hidden_dim
            )
            attention = _segmented_edge_softmax(
                score, token_index, token_count
            )
            group_context = binding.new_zeros((token_count, self.hidden_dim))
            group_context.index_add_(
                0,
                token_index,
                attention.unsqueeze(-1) * binding[binding_index],
            )

            receiver_index = binding_index[shared_mask]
            receiver = binding[receiver_index]
            attended = group_context[token_index[shared_mask]]
            shared_token = token_state[shared_mask]
            message = torch.tanh(
                attended
                + self.relative_context(receiver - attended)
                + self.token_context(shared_token)
                + relation_state.unsqueeze(0)
            )
            mode = binding_participation_mode(relation)
            channel_sum[mode].index_add_(0, receiver_index, message)
            channel_count[mode].index_add_(
                0,
                receiver_index,
                binding.new_ones((receiver_index.shape[0], 1)),
            )
            cardinality_sum[mode].index_add_(
                0,
                receiver_index,
                torch.log1p(peer_count[shared_mask]),
            )

        contexts = [
            channel_sum[mode] / channel_count[mode].clamp_min(1.0)
            for mode in TOKEN_BINDING_INTERACTION_MODES
        ]
        cardinalities = [
            cardinality_sum[mode] / channel_count[mode].clamp_min(1.0)
            for mode in TOKEN_BINDING_INTERACTION_MODES
        ]
        fusion_input = torch.cat(
            [binding, *contexts, *cardinalities], dim=-1
        )
        proposal = self.fusion(fusion_input)
        gate = torch.sigmoid(self.update_gate(fusion_input))
        has_interaction = sum(channel_count.values()).squeeze(-1) > 0
        updated = binding + gate * proposal
        return torch.where(has_interaction.unsqueeze(-1), updated, binding)


class SemanticAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled = values.new_zeros((num_graphs, values.shape[-1]))
        if values.shape[0] == 0:
            return pooled
        scores = self.score(values).squeeze(-1)
        weights = _segmented_edge_softmax(scores, graph_index, num_graphs)
        pooled.index_add_(0, graph_index, weights.unsqueeze(-1) * values)
        return pooled


def _segmented_edge_softmax(
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
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


class CompactBindingSemanticPolicyNetwork(BindingSemanticPolicyNetwork):
    """Lean V5 policy retaining reference propagation and Binding composition.

    Node type is represented by separate type-specific encoders, so the extra
    shared type embedding is omitted. Shared-Token competition remains an
    optional ablation controlled by ``competition_layers``.
    """

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=competition_layers,
            dropout=dropout,
        )
        del self.type_embedding

    def _encode_nodes(
        self, batch: V5TokenBindingBatch
    ) -> Dict[str, torch.Tensor]:
        hidden: Dict[str, torch.Tensor] = {}
        for node_type in V5_NODE_TYPES:
            count = batch.node_numeric[node_type].shape[0]
            base = batch.node_numeric[node_type].new_zeros((count, self.hidden_dim))
            if node_type in self.numeric_projection:
                base = base + self.numeric_projection[node_type](
                    batch.node_numeric[node_type]
                )
            if node_type in self.categorical_projection:
                categories = self.category_embedding(
                    batch.node_categorical[node_type]
                ).reshape(
                    count,
                    len(V5_CATEGORICAL_FIELDS[node_type]) * self.category_dim,
                )
                base = base + self.categorical_projection[node_type](categories)
            hidden[node_type] = self.input_norm[node_type](F.relu(base))
        return hidden


class RoleAgnosticBindingPolicyNetwork(CompactBindingSemanticPolicyNetwork):
    """Compact V5 ablation with one shared Token-participation channel."""

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        if competition_layers != 0:
            raise ValueError(
                "RoleAgnosticBindingPolicyNetwork does not use competition "
                "layers"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=0,
            dropout=dropout,
        )
        self.binding_composer = RoleAgnosticBindingComposer(
            hidden_dim, dropout=dropout
        )


class ReferenceFreeTokenPolicyNetwork(CompactBindingSemanticPolicyNetwork):
    """Capacity-matched ablation without Token-reference propagation."""

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        if competition_layers != 0:
            raise ValueError(
                "ReferenceFreeTokenPolicyNetwork does not use competition layers"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=0,
            competition_layers=0,
            dropout=dropout,
        )
        self.reference_layers = nn.ModuleList([
            ReferenceFreeTokenUpdateLayer(hidden_dim, dropout=dropout)
            for _ in range(reference_layers)
        ])


class LocalActorContextPolicyNetwork(CompactBindingSemanticPolicyNetwork):
    """Compact V5 ablation whose Actor sees no pooled graph context.

    Reference propagation and role-conditioned Binding composition are kept.
    Only the state-wide context concatenated to each action representation is
    removed. The Critic still observes the pooled graph state.
    """

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        if competition_layers != 0:
            raise ValueError(
                "LocalActorContextPolicyNetwork does not use competition layers"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=0,
            dropout=dropout,
        )
        del self.context_projection
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.ReLU(),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: V5TokenBindingBatch) -> BindingPolicyOutput:
        hidden = self._encode_nodes(batch)
        for layer in self.reference_layers:
            hidden = layer(hidden, batch)

        hidden["binding"] = self.binding_composer(hidden, batch)
        pooled = self._pool(hidden, batch)
        binding_hidden = hidden["binding"][batch.action_binding_index]
        if binding_hidden.shape[0]:
            logits = self.actor(binding_hidden).squeeze(-1)
        else:
            logits = pooled.new_empty((0,))
        probabilities = segmented_softmax(
            logits, batch.action_batch, batch.num_graphs
        )
        values = self.critic(pooled).squeeze(-1)
        return BindingPolicyOutput(logits, probabilities, values)


class SharedTokenInteractionPolicyNetwork(CompactBindingSemanticPolicyNetwork):
    """Compact policy with application-agnostic shared-Token interaction."""

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        if competition_layers <= 0:
            raise ValueError(
                "SharedTokenInteractionPolicyNetwork requires at least one "
                "interaction layer"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=0,
            dropout=dropout,
        )
        self.competition_layers = nn.ModuleList([
            VectorizedSharedTokenInteractionLayer(
                hidden_dim,
                relation_count=len(schema.relation_values),
                dropout=dropout,
            )
            for _ in range(competition_layers)
        ])


class RoleAwareBindingAttentionPolicyNetwork(CompactBindingSemanticPolicyNetwork):
    """Compact policy with role-aware Token-mediated Binding attention."""

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        if competition_layers <= 0:
            raise ValueError(
                "RoleAwareBindingAttentionPolicyNetwork requires at least "
                "one interaction layer"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=0,
            dropout=dropout,
        )
        self.competition_layers = nn.ModuleList([
            RoleAwareTokenBindingAttentionLayer(
                hidden_dim,
                relation_count=len(schema.relation_values),
                dropout=dropout,
            )
            for _ in range(competition_layers)
        ])
