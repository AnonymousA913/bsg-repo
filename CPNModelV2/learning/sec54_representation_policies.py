from __future__ import annotations

"""Controlled representation baselines for Section 5.4.1.

All policies consume the same schema-driven Token-Binding batch and return one
categorical logit per enabled Binding.  They deliberately differ only in the
state encoder used before the shared actor-critic readout.
"""

from typing import Dict, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from CPNModelV2.graph.generic_tensorizer import (
    BINDING_NODE_TYPE,
    GenericGraphFeatureSchema,
    GenericTokenBindingBatch,
)
from CPNModelV2.learning.token_binding_policy import BindingPolicyOutput


POLICY_MASKED_MLP = "sec54_masked_mlp_petrirl_style"
POLICY_PNC_PPO = "sec54_pnc_ppo"
POLICY_PETRI_NET_GCN_PPO = "sec54_petri_net_gcn_ppo"
POLICY_AG_PPO = "sec54_assignment_graph_ppo"
POLICY_BSG_RL_SMALL = "sec54_bsg_rl_small_capacity_matched"


class _GenericNodeEncoder(nn.Module):
    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        category_dim: int,
    ) -> None:
        super().__init__()
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
        self.type_embedding = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim))
            for _ in schema.node_types
        ])
        self.input_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in schema.node_types
        ])
        for embedding in self.type_embedding:
            nn.init.normal_(embedding, std=0.02)

    def forward(
        self, batch: GenericTokenBindingBatch
    ) -> Dict[str, torch.Tensor]:
        hidden: Dict[str, torch.Tensor] = {}
        for index, node_type in enumerate(self.schema.node_types):
            count = batch.node_numeric[node_type].shape[0]
            base = self.type_embedding[index].unsqueeze(0).expand(count, -1)
            if self.schema.numeric_fields[node_type]:
                base = base + self.numeric_projection[index](
                    batch.node_numeric[node_type]
                )
            if self.schema.categorical_fields[node_type]:
                categorical = self.category_embedding(
                    batch.node_categorical[node_type]
                ).reshape(
                    count,
                    len(self.schema.categorical_fields[node_type])
                    * self.category_dim,
                )
                base = base + self.categorical_projection[index](categorical)
            hidden[node_type] = self.input_norm[index](F.relu(base))
        return hidden


class _RepresentationPolicyBase(nn.Module):
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
        if reference_layers < 0 or competition_layers != 0:
            raise ValueError(
                "Section 5.4 baselines require non-negative encoder layers "
                "and competition_layers=0"
            )
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.encoder_layers = reference_layers
        self.node_encoder = _GenericNodeEncoder(
            schema, hidden_dim, category_dim
        )
        self.context_projection = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _readout(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: GenericTokenBindingBatch,
    ) -> BindingPolicyOutput:
        pooled = _token_binding_mean_max(hidden, batch)
        context = self.context_projection(pooled)
        binding = hidden[BINDING_NODE_TYPE][batch.action_binding_index]
        if binding.shape[0]:
            logits = self.actor(torch.cat([
                binding,
                context[batch.action_batch],
            ], dim=-1)).squeeze(-1)
        else:
            logits = context.new_empty((0,))
        probabilities = _segmented_softmax(
            logits, batch.action_batch, batch.num_graphs
        )
        values = self.critic(pooled).squeeze(-1)
        return BindingPolicyOutput(logits, probabilities, values)


class MaskedMLPPolicyNetwork(_RepresentationPolicyBase):
    """PetriRL-style masked MLP adapted to variable Binding actions.

    The state is a cardinality-independent flattened mean/max summary.  Each
    candidate receives the role-agnostic mean of its participant Token
    features; there is no learned graph propagation or reference-edge use.
    """

    def __init__(self, schema: GenericGraphFeatureSchema, **kwargs: object) -> None:
        super().__init__(schema, **kwargs)
        self.candidate_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.candidate_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, batch: GenericTokenBindingBatch) -> BindingPolicyOutput:
        hidden = self.node_encoder(batch)
        binding = hidden[BINDING_NODE_TYPE]
        incident, counts = _incident_token_sum(hidden, batch)
        proposal = F.relu(self.candidate_projection(
            incident / counts.clamp_min(1.0)
        ))
        hidden[BINDING_NODE_TYPE] = torch.where(
            counts > 0,
            self.candidate_norm(binding + proposal),
            binding,
        )
        return self._readout(hidden, batch)


class PNCPPOPolicyNetwork(_RepresentationPolicyBase):
    """PNC-style bipartite propagation with a categorical PPO readout."""

    def __init__(
        self, schema: GenericGraphFeatureSchema, *, dropout: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(schema, dropout=dropout, **kwargs)
        self.layers = nn.ModuleList([
            _PNCPropagationLayer(schema, self.hidden_dim, dropout=dropout)
            for _ in range(self.encoder_layers)
        ])

    def forward(self, batch: GenericTokenBindingBatch) -> BindingPolicyOutput:
        hidden = self.node_encoder(batch)
        for layer in self.layers:
            hidden = layer(hidden, batch)
        return self._readout(hidden, batch)


class PetriNetGCNPPOPolicyNetwork(_RepresentationPolicyBase):
    """Direction- and arc-role-aware Petri-net-GCN adapted to Binding PPO."""

    def __init__(
        self, schema: GenericGraphFeatureSchema, *, dropout: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(schema, dropout=dropout, **kwargs)
        self.layers = nn.ModuleList([
            _PetriNetGCNPropagationLayer(
                schema, self.hidden_dim, dropout=dropout
            )
            for _ in range(self.encoder_layers)
        ])

    def forward(self, batch: GenericTokenBindingBatch) -> BindingPolicyOutput:
        hidden = self.node_encoder(batch)
        for layer in self.layers:
            hidden = layer(hidden, batch)
        return self._readout(hidden, batch)


class _PNCPropagationLayer(nn.Module):
    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.token_to_binding = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in schema.token_types
        ])
        self.p2t_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.h0 = nn.ParameterList([
            nn.Parameter(torch.randn(1, hidden_dim))
            for _ in schema.token_types
        ])
        self.h1 = nn.ParameterList([
            nn.Parameter(torch.randn(1, hidden_dim))
            for _ in schema.token_types
        ])
        self.token_update = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in schema.token_types
        ])
        self.dropout = nn.Dropout(dropout)
        self.binding_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in schema.token_types
        ])

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: GenericTokenBindingBatch,
    ) -> Dict[str, torch.Tensor]:
        updated = dict(hidden)
        binding = hidden[BINDING_NODE_TYPE]
        binding_sum = binding.new_zeros(binding.shape)
        binding_count = binding.new_zeros((binding.shape[0], 1))
        token_edges: Dict[str, list[Tuple[torch.Tensor, torch.Tensor]]] = {
            node_type: [] for node_type in self.schema.token_types
        }
        type_index = {
            node_type: index
            for index, node_type in enumerate(self.schema.token_types)
        }
        for canonical, edge_index in batch.participation_edge_index.items():
            source_type, _role, target_type = canonical
            if target_type != BINDING_NODE_TYPE or source_type not in type_index:
                continue
            token_index, binding_index = edge_index
            if token_index.numel() == 0:
                continue
            index = type_index[source_type]
            message = self.token_to_binding[index](
                hidden[source_type][token_index]
            )
            binding_sum.index_add_(0, binding_index, message)
            binding_count.index_add_(
                0, binding_index, message.new_ones((message.shape[0], 1))
            )
            token_edges[source_type].append((token_index, binding_index))
        binding_proposal = F.leaky_relu(binding_sum + self.p2t_bias)
        updated_binding = F.leaky_relu(self.binding_norm(
            binding + self.dropout(binding_proposal)
        ))
        updated[BINDING_NODE_TYPE] = torch.where(
            binding_count > 0, updated_binding, binding
        )

        for node_type, edge_groups in token_edges.items():
            if not edge_groups:
                continue
            index = type_index[node_type]
            token = hidden[node_type]
            token_sum = token.new_zeros(token.shape)
            token_count = token.new_zeros((token.shape[0], 1))
            for token_index, binding_index in edge_groups:
                message = updated[BINDING_NODE_TYPE][binding_index]
                token_sum.index_add_(0, token_index, message)
                token_count.index_add_(
                    0, token_index, message.new_ones((message.shape[0], 1))
                )
            token_mean = token_sum / token_count.clamp_min(1.0)
            gated = self.h0[index] * token + self.h1[index] * token_mean
            proposal = self.token_update[index](gated)
            normalized = F.leaky_relu(self.token_norm[index](
                token + self.dropout(proposal)
            ))
            updated[node_type] = torch.where(
                token_count > 0, normalized, token
            )
        return updated


class _PetriNetGCNPropagationLayer(nn.Module):
    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.role_embedding = nn.Embedding(
            len(schema.relation_values), hidden_dim
        )
        self.p2t = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in schema.token_types
        ])
        self.t2p = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in schema.token_types
        ])
        self.binding_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in schema.token_types
        ])
        self.binding_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in schema.token_types
        ])

    def forward(
        self,
        hidden: Mapping[str, torch.Tensor],
        batch: GenericTokenBindingBatch,
    ) -> Dict[str, torch.Tensor]:
        updated = dict(hidden)
        binding = hidden[BINDING_NODE_TYPE]
        binding_sum = binding.new_zeros(binding.shape)
        binding_count = binding.new_zeros((binding.shape[0], 1))
        edge_groups: Dict[
            str, list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {node_type: [] for node_type in self.schema.token_types}
        type_index = {
            node_type: index
            for index, node_type in enumerate(self.schema.token_types)
        }
        for canonical, edge_index in batch.participation_edge_index.items():
            source_type, _role, target_type = canonical
            if target_type != BINDING_NODE_TYPE or source_type not in type_index:
                continue
            token_index, binding_index = edge_index
            if token_index.numel() == 0:
                continue
            index = type_index[source_type]
            role = self.role_embedding.weight[batch.relation_ids[canonical]]
            message = torch.tanh(
                self.p2t[index](hidden[source_type][token_index]) + role
            )
            binding_sum.index_add_(0, binding_index, message)
            binding_count.index_add_(
                0, binding_index, message.new_ones((message.shape[0], 1))
            )
            edge_groups[source_type].append(
                (token_index, binding_index, role)
            )
        binding_mean = binding_sum / binding_count.clamp_min(1.0)
        binding_proposal = self.binding_update(torch.cat([
            binding, binding_mean
        ], dim=-1))
        normalized_binding = self.binding_norm(binding + binding_proposal)
        updated[BINDING_NODE_TYPE] = torch.where(
            binding_count > 0, normalized_binding, binding
        )

        for node_type, groups in edge_groups.items():
            if not groups:
                continue
            index = type_index[node_type]
            token = hidden[node_type]
            token_sum = token.new_zeros(token.shape)
            token_count = token.new_zeros((token.shape[0], 1))
            for token_index, binding_index, role in groups:
                message = torch.tanh(
                    self.t2p[index](
                        updated[BINDING_NODE_TYPE][binding_index]
                    ) + role
                )
                token_sum.index_add_(0, token_index, message)
                token_count.index_add_(
                    0, token_index, message.new_ones((message.shape[0], 1))
                )
            token_mean = token_sum / token_count.clamp_min(1.0)
            proposal = self.token_update[index](torch.cat([
                token, token_mean
            ], dim=-1))
            normalized = self.token_norm[index](token + proposal)
            updated[node_type] = torch.where(
                token_count > 0, normalized, token
            )
        return updated


def _incident_token_sum(
    hidden: Mapping[str, torch.Tensor],
    batch: GenericTokenBindingBatch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    binding = hidden[BINDING_NODE_TYPE]
    sums = binding.new_zeros(binding.shape)
    counts = binding.new_zeros((binding.shape[0], 1))
    for canonical, edge_index in batch.participation_edge_index.items():
        source_type, _role, target_type = canonical
        if target_type != BINDING_NODE_TYPE:
            continue
        token_index, binding_index = edge_index
        values = hidden[source_type][token_index]
        sums.index_add_(0, binding_index, values)
        counts.index_add_(
            0, binding_index, values.new_ones((values.shape[0], 1))
        )
    return sums, counts


def _token_binding_mean_max(
    hidden: Mapping[str, torch.Tensor],
    batch: GenericTokenBindingBatch,
) -> torch.Tensor:
    token_values = [hidden[node_type] for node_type in batch.schema.token_types]
    token_batches = [
        batch.node_batch[node_type] for node_type in batch.schema.token_types
    ]
    tokens = torch.cat(token_values, dim=0)
    token_batch = torch.cat(token_batches, dim=0)
    bindings = hidden[BINDING_NODE_TYPE]
    binding_batch = batch.node_batch[BINDING_NODE_TYPE]
    token_mean, token_max = _segment_mean_max(
        tokens, token_batch, batch.num_graphs
    )
    binding_mean, binding_max = _segment_mean_max(
        bindings, binding_batch, batch.num_graphs
    )
    return torch.cat([
        token_mean, token_max, binding_mean, binding_max
    ], dim=-1)


def _segment_mean_max(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    width = values.shape[-1]
    sums = values.new_zeros((num_segments, width))
    counts = values.new_zeros((num_segments, 1))
    maxima = values.new_full((num_segments, width), -torch.inf)
    if values.shape[0]:
        sums.index_add_(0, segment_ids, values)
        counts.index_add_(
            0, segment_ids, values.new_ones((values.shape[0], 1))
        )
        maxima.scatter_reduce_(
            0,
            segment_ids.unsqueeze(-1).expand(-1, width),
            values,
            reduce="amax",
            include_self=True,
        )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return sums / counts.clamp_min(1.0), maxima


def _segmented_softmax(
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
