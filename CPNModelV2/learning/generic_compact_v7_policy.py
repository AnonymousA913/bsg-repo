from __future__ import annotations

"""Generic Compact V7 with the role-semantic Compact V5 C3 operator."""

from typing import Mapping

import torch
from torch import nn

from CPNModelV2.graph.generic_tensorizer import (
    BINDING_NODE_TYPE,
    GenericGraphFeatureSchema,
    GenericTokenBindingBatch,
)
from CPNModelV2.learning.generic_compact_policy import (
    GenericCompactPolicyNetwork,
)


class V7RoleSemanticBindingComposer(nn.Module):
    """Compose Bindings through schema-declared participation-role channels.

    This is the domain-independent counterpart of Compact V5's C3 stage. Each
    role receives its own projection and Binding-conditioned gate, while role
    names and channel count come exclusively from the fitted graph schema.
    """

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.role_to_index = {
            role: index for index, role in enumerate(schema.participation_roles)
        }
        self.role_projection = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in schema.participation_roles
        ])
        self.role_embedding = nn.Embedding(
            len(schema.relation_values), hidden_dim
        )
        self.role_gate = nn.ModuleList([
            nn.Linear(2 * hidden_dim, 1)
            for _ in schema.participation_roles
        ])
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
        role_sums = [
            binding.new_zeros(binding.shape)
            for _ in self.schema.participation_roles
        ]
        role_weights = [
            binding.new_zeros((binding.shape[0], 1))
            for _ in self.schema.participation_roles
        ]

        for canonical, edge_index in batch.participation_edge_index.items():
            source_type, role, target_type = canonical
            if target_type != BINDING_NODE_TYPE:
                continue
            role_index = self.role_to_index.get(role)
            if role_index is None:
                raise ValueError(
                    f"Participation role absent from V7 schema: {role}"
                )
            token_index, binding_index = edge_index
            source = hidden[source_type][token_index]
            target = binding[binding_index]
            relation_context = self.role_embedding.weight[
                batch.relation_ids[canonical]
            ].unsqueeze(0)
            transformed = (
                self.role_projection[role_index](source) + relation_context
            )
            gate = torch.sigmoid(self.role_gate[role_index](torch.cat(
                [transformed, target], dim=-1
            )))
            role_sums[role_index].index_add_(
                0, binding_index, gate * transformed
            )
            role_weights[role_index].index_add_(0, binding_index, gate)

        normalized = [
            role_sum / weight.clamp_min(1e-8)
            for role_sum, weight in zip(role_sums, role_weights)
        ]
        proposal = self.fusion(torch.cat([binding, *normalized], dim=-1))
        return self.norm(binding + proposal)


class GenericCompactV7PolicyNetwork(GenericCompactPolicyNetwork):
    """Generic Compact policy with the restored role-semantic C3 stage."""

    def _make_binding_composer(
        self,
        schema: GenericGraphFeatureSchema,
        hidden_dim: int,
        *,
        dropout: float,
    ) -> nn.Module:
        return V7RoleSemanticBindingComposer(
            schema, hidden_dim, dropout=dropout
        )
