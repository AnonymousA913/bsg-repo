from __future__ import annotations

from CPNModelV2.graph.generic_tensorizer import GenericGraphFeatureSchema
from CPNModelV2.learning.generic_compact_policy import (
    GenericCompactPolicyNetwork,
)


class GymPNGenericCompactPolicy(GenericCompactPolicyNetwork):
    """Generic Compact V5 with the shared GymPN runner configuration shape."""

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        factor_rounds: int = 0,
        dropout: float = 0.0,
    ) -> None:
        if factor_rounds != 0:
            raise ValueError(
                "BSG-PPO uses Generic Compact V5 and requires factor_rounds=0"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=0,
            dropout=dropout,
        )
