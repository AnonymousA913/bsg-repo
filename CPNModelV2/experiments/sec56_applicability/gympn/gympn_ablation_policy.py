from __future__ import annotations

"""GymPN constructor adapter for the Section 5.5 semantic ablations."""

from CPNModelV2.graph.generic_tensorizer import GenericGraphFeatureSchema
from CPNModelV2.learning.generic_compact_ablation_policy import (
    FULL_VARIANT,
    GenericCompactAblationPolicyNetwork,
)


class GymPNGenericCompactAblationPolicy(
    GenericCompactAblationPolicyNetwork
):
    """Expose the Generic Compact ablations through the GymPN runner API.

    The shared GymPN runner names its historical structural-round argument
    ``factor_rounds``.  Generic Compact V5 does not use those rounds, so the
    adapter validates that it remains zero and maps the remaining arguments to
    the controlled Section 5.5 ablation implementation.
    """

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        factor_rounds: int = 0,
        dropout: float = 0.0,
        ablation_variant: str = FULL_VARIANT,
    ) -> None:
        if factor_rounds != 0:
            raise ValueError(
                "GymPN BSG-RL ablations require factor_rounds=0"
            )
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=0,
            dropout=dropout,
            ablation_variant=ablation_variant,
        )
