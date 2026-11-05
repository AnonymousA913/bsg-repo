"""Build one executable FJSP CPN state and run a BSG-RL forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from CPNModelV2.graph.generic_tensorizer import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.integration.fjsp_v5_generic_adapter import (
    FJSPV5GenericGraphBuilder,
)
from CPNModelV2.learning.generic_compact_v7_policy import (
    GenericCompactV7PolicyNetwork,
)
from CPNModelV2.learning.instances import SongSD1Config, SongSD1V5NetFactory


def main() -> None:
    config = SongSD1Config(
        num_jobs=2,
        num_machines=2,
        min_operations=2,
        max_operations=2,
        min_eligible_machines=1,
        max_eligible_machines=2,
    )
    net = SongSD1V5NetFactory(config)(seed=7)
    net.advance_until_decision_or_terminal()
    graph = FJSPV5GenericGraphBuilder().build(net)

    schema = GenericGraphFeatureSchema.fit([graph])
    normalizer = GenericNumericFeatureNormalizer.fit([graph], schema)
    batch = GenericGraphTensorizer(
        schema,
        normalizer=normalizer,
    ).tensorize(graph)
    policy = GenericCompactV7PolicyNetwork(
        schema,
        hidden_dim=32,
        category_dim=8,
        reference_layers=2,
        competition_layers=0,
    )
    with torch.no_grad():
        output = policy(batch)

    print(f"enabled bindings: {len(graph.action_keys)}")
    print(f"logits shape: {tuple(output.logits.shape)}")
    print(f"state value: {float(output.values[0]):.6f}")


if __name__ == "__main__":
    main()
