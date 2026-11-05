from __future__ import annotations

import unittest

import torch

from CPNModelV2.graph.generic_tensorizer import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.integration.fjsp_v5_generic_adapter import (
    FJSPV5GenericGraphBuilder,
)
from CPNModelV2.learning import (
    BSG_RL_SMALL_CATEGORY_DIM,
    BSG_RL_SMALL_HIDDEN_DIM,
    BSG_RL_SMALL_REFERENCE_LAYERS,
    BSGRLSmallPolicyNetwork,
    SongSD1Config,
    SongSD1V5NetFactory,
)


class BSGRLSmallTest(unittest.TestCase):
    @staticmethod
    def _schema_and_batch():
        config = SongSD1Config(
            num_jobs=2,
            num_machines=2,
            min_operations=2,
            max_operations=2,
            min_eligible_machines=1,
            max_eligible_machines=2,
        )
        net = SongSD1V5NetFactory(config)(seed=23)
        net.advance_until_decision_or_terminal()
        graph = FJSPV5GenericGraphBuilder().build(net)
        schema = GenericGraphFeatureSchema.fit([graph])
        normalizer = GenericNumericFeatureNormalizer.fit([graph], schema)
        batch = GenericGraphTensorizer(
            schema, normalizer=normalizer
        ).tensorize(graph)
        return schema, batch

    def test_fixed_capacity_policy_runs_with_full_semantics(self) -> None:
        schema, batch = self._schema_and_batch()
        policy = BSGRLSmallPolicyNetwork(schema)

        self.assertEqual(policy.hidden_dim, BSG_RL_SMALL_HIDDEN_DIM)
        self.assertEqual(policy.category_dim, BSG_RL_SMALL_CATEGORY_DIM)
        self.assertEqual(
            len(policy.reference_layers), BSG_RL_SMALL_REFERENCE_LAYERS
        )
        with torch.no_grad():
            output = policy(batch)
        self.assertEqual(
            output.logits.shape[0], batch.action_binding_index.shape[0]
        )
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_capacity_configuration_cannot_drift(self) -> None:
        schema, _batch = self._schema_and_batch()
        with self.assertRaisesRegex(ValueError, "fixed capacity-control"):
            BSGRLSmallPolicyNetwork(schema, hidden_dim=32)


if __name__ == "__main__":
    unittest.main()
