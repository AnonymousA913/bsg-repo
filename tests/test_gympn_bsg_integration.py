from __future__ import annotations

import unittest

import torch

from CPNModelV2.graph.binding_effect import BindingEffectGraph
from CPNModelV2.graph.generic_tensorizer import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
)
from CPNModelV2.integration.gympn_bsg_adapter import (
    build_bsg_paper_suite_environment,
)
from CPNModelV2.integration.gympn_scenarios import PAPER_SCENARIO_NAMES
from CPNModelV2.learning.gympn_bsg_policy import GymPNGenericCompactPolicy


class GymPNBSGIntegrationTest(unittest.TestCase):
    def test_case_multiplier_increases_concurrent_bindings(self) -> None:
        base = build_bsg_paper_suite_environment(
            "sequence_joint",
            horizon=4,
            case_multiplier=1,
        ).reset(seed=23)
        scaled = build_bsg_paper_suite_environment(
            "sequence_joint",
            horizon=4,
            case_multiplier=2,
        ).reset(seed=23)

        self.assertEqual(base.metadata["case_multiplier"], 1)
        self.assertEqual(scaled.metadata["case_multiplier"], 2)
        self.assertGreater(len(scaled.action_keys), len(base.action_keys))

    def test_all_scenarios_support_controlled_load_levels(self) -> None:
        for case_multiplier in (1, 2, 4):
            for scenario_name in PAPER_SCENARIO_NAMES:
                graph = build_bsg_paper_suite_environment(
                    scenario_name,
                    horizon=4,
                    case_multiplier=case_multiplier,
                ).reset(seed=31)
                graph.validate()
                self.assertEqual(
                    graph.metadata["case_multiplier"],
                    case_multiplier,
                )
                self.assertGreater(len(graph.action_keys), 0)

    def test_large_namespaced_seed_is_supported(self) -> None:
        env = build_bsg_paper_suite_environment(
            PAPER_SCENARIO_NAMES[0],
            horizon=4,
        )
        graph = env.reset(seed=9_010_000_000)
        self.assertGreater(len(graph.action_keys), 0)

    def test_all_scenarios_expose_plain_token_binding_graphs(self) -> None:
        graphs = []
        for scenario_name in PAPER_SCENARIO_NAMES:
            env = build_bsg_paper_suite_environment(scenario_name, horizon=4)
            graph = env.reset(seed=17)
            graph.validate()
            self.assertNotIsInstance(graph, BindingEffectGraph)
            self.assertGreater(len(graph.action_keys), 0)
            self.assertFalse(graph.metadata["binding_effects_exposed"])
            graphs.append(graph)

        schema = GenericGraphFeatureSchema.fit(graphs)
        normalizer = GenericNumericFeatureNormalizer.fit(graphs, schema)
        batch = GenericGraphTensorizer(
            schema,
            normalizer=normalizer,
            validate_graphs=True,
        ).tensorize(graphs)
        policy = GymPNGenericCompactPolicy(
            schema,
            hidden_dim=16,
            category_dim=8,
            reference_layers=1,
            factor_rounds=0,
        )
        with torch.no_grad():
            output = policy(batch)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.values).all())


if __name__ == "__main__":
    unittest.main()
