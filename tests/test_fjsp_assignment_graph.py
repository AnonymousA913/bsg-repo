from __future__ import annotations

import importlib.util
import unittest

import torch

from CPNModelV2.learning import SongSD1Config, SongSD1V5NetFactory


HAS_TORCH_GEOMETRIC = importlib.util.find_spec("torch_geometric") is not None


@unittest.skipUnless(
    HAS_TORCH_GEOMETRIC,
    "FJSP AG-PPO requires the optional torch-geometric dependency",
)
class FJSPAssignmentGraphTest(unittest.TestCase):
    @staticmethod
    def _graphs():
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPAssignmentGraphBuilder,
        )

        config = SongSD1Config(
            num_jobs=3,
            num_machines=3,
            min_operations=2,
            max_operations=3,
            min_eligible_machines=1,
            max_eligible_machines=3,
        )
        net = SongSD1V5NetFactory(config)(101)
        net.advance_until_decision_or_terminal()
        builder = FJSPAssignmentGraphBuilder(validate=True)
        first = builder.build(net)
        net.step(next(iter(first.action_keys.values())))
        second = builder.build(net)
        return first, second

    def test_builder_preserves_actions_and_fixed_metadata(self) -> None:
        first, second = self._graphs()
        self.assertEqual(
            len(first.action_keys),
            first.assignment_graph["a_transition"].num_nodes,
        )
        self.assertEqual(
            first.assignment_graph.metadata(),
            second.assignment_graph.metadata(),
        )
        self.assertFalse(any(
            "reference" in relation
            for _source, relation, _target in first.assignment_graph.edge_types
        ))
        self.assertFalse(first.metadata["reference_edges_visible"])
        self.assertFalse(first.metadata["binding_role_labels_visible"])

    def test_shared_assignment_graph_policy_produces_valid_batch(self) -> None:
        from CPNModelV2.integration.bpm2024_assignment_graph import (
            AssignmentGraphFeatureSchema,
            AssignmentGraphTensorizer,
        )
        from CPNModelV2.integration.fjsp_assignment_graph import (
            FJSPAssignmentGraphNoOpNormalizer,
            FJSPControlledAssignmentGraphPolicy,
        )

        graphs = self._graphs()
        schema = AssignmentGraphFeatureSchema.fit(graphs)
        normalizer = FJSPAssignmentGraphNoOpNormalizer.fit(graphs, schema)
        batch = AssignmentGraphTensorizer(
            schema, normalizer=normalizer, validate_graphs=True
        ).tensorize(graphs)
        policy = FJSPControlledAssignmentGraphPolicy(
            schema,
            hidden_dim=16,
            category_dim=8,
            reference_layers=2,
            competition_layers=0,
        )
        output = policy(batch)
        self.assertEqual(output.logits.shape, (len(batch.action_keys),))
        self.assertEqual(output.values.shape, (2,))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.values).all())
        for graph_index in range(batch.num_graphs):
            probabilities = output.probabilities[
                batch.action_batch == graph_index
            ]
            self.assertTrue(torch.allclose(
                probabilities.sum(), torch.tensor(1.0), atol=1e-6
            ))
        (output.logits.mean() + output.values.mean()).backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in policy.parameters()
            if parameter.requires_grad
        ))


if __name__ == "__main__":
    unittest.main()
