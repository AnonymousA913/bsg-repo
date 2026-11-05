from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GYMPN_ROOT = PROJECT_ROOT / "gympn-main"
if str(GYMPN_ROOT) not in sys.path:
    sys.path.insert(0, str(GYMPN_ROOT))

try:
    import gympn  # noqa: F401
    import simpn  # noqa: F401
except ModuleNotFoundError:
    GYMPN_AVAILABLE = False
else:
    GYMPN_AVAILABLE = True

if GYMPN_AVAILABLE:
    from CPNModelV2.integration.bpm2024_assignment_graph import (
        AssignmentGraphFeatureSchema,
        AssignmentGraphTensorizer,
    )
    from CPNModelV2.integration.gympn_assignment_graph import (
        AssignmentGraphNoOpNormalizer,
        GymPNControlledAssignmentGraphPolicy,
        build_ag_complex_suite_environment,
    )
    from CPNModelV2.integration.gympn_complex_scenarios import (
        COMPLEXITY_CONFIGS,
    )
    from CPNModelV2.integration.gympn_scenarios import PAPER_SCENARIO_NAMES


@unittest.skipUnless(GYMPN_AVAILABLE, "GymPN/SimPN dependencies are unavailable")
class GymPNAssignmentGraphTest(unittest.TestCase):
    def test_all_complex_scenarios_map_only_executable_actions(self) -> None:
        for level in COMPLEXITY_CONFIGS:
            for scenario_name in PAPER_SCENARIO_NAMES:
                with self.subTest(level=level, scenario=scenario_name):
                    env = build_ag_complex_suite_environment(
                        scenario_name,
                        level,
                        horizon=10,
                        validate_graphs=True,
                    )
                    graph = env.reset(seed=31_000)
                    self.assertEqual(
                        len(graph.action_keys),
                        len(env.problem.pn_actions),
                    )
                    self.assertEqual(
                        len(graph.assignment_action_indices),
                        len(set(graph.assignment_action_indices)),
                    )
                    node_count = int(
                        graph.assignment_graph["a_transition"].num_nodes
                    )
                    self.assertEqual(node_count, len(graph.action_keys))
                    self.assertTrue(all(
                        0 <= index < node_count
                        for index in graph.assignment_action_indices
                    ))
                    # The observation-only numeric copy must not replace the
                    # executable problem's semantic string identifiers.
                    string_values = [
                        value
                        for place in env.problem.places
                        for token in place.marking
                        if isinstance(token.value, dict)
                        for value in token.value.values()
                        if isinstance(value, str)
                    ]
                    self.assertTrue(string_values)
                    metadata = graph.assignment_graph.metadata()
                    step = env.step(next(iter(graph.action_keys.values())))
                    self.assertIsNotNone(step.observation)
                    if not step.terminated:
                        self.assertEqual(
                            step.observation.assignment_graph.metadata(),
                            metadata,
                        )

    def test_schema_tensorizer_and_han_policy_are_ppo_compatible(self) -> None:
        graphs = [
            build_ag_complex_suite_environment(
                "sequence_joint", "C1", horizon=10
            ).reset(seed=seed)
            for seed in (31_000, 31_001)
        ]
        schema = AssignmentGraphFeatureSchema.fit(graphs)
        normalizer = AssignmentGraphNoOpNormalizer.fit(graphs, schema)
        restored = AssignmentGraphNoOpNormalizer.from_dict(
            normalizer.to_dict(), schema
        )
        batch = AssignmentGraphTensorizer(
            schema,
            normalizer=restored,
        ).tensorize(graphs)
        policy = GymPNControlledAssignmentGraphPolicy(
            schema,
            hidden_dim=16,
            category_dim=8,
            reference_layers=2,
            factor_rounds=0,
        )
        output = policy(batch)
        self.assertEqual(output.logits.numel(), len(batch.action_keys))
        self.assertEqual(tuple(output.values.shape), (2,))
        for graph_index in range(2):
            probability_sum = output.probabilities[
                batch.action_batch == graph_index
            ].sum()
            self.assertTrue(torch.allclose(
                probability_sum,
                probability_sum.new_tensor(1.0),
                atol=1e-5,
            ))


if __name__ == "__main__":
    unittest.main()
