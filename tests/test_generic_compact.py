from __future__ import annotations

import math
import unittest

import torch

from CPNModelV2.graph.generic_tensorizer import (
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
    _matrix,
    participation_relation,
)
from CPNModelV2.graph.schema import GraphEdge, GraphNode, TokenBindingGraph
from CPNModelV2.integration.fjsp_v5_generic_adapter import (
    FJSPV5GenericGraphBuilder,
)
from CPNModelV2.learning.generic_compact_policy import (
    GenericCompactPolicyNetwork,
)
from CPNModelV2.learning.environment import CPNGraphEnv
from CPNModelV2.learning.instances import SongSD1Config, SongSD1V5NetFactory
from CPNModelV2.learning.ppo import PPOConfig, TokenBindingPPO


class GenericCompactTest(unittest.TestCase):
    def test_categorical_short_rows_use_reserved_missing_value(self) -> None:
        with self.assertWarnsRegex(
            RuntimeWarning, "padded with the reserved missing value"
        ):
            values = _matrix(
                [[3], []],
                1,
                torch.long,
                torch.device("cpu"),
                pad_value=0,
            )
        self.assertEqual(values.tolist(), [[3], [0]])

        with self.assertRaisesRegex(
            ValueError, "Feature matrix row width mismatch"
        ):
            _matrix(
                [[1.0], []],
                1,
                torch.float32,
                torch.device("cpu"),
            )

    def test_fjsp_adapter_preserves_enabled_actions_and_semantics(self) -> None:
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

        self.assertEqual(len(net.control_bindings()), len(graph.action_keys))
        self.assertEqual(
            {"operation", "machine", "eligibility", "precedence", "binding"},
            {node.node_type for node in graph.nodes.values()},
        )
        self.assertTrue(any(
            edge.edge_type.startswith("references::") for edge in graph.edges
        ))
        self.assertTrue(any(
            edge.edge_type.startswith("participates::") for edge in graph.edges
        ))

        schema = GenericGraphFeatureSchema.fit([graph])
        normalizer = GenericNumericFeatureNormalizer.fit([graph], schema)
        batch = GenericGraphTensorizer(
            schema, normalizer=normalizer
        ).tensorize(graph)
        policy = GenericCompactPolicyNetwork(
            schema,
            hidden_dim=16,
            category_dim=8,
            reference_layers=1,
        )
        with torch.no_grad():
            output = policy(batch)
        self.assertEqual(output.logits.shape, (len(graph.action_keys),))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.allclose(
            output.probabilities.sum(), torch.tensor(1.0), atol=1e-6
        ))

    def test_single_token_type_without_references_is_valid(self) -> None:
        graph = TokenBindingGraph(time=0.0)
        graph.add_node(GraphNode(
            node_id="token:0",
            node_type="item",
            numeric={"priority": 1.0},
            categorical={"current_place": "waiting"},
        ))
        graph.add_node(GraphNode(
            node_id="token:1",
            node_type="item",
            numeric={"priority": 2.0},
            categorical={"current_place": "waiting"},
        ))
        for index in range(2):
            binding_id = f"binding:{index}"
            graph.add_node(GraphNode(
                node_id=binding_id,
                node_type="binding",
                categorical={"transition_name": "select"},
            ))
            graph.add_edge(GraphEdge(
                source=f"token:{index}",
                target=binding_id,
                edge_type=participation_relation("consume:item"),
            ))
            graph.action_keys[binding_id] = ("select", index)
        graph.validate()

        schema = GenericGraphFeatureSchema.fit([graph])
        self.assertEqual(schema.token_types, ("item",))
        self.assertEqual(schema.reference_relations, ())
        batch = GenericGraphTensorizer(schema).tensorize(graph)
        policy = GenericCompactPolicyNetwork(
            schema,
            hidden_dim=16,
            category_dim=8,
            reference_layers=2,
        )
        with torch.no_grad():
            output = policy(batch)

        self.assertEqual(output.logits.shape, (2,))
        self.assertEqual(output.values.shape, (1,))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.allclose(
            output.probabilities.sum(), torch.tensor(1.0), atol=1e-6
        ))

    def test_small_cpu_ppo_update_is_finite(self) -> None:
        config = SongSD1Config(
            num_jobs=2,
            num_machines=2,
            min_operations=2,
            max_operations=2,
            min_eligible_machines=1,
            max_eligible_machines=2,
        )
        factory = SongSD1V5NetFactory(config)
        builder = FJSPV5GenericGraphBuilder(validate=False)
        calibration = []
        for seed in (31, 32):
            net = factory(seed)
            net.advance_until_decision_or_terminal()
            calibration.append(builder.build(net))
        schema = GenericGraphFeatureSchema.fit(calibration)
        normalizer = GenericNumericFeatureNormalizer.fit(calibration, schema)
        trainer = TokenBindingPPO(
            GenericCompactPolicyNetwork(
                schema,
                hidden_dim=16,
                category_dim=8,
                reference_layers=1,
            ),
            GenericGraphTensorizer(
                schema, normalizer=normalizer, validate_graphs=False
            ),
            config=PPOConfig(
                update_epochs=1,
                minibatch_size=16,
                device_microbatch_size=8,
            ),
            device="cpu",
        )
        environments = [
            CPNGraphEnv(factory, graph_builder=builder) for _ in range(2)
        ]
        episodes = trainer.collect_episodes_batched(
            environments, seeds=[41, 42]
        )
        metrics = trainer.update(episodes, seed=43)

        self.assertEqual(len(episodes), 2)
        self.assertTrue(all(episode.makespan > 0 for episode in episodes))
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_streaming_evaluation_matches_rollout_evaluation(self) -> None:
        config = SongSD1Config(
            num_jobs=2,
            num_machines=2,
            min_operations=2,
            max_operations=2,
            min_eligible_machines=1,
            max_eligible_machines=2,
        )
        factory = SongSD1V5NetFactory(config)
        builder = FJSPV5GenericGraphBuilder(validate=False)
        calibration = []
        for seed in (51, 52):
            net = factory(seed)
            net.advance_until_decision_or_terminal()
            calibration.append(builder.build(net))
        schema = GenericGraphFeatureSchema.fit(calibration)
        normalizer = GenericNumericFeatureNormalizer.fit(calibration, schema)
        trainer = TokenBindingPPO(
            GenericCompactPolicyNetwork(
                schema,
                hidden_dim=16,
                category_dim=8,
                reference_layers=1,
            ),
            GenericGraphTensorizer(
                schema, normalizer=normalizer, validate_graphs=False
            ),
            config=PPOConfig(update_epochs=1, minibatch_size=16),
            device="cpu",
        )
        seeds = [61, 62]
        rollout_results = trainer.collect_episodes_batched(
            [CPNGraphEnv(factory, graph_builder=builder) for _ in seeds],
            seeds=seeds,
            deterministic=True,
        )
        evaluation_results = trainer.evaluate_episodes_batched(
            [CPNGraphEnv(factory, graph_builder=builder) for _ in seeds],
            seeds=seeds,
            deterministic=True,
        )

        self.assertEqual(
            [result.makespan for result in evaluation_results],
            [result.makespan for result in rollout_results],
        )
        self.assertEqual(
            [result.episode_return for result in evaluation_results],
            [result.episode_return for result in rollout_results],
        )
        self.assertEqual(
            [result.decision_count for result in evaluation_results],
            [result.decision_count for result in rollout_results],
        )


if __name__ == "__main__":
    unittest.main()
