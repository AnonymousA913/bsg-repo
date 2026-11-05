from __future__ import annotations

from collections import defaultdict
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
    from CPNModelV2.graph.generic_tensorizer import (
        GenericGraphFeatureSchema,
        GenericGraphTensorizer,
    )
    from CPNModelV2.integration.gympn_decision_adapter import GymPNDecisionEnv
    from CPNModelV2.integration.gympn_semantic_stress_scenarios import (
        DIRECTED_DEPENDENCY,
        ORDERED_BINDING,
        TYPED_REFERENCE,
        semantic_stress_scenario_factory,
    )
    from CPNModelV2.learning.generic_compact_ablation_policy import (
        FLAT_BINDING_VARIANT,
        NO_DIRECTION_VARIANT,
        NO_REFERENCE_VARIANT,
        NO_REFERENCE_FLAT_BINDING_VARIANT,
        NO_RELATION_DIRECTION_VARIANT,
        NO_RELATION_TYPE_VARIANT,
        GenericCompactAblationPolicyNetwork,
    )


@unittest.skipUnless(GYMPN_AVAILABLE, "GymPN/SimPN dependencies are unavailable")
class GymPNSemanticStressScenarioTest(unittest.TestCase):
    def test_directed_dependency_is_direction_identifiable_only(self) -> None:
        env, graph = _reset(DIRECTED_DEPENDENCY)
        self.assertEqual(len(graph.action_keys), 8)

        candidate_by_binding = _single_participant_by_binding(
            graph,
            "participates::consume::candidate",
        )
        actions_by_pair = _actions_by_pair(graph, candidate_by_binding)
        next_edges = graph.edges_of_type("references::next")
        self.assertEqual(len(next_edges), 4)

        payloads = set()
        for pair_actions in actions_by_pair.values():
            self.assertEqual(len(pair_actions), 2)
            candidates = [candidate_by_binding[action] for action in pair_actions]
            first, second = (graph.nodes[node_id] for node_id in candidates)
            self.assertEqual(first.node_type, second.node_type)
            self.assertEqual(first.numeric, second.numeric)
            self.assertEqual(first.categorical, second.categorical)
            self.assertNotIn("value_pair_id", first.numeric)
            payloads.add(first.numeric["value_payload"])
            pair_edge = next(
                edge for edge in next_edges
                if edge.source in candidates or edge.target in candidates
            )
            self.assertEqual({pair_edge.source, pair_edge.target}, set(candidates))
        self.assertGreater(len(payloads), 1)

        no_reference = _action_logits(graph, NO_REFERENCE_VARIANT)
        no_relation_direction = _action_logits(
            graph,
            NO_RELATION_DIRECTION_VARIANT,
        )
        no_direction = _action_logits(graph, NO_DIRECTION_VARIANT)
        joint_semantic_ablation = _action_logits(
            graph,
            NO_REFERENCE_FLAT_BINDING_VARIANT,
        )
        for pair_actions in actions_by_pair.values():
            left, right = pair_actions
            self.assertAlmostEqual(
                no_reference[left], no_reference[right], places=6
            )
            self.assertAlmostEqual(
                no_relation_direction[left],
                no_relation_direction[right],
                places=6,
            )
            self.assertAlmostEqual(
                no_direction[left], no_direction[right], places=6
            )
            self.assertAlmostEqual(
                joint_semantic_ablation[left],
                joint_semantic_ablation[right],
                places=6,
            )

        correct_binding = next(
            binding_id
            for binding_id, candidate_id in candidate_by_binding.items()
            if any(edge.source == candidate_id for edge in next_edges)
        )
        reward = env.step(graph.action_keys[correct_binding]).reward
        self.assertEqual(reward, 1.0)

    def test_typed_reference_is_relation_type_identifiable_only(self) -> None:
        env, graph = _reset(TYPED_REFERENCE)
        self.assertEqual(len(graph.action_keys), 8)

        candidate_by_binding = _single_participant_by_binding(
            graph,
            "participates::consume::candidate",
        )
        actions_by_pair = _actions_by_pair(graph, candidate_by_binding)
        supports_edges = graph.edges_of_type("references::supports")
        blocks_edges = graph.edges_of_type("references::blocks")
        self.assertEqual(len(supports_edges), 4)
        self.assertEqual(len(blocks_edges), 4)

        payloads = set()
        for pair_actions in actions_by_pair.values():
            self.assertEqual(len(pair_actions), 2)
            candidates = [candidate_by_binding[action] for action in pair_actions]
            first, second = (graph.nodes[node_id] for node_id in candidates)
            self.assertEqual(first.node_type, second.node_type)
            self.assertEqual(first.numeric, second.numeric)
            self.assertEqual(first.categorical, second.categorical)
            self.assertNotIn("value_pair_id", first.numeric)
            payloads.add(first.numeric["value_payload"])
            pair_support = next(
                edge for edge in supports_edges if edge.source in candidates
            )
            pair_block = next(
                edge for edge in blocks_edges if edge.source in candidates
            )
            self.assertNotEqual(pair_support.source, pair_block.source)
            self.assertEqual(pair_support.target, pair_block.target)
        self.assertGreater(len(payloads), 1)

        no_reference = _action_logits(graph, NO_REFERENCE_VARIANT)
        no_relation_type = _action_logits(graph, NO_RELATION_TYPE_VARIANT)
        for pair_actions in actions_by_pair.values():
            left, right = pair_actions
            self.assertAlmostEqual(
                no_reference[left], no_reference[right], places=6
            )
            self.assertAlmostEqual(
                no_relation_type[left], no_relation_type[right], places=6
            )

        correct_binding = next(
            binding_id
            for binding_id, candidate_id in candidate_by_binding.items()
            if any(edge.source == candidate_id for edge in supports_edges)
        )
        reward = env.step(graph.action_keys[correct_binding]).reward
        self.assertEqual(reward, 1.0)

    def test_ordered_binding_is_role_identifiable_only(self) -> None:
        env, graph = _reset(ORDERED_BINDING)
        self.assertEqual(len(graph.action_keys), 8)

        source_by_binding = _single_participant_by_binding(
            graph,
            "participates::consume::source",
        )
        target_by_binding = _single_participant_by_binding(
            graph,
            "participates::consume::target",
        )
        slot_by_binding = _single_participant_by_binding(
            graph,
            "participates::consume::pair_slot",
        )
        actions_by_pair = _actions_by_pair(graph, source_by_binding)

        observed_ranks = set()
        for pair_actions in actions_by_pair.values():
            self.assertEqual(len(pair_actions), 2)
            first, second = pair_actions
            self.assertEqual(
                source_by_binding[first],
                target_by_binding[second],
            )
            self.assertEqual(
                target_by_binding[first],
                source_by_binding[second],
            )
            self.assertEqual(slot_by_binding[first], slot_by_binding[second])
            self.assertNotIn(
                "value_pair_id",
                graph.nodes[source_by_binding[first]].numeric,
            )
            observed_ranks.update((
                graph.nodes[source_by_binding[first]].numeric["value_rank"],
                graph.nodes[target_by_binding[first]].numeric["value_rank"],
            ))
            first_flat = sorted((
                source_by_binding[first],
                target_by_binding[first],
                slot_by_binding[first],
            ))
            second_flat = sorted((
                source_by_binding[second],
                target_by_binding[second],
                slot_by_binding[second],
            ))
            self.assertEqual(first_flat, second_flat)
        self.assertGreater(len(observed_ranks), 2)

        flat_logits = _action_logits(graph, FLAT_BINDING_VARIANT)
        joint_semantic_logits = _action_logits(
            graph,
            NO_REFERENCE_FLAT_BINDING_VARIANT,
        )
        for pair_actions in actions_by_pair.values():
            left, right = pair_actions
            self.assertAlmostEqual(flat_logits[left], flat_logits[right], places=6)
            self.assertAlmostEqual(
                joint_semantic_logits[left],
                joint_semantic_logits[right],
                places=6,
            )

        correct_binding = next(
            binding_id
            for binding_id in graph.action_keys
            if (
                graph.nodes[source_by_binding[binding_id]].numeric["value_rank"]
                < graph.nodes[target_by_binding[binding_id]].numeric["value_rank"]
            )
        )
        reward = env.step(graph.action_keys[correct_binding]).reward
        self.assertEqual(reward, 1.0)

    def test_counterfactual_interventions_flip_the_correct_choice(self) -> None:
        for scenario_name in (
            DIRECTED_DEPENDENCY,
            TYPED_REFERENCE,
            ORDERED_BINDING,
        ):
            _base_env, base_graph = _reset(scenario_name, counterfactual=False)
            _cf_env, cf_graph = _reset(scenario_name, counterfactual=True)
            self.assertEqual(
                tuple(base_graph.action_keys),
                tuple(cf_graph.action_keys),
            )
            base_correct = _correct_choice_by_pair(base_graph)
            cf_correct = _correct_choice_by_pair(cf_graph)
            self.assertEqual(set(base_correct), set(cf_correct))
            for pair_id in base_correct:
                self.assertNotEqual(base_correct[pair_id], cf_correct[pair_id])


def _reset(scenario_name: str, *, counterfactual: bool = False):
    env = GymPNDecisionEnv(
        semantic_stress_scenario_factory(
            scenario_name,
            pair_count=4,
            counterfactual=counterfactual,
        ),
        horizon=10.0,
    )
    return env, env.reset(seed=9000)


def _single_participant_by_binding(graph, role: str):
    result = {}
    for edge in graph.edges_of_type(role):
        if edge.target in result:
            raise AssertionError(f"Binding has multiple participants for {role}")
        result[edge.target] = edge.source
    if set(result) != set(graph.action_keys):
        raise AssertionError(f"Role {role} does not cover every action")
    return result


def _actions_by_pair(graph, participant_by_binding):
    grouped = defaultdict(list)
    for binding_id in graph.action_keys:
        participant = graph.nodes[participant_by_binding[binding_id]]
        object_id = str(participant.identity["object_id"])
        grouped[int(object_id.split(":")[1])].append(binding_id)
    return grouped


def _correct_choice_by_pair(graph):
    scenario_name = graph.metadata["scenario_name"]
    if scenario_name == DIRECTED_DEPENDENCY:
        correct_nodes = {
            edge.source for edge in graph.edges_of_type("references::next")
        }
    elif scenario_name == TYPED_REFERENCE:
        correct_nodes = {
            edge.source for edge in graph.edges_of_type("references::supports")
        }
    else:
        source_by_binding = _single_participant_by_binding(
            graph,
            "participates::consume::source",
        )
        target_by_binding = _single_participant_by_binding(
            graph,
            "participates::consume::target",
        )
        correct_nodes = {
            source_by_binding[binding_id]
            for binding_id in graph.action_keys
            if (
                graph.nodes[source_by_binding[binding_id]].numeric["value_rank"]
                < graph.nodes[target_by_binding[binding_id]].numeric["value_rank"]
            )
        }
    result = {}
    for participant_id in correct_nodes:
        object_id = str(graph.nodes[participant_id].identity["object_id"])
        pair_id = int(object_id.split(":")[1])
        result[pair_id] = object_id
    return result


def _action_logits(graph, variant: str):
    schema = GenericGraphFeatureSchema.fit([graph])
    batch = GenericGraphTensorizer(schema).tensorize(graph)
    torch.manual_seed(17)
    model = GenericCompactAblationPolicyNetwork(
        schema,
        hidden_dim=16,
        category_dim=8,
        reference_layers=1,
        competition_layers=0,
        dropout=0.0,
        ablation_variant=variant,
    )
    model.eval()
    with torch.no_grad():
        logits = model(batch).logits.tolist()
    return dict(zip(batch.action_node_ids, logits))


if __name__ == "__main__":
    unittest.main()
