from __future__ import annotations

from pathlib import Path
import sys
import unittest


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
    from CPNModelV2.integration.gympn_assignment_graph import (
        GymPNAssignmentGraphEnv,
    )
    from CPNModelV2.integration.gympn_bsg_adapter import GymPNBSGEnv
    from CPNModelV2.integration.gympn_semantic_matrix_scenarios import (
        FORMAL_AMBIGUITY_RATIOS,
        SEMANTIC_MATRIX_SCALES,
        ambiguity_group_ids,
        semantic_matrix_scenario_factory,
    )


@unittest.skipUnless(GYMPN_AVAILABLE, "GymPN/SimPN dependencies are unavailable")
class GymPNSemanticMatrixScenarioTest(unittest.TestCase):
    def test_formal_ratios_are_exact_and_nested(self) -> None:
        for scale in SEMANTIC_MATRIX_SCALES.values():
            previous = set()
            for ratio in FORMAL_AMBIGUITY_RATIOS:
                current = set(ambiguity_group_ids(
                    19000,
                    scale.decision_groups,
                    ratio,
                ))
                self.assertEqual(
                    len(current),
                    int(ratio * scale.decision_groups),
                )
                self.assertTrue(previous <= current)
                previous = current

    def test_local_and_semantic_oracles_solve_their_endpoints(self) -> None:
        for ratio, oracle in ((0.0, _local_oracle), (1.0, _semantic_oracle)):
            env = GymPNBSGEnv(
                semantic_matrix_scenario_factory(
                    "C2",
                    ambiguity_ratios=(ratio,),
                ),
                horizon=30.0,
            )
            graph = env.reset(seed=9000)
            while True:
                step = env.step(oracle(env, graph))
                graph = step.observation
                if step.terminated:
                    self.assertEqual(
                        step.info["correct_decisions"],
                        SEMANTIC_MATRIX_SCALES["C2"].decision_groups,
                    )
                    break

    def test_group_identity_is_hidden_from_bsg_features(self) -> None:
        env = GymPNBSGEnv(
            semantic_matrix_scenario_factory("C1", ambiguity_ratios=(0.5,)),
            horizon=30.0,
        )
        graph = env.reset(seed=9000)
        for node in graph.nodes.values():
            self.assertFalse(any("group_id" in key for key in node.numeric))
            self.assertFalse(any("group_id" in key for key in node.categorical))

    def test_assignment_graph_handles_repeated_input_place_roles(self) -> None:
        factory = semantic_matrix_scenario_factory(
            "C1",
            ambiguity_ratios=(1.0,),
        )
        bsg_graph = GymPNBSGEnv(factory, horizon=30.0).reset(seed=9000)
        ag_graph = GymPNAssignmentGraphEnv(factory, horizon=30.0).reset(seed=9000)
        self.assertEqual(len(ag_graph.action_keys), len(bsg_graph.action_keys))
        self.assertGreater(len(ag_graph.action_keys), 0)


def _binding_values(env, action_key):
    binding_index = int(action_key[2])
    participants, _time, transition = env.problem.pn_actions[binding_index]
    return [token.value for _place, token in participants], str(transition._id)


def _local_oracle(env, graph):
    ranked = []
    for action_key in graph.action_keys.values():
        values, transition_name = _binding_values(env, action_key)
        if transition_name == "select_ordered_binding":
            score = sum(float(value.get("local_priority", 0.0)) for value in values)
        else:
            score = float(values[0].get("local_priority", 0.0))
        ranked.append((score, action_key))
    return max(ranked, key=lambda item: item[0])[1]


def _semantic_oracle(env, graph):
    ranked = []
    for action_key in graph.action_keys.values():
        values, transition_name = _binding_values(env, action_key)
        if transition_name == "select_direction_candidate":
            score = float(bool(str(values[0]["next_ref"])))
        elif transition_name == "select_typed_candidate":
            score = float(bool(str(values[0]["supports_ref"])))
        else:
            source, target, _slot = values
            top_pair = (
                float(source["local_priority"]) > 0.5
                and float(target["local_priority"]) > 0.5
            )
            score = float(
                top_pair and float(source["rank"]) < float(target["rank"])
            )
        ranked.append((score, action_key))
    return max(ranked, key=lambda item: item[0])[1]


if __name__ == "__main__":
    unittest.main()
