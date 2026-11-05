from __future__ import annotations

from pathlib import Path
import random
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
    from CPNModelV2.integration.gympn_scenarios import (
        PAPER_SCENARIO_NAMES,
        build_minimal_task_assignment_problem,
        paper_scenario_factory,
    )
    from CPNModelV2.integration.gympn_decision_adapter import GymPNDecisionEnv


@unittest.skipUnless(GYMPN_AVAILABLE, "GymPN/SimPN dependencies are unavailable")
class GymPNDecisionIntegrationTest(unittest.TestCase):
    def test_all_paper_scenarios_build_valid_action_graphs(self) -> None:
        for scenario_name in PAPER_SCENARIO_NAMES:
            with self.subTest(scenario=scenario_name):
                env = GymPNDecisionEnv(
                    paper_scenario_factory(scenario_name),
                    horizon=4,
                )
                graph = env.reset(seed=7)
                graph.validate()
                self.assertGreater(len(graph.action_keys), 0)
                self.assertEqual(
                    scenario_name,
                    graph.metadata["scenario_name"],
                )
                self.assertEqual(
                    {"binding"},
                    {
                        graph.nodes[node_id].node_type
                        for node_id in graph.action_keys
                    },
                )

    def test_all_paper_scenarios_terminate_under_random_actions(self) -> None:
        for scenario_index, scenario_name in enumerate(PAPER_SCENARIO_NAMES):
            with self.subTest(scenario=scenario_name):
                seed = 100 + scenario_index
                env = GymPNDecisionEnv(
                    paper_scenario_factory(scenario_name),
                    horizon=4,
                )
                graph = env.reset(seed=seed)
                rng = random.Random(seed)
                for _step_index in range(200):
                    step = env.step(
                        rng.choice(list(graph.action_keys.values()))
                    )
                    graph = step.observation
                    if step.terminated:
                        break
                else:
                    self.fail(f"Scenario did not terminate: {scenario_name}")
                self.assertEqual(
                    step.info["makespan"], -step.info["completed_tasks"]
                )

    def test_minimal_marking_becomes_binding_effect_graph(self) -> None:
        env = GymPNDecisionEnv(
            build_minimal_task_assignment_problem,
            horizon=4,
        )
        graph = env.reset(seed=7)
        graph.validate()
        self.assertEqual(len(graph.action_keys), 4)
        self.assertEqual(
            {"color__num_task_type", "color__num_code_employee", "binding"},
            {node.node_type for node in graph.nodes.values()},
        )
        self.assertFalse(any(
            edge.edge_type.startswith("references::") for edge in graph.edges
        ))
        delays = {
            item.numeric_delta["available_time_delta"]
            for effect in graph.binding_effects.values()
            for item in effect.token_effects
        }
        self.assertEqual(delays, {1.0, 2.0})
        self.assertTrue(all(
            item.after_place == "busy"
            for effect in graph.binding_effects.values()
            for item in effect.token_effects
        ))

    def test_random_episode_terminates_at_finite_horizon(self) -> None:
        env = GymPNDecisionEnv(
            build_minimal_task_assignment_problem,
            horizon=4,
        )
        graph = env.reset(seed=11)
        rng = random.Random(11)
        terminated = False
        steps = 0
        while not terminated:
            step = env.step(rng.choice(list(graph.action_keys.values())))
            graph = step.observation
            terminated = step.terminated
            steps += 1
            self.assertLess(steps, 100)
        self.assertGreater(steps, 0)
        self.assertGreaterEqual(step.info["completed_tasks"], 0)
        self.assertEqual(step.info["makespan"], -step.info["completed_tasks"])

if __name__ == "__main__":
    unittest.main()
