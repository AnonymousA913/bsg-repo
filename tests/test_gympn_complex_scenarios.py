from __future__ import annotations

from pathlib import Path
import random
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
        GenericNumericFeatureNormalizer,
    )
    from CPNModelV2.integration.gympn_bsg_adapter import (
        build_bsg_complex_suite_environment,
        build_bsg_paper_suite_environment,
    )
    from CPNModelV2.integration.gympn_complex_scenarios import (
        COMPLEXITY_CONFIGS,
        COMPLEXITY_LEVEL_NAMES,
        PAPER_COMPLEXITY_LABELS,
        PAPER_COMPLEXITY_LEVELS,
        _advance_stream,
        build_complexity_scenario_problem,
    )
    from CPNModelV2.integration.gympn_scenarios import PAPER_SCENARIO_NAMES
    from CPNModelV2.learning.gympn_bsg_policy import GymPNGenericCompactPolicy


@unittest.skipUnless(GYMPN_AVAILABLE, "GymPN/SimPN dependencies are unavailable")
class GymPNComplexScenarioTest(unittest.TestCase):
    def test_p0_delegates_to_source_faithful_scenarios(self) -> None:
        for scenario_name in PAPER_SCENARIO_NAMES:
            with self.subTest(scenario=scenario_name):
                source_env = build_bsg_paper_suite_environment(
                    scenario_name,
                    horizon=10,
                )
                p0_env = build_bsg_complex_suite_environment(
                    scenario_name,
                    "P0",
                    horizon=10,
                )
                source_graph = source_env.reset(seed=13)
                p0_graph = p0_env.reset(seed=13)
                for _step_index in range(500):
                    self.assertEqual(
                        len(source_graph.action_keys),
                        len(p0_graph.action_keys),
                    )
                    source_step = source_env.step(
                        next(iter(source_graph.action_keys.values()))
                    )
                    p0_step = p0_env.step(
                        next(iter(p0_graph.action_keys.values()))
                    )
                    self.assertEqual(source_step.reward, p0_step.reward)
                    self.assertEqual(source_step.terminated, p0_step.terminated)
                    self.assertEqual(
                        source_step.info["decision_count"],
                        p0_step.info["decision_count"],
                    )
                    source_graph = source_step.observation
                    p0_graph = p0_step.observation
                    if source_step.terminated:
                        self.assertEqual(
                            source_step.info["completed_tasks"],
                            p0_step.info["completed_tasks"],
                        )
                        break
                else:
                    self.fail(f"P0 trace did not terminate: {scenario_name}")

    def test_all_scenarios_and_levels_build_valid_graphs(self) -> None:
        for level in COMPLEXITY_LEVEL_NAMES:
            for scenario_name in PAPER_SCENARIO_NAMES:
                with self.subTest(level=level, scenario=scenario_name):
                    graph = build_bsg_complex_suite_environment(
                        scenario_name,
                        level,
                        horizon=4,
                        validate_graphs=True,
                    ).reset(seed=17)
                    graph.validate()
                    self.assertGreater(len(graph.action_keys), 0)
                    self.assertEqual(level, graph.metadata["complexity_level"])
                    self.assertEqual(
                        level != "P0",
                        graph.metadata["semantic_references_enabled"],
                    )

    def test_paper_ladder_configuration_is_frozen(self) -> None:
        expected = {
            "C1": (3, 2, 2, 4, 3, 4, 0.85),
            "C1_5": (3, 3, 3, 4, 3, 6, 0.95),
            "C2": (4, 4, 4, 5, 3, 8, 1.05),
            "C2_25": (5, 4, 4, 5, 3, 10, 1.10),
            "C2_5": (5, 5, 5, 5, 3, 12, 1.15),
        }
        self.assertEqual(tuple(expected), PAPER_COMPLEXITY_LEVELS)
        self.assertEqual(
            tuple(PAPER_COMPLEXITY_LABELS.values()),
            ("L1", "L2", "L3", "L4", "L5"),
        )
        for level, values in expected.items():
            config = COMPLEXITY_CONFIGS[level]
            self.assertEqual(
                (
                    config.flow_size,
                    config.case_streams,
                    config.case_classes,
                    config.resources_per_pool,
                    config.eligibility_per_activity,
                    config.processing_time_max,
                    config.target_offered_load,
                ),
                values,
            )

    def test_initial_action_width_is_nondecreasing_on_paper_ladder(self) -> None:
        for scenario_name in PAPER_SCENARIO_NAMES:
            with self.subTest(scenario=scenario_name):
                counts = []
                for level in PAPER_COMPLEXITY_LEVELS:
                    graph = build_bsg_complex_suite_environment(
                        scenario_name,
                        level,
                        horizon=4,
                    ).reset(seed=23)
                    counts.append(len(graph.action_keys))
                self.assertEqual(counts, sorted(counts))

    def test_extended_levels_expose_reference_and_role_edges(self) -> None:
        for level in PAPER_COMPLEXITY_LEVELS:
            graph = build_bsg_complex_suite_environment(
                "parallel_joint",
                level,
                horizon=4,
                validate_graphs=True,
            ).reset(seed=29)
            edge_types = {edge.edge_type for edge in graph.edges}
            self.assertTrue(any(
                edge_type.startswith("references::")
                for edge_type in edge_types
            ))
            self.assertIn("references::same_case", edge_types)
            self.assertTrue(any(
                edge_type.startswith("participates::")
                for edge_type in edge_types
            ))
            self.assertTrue({
                "participates::consume::work",
                "participates::consume::resource",
                "participates::read::eligibility",
            }.issubset(edge_types))

    def test_arrival_load_matches_predeclared_uniform_assignment_target(self) -> None:
        for level, config in COMPLEXITY_CONFIGS.items():
            with self.subTest(level=level):
                problem = build_complexity_scenario_problem(
                    "exclusive_choice_joint",
                    level,
                    seed=31,
                )
                arrival_rate = config.case_streams / problem.gympn_arrival_interval
                random_mean_time = sum(
                    1 + round(
                        rank
                        / float(config.eligibility_per_activity - 1)
                        * (config.processing_time_max - 1)
                    )
                    for rank in range(config.eligibility_per_activity)
                ) / float(config.eligibility_per_activity)
                random_capacity = config.resources_per_pool / random_mean_time
                realized_load = arrival_rate / random_capacity
                self.assertAlmostEqual(
                    realized_load,
                    config.target_offered_load,
                    places=7,
                )

    def test_each_resource_is_fastest_for_some_activity_class(self) -> None:
        for level, config in COMPLEXITY_CONFIGS.items():
            problem = build_complexity_scenario_problem(
                "sequence_joint",
                level,
                seed=31,
            )
            eligibility = next(
                place for place in problem.places
                if place._id == "eligibility_catalog"
            )
            fastest_resources = {
                token.value["resource_ref"]
                for token in eligibility.marking
                if float(token.value["processing_time"]) == 1.0
            }
            self.assertEqual(
                len(fastest_resources),
                config.resources_per_pool,
            )

    def test_initial_streams_are_balanced_over_case_classes(self) -> None:
        for level, config in COMPLEXITY_CONFIGS.items():
            with self.subTest(level=level):
                problem = build_complexity_scenario_problem(
                    "exclusive_choice_disjoint",
                    level,
                    seed=31,
                )
                arrival = next(
                    place for place in problem.places if place._id == "arrival"
                )
                first_classes = set()
                second_classes = set()
                for token in arrival.marking:
                    _case_id, case_class, next_token = _advance_stream(
                        token.value,
                        config,
                        arrival_interval=1.0,
                    )
                    first_classes.add(case_class)
                    _case_id, case_class, _next_token = _advance_stream(
                        next_token.value,
                        config,
                        arrival_interval=1.0,
                    )
                    second_classes.add(case_class)
                self.assertEqual(len(first_classes), config.case_classes)
                self.assertEqual(len(second_classes), config.case_classes)
                self.assertNotEqual(first_classes, set())

    def test_random_actions_terminate_at_finite_horizon(self) -> None:
        for scenario_index, scenario_name in enumerate(PAPER_SCENARIO_NAMES):
            with self.subTest(scenario=scenario_name):
                seed = 1000 + scenario_index
                env = build_bsg_complex_suite_environment(
                    scenario_name,
                    "C3",
                    horizon=3,
                )
                graph = env.reset(seed=seed)
                rng = random.Random(seed)
                for _step_index in range(2000):
                    step = env.step(
                        rng.choice(list(graph.action_keys.values()))
                    )
                    graph = step.observation
                    if step.terminated:
                        break
                else:
                    self.fail(f"Scenario did not terminate: {scenario_name}")
                self.assertGreaterEqual(step.info["completed_tasks"], 0.0)

    def test_bsg_rl_forward_pass_covers_all_extended_schemas(self) -> None:
        graphs = []
        for level in PAPER_COMPLEXITY_LEVELS:
            for scenario_name in PAPER_SCENARIO_NAMES:
                graphs.append(build_bsg_complex_suite_environment(
                    scenario_name,
                    level,
                    horizon=4,
                    validate_graphs=True,
                ).reset(seed=41))
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
            reference_layers=2,
            factor_rounds=0,
        )
        with torch.no_grad():
            output = policy(batch)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.values).all())

    def test_declared_schema_covers_late_runtime_token_shapes(self) -> None:
        calibration_graph = build_bsg_complex_suite_environment(
            "loop_disjoint",
            "C1",
            horizon=100,
            validate_graphs=True,
        ).reset(seed=9_010_000_000)
        schema = GenericGraphFeatureSchema.fit([calibration_graph])
        normalizer = GenericNumericFeatureNormalizer.fit(
            [calibration_graph], schema
        )
        tensorizer = GenericGraphTensorizer(
            schema,
            normalizer=normalizer,
            validate_graphs=True,
        )
        initial_types = {
            node.node_type for node in calibration_graph.nodes.values()
            if node.node_type != "binding"
        }
        declared_only = set(schema.token_types) - initial_types
        self.assertTrue(declared_only)

        seed = 9_020_010_005
        env = build_bsg_complex_suite_environment(
            "loop_disjoint", "C1", horizon=100
        )
        graph = env.reset(seed=seed)
        rng = random.Random(seed)
        observed_types = set()
        for _step_index in range(1000):
            tensorizer.tensorize(graph)
            observed_types.update(
                node.node_type for node in graph.nodes.values()
            )
            step = env.step(
                rng.choice(list(graph.action_keys.values()))
            )
            graph = step.observation
            if step.terminated:
                break
        else:
            self.fail("Late-token schema regression episode did not terminate")
        self.assertTrue(observed_types & declared_only)


if __name__ == "__main__":
    unittest.main()
