from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Tuple

from CPNModelV2.integration.gympn_scenarios import paper_scenario_factory
from CPNModelV2.integration.gympn_decision_adapter import GymPNDecisionEnv
from CPNModelV2.integration.gympn_semantic_stress_scenarios import (
    semantic_stress_scenario_factory,
)


def build_paper_suite_environment(
    scenario_name: str,
    *,
    horizon: float,
    case_multiplier: int = 1,
) -> GymPNDecisionEnv:
    return GymPNDecisionEnv(
        paper_scenario_factory(
            scenario_name,
            case_multiplier=case_multiplier,
        ),
        horizon=horizon,
    )


def build_semantic_stress_environment(
    scenario_name: str,
    *,
    pair_count: int = 8,
    horizon: float = 10.0,
    counterfactual: bool = False,
) -> GymPNDecisionEnv:
    """Build one controlled semantic-identifiability environment."""

    return GymPNDecisionEnv(
        semantic_stress_scenario_factory(
            scenario_name,
            pair_count=pair_count,
            counterfactual=counterfactual,
        ),
        horizon=horizon,
    )


def shortest_effect_action(graph) -> Tuple:
    """Select the Binding with the shortest previewed availability delay."""

    if not graph.action_keys:
        raise ValueError("Cannot select an action from an empty action set")
    ranked = []
    for binding_node_id, action_key in graph.action_keys.items():
        effect = graph.binding_effects[binding_node_id]
        delays = [
            max(
                0.0,
                float(token_effect.numeric_delta.get(
                    "available_time_delta", 0.0
                )),
            )
            for token_effect in effect.token_effects
        ]
        ranked.append((max(delays, default=0.0), binding_node_id, action_key))
    return min(ranked, key=lambda item: (item[0], item[1]))[2]


def evaluate_rule(
    scenario_name: str,
    *,
    horizon: float,
    seeds: Iterable[int],
    rule: str,
    case_multiplier: int = 1,
) -> List[float]:
    scores = []
    for seed in seeds:
        env = build_paper_suite_environment(
            scenario_name,
            horizon=horizon,
            case_multiplier=case_multiplier,
        )
        graph = env.reset(seed=int(seed))
        rng = random.Random(int(seed))
        while True:
            if rule == "random":
                action = rng.choice(list(graph.action_keys.values()))
            elif rule == "shortest_effect":
                action = shortest_effect_action(graph)
            else:
                raise ValueError(f"Unknown GymPN rule: {rule}")
            step = env.step(action)
            graph = step.observation
            if step.terminated:
                scores.append(float(step.info["completed_tasks"]))
                break
    return scores


def macro_average(values_by_scenario: Mapping[str, Iterable[float]]) -> float:
    scenario_means = []
    for values in values_by_scenario.values():
        sequence = list(values)
        if not sequence:
            raise ValueError("Every scenario must contain at least one score")
        scenario_means.append(sum(sequence) / len(sequence))
    if not scenario_means:
        raise ValueError("At least one scenario is required")
    return sum(scenario_means) / len(scenario_means)
