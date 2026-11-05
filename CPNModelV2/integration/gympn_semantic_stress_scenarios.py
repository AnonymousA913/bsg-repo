from __future__ import annotations

"""Controlled GymPN scenarios for semantic-component stress tests.

The source-faithful GymPN scenarios remain unchanged.  The two models in this
module are deliberately small identifiability tests:

``directed_dependency``
    Each decision chooses one of two locally indistinguishable candidate
    Tokens.  The rewarded candidate is the source of a directed ``next``
    reference.  Erasing reference propagation or reference direction makes
    the two actions indistinguishable.

``ordered_binding``
    Each decision exposes both orderings of the same two candidate Tokens.
    The rewarded ordering places the lower-ranked Token in the ``source``
    role.  Flat participant pooling receives the same Token multiset for both
    actions and therefore cannot distinguish their order.

``typed_reference``
    Each decision chooses between two locally indistinguishable candidate
    Tokens connected to the same anchor in the same direction. The rewarded
    candidate is distinguished only by the reference-relation type.

These are controlled extensions, not additional source-faithful GymPN paper
benchmarks.  Their purpose is to test whether a representation preserves a
specific semantic distinction, rather than whether it can absorb more nodes.
"""

from dataclasses import dataclass
from functools import partial
import operator
import random
from typing import Any, Callable, Mapping, Optional, Tuple

from gympn.simulator import GymProblem


DIRECTED_DEPENDENCY = "directed_dependency"
ORDERED_BINDING = "ordered_binding"
TYPED_REFERENCE = "typed_reference"
SEMANTIC_STRESS_SCENARIO_NAMES: Tuple[str, ...] = (
    DIRECTED_DEPENDENCY,
    TYPED_REFERENCE,
    ORDERED_BINDING,
)


@dataclass(frozen=True)
class GymPNSemanticStressConfig:
    """Size of one lightweight semantic-identifiability episode."""

    pair_count: int = 8
    counterfactual: bool = False

    def validate(self) -> None:
        try:
            value = operator.index(self.pair_count)
        except TypeError as exc:
            raise TypeError("pair_count must be an integer") from exc
        if value < 1:
            raise ValueError("pair_count must be at least 1")


NetFactory = Callable[[Optional[int]], GymProblem]


def semantic_stress_scenario_factory(
    scenario_name: str,
    *,
    pair_count: int = 8,
    counterfactual: bool = False,
) -> NetFactory:
    """Return the seed-compatible factory expected by ``GymPNDecisionEnv``."""

    _validate_scenario_name(scenario_name)
    config = GymPNSemanticStressConfig(
        pair_count=pair_count,
        counterfactual=counterfactual,
    )
    config.validate()
    return partial(
        build_semantic_stress_problem,
        scenario_name,
        config=config,
    )


def build_semantic_stress_problem(
    scenario_name: str,
    seed: Optional[int] = None,
    *,
    config: Optional[GymPNSemanticStressConfig] = None,
) -> GymProblem:
    """Build one controlled semantic stress scenario."""

    _validate_scenario_name(scenario_name)
    active_config = config or GymPNSemanticStressConfig()
    active_config.validate()
    if scenario_name == DIRECTED_DEPENDENCY:
        return _build_directed_dependency(active_config, seed)
    if scenario_name == TYPED_REFERENCE:
        return _build_typed_reference(active_config, seed)
    if scenario_name == ORDERED_BINDING:
        return _build_ordered_binding(active_config, seed)
    raise AssertionError(f"Unhandled semantic stress scenario: {scenario_name}")


def _new_problem(
    scenario_name: str,
    config: GymPNSemanticStressConfig,
    *,
    stress_target: str,
) -> GymProblem:
    problem = GymProblem(debugging=False, causal_rl=False)
    problem._debugging = False
    problem.gympn_scenario_name = scenario_name
    problem.gympn_case_multiplier = int(config.pair_count)
    problem.gympn_complexity_family = "semantic_stress_v1"
    problem.gympn_semantic_references = True
    problem.gympn_semantic_stress_target = str(stress_target)
    problem.gympn_semantic_stress_pair_count = int(config.pair_count)
    problem.gympn_semantic_stress_counterfactual = bool(config.counterfactual)
    problem.gympn_declared_reference_relations = ()
    problem.gympn_terminal_objective_name = "negative_correct_decisions"
    problem.gympn_terminal_score_name = "correct_decisions"
    problem.gympn_participation_roles = {}
    return problem


def _build_directed_dependency(
    config: GymPNSemanticStressConfig,
    seed: Optional[int],
) -> GymProblem:
    problem = _new_problem(
        DIRECTED_DEPENDENCY,
        config,
        stress_target="reference_direction",
    )
    candidates = problem.add_var(
        "dependency_candidates",
        var_attributes=["object_id", "pair_id", "payload", "next_ref"],
    )
    slots = problem.add_var(
        "dependency_slots",
        var_attributes=["object_id", "pair_id"],
    )
    problem.unobservable_token_attrs[candidates._id] = ["pair_id"]
    problem.unobservable_token_attrs[slots._id] = ["pair_id"]
    problem.gympn_declared_reference_relations = ("next",)
    problem.gympn_declared_token_samples = (
        (
            "dependency_candidates",
            {
                "object_id": "dependency:0:source",
                "pair_id": 0,
                "payload": 0.0,
                "next_ref": "dependency:0:target",
            },
        ),
        (
            "dependency_slots",
            {"object_id": "dependency-slot:0", "pair_id": 0},
        ),
    )

    seed_value = int(seed or 0)
    rng = random.Random(seed_value)
    for pair_id in range(config.pair_count):
        first_id = f"dependency:{pair_id}:a"
        second_id = f"dependency:{pair_id}:b"
        payload = rng.uniform(-1.0, 1.0)
        first = {
            "object_id": first_id,
            "pair_id": pair_id,
            "payload": payload,
            "next_ref": "" if config.counterfactual else second_id,
        }
        second = {
            "object_id": second_id,
            "pair_id": pair_id,
            "payload": payload,
            "next_ref": first_id if config.counterfactual else "",
        }
        # Balance the physical/action ordering within every episode.  A model
        # that cannot read direction must not obtain a spurious advantage from
        # deterministic argmax tie-breaking.
        ordered = (first, second)
        if (seed_value + pair_id) % 2:
            ordered = tuple(reversed(ordered))
        for candidate in ordered:
            candidates.put(candidate)
        slots.put({"object_id": f"dependency-slot:{pair_id}", "pair_id": pair_id})

    def same_pair(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> bool:
        return int(candidate["pair_id"]) == int(slot["pair_id"])

    def select_candidate(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ):
        del candidate, slot
        return []

    def dependency_reward(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> float:
        del slot
        return float(bool(str(candidate["next_ref"])))

    transition_name = "select_dependency_source"
    problem.add_action(
        [candidates, slots],
        [],
        select_candidate,
        name=transition_name,
        guard=same_pair,
        reward_function=dependency_reward,
    )
    problem.gympn_participation_roles[transition_name] = (
        "consume::candidate",
        "consume::pair_slot",
    )
    return problem


def _build_typed_reference(
    config: GymPNSemanticStressConfig,
    seed: Optional[int],
) -> GymProblem:
    problem = _new_problem(
        TYPED_REFERENCE,
        config,
        stress_target="reference_type",
    )
    candidates = problem.add_var(
        "typed_candidates",
        var_attributes=[
            "object_id", "pair_id", "payload", "supports_ref", "blocks_ref",
        ],
    )
    anchors = problem.add_var(
        "typed_anchors",
        var_attributes=["object_id", "pair_id", "context"],
    )
    slots = problem.add_var(
        "typed_slots",
        var_attributes=["object_id", "pair_id"],
    )
    problem.unobservable_token_attrs[candidates._id] = ["pair_id"]
    problem.unobservable_token_attrs[anchors._id] = ["pair_id"]
    problem.unobservable_token_attrs[slots._id] = ["pair_id"]
    problem.gympn_declared_reference_relations = ("supports", "blocks")
    problem.gympn_declared_token_samples = (
        (
            "typed_candidates",
            {
                "object_id": "typed:0:a",
                "pair_id": 0,
                "payload": 0.0,
                "supports_ref": "typed-anchor:0",
                "blocks_ref": "",
            },
        ),
        (
            "typed_anchors",
            {"object_id": "typed-anchor:0", "pair_id": 0, "context": 0.0},
        ),
        (
            "typed_slots",
            {"object_id": "typed-slot:0", "pair_id": 0},
        ),
    )

    seed_value = int(seed or 0)
    rng = random.Random(seed_value)
    for pair_id in range(config.pair_count):
        anchor_id = f"typed-anchor:{pair_id}"
        first_id = f"typed:{pair_id}:a"
        second_id = f"typed:{pair_id}:b"
        payload = rng.uniform(-1.0, 1.0)
        context = rng.uniform(-1.0, 1.0)
        first_supports = not config.counterfactual
        first = {
            "object_id": first_id,
            "pair_id": pair_id,
            "payload": payload,
            "supports_ref": anchor_id if first_supports else "",
            "blocks_ref": "" if first_supports else anchor_id,
        }
        second = {
            "object_id": second_id,
            "pair_id": pair_id,
            "payload": payload,
            "supports_ref": "" if first_supports else anchor_id,
            "blocks_ref": anchor_id if first_supports else "",
        }
        ordered = (first, second)
        if (seed_value + pair_id) % 2:
            ordered = tuple(reversed(ordered))
        for candidate in ordered:
            candidates.put(candidate)
        anchors.put({
            "object_id": anchor_id,
            "pair_id": pair_id,
            "context": context,
        })
        slots.put({"object_id": f"typed-slot:{pair_id}", "pair_id": pair_id})

    def same_pair(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> bool:
        return int(candidate["pair_id"]) == int(slot["pair_id"])

    def select_candidate(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ):
        del candidate, slot
        return []

    def typed_reward(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> float:
        del slot
        return float(bool(str(candidate["supports_ref"])))

    transition_name = "select_supporting_reference"
    problem.add_action(
        [candidates, slots],
        [],
        select_candidate,
        name=transition_name,
        guard=same_pair,
        reward_function=typed_reward,
    )
    problem.gympn_participation_roles[transition_name] = (
        "consume::candidate",
        "consume::pair_slot",
    )
    return problem


def _build_ordered_binding(
    config: GymPNSemanticStressConfig,
    seed: Optional[int],
) -> GymProblem:
    problem = _new_problem(
        ORDERED_BINDING,
        config,
        stress_target="binding_role_order",
    )
    candidates = problem.add_var(
        "ordered_candidates",
        var_attributes=["object_id", "pair_id", "rank"],
    )
    slots = problem.add_var(
        "ordered_slots",
        var_attributes=["object_id", "pair_id"],
    )
    problem.unobservable_token_attrs[candidates._id] = ["pair_id"]
    problem.unobservable_token_attrs[slots._id] = ["pair_id"]
    problem.gympn_declared_token_samples = (
        (
            "ordered_candidates",
            {"object_id": "ordered:0:a", "pair_id": 0, "rank": 0.0},
        ),
        (
            "ordered_slots",
            {"object_id": "ordered-slot:0", "pair_id": 0},
        ),
    )

    seed_value = int(seed or 0)
    rng = random.Random(seed_value)
    for pair_id in range(config.pair_count):
        first_rank = rng.uniform(-1.0, 1.0)
        second_rank = rng.uniform(-1.0, 1.0)
        while abs(first_rank - second_rank) < 0.05:
            second_rank = rng.uniform(-1.0, 1.0)
        if config.counterfactual:
            first_rank, second_rank = second_rank, first_rank
        first = {
            "object_id": f"ordered:{pair_id}:a",
            "pair_id": pair_id,
            "rank": first_rank,
        }
        second = {
            "object_id": f"ordered:{pair_id}:b",
            "pair_id": pair_id,
            "rank": second_rank,
        }
        ordered = (first, second)
        if (seed_value + pair_id) % 2:
            ordered = tuple(reversed(ordered))
        for candidate in ordered:
            candidates.put(candidate)
        slots.put({"object_id": f"ordered-slot:{pair_id}", "pair_id": pair_id})

    def valid_ordering(
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> bool:
        return (
            str(source["object_id"]) != str(target["object_id"])
            and int(source["pair_id"]) == int(target["pair_id"])
            and int(source["pair_id"]) == int(slot["pair_id"])
        )

    def execute_ordering(
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        slot: Mapping[str, Any],
    ):
        del source, target, slot
        return []

    def ordering_reward(
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> float:
        del slot
        return float(float(source["rank"]) < float(target["rank"]))

    transition_name = "select_ordered_binding"
    problem.add_action(
        [candidates, candidates, slots],
        [],
        execute_ordering,
        name=transition_name,
        guard=valid_ordering,
        reward_function=ordering_reward,
    )
    problem.gympn_participation_roles[transition_name] = (
        "consume::source",
        "consume::target",
        "consume::pair_slot",
    )
    return problem


def _validate_scenario_name(value: str) -> None:
    if value not in SEMANTIC_STRESS_SCENARIO_NAMES:
        choices = ", ".join(SEMANTIC_STRESS_SCENARIO_NAMES)
        raise ValueError(
            f"Unknown GymPN semantic stress scenario {value!r}; choose {choices}"
        )


__all__ = [
    "DIRECTED_DEPENDENCY",
    "GymPNSemanticStressConfig",
    "ORDERED_BINDING",
    "SEMANTIC_STRESS_SCENARIO_NAMES",
    "TYPED_REFERENCE",
    "build_semantic_stress_problem",
    "semantic_stress_scenario_factory",
]
