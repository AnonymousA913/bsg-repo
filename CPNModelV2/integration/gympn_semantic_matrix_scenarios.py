from __future__ import annotations

"""Mixed GymPN scenarios for the scale--semantic-ambiguity matrix.

The formal matrix varies two independent factors:

* the existing Section 5.6 structural scale, L1--L5; and
* the fraction of decision groups for which local attributes are insufficient.

Every episode mixes reference-direction, reference-type, and binding-role
decisions.  At an unambiguous group a visible local priority is sufficient.  At
an ambiguous group the same executable information is retained, but the
correct choice additionally depends on the corresponding CPN semantics.  The
ambiguity sets are nested for a fixed seed, which makes comparisons between
ratios paired rather than independent.
"""

from dataclasses import dataclass
from functools import partial
import hashlib
import operator
import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from gympn.simulator import GymProblem
from simpn.simulator import SimToken

from CPNModelV2.integration.gympn_complex_scenarios import (
    PAPER_COMPLEXITY_LABELS,
    PAPER_COMPLEXITY_LEVELS,
)


SEMANTIC_MATRIX_SCENARIO = "mixed_semantic"
REFERENCE_DIRECTION = "reference_direction"
REFERENCE_TYPE = "reference_type"
BINDING_ROLE = "binding_role"
SEMANTIC_FAMILIES: Tuple[str, ...] = (
    REFERENCE_DIRECTION,
    REFERENCE_TYPE,
    BINDING_ROLE,
)
FORMAL_AMBIGUITY_RATIOS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class GymPNSemanticMatrixScale:
    """One structural point on the L1--L5 matrix axis."""

    level: str
    decision_groups: int
    typed_candidates: int
    ordered_candidates: int
    context_tokens_per_group: int

    @property
    def paper_label(self) -> str:
        return PAPER_COMPLEXITY_LABELS[self.level]

    def validate(self) -> None:
        if self.level not in PAPER_COMPLEXITY_LEVELS:
            raise ValueError(f"Unknown paper complexity level: {self.level}")
        values = (
            self.decision_groups,
            self.typed_candidates,
            self.ordered_candidates,
            self.context_tokens_per_group,
        )
        if any(operator.index(value) < 1 for value in values):
            raise ValueError("Semantic-matrix scale values must be positive")
        if self.typed_candidates < 2 or self.ordered_candidates < 2:
            raise ValueError("Semantic decisions require at least two candidates")


SEMANTIC_MATRIX_SCALES: Dict[str, GymPNSemanticMatrixScale] = {
    "C1": GymPNSemanticMatrixScale("C1", 4, 2, 2, 1),
    "C1_5": GymPNSemanticMatrixScale("C1_5", 8, 3, 3, 1),
    "C2": GymPNSemanticMatrixScale("C2", 12, 4, 4, 2),
    "C2_25": GymPNSemanticMatrixScale("C2_25", 16, 5, 5, 3),
    "C2_5": GymPNSemanticMatrixScale("C2_5", 20, 6, 6, 4),
}


@dataclass(frozen=True)
class GymPNSemanticMatrixConfig:
    """Configuration of one matrix episode distribution."""

    complexity_level: str
    ambiguity_ratios: Tuple[float, ...] = FORMAL_AMBIGUITY_RATIOS

    def validate(self) -> None:
        if self.complexity_level not in SEMANTIC_MATRIX_SCALES:
            choices = ", ".join(PAPER_COMPLEXITY_LEVELS)
            raise ValueError(
                f"Unknown semantic-matrix level {self.complexity_level!r}; "
                f"choose {choices}"
            )
        if not self.ambiguity_ratios:
            raise ValueError("At least one ambiguity ratio is required")
        for ratio in self.ambiguity_ratios:
            _validate_ratio(ratio)


NetFactory = Callable[[Optional[int]], GymProblem]


def semantic_matrix_scenario_factory(
    complexity_level: str,
    *,
    ambiguity_ratios: Sequence[float] = FORMAL_AMBIGUITY_RATIOS,
) -> NetFactory:
    """Return a seed-compatible mixed-semantic GymPN factory.

    A single ratio creates a fixed matrix cell.  Multiple ratios create a
    deterministic uniform training mixture: the episode seed selects one ratio
    without changing the model or checkpoint.
    """

    config = GymPNSemanticMatrixConfig(
        complexity_level=str(complexity_level),
        ambiguity_ratios=tuple(float(value) for value in ambiguity_ratios),
    )
    config.validate()
    return partial(build_semantic_matrix_problem, config=config)


def select_ambiguity_ratio(
    seed: Optional[int],
    ratios: Sequence[float],
) -> float:
    """Select a reproducible, approximately uniform ratio from a distribution."""

    values = tuple(float(value) for value in ratios)
    if not values:
        raise ValueError("At least one ambiguity ratio is required")
    for value in values:
        _validate_ratio(value)
    if len(values) == 1:
        return values[0]
    # The shared runner allocates consecutive episode seeds.  Modulo selection
    # therefore gives every block of five episodes one instance from each
    # formal ratio, rather than merely balancing them in expectation.
    return values[int(seed or 0) % len(values)]


def ambiguity_group_ids(
    seed: Optional[int],
    decision_groups: int,
    ambiguity_ratio: float,
) -> Tuple[int, ...]:
    """Return an exactly sized, nested ambiguity set for a fixed seed.

    Groups are ranked once by a stable seed-dependent hash.  Increasing the
    ratio only adds groups and never removes previously ambiguous groups.
    """

    if decision_groups < 1:
        raise ValueError("decision_groups must be positive")
    ratio = _validate_ratio(ambiguity_ratio)
    count = int(round(ratio * decision_groups))
    ranked = sorted(
        range(decision_groups),
        key=lambda group_id: _stable_rank(seed, group_id),
    )
    return tuple(sorted(ranked[:count]))


def semantic_family_for_group(seed: Optional[int], group_id: int) -> str:
    """Assign balanced semantic families while rotating their seed alignment."""

    offset = _stable_rank(seed, -1) % len(SEMANTIC_FAMILIES)
    return SEMANTIC_FAMILIES[(int(group_id) + offset) % len(SEMANTIC_FAMILIES)]


def build_semantic_matrix_problem(
    seed: Optional[int] = None,
    *,
    config: GymPNSemanticMatrixConfig,
) -> GymProblem:
    """Build one paired scale--ambiguity episode."""

    config.validate()
    scale = SEMANTIC_MATRIX_SCALES[config.complexity_level]
    scale.validate()
    ratio = select_ambiguity_ratio(seed, config.ambiguity_ratios)
    ambiguous_ids = set(
        ambiguity_group_ids(seed, scale.decision_groups, ratio)
    )
    seed_value = int(seed or 0)
    rng = random.Random(seed_value)

    problem = GymProblem(debugging=False, causal_rl=False)
    problem._debugging = False
    problem.gympn_scenario_name = SEMANTIC_MATRIX_SCENARIO
    problem.gympn_case_multiplier = 1
    problem.gympn_complexity_level = config.complexity_level
    problem.gympn_paper_complexity_level = scale.paper_label
    problem.gympn_complexity_family = "semantic_ambiguity_matrix_v1"
    problem.gympn_semantic_references = True
    problem.gympn_declared_reference_relations = (
        "next", "supports", "blocks",
    )
    problem.gympn_terminal_objective_name = "negative_correct_decisions"
    problem.gympn_terminal_score_name = "correct_decisions"
    problem.gympn_participation_roles = {}
    problem.gympn_semantic_ambiguity_ratio = ratio
    problem.gympn_semantic_ambiguity_count = len(ambiguous_ids)
    problem.gympn_semantic_ambiguity_groups = tuple(sorted(ambiguous_ids))
    problem.gympn_semantic_decision_groups = scale.decision_groups
    problem.gympn_semantic_families = SEMANTIC_FAMILIES

    direction_candidates = problem.add_var(
        "direction_candidates",
        var_attributes=[
            "object_id", "group_id", "local_priority", "next_ref",
        ],
    )
    direction_slots = problem.add_var(
        "direction_slots",
        var_attributes=["object_id", "group_id"],
    )
    type_candidates = problem.add_var(
        "type_candidates",
        var_attributes=[
            "object_id", "group_id", "local_priority",
            "supports_ref", "blocks_ref",
        ],
    )
    type_anchors = problem.add_var(
        "type_anchors",
        var_attributes=["object_id", "group_id", "context"],
    )
    type_slots = problem.add_var(
        "type_slots",
        var_attributes=["object_id", "group_id"],
    )
    role_candidates = problem.add_var(
        "role_candidates",
        var_attributes=[
            "object_id", "group_id", "local_priority", "rank",
        ],
    )
    role_slots = problem.add_var(
        "role_slots",
        var_attributes=["object_id", "group_id"],
    )
    context = problem.add_var(
        "semantic_context",
        var_attributes=["object_id", "group_id", "context"],
    )
    completed = problem.add_var(
        "completed_decisions",
        var_attributes=["object_id", "group_id", "success"],
    )

    for place in (
        direction_candidates,
        direction_slots,
        type_candidates,
        type_anchors,
        type_slots,
        role_candidates,
        role_slots,
        context,
        completed,
    ):
        problem.unobservable_token_attrs[place._id] = ["group_id"]

    problem.gympn_declared_token_samples = (
        (
            "direction_candidates",
            {
                "object_id": "direction:0:a",
                "group_id": 0,
                "local_priority": 0.0,
                "next_ref": "direction:0:b",
            },
        ),
        (
            "direction_slots",
            {"object_id": "direction-slot:0", "group_id": 0},
        ),
        (
            "type_candidates",
            {
                "object_id": "type:0:a",
                "group_id": 0,
                "local_priority": 0.0,
                "supports_ref": "type-anchor:0",
                "blocks_ref": "",
            },
        ),
        (
            "type_anchors",
            {"object_id": "type-anchor:0", "group_id": 0, "context": 0.0},
        ),
        ("type_slots", {"object_id": "type-slot:0", "group_id": 0}),
        (
            "role_candidates",
            {
                "object_id": "role:0:a",
                "group_id": 0,
                "local_priority": 0.0,
                "rank": 0.0,
            },
        ),
        ("role_slots", {"object_id": "role-slot:0", "group_id": 0}),
        (
            "semantic_context",
            {"object_id": "context:0:0", "group_id": 0, "context": 0.0},
        ),
        (
            "completed_decisions",
            {"object_id": "completed:0", "group_id": 0, "success": 1.0},
        ),
    )

    for group_id in range(scale.decision_groups):
        family = semantic_family_for_group(seed, group_id)
        is_ambiguous = group_id in ambiguous_ids
        for context_index in range(scale.context_tokens_per_group):
            context.put({
                "object_id": f"context:{group_id}:{context_index}",
                "group_id": group_id,
                "context": rng.uniform(-1.0, 1.0),
            })
        if family == REFERENCE_DIRECTION:
            _seed_direction_group(
                direction_candidates,
                direction_slots,
                group_id=group_id,
                ambiguous=is_ambiguous,
                rng=rng,
            )
        elif family == REFERENCE_TYPE:
            _seed_type_group(
                type_candidates,
                type_anchors,
                type_slots,
                group_id=group_id,
                candidate_count=scale.typed_candidates,
                ambiguous=is_ambiguous,
                rng=rng,
            )
        else:
            _seed_role_group(
                role_candidates,
                role_slots,
                group_id=group_id,
                candidate_count=scale.ordered_candidates,
                ambiguous=is_ambiguous,
                rng=rng,
            )

    def same_group(candidate: Mapping[str, Any], slot: Mapping[str, Any]) -> bool:
        return int(candidate["group_id"]) == int(slot["group_id"])

    def choose_direction(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ):
        group_id = int(slot["group_id"])
        success = (
            bool(str(candidate["next_ref"]))
            if group_id in ambiguous_ids
            else float(candidate["local_priority"]) > 0.5
        )
        return [_completion_token(group_id, success)]

    def direction_reward(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> float:
        group_id = int(slot["group_id"])
        if group_id in ambiguous_ids:
            return float(bool(str(candidate["next_ref"])))
        return float(float(candidate["local_priority"]) > 0.5)

    direction_name = "select_direction_candidate"
    problem.add_action(
        [direction_candidates, direction_slots],
        [completed],
        choose_direction,
        name=direction_name,
        guard=same_group,
        reward_function=direction_reward,
    )
    problem.gympn_participation_roles[direction_name] = (
        "consume::candidate", "consume::decision_slot",
    )

    def choose_type(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ):
        group_id = int(slot["group_id"])
        success = (
            bool(str(candidate["supports_ref"]))
            if group_id in ambiguous_ids
            else float(candidate["local_priority"]) > 0.5
        )
        return [_completion_token(group_id, success)]

    def type_reward(
        candidate: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> float:
        group_id = int(slot["group_id"])
        if group_id in ambiguous_ids:
            return float(bool(str(candidate["supports_ref"])))
        return float(float(candidate["local_priority"]) > 0.5)

    type_name = "select_typed_candidate"
    problem.add_action(
        [type_candidates, type_slots],
        [completed],
        choose_type,
        name=type_name,
        guard=same_group,
        reward_function=type_reward,
    )
    problem.gympn_participation_roles[type_name] = (
        "consume::candidate", "consume::decision_slot",
    )

    def valid_role_order(
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> bool:
        return (
            str(source["object_id"]) != str(target["object_id"])
            and int(source["group_id"]) == int(target["group_id"])
            and int(source["group_id"]) == int(slot["group_id"])
        )

    def role_success(
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        slot: Mapping[str, Any],
    ) -> bool:
        group_id = int(slot["group_id"])
        selected_priorities = sorted((
            float(source["local_priority"]),
            float(target["local_priority"]),
        ))
        # The two selected Tokens must be the visible top-priority pair.  This
        # condition is symmetric and therefore remains available to Flat.
        correct_pair = selected_priorities[0] > 0.5
        if not correct_pair:
            return False
        if group_id not in ambiguous_ids:
            return True
        # Only ambiguous groups additionally require the ordered source/target
        # role semantics preserved by the full binding composer.
        return float(source["rank"]) < float(target["rank"])

    def choose_role(
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        slot: Mapping[str, Any],
    ):
        group_id = int(slot["group_id"])
        return [_completion_token(group_id, role_success(source, target, slot))]

    role_name = "select_ordered_binding"
    problem.add_action(
        [role_candidates, role_candidates, role_slots],
        [completed],
        choose_role,
        name=role_name,
        guard=valid_role_order,
        reward_function=lambda source, target, slot: float(
            role_success(source, target, slot)
        ),
    )
    problem.gympn_participation_roles[role_name] = (
        "consume::source", "consume::target", "consume::decision_slot",
    )
    return problem


def _seed_direction_group(
    candidates: Any,
    slots: Any,
    *,
    group_id: int,
    ambiguous: bool,
    rng: random.Random,
) -> None:
    first_id = f"direction:{group_id}:a"
    second_id = f"direction:{group_id}:b"
    direction_source = rng.randrange(2)
    local_target = rng.randrange(2)
    values = []
    for index, object_id in enumerate((first_id, second_id)):
        other_id = second_id if index == 0 else first_id
        values.append({
            "object_id": object_id,
            "group_id": group_id,
            "local_priority": (
                0.0 if ambiguous else float(index == local_target)
            ),
            "next_ref": other_id if index == direction_source else "",
        })
    if rng.random() < 0.5:
        values.reverse()
    for value in values:
        candidates.put(value)
    slots.put({"object_id": f"direction-slot:{group_id}", "group_id": group_id})


def _seed_type_group(
    candidates: Any,
    anchors: Any,
    slots: Any,
    *,
    group_id: int,
    candidate_count: int,
    ambiguous: bool,
    rng: random.Random,
) -> None:
    anchor_id = f"type-anchor:{group_id}"
    support_index = rng.randrange(candidate_count)
    local_target = rng.randrange(candidate_count)
    values = []
    for index in range(candidate_count):
        supports = index == support_index
        values.append({
            "object_id": f"type:{group_id}:{index}",
            "group_id": group_id,
            "local_priority": (
                0.0 if ambiguous else float(index == local_target)
            ),
            "supports_ref": anchor_id if supports else "",
            "blocks_ref": "" if supports else anchor_id,
        })
    rng.shuffle(values)
    for value in values:
        candidates.put(value)
    anchors.put({
        "object_id": anchor_id,
        "group_id": group_id,
        "context": rng.uniform(-1.0, 1.0),
    })
    slots.put({"object_id": f"type-slot:{group_id}", "group_id": group_id})


def _seed_role_group(
    candidates: Any,
    slots: Any,
    *,
    group_id: int,
    candidate_count: int,
    ambiguous: bool,
    rng: random.Random,
) -> None:
    del ambiguous  # Ambiguity changes the reward rule, not visible Token facts.
    top_pair = set(rng.sample(range(candidate_count), 2))
    values = []
    ranks = rng.sample(range(candidate_count * 4), candidate_count)
    for index in range(candidate_count):
        values.append({
            "object_id": f"role:{group_id}:{index}",
            "group_id": group_id,
            "local_priority": 1.0 if index in top_pair else 0.0,
            "rank": float(ranks[index]) / float(max(1, candidate_count * 4 - 1)),
        })
    rng.shuffle(values)
    for value in values:
        candidates.put(value)
    slots.put({"object_id": f"role-slot:{group_id}", "group_id": group_id})


def _completion_token(group_id: int, success: bool) -> SimToken:
    return SimToken({
        "object_id": f"completed:{group_id}",
        "group_id": int(group_id),
        "success": float(bool(success)),
    })


def _stable_rank(seed: Optional[int], group_id: int) -> int:
    payload = f"semantic-matrix:{int(seed or 0)}:{int(group_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_ratio(value: float) -> float:
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Semantic ambiguity ratio must lie in [0, 1]")
    return ratio


__all__ = [
    "BINDING_ROLE",
    "FORMAL_AMBIGUITY_RATIOS",
    "GymPNSemanticMatrixConfig",
    "GymPNSemanticMatrixScale",
    "REFERENCE_DIRECTION",
    "REFERENCE_TYPE",
    "SEMANTIC_FAMILIES",
    "SEMANTIC_MATRIX_SCALES",
    "SEMANTIC_MATRIX_SCENARIO",
    "ambiguity_group_ids",
    "build_semantic_matrix_problem",
    "select_ambiguity_ratio",
    "semantic_family_for_group",
    "semantic_matrix_scenario_factory",
]
