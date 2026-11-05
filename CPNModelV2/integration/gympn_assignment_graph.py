from __future__ import annotations

"""Controlled GymPN Assignment Graph adapter for the Section 5.6 baseline.

GymPN's published graph observation assumes that every token attribute is
numeric.  The relational-complexity suite deliberately uses explicit object
identifiers and references.  This module makes a deep observation-only copy,
encodes those identifiers deterministically, and then calls GymPN's original
expanded Assignment Graph builder.  The executable problem and its tokens are
never changed.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from types import MethodType
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch_geometric.data import HeteroData
from gympn.simulator import GymProblem
from simpn.simulator import SimVar, SimVarQueue

from CPNModelV2.graph.schema import GraphNode, TokenBindingGraph
from CPNModelV2.integration.bpm2024_assignment_graph import (
    AssignmentGraphFeatureSchema,
    ControlledAssignmentGraphPolicy,
)
from CPNModelV2.integration.gympn_complex_scenarios import (
    complexity_scenario_factory,
)
from CPNModelV2.integration.gympn_decision_adapter import (
    GYMPN_ACTION_KEY,
    GymPNDecisionEnv,
)


_STRUCTURED_ID = re.compile(r"^(?P<kind>[a-z_]+):(?P<body>.+)$")
_KIND_CODES = {
    "flow": 1,
    "activity": 2,
    "pool": 3,
    "resource": 4,
    "eligibility": 5,
    "case": 6,
}
_SYMBOL_CODES = {
    "": 0,
    "sequence": 1,
    "parallel": 2,
    "loop": 3,
    "exclusive_choice": 4,
    "joint": 5,
    "disjoint": 6,
}
_NESTED_ATTRIBUTE_FIELDS = {
    "work": (
        "object_id", "case_id", "case_class", "activity_ref", "rework_count",
    ),
    "resource": ("object_id", "resource_id", "pool_ref"),
    "eligibility": (
        "object_id", "activity_ref", "resource_ref", "case_class",
        "processing_time", "quality",
    ),
}


class GymPNAssignmentGraphBuilder:
    """Expose GymPN's expanded Assignment Graph through the common action API."""

    def __init__(self, *, validate: bool = True) -> None:
        self.validate = bool(validate)

    def build(self, problem: Any, *, decision_id: int = 0) -> TokenBindingGraph:
        executable_actions = list(getattr(problem, "pn_actions", ()))
        source = _numeric_graph_observation(problem)
        assignment_graph = _canonicalize_assignment_graph(
            source["graph"], problem
        )
        observed_actions = list(source["actions_dict"])
        mapping = _match_observed_actions(executable_actions, observed_actions)
        raw_action_node_count = int(
            assignment_graph["a_transition"].num_nodes
        )
        if raw_action_node_count != len(observed_actions):
            raise ValueError(
                "GymPN Assignment Graph action-node count differs from its "
                f"action map: {raw_action_node_count} != "
                f"{len(observed_actions)}"
            )
        assignment_graph = _select_executable_action_nodes(
            assignment_graph, mapping
        )

        graph = TokenBindingGraph(
            time=float(problem.clock),
            metadata={
                "observation": "gympn_assignment_graph",
                "adapter": type(self).__name__,
                "cpn_family": "action_evolution_petri_net",
                "scenario_name": str(
                    getattr(problem, "gympn_scenario_name", "unspecified")
                ),
                "complexity_level": str(
                    getattr(problem, "gympn_complexity_level", "unspecified")
                ),
                "identifier_encoding": "deterministic_observation_copy_v1",
                "executable_problem_mutated": False,
            },
        )
        action_node_indices: List[int] = []
        for executable_index, action_node_index in enumerate(range(len(mapping))):
            transition = executable_actions[executable_index][2]
            node_id = f"binding:{action_node_index:06d}"
            graph.add_node(GraphNode(
                node_id=node_id,
                node_type="binding",
                identity={
                    "decision_id": int(decision_id),
                    "binding_index": int(executable_index),
                    "assignment_action_index": int(action_node_index),
                },
            ))
            graph.action_keys[node_id] = (
                GYMPN_ACTION_KEY,
                int(decision_id),
                int(executable_index),
                str(transition._id),
            )
            action_node_indices.append(int(action_node_index))

        if "a_transition" not in assignment_graph.node_types:
            if executable_actions:
                raise ValueError("GymPN Assignment Graph contains no action nodes")
            # GymPN omits this node type at a terminal marking.  The common
            # environment still returns a terminal observation, so provide a
            # correctly shaped empty placeholder; PPO never selects from it.
            assignment_graph["a_transition"].x = torch.empty(
                (0, len(problem.actions)), dtype=torch.float32
            )
        action_node_count = int(assignment_graph["a_transition"].num_nodes)
        if action_node_count != len(executable_actions):
            raise ValueError("Filtered Assignment Graph action count changed")
        graph.assignment_graph = assignment_graph
        graph.assignment_action_indices = tuple(action_node_indices)
        if self.validate:
            graph.validate()
        return graph


class GymPNAssignmentGraphEnv(GymPNDecisionEnv):
    """Decision-epoch GymPN environment for the controlled AG_PPO baseline."""

    def __init__(
        self,
        net_factory: Any,
        *,
        horizon: float = 100.0,
        validate_graphs: bool = False,
    ) -> None:
        super().__init__(
            net_factory,
            horizon=horizon,
            graph_builder=GymPNAssignmentGraphBuilder(validate=validate_graphs),
        )


def build_ag_complex_suite_environment(
    scenario_name: str,
    complexity_level: str,
    *,
    horizon: float = 100.0,
    validate_graphs: bool = False,
) -> GymPNAssignmentGraphEnv:
    return GymPNAssignmentGraphEnv(
        complexity_scenario_factory(scenario_name, complexity_level),
        horizon=horizon,
        validate_graphs=validate_graphs,
    )


@dataclass(frozen=True)
class AssignmentGraphNoOpNormalizer:
    """Compatibility object; AG statistics live inside its feature schema."""

    format: str = "assignment_graph_schema_normalization_v1"

    @classmethod
    def fit(
        cls,
        _graphs: Iterable[TokenBindingGraph],
        _schema: AssignmentGraphFeatureSchema,
    ) -> "AssignmentGraphNoOpNormalizer":
        return cls()

    def to_dict(self) -> Dict[str, str]:
        return {"format": self.format}

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        _schema: AssignmentGraphFeatureSchema,
    ) -> "AssignmentGraphNoOpNormalizer":
        value = str(values.get("format", ""))
        if value != cls().format:
            raise ValueError(f"Unsupported Assignment Graph normalizer: {value}")
        return cls()


class GymPNControlledAssignmentGraphPolicy(ControlledAssignmentGraphPolicy):
    """HAN policy accepting the common runner's representation arguments."""

    def __init__(
        self,
        schema: AssignmentGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 0,
        factor_rounds: int = 0,
        dropout: float = 0.0,
    ) -> None:
        # Unused values remain explicit in checkpoints, making the common PPO
        # protocol directly auditable across BSG_RL and AG_PPO.
        self.category_dim = int(category_dim)
        self.reference_layers = int(reference_layers)
        self.factor_rounds = int(factor_rounds)
        super().__init__(schema, hidden_dim=hidden_dim, dropout=dropout)


def _numericize_problem_tokens(problem: Any) -> None:
    for place in problem.places:
        attributes = list(problem.var_attributes.get(place._id, ()))
        expanded_attributes = []
        for attribute in attributes:
            nested_fields = _NESTED_ATTRIBUTE_FIELDS.get(str(attribute))
            if nested_fields is None:
                expanded_attributes.append(attribute)
            else:
                expanded_attributes.extend(
                    f"{attribute}__{field}" for field in nested_fields
                )
        if expanded_attributes:
            problem.var_attributes[place._id] = expanded_attributes
        for token in place.marking:
            token.value = _numericize_token_value(token.value)


_MISSING = object()


def _numeric_graph_observation(problem: Any) -> Mapping[str, Any]:
    """Build GymPN's numeric graph without copying or mutating the executor.

    Some SimPN releases implement dynamic variable access through ``__getattr__``
    and cannot be deep-copied safely.  The graph builder is synchronous, so we
    snapshot the small set of fields touched by numericization and by
    ``get_graph_observation``, restore them in ``finally``, and return only the
    detached tensor graph plus its observation-time action list.
    """

    token_values = []
    seen_tokens = set()
    for place in problem.places:
        for token in place.marking:
            token_identity = id(token)
            if token_identity in seen_tokens:
                continue
            seen_tokens.add(token_identity)
            token_values.append((token, token.value))
    original_var_attributes = dict(problem.var_attributes)
    state_fields = (
        "pn_actions", "expanded_pn", "metadata",
        "expand_no_future_tokens",
    )
    original_state = {
        name: problem.__dict__.get(name, _MISSING) for name in state_fields
    }
    try:
        _numericize_problem_tokens(problem)
        if _has_repeated_action_input_place(problem):
            # Released GymPN versions collapse repeated occurrences of one
            # input place through a dictionary keyed by place id.  That makes
            # ordered source/target bindings impossible to match to Assignment
            # Graph nodes.  Install an observation-only expansion for this
            # case; the executable CPN and its action order remain untouched.
            problem.expand_no_future_tokens = MethodType(
                _expand_observed_bindings_preserving_roles,
                problem,
            )
        return problem.get_graph_observation(
            normalize=False,
            remove_empty_nodes=True,
            add_self_loops=True,
        )
    finally:
        for token, original_value in token_values:
            token.value = original_value
        problem.var_attributes.clear()
        problem.var_attributes.update(original_var_attributes)
        for name, original_value in original_state.items():
            if original_value is _MISSING:
                problem.__dict__.pop(name, None)
            else:
                problem.__dict__[name] = original_value


def _has_repeated_action_input_place(problem: Any) -> bool:
    for transition in problem.actions:
        place_ids = [
            str(place._id)
            for place in transition.incoming
            if isinstance(place, SimVar)
        ]
        if len(place_ids) != len(set(place_ids)):
            return True
    return False


def _expand_observed_bindings_preserving_roles(self: Any):
    """Build one expanded action node per observed ordered binding.

    This is a narrow compatibility replacement for GymPN's
    ``expand_no_future_tokens``.  It is installed only when an action contains
    repeated occurrences of one input place.  Each occurrence is preserved in
    order, while all ordinary GymPN callers continue to use the upstream
    expansion implementation.
    """

    expanded_pn = GymProblem(has_var_attrs=True)
    expanded_pn.arcs = []
    expanded_place_by_token = {}
    expanded_places_by_source: Dict[str, List[Any]] = {}

    for place in self.places:
        valid_tokens = [token for token in place.marking if token.time <= self.clock]
        new_places = []
        if valid_tokens:
            for index, token in enumerate(valid_tokens):
                new_place = expanded_pn.add_var(
                    name=f"{place._id}.{index}",
                    var_attributes=self.var_attributes[place._id],
                )
                new_place.put(token.value, time=token.time)
                if hasattr(token, "_id"):
                    setattr(new_place.marking[0], "_id", token._id)
                expanded_place_by_token[id(token)] = new_place
                new_places.append(new_place)
        else:
            new_places.append(expanded_pn.add_var(
                name=f"{place._id}.0",
                var_attributes=self.var_attributes[place._id],
            ))
        expanded_places_by_source[str(place._id)] = new_places

    # Semantic-matrix scenarios contain actions only.  Copying events here as
    # structural arcs keeps the fallback valid if a later mixed model adds an
    # automatic evolution without changing its decision semantics.
    for transition in self.events:
        incoming = []
        outgoing = []
        for place in transition.incoming:
            if isinstance(place, SimVar):
                incoming.extend(expanded_places_by_source[str(place._id)])
            elif isinstance(place, SimVarQueue):
                incoming.append(place)
        for place in transition.outgoing:
            if isinstance(place, SimVar):
                outgoing.extend(expanded_places_by_source[str(place._id)])
            elif isinstance(place, SimVarQueue):
                outgoing.append(place)
        expanded_transition = expanded_pn.add_event(
            incoming,
            outgoing,
            behavior=transition.behavior,
            name=transition._id,
        )
        for place in expanded_transition.incoming:
            expanded_pn.arcs.append((place, expanded_transition))
        for place in expanded_transition.outgoing:
            expanded_pn.arcs.append((expanded_transition, place))

    transition_binding_map = []
    action_index = 0
    for transition in self.actions:
        for binding, time in self.event_bindings(transition):
            if time > self.clock:
                continue
            transition_binding_map.append([binding, time, transition])
            incoming = []
            for _place, token in binding:
                expanded_place = expanded_place_by_token.get(id(token))
                if expanded_place is None:
                    raise ValueError(
                        "Observed GymPN Binding contains a future or unknown Token"
                    )
                incoming.append(expanded_place)
            outgoing = []
            for place in transition.outgoing:
                if isinstance(place, SimVar):
                    outgoing.extend(expanded_places_by_source[str(place._id)])
                elif isinstance(place, SimVarQueue):
                    outgoing.append(place)
            dummy_behavior = self._make_dummy_behavior(len(incoming))
            expanded_transition = expanded_pn.add_action(
                incoming,
                outgoing,
                behavior=dummy_behavior,
                name=f"{transition._id}.{action_index}",
            )
            action_index += 1
            for place in expanded_transition.incoming:
                expanded_pn.arcs.append((place, expanded_transition))
            for place in expanded_transition.outgoing:
                expanded_pn.arcs.append((expanded_transition, place))
    return expanded_pn, transition_binding_map


def _canonicalize_assignment_graph(
    graph: HeteroData,
    problem: Any,
) -> HeteroData:
    """Keep node/edge metadata fixed when a dynamic arc has no instances."""

    visible_places = [
        place for place in problem.places
        if place._id not in problem.unobservable_simvars
    ]
    node_types = sorted(place._id for place in visible_places)
    node_types.extend(("a_transition", "e_transition"))
    canonical = HeteroData()
    for node_type in node_types:
        if node_type in graph.node_types:
            canonical[node_type].x = graph[node_type].x
        elif node_type == "a_transition":
            canonical[node_type].x = torch.empty(
                (0, len(problem.actions)), dtype=torch.float32
            )
        elif node_type == "e_transition":
            canonical[node_type].x = torch.empty(
                (0, len(problem.events)), dtype=torch.float32
            )
        else:
            canonical[node_type].x = torch.empty(
                (0, len(problem.var_attributes[node_type])),
                dtype=torch.float32,
            )

    visible_ids = {place._id for place in visible_places}
    edge_types = set()
    for transition_type, transitions in (
        ("a_transition", problem.actions),
        ("e_transition", problem.events),
    ):
        for transition in transitions:
            for place in transition.incoming:
                if place._id in visible_ids:
                    edge_types.add((
                        place._id,
                        f"edge_to_{transition_type}",
                        transition_type,
                    ))
            for place in transition.outgoing:
                if place._id in visible_ids:
                    edge_types.add((
                        transition_type,
                        f"edge_to_{place._id}",
                        place._id,
                    ))
    edge_types.add(("a_transition", "self_loop", "a_transition"))
    for edge_type in sorted(edge_types):
        if edge_type in graph.edge_types:
            canonical[edge_type].edge_index = graph[edge_type].edge_index
        else:
            canonical[edge_type].edge_index = torch.empty(
                (2, 0), dtype=torch.int64
            )
    return canonical


def _select_executable_action_nodes(
    graph: HeteroData,
    selected_indices: Sequence[int],
) -> HeteroData:
    """Remove structural Cartesian-product nodes rejected by GymPN guards."""

    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("Executable Assignment Graph action indices repeat")
    action_count = int(graph["a_transition"].num_nodes)
    if any(not 0 <= int(index) < action_count for index in selected_indices):
        raise ValueError("Executable Assignment Graph action index is invalid")
    selected = [int(index) for index in selected_indices]
    remap = torch.full((action_count,), -1, dtype=torch.long)
    if selected:
        old_indices = torch.tensor(selected, dtype=torch.long)
        remap[old_indices] = torch.arange(len(selected), dtype=torch.long)
    else:
        old_indices = torch.empty((0,), dtype=torch.long)

    filtered = HeteroData()
    for node_type in graph.node_types:
        if node_type == "a_transition":
            filtered[node_type].x = graph[node_type].x[old_indices]
        else:
            filtered[node_type].x = graph[node_type].x
    for edge_type in graph.edge_types:
        source_type, _relation, target_type = edge_type
        edge_index = graph[edge_type].edge_index
        keep = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if source_type == "a_transition":
            keep &= remap[edge_index[0]] >= 0
        if target_type == "a_transition":
            keep &= remap[edge_index[1]] >= 0
        edge_index = edge_index[:, keep].clone()
        if source_type == "a_transition" and edge_index.numel():
            edge_index[0] = remap[edge_index[0]]
        if target_type == "a_transition" and edge_index.numel():
            edge_index[1] = remap[edge_index[1]]
        filtered[edge_type].edge_index = edge_index
    return filtered


def _numericize_token_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return _numericize_value(value)
    flattened: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            for nested_key, nested_value in item.items():
                flattened[f"{key}__{nested_key}"] = _numericize_value(
                    nested_value
                )
        else:
            flattened[str(key)] = _numericize_value(item)
    return flattened


def _numericize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _numericize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_numericize_value(item) for item in value)
    if isinstance(value, list):
        return [_numericize_value(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(_encode_symbol(value))
    raise TypeError(f"Unsupported GymPN token attribute type: {type(value)!r}")


def _encode_symbol(value: str) -> int:
    if value in _SYMBOL_CODES:
        return _SYMBOL_CODES[value]
    match = _STRUCTURED_ID.fullmatch(value)
    if match is not None and match.group("kind") in _KIND_CODES:
        paired = 0
        for part in match.group("body").split(":"):
            if part.isdigit():
                component = int(part) + 1
            else:
                digest = hashlib.blake2b(
                    part.encode("utf-8"), digest_size=2
                ).digest()
                component = int.from_bytes(digest, "big") % 997 + 1
            total = paired + component
            paired = total * (total + 1) // 2 + component
        return _KIND_CODES[match.group("kind")] * 1_000_000 + paired
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=6).digest()
    return 10_000_000 + int.from_bytes(digest, "big") % 1_000_000


def _match_observed_actions(
    executable_actions: Sequence[Tuple[Any, ...]],
    observed_actions: Sequence[Tuple[Any, ...]],
) -> List[int]:
    # GymPN's expansion enumerates the structural Cartesian product and does
    # not reapply action guards.  The environment's ``pn_actions`` is the
    # authoritative executable subset.  Map that subset to graph node indices
    # and expose only those indices to PPO.
    by_signature: Dict[str, List[int]] = {}
    for index, binding in enumerate(observed_actions):
        by_signature.setdefault(_binding_signature(binding), []).append(index)
    mapping: List[int] = []
    for binding in executable_actions:
        signature = _binding_signature(binding)
        matches = by_signature.get(signature, [])
        if not matches:
            raise ValueError(
                "Executable GymPN action has no Assignment Graph node: "
                f"{signature}"
            )
        mapping.append(matches.pop(0))
    return mapping


def _binding_signature(binding: Tuple[Any, ...]) -> str:
    participants, binding_time, transition = binding
    values = {
        "transition": str(transition._id),
        "time": round(float(binding_time), 12),
        "participants": [
            {
                "place": str(place._id),
                "token_time": round(float(token.time), 12),
                "value": _numericize_token_value(token.value),
            }
            for place, token in participants
        ],
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))
