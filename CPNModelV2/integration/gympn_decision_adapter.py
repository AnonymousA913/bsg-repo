from __future__ import annotations

import copy
import random
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from CPNModelV2.graph.binding_effect import (
    BindingEffect,
    BindingEffectGraph,
    TokenEffect,
)
from CPNModelV2.graph.generic_tensorizer import (
    participation_relation,
    reference_relation,
)
from CPNModelV2.graph.schema import GraphEdge, GraphNode, TokenBindingGraph
from CPNModelV2.learning.environment import CPNGraphStep


GYMPN_ACTION_KEY = "gympn_binding"
NetFactory = Callable[[Optional[int]], Any]


class GymPNBindingEffectGraphBuilder:
    """Adapt a GymPN A-E PN marking to a Token-Binding-Effect graph."""

    def __init__(
        self,
        *,
        validate: bool = True,
        include_effects: bool = True,
    ) -> None:
        self.validate = bool(validate)
        self.include_effects = bool(include_effects)

    def build(
        self,
        problem: Any,
        *,
        decision_id: int = 0,
    ) -> TokenBindingGraph:
        semantic_references = bool(
            getattr(problem, "gympn_semantic_references", False)
        )
        graph_class = BindingEffectGraph if self.include_effects else TokenBindingGraph
        graph = graph_class(
            time=float(problem.clock),
            metadata={
                "observation": "gympn_binding_effect_graph",
                "adapter": type(self).__name__,
                "cpn_family": "action_evolution_petri_net",
                "static_pt_nodes_materialized": False,
                "effect_semantics": "transition_behavior_preview",
                "reference_relations_optional": not semantic_references,
                "semantic_references_enabled": semantic_references,
                "token_types_runtime_declared": True,
                "scenario_name": str(
                    getattr(problem, "gympn_scenario_name", "unspecified")
                ),
                "case_multiplier": int(
                    getattr(problem, "gympn_case_multiplier", 1)
                ),
            },
        )
        _copy_complexity_metadata(problem, graph.metadata)
        if semantic_references:
            declared_relations = {
                "activity", "flow", "next", "pool", "resource", "same_case",
            }
            declared_relations.update(
                str(name)
                for name in getattr(
                    problem,
                    "gympn_declared_reference_relations",
                    (),
                )
            )
            graph.metadata["declared_reference_relations"] = tuple(
                reference_relation(name)
                for name in sorted(declared_relations)
            )
        token_nodes: Dict[int, str] = {}
        binary_by_type: Dict[str, set[str]] = {}
        visible_places = [
            place
            for place in problem.places
            if place._id not in problem.unobservable_simvars
        ]
        for place in visible_places:
            visible_tokens = [
                token for token in place.marking
                if float(token.time) <= float(problem.clock) + 1e-12
            ]
            for token_index, token in enumerate(visible_tokens):
                node = _gympn_token_node(
                    place,
                    token,
                    token_index=token_index,
                    now=float(problem.clock),
                    hidden_attributes=problem.unobservable_token_attrs.get(
                        place._id, ()
                    ),
                    categorical_attributes=problem.categorical_token_attrs.get(
                        place._id, {}
                    ),
                    semantic_references=semantic_references,
                )
                graph.add_node(node)
                token_nodes[id(token)] = node.node_id
                binary_by_type.setdefault(node.node_type, set()).add(
                    "available_time_known"
                )

        declared_node_features = _declared_node_features(
            problem,
            now=float(problem.clock),
            semantic_references=semantic_references,
        )
        if declared_node_features:
            graph.metadata["declared_node_features"] = declared_node_features
            for declaration in declared_node_features:
                binary_by_type.setdefault(
                    str(declaration["node_type"]), set()
                ).add("available_time_known")

        if semantic_references:
            _add_semantic_reference_edges(graph)

        bindings = list(getattr(problem, "pn_actions", ()))
        declared_roles = getattr(problem, "gympn_participation_roles", {})
        for binding_index, timed_binding in enumerate(bindings):
            participants, _time, transition = timed_binding
            binding_node_id = f"binding:{binding_index:06d}"
            graph.add_node(GraphNode(
                node_id=binding_node_id,
                node_type="binding",
                categorical={"transition_name": str(transition._id)},
                identity={
                    "decision_id": int(decision_id),
                    "binding_index": int(binding_index),
                },
            ))
            graph.action_keys[binding_node_id] = (
                GYMPN_ACTION_KEY,
                int(decision_id),
                int(binding_index),
                str(transition._id),
            )
            participant_records = []
            transition_roles = tuple(
                declared_roles.get(str(transition._id), ())
            )
            if transition_roles and len(transition_roles) != len(participants):
                raise ValueError(
                    "GymPN participation-role declaration for "
                    f"{transition._id!s} has {len(transition_roles)} roles for "
                    f"{len(participants)} inputs"
                )
            for participant_index, (place, token) in enumerate(participants):
                token_node_id = token_nodes.get(id(token))
                if token_node_id is None:
                    raise ValueError(
                        "Enabled GymPN Binding refers to an invisible or future "
                        f"Token in place {place._id}"
                    )
                raw_role = (
                    transition_roles[participant_index]
                    if transition_roles
                    else f"consume::input_{participant_index}"
                )
                role = participation_relation(raw_role)
                graph.add_edge(GraphEdge(
                    source=token_node_id,
                    target=binding_node_id,
                    edge_type=role,
                    categorical={"input_place": str(place._id)},
                ))
                participant_records.append((
                    token_node_id,
                    role,
                    place,
                    token,
                ))
            if self.include_effects:
                graph.binding_effects[binding_node_id] = _preview_binding_effect(
                    graph,
                    binding_node_id,
                    timed_binding,
                    participant_records,
                    now=float(problem.clock),
                )

        graph.metadata["binary_numeric_fields"] = {
            node_type: sorted(fields)
            for node_type, fields in binary_by_type.items()
        }
        if self.validate:
            graph.validate()
        return graph


class GymPNDecisionEnv:
    """Decision-epoch wrapper exposing every executable GymPN Binding."""

    def __init__(
        self,
        net_factory: NetFactory,
        *,
        horizon: float = 10.0,
        graph_builder: Optional[GymPNBindingEffectGraphBuilder] = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("GymPN horizon must be positive")
        self.net_factory = net_factory
        self.horizon = float(horizon)
        self.graph_builder = graph_builder or GymPNBindingEffectGraphBuilder()
        self.problem: Any = None
        self.net: Any = None
        self.observation: Optional[BindingEffectGraph] = None
        self.decision_count = 0
        self.episode_return = 0.0
        self._terminated = False
        self._rng = random.Random()

    def reset(self, *, seed: Optional[int] = None) -> BindingEffectGraph:
        if seed is not None:
            random.seed(seed)
            # NumPy's legacy RandomState accepts only unsigned 32-bit seeds,
            # while experiment-level seed namespaces can be much larger.
            np.random.seed(int(seed) % (2**32))
            torch.manual_seed(seed)
            self._rng.seed(seed)
        self.problem = self.net_factory(seed)
        self.net = self.problem
        self.problem.length = self.horizon
        self.problem.reward = 0.0
        self.decision_count = 0
        self.episode_return = 0.0
        self._terminated = False
        self._advance_to_decision()
        if self._terminated:
            raise RuntimeError("GymPN scenario terminated before its first action")
        self.observation = self.graph_builder.build(
            self.problem, decision_id=self.decision_count
        )
        return self.observation

    def step(self, binding_key: Tuple[Any, ...]) -> CPNGraphStep:
        if self.problem is None or self.observation is None:
            raise RuntimeError("Environment must be reset before step")
        if self._terminated:
            raise RuntimeError("Cannot step a terminated GymPN episode")
        if len(binding_key) != 4 or binding_key[0] != GYMPN_ACTION_KEY:
            raise ValueError(f"Invalid GymPN Binding key: {binding_key}")
        _, decision_id, binding_index, transition_name = binding_key
        if int(decision_id) != self.decision_count:
            raise ValueError("Stale GymPN Binding from an earlier decision epoch")
        bindings = list(self.problem.pn_actions)
        if not 0 <= int(binding_index) < len(bindings):
            raise ValueError(f"GymPN Binding index out of range: {binding_index}")
        timed_binding = bindings[int(binding_index)]
        if str(timed_binding[2]._id) != str(transition_name):
            raise ValueError("GymPN Binding transition identity changed")

        reward_before = float(self.problem.reward)
        result_tokens = self.problem.fire(timed_binding)
        self.problem.update_reward(timed_binding, result_tokens)
        self.decision_count += 1
        self._advance_to_decision()
        reward = float(self.problem.reward) - reward_before
        self.episode_return += reward
        self.observation = self.graph_builder.build(
            self.problem, decision_id=self.decision_count
        )
        info: Dict[str, Any] = {
            "time": float(self.problem.clock),
            "decision_count": self.decision_count,
            "episode_return": self.episode_return,
            "pn_reward": float(self.problem.reward),
            "enabled_actions": len(self.observation.action_keys),
            "horizon": self.horizon,
        }
        if self._terminated:
            # TokenBindingPPO names the minimized episode metric ``makespan``.
            # Here it stores the negative scenario score so lower remains
            # better.  Ordinary GymPN workflows retain the historical
            # completed-task names; controlled stress scenarios may declare a
            # more precise score label without changing the PPO interface.
            objective_name = str(getattr(
                self.problem,
                "gympn_terminal_objective_name",
                "negative_completed_tasks",
            ))
            score_name = str(getattr(
                self.problem,
                "gympn_terminal_score_name",
                "completed_tasks",
            ))
            info.update({
                "makespan": -float(self.problem.reward),
                "objective_name": objective_name,
                score_name: float(self.problem.reward),
            })
        return CPNGraphStep(
            observation=self.observation,
            reward=reward,
            terminated=self._terminated,
            info=info,
        )

    def _advance_to_decision(self) -> None:
        while True:
            bindings, active_model = self.problem.bindings()
            if not active_model or float(self.problem.clock) > self.horizon:
                self.problem.pn_actions = []
                self._terminated = True
                return
            if self.problem.network_tag.is_action():
                self.problem.pn_actions = list(bindings)
                if bindings:
                    return
                self._terminated = True
                return
            if not bindings:
                self.problem.pn_actions = []
                self._terminated = True
                return
            timed_binding = self._rng.choice(bindings)
            result_tokens = self.problem.fire(timed_binding)
            self.problem.update_reward(timed_binding, result_tokens)


def _gympn_token_node(
    place: Any,
    token: Any,
    *,
    token_index: int,
    now: float,
    hidden_attributes: Sequence[str],
    categorical_attributes: Mapping[str, Sequence[Any]],
    semantic_references: bool = False,
) -> GraphNode:
    flattened = _flatten_value(token.value)
    hidden = set(hidden_attributes)
    numeric: Dict[str, float] = {
        "available_time_delta": float(token.time) - now,
        "available_time_known": 1.0,
    }
    categorical: Dict[str, str] = {"current_place": str(place._id)}
    identity: Dict[str, Any] = {
        "place_id": str(place._id),
        "token_index": int(token_index),
    }
    semantic_links: Dict[str, str] = {}
    type_fields = []
    for field_name, value in flattened:
        leaf_name = field_name.split(".")[-1]
        if leaf_name in hidden:
            continue
        safe_name = _safe_name(field_name)
        if semantic_references and field_name == "object_id":
            identity["object_id"] = str(value)
            continue
        if semantic_references and leaf_name.endswith("_ref"):
            relation_name = _safe_name(leaf_name[:-4])
            if str(value):
                semantic_links[relation_name] = str(value)
            continue
        if semantic_references and leaf_name == "case_id":
            identity["case_id"] = str(value)
            continue
        if leaf_name in categorical_attributes or isinstance(value, str):
            categorical[f"value_{safe_name}"] = str(value)
            type_fields.append(f"cat_{safe_name}")
        elif isinstance(value, (bool, int, float, np.number)):
            numeric[f"value_{safe_name}"] = float(value)
            type_fields.append(f"num_{safe_name}")
        else:
            categorical[f"value_{safe_name}"] = repr(value)
            type_fields.append(f"cat_{safe_name}")
    signature = "__".join(sorted(type_fields)) or "empty"
    if semantic_links:
        identity["semantic_references"] = semantic_links
    node_type = f"color__{signature}"
    return GraphNode(
        node_id=f"token:{place._id}:{token_index:06d}",
        node_type=node_type,
        numeric=numeric,
        categorical=categorical,
        identity=identity,
    )


def _declared_node_features(
    problem: Any,
    *,
    now: float,
    semantic_references: bool,
) -> List[Dict[str, Any]]:
    declarations = []
    for sample_index, sample in enumerate(
        getattr(problem, "gympn_declared_token_samples", ())
    ):
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError("Invalid GymPN declared Token sample")
        place_id, value = sample
        place_id = str(place_id)
        node = _gympn_token_node(
            SimpleNamespace(_id=place_id),
            SimpleNamespace(value=value, time=now),
            token_index=sample_index,
            now=now,
            hidden_attributes=problem.unobservable_token_attrs.get(
                place_id, ()
            ),
            categorical_attributes=problem.categorical_token_attrs.get(
                place_id, {}
            ),
            semantic_references=semantic_references,
        )
        declarations.append({
            "node_type": node.node_type,
            "numeric": dict(node.numeric),
            "categorical": dict(node.categorical),
        })
    return declarations


def _add_semantic_reference_edges(graph: TokenBindingGraph) -> None:
    object_nodes: Dict[str, str] = {}
    case_nodes: Dict[str, List[str]] = {}
    for node in graph.nodes.values():
        object_id = node.identity.get("object_id")
        if object_id is not None:
            object_key = str(object_id)
            if object_key in object_nodes:
                raise ValueError(f"Duplicate GymPN semantic object id: {object_key}")
            object_nodes[object_key] = node.node_id
        case_id = node.identity.get("case_id")
        if case_id is not None:
            case_nodes.setdefault(str(case_id), []).append(node.node_id)

    for node in list(graph.nodes.values()):
        links = node.identity.get("semantic_references", {})
        if not isinstance(links, Mapping):
            continue
        for relation_name, target_object_id in sorted(links.items()):
            target = object_nodes.get(str(target_object_id))
            if target is None:
                # A referenced resource may be busy and therefore not visible
                # at the current decision epoch.  The fact remains in the
                # Token identity, but no dangling policy edge is emitted.
                continue
            graph.add_edge(GraphEdge(
                source=node.node_id,
                target=target,
                edge_type=reference_relation(str(relation_name)),
            ))

    # Parallel branch Tokens carry the same case identity.  A star is enough
    # because reference propagation traverses semantic edges in both
    # directions and avoids a quadratic same-case clique.
    for node_ids in case_nodes.values():
        ordered = sorted(set(node_ids))
        if len(ordered) < 2:
            continue
        anchor = ordered[0]
        for target in ordered[1:]:
            graph.add_edge(GraphEdge(
                source=anchor,
                target=target,
                edge_type=reference_relation("same_case"),
            ))


def _copy_complexity_metadata(
    problem: Any,
    metadata: Dict[str, Any],
) -> None:
    attribute_names = (
        "gympn_complexity_level",
        "gympn_complexity_family",
        "gympn_flow_size",
        "gympn_case_streams",
        "gympn_case_classes",
        "gympn_resources_per_pool",
        "gympn_eligibility_per_activity",
        "gympn_processing_time_max",
        "gympn_target_offered_load",
        "gympn_resource_layout",
        "gympn_arrival_interval",
        "gympn_semantic_stress_target",
        "gympn_semantic_stress_pair_count",
        "gympn_semantic_stress_counterfactual",
    )
    for attribute_name in attribute_names:
        if hasattr(problem, attribute_name):
            metadata[attribute_name.removeprefix("gympn_")] = getattr(
                problem,
                attribute_name,
            )


def _preview_binding_effect(
    graph: BindingEffectGraph,
    binding_node_id: str,
    timed_binding: Tuple[Any, ...],
    participant_records: Sequence[Tuple[str, str, Any, Any]],
    *,
    now: float,
) -> BindingEffect:
    participants, _time, transition = timed_binding
    values = [copy.deepcopy(token.value) for _place, token in participants]
    rng_state = _capture_rng_state()
    try:
        outputs = transition.behavior(*values)
    finally:
        _restore_rng_state(rng_state)
    if not isinstance(outputs, list) or len(outputs) != len(transition.outgoing):
        raise ValueError(
            f"GymPN transition {transition._id} returned an invalid preview"
        )

    produced = [
        (place, token)
        for place, token in zip(transition.outgoing, outputs)
        if token is not None
    ]
    token_effects: List[TokenEffect] = []
    removed = 0
    for token_node_id, role, input_place, input_token in participant_records:
        descendant = next(
            (
                (output_place, output_token)
                for output_place, output_token in produced
                if _contains_value(output_token.value, input_token.value)
            ),
            None,
        )
        before = graph.nodes[token_node_id]
        if descendant is None:
            removed += 1
            token_effects.append(TokenEffect(
                token_node_id=token_node_id,
                participation_role=role,
                arc_mode="consume",
                before_place=str(input_place._id),
                after_place="<CONSUMED>",
                numeric_delta={
                    field: -float(value)
                    for field, value in before.numeric.items()
                },
                survives=False,
            ))
            continue
        output_place, output_token = descendant
        after_numeric = dict(before.numeric)
        after_numeric["available_time_delta"] = float(output_token.delay)
        token_effects.append(TokenEffect(
            token_node_id=token_node_id,
            participation_role=role,
            arc_mode="consume",
            before_place=str(input_place._id),
            after_place=str(output_place._id),
            numeric_delta={
                field: float(after_numeric.get(field, 0.0)) - float(value)
                for field, value in before.numeric.items()
            },
            survives=True,
        ))
    return BindingEffect(
        binding_node_id=binding_node_id,
        token_effects=tuple(token_effects),
        created_tokens=len(produced),
        removed_tokens=removed,
    )


def _flatten_value(value: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    if isinstance(value, Mapping):
        result = []
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_flatten_value(value[key], child))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            result.extend(_flatten_value(item, child))
        return result
    return [(prefix or "value", value)]


def _contains_value(container: Any, target: Any) -> bool:
    if container == target:
        return True
    if isinstance(container, Mapping):
        return any(_contains_value(value, target) for value in container.values())
    if isinstance(container, (tuple, list)):
        return any(_contains_value(value, target) for value in container)
    return False


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_") or "value"


def _capture_rng_state() -> Tuple[Any, Any, torch.Tensor, Optional[List[torch.Tensor]]]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng_state(
    state: Tuple[Any, Any, torch.Tensor, Optional[List[torch.Tensor]]],
) -> None:
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
