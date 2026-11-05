from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from CPNModelV2.graph.generic_tensorizer import (
    BINDING_NODE_TYPE,
    CanonicalEdgeType,
    GenericGraphFeatureSchema,
    GenericGraphTensorizer,
    GenericNumericFeatureNormalizer,
    GenericTokenBindingBatch,
)
from CPNModelV2.graph.schema import TokenBindingGraph


EFFECT_FIXED_FIELDS: Tuple[str, ...] = (
    "is_consumed",
    "is_read",
    "survives",
    "place_changed",
    "numeric_changed",
    "reference_changed",
)

EFFECT_SUMMARY_FIELDS: Tuple[str, ...] = (
    "consume_fraction",
    "read_fraction",
    "changed_fraction",
    "place_change_fraction",
    "numeric_change_fraction",
    "reference_change_fraction",
    "created_fraction",
    "removed_fraction",
)


@dataclass(frozen=True)
class TokenEffect:
    """Immediate effect of one enabled Binding on one participating Token."""

    token_node_id: str
    participation_role: str
    arc_mode: str
    before_place: str
    after_place: str
    numeric_delta: Mapping[str, float] = field(default_factory=dict)
    reference_changes: int = 0
    survives: bool = True

    @property
    def place_changed(self) -> bool:
        return self.before_place != self.after_place

    @property
    def numeric_changes(self) -> int:
        return sum(abs(float(value)) > 1e-12 for value in self.numeric_delta.values())

    @property
    def changed(self) -> bool:
        return (
            not self.survives
            or self.place_changed
            or self.numeric_changes > 0
            or self.reference_changes > 0
        )


@dataclass(frozen=True)
class BindingEffect:
    """One executable candidate and its marking-local immediate effect."""

    binding_node_id: str
    token_effects: Tuple[TokenEffect, ...]
    created_tokens: int = 0
    removed_tokens: int = 0


@dataclass
class BindingEffectGraph(TokenBindingGraph):
    """Token-Binding graph augmented with one effect record per action."""

    binding_effects: Dict[str, BindingEffect] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if set(self.binding_effects) != set(self.action_keys):
            missing = set(self.action_keys) - set(self.binding_effects)
            extra = set(self.binding_effects) - set(self.action_keys)
            raise ValueError(
                "Binding effects must match action nodes exactly: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for binding_node_id, effect in self.binding_effects.items():
            if effect.binding_node_id != binding_node_id:
                raise ValueError(f"Effect key mismatch for {binding_node_id}")
            for token_effect in effect.token_effects:
                node = self.nodes.get(token_effect.token_node_id)
                if node is None or node.node_type == BINDING_NODE_TYPE:
                    raise ValueError(
                        "Effect participant must refer to a Token node: "
                        f"{token_effect.token_node_id}"
                    )
                if token_effect.arc_mode not in {"consume", "read"}:
                    raise ValueError(
                        f"Unsupported effect arc mode: {token_effect.arc_mode}"
                    )


@dataclass
class BindingEffectBatch(GenericTokenBindingBatch):
    """Generic batch plus candidate-specific executable effect channels."""

    effect_summary: torch.Tensor
    effect_edge_index: Dict[CanonicalEdgeType, torch.Tensor]
    effect_edge_numeric: Dict[CanonicalEdgeType, torch.Tensor]
    effect_place_categorical: Dict[CanonicalEdgeType, torch.Tensor]

    def validate(self) -> None:
        super().validate()
        binding_count = len(self.node_ids[BINDING_NODE_TYPE])
        if self.effect_summary.shape != (
            binding_count,
            len(EFFECT_SUMMARY_FIELDS),
        ):
            raise ValueError("Invalid effect summary shape")
        if not torch.isfinite(self.effect_summary).all():
            raise ValueError("Non-finite effect summary")
        if set(self.effect_edge_index) != set(self.effect_edge_numeric):
            raise ValueError("Effect edge/numeric relation sets differ")
        if set(self.effect_edge_index) != set(self.effect_place_categorical):
            raise ValueError("Effect edge/place relation sets differ")
        for canonical, edge_index in self.effect_edge_index.items():
            source_type, _role, target_type = canonical
            if source_type not in self.schema.token_types:
                raise ValueError(f"Unknown effect source type: {source_type}")
            if target_type != BINDING_NODE_TYPE:
                raise ValueError("Effect edges must target Binding nodes")
            edge_count = edge_index.shape[1]
            expected_width = len(EFFECT_FIXED_FIELDS) + len(
                self.schema.numeric_fields[source_type]
            )
            if edge_index.shape[0] != 2:
                raise ValueError(f"Invalid effect edge index for {canonical}")
            if self.effect_edge_numeric[canonical].shape != (
                edge_count,
                expected_width,
            ):
                raise ValueError(f"Invalid effect numeric shape for {canonical}")
            if self.effect_place_categorical[canonical].shape != (edge_count, 2):
                raise ValueError(f"Invalid effect place shape for {canonical}")

    def to(self, device: Union[str, torch.device]) -> "BindingEffectBatch":
        target = torch.device(device)
        return BindingEffectBatch(
            num_graphs=self.num_graphs,
            node_numeric={key: value.to(target) for key, value in self.node_numeric.items()},
            node_categorical={
                key: value.to(target) for key, value in self.node_categorical.items()
            },
            node_batch={key: value.to(target) for key, value in self.node_batch.items()},
            node_ids={key: list(value) for key, value in self.node_ids.items()},
            reference_edge_index={
                key: value.to(target) for key, value in self.reference_edge_index.items()
            },
            participation_edge_index={
                key: value.to(target)
                for key, value in self.participation_edge_index.items()
            },
            relation_ids=dict(self.relation_ids),
            action_binding_index=self.action_binding_index.to(target),
            action_batch=self.action_batch.to(target),
            action_node_ids=list(self.action_node_ids),
            action_keys=list(self.action_keys),
            graph_times=self.graph_times.to(target),
            schema=self.schema,
            effect_summary=self.effect_summary.to(target),
            effect_edge_index={
                key: value.to(target) for key, value in self.effect_edge_index.items()
            },
            effect_edge_numeric={
                key: value.to(target)
                for key, value in self.effect_edge_numeric.items()
            },
            effect_place_categorical={
                key: value.to(target)
                for key, value in self.effect_place_categorical.items()
            },
        )


class BindingEffectTensorizer:
    """Tensorize a Binding-Effect graph without adding domain-specific fields."""

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        *,
        normalizer: Optional[GenericNumericFeatureNormalizer] = None,
        validate_graphs: bool = True,
    ) -> None:
        self.schema = schema
        self.normalizer = normalizer
        self.validate_graphs = bool(validate_graphs)
        self.base_tensorizer = GenericGraphTensorizer(
            schema,
            normalizer=normalizer,
            validate_graphs=validate_graphs,
        )

    def tensorize(
        self,
        graphs: Union[BindingEffectGraph, Sequence[BindingEffectGraph]],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> BindingEffectBatch:
        graph_list = [graphs] if isinstance(graphs, BindingEffectGraph) else list(graphs)
        if not graph_list:
            raise ValueError("At least one Binding-Effect graph is required")
        if any(not isinstance(graph, BindingEffectGraph) for graph in graph_list):
            raise TypeError("BindingEffectTensorizer requires BindingEffectGraph inputs")
        if self.validate_graphs:
            for graph in graph_list:
                graph.validate()

        base = self.base_tensorizer.tensorize(graph_list, device=device)
        category_to_index = self.schema.category_to_index
        target = torch.device(device)
        summaries: List[List[float]] = []
        edge_pairs: Dict[CanonicalEdgeType, List[Tuple[int, int]]] = {}
        edge_numeric: Dict[CanonicalEdgeType, List[List[float]]] = {}
        edge_places: Dict[CanonicalEdgeType, List[List[int]]] = {}
        offsets = {node_type: 0 for node_type in self.schema.node_types}

        for graph in graph_list:
            local_nodes: Dict[str, Tuple[str, int]] = {}
            local_counts = {node_type: 0 for node_type in self.schema.node_types}
            for node in graph.nodes.values():
                node_type = node.node_type
                local_nodes[node.node_id] = (
                    node_type,
                    offsets[node_type] + local_counts[node_type],
                )
                local_counts[node_type] += 1

            binding_nodes = [
                node.node_id
                for node in graph.nodes.values()
                if node.node_type == BINDING_NODE_TYPE
            ]
            for binding_node_id in binding_nodes:
                effect = graph.binding_effects[binding_node_id]
                summaries.append(_effect_summary(effect, graph, self.schema))
                _, binding_index = local_nodes[binding_node_id]
                for token_effect in effect.token_effects:
                    source_type, token_index = local_nodes[token_effect.token_node_id]
                    canonical = (
                        source_type,
                        token_effect.participation_role,
                        BINDING_NODE_TYPE,
                    )
                    edge_pairs.setdefault(canonical, []).append(
                        (token_index, binding_index)
                    )
                    edge_numeric.setdefault(canonical, []).append(
                        self._effect_numeric(source_type, token_effect)
                    )
                    edge_places.setdefault(canonical, []).append([
                        category_to_index.get(
                            str(token_effect.before_place),
                            self.schema.unknown_category_index,
                        ),
                        category_to_index.get(
                            str(token_effect.after_place),
                            self.schema.unknown_category_index,
                        ),
                    ])
            for node_type, count in local_counts.items():
                offsets[node_type] += count

        batch = BindingEffectBatch(
            **vars(base),
            effect_summary=torch.tensor(
                summaries, dtype=torch.float32, device=target
            ).reshape(-1, len(EFFECT_SUMMARY_FIELDS)),
            effect_edge_index={
                canonical: torch.tensor(
                    pairs, dtype=torch.long, device=target
                ).t().contiguous()
                for canonical, pairs in edge_pairs.items()
            },
            effect_edge_numeric={
                canonical: torch.tensor(
                    rows, dtype=torch.float32, device=target
                ).reshape(
                    len(rows),
                    len(EFFECT_FIXED_FIELDS)
                    + len(self.schema.numeric_fields[canonical[0]]),
                )
                for canonical, rows in edge_numeric.items()
            },
            effect_place_categorical={
                canonical: torch.tensor(
                    rows, dtype=torch.long, device=target
                ).reshape(len(rows), 2)
                for canonical, rows in edge_places.items()
            },
        )
        batch.validate()
        return batch

    def _effect_numeric(
        self,
        node_type: str,
        effect: TokenEffect,
    ) -> List[float]:
        fields = self.schema.numeric_fields[node_type]
        binary = set(self.schema.binary_numeric_fields[node_type])
        scaled_delta = []
        for index, field_name in enumerate(fields):
            value = float(effect.numeric_delta.get(field_name, 0.0))
            if self.normalizer is not None and field_name not in binary:
                value /= self.normalizer.scales[node_type][index]
            scaled_delta.append(value)
        return [
            float(effect.arc_mode == "consume"),
            float(effect.arc_mode == "read"),
            float(effect.survives),
            float(effect.place_changed),
            float(effect.numeric_changes > 0),
            float(effect.reference_changes > 0),
            *scaled_delta,
        ]


def fit_binding_effect_schema(
    graphs: Sequence[BindingEffectGraph],
) -> GenericGraphFeatureSchema:
    """Fit the generic schema and retain places only visible in Effect previews."""

    graph_list = list(graphs)
    schema = GenericGraphFeatureSchema.fit(graph_list)
    effect_categories = {
        place
        for graph in graph_list
        for effect in graph.binding_effects.values()
        for token_effect in effect.token_effects
        for place in (token_effect.before_place, token_effect.after_place)
    }
    categories = (*schema.category_values, *sorted(
        effect_categories - set(schema.category_values)
    ))
    return GenericGraphFeatureSchema(
        token_types=schema.token_types,
        numeric_fields=schema.numeric_fields,
        categorical_fields=schema.categorical_fields,
        category_values=categories,
        reference_relations=schema.reference_relations,
        participation_roles=schema.participation_roles,
        binary_numeric_fields=schema.binary_numeric_fields,
    )


def _effect_summary(
    effect: BindingEffect,
    graph: BindingEffectGraph,
    schema: GenericGraphFeatureSchema,
) -> List[float]:
    participants = effect.token_effects
    count = max(1, len(participants))
    total_numeric_fields = max(
        1,
        sum(
            len(schema.numeric_fields[graph.nodes[item.token_node_id].node_type])
            for item in participants
        ),
    )
    return [
        sum(item.arc_mode == "consume" for item in participants) / count,
        sum(item.arc_mode == "read" for item in participants) / count,
        sum(item.changed for item in participants) / count,
        sum(item.place_changed for item in participants) / count,
        sum(item.numeric_changes for item in participants) / total_numeric_fields,
        sum(item.reference_changes > 0 for item in participants) / count,
        effect.created_tokens / count,
        effect.removed_tokens / count,
    ]
