from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from CPNModelV2.core.fjsp_cpn_v2 import T_START
from CPNModelV2.core.fjsp_cpn_v5 import (
    MACHINE_PLACES,
    OPERATION_PLACES,
    RELATION_PLACES,
)
from CPNModelV2.graph.schema import TokenBindingGraph
from CPNModelV2.graph.v5_builder import (
    BINDING_ROLE_RELATIONS,
    POLICY_NODE_TYPES,
    REFERENCE_RELATIONS,
)


V5_NODE_TYPES = POLICY_NODE_TYPES

V5_NUMERIC_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "operation": (
        "relative_position",
        "num_operations",
        "priority",
        "release_time_delta",
        "release_time_known",
        "due_time_delta",
        "due_time_known",
        "ready_time_delta",
        "ready_time_known",
        "start_time_delta",
        "start_time_known",
        "finish_time_delta",
        "finish_time_known",
    ),
    "machine": (
        "is_busy",
        "available_time_delta",
        "available_time_known",
    ),
    "eligibility": ("processing_time",),
    "precedence": (),
    "binding": (),
}

V5_CATEGORICAL_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "operation": ("current_place",),
    "machine": ("current_place",),
    "eligibility": ("current_place",),
    "precedence": ("current_place",),
    "binding": ("transition_name",),
}

MISSING_CATEGORY = "<MISSING>"
UNKNOWN_CATEGORY = "<UNK>"
UNKNOWN_RELATION = "<UNK_RELATION>"
CanonicalEdgeType = Tuple[str, str, str]

BINARY_NUMERIC_FIELDS = frozenset({
    "is_busy",
    "release_time_known",
    "due_time_known",
    "ready_time_known",
    "start_time_known",
    "finish_time_known",
    "available_time_known",
})


@dataclass(frozen=True)
class V5GraphFeatureSchema:
    category_values: Tuple[str, ...]
    relation_values: Tuple[str, ...]

    @classmethod
    def fit(cls, graphs: Iterable[TokenBindingGraph]) -> "V5GraphFeatureSchema":
        categories = {
            MISSING_CATEGORY,
            UNKNOWN_CATEGORY,
            T_START,
            *OPERATION_PLACES,
            *MACHINE_PLACES,
            *RELATION_PLACES,
        }
        relations = {
            UNKNOWN_RELATION,
            *REFERENCE_RELATIONS,
            *BINDING_ROLE_RELATIONS,
        }
        for graph in graphs:
            graph.validate()
            _validate_v5_graph(graph)
            for node in graph.nodes.values():
                categories.update(str(value) for value in node.categorical.values())
            relations.update(edge.edge_type for edge in graph.edges)
        return cls(
            category_values=_reserved_first(
                categories, MISSING_CATEGORY, UNKNOWN_CATEGORY
            ),
            relation_values=_reserved_first(relations, UNKNOWN_RELATION),
        )

    @property
    def category_to_index(self) -> Dict[str, int]:
        return {value: index for index, value in enumerate(self.category_values)}

    @property
    def relation_to_index(self) -> Dict[str, int]:
        return {value: index for index, value in enumerate(self.relation_values)}

    @property
    def unknown_category_index(self) -> int:
        return self.category_to_index[UNKNOWN_CATEGORY]

    @property
    def unknown_relation_index(self) -> int:
        return self.relation_to_index[UNKNOWN_RELATION]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "category_values": list(self.category_values),
            "relation_values": list(self.relation_values),
        }

    @classmethod
    def from_dict(
        cls, values: Mapping[str, Sequence[str]]
    ) -> "V5GraphFeatureSchema":
        return cls(
            category_values=tuple(values["category_values"]),
            relation_values=tuple(values["relation_values"]),
        )


@dataclass(frozen=True)
class V5NumericFeatureNormalizer:
    means: Mapping[str, Tuple[float, ...]]
    scales: Mapping[str, Tuple[float, ...]]

    @classmethod
    def fit(
        cls,
        graphs: Iterable[TokenBindingGraph],
        *,
        minimum_scale: float = 1e-6,
    ) -> "V5NumericFeatureNormalizer":
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        rows: Dict[str, List[List[float]]] = {
            node_type: [] for node_type in V5_NODE_TYPES
        }
        for graph in graphs:
            _validate_v5_graph(graph)
            for node in graph.nodes.values():
                rows[node.node_type].append([
                    float(node.numeric.get(field, 0.0))
                    for field in V5_NUMERIC_FIELDS[node.node_type]
                ])
        means: Dict[str, Tuple[float, ...]] = {}
        scales: Dict[str, Tuple[float, ...]] = {}
        for node_type in V5_NODE_TYPES:
            width = len(V5_NUMERIC_FIELDS[node_type])
            if not rows[node_type] or width == 0:
                means[node_type] = (0.0,) * width
                scales[node_type] = (1.0,) * width
                continue
            columns = list(zip(*rows[node_type]))
            type_means = tuple(sum(column) / len(column) for column in columns)
            type_scales = []
            for column, feature_mean in zip(columns, type_means):
                variance = sum(
                    (value - feature_mean) ** 2 for value in column
                ) / len(column)
                scale = math.sqrt(variance)
                type_scales.append(scale if scale >= minimum_scale else 1.0)
            means[node_type] = type_means
            scales[node_type] = tuple(type_scales)
        return cls(means, scales)

    def transform(self, node_type: str, row: Sequence[float]) -> List[float]:
        if len(row) != len(V5_NUMERIC_FIELDS[node_type]):
            raise ValueError(f"Invalid numeric width for {node_type}")
        return [
            float(value)
            if field in BINARY_NUMERIC_FIELDS
            else (float(value) - mean) / scale
            for field, value, mean, scale in zip(
                V5_NUMERIC_FIELDS[node_type],
                row,
                self.means[node_type],
                self.scales[node_type],
            )
        ]

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            "means": {
                node_type: list(self.means[node_type])
                for node_type in V5_NODE_TYPES
            },
            "scales": {
                node_type: list(self.scales[node_type])
                for node_type in V5_NODE_TYPES
            },
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Mapping[str, Sequence[float]]],
    ) -> "V5NumericFeatureNormalizer":
        return cls(
            means={
                node_type: tuple(
                    float(value) for value in values["means"][node_type]
                )
                for node_type in V5_NODE_TYPES
            },
            scales={
                node_type: tuple(
                    float(value) for value in values["scales"][node_type]
                )
                for node_type in V5_NODE_TYPES
            },
        )


@dataclass
class V5TokenBindingBatch:
    num_graphs: int
    node_numeric: Dict[str, torch.Tensor]
    node_categorical: Dict[str, torch.Tensor]
    node_batch: Dict[str, torch.Tensor]
    node_ids: Dict[str, List[str]]
    reference_edge_index: Dict[CanonicalEdgeType, torch.Tensor]
    binding_role_edge_index: Dict[CanonicalEdgeType, torch.Tensor]
    relation_ids: Dict[CanonicalEdgeType, int]
    action_binding_index: torch.Tensor
    action_batch: torch.Tensor
    action_node_ids: List[str]
    action_keys: List[Tuple[Any, ...]]
    graph_times: torch.Tensor
    schema: V5GraphFeatureSchema

    def validate(self) -> None:
        if self.num_graphs <= 0:
            raise ValueError("A V5 batch must contain at least one graph")
        for node_type in V5_NODE_TYPES:
            count = len(self.node_ids[node_type])
            if self.node_numeric[node_type].shape != (
                count,
                len(V5_NUMERIC_FIELDS[node_type]),
            ):
                raise ValueError(f"Invalid V5 numeric shape for {node_type}")
            if self.node_categorical[node_type].shape != (
                count,
                len(V5_CATEGORICAL_FIELDS[node_type]),
            ):
                raise ValueError(f"Invalid V5 categorical shape for {node_type}")
            if self.node_batch[node_type].shape != (count,):
                raise ValueError(f"Invalid V5 batch vector for {node_type}")
            if not torch.isfinite(self.node_numeric[node_type]).all():
                raise ValueError(f"Non-finite V5 features for {node_type}")
        for edge_map in (
            self.reference_edge_index,
            self.binding_role_edge_index,
        ):
            for canonical, edge_index in edge_map.items():
                if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                    raise ValueError(f"Invalid edge index for {canonical}")
                source_type, _, target_type = canonical
                if edge_index.numel():
                    if int(edge_index[0].max()) >= len(self.node_ids[source_type]):
                        raise ValueError(f"Source index out of range for {canonical}")
                    if int(edge_index[1].max()) >= len(self.node_ids[target_type]):
                        raise ValueError(f"Target index out of range for {canonical}")
        action_count = len(self.action_keys)
        if self.action_binding_index.shape != (action_count,):
            raise ValueError("Invalid action_binding_index")
        if self.action_batch.shape != (action_count,):
            raise ValueError("Invalid action_batch")

    def to(self, device: Union[str, torch.device]) -> "V5TokenBindingBatch":
        target = torch.device(device)
        return V5TokenBindingBatch(
            num_graphs=self.num_graphs,
            node_numeric={k: v.to(target) for k, v in self.node_numeric.items()},
            node_categorical={
                k: v.to(target) for k, v in self.node_categorical.items()
            },
            node_batch={k: v.to(target) for k, v in self.node_batch.items()},
            node_ids={k: list(v) for k, v in self.node_ids.items()},
            reference_edge_index={
                k: v.to(target) for k, v in self.reference_edge_index.items()
            },
            binding_role_edge_index={
                k: v.to(target) for k, v in self.binding_role_edge_index.items()
            },
            relation_ids=dict(self.relation_ids),
            action_binding_index=self.action_binding_index.to(target),
            action_batch=self.action_batch.to(target),
            action_node_ids=list(self.action_node_ids),
            action_keys=list(self.action_keys),
            graph_times=self.graph_times.to(target),
            schema=self.schema,
        )

    @property
    def device(self) -> torch.device:
        return self.graph_times.device


class V5GraphTensorizer:
    """Tensorize V5 graphs while keeping reference and Binding-role edges apart."""

    def __init__(
        self,
        schema: V5GraphFeatureSchema,
        *,
        normalizer: Optional[V5NumericFeatureNormalizer] = None,
        validate_graphs: bool = True,
    ) -> None:
        self.schema = schema
        self.normalizer = normalizer
        self.validate_graphs = bool(validate_graphs)

    def tensorize(
        self,
        graphs: Union[TokenBindingGraph, Sequence[TokenBindingGraph]],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> V5TokenBindingBatch:
        graph_list = [graphs] if isinstance(graphs, TokenBindingGraph) else list(graphs)
        if not graph_list:
            raise ValueError("At least one V5 graph is required")
        if self.validate_graphs:
            for graph in graph_list:
                graph.validate()
                _validate_v5_graph(graph)

        numeric_rows = {node_type: [] for node_type in V5_NODE_TYPES}
        categorical_rows = {node_type: [] for node_type in V5_NODE_TYPES}
        node_batches = {node_type: [] for node_type in V5_NODE_TYPES}
        node_ids = {node_type: [] for node_type in V5_NODE_TYPES}
        references: List[Dict[str, Tuple[str, int]]] = []
        category_to_index = self.schema.category_to_index

        for graph_index, graph in enumerate(graph_list):
            graph_references: Dict[str, Tuple[str, int]] = {}
            for node in graph.nodes.values():
                node_type = node.node_type
                local_index = len(node_ids[node_type])
                graph_references[node.node_id] = (node_type, local_index)
                node_ids[node_type].append(node.node_id)
                node_batches[node_type].append(graph_index)
                numeric = [
                    float(node.numeric.get(field, 0.0))
                    for field in V5_NUMERIC_FIELDS[node_type]
                ]
                if self.normalizer is not None:
                    numeric = self.normalizer.transform(node_type, numeric)
                numeric_rows[node_type].append(numeric)
                categorical_rows[node_type].append([
                    category_to_index.get(
                        str(node.categorical.get(field, MISSING_CATEGORY)),
                        self.schema.unknown_category_index,
                    )
                    for field in V5_CATEGORICAL_FIELDS[node_type]
                ])
            references.append(graph_references)

        reference_pairs: Dict[CanonicalEdgeType, List[Tuple[int, int]]] = {}
        binding_pairs: Dict[CanonicalEdgeType, List[Tuple[int, int]]] = {}
        for graph, graph_references in zip(graph_list, references):
            for edge in graph.edges:
                source_type, source_index = graph_references[edge.source]
                target_type, target_index = graph_references[edge.target]
                canonical = (source_type, edge.edge_type, target_type)
                pair = (source_index, target_index)
                destination = (
                    binding_pairs
                    if edge.edge_type.startswith("binding_")
                    else reference_pairs
                )
                destination.setdefault(canonical, []).append(pair)

        action_binding_index: List[int] = []
        action_batch: List[int] = []
        action_node_ids: List[str] = []
        action_keys: List[Tuple[Any, ...]] = []
        for graph_index, (graph, graph_references) in enumerate(
            zip(graph_list, references)
        ):
            for node_id, binding_key in graph.action_keys.items():
                node_type, node_index = graph_references[node_id]
                if node_type != "binding":
                    raise ValueError(f"Action is not a Binding: {node_id}")
                action_binding_index.append(node_index)
                action_batch.append(graph_index)
                action_node_ids.append(node_id)
                action_keys.append(binding_key)

        target = torch.device(device)
        relation_to_index = self.schema.relation_to_index
        all_canonical = set(reference_pairs) | set(binding_pairs)
        batch = V5TokenBindingBatch(
            num_graphs=len(graph_list),
            node_numeric={
                node_type: _matrix(
                    numeric_rows[node_type],
                    len(V5_NUMERIC_FIELDS[node_type]),
                    torch.float32,
                    target,
                )
                for node_type in V5_NODE_TYPES
            },
            node_categorical={
                node_type: _matrix(
                    categorical_rows[node_type],
                    len(V5_CATEGORICAL_FIELDS[node_type]),
                    torch.long,
                    target,
                )
                for node_type in V5_NODE_TYPES
            },
            node_batch={
                node_type: torch.tensor(
                    node_batches[node_type], dtype=torch.long, device=target
                )
                for node_type in V5_NODE_TYPES
            },
            node_ids=node_ids,
            reference_edge_index={
                canonical: torch.tensor(
                    pairs, dtype=torch.long, device=target
                ).t().contiguous()
                for canonical, pairs in reference_pairs.items()
            },
            binding_role_edge_index={
                canonical: torch.tensor(
                    pairs, dtype=torch.long, device=target
                ).t().contiguous()
                for canonical, pairs in binding_pairs.items()
            },
            relation_ids={
                canonical: relation_to_index.get(
                    canonical[1], self.schema.unknown_relation_index
                )
                for canonical in all_canonical
            },
            action_binding_index=torch.tensor(
                action_binding_index, dtype=torch.long, device=target
            ),
            action_batch=torch.tensor(action_batch, dtype=torch.long, device=target),
            action_node_ids=action_node_ids,
            action_keys=action_keys,
            graph_times=torch.tensor(
                [graph.time for graph in graph_list],
                dtype=torch.float32,
                device=target,
            ),
            schema=self.schema,
        )
        batch.validate()
        return batch


def _validate_v5_graph(graph: TokenBindingGraph) -> None:
    unexpected = {
        node.node_type for node in graph.nodes.values()
        if node.node_type not in V5_NODE_TYPES
    }
    if unexpected:
        raise ValueError(f"Non-V5 node types in graph: {sorted(unexpected)}")


def _matrix(
    rows: List[List[Union[int, float]]],
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=dtype, device=device)
    return torch.tensor(rows, dtype=dtype, device=device).reshape(len(rows), width)


def _reserved_first(
    values: Iterable[str], first: str, second: Optional[str] = None
) -> Tuple[str, ...]:
    reserved = {first} if second is None else {first, second}
    prefix = (first,) if second is None else (first, second)
    return (*prefix, *sorted(set(values) - reserved))
