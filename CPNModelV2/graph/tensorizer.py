from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from CPNModelV2.core.fjsp_cpn_v3 import EVENT_TYPES
from CPNModelV2.graph.schema import TokenBindingGraph


# 规定了节点的种类，说实话有一点太多了
NODE_TYPES: Tuple[str, ...] = (
    "place",
    "transition",
    "job",
    "operation",
    "machine",
    "eligibility",
    "precedence",
    "event",
    "binding",
)

# 规定了每种节点到底吃哪些连续特征
NUMERIC_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "place": (),
    "transition": ("controlled", "priority"),
    "job": (
        "num_operations",
        "priority",
        "release_time_delta",
        "release_time_known",
        "due_time_delta",
        "due_time_known",
        "completion_time_delta",
        "completion_time_known",
    ),
    "operation": (
        "op_index",
        "num_operations",
        "attempt_count",
        "failed_attempt_count",
        "ready_time_delta",
        "ready_time_known",
        "start_time_delta",
        "start_time_known",
        "finish_time_delta",
        "finish_time_known",
    ),
    "machine": (
        "failure_count",
        "has_bound_operation",
        "available_time_delta",
        "available_time_known",
    ),
    "eligibility": ("processing_time",),
    "precedence": (),
    "event": (
        "event_time_delta",
        "event_time_known",
        "reveal_time_delta",
        "reveal_time_known",
        "applied_time_delta",
        "applied_time_known",
        "payload_operation_count",
        "payload_eligibility_count",
        "payload_precedence_count",
    ),
    "binding": (
        "consume_count",
        "read_count",
        "processing_time",
        "processing_time_known",
    ),
}

CATEGORICAL_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "place": ("place_name", "token_kind"),
    "transition": ("transition_name",),
    "job": ("current_place",),
    "operation": ("current_place",),
    "machine": ("current_place",),
    "eligibility": ("current_place",),
    "precedence": ("current_place",),
    "event": ("current_place", "event_type", "event_status"),
    "binding": ("transition_name",),
}

MISSING_CATEGORY = "<MISSING>"
UNKNOWN_CATEGORY = "<UNK>"
MISSING_RELATION = "<MISSING_RELATION>"
UNKNOWN_RELATION = "<UNK_RELATION>"

CanonicalEdgeType = Tuple[str, str, str]


@dataclass(frozen=True)
class GraphFeatureSchema:
    """Frozen vocabularies used by both training and inference graphs."""

    category_values: Tuple[str, ...]
    relation_values: Tuple[str, ...]

    @classmethod
    def fit(cls, graphs: Iterable[TokenBindingGraph]) -> "GraphFeatureSchema":
        categories = {
            MISSING_CATEGORY,
            UNKNOWN_CATEGORY,
            "pending",
            "applied",
            "operation",
            "machine",
            "job",
            "eligibility",
            "precedence",
            "event",
            *EVENT_TYPES,
        }
        relations = {MISSING_RELATION, UNKNOWN_RELATION}
        for graph in graphs:
            graph.validate()
            for node in graph.nodes.values():
                categories.update(str(value) for value in node.categorical.values())
            relations.update(edge.edge_type for edge in graph.edges)
            relations.update(f"rev:{edge.edge_type}" for edge in graph.edges)
        return cls(
            category_values=_reserved_first(
                categories, MISSING_CATEGORY, UNKNOWN_CATEGORY
            ),
            relation_values=_reserved_first(
                relations, MISSING_RELATION, UNKNOWN_RELATION
            ),
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
    def from_dict(cls, values: Mapping[str, Sequence[str]]) -> "GraphFeatureSchema":
        return cls(
            category_values=tuple(values["category_values"]),
            relation_values=tuple(values["relation_values"]),
        )

# 此处做的是标准化，而且是按照node type分别统计
# 例如：Operation加工相关时间可能量级是几十，Machine的available time可能是几百，因此需要根据不同节点的类型分别进行标准化
@dataclass(frozen=True)
class NumericFeatureNormalizer:
    """Offline per-node-type standardization fitted from calibration graphs."""

    means: Mapping[str, Tuple[float, ...]]
    scales: Mapping[str, Tuple[float, ...]]

    @classmethod
    def fit(
        cls,
        graphs: Iterable[TokenBindingGraph],
        *,
        minimum_scale: float = 1e-6,
    ) -> "NumericFeatureNormalizer":
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        values: Dict[str, List[List[float]]] = {
            node_type: [] for node_type in NODE_TYPES
        }
        for graph in graphs:
            graph.validate()
            for node in graph.nodes.values():
                values[node.node_type].append([
                    float(node.numeric.get(field_name, 0.0))
                    for field_name in NUMERIC_FIELDS[node.node_type]
                ])
        means: Dict[str, Tuple[float, ...]] = {}
        scales: Dict[str, Tuple[float, ...]] = {}
        for node_type in NODE_TYPES:
            width = len(NUMERIC_FIELDS[node_type])
            rows = values[node_type]
            if not rows:
                means[node_type] = (0.0,) * width
                scales[node_type] = (1.0,) * width
                continue
            columns = list(zip(*rows)) if width else []
            type_means = tuple(
                sum(column) / len(column) for column in columns
            )
            type_scales = []
            for column, feature_mean in zip(columns, type_means):
                variance = sum(
                    (value - feature_mean) ** 2 for value in column
                ) / len(column)
                scale = math.sqrt(variance)
                type_scales.append(scale if scale >= minimum_scale else 1.0)
            means[node_type] = type_means
            scales[node_type] = tuple(type_scales)
        return cls(means=means, scales=scales)

    def transform(self, node_type: str, row: Sequence[float]) -> List[float]:
        if len(row) != len(NUMERIC_FIELDS[node_type]):
            raise ValueError(f"Invalid numeric row width for {node_type}")
        return [
            (float(value) - feature_mean) / feature_scale
            for value, feature_mean, feature_scale in zip(
                row, self.means[node_type], self.scales[node_type]
            )
        ]

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            "means": {
                node_type: list(self.means[node_type]) for node_type in NODE_TYPES
            },
            "scales": {
                node_type: list(self.scales[node_type]) for node_type in NODE_TYPES
            },
        }

    @classmethod
    def from_dict(
        cls, values: Mapping[str, Mapping[str, Sequence[float]]]
    ) -> "NumericFeatureNormalizer":
        return cls(
            means={
                node_type: tuple(float(value) for value in values["means"][node_type])
                for node_type in NODE_TYPES
            },
            scales={
                node_type: tuple(float(value) for value in values["scales"][node_type])
                for node_type in NODE_TYPES
            },
        )


# 这是真正重要的对象，他是网络最终真正收到的数据。
@dataclass
class TokenBindingBatch:
    """A disjoint batch of typed Token-Binding graphs."""

    num_graphs: int
    node_numeric: Dict[str, torch.Tensor]       #
    node_categorical: Dict[str, torch.Tensor]
    node_batch: Dict[str, torch.Tensor]
    node_ids: Dict[str, List[str]]
    edge_index: Dict[CanonicalEdgeType, torch.Tensor]
    edge_relation_ids: Dict[CanonicalEdgeType, int]
    action_binding_index: torch.Tensor
    action_batch: torch.Tensor
    action_node_ids: List[str]
    action_keys: List[Tuple[Any, ...]]
    graph_times: torch.Tensor
    schema: GraphFeatureSchema

    def validate(self) -> None:
        if self.num_graphs <= 0:
            raise ValueError("A tensor batch must contain at least one graph")
        for node_type in NODE_TYPES:
            numeric = self.node_numeric[node_type]
            categorical = self.node_categorical[node_type]
            batches = self.node_batch[node_type]
            expected_nodes = len(self.node_ids[node_type])
            if numeric.shape != (expected_nodes, len(NUMERIC_FIELDS[node_type])):
                raise ValueError(f"Invalid numeric shape for {node_type}: {numeric.shape}")
            if categorical.shape != (
                expected_nodes,
                len(CATEGORICAL_FIELDS[node_type]),
            ):
                raise ValueError(
                    f"Invalid categorical shape for {node_type}: {categorical.shape}"
                )
            if batches.shape != (expected_nodes,):
                raise ValueError(f"Invalid batch vector for {node_type}")
            if not torch.isfinite(numeric).all():
                raise ValueError(f"Non-finite numeric tensor for {node_type}")
        for canonical_type, index in self.edge_index.items():
            if index.ndim != 2 or index.shape[0] != 2:
                raise ValueError(f"Invalid edge_index for {canonical_type}")
            source_type, _, target_type = canonical_type
            if index.numel():
                if int(index[0].max()) >= len(self.node_ids[source_type]):
                    raise ValueError(f"Source edge index out of range: {canonical_type}")
                if int(index[1].max()) >= len(self.node_ids[target_type]):
                    raise ValueError(f"Target edge index out of range: {canonical_type}")
        action_count = len(self.action_keys)
        if self.action_binding_index.shape != (action_count,):
            raise ValueError("Invalid action_binding_index shape")
        if self.action_batch.shape != (action_count,):
            raise ValueError("Invalid action_batch shape")
        if len(self.action_node_ids) != action_count:
            raise ValueError("Action node IDs and action keys differ in length")
        if action_count and int(self.action_binding_index.max()) >= len(
            self.node_ids["binding"]
        ):
            raise ValueError("Binding action index out of range")

    def to(self, device: Union[str, torch.device]) -> "TokenBindingBatch":
        target = torch.device(device)
        return TokenBindingBatch(
            num_graphs=self.num_graphs,
            node_numeric={key: value.to(target) for key, value in self.node_numeric.items()},
            node_categorical={
                key: value.to(target) for key, value in self.node_categorical.items()
            },
            node_batch={key: value.to(target) for key, value in self.node_batch.items()},
            node_ids={key: list(value) for key, value in self.node_ids.items()},
            edge_index={key: value.to(target) for key, value in self.edge_index.items()},
            edge_relation_ids=dict(self.edge_relation_ids),
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

    def node_counts(self) -> Dict[str, int]:
        return {node_type: len(self.node_ids[node_type]) for node_type in NODE_TYPES}


class GraphTensorizer:
    """Convert framework-neutral observations into fixed-width PyTorch tensors."""

    def __init__(
        self,
        schema: GraphFeatureSchema,
        *,
        normalizer: Optional[NumericFeatureNormalizer] = None,
        add_reverse_edges: bool = True,
    ) -> None:
        self.schema = schema
        self.normalizer = normalizer
        self.add_reverse_edges = add_reverse_edges

    def tensorize(
        self,
        graphs: Union[TokenBindingGraph, Sequence[TokenBindingGraph]],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> TokenBindingBatch:
        graph_list = [graphs] if isinstance(graphs, TokenBindingGraph) else list(graphs)
        if not graph_list:
            raise ValueError("At least one TokenBindingGraph is required")
        for graph in graph_list:
            graph.validate()

        category_to_index = self.schema.category_to_index
        relation_to_index = self.schema.relation_to_index
        unknown_category = self.schema.unknown_category_index
        unknown_relation = self.schema.unknown_relation_index

        numeric_rows: Dict[str, List[List[float]]] = {
            node_type: [] for node_type in NODE_TYPES
        }
        categorical_rows: Dict[str, List[List[int]]] = {
            node_type: [] for node_type in NODE_TYPES
        }
        node_batches: Dict[str, List[int]] = {
            node_type: [] for node_type in NODE_TYPES
        }
        node_ids: Dict[str, List[str]] = {
            node_type: [] for node_type in NODE_TYPES
        }
        references: List[Dict[str, Tuple[str, int]]] = []

        for graph_index, graph in enumerate(graph_list):
            graph_references: Dict[str, Tuple[str, int]] = {}
            for node in graph.nodes.values():
                if node.node_type not in NUMERIC_FIELDS:
                    raise ValueError(f"Unsupported node type: {node.node_type}")
                node_type = node.node_type
                local_index = len(node_ids[node_type])
                graph_references[node.node_id] = (node_type, local_index)
                node_ids[node_type].append(node.node_id)
                node_batches[node_type].append(graph_index)
                numeric_row = [
                    float(node.numeric.get(field_name, 0.0))
                    for field_name in NUMERIC_FIELDS[node_type]
                ]
                if self.normalizer is not None:
                    numeric_row = self.normalizer.transform(node_type, numeric_row)
                numeric_rows[node_type].append(numeric_row)
                categorical_rows[node_type].append([
                    category_to_index.get(
                        str(node.categorical.get(field_name, MISSING_CATEGORY)),
                        unknown_category,
                    )
                    for field_name in CATEGORICAL_FIELDS[node_type]
                ])
            references.append(graph_references)

        edge_pairs: Dict[CanonicalEdgeType, List[Tuple[int, int]]] = {}
        for graph, graph_references in zip(graph_list, references):
            for edge in graph.edges:
                source_type, source_index = graph_references[edge.source]
                target_type, target_index = graph_references[edge.target]
                canonical = (source_type, edge.edge_type, target_type)
                edge_pairs.setdefault(canonical, []).append(
                    (source_index, target_index)
                )
                if self.add_reverse_edges:
                    reverse = (target_type, f"rev:{edge.edge_type}", source_type)
                    edge_pairs.setdefault(reverse, []).append(
                        (target_index, source_index)
                    )

        action_binding_index: List[int] = []
        action_batch: List[int] = []
        action_node_ids: List[str] = []
        action_keys: List[Tuple[Any, ...]] = []
        for graph_index, (graph, graph_references) in enumerate(
            zip(graph_list, references)
        ):
            for action_node_id, action_key in graph.action_keys.items():
                node_type, binding_index = graph_references[action_node_id]
                if node_type != "binding":
                    raise ValueError(f"Action is not a Binding node: {action_node_id}")
                action_binding_index.append(binding_index)
                action_batch.append(graph_index)
                action_node_ids.append(action_node_id)
                action_keys.append(action_key)

        target = torch.device(device)
        batch = TokenBindingBatch(
            num_graphs=len(graph_list),
            node_numeric={
                node_type: _matrix(
                    numeric_rows[node_type], len(NUMERIC_FIELDS[node_type]),
                    torch.float32, target
                )
                for node_type in NODE_TYPES
            },
            node_categorical={
                node_type: _matrix(
                    categorical_rows[node_type], len(CATEGORICAL_FIELDS[node_type]),
                    torch.long, target
                )
                for node_type in NODE_TYPES
            },
            node_batch={
                node_type: torch.tensor(
                    node_batches[node_type], dtype=torch.long, device=target
                )
                for node_type in NODE_TYPES
            },
            node_ids=node_ids,
            edge_index={
                canonical: torch.tensor(
                    pairs, dtype=torch.long, device=target
                ).t().contiguous()
                for canonical, pairs in edge_pairs.items()
            },
            edge_relation_ids={
                canonical: relation_to_index.get(canonical[1], unknown_relation)
                for canonical in edge_pairs
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


def _matrix(
    rows: List[List[Union[int, float]]],
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=dtype, device=device)
    return torch.tensor(rows, dtype=dtype, device=device).reshape(len(rows), width)


def _reserved_first(values: Iterable[str], first: str, second: str) -> Tuple[str, ...]:
    remainder = sorted(set(values) - {first, second})
    return (first, second, *remainder)
