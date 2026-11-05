from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from CPNModelV2.graph.schema import TokenBindingGraph


BINDING_NODE_TYPE = "binding"
PARTICIPATION_PREFIX = "participates::"
REFERENCE_PREFIX = "references::"
MISSING_CATEGORY = "<MISSING>"
UNKNOWN_CATEGORY = "<UNK>"
UNKNOWN_RELATION = "<UNK_RELATION>"
CanonicalEdgeType = Tuple[str, str, str]


def participation_relation(role: str) -> str:
    return f"{PARTICIPATION_PREFIX}{role}"


def reference_relation(relation: str) -> str:
    return f"{REFERENCE_PREFIX}{relation}"


@dataclass(frozen=True)
class GenericGraphFeatureSchema:
    """Runtime schema for a domain-independent Token-Binding graph.

    Token types and their feature fields are discovered from calibration
    markings plus optional model-declared Token shapes. The policy therefore
    receives the same abstract objects for a one-colour CPN and a heterogeneous
    attributed CPN; only this schema changes.
    """

    token_types: Tuple[str, ...]
    numeric_fields: Mapping[str, Tuple[str, ...]]
    categorical_fields: Mapping[str, Tuple[str, ...]]
    category_values: Tuple[str, ...]
    reference_relations: Tuple[str, ...]
    participation_roles: Tuple[str, ...]
    binary_numeric_fields: Mapping[str, Tuple[str, ...]]

    @classmethod
    def fit(cls, graphs: Iterable[TokenBindingGraph]) -> "GenericGraphFeatureSchema":
        graph_list = list(graphs)
        if not graph_list:
            raise ValueError("Generic schema fitting requires at least one graph")

        token_types = set()
        numeric: Dict[str, set[str]] = {}
        categorical: Dict[str, set[str]] = {}
        categories = {MISSING_CATEGORY, UNKNOWN_CATEGORY}
        references = set()
        roles = set()
        declared_role_order: List[str] = []
        declared_role_set = set()
        binary: Dict[str, set[str]] = {}

        for graph in graph_list:
            graph.validate()
            _validate_generic_graph(graph)
            declared_binary = graph.metadata.get("binary_numeric_fields", {})
            declared_references = graph.metadata.get(
                "declared_reference_relations", ()
            )
            if isinstance(declared_references, (str, bytes)):
                raise ValueError(
                    "declared_reference_relations metadata must be a sequence"
                )
            for raw_relation in declared_references:
                relation = str(raw_relation)
                if not relation.startswith(REFERENCE_PREFIX):
                    raise ValueError(
                        "Declared reference relation lacks the generic prefix: "
                        f"{relation}"
                    )
                references.add(relation)
            declared_features = graph.metadata.get(
                "declared_node_features", ()
            )
            if isinstance(declared_features, (str, bytes, Mapping)):
                raise ValueError(
                    "declared_node_features metadata must be a sequence"
                )
            for declaration in declared_features:
                if not isinstance(declaration, Mapping):
                    raise ValueError(
                        "Each declared node feature must be a mapping"
                    )
                node_type = str(declaration["node_type"])
                if node_type == BINDING_NODE_TYPE:
                    raise ValueError("A declared Token cannot be a Binding")
                declared_numeric = declaration.get("numeric", {})
                declared_categorical = declaration.get("categorical", {})
                if not isinstance(declared_numeric, Mapping) or not isinstance(
                    declared_categorical, Mapping
                ):
                    raise ValueError(
                        "Declared numeric/categorical features must be mappings"
                    )
                token_types.add(node_type)
                numeric.setdefault(node_type, set()).update(declared_numeric)
                categorical.setdefault(node_type, set()).update(
                    declared_categorical
                )
                categories.update(
                    str(value) for value in declared_categorical.values()
                )
                binary.setdefault(node_type, set()).update(
                    str(value) for value in declared_binary.get(node_type, ())
                )
            declared_roles = graph.metadata.get("participation_roles", ())
            if isinstance(declared_roles, (str, bytes)):
                raise ValueError("participation_roles metadata must be a sequence")
            for raw_role in declared_roles:
                role = str(raw_role)
                if not role.startswith(PARTICIPATION_PREFIX):
                    raise ValueError(
                        "Declared participation role lacks the generic prefix: "
                        f"{role}"
                    )
                if role not in declared_role_set:
                    declared_role_order.append(role)
                    declared_role_set.add(role)
                roles.add(role)
            for node in graph.nodes.values():
                node_type = node.node_type
                if node_type != BINDING_NODE_TYPE:
                    token_types.add(node_type)
                numeric.setdefault(node_type, set()).update(node.numeric)
                categorical.setdefault(node_type, set()).update(node.categorical)
                categories.update(str(value) for value in node.categorical.values())
                binary.setdefault(node_type, set()).update(
                    str(value) for value in declared_binary.get(node_type, ())
                )
            for edge in graph.edges:
                if edge.edge_type.startswith(PARTICIPATION_PREFIX):
                    roles.add(edge.edge_type)
                elif edge.edge_type.startswith(REFERENCE_PREFIX):
                    references.add(edge.edge_type)

        ordered_token_types = tuple(sorted(token_types))
        node_types = (*ordered_token_types, BINDING_NODE_TYPE)
        return cls(
            token_types=ordered_token_types,
            numeric_fields={
                node_type: tuple(sorted(numeric.get(node_type, ())))
                for node_type in node_types
            },
            categorical_fields={
                node_type: tuple(sorted(categorical.get(node_type, ())))
                for node_type in node_types
            },
            category_values=_reserved_first(
                categories, MISSING_CATEGORY, UNKNOWN_CATEGORY
            ),
            reference_relations=tuple(sorted(references)),
            participation_roles=tuple([
                *declared_role_order,
                *sorted(roles - declared_role_set),
            ]),
            binary_numeric_fields={
                node_type: tuple(sorted(binary.get(node_type, ())))
                for node_type in node_types
            },
        )

    @property
    def node_types(self) -> Tuple[str, ...]:
        return (*self.token_types, BINDING_NODE_TYPE)

    @property
    def category_to_index(self) -> Dict[str, int]:
        return {value: index for index, value in enumerate(self.category_values)}

    @property
    def relation_values(self) -> Tuple[str, ...]:
        return (
            UNKNOWN_RELATION,
            *self.reference_relations,
            *self.participation_roles,
        )

    @property
    def relation_to_index(self) -> Dict[str, int]:
        return {value: index for index, value in enumerate(self.relation_values)}

    @property
    def unknown_category_index(self) -> int:
        return self.category_to_index[UNKNOWN_CATEGORY]

    @property
    def unknown_relation_index(self) -> int:
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_types": list(self.token_types),
            "numeric_fields": {
                key: list(value) for key, value in self.numeric_fields.items()
            },
            "categorical_fields": {
                key: list(value) for key, value in self.categorical_fields.items()
            },
            "category_values": list(self.category_values),
            "reference_relations": list(self.reference_relations),
            "participation_roles": list(self.participation_roles),
            "binary_numeric_fields": {
                key: list(value)
                for key, value in self.binary_numeric_fields.items()
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GenericGraphFeatureSchema":
        token_types = tuple(values["token_types"])
        node_types = (*token_types, BINDING_NODE_TYPE)
        return cls(
            token_types=token_types,
            numeric_fields={
                node_type: tuple(values["numeric_fields"].get(node_type, ()))
                for node_type in node_types
            },
            categorical_fields={
                node_type: tuple(
                    values["categorical_fields"].get(node_type, ())
                )
                for node_type in node_types
            },
            category_values=tuple(values["category_values"]),
            reference_relations=tuple(values.get("reference_relations", ())),
            participation_roles=tuple(values.get("participation_roles", ())),
            binary_numeric_fields={
                node_type: tuple(
                    values.get("binary_numeric_fields", {}).get(node_type, ())
                )
                for node_type in node_types
            },
        )


@dataclass(frozen=True)
class GenericNumericFeatureNormalizer:
    means: Mapping[str, Tuple[float, ...]]
    scales: Mapping[str, Tuple[float, ...]]
    schema: GenericGraphFeatureSchema

    @classmethod
    def fit(
        cls,
        graphs: Iterable[TokenBindingGraph],
        schema: GenericGraphFeatureSchema,
        *,
        minimum_scale: float = 1e-6,
    ) -> "GenericNumericFeatureNormalizer":
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        rows: Dict[str, List[List[float]]] = {
            node_type: [] for node_type in schema.node_types
        }
        for graph in graphs:
            _validate_graph_against_schema(graph, schema)
            for declaration in graph.metadata.get(
                "declared_node_features", ()
            ):
                node_type = str(declaration["node_type"])
                declared_numeric = declaration.get("numeric", {})
                rows[node_type].append([
                    float(declared_numeric.get(field, 0.0))
                    for field in schema.numeric_fields[node_type]
                ])
            for node in graph.nodes.values():
                rows[node.node_type].append([
                    float(node.numeric.get(field, 0.0))
                    for field in schema.numeric_fields[node.node_type]
                ])

        means: Dict[str, Tuple[float, ...]] = {}
        scales: Dict[str, Tuple[float, ...]] = {}
        for node_type in schema.node_types:
            fields = schema.numeric_fields[node_type]
            if not fields or not rows[node_type]:
                means[node_type] = (0.0,) * len(fields)
                scales[node_type] = (1.0,) * len(fields)
                continue
            columns = list(zip(*rows[node_type]))
            type_means = tuple(sum(column) / len(column) for column in columns)
            type_scales = []
            for field, column, feature_mean in zip(fields, columns, type_means):
                if field in schema.binary_numeric_fields[node_type]:
                    type_scales.append(1.0)
                    continue
                variance = sum(
                    (value - feature_mean) ** 2 for value in column
                ) / len(column)
                scale = math.sqrt(variance)
                type_scales.append(scale if scale >= minimum_scale else 1.0)
            means[node_type] = type_means
            scales[node_type] = tuple(type_scales)
        return cls(means=means, scales=scales, schema=schema)

    def transform(self, node_type: str, row: Sequence[float]) -> List[float]:
        fields = self.schema.numeric_fields[node_type]
        if len(row) != len(fields):
            raise ValueError(f"Invalid numeric width for {node_type}")
        binary = set(self.schema.binary_numeric_fields[node_type])
        return [
            float(value)
            if field in binary
            else (float(value) - mean) / scale
            for field, value, mean, scale in zip(
                fields,
                row,
                self.means[node_type],
                self.scales[node_type],
            )
        ]

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            "means": {key: list(value) for key, value in self.means.items()},
            "scales": {key: list(value) for key, value in self.scales.items()},
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Mapping[str, Sequence[float]]],
        schema: GenericGraphFeatureSchema,
    ) -> "GenericNumericFeatureNormalizer":
        return cls(
            means={
                node_type: tuple(
                    float(value) for value in values["means"][node_type]
                )
                for node_type in schema.node_types
            },
            scales={
                node_type: tuple(
                    float(value) for value in values["scales"][node_type]
                )
                for node_type in schema.node_types
            },
            schema=schema,
        )


@dataclass
class GenericTokenBindingBatch:
    num_graphs: int
    node_numeric: Dict[str, torch.Tensor]
    node_categorical: Dict[str, torch.Tensor]
    node_batch: Dict[str, torch.Tensor]
    node_ids: Dict[str, List[str]]
    reference_edge_index: Dict[CanonicalEdgeType, torch.Tensor]
    participation_edge_index: Dict[CanonicalEdgeType, torch.Tensor]
    relation_ids: Dict[CanonicalEdgeType, int]
    action_binding_index: torch.Tensor
    action_batch: torch.Tensor
    action_node_ids: List[str]
    action_keys: List[Tuple[Any, ...]]
    graph_times: torch.Tensor
    schema: GenericGraphFeatureSchema

    def validate(self) -> None:
        if self.num_graphs <= 0:
            raise ValueError("A generic batch must contain at least one graph")
        for node_type in self.schema.node_types:
            count = len(self.node_ids[node_type])
            if self.node_numeric[node_type].shape != (
                count,
                len(self.schema.numeric_fields[node_type]),
            ):
                raise ValueError(f"Invalid numeric shape for {node_type}")
            if self.node_categorical[node_type].shape != (
                count,
                len(self.schema.categorical_fields[node_type]),
            ):
                raise ValueError(f"Invalid categorical shape for {node_type}")
            if self.node_batch[node_type].shape != (count,):
                raise ValueError(f"Invalid batch vector for {node_type}")
            if not torch.isfinite(self.node_numeric[node_type]).all():
                raise ValueError(f"Non-finite features for {node_type}")
        for edge_map in (
            self.reference_edge_index,
            self.participation_edge_index,
        ):
            for canonical, edge_index in edge_map.items():
                if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                    raise ValueError(f"Invalid edge index for {canonical}")
        if self.action_binding_index.shape != (len(self.action_keys),):
            raise ValueError("Invalid action_binding_index")
        if self.action_batch.shape != (len(self.action_keys),):
            raise ValueError("Invalid action_batch")

    def to(self, device: Union[str, torch.device]) -> "GenericTokenBindingBatch":
        target = torch.device(device)
        return GenericTokenBindingBatch(
            num_graphs=self.num_graphs,
            node_numeric={key: value.to(target) for key, value in self.node_numeric.items()},
            node_categorical={
                key: value.to(target) for key, value in self.node_categorical.items()
            },
            node_batch={key: value.to(target) for key, value in self.node_batch.items()},
            node_ids={key: list(value) for key, value in self.node_ids.items()},
            reference_edge_index={
                key: value.to(target)
                for key, value in self.reference_edge_index.items()
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
        )

    @property
    def device(self) -> torch.device:
        return self.graph_times.device


class GenericGraphTensorizer:
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

    def tensorize(
        self,
        graphs: Union[TokenBindingGraph, Sequence[TokenBindingGraph]],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> GenericTokenBindingBatch:
        graph_list = [graphs] if isinstance(graphs, TokenBindingGraph) else list(graphs)
        if not graph_list:
            raise ValueError("At least one generic graph is required")
        if self.validate_graphs:
            for graph in graph_list:
                graph.validate()
                _validate_graph_against_schema(graph, self.schema)

        numeric_rows = {node_type: [] for node_type in self.schema.node_types}
        categorical_rows = {node_type: [] for node_type in self.schema.node_types}
        node_batches = {node_type: [] for node_type in self.schema.node_types}
        node_ids = {node_type: [] for node_type in self.schema.node_types}
        references: List[Dict[str, Tuple[str, int]]] = []
        # Freeze one field layout for the complete batch. Besides avoiding
        # repeated mapping lookups, this guarantees that row construction and
        # final tensor widths use the same immutable schema view.
        numeric_fields = {
            node_type: tuple(self.schema.numeric_fields[node_type])
            for node_type in self.schema.node_types
        }
        categorical_fields = {
            node_type: tuple(self.schema.categorical_fields[node_type])
            for node_type in self.schema.node_types
        }
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
                    for field in numeric_fields[node_type]
                ]
                if self.normalizer is not None:
                    numeric = self.normalizer.transform(node_type, numeric)
                numeric_rows[node_type].append(numeric)
                categorical_rows[node_type].append([
                    category_to_index.get(
                        str(node.categorical.get(field, MISSING_CATEGORY)),
                        self.schema.unknown_category_index,
                    )
                    for field in categorical_fields[node_type]
                ])
            references.append(graph_references)

        reference_pairs: Dict[CanonicalEdgeType, List[Tuple[int, int]]] = {}
        participation_pairs: Dict[CanonicalEdgeType, List[Tuple[int, int]]] = {}
        for graph, graph_references in zip(graph_list, references):
            for edge in graph.edges:
                source_type, source_index = graph_references[edge.source]
                target_type, target_index = graph_references[edge.target]
                canonical = (source_type, edge.edge_type, target_type)
                destination = (
                    participation_pairs
                    if edge.edge_type.startswith(PARTICIPATION_PREFIX)
                    else reference_pairs
                )
                destination.setdefault(canonical, []).append(
                    (source_index, target_index)
                )

        action_binding_index: List[int] = []
        action_batch: List[int] = []
        action_node_ids: List[str] = []
        action_keys: List[Tuple[Any, ...]] = []
        for graph_index, (graph, graph_references) in enumerate(
            zip(graph_list, references)
        ):
            for node_id, binding_key in graph.action_keys.items():
                node_type, node_index = graph_references[node_id]
                if node_type != BINDING_NODE_TYPE:
                    raise ValueError(f"Action is not a Binding: {node_id}")
                action_binding_index.append(node_index)
                action_batch.append(graph_index)
                action_node_ids.append(node_id)
                action_keys.append(binding_key)

        target = torch.device(device)
        relation_to_index = self.schema.relation_to_index
        all_canonical = set(reference_pairs) | set(participation_pairs)
        batch = GenericTokenBindingBatch(
            num_graphs=len(graph_list),
            node_numeric={
                node_type: _matrix(
                    numeric_rows[node_type],
                    len(numeric_fields[node_type]),
                    torch.float32,
                    target,
                )
                for node_type in self.schema.node_types
            },
            node_categorical={
                node_type: _matrix(
                    categorical_rows[node_type],
                    len(categorical_fields[node_type]),
                    torch.long,
                    target,
                    pad_value=category_to_index[MISSING_CATEGORY],
                )
                for node_type in self.schema.node_types
            },
            node_batch={
                node_type: torch.tensor(
                    node_batches[node_type], dtype=torch.long, device=target
                )
                for node_type in self.schema.node_types
            },
            node_ids=node_ids,
            reference_edge_index={
                canonical: torch.tensor(
                    pairs, dtype=torch.long, device=target
                ).t().contiguous()
                for canonical, pairs in reference_pairs.items()
            },
            participation_edge_index={
                canonical: torch.tensor(
                    pairs, dtype=torch.long, device=target
                ).t().contiguous()
                for canonical, pairs in participation_pairs.items()
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


def _validate_generic_graph(graph: TokenBindingGraph) -> None:
    for edge in graph.edges:
        is_participation = edge.edge_type.startswith(PARTICIPATION_PREFIX)
        is_reference = edge.edge_type.startswith(REFERENCE_PREFIX)
        if is_participation == is_reference:
            raise ValueError(
                f"Generic edge must be exactly one semantic kind: {edge.edge_type}"
            )
        if is_participation:
            if graph.nodes[edge.source].node_type == BINDING_NODE_TYPE:
                raise ValueError("Participation edge must originate at a Token")
            if graph.nodes[edge.target].node_type != BINDING_NODE_TYPE:
                raise ValueError("Participation edge must target a Binding")
        elif (
            graph.nodes[edge.source].node_type == BINDING_NODE_TYPE
            or graph.nodes[edge.target].node_type == BINDING_NODE_TYPE
        ):
            raise ValueError("Reference edges must connect Token instances")


def _validate_graph_against_schema(
    graph: TokenBindingGraph,
    schema: GenericGraphFeatureSchema,
) -> None:
    _validate_generic_graph(graph)
    unexpected_types = {
        node.node_type for node in graph.nodes.values()
        if node.node_type not in schema.node_types
    }
    if unexpected_types:
        raise ValueError(f"Node types absent from schema: {sorted(unexpected_types)}")
    unexpected_edges = {
        edge.edge_type for edge in graph.edges
        if edge.edge_type not in schema.reference_relations
        and edge.edge_type not in schema.participation_roles
    }
    if unexpected_edges:
        raise ValueError(f"Relations absent from schema: {sorted(unexpected_edges)}")


def _matrix(
    rows: List[List[Union[int, float]]],
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    pad_value: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=dtype, device=device)
    invalid = [
        (index, len(row))
        for index, row in enumerate(rows)
        if len(row) != width
    ]
    if invalid:
        can_pad = (
            pad_value is not None
            and all(actual < width for _index, actual in invalid)
        )
        if can_pad:
            warnings.warn(
                "Short categorical feature rows were padded with the "
                f"reserved missing value: expected {width}; "
                + ", ".join(
                    f"row {index}: {actual}"
                    for index, actual in invalid[:8]
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            rows = [
                [
                    *row,
                    *([pad_value] * (width - len(row))),
                ]
                for row in rows
            ]
            invalid = []
    if invalid:
        preview = ", ".join(
            f"row {index}: {actual}"
            for index, actual in invalid[:8]
        )
        raise ValueError(
            "Feature matrix row width mismatch: "
            f"expected {width}; {preview}"
        )
    flat = [value for row in rows for value in row]
    return torch.tensor(
        flat, dtype=dtype, device=device
    ).reshape(len(rows), width)


def _reserved_first(
    values: Iterable[str],
    first: str,
    second: Optional[str] = None,
) -> Tuple[str, ...]:
    reserved = {first} if second is None else {first, second}
    prefix = (first,) if second is None else (first, second)
    return (*prefix, *sorted(set(values) - reserved))
