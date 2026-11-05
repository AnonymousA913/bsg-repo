from __future__ import annotations

"""Isolated ablation variants for Generic Compact V5.

This module deliberately does *not* modify ``generic_compact_policy.py``.
It subclasses the frozen production network and changes only forward information
flow.  The inherited constructor, registered modules, parameter names, and
state-dict layout remain identical to ``GenericCompactPolicyNetwork``.

Variants
--------
``full``
    Exact production forward pass (delegates to ``super().forward``).
``no_reference``
    Bypass all token-reference message-passing layers (C2).
``no_relation_direction``
    Keep reference topology and Q/K/V propagation, but remove relation and
    direction embeddings from keys/messages.
``no_relation_type``
    Keep reference topology and propagation direction, but remove relation-type
    embeddings from keys/messages.
``no_direction``
    Keep reference topology and relation-type embeddings, but remove the
    original/reverse propagation-direction embeddings.
``flat_binding``
    Keep token participants but remove binding-role identity by pooling all
    participants together and repeating the same summary in every original role
    slot.  The existing fusion MLP is reused unchanged.
``no_reference_flat_binding``
    Joint semantic ablation. Bypass token-reference message passing and erase
    binding-role identity at the same time. This tests whether the two semantic
    stages can compensate for one another when removed separately.
"""

import math
from typing import Dict, List, Mapping, Tuple

import torch

from CPNModelV2.graph.generic_tensorizer import (
    BINDING_NODE_TYPE,
    GenericGraphFeatureSchema,
    GenericTokenBindingBatch,
)
from CPNModelV2.learning.generic_compact_policy import (
    GenericCompactPolicyNetwork,
    GenericReferenceAttentionLayer,
    GenericRoleConditionedBindingComposer,
    _segmented_softmax,
)
from CPNModelV2.learning.token_binding_policy import BindingPolicyOutput


FULL_VARIANT = "full"
NO_REFERENCE_VARIANT = "no_reference"
NO_RELATION_DIRECTION_VARIANT = "no_relation_direction"
NO_RELATION_TYPE_VARIANT = "no_relation_type"
NO_DIRECTION_VARIANT = "no_direction"
FLAT_BINDING_VARIANT = "flat_binding"
NO_REFERENCE_FLAT_BINDING_VARIANT = "no_reference_flat_binding"
GENERIC_COMPACT_ABLATION_VARIANTS = (
    FULL_VARIANT,
    NO_REFERENCE_VARIANT,
    NO_RELATION_DIRECTION_VARIANT,
    NO_RELATION_TYPE_VARIANT,
    NO_DIRECTION_VARIANT,
    FLAT_BINDING_VARIANT,
    NO_REFERENCE_FLAT_BINDING_VARIANT,
)


def normalize_ablation_variant(variant: str) -> str:
    normalized = str(variant).strip().lower()
    if normalized not in GENERIC_COMPACT_ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown Generic Compact ablation variant {variant!r}; "
            f"expected one of {GENERIC_COMPACT_ABLATION_VARIANTS}"
        )
    return normalized


class GenericCompactAblationPolicyNetwork(GenericCompactPolicyNetwork):
    """Generic Compact V5 with isolated, parameter-matched semantic ablations.

    The base class constructor is used unchanged.  Consequently, for the same
    PyTorch RNG state, an instance of this class has exactly the same initial
    registered parameters as ``GenericCompactPolicyNetwork``.  Existing Full
    checkpoints can also be loaded with ``strict=True``.
    """

    def __init__(
        self,
        schema: GenericGraphFeatureSchema,
        *,
        hidden_dim: int = 64,
        category_dim: int = 16,
        reference_layers: int = 2,
        competition_layers: int = 0,
        dropout: float = 0.0,
        ablation_variant: str = FULL_VARIANT,
    ) -> None:
        # Important: do not register or replace any neural modules here.
        # This keeps initialization order and state_dict layout identical to the
        # frozen production GenericCompactPolicyNetwork.
        super().__init__(
            schema,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            reference_layers=reference_layers,
            competition_layers=competition_layers,
            dropout=dropout,
        )
        self.ablation_variant = normalize_ablation_variant(ablation_variant)

    def forward(self, batch: GenericTokenBindingBatch) -> BindingPolicyOutput:
        # Full is delegated verbatim to the frozen production implementation.
        # This makes backward compatibility a structural guarantee rather than
        # a near-duplicate reimplementation.
        if self.ablation_variant == FULL_VARIANT:
            return super().forward(batch)

        hidden = self._encode_nodes(batch)

        if self.ablation_variant not in {
            NO_REFERENCE_VARIANT,
            NO_REFERENCE_FLAT_BINDING_VARIANT,
        }:
            for layer in self.reference_layers:
                if self.ablation_variant in {
                    NO_RELATION_DIRECTION_VARIANT,
                    NO_RELATION_TYPE_VARIANT,
                    NO_DIRECTION_VARIANT,
                }:
                    hidden = _reference_forward_with_semantic_mask(
                        layer,
                        hidden,
                        batch,
                        use_relation=(
                            self.ablation_variant != NO_RELATION_DIRECTION_VARIANT
                            and self.ablation_variant != NO_RELATION_TYPE_VARIANT
                        ),
                        use_direction=(
                            self.ablation_variant != NO_RELATION_DIRECTION_VARIANT
                            and self.ablation_variant != NO_DIRECTION_VARIANT
                        ),
                    )
                else:
                    hidden = layer(hidden, batch)

        if self.ablation_variant in {
            FLAT_BINDING_VARIANT,
            NO_REFERENCE_FLAT_BINDING_VARIANT,
        }:
            hidden[BINDING_NODE_TYPE] = _flat_binding_forward(
                self.binding_composer,
                hidden,
                batch,
            )
        else:
            hidden[BINDING_NODE_TYPE] = self.binding_composer(hidden, batch)

        return _policy_head_forward(self, hidden, batch)


def _reference_forward_with_semantic_mask(
    layer: GenericReferenceAttentionLayer,
    hidden: Dict[str, torch.Tensor],
    batch: GenericTokenBindingBatch,
    *,
    use_relation: bool,
    use_direction: bool,
) -> Dict[str, torch.Tensor]:
    """Run one C2 layer with selected edge-label channels disabled.

    The same layer parameters remain registered on the model, keeping every
    semantic ablation parameter matched to the Full network.
    """

    incoming: Dict[
        str,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ] = {
        node_type: []
        for node_type in layer.schema.token_types
    }

    for canonical, edge_index in batch.reference_edge_index.items():
        source_type, _relation, target_type = canonical
        source_index, target_index = edge_index
        relation_id = batch.relation_ids[canonical]
        _collect_direction_with_semantic_mask(
            layer,
            incoming[target_type],
            hidden[source_type],
            hidden[target_type],
            source_index,
            target_index,
            relation_id=relation_id,
            direction=0,
            use_relation=use_relation,
            use_direction=use_direction,
        )
        _collect_direction_with_semantic_mask(
            layer,
            incoming[source_type],
            hidden[target_type],
            hidden[source_type],
            target_index,
            source_index,
            relation_id=relation_id,
            direction=1,
            use_relation=use_relation,
            use_direction=use_direction,
        )

    updated = dict(hidden)
    for index, node_type in enumerate(layer.schema.token_types):
        current = hidden[node_type]
        if not incoming[node_type] or current.shape[0] == 0:
            continue
        targets = torch.cat([item[0] for item in incoming[node_type]])
        scores = torch.cat([item[1] for item in incoming[node_type]])
        messages = torch.cat([item[2] for item in incoming[node_type]])
        weights = _segmented_softmax(scores, targets, current.shape[0])
        aggregate = current.new_zeros(current.shape)
        aggregate.index_add_(0, targets, weights.unsqueeze(-1) * messages)
        proposal = layer.update[index](torch.cat([current, aggregate], dim=-1))
        updated[node_type] = layer.norm[index](current + proposal)
    return updated


def _collect_direction_with_semantic_mask(
    layer: GenericReferenceAttentionLayer,
    destination: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    source: torch.Tensor,
    target: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    *,
    relation_id: int,
    direction: int,
    use_relation: bool,
    use_direction: bool,
) -> None:
    if source_index.numel() == 0:
        return
    source_key = layer.key(source[source_index])
    source_value = layer.value(source[source_index])
    if use_relation:
        source_key = source_key + layer.relation_key.weight[relation_id]
        source_value = source_value + layer.relation_value.weight[relation_id]
    if use_direction:
        source_key = source_key + layer.direction_key.weight[direction]
        source_value = source_value + layer.direction_value.weight[direction]
    score = (
        layer.query(target[target_index]) * source_key
    ).sum(-1) / math.sqrt(layer.hidden_dim)
    destination.append((target_index, score, source_value))


def _flat_binding_forward(
    composer: GenericRoleConditionedBindingComposer,
    hidden: Mapping[str, torch.Tensor],
    batch: GenericTokenBindingBatch,
) -> torch.Tensor:
    """Compose a binding after erasing participant-role identity.

    All participants are mean-pooled into one summary.  That same summary is
    repeated into every original role slot, preserving the input width and all
    parameters of the production fusion MLP.
    """

    binding = hidden[BINDING_NODE_TYPE]
    flat_channel = binding.new_zeros(binding.shape)
    flat_count = binding.new_zeros((binding.shape[0], 1))

    for canonical, edge_index in batch.participation_edge_index.items():
        source_type, role, target_type = canonical
        if target_type != BINDING_NODE_TYPE:
            continue
        # Validate against the fitted schema, but deliberately do not use the
        # role index in the aggregation itself.
        if role not in composer.role_to_index:
            raise KeyError(f"Participation role absent from composer schema: {role}")
        token_index, binding_index = edge_index
        flat_channel.index_add_(
            0,
            binding_index,
            hidden[source_type][token_index],
        )
        flat_count.index_add_(
            0,
            binding_index,
            binding.new_ones((binding_index.shape[0], 1)),
        )

    flat_summary = flat_channel / flat_count.clamp_min(1.0)
    repeated = [flat_summary for _ in composer.role_to_index]
    proposal = composer.fusion(torch.cat([binding, *repeated], dim=-1))
    return composer.norm(binding + proposal)


def _policy_head_forward(
    model: GenericCompactPolicyNetwork,
    hidden: Mapping[str, torch.Tensor],
    batch: GenericTokenBindingBatch,
) -> BindingPolicyOutput:
    """Production C4 logic, reused for ablation-only forward paths."""

    pooled = torch.cat([
        model.poolers[index](
            hidden[node_type],
            batch.node_batch[node_type],
            batch.num_graphs,
        )
        for index, node_type in enumerate(model.schema.node_types)
    ], dim=-1)
    context = model.context_projection(pooled)
    binding_hidden = hidden[BINDING_NODE_TYPE][batch.action_binding_index]
    if binding_hidden.shape[0]:
        logits = model.actor(torch.cat([
            binding_hidden,
            context[batch.action_batch],
        ], dim=-1)).squeeze(-1)
    else:
        logits = context.new_empty((0,))
    probabilities = _segmented_softmax(
        logits,
        batch.action_batch,
        batch.num_graphs,
    )
    values = model.critic(pooled).squeeze(-1)
    return BindingPolicyOutput(logits, probabilities, values)
