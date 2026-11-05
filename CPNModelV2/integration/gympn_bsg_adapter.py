from __future__ import annotations

from typing import Any, Callable, Optional

from CPNModelV2.graph.schema import TokenBindingGraph
from CPNModelV2.integration.gympn_complex_scenarios import (
    complexity_scenario_factory,
)
from CPNModelV2.integration.gympn_scenarios import paper_scenario_factory
from CPNModelV2.integration.gympn_decision_adapter import (
    GymPNBindingEffectGraphBuilder,
    GymPNDecisionEnv,
)


NetFactory = Callable[[Optional[int]], Any]


class GymPNBindingGraphBuilder:
    """Expose GymPN Tokens and enabled Bindings without effect previews.

    The established GymPN adapter remains the single source of truth for
    visible Tokens, action keys, and participant roles.  This projection drops
    the previewed after-action effects so Generic Compact V5 receives exactly
    its Token--Binding observation contract.
    """

    def __init__(self, *, validate: bool = True) -> None:
        self.validate = bool(validate)
        self._source = GymPNBindingEffectGraphBuilder(
            validate=validate,
            include_effects=False,
        )

    def build(self, problem: Any, *, decision_id: int = 0) -> TokenBindingGraph:
        graph = self._source.build(problem, decision_id=decision_id)
        graph.metadata.update({
            "observation": "gympn_token_binding_graph",
            "adapter": type(self).__name__,
            "binding_effects_exposed": False,
        })
        graph.metadata.pop("effect_semantics", None)
        if self.validate:
            graph.validate()
        return graph


class GymPNBSGEnv(GymPNDecisionEnv):
    """GymPN decision-epoch environment for Generic Compact V5 / BSG-PPO."""

    def __init__(
        self,
        net_factory: NetFactory,
        *,
        horizon: float = 10.0,
        validate_graphs: bool = False,
    ) -> None:
        super().__init__(
            net_factory,
            horizon=horizon,
            graph_builder=GymPNBindingGraphBuilder(validate=validate_graphs),
        )


def build_bsg_paper_suite_environment(
    scenario_name: str,
    *,
    horizon: float = 10.0,
    case_multiplier: int = 1,
) -> GymPNBSGEnv:
    return GymPNBSGEnv(
        paper_scenario_factory(
            scenario_name,
            case_multiplier=case_multiplier,
        ),
        horizon=horizon,
        validate_graphs=False,
    )


def build_bsg_complex_suite_environment(
    scenario_name: str,
    complexity_level: str,
    *,
    horizon: float = 100.0,
    validate_graphs: bool = False,
) -> GymPNBSGEnv:
    """Build a predeclared environment for the Section 5.6 GymPN study."""

    return GymPNBSGEnv(
        complexity_scenario_factory(scenario_name, complexity_level),
        horizon=horizon,
        validate_graphs=validate_graphs,
    )
