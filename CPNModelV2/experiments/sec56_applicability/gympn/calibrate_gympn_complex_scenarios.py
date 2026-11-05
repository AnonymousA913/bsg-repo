from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _project_root = next(
        parent for parent in _Path(__file__).resolve().parents
        if (parent / "CPNModelV2").is_dir()
    )
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "gympn-main"))

import csv
from pathlib import Path
import random
from statistics import mean
from typing import Any, Dict, Iterable, Tuple

from CPNModelV2.graph.generic_tensorizer import (
    PARTICIPATION_PREFIX,
    REFERENCE_PREFIX,
)
from CPNModelV2.integration.gympn_bsg_adapter import (
    build_bsg_complex_suite_environment,
)
from CPNModelV2.integration.gympn_complex_scenarios import (
    PAPER_COMPLEXITY_LABELS,
    PAPER_COMPLEXITY_LEVELS,
)
from CPNModelV2.integration.gympn_scenarios import PAPER_SCENARIO_NAMES


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RUN_PROFILE = "structure"  # "structure", "pilot", or "full"
LEVELS = PAPER_COMPLEXITY_LEVELS
SCENARIOS = PAPER_SCENARIO_NAMES
RULES = ("random", "shortest_processing")

PROFILES = {
    # No rollouts: validates all 40 formal graph structures in a few seconds.
    "structure": {"horizon": 100.0, "seeds": ()},
    # One paired stream for iterative environment calibration.
    "pilot": {"horizon": 100.0, "seeds": (31_000,)},
    # Frozen-environment diagnostic; not a learned-model paper result.
    "full": {"horizon": 100.0, "seeds": (31_000, 31_001, 31_002)},
}
if RUN_PROFILE not in PROFILES:
    raise ValueError(f"Unknown calibration profile: {RUN_PROFILE}")
HORIZON = float(PROFILES[RUN_PROFILE]["horizon"])
EVALUATION_SEEDS = tuple(PROFILES[RUN_PROFILE]["seeds"])

PACKAGE_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if parent.name == "CPNModelV2"
)
OUTPUT_DIR = (
    PACKAGE_ROOT / "results" / "sec56_applicability"
    / "gympn_relational_complexity" / "calibration" / "paper_ladder_v2"
)
# =============================================================================


def _choose_action(graph: Any, rule: str, rng: random.Random) -> Tuple[Any, ...]:
    if rule == "random":
        return rng.choice(list(graph.action_keys.values()))
    if rule != "shortest_processing":
        raise ValueError(f"Unknown calibration rule: {rule}")

    processing_by_binding: Dict[str, list[float]] = {
        binding_node_id: [] for binding_node_id in graph.action_keys
    }
    for edge in graph.edges:
        if edge.target not in processing_by_binding:
            continue
        value = graph.nodes[edge.source].numeric.get("value_processing_time")
        if value is not None:
            processing_by_binding[edge.target].append(float(value))

    ranked = []
    for binding_node_id, action_key in graph.action_keys.items():
        processing_times = processing_by_binding[binding_node_id]
        if not processing_times:
            raise ValueError(
                f"Binding {binding_node_id} has no visible processing-time fact"
            )
        ranked.append((min(processing_times), binding_node_id, action_key))
    return min(ranked, key=lambda item: (item[0], item[1]))[2]


def _rollout(
    scenario_name: str,
    level: str,
    seed: int,
    rule: str,
) -> Dict[str, Any]:
    env = build_bsg_complex_suite_environment(
        scenario_name,
        level,
        horizon=HORIZON,
    )
    graph = env.reset(seed=seed)
    rng = random.Random(seed)
    action_counts = []
    token_counts = []
    reference_counts = []
    while True:
        action_counts.append(len(graph.action_keys))
        token_counts.append(sum(
            node.node_type != "binding" for node in graph.nodes.values()
        ))
        reference_counts.append(sum(
            edge.edge_type.startswith(REFERENCE_PREFIX) for edge in graph.edges
        ))
        step = env.step(_choose_action(graph, rule, rng))
        graph = step.observation
        if step.terminated:
            return {
                "scenario": scenario_name,
                "level": level,
                "paper_level": PAPER_COMPLEXITY_LABELS[level],
                "seed": seed,
                "rule": rule,
                "completed_cases": float(step.info["completed_tasks"]),
                "decision_count": int(step.info["decision_count"]),
                "mean_enabled_bindings": mean(action_counts),
                "max_enabled_bindings": max(action_counts),
                "mean_visible_tokens": mean(token_counts),
                "mean_reference_edges": mean(reference_counts),
            }


def _initial_structure(scenario_name: str, level: str) -> Dict[str, Any]:
    graph = build_bsg_complex_suite_environment(
        scenario_name,
        level,
        horizon=HORIZON,
        validate_graphs=True,
    ).reset(seed=31_000)
    return {
        "scenario": scenario_name,
        "level": level,
        "paper_level": PAPER_COMPLEXITY_LABELS[level],
        "nodes": len(graph.nodes),
        "visible_tokens": sum(
            node.node_type != "binding" for node in graph.nodes.values()
        ),
        "enabled_bindings": len(graph.action_keys),
        "reference_edges": sum(
            edge.edge_type.startswith(REFERENCE_PREFIX) for edge in graph.edges
        ),
        "participation_edges": sum(
            edge.edge_type.startswith(PARTICIPATION_PREFIX)
            for edge in graph.edges
        ),
        "arrival_interval": float(graph.metadata["arrival_interval"]),
        "target_offered_load": float(graph.metadata["target_offered_load"]),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    structure_rows = [
        _initial_structure(scenario_name, level)
        for level in LEVELS
        for scenario_name in SCENARIOS
    ]
    _write_csv(OUTPUT_DIR / "initial_structure.csv", structure_rows)
    print(f"Wrote {len(structure_rows)} structural rows to {OUTPUT_DIR}")
    if EVALUATION_SEEDS:
        rollout_rows = [
            _rollout(scenario_name, level, seed, rule)
            for level in LEVELS
            for scenario_name in SCENARIOS
            for rule in RULES
            for seed in EVALUATION_SEEDS
        ]
        _write_csv(OUTPUT_DIR / "calibration_rollouts.csv", rollout_rows)
        print(f"Wrote {len(rollout_rows)} rollout rows to {OUTPUT_DIR}")
    else:
        print("Structure profile selected; rollout calibration was skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
