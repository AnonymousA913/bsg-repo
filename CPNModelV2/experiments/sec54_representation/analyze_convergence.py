from __future__ import annotations

"""Create a descriptive convergence audit for Section 5.4 training runs."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    PACKAGE_ROOT
    / "results"
    / "sec54_representation"
    / "training"
    / "formal_500x20"
    / "SD1"
)
OUTPUT_ROOT = (
    PACKAGE_ROOT / "results" / "sec54_representation" / "analysis"
)

METHOD_LABELS = {
    "ag_ppo": "AG-PPO",
    "masked_mlp": "PetriRL-style MLP",
    "pnc_ppo": "PNC-PPO",
    "petri_net_gcn_ppo": "PN-GCN-PPO",
    "generic_compact_v5": "BSG-RL",
}
METHOD_COLORS = {
    "ag_ppo": "#C44E52",
    "masked_mlp": "#7A8B99",
    "pnc_ppo": "#D49A6A",
    "petri_net_gcn_ppo": "#7B6FAE",
    "generic_compact_v5": "#287D8E",
}
SIZE_ORDER = ("10x5", "15x10", "20x5", "20x10")
METHOD_ORDER = tuple(METHOD_LABELS)


def _read_run(history_path: Path) -> tuple[pd.DataFrame, dict]:
    method = history_path.parents[1].name
    size = history_path.parents[2].name
    seed = int(history_path.parent.name.removeprefix("seed_"))
    frame = pd.read_csv(history_path)
    frame = frame.dropna(subset=["validation_mean_makespan"]).copy()
    frame["iteration"] = frame["iteration"].astype(int)
    frame["method"] = method
    frame["size"] = size
    frame["seed"] = seed

    final_iteration = int(frame["iteration"].max())
    complete = final_iteration >= 500
    best_row = frame.loc[frame["validation_mean_makespan"].idxmin()]
    tail = frame[
        frame["iteration"] >= max(0, final_iteration - 100)
    ].copy()

    slope = np.nan
    p_value = np.nan
    relative_improvement = np.nan
    half_improvement = np.nan
    if complete and len(tail) >= 5:
        regression = linregress(
            tail["iteration"], tail["validation_mean_makespan"]
        )
        slope = float(regression.slope * 100.0)
        p_value = float(regression.pvalue)
        relative_improvement = float(
            -slope / tail["validation_mean_makespan"].mean() * 100.0
        )
        midpoint = len(tail) // 2
        first_mean = tail.iloc[:midpoint][
            "validation_mean_makespan"
        ].mean()
        second_mean = tail.iloc[midpoint:][
            "validation_mean_makespan"
        ].mean()
        half_improvement = float(
            (first_mean - second_mean) / first_mean * 100.0
        )

    if not complete:
        assessment = "incomplete"
    elif relative_improvement > 0.5 and p_value < 0.10:
        assessment = "still improving"
    elif (
        int(best_row["iteration"]) >= 480
        or relative_improvement > 0.5
    ):
        assessment = "borderline"
    else:
        assessment = "plateau-like"

    summary = {
        "size": size,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": seed,
        "complete_to_500": complete,
        "final_iteration": final_iteration,
        "best_iteration": int(best_row["iteration"]),
        "best_validation_makespan": float(
            best_row["validation_mean_makespan"]
        ),
        "final_validation_makespan": float(
            frame.iloc[-1]["validation_mean_makespan"]
        ),
        "final_gap_from_best_percent": float(
            (
                frame.iloc[-1]["validation_mean_makespan"]
                / best_row["validation_mean_makespan"]
                - 1.0
            )
            * 100.0
        ),
        "last100_slope_makespan_per_100_iterations": slope,
        "last100_relative_improvement_percent": relative_improvement,
        "last100_half_to_half_improvement_percent": half_improvement,
        "last100_slope_p_value_descriptive": p_value,
        "assessment": assessment,
    }
    return frame, summary


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for history_path in sorted(
        INPUT_ROOT.glob("*/*/seed_*/validation_history.csv")
    ):
        frame, summary = _read_run(history_path)
        frames.append(frame)
        summaries.append(summary)
    if not frames:
        raise FileNotFoundError(f"No validation histories under {INPUT_ROOT}")
    curves = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    summary["size"] = pd.Categorical(
        summary["size"], SIZE_ORDER, ordered=True
    )
    summary["method"] = pd.Categorical(
        summary["method"], METHOD_ORDER, ordered=True
    )
    summary = summary.sort_values(["size", "method"]).reset_index(drop=True)
    return curves, summary


def plot_curves(curves: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    figure, axes = plt.subplots(
        2, 2, figsize=(7.1, 4.8), sharex=True, constrained_layout=True
    )
    for axis, size in zip(axes.flat, SIZE_ORDER):
        size_data = curves[curves["size"] == size]
        for method in METHOD_ORDER:
            run = size_data[size_data["method"] == method].sort_values(
                "iteration"
            )
            if run.empty:
                continue
            color = METHOD_COLORS[method]
            complete = int(run["iteration"].max()) >= 500
            axis.plot(
                run["iteration"],
                run["validation_mean_makespan"],
                color=color,
                alpha=0.28,
                linewidth=0.8,
                marker="o",
                markersize=1.8,
            )
            smooth = run["validation_mean_makespan"].rolling(
                3, center=True, min_periods=1
            ).mean()
            axis.plot(
                run["iteration"],
                smooth,
                color=color,
                linewidth=1.6,
                linestyle="-" if complete else "--",
                label=METHOD_LABELS[method],
            )
            best = run.loc[run["validation_mean_makespan"].idxmin()]
            axis.scatter(
                [best["iteration"]],
                [best["validation_mean_makespan"]],
                color=color,
                marker="*",
                s=34,
                edgecolors="white",
                linewidths=0.4,
                zorder=5,
            )
        axis.axvspan(400, 500, color="#D9D9D9", alpha=0.18, linewidth=0)
        axis.set_title(size, fontweight="bold")
        axis.set_xlim(0, 500)
        axis.grid(axis="y", color="#D7D7D7", linewidth=0.45, alpha=0.7)
        axis.set_ylabel("Validation makespan (lower is better)")
        axis.set_xlabel("Training iteration")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.suptitle(
        "Section 5.4 convergence audit — seed 9000 only",
        fontsize=9,
        fontweight="bold",
    )
    return figure


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    curves, summary = load_runs()
    summary.to_csv(OUTPUT_ROOT / "sec54_convergence_audit.csv", index=False)
    figure = plot_curves(curves, summary)
    stem = OUTPUT_ROOT / "sec54_seed9000_convergence_audit"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(summary.to_string(index=False))
    print(f"Wrote convergence audit to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
