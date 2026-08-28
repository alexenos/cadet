"""Regenerate the paper's figures from the model and from sweep results.

    python scripts/make_figures.py --figures 3 2 --out figures/

Figure 3 (conditional visibility vs. block-averaged cloud measurement) and the
Figure 2-style environment snapshot depend only on the models, so they can be
produced without training anything.  Figures 4-6 read ``results/sweep.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadet.clouds import sigma_for_lookahead_width  # noqa: E402
from cadet.config import (  # noqa: E402
    CONTROLLERS,
    LOOKAHEAD_WIDTHS,
    CloudConfig,
    make_env_config,
)
from cadet.experiments import load_results  # noqa: E402
from cadet.visibility import PAPER_SIGMA_A, conditional_visibility  # noqa: E402

BUDGET_COLOURS = {100.0: "#d62728", 150.0: "#2ca02c", 1500.0: "#1f77b4"}
CONTROLLER_STYLE = {"cadet": ("o", "--"), "cadet-plan": ("s", "-")}


def figure_visibility(out: Path, use_paper_sigma: bool = False) -> Path:
    """Figure 3: conditional visibility probability vs. ``Y_A``."""
    cfg = CloudConfig()
    y = np.linspace(0.001, 0.999, 500)
    figure, axis = plt.subplots(figsize=(6.0, 4.5), dpi=150)
    for width in LOOKAHEAD_WIDTHS:
        sigma = PAPER_SIGMA_A[width] if use_paper_sigma else sigma_for_lookahead_width(width, cfg)
        axis.plot(
            y,
            conditional_visibility(y, sigma, cfg.tau),
            label=f"FOV = {width} px ($\\sigma_A$ = {sigma:.2f})",
        )
    axis.axvline(cfg.tau, color="grey", linestyle=":", label=f"Threshold = {cfg.tau}")
    axis.set_xlabel("Observed block-averaged cloud fraction $Y_A$")
    axis.set_ylabel("Probability target is clear")
    axis.set_title("Visibility probability as a function\nof block average cloud measurement")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = out / "figure3_visibility.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_environment(out: Path) -> Path:
    """Figure 2: a snapshot of the AoR after a lookahead and a few rolls."""
    from cadet.env import (
        MOVE_LEFT,
        MOVE_NOOP,
        SENSE_LOOKAHEAD,
        SENSE_NOOP,
        DynamicTaskingEnv,
    )
    from cadet.render import save_episode_figure

    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=300))
    env.reset(seed=7)
    env.step([MOVE_NOOP, SENSE_LOOKAHEAD])
    for _ in range(5):
        env.step([MOVE_NOOP, SENSE_NOOP])
    for _ in range(5):
        env.step([MOVE_LEFT, SENSE_NOOP])
    path = out / "figure2_environment.png"
    save_episode_figure(env, str(path), title="Area of regard at epoch 11")
    return path


def figure_captures(out: Path, results) -> Path:
    """Figure 4: targets captured vs. lookahead width, across budgets."""
    figure, axis = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    widths = sorted({r["lookahead_width"] for r in results})

    for controller in CONTROLLERS:
        marker, linestyle = CONTROLLER_STYLE[controller]
        for budget in sorted({float(r["budget"]) for r in results}):
            rows = sorted(
                (
                    r
                    for r in results
                    if r["controller"] == controller and float(r["budget"]) == budget
                ),
                key=lambda r: r["lookahead_width"],
            )
            if not rows:
                continue
            axis.plot(
                [r["lookahead_width"] for r in rows],
                [r["captured_targets"] for r in rows],
                marker=marker,
                linestyle=linestyle,
                color=BUDGET_COLOURS.get(budget, "grey"),
                label=f"{controller}, $\\bar P$={budget:.0f}",
            )

    ssp = np.mean([r["ssp_captured"] for r in results])
    oracle = np.mean([r["oracle_captured"] for r in results])
    axis.axhline(ssp, color="#e377c2", linestyle="-", linewidth=2, label="SSP")
    axis.axhline(oracle, color="#17becf", linestyle="-", linewidth=2, label="Oracle")

    axis.set_xscale("log", base=2)
    axis.set_xticks(widths)
    axis.set_xticklabels([str(w) for w in widths])
    axis.set_xlabel("Lookahead width (pixels)")
    axis.set_ylabel("Targets captured per episode")
    axis.set_title("Targets Captured vs. Lookahead Width Across Power Budgets")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = out / "figure4_captures.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_accuracy(out: Path, results) -> Path:
    """Figure 5: capture accuracy vs. total captures."""
    figure, axis = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    sizes = {8: 30, 16: 60, 32: 100, 64: 150}
    for row in results:
        marker = CONTROLLER_STYLE[row["controller"]][0]
        axis.scatter(
            row["captured_targets"],
            row["capture_accuracy"],
            s=sizes.get(row["lookahead_width"], 60),
            marker=marker,
            color=BUDGET_COLOURS.get(float(row["budget"]), "grey"),
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4,
        )
    axis.set_xlabel("Targets captured per episode")
    axis.set_ylabel("Capture accuracy")
    axis.set_title("Capture Accuracy vs. Targets Captured")
    axis.grid(alpha=0.25)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour, label=f"$\\bar P$={b:.0f}")
        for b, colour in BUDGET_COLOURS.items()
    ] + [
        plt.Line2D([], [], marker=CONTROLLER_STYLE[c][0], linestyle="", color="grey", label=c)
        for c in CONTROLLERS
    ]
    axis.legend(handles=handles, fontsize=8)
    figure.tight_layout()
    path = out / "figure5_accuracy.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_behaviour(out: Path, results) -> Path:
    """Figure 6: sensing and movement action composition for CADET-Plan."""
    rows = [r for r in results if r["controller"] == "cadet-plan"]
    if not rows:
        rows = results
    rows = sorted(rows, key=lambda r: (float(r["budget"]), r["lookahead_width"]))
    labels = [f"{int(r['budget'])}\nn={r['lookahead_width']}" for r in rows]
    index = np.arange(len(rows))

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), dpi=150, sharex=True)

    sensing = [
        ("Payload", "n_payload", "#d62728"),
        ("Lookahead", "n_lookahead", "#1f77b4"),
        ("No-op", "n_sense_noop", "#bbbbbb"),
    ]
    bottom = np.zeros(len(rows))
    for label, key, colour in sensing:
        values = np.array([r.get(key, 0.0) for r in rows])
        axes[0].bar(index, values, bottom=bottom, label=label, color=colour)
        bottom += values
    axes[0].set_ylabel("Actions per episode")
    axes[0].set_title("Sensing actions")
    axes[0].legend(fontsize=8)

    movement = [
        ("Direct roll", "n_roll", "#2ca02c"),
        ("Delegate to planner", "n_delegate", "#9467bd"),
        ("No-op", "n_move_noop", "#bbbbbb"),
    ]
    bottom = np.zeros(len(rows))
    for label, key, colour in movement:
        values = np.array([r.get(key, 0.0) for r in rows])
        axes[1].bar(index, values, bottom=bottom, label=label, color=colour)
        bottom += values
    axes[1].set_ylabel("Actions per episode")
    axes[1].set_title("Move actions")
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_xlabel("Power budget and lookahead width (pixels)")

    figure.suptitle("Behavioural Analysis: Sensing and Motion Composition")
    figure.tight_layout()
    path = out / "figure6_behaviour.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", nargs="+", default=["2", "3", "4", "5", "6"], help="which figures"
    )
    parser.add_argument("--results", default="results/sweep.json")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--paper-sigma", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)

    written = []
    if "3" in args.figures:
        written.append(figure_visibility(out, args.paper_sigma))
    if "2" in args.figures:
        written.append(figure_environment(out))
    for key, builder in (("4", figure_captures), ("5", figure_accuracy), ("6", figure_behaviour)):
        if key in args.figures:
            if not results:
                print(f"no results at {args.results}; skipping figure {key}")
                continue
            written.append(builder(out, results))

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
