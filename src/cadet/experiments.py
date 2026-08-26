"""Sweep runner reproducing the paper's 24-cell experimental grid.

Three factors are crossed ("Experimental Design"):

* lookahead footprint width ``n`` in ``{8, 16, 32, 64}``,
* average power budget ``P_bar`` in ``{100, 150, 1500}``,
* controller in ``{cadet, cadet-plan}``.

The full sweep at the paper's 30M timesteps per cell is a large compute job;
``--profile quick`` and ``--profile smoke`` run the same code at reduced scale
for development and CI.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CONTROLLERS,
    LOOKAHEAD_WIDTHS,
    POWER_BUDGETS,
    TrainConfig,
)
from .evaluate import evaluate_configuration, summarise
from .train import run_name, train

__all__ = ["PROFILES", "run_sweep", "load_results", "table_two", "table_three"]

#: Named training-scale presets.  ``paper`` is Table 5; the others keep every
#: hyperparameter but shrink the horizon (and the curriculum proportionally).
PROFILES: dict[str, dict[str, int]] = {
    "paper": {
        "total_timesteps": 30_000_000,
        "warmup_steps": 1_000_000,
        "taper_steps": 10_000_000,
    },
    "quick": {
        "total_timesteps": 2_000_000,
        "warmup_steps": 66_000,
        "taper_steps": 660_000,
    },
    "smoke": {
        "total_timesteps": 20_000,
        "warmup_steps": 2_000,
        "taper_steps": 6_000,
    },
}


def _train_config(profile: str, base: TrainConfig | None = None, **overrides: Any):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected {sorted(PROFILES)}.")
    base = base or TrainConfig()
    return replace(base, **PROFILES[profile], **overrides)


def run_sweep(
    controllers: Sequence[str] = CONTROLLERS,
    lookahead_widths: Sequence[int] = LOOKAHEAD_WIDTHS,
    budgets: Sequence[float] = POWER_BUDGETS,
    profile: str = "quick",
    output_dir: str | Path = "runs",
    results_path: str | Path = "results/sweep.json",
    n_episodes: int = 20,
    episode_length: int = 3000,
    n_envs: int | None = None,
    device: str = "auto",
    seed: int = 0,
    use_subproc: bool = False,
    use_paper_sigma: bool = False,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Train and evaluate every requested cell, appending results to JSON."""
    overrides: dict[str, Any] = {"seed": seed, "device": device}
    if n_envs is not None:
        overrides["n_envs"] = n_envs
    train_config = _train_config(profile, **overrides)

    output_dir = Path(output_dir)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = load_results(results_path)
    done = {
        (r["controller"], r["lookahead_width"], r["budget"]) for r in results
    }

    cells = list(product(controllers, budgets, lookahead_widths))
    for index, (controller, budget, width) in enumerate(cells, start=1):
        key = (controller, width, float(budget))
        if skip_existing and key in done:
            print(f"[{index}/{len(cells)}] {key} already recorded; skipping.")
            continue

        name = run_name(controller, width, budget)
        print(f"[{index}/{len(cells)}] training {name} (profile={profile})")
        started = time.time()
        model = train(
            controller=controller,
            lookahead_width=width,
            budget=float(budget),
            train_config=train_config,
            output_dir=output_dir,
            use_subproc=use_subproc,
            use_paper_sigma=use_paper_sigma,
            progress_bar=False,
        )
        elapsed = time.time() - started

        record = evaluate_configuration(
            model,
            controller=controller,
            lookahead_width=width,
            budget=float(budget),
            n_episodes=n_episodes,
            episode_length=episode_length,
            use_paper_sigma=use_paper_sigma,
        )
        record["profile"] = profile
        record["train_seconds"] = elapsed
        record.pop("episodes", None)
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2))
        print(summarise([record]))

    return results


def load_results(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _lookup(results, controller, width, budget, key):
    for row in results:
        if (
            row["controller"] == controller
            and row["lookahead_width"] == width
            and float(row["budget"]) == float(budget)
        ):
            return row.get(key, float("nan"))
    return float("nan")


def table_two(results: Sequence[dict[str, Any]]) -> str:
    """Table 2-style performance summary: targets, gap closure and delta."""
    widths = sorted({r["lookahead_width"] for r in results})
    budgets = sorted({float(r["budget"]) for r in results})
    lines = ["Performance Summary (CADET / CADET-Plan)", ""]
    header = f"{'P_bar':>7}  {'metric':<8}" + "".join(f"{w:>18}" for w in widths)
    lines.append(header)
    lines.append("-" * len(header))
    for budget in budgets:
        for metric, key in (("Targets", "captured_targets"), ("Gap %", "gap_closed_pct")):
            cells = []
            for width in widths:
                base = _lookup(results, "cadet", width, budget, key)
                plan = _lookup(results, "cadet-plan", width, budget, key)
                cells.append(f"{base:>8.1f}/{plan:<9.1f}")
            lines.append(f"{budget:>7.0f}  {metric:<8}" + "".join(cells))
        deltas = []
        for width in widths:
            base = _lookup(results, "cadet", width, budget, "captured_targets")
            plan = _lookup(results, "cadet-plan", width, budget, "captured_targets")
            deltas.append(f"{plan - base:>+18.1f}")
        lines.append(f"{'':>7}  {'Delta':<8}" + "".join(deltas))
        lines.append("")
    return "\n".join(lines)


def table_three(results: Sequence[dict[str, Any]]) -> str:
    """Table 3-style capture accuracy grid."""
    widths = sorted({r["lookahead_width"] for r in results})
    budgets = sorted({float(r["budget"]) for r in results})
    lines = ["Capture Accuracy", ""]
    header = f"{'controller':<12}{'P_bar':>7}" + "".join(f"{'n=' + str(w):>9}" for w in widths)
    lines.append(header)
    lines.append("-" * len(header))
    for controller in CONTROLLERS:
        for budget in budgets:
            cells = "".join(
                f"{_lookup(results, controller, w, budget, 'capture_accuracy'):>9.3f}"
                for w in widths
            )
            lines.append(f"{controller:<12}{budget:>7.0f}{cells}")
    return "\n".join(lines)


def table_energy(results: Sequence[dict[str, Any]]) -> str:
    """Table 1-style normalised energy usage for the constrained budgets."""
    widths = sorted({r["lookahead_width"] for r in results})
    budgets = [b for b in sorted({float(r["budget"]) for r in results}) if b < 1000]
    lines = ["Normalised Energy Usage (E_bar / P_bar)", ""]
    header = f"{'controller':<12}{'P_bar':>7}" + "".join(f"{'n=' + str(w):>9}" for w in widths)
    lines.append(header)
    lines.append("-" * len(header))
    for controller in CONTROLLERS:
        for budget in budgets:
            cells = "".join(
                f"{_lookup(results, controller, w, budget, 'normalised_power'):>9.2f}"
                for w in widths
            )
            lines.append(f"{controller:<12}{budget:>7.0f}{cells}")
    return "\n".join(lines)


def mean_gap_closed(results: Sequence[dict[str, Any]], controller: str) -> float:
    """Average gap closure, with sub-baseline cells truncated to 0 as in Table 2."""
    values = [
        max(0.0, r["gap_closed_pct"])
        for r in results
        if r["controller"] == controller and np.isfinite(r["gap_closed_pct"])
    ]
    return float(np.mean(values)) if values else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    parser.add_argument(
        "--lookahead", nargs="+", type=int, default=list(LOOKAHEAD_WIDTHS)
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=list(POWER_BUDGETS))
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--results", default="results/sweep.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=3000)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--paper-sigma", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="ignore existing results")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = run_sweep(
        controllers=args.controllers,
        lookahead_widths=args.lookahead,
        budgets=args.budgets,
        profile=args.profile,
        output_dir=args.output_dir,
        results_path=args.results,
        n_episodes=args.episodes,
        episode_length=args.episode_length,
        n_envs=args.n_envs,
        device=args.device,
        seed=args.seed,
        use_subproc=args.subproc,
        use_paper_sigma=args.paper_sigma,
        skip_existing=not args.rerun,
    )
    print()
    print(table_two(results))
    print(table_three(results))
    print()
    print(table_energy(results))
    print()
    for controller in CONTROLLERS:
        print(f"mean gap closed, {controller}: {mean_gap_closed(results, controller):.1f}%")


if __name__ == "__main__":  # pragma: no cover
    main()
