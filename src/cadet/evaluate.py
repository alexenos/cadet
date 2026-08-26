"""Evaluation harness and the metrics reported in the paper.

Learned policies act deterministically at evaluation time and every controller
-- CADET, CADET-Plan, SSP and Oracle -- sees the *same* sequence of episodes,
which are disjoint from the training seeds.  Evaluation episodes are 3,000
epochs long (10x training) with the target count scaled to preserve density.

Metrics ("Metrics"):

``captured_targets``
    Cloud-free targets successfully imaged by the payload sensor.
``capture_accuracy``
    Fraction of payload captures that landed on a cloud-free target.
``normalised_power``
    Mean power per decision epoch divided by the budget ``P_bar``.
``behaviour``
    Relative frequencies of lookahead, payload, maneuver, no-op and planner
    invocations.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import gap_closed, run_baseline
from .config import EnvConfig, make_env_config
from .env import DynamicTaskingEnv

__all__ = [
    "EVAL_SEEDS",
    "rollout",
    "evaluate_policy",
    "evaluate_configuration",
    "summarise",
]

#: Evaluation episodes are drawn from a seed range disjoint from training.
EVAL_SEED_OFFSET = 1_000_000


def EVAL_SEEDS(n_episodes: int = 100) -> list[int]:
    """Deterministic evaluation seeds, disjoint from any training seed."""
    return [EVAL_SEED_OFFSET + i for i in range(n_episodes)]


def rollout(
    env: DynamicTaskingEnv,
    model: Any,
    seed: int,
    deterministic: bool = True,
) -> dict[str, float]:
    """Run one full episode of a learned policy and return its statistics."""
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs[None], deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(np.asarray(action).reshape(-1))
        done = terminated or truncated
    stats = env.statistics()
    stats["seed"] = float(seed)
    return stats


def evaluate_policy(
    model: Any,
    env_config: EnvConfig,
    seeds: Sequence[int],
    deterministic: bool = True,
    use_paper_sigma: bool = False,
) -> list[dict[str, float]]:
    """Evaluate a learned policy over a fixed sequence of episodes."""
    env = DynamicTaskingEnv(env_config, use_paper_sigma=use_paper_sigma)
    return [rollout(env, model, int(seed), deterministic) for seed in seeds]


def evaluate_configuration(
    model: Any,
    controller: str,
    lookahead_width: int,
    budget: float,
    n_episodes: int = 20,
    episode_length: int = 3000,
    use_paper_sigma: bool = False,
) -> dict[str, Any]:
    """Score one learned controller against both baselines on shared episodes."""
    env_config = make_env_config(
        lookahead_width=lookahead_width,
        budget=budget,
        controller=controller,
        episode_length=episode_length,
    )
    seeds = EVAL_SEEDS(n_episodes)

    policy_records = evaluate_policy(
        model, env_config, seeds, use_paper_sigma=use_paper_sigma
    )

    baseline_env = DynamicTaskingEnv(env_config, use_paper_sigma=use_paper_sigma)
    ssp_records, oracle_records = [], []
    for seed in seeds:
        baseline_env.reset(seed=int(seed))
        ssp_records.append(run_baseline(baseline_env, "ssp").to_dict())
        oracle_records.append(run_baseline(baseline_env, "oracle").to_dict())

    policy_mean = _mean_of("captured_targets", policy_records)
    ssp_mean = _mean_of("captured_targets", ssp_records)
    oracle_mean = _mean_of("captured_targets", oracle_records)

    return {
        "controller": controller,
        "lookahead_width": lookahead_width,
        "budget": budget,
        "n_episodes": n_episodes,
        "episode_length": episode_length,
        "captured_targets": policy_mean,
        "capture_accuracy": _mean_of("capture_accuracy", policy_records),
        "normalised_power": _mean_of("normalised_power", policy_records),
        "mean_power": _mean_of("mean_power", policy_records),
        "n_lookahead": _mean_of("n_lookahead", policy_records),
        "n_payload": _mean_of("n_payload", policy_records),
        "n_sense_noop": _mean_of("n_sense_noop", policy_records),
        "n_roll": _mean_of("n_roll", policy_records),
        "n_delegate": _mean_of("n_delegate", policy_records),
        "n_move_noop": _mean_of("n_move_noop", policy_records),
        "ssp_captured": ssp_mean,
        "oracle_captured": oracle_mean,
        "gap_closed_pct": gap_closed(policy_mean, ssp_mean, oracle_mean),
        "episodes": policy_records,
    }


def _mean_of(key: str, records: Iterable[dict[str, float]]) -> float:
    values = [record[key] for record in records if key in record]
    return float(np.mean(values)) if values else float("nan")


def summarise(results: Sequence[dict[str, Any]]) -> str:
    """Render a Table 2-style summary of a completed sweep."""
    lines = [
        f"{'controller':<12} {'n':>4} {'P_bar':>7} {'targets':>9} "
        f"{'gap %':>7} {'accuracy':>9} {'power/P':>8}",
        "-" * 62,
    ]
    for row in results:
        lines.append(
            f"{row['controller']:<12} {row['lookahead_width']:>4} "
            f"{row['budget']:>7.0f} {row['captured_targets']:>9.1f} "
            f"{row['gap_closed_pct']:>7.1f} {row['capture_accuracy']:>9.3f} "
            f"{row['normalised_power']:>8.2f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to a saved Stable-Baselines3 model")
    parser.add_argument("--controller", default="cadet", choices=["cadet", "cadet-plan"])
    parser.add_argument("--lookahead", type=int, default=32, choices=[8, 16, 32, 64])
    parser.add_argument("--budget", type=float, default=150.0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=3000)
    parser.add_argument("--paper-sigma", action="store_true")
    parser.add_argument("--output", default=None, help="write results as JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    from stable_baselines3 import PPO

    args = build_parser().parse_args(argv)
    model = PPO.load(args.model, device="cpu")
    result = evaluate_configuration(
        model,
        controller=args.controller,
        lookahead_width=args.lookahead,
        budget=args.budget,
        n_episodes=args.episodes,
        episode_length=args.episode_length,
        use_paper_sigma=args.paper_sigma,
    )
    print(summarise([result]))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
