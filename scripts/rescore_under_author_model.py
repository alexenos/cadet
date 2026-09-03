"""Re-score this repo's models under the paper author's cloud-field convention.

The author's implementation models the cloud field at the scale of the *lookahead*
sensor and sets ``is_cloud_free = is_observed_cloud_free``.  That is now the
environment's default (``CloudConfig.field_scale = "lookahead"``), so this script
just evaluates against it: it regenerates the tables in section 2 of
``docs/shortfall-resolved.md``, which were first measured before the environment
implemented the convention.

The checkpoints under ``runs/`` were trained against the sub-pixel field, so their
rows here are a *transfer* measurement rather than a reproduction.  Pass
``--field-scale subpixel`` to score them under the model they were trained on.

Two tables are produced:

``baselines``
    SSP and Oracle across all four lookahead widths.  Needs no trained policy.
``policies``
    The 30M-timestep checkpoints re-scored end to end.  Skipped when the run
    directories are absent, since they are untracked and regenerable.

Usage:
    python scripts/rescore_under_author_model.py [--episodes 20]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cadet.baselines import gap_closed, run_baseline  # noqa: E402
from cadet.config import LOOKAHEAD_WIDTHS, make_env_config  # noqa: E402
from cadet.env import DynamicTaskingEnv  # noqa: E402
from cadet.evaluate import EVAL_SEEDS  # noqa: E402

#: The two cells trained at paper scale, with their published counterparts.
CHECKPOINTS = (
    ("runs/cadet_n32_P150_paper_DONE/model.zip", 150.0, 255.5, 56.1, 0.714),
    ("runs/cadet_n32_P1500/model.zip", 1500.0, 258.8, 60.1, 0.202),
)


def _with_scale(cfg, field_scale: str):
    """Copy of an env config with the cloud field drawn at ``field_scale``."""
    return cfg.replace(
        clouds=dataclasses.replace(
            cfg.clouds,
            field_scale=field_scale,
            # A sub-pixel field realises the discrepancy itself; drawing it
            # again would double-count.
            truth_noise=(field_scale == "lookahead"),
        )
    )


def baseline_table(n_episodes: int, episode_length: int, field_scale: str) -> None:
    """SSP and Oracle at every lookahead width."""
    print("\nBaselines by lookahead width")
    print(f"{'n':>4} {'pixel km':>9} {'clear frac':>11} {'SSP':>8} {'Oracle':>8}")
    for width in LOOKAHEAD_WIDTHS:
        cfg = _with_scale(
            make_env_config(width, 150.0, "cadet", episode_length=episode_length),
            field_scale,
        )
        env = DynamicTaskingEnv(cfg)
        ssp, oracle, frac = [], [], []
        for seed in EVAL_SEEDS(n_episodes):
            env.reset(seed=int(seed))
            ssp.append(run_baseline(env, "ssp").captured_targets)
            oracle.append(run_baseline(env, "oracle").captured_targets)
            frac.append(float(np.mean(env.cloud.values < cfg.clouds.tau)))
        side = env._block_sub * cfg.clouds.subpixel_km
        print(
            f"{width:>4} {side:>9.2f} {np.mean(frac):>11.3f} "
            f"{np.mean(ssp):>8.1f} {np.mean(oracle):>8.1f}"
        )


def policy_table(n_episodes: int, episode_length: int, field_scale: str) -> None:
    """Re-score the trained checkpoints, if their run directories are present."""
    from stable_baselines3 import PPO

    print("\nTrained policies, n = 32 (paper values in parentheses)")
    for rel, budget, paper_targets, paper_gap, paper_acc in CHECKPOINTS:
        path = ROOT / rel
        if not path.exists():
            print(f"  P={budget:.0f}: {rel} not present -- skipped")
            continue

        cfg = _with_scale(
            make_env_config(32, budget, "cadet", episode_length=episode_length),
            field_scale,
        )
        model = PPO.load(str(path), device="cpu")
        env = DynamicTaskingEnv(cfg)
        captured, actions, ssp, oracle, agree = [], [], [], [], []
        for seed in EVAL_SEEDS(n_episodes):
            obs, _ = env.reset(seed=int(seed))
            done = False
            while not done:
                action, _ = model.predict(obs[None], deterministic=True)
                obs, _, terminated, truncated, _ = env.step(np.asarray(action).reshape(-1))
                done = terminated or truncated
            captured.append(env.n_targets_captured_clear)
            actions.append(env.n_capture_attempts)
            # Under field_scale="lookahead" these agree exactly, by construction.
            agree.append(float(np.mean(env.target_visible == (env.target_pvis_if_observed > 0.5))))
            ssp.append(run_baseline(env, "ssp").captured_targets)
            oracle.append(run_baseline(env, "oracle").captured_targets)

        n = len(captured)
        mean = float(np.mean(captured))
        stderr = float(np.std(captured, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        gap = gap_closed(mean, float(np.mean(ssp)), float(np.mean(oracle)))
        accuracy = mean / float(np.mean(actions))
        print(
            f"  P={budget:>6.0f}  captured {mean:6.1f} +/- {stderr:.1f} "
            f"({paper_targets})   gap {gap:5.1f}% ({paper_gap}%)   "
            f"accuracy {accuracy:.3f} ({paper_acc})   "
            f"truth==observed {np.mean(agree):.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=3000)
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="skip the checkpoints; runs in well under a minute",
    )
    parser.add_argument(
        "--field-scale",
        default="lookahead",
        choices=["lookahead", "subpixel"],
        help="cloud field resolution; 'lookahead' is the paper's convention",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    baseline_table(args.episodes, args.episode_length, args.field_scale)
    if not args.baselines_only:
        policy_table(args.episodes, args.episode_length, args.field_scale)


if __name__ == "__main__":  # pragma: no cover
    main()
