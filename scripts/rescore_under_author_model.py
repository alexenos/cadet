"""Re-score this repo's models under the paper author's cloud-field convention.

The author's implementation models the cloud field at the scale of the *lookahead*
sensor and sets ``is_cloud_free = is_observed_cloud_free``; this repo models it at
sub-pixel scale, reads ground truth pointwise, and block-averages to produce the
observation.  See ``docs/shortfall-resolved.md`` for what that difference costs.

Rather than edit the environment, this script swaps the field sampler for one that
draws the latent GRF on the lookahead-pixel grid and holds it constant within a
pixel.  Then ``Y(p) = Y_A`` identically, the observed-cloud-free test *is* the
ground truth, and the marginal cloud-free fraction stays ``Phi(-beta/alpha)``.
That makes it a measurement of the author's convention against the code as it
stands, with nothing else changed.

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

import cadet.env as env_mod  # noqa: E402
from cadet.baselines import gap_closed, run_baseline  # noqa: E402
from cadet.clouds import CloudField, sample_latent_field  # noqa: E402
from cadet.config import make_env_config  # noqa: E402
from cadet.env import DynamicTaskingEnv  # noqa: E402
from cadet.evaluate import EVAL_SEEDS  # noqa: E402

#: Sub-cells spanned by one lookahead pixel, at ``subpixels_per_cell = 4``.
BLOCK_SUB = {8: 1, 16: 2, 32: 4, 64: 8}

#: The two cells trained at paper scale, with their published counterparts.
CHECKPOINTS = (
    ("runs/cadet_n32_P150_paper_DONE/model.zip", 150.0, 255.5, 56.1, 0.714),
    ("runs/cadet_n32_P1500/model.zip", 1500.0, 258.8, 60.1, 0.202),
)

_block = {"sub": 4}


def pixel_scale_field(world_shape, cfg, rng) -> CloudField:
    """Draw the latent field on the lookahead-pixel grid, then replicate it.

    Sampling on a grid of spacing ``b * d`` is equivalent to sampling on a grid of
    spacing ``d`` with the length scale divided by ``b``, so this reuses the
    existing sampler (and its cached spectrum) rather than duplicating it.
    """
    b = _block["sub"]
    s = cfg.subpixels_per_cell
    coarse = dataclasses.replace(cfg, length_scale_km=cfg.length_scale_km / b)
    shape = (world_shape[0] * s // b, world_shape[1] * s // b)
    latent = sample_latent_field(shape, coarse, rng)
    values = 1.0 / (1.0 + np.exp(-(cfg.alpha * latent + cfg.beta), dtype=np.float32))
    values = np.repeat(np.repeat(values, b, axis=0), b, axis=1)
    return CloudField(values=values.astype(np.float32), cfg=cfg)


def baseline_table(n_episodes: int, episode_length: int) -> None:
    """SSP and Oracle at every lookahead width, under the author's field model."""
    print("\nBaselines by lookahead width")
    print(f"{'n':>4} {'pixel km':>9} {'clear frac':>11} {'SSP':>8} {'Oracle':>8}")
    for width, block in BLOCK_SUB.items():
        _block["sub"] = block
        cfg = make_env_config(width, 150.0, "cadet", episode_length=episode_length)
        env = DynamicTaskingEnv(cfg)
        ssp, oracle, frac = [], [], []
        for seed in EVAL_SEEDS(n_episodes):
            env.reset(seed=int(seed))
            ssp.append(run_baseline(env, "ssp").captured_targets)
            oracle.append(run_baseline(env, "oracle").captured_targets)
            frac.append(float(np.mean(env.cloud.values < cfg.clouds.tau)))
        side = block * cfg.clouds.subpixel_km
        print(
            f"{width:>4} {side:>9.2f} {np.mean(frac):>11.3f} "
            f"{np.mean(ssp):>8.1f} {np.mean(oracle):>8.1f}"
        )


def policy_table(n_episodes: int, episode_length: int) -> None:
    """Re-score the trained checkpoints, if their run directories are present."""
    from stable_baselines3 import PPO

    _block["sub"] = BLOCK_SUB[32]
    print("\nTrained policies, n = 32 (paper values in parentheses)")
    for rel, budget, paper_targets, paper_gap, paper_acc in CHECKPOINTS:
        path = ROOT / rel
        if not path.exists():
            print(f"  P={budget:.0f}: {rel} not present -- skipped")
            continue

        cfg = make_env_config(32, budget, "cadet", episode_length=episode_length)
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
            # Under this model the two must agree exactly, by construction.
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    env_mod.sample_cloud_field = pixel_scale_field
    baseline_table(args.episodes, args.episode_length)
    if not args.baselines_only:
        policy_table(args.episodes, args.episode_length)


if __name__ == "__main__":  # pragma: no cover
    main()
