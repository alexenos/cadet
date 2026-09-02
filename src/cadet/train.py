"""Train CADET / CADET-Plan with constrained PPO.

Usage::

    python -m cadet.train --controller cadet-plan --lookahead 32 --budget 150

The policy is optimised with PPO on the augmented reward ``r - lambda * c``
while ``lambda`` is driven by projected dual ascent on the discounted power
constraint (see :mod:`cadet.lagrangian`).  Hyperparameters default to Table 5.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .config import EnvConfig, TrainConfig, make_env_config
from .env import DynamicTaskingEnv
from .lagrangian import (
    EpisodeStatsCallback,
    LagrangeMultiplierCallback,
    LagrangeState,
    LagrangianRewardWrapper,
)
from .policies import count_parameters, policy_kwargs

__all__ = ["build_vec_env", "train", "run_name"]


def run_name(controller: str, lookahead_width: int, budget: float) -> str:
    """Canonical identifier for one cell of the experimental grid."""
    return f"{controller}_n{lookahead_width}_P{int(budget)}"


def _has_tensorboard() -> bool:
    """Whether the optional tensorboard package is importable."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


def _tensorboard_dir(path: Path) -> str | None:
    """Tensorboard log directory, or ``None`` when tensorboard is unavailable.

    Stable-Baselines3 raises if ``tensorboard_log`` is set without the package
    installed; training should not be gated on an optional logging dependency.
    """
    try:  # pragma: no cover - depends on the installed environment
        import tensorboard  # noqa: F401
    except ImportError:
        print("tensorboard not installed; skipping tensorboard logging.")
        return None
    return str(path)


def build_vec_env(
    env_config: EnvConfig,
    n_envs: int,
    seed: int = 0,
    use_subproc: bool = False,
    use_paper_sigma: bool = False,
):
    """Vectorised, Monitor-wrapped environments for training."""

    def factory() -> Callable[[], DynamicTaskingEnv]:
        return DynamicTaskingEnv(env_config, use_paper_sigma=use_paper_sigma)

    vec_cls = SubprocVecEnv if use_subproc and n_envs > 1 else DummyVecEnv
    return make_vec_env(
        factory,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_cls,
    )


def train(
    controller: str = "cadet",
    lookahead_width: int = 32,
    budget: float = 150.0,
    train_config: TrainConfig | None = None,
    output_dir: str | Path = "runs",
    use_subproc: bool = False,
    use_paper_sigma: bool = False,
    checkpoint_freq: int = 0,
    progress_bar: bool = True,
    truth_noise: bool = False,
) -> PPO:
    """Train one controller configuration and save the model and metadata."""
    train_config = train_config or TrainConfig()
    env_config = make_env_config(
        lookahead_width=lookahead_width,
        budget=budget,
        controller=controller,
        episode_length=300,
        truth_noise=truth_noise,
    )

    name = run_name(controller, lookahead_width, budget)
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    venv = build_vec_env(
        env_config,
        n_envs=train_config.n_envs,
        seed=train_config.seed,
        use_subproc=use_subproc,
        use_paper_sigma=use_paper_sigma,
    )

    # mu = 1 / (1 - gamma); with cost normalisation the threshold is mu * slack.
    mu = 1.0 / (1.0 - train_config.gamma)
    state = LagrangeState(
        value=train_config.lambda_init,
        slack=train_config.initial_slack,
        mu=mu,
        budget=1.0 if env_config.power.normalise_by_budget else budget,
    )
    venv = LagrangianRewardWrapper(venv, state, gamma=train_config.gamma)

    model = PPO(
        "CnnPolicy",
        venv,
        learning_rate=train_config.learning_rate,
        n_steps=train_config.n_steps,
        batch_size=train_config.batch_size,
        n_epochs=train_config.n_epochs,
        gamma=train_config.gamma,
        gae_lambda=train_config.gae_lambda,
        clip_range=train_config.clip_range,
        ent_coef=train_config.ent_coef,
        vf_coef=train_config.vf_coef,
        max_grad_norm=train_config.max_grad_norm,
        policy_kwargs=policy_kwargs(train_config.features_dim),
        tensorboard_log=_tensorboard_dir(out / "tb"),
        seed=train_config.seed,
        device=train_config.device,
        verbose=1,
    )

    callbacks = [
        LagrangeMultiplierCallback(
            venv,
            learning_rate=train_config.lambda_lr,
            warmup_steps=train_config.warmup_steps,
            taper_steps=train_config.taper_steps,
            initial_slack=train_config.initial_slack,
        ),
        EpisodeStatsCallback(),
    ]
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(checkpoint_freq // train_config.n_envs, 1),
                save_path=str(out / "checkpoints"),
                name_prefix="ppo",
            )
        )

    # SB3's stdout tables are the only record of the run unless we ask for a
    # durable one. Tensorboard is optional (and not installed by default), so
    # always write progress.csv -- scripts/plot_training.py reads it.
    model.set_logger(
        configure(str(out), ["stdout", "csv"] + (["tensorboard"] if _has_tensorboard() else []))
    )

    metadata = {
        "controller": controller,
        "lookahead_width": lookahead_width,
        "budget": budget,
        "policy_parameters": count_parameters(model.policy),
        "train_config": train_config.to_dict(),
        # Recorded so a run report can say which cloud model produced its
        # numbers; runs predating 2026-09-01 have no such key and were trained
        # against the sub-pixel field.  See docs/shortfall-resolved.md.
        "field_scale": env_config.clouds.field_scale,
        "truth_noise": env_config.clouds.truth_noise,
        "sigma_a": DynamicTaskingEnv(
            env_config, use_paper_sigma=use_paper_sigma
        ).visibility.sigma_a,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[{name}] policy parameters: {metadata['policy_parameters']:,}")
    print(f"[{name}] sigma_A: {metadata['sigma_a']:.3f}")

    model.learn(
        total_timesteps=train_config.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=progress_bar,
    )
    model.save(str(out / "model"))
    (out / "lambda.json").write_text(
        json.dumps(
            {
                "lambda": state.value,
                "slack": state.slack,
                "threshold": state.threshold,
                "jc_hat": state.last_jc,
            },
            indent=2,
        )
    )
    venv.close()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="cadet", choices=["cadet", "cadet-plan"])
    parser.add_argument("--lookahead", type=int, default=32, choices=[8, 16, 32, 64])
    parser.add_argument("--budget", type=float, default=150.0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subproc", action="store_true", help="use SubprocVecEnv")
    parser.add_argument(
        "--paper-sigma",
        action="store_true",
        help="pin sigma_A to the values printed in Figure 3",
    )
    parser.add_argument(
        "--truth-noise",
        action="store_true",
        help="pay the reward on a Prop-1 draw; metrics stay deterministic",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {"seed": args.seed, "device": args.device}
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
        # Keep the curriculum proportional for short debugging runs.
        default = TrainConfig()
        scale = args.timesteps / default.total_timesteps
        overrides["warmup_steps"] = max(int(default.warmup_steps * scale), 1)
        overrides["taper_steps"] = max(int(default.taper_steps * scale), 1)
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs

    train(
        controller=args.controller,
        lookahead_width=args.lookahead,
        budget=args.budget,
        train_config=TrainConfig(**{**TrainConfig().to_dict(), **overrides}),
        output_dir=args.output_dir,
        use_subproc=args.subproc,
        use_paper_sigma=args.paper_sigma,
        checkpoint_freq=args.checkpoint_freq,
        progress_bar=not args.no_progress,
        truth_noise=args.truth_noise,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
