"""cadet -- Cloud-Aware Dynamic Earth-observation Tasking.

A reimplementation of

    N. Nordlund, T. Upthegrove, L. Tassiulas,
    "Energy-Aware Dynamic Tasking for Earth Observing Satellites with Deep
    Reinforcement Learning", SSC26-IX-06, 40th Annual Small Satellite Conference.

The package provides:

* :mod:`cadet.clouds` -- latent Gaussian random field cloud model and the
  point-to-block discrepancy variance ``sigma_A`` (Appendix A.1, Eq. 10).
* :mod:`cadet.visibility` -- the closed-form conditional visibility model of
  Proposition 1.
* :mod:`cadet.env` -- the CPOMDP as a Gymnasium environment with joint
  maneuvering and sensing actions.
* :mod:`cadet.planner` -- an exact dynamic-programming SSP solver, used both for
  the baselines and as the ``delegate`` action of CADET-Plan.
* :mod:`cadet.lagrangian` -- primal-dual constrained RL with a learned Lagrange
  multiplier on the power budget.
* :mod:`cadet.train` / :mod:`cadet.evaluate` -- PPO training and the paper's
  evaluation protocol.
"""

from .config import (
    AOR_HEIGHT,
    AOR_WIDTH,
    CONTROLLERS,
    LOOKAHEAD_WIDTHS,
    POWER_BUDGETS,
    CloudConfig,
    EnvConfig,
    PowerConfig,
    SensorConfig,
    TrainConfig,
    make_env_config,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AOR_HEIGHT",
    "AOR_WIDTH",
    "CONTROLLERS",
    "LOOKAHEAD_WIDTHS",
    "POWER_BUDGETS",
    "CloudConfig",
    "EnvConfig",
    "PowerConfig",
    "SensorConfig",
    "TrainConfig",
    "make_env_config",
]


def __getattr__(name: str):
    """Lazily expose the heavier submodules without importing torch on import."""
    if name == "DynamicTaskingEnv":
        from .env import DynamicTaskingEnv

        return DynamicTaskingEnv
    if name == "VisibilityModel":
        from .visibility import VisibilityModel

        return VisibilityModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
