"""Offline planning baselines that bracket achievable performance.

Both baselines solve the power-unconstrained, agility-feasible SSP exactly with
the dynamic program in :mod:`cadet.planner`, take no lookahead observations, and
fire the payload sensor at every decision epoch.  They differ only in the task
utilities they are given:

``SSP``
    The optimistic preplanned schedule: every target is assumed cloud free.  The
    resulting trajectory is scored against the realised cloud field, so captures
    of obscured targets earn nothing.  This is the lower bound.

``Oracle``
    Perfect pointwise cloud knowledge: only genuinely cloud-free targets carry
    utility.  Not realisable in practice, but an upper bound for an
    agility-feasible planner with no power constraint.

Reported performance is the fraction of the gap between the two that a learned
controller closes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import AOR_WIDTH, EnvConfig
from .env import DynamicTaskingEnv
from .planner import solve_roll_trajectory

__all__ = ["BaselineResult", "run_baseline", "evaluate_baselines", "gap_closed"]

BaselineName = Literal["ssp", "oracle"]


@dataclass
class BaselineResult:
    """Per-episode outcome of one offline baseline."""

    name: str
    captured_targets: int
    targets_imaged: int
    capture_accuracy: float
    mean_power: float
    normalised_power: float
    n_roll: int
    columns: np.ndarray

    def to_dict(self) -> dict[str, float]:
        return {
            "controller": self.name,
            "captured_targets": float(self.captured_targets),
            "targets_imaged": float(self.targets_imaged),
            "capture_accuracy": float(self.capture_accuracy),
            "mean_power": float(self.mean_power),
            "normalised_power": float(self.normalised_power),
            "n_roll": float(self.n_roll),
        }


def run_baseline(
    env: DynamicTaskingEnv,
    name: BaselineName,
    start_col: int | None = AOR_WIDTH // 2,
) -> BaselineResult:
    """Plan and score one baseline against the cloud field already sampled.

    The environment must have been ``reset`` -- the baseline reads its targets
    and realised visibilities directly rather than stepping it, since the plan is
    computed offline in one shot.
    """
    horizon = env.cfg.episode_length
    if name == "ssp":
        utility = env.nominal_utility()[:horizon]
    elif name == "oracle":
        utility = env.oracle_utility()[:horizon]
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"Unknown baseline {name!r}; expected 'ssp' or 'oracle'.")

    columns, _ = solve_roll_trajectory(
        utility,
        start_col=start_col,
        footprint_width=env.cfg.sensors.payload_width,
    )

    half = env.cfg.sensors.payload_width // 2
    captured_clear = 0
    imaged = 0
    for t, col in enumerate(columns):
        lo = max(0, int(col) - half)
        hi = min(AOR_WIDTH, int(col) + half + 1)
        ids = env.target_index[t, lo:hi]
        ids = ids[ids >= 0]
        if ids.size:
            imaged += int(ids.size)
            captured_clear += int(np.count_nonzero(env.target_visible[ids]))

    n_roll = int(np.count_nonzero(np.diff(columns)))
    power = env.cfg.power
    total_power = horizon * power.payload + n_roll * power.roll
    mean_power = total_power / max(horizon, 1)

    return BaselineResult(
        name=name,
        captured_targets=captured_clear,
        targets_imaged=imaged,
        capture_accuracy=captured_clear / imaged if imaged else 0.0,
        mean_power=mean_power,
        normalised_power=mean_power / power.budget,
        n_roll=n_roll,
        columns=columns,
    )


def evaluate_baselines(
    config: EnvConfig,
    seeds: Iterable[int],
    names: Iterable[BaselineName] = ("ssp", "oracle"),
    use_paper_sigma: bool = False,
) -> list[dict[str, float]]:
    """Run every baseline over a shared sequence of evaluation episodes."""
    env = DynamicTaskingEnv(config, use_paper_sigma=use_paper_sigma)
    records: list[dict[str, float]] = []
    for seed in seeds:
        env.reset(seed=int(seed))
        for name in names:
            record = run_baseline(env, name).to_dict()
            record["seed"] = int(seed)
            records.append(record)
    return records


def gap_closed(value: float, lower: float, upper: float) -> float:
    """Fraction of the SSP-to-Oracle gap closed by ``value``, as a percentage.

    Returns ``nan`` when the bounds coincide.  Values are not clipped, so a
    controller below the SSP baseline reports a negative gap closure -- the
    paper truncates these to 0 in Table 2.
    """
    span = upper - lower
    if abs(span) < 1e-12:
        return float("nan")
    return 100.0 * (value - lower) / span
