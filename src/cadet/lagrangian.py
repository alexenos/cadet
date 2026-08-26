"""Primal-dual machinery for the constrained objective.

The CPOMDP is solved through the Lagrangian relaxation of Equations (2)-(4):

    L(pi, lambda) = J_r(pi) - lambda * (J_c(pi) - mu * P_bar),   mu = 1 / (1 - gamma)

For a fixed multiplier the policy simply maximises the *augmented* reward
``r(s, a) - lambda * c(s, a)``, which is what
:class:`LagrangianRewardWrapper` hands to PPO.  The multiplier itself is updated
by projected dual ascent on a Monte-Carlo estimate of the discounted constraint
return,

    lambda <- [ lambda + eta * (J_c_hat - mu * P_bar) ]_+

implemented by :class:`LagrangeMultiplierCallback`.

Replacing hand-tuned reward penalties with a learned multiplier is one of the
three departures from prior DRL work claimed in the paper: the agent ends up
*pricing* the energy cost of each action rather than obeying a fixed penalty.

A warm-up curriculum relaxes the budget by a factor ``s > 1`` and then tapers it
back to 1, which stops the policy from converging prematurely to an overly
conservative behaviour before it has learned that lookahead sensing pays off.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

__all__ = [
    "LagrangeState",
    "LagrangianRewardWrapper",
    "LagrangeMultiplierCallback",
    "EpisodeStatsCallback",
]


@dataclass
class LagrangeState:
    """Mutable multiplier shared between the vec-env wrapper and the callback."""

    value: float = 0.0
    #: Current budget slack ``s``; the effective threshold is ``mu * s``.
    slack: float = 1.0
    #: ``mu = 1 / (1 - gamma)``, the effective horizon.
    mu: float = 100.0
    #: Constraint threshold in the *same units* as the accumulated cost.  With
    #: cost normalisation this is ``mu``; otherwise ``mu * P_bar``.
    budget: float = 1.0
    #: Most recent Monte-Carlo estimate of ``J_c``.
    last_jc: float = float("nan")

    @property
    def threshold(self) -> float:
        return self.mu * self.budget * self.slack

    def update(self, jc_hat: float, learning_rate: float) -> float:
        """One projected dual-ascent step; returns the new multiplier."""
        self.last_jc = float(jc_hat)
        self.value = max(0.0, self.value + learning_rate * (jc_hat - self.threshold))
        return self.value


class LagrangianRewardWrapper(VecEnvWrapper):
    """Converts environment rewards into augmented rewards ``r - lambda * c``.

    The wrapper also accumulates the discounted constraint return of each
    episode, ``sum_t gamma^t c_t``, and exposes the completed values through
    :attr:`episode_costs` for the dual update.

    It sits *outside* the per-environment ``Monitor``, so episode statistics
    logged by Stable-Baselines3 (``ep_rew_mean``) remain the true number of
    cloud-free targets captured rather than the penalised objective.
    """

    def __init__(
        self,
        venv: VecEnv,
        state: LagrangeState,
        gamma: float = 0.99,
        history: int = 64,
    ) -> None:
        super().__init__(venv)
        self.state = state
        self.gamma = gamma
        self.episode_costs: deque[float] = deque(maxlen=history)
        self._discounted = np.zeros(venv.num_envs, dtype=np.float64)
        self._undiscounted = np.zeros(venv.num_envs, dtype=np.float64)
        self._discount = np.ones(venv.num_envs, dtype=np.float64)

    def reset(self):
        self._discounted[:] = 0.0
        self._undiscounted[:] = 0.0
        self._discount[:] = 1.0
        return self.venv.reset()

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        costs = np.array([info.get("cost", 0.0) for info in infos], dtype=np.float64)

        self._discounted += self._discount * costs
        self._undiscounted += costs
        self._discount *= self.gamma

        augmented = np.asarray(rewards, dtype=np.float64) - self.state.value * costs

        for index, done in enumerate(dones):
            infos[index]["augmented_reward"] = float(augmented[index])
            if done:
                infos[index]["discounted_cost"] = float(self._discounted[index])
                infos[index]["undiscounted_cost"] = float(self._undiscounted[index])
                self.episode_costs.append(float(self._discounted[index]))
                self._discounted[index] = 0.0
                self._undiscounted[index] = 0.0
                self._discount[index] = 1.0

        return observations, augmented.astype(np.float32), dones, infos

    def drain_costs(self) -> list[float]:
        """Return and clear the completed episode constraint returns."""
        values = list(self.episode_costs)
        self.episode_costs.clear()
        return values


class LagrangeMultiplierCallback(BaseCallback):
    """Dual ascent on ``lambda`` plus the budget-slack curriculum.

    Runs once per PPO rollout, which is the natural batch boundary for the
    Monte-Carlo estimator ``J_c_hat = (1/N) sum_i sum_t gamma^t c_t``.
    """

    def __init__(
        self,
        wrapper: LagrangianRewardWrapper,
        learning_rate: float = 1e-3,
        warmup_steps: int = 1_000_000,
        taper_steps: int = 10_000_000,
        initial_slack: float = 5.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.wrapper = wrapper
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.taper_steps = taper_steps
        self.initial_slack = initial_slack

    def current_slack(self, timesteps: int) -> float:
        """Hold ``initial_slack`` through warm-up, then taper linearly to 1."""
        if timesteps <= self.warmup_steps:
            return self.initial_slack
        if self.taper_steps <= 0:
            return 1.0
        progress = (timesteps - self.warmup_steps) / self.taper_steps
        if progress >= 1.0:
            return 1.0
        return self.initial_slack + progress * (1.0 - self.initial_slack)

    def _on_step(self) -> bool:  # pragma: no cover - required by the interface
        return True

    def _on_rollout_end(self) -> None:
        state = self.wrapper.state
        state.slack = self.current_slack(self.num_timesteps)

        costs = self.wrapper.drain_costs()
        if costs:
            state.update(float(np.mean(costs)), self.learning_rate)

        self.logger.record("constraint/lambda", state.value)
        self.logger.record("constraint/slack", state.slack)
        self.logger.record("constraint/threshold", state.threshold)
        if costs:
            self.logger.record("constraint/jc_hat", state.last_jc)
            self.logger.record(
                "constraint/violation", state.last_jc - state.threshold
            )


class EpisodeStatsCallback(BaseCallback):
    """Logs the environment's own episode metrics to the SB3 logger.

    Surfaces captured targets, capture accuracy and normalised power -- the
    quantities reported in the paper's Metrics section -- alongside the usual
    PPO diagnostics.
    """

    def __init__(self, history: int = 32, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.history = history
        self._buffers: dict[str, deque[float]] = {}

    def _record(self, key: str, value: float) -> None:
        buffer = self._buffers.setdefault(key, deque(maxlen=self.history))
        buffer.append(float(value))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            stats = info.get("episode_stats")
            if stats:
                for key, value in stats.items():
                    self._record(key, value)
            if "discounted_cost" in info:
                self._record("discounted_cost", info["discounted_cost"])
        return True

    def _on_rollout_end(self) -> None:
        for key, buffer in self._buffers.items():
            if buffer:
                self.logger.record(f"episode/{key}", float(np.mean(buffer)))
