"""Tests for the primal-dual constrained-RL machinery."""

import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env

from cadet.config import make_env_config
from cadet.env import MOVE_NOOP, SENSE_NOOP, SENSE_PAYLOAD, DynamicTaskingEnv
from cadet.lagrangian import (
    LagrangeMultiplierCallback,
    LagrangeState,
    LagrangianRewardWrapper,
)


def make_wrapped(n_envs=2, episode_length=20, budget=150.0, gamma=0.99, lam=0.0):
    config = make_env_config(32, budget, "cadet", episode_length=episode_length)
    venv = make_vec_env(lambda: DynamicTaskingEnv(config), n_envs=n_envs, seed=0)
    state = LagrangeState(value=lam, slack=1.0, mu=1.0 / (1.0 - gamma), budget=1.0)
    return LagrangianRewardWrapper(venv, state, gamma=gamma), state


# ---------------------------------------------------------------------------
# Dual variable
# ---------------------------------------------------------------------------
def test_multiplier_rises_when_the_constraint_is_violated():
    state = LagrangeState(value=0.0, slack=1.0, mu=100.0, budget=1.0)
    state.update(jc_hat=150.0, learning_rate=1e-3)
    assert state.value == pytest.approx(0.05)


def test_multiplier_falls_when_the_policy_is_frugal():
    state = LagrangeState(value=1.0, slack=1.0, mu=100.0, budget=1.0)
    state.update(jc_hat=50.0, learning_rate=1e-3)
    assert state.value == pytest.approx(0.95)


def test_multiplier_is_projected_onto_the_non_negative_orthant():
    state = LagrangeState(value=0.01, slack=1.0, mu=100.0, budget=1.0)
    state.update(jc_hat=0.0, learning_rate=1.0)
    assert state.value == 0.0


def test_threshold_scales_with_slack_and_budget():
    state = LagrangeState(mu=100.0, budget=1.0, slack=5.0)
    assert state.threshold == pytest.approx(500.0)
    state.slack = 1.0
    assert state.threshold == pytest.approx(100.0)
    state.budget = 150.0
    assert state.threshold == pytest.approx(15000.0)


def test_dual_ascent_drives_a_constant_cost_policy_to_its_fixed_point():
    """With a fixed J_c the multiplier converges to a stationary point at
    the threshold, or to zero when the constraint is slack."""
    state = LagrangeState(value=0.0, slack=1.0, mu=100.0, budget=1.0)
    for _ in range(500):
        state.update(jc_hat=100.0, learning_rate=1e-2)
    assert state.value == pytest.approx(0.0, abs=1e-9)

    state = LagrangeState(value=5.0, slack=1.0, mu=100.0, budget=1.0)
    for _ in range(2000):
        state.update(jc_hat=90.0, learning_rate=1e-2)
    assert state.value == 0.0


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
def test_slack_curriculum_holds_then_tapers():
    wrapper, _ = make_wrapped()
    callback = LagrangeMultiplierCallback(
        wrapper, warmup_steps=1000, taper_steps=1000, initial_slack=5.0
    )
    assert callback.current_slack(0) == 5.0
    assert callback.current_slack(1000) == 5.0
    assert callback.current_slack(1500) == pytest.approx(3.0)
    assert callback.current_slack(2000) == pytest.approx(1.0)
    assert callback.current_slack(10_000) == 1.0
    wrapper.close()


def test_slack_is_monotonically_non_increasing():
    wrapper, _ = make_wrapped()
    callback = LagrangeMultiplierCallback(
        wrapper, warmup_steps=100, taper_steps=900, initial_slack=5.0
    )
    values = [callback.current_slack(t) for t in range(0, 2000, 25)]
    # Deliberately ragged: this zips each value against its successor.
    pairs = zip(values, values[1:], strict=False)
    assert all(b <= a + 1e-12 for a, b in pairs)
    wrapper.close()


# ---------------------------------------------------------------------------
# Reward augmentation
# ---------------------------------------------------------------------------
def test_zero_multiplier_leaves_rewards_untouched():
    wrapper, _ = make_wrapped(lam=0.0)
    wrapper.reset()
    actions = np.array([[MOVE_NOOP, SENSE_PAYLOAD]] * wrapper.num_envs)
    _, rewards, _, infos = wrapper.step(actions)
    for reward, info in zip(rewards, infos, strict=True):
        assert reward == pytest.approx(info["reward"])
    wrapper.close()


def test_augmented_reward_prices_the_action_cost():
    wrapper, state = make_wrapped(lam=0.5)
    wrapper.reset()
    actions = np.array([[MOVE_NOOP, SENSE_PAYLOAD]] * wrapper.num_envs)
    _, rewards, _, infos = wrapper.step(actions)
    for reward, info in zip(rewards, infos, strict=True):
        assert reward == pytest.approx(info["reward"] - state.value * info["cost"])
    wrapper.close()


def test_discounted_cost_accumulates_with_the_discount_factor():
    gamma = 0.9
    wrapper, _ = make_wrapped(n_envs=1, episode_length=5, gamma=gamma)
    wrapper.reset()
    actions = np.array([[MOVE_NOOP, SENSE_PAYLOAD]])
    costs = []
    for _ in range(5):
        _, _, dones, infos = wrapper.step(actions)
        costs.append(infos[0]["cost"])
    expected = sum(gamma**t * c for t, c in enumerate(costs))
    assert infos[0]["discounted_cost"] == pytest.approx(expected)
    wrapper.close()


def test_costs_reset_between_episodes():
    wrapper, _ = make_wrapped(n_envs=1, episode_length=4)
    wrapper.reset()
    actions = np.array([[MOVE_NOOP, SENSE_NOOP]])
    for _ in range(8):
        wrapper.step(actions)
    assert len(wrapper.episode_costs) == 2
    assert all(cost == pytest.approx(0.0) for cost in wrapper.episode_costs)
    wrapper.close()


def test_drain_costs_empties_the_buffer():
    wrapper, _ = make_wrapped(n_envs=1, episode_length=4)
    wrapper.reset()
    for _ in range(4):
        wrapper.step(np.array([[MOVE_NOOP, SENSE_PAYLOAD]]))
    assert wrapper.drain_costs()
    assert wrapper.drain_costs() == []
    wrapper.close()


def test_idle_policy_satisfies_the_constraint_and_busy_one_violates_it():
    """Sanity-check the constraint scale: capturing every epoch must blow the
    P_bar = 150 budget, while doing nothing must not."""
    gamma = 0.99
    for action, violates in (
        ([MOVE_NOOP, SENSE_NOOP], False),
        ([MOVE_NOOP, SENSE_PAYLOAD], True),
    ):
        wrapper, state = make_wrapped(n_envs=1, episode_length=300, gamma=gamma)
        wrapper.reset()
        for _ in range(300):
            wrapper.step(np.array([action]))
        jc = wrapper.episode_costs[-1]
        assert (jc > state.threshold) == violates
        wrapper.close()
