"""Tests for the SSP and Oracle offline planning baselines."""

import numpy as np
import pytest

from cadet.baselines import evaluate_baselines, gap_closed, run_baseline
from cadet.config import make_env_config
from cadet.env import DynamicTaskingEnv


@pytest.fixture(scope="module")
def env():
    return DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=600))


def test_oracle_dominates_ssp(env):
    """Perfect cloud knowledge must never do worse than the blind schedule."""
    for seed in range(5):
        env.reset(seed=seed)
        ssp = run_baseline(env, "ssp")
        oracle = run_baseline(env, "oracle")
        assert oracle.captured_targets >= ssp.captured_targets


def test_ssp_accuracy_tracks_the_cloud_free_fraction(env):
    """Blind scheduling captures a representative sample, so accuracy ~ 1/3."""
    accuracies = []
    for seed in range(5):
        env.reset(seed=seed)
        accuracies.append(run_baseline(env, "ssp").capture_accuracy)
    assert np.mean(accuracies) == pytest.approx(1 / 3, abs=0.07)


def test_oracle_accuracy_is_high(env):
    env.reset(seed=0)
    assert run_baseline(env, "oracle").capture_accuracy > 0.7


def test_baseline_trajectory_is_agility_feasible(env):
    env.reset(seed=1)
    result = run_baseline(env, "oracle")
    assert result.columns.size == env.cfg.episode_length
    assert np.all(np.abs(np.diff(result.columns)) <= 1)


def test_baseline_starts_at_the_requested_pointing_state(env):
    env.reset(seed=2)
    assert run_baseline(env, "ssp", start_col=7).columns[0] == 7


def test_unknown_baseline_is_rejected(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        run_baseline(env, "magic")


def test_baselines_bracket_the_paper_operating_point():
    """3,000-epoch episodes should land near the paper's ~194 / ~295 bounds."""
    config = make_env_config(32, 150.0, "cadet", episode_length=3000)
    records = evaluate_baselines(config, seeds=range(3))
    ssp = np.mean([r["captured_targets"] for r in records if r["controller"] == "ssp"])
    oracle = np.mean(
        [r["captured_targets"] for r in records if r["controller"] == "oracle"]
    )
    assert 170 < ssp < 235
    assert 265 < oracle < 320
    assert oracle > ssp


def test_baselines_consume_more_than_the_constrained_budgets():
    """Both baselines are power-unconstrained and image at every epoch."""
    config = make_env_config(32, 150.0, "cadet", episode_length=300)
    env = DynamicTaskingEnv(config)
    env.reset(seed=0)
    result = run_baseline(env, "ssp")
    assert result.mean_power >= config.power.payload
    assert result.normalised_power > 1.0


def test_gap_closed_endpoints():
    assert gap_closed(100.0, 100.0, 200.0) == pytest.approx(0.0)
    assert gap_closed(200.0, 100.0, 200.0) == pytest.approx(100.0)
    assert gap_closed(150.0, 100.0, 200.0) == pytest.approx(50.0)
    assert gap_closed(90.0, 100.0, 200.0) == pytest.approx(-10.0)
    assert np.isnan(gap_closed(1.0, 5.0, 5.0))


def test_evaluate_baselines_covers_every_seed():
    config = make_env_config(16, 150.0, "cadet", episode_length=200)
    records = evaluate_baselines(config, seeds=[0, 1])
    assert len(records) == 4
    assert {r["seed"] for r in records} == {0, 1}
