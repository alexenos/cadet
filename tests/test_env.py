"""Tests for the dynamic tasking CPOMDP environment."""

import numpy as np
import pytest

from cadet.config import (
    AOR_HEIGHT,
    AOR_WIDTH,
    N_OBS_CHANNELS,
    CloudConfig,
    PowerConfig,
    SensorConfig,
    make_env_config,
)
from cadet.env import (
    CH_CLOUD_MASK,
    CH_CLOUD_VALUE,
    CH_FOOTPRINT,
    CH_TARGETS_CLEAR,
    CH_TARGETS_OBSCURED,
    CH_TARGETS_TOTAL,
    CH_TARGETS_UNOBSERVED,
    CH_VISIBILITY,
    MOVE_DELEGATE,
    MOVE_LEFT,
    MOVE_NOOP,
    MOVE_RIGHT,
    SENSE_LOOKAHEAD,
    SENSE_NOOP,
    SENSE_PAYLOAD,
    DynamicTaskingEnv,
)


@pytest.fixture(scope="module")
def env():
    return DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=100))


@pytest.fixture(scope="module")
def plan_env():
    return DynamicTaskingEnv(
        make_env_config(32, 150.0, "cadet-plan", episode_length=100)
    )


# ---------------------------------------------------------------------------
# Spaces and API conformance
# ---------------------------------------------------------------------------
def test_observation_shape_and_bounds(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (N_OBS_CHANNELS, AOR_HEIGHT, AOR_WIDTH)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_action_space_sizes(env, plan_env):
    assert list(env.action_space.nvec) == [3, 3]
    assert list(plan_env.action_space.nvec) == [4, 3]


def test_passes_the_gymnasium_api_checker():
    from gymnasium.utils.env_checker import check_env

    env = DynamicTaskingEnv(make_env_config(16, 150.0, "cadet-plan", episode_length=20))
    check_env(env, skip_render_check=True)


def test_reset_is_deterministic_given_a_seed(env):
    first, _ = env.reset(seed=42)
    targets_first = env.target_visible.copy()
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(targets_first, env.target_visible)


def test_different_seeds_give_different_worlds(env):
    env.reset(seed=1)
    a = env.target_row.copy()
    env.reset(seed=2)
    assert not np.array_equal(a, env.target_row)


def test_episode_truncates_at_the_horizon():
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=25))
    env.reset(seed=0)
    for step in range(25):
        _, _, terminated, truncated, _ = env.step([MOVE_NOOP, SENSE_NOOP])
        assert not terminated
        assert truncated == (step == 24)


def test_invalid_actions_are_rejected(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step([9, SENSE_NOOP])
    with pytest.raises(ValueError):
        env.step([MOVE_NOOP, 7])
    with pytest.raises(ValueError):
        env.step([MOVE_DELEGATE, SENSE_NOOP])  # CADET has no delegate action


# ---------------------------------------------------------------------------
# Targets and cloud statistics
# ---------------------------------------------------------------------------
def test_targets_are_unique_per_cell(env):
    env.reset(seed=5)
    cells = env.target_row.astype(np.int64) * AOR_WIDTH + env.target_col
    assert np.unique(cells).size == cells.size


def test_target_count_matches_the_configuration(env):
    env.reset(seed=5)
    assert env.target_row.size == env.cfg.n_targets


def test_cloud_free_target_fraction_is_about_one_third():
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=3000))
    fractions = [
        (env.reset(seed=s), env.target_visible.mean())[1] for s in range(3)
    ]
    assert np.mean(fractions) == pytest.approx(1 / 3, abs=0.05)


# ---------------------------------------------------------------------------
# Maneuvering
# ---------------------------------------------------------------------------
def test_roll_moves_one_pointing_state(env):
    env.reset(seed=0)
    start = env.roll_col
    env.step([MOVE_RIGHT, SENSE_NOOP])
    assert env.roll_col == start + 1
    env.step([MOVE_LEFT, SENSE_NOOP])
    assert env.roll_col == start


def test_roll_saturates_at_the_maximum_angle(env):
    env.reset(seed=0)
    for _ in range(AOR_WIDTH + 10):
        env.step([MOVE_LEFT, SENSE_NOOP])
    assert env.roll_col == 0
    env.reset(seed=0)
    for _ in range(AOR_WIDTH + 10):
        env.step([MOVE_RIGHT, SENSE_NOOP])
    assert env.roll_col == AOR_WIDTH - 1


def test_payload_footprint_tracks_the_roll_state(env):
    env.reset(seed=0)
    env.step([MOVE_RIGHT, SENSE_NOOP])
    lo, hi = env.payload_columns()
    assert lo <= env.roll_col < hi
    assert hi - lo <= env.cfg.sensors.payload_width


# ---------------------------------------------------------------------------
# Power costs
# ---------------------------------------------------------------------------
def test_action_costs_follow_the_power_model():
    cfg = make_env_config(32, 150.0, "cadet-plan", episode_length=50)
    env = DynamicTaskingEnv(cfg)
    power = cfg.power

    env.reset(seed=0)
    _, _, _, _, info = env.step([MOVE_NOOP, SENSE_NOOP])
    assert info["raw_cost"] == 0.0

    env.reset(seed=0)
    _, _, _, _, info = env.step([MOVE_RIGHT, SENSE_NOOP])
    assert info["raw_cost"] == pytest.approx(power.roll)

    env.reset(seed=0)
    _, _, _, _, info = env.step([MOVE_NOOP, SENSE_LOOKAHEAD])
    assert info["raw_cost"] == pytest.approx(power.lookahead)

    env.reset(seed=0)
    _, _, _, _, info = env.step([MOVE_NOOP, SENSE_PAYLOAD])
    assert info["raw_cost"] == pytest.approx(power.payload)


def test_delegation_adds_the_planner_cost(plan_env):
    plan_env.reset(seed=0)
    _, _, _, _, info = plan_env.step([MOVE_DELEGATE, SENSE_NOOP])
    power = plan_env.cfg.power
    assert info["raw_cost"] in (
        pytest.approx(power.planner),
        pytest.approx(power.planner + power.roll),
    )


def test_cost_is_normalised_by_the_budget():
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=10))
    env.reset(seed=0)
    _, _, _, _, info = env.step([MOVE_NOOP, SENSE_PAYLOAD])
    assert info["cost"] == pytest.approx(info["raw_cost"] / 150.0)


def test_unnormalised_cost_mode():
    cfg = make_env_config(32, 150.0, "cadet", episode_length=10)
    cfg = cfg.replace(power=PowerConfig(budget=150.0, normalise_by_budget=False))
    env = DynamicTaskingEnv(cfg)
    env.reset(seed=0)
    _, _, _, _, info = env.step([MOVE_NOOP, SENSE_PAYLOAD])
    assert info["cost"] == pytest.approx(info["raw_cost"])


# ---------------------------------------------------------------------------
# Sensing and partial observability
# ---------------------------------------------------------------------------
def test_lookahead_marks_the_observation_mask(env):
    obs, _ = env.reset(seed=3)
    assert obs[CH_CLOUD_MASK].sum() == 0
    obs, _, _, _, _ = env.step([MOVE_NOOP, SENSE_LOOKAHEAD])
    assert obs[CH_CLOUD_MASK].sum() > 0
    # The mask covers the footprint, offset ahead of nadir.
    sensors = env.cfg.sensors
    assert obs[CH_CLOUD_MASK, : sensors.lookahead_offset - 1].sum() == 0


def test_lookahead_only_reveals_targets_inside_the_footprint():
    cfg = make_env_config(8, 150.0, "cadet", episode_length=100)
    env = DynamicTaskingEnv(cfg)
    env.reset(seed=4)
    env.step([MOVE_NOOP, SENSE_LOOKAHEAD])

    row0, row1, col0, col1 = env.lookahead_window()
    # Recompute the window at the pre-step epoch (step advanced t by one).
    row0, row1 = row0 - 1, row1 - 1

    observed = np.flatnonzero(env.target_observed)
    for index in observed:
        assert row0 <= env.target_row[index] < row1
        assert col0 <= env.target_col[index] < col1


def test_no_sensing_reveals_nothing(env):
    env.reset(seed=6)
    for _ in range(10):
        env.step([MOVE_RIGHT, SENSE_NOOP])
    assert not env.target_observed.any()
    assert env.obs_mask.sum() == 0


def test_unobserved_targets_carry_the_prior_belief(env):
    obs, _ = env.reset(seed=7)
    live = obs[CH_TARGETS_TOTAL] > 0
    assert np.allclose(obs[CH_VISIBILITY][live], env.visibility.prior)


def test_observed_targets_are_split_into_clear_and_obscured(env):
    env.reset(seed=8)
    obs, _, _, _, _ = env.step([MOVE_NOOP, SENSE_LOOKAHEAD])
    total_observed = obs[CH_TARGETS_CLEAR].sum() + obs[CH_TARGETS_OBSCURED].sum()
    assert total_observed == pytest.approx(env.target_observed.sum())
    # Channel 2 counts what is left unobserved.
    assert obs[CH_TARGETS_UNOBSERVED].sum() == pytest.approx(
        obs[CH_TARGETS_TOTAL].sum() - total_observed
    )


def test_belief_channel_is_calibrated_against_the_truth():
    """Averaged over many observations, beliefs should match realised visibility."""
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=300))
    beliefs, truths = [], []
    for seed in range(4):
        env.reset(seed=seed)
        for _ in range(300):
            env.step([MOVE_NOOP, SENSE_LOOKAHEAD])
        observed = env.target_observed
        beliefs.append(env.target_belief[observed])
        truths.append(env.target_visible[observed])
    belief = np.concatenate(beliefs)
    truth = np.concatenate(truths)
    assert belief.mean() == pytest.approx(truth.mean(), abs=0.06)


def test_cloud_value_channel_reports_block_averages(env):
    env.reset(seed=9)
    obs, _, _, _, _ = env.step([MOVE_NOOP, SENSE_LOOKAHEAD])
    values = obs[CH_CLOUD_VALUE][obs[CH_CLOUD_MASK] > 0]
    assert values.size > 0
    assert values.min() >= 0.0 and values.max() <= 1.0


# ---------------------------------------------------------------------------
# Payload captures and reward
# ---------------------------------------------------------------------------
def test_reward_only_counts_cloud_free_captures():
    env = DynamicTaskingEnv(make_env_config(32, 1500.0, "cadet", episode_length=300))
    env.reset(seed=10)
    total = 0.0
    for _ in range(300):
        _, reward, _, _, _ = env.step([MOVE_NOOP, SENSE_PAYLOAD])
        total += reward
    stats = env.statistics()
    assert total == stats["captured_targets"]
    assert stats["captured_targets"] <= stats["targets_imaged"]
    # The paper's denominator is capture *actions*, not targets encountered.
    assert stats["capture_accuracy"] == pytest.approx(
        stats["captured_targets"] / stats["capture_attempts"]
    )


def test_captured_targets_leave_the_target_channels():
    env = DynamicTaskingEnv(make_env_config(32, 1500.0, "cadet", episode_length=300))
    env.reset(seed=11)
    before = env.raster_total.sum()
    for _ in range(50):
        env.step([MOVE_NOOP, SENSE_PAYLOAD])
    assert env.raster_total.sum() <= before
    assert env.raster_total.sum() == before - env.n_targets_captured


def test_a_target_cannot_be_captured_twice():
    """Each target has exactly one access time, so repetition is impossible."""
    env = DynamicTaskingEnv(make_env_config(32, 1500.0, "cadet", episode_length=300))
    env.reset(seed=12)
    for _ in range(300):
        env.step([MOVE_NOOP, SENSE_PAYLOAD])
    assert env.n_targets_captured == int(env.target_captured.sum())


def test_no_reward_without_a_payload_action():
    env = DynamicTaskingEnv(make_env_config(32, 1500.0, "cadet", episode_length=100))
    env.reset(seed=13)
    for _ in range(100):
        _, reward, _, _, _ = env.step([MOVE_RIGHT, SENSE_LOOKAHEAD])
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Planner delegation
# ---------------------------------------------------------------------------
def test_delegation_beats_random_rolling(plan_env):
    """Delegating maneuvers should image more targets than rolling at random."""
    rng = np.random.default_rng(0)
    delegated, random_roll = [], []
    for seed in range(4):
        plan_env.reset(seed=seed)
        for _ in range(100):
            plan_env.step([MOVE_DELEGATE, SENSE_PAYLOAD])
        delegated.append(plan_env.statistics()["captured_targets"])

        plan_env.reset(seed=seed)
        for _ in range(100):
            plan_env.step([int(rng.integers(0, 3)), SENSE_PAYLOAD])
        random_roll.append(plan_env.statistics()["captured_targets"])
    assert np.mean(delegated) > np.mean(random_roll)


def test_lookahead_improves_delegated_planning():
    """Beliefs from lookahead should steer the planner towards clear targets.

    Both policies delegate every maneuver; the informed one spends alternate
    epochs on lookahead sensing.  Both fire on the same number of epochs, many
    of them empty, so they are compared on *conversion* -- cloud-free captures
    per target imaged -- rather than on ``capture_accuracy``, whose denominator
    counts empty shots and therefore measures shot timing rather than the value
    of the information.
    """
    cfg = make_env_config(64, 1500.0, "cadet-plan", episode_length=300)
    env = DynamicTaskingEnv(cfg)

    def conversion() -> float:
        stats = env.statistics()
        imaged = stats["targets_imaged"]
        return stats["captured_targets"] / imaged if imaged else 0.0

    informed, blind = [], []
    for seed in range(10):
        env.reset(seed=seed)
        for step in range(300):
            sense = SENSE_LOOKAHEAD if step % 2 == 0 else SENSE_PAYLOAD
            env.step([MOVE_DELEGATE, sense])
        informed.append(conversion())

        env.reset(seed=seed)
        for step in range(300):
            sense = SENSE_NOOP if step % 2 == 0 else SENSE_PAYLOAD
            env.step([MOVE_DELEGATE, sense])
        blind.append(conversion())

    # Without observations the planner sees a uniform prior and can only chase
    # target density, so its conversion is the ambient cloud-free fraction.
    assert np.mean(blind) == pytest.approx(1 / 3, abs=0.1)
    assert np.mean(informed) > np.mean(blind) + 0.05
    assert sum(i > b for i, b in zip(informed, blind, strict=True)) >= 7


def test_planner_utility_uses_current_beliefs(plan_env):
    plan_env.reset(seed=14)
    prior_utility = plan_env._planner_utility().copy()
    assert np.allclose(
        prior_utility[prior_utility > 0], plan_env.visibility.prior
    )
    plan_env.step([MOVE_NOOP, SENSE_LOOKAHEAD])
    updated = plan_env._planner_utility()
    assert not np.allclose(np.unique(updated[updated > 0]), plan_env.visibility.prior)


# ---------------------------------------------------------------------------
# Observation geometry
# ---------------------------------------------------------------------------
def test_footprint_channel_encodes_both_sensors(env):
    obs, _ = env.reset(seed=15)
    footprint = obs[CH_FOOTPRINT]
    assert footprint[0].max() == 1.0  # payload at nadir
    sensors = env.cfg.sensors
    band = footprint[
        sensors.lookahead_offset : sensors.lookahead_offset + sensors.lookahead_height
    ]
    assert band.max() == pytest.approx(0.5)
    assert band.sum() > 0


def test_widest_lookahead_always_spans_the_aor():
    """At n = 64 the footprint covers every column regardless of roll."""
    env = DynamicTaskingEnv(make_env_config(64, 150.0, "cadet", episode_length=50))
    env.reset(seed=16)
    for _ in range(20):
        env.step([MOVE_LEFT, SENSE_NOOP])
    _, _, col0, col1 = env.lookahead_window()
    assert (col0, col1) == (0, AOR_WIDTH)


def test_aor_scrolls_one_row_per_epoch():
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=100))
    obs0, _ = env.reset(seed=17)
    obs1, _, _, _, _ = env.step([MOVE_NOOP, SENSE_NOOP])
    # Row r of the new AoR is row r + 1 of the old one, for the static channels.
    np.testing.assert_allclose(
        obs1[CH_TARGETS_TOTAL][:-1], obs0[CH_TARGETS_TOTAL][1:]
    )


def test_statistics_are_reset_between_episodes(env):
    env.reset(seed=18)
    for _ in range(20):
        env.step([MOVE_RIGHT, SENSE_PAYLOAD])
    assert env.statistics()["n_payload"] == 20
    env.reset(seed=18)
    assert env.statistics()["n_payload"] == 0
    assert env.total_power == 0.0


def test_ansi_render_produces_the_aor(env):
    env.reset(seed=19)
    env.render_mode = "ansi"
    text = env.render()
    assert isinstance(text, str)
    assert len(text.splitlines()) == AOR_HEIGHT + 2


def test_rgb_render_shape(env):
    env.reset(seed=20)
    env.render_mode = "rgb_array"
    frame = env.render()
    assert frame.shape == (AOR_HEIGHT * 8, AOR_WIDTH * 8, 3)
    assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def test_lookahead_geometry_is_validated():
    with pytest.raises(ValueError):
        SensorConfig(lookahead_offset=10, lookahead_height=10)


def test_even_payload_width_is_rejected():
    with pytest.raises(ValueError):
        SensorConfig(payload_width=4)


def test_subpixel_resolution_must_tile_lookahead_pixels():
    cfg = make_env_config(8, 150.0, "cadet", episode_length=20)
    cfg = cfg.replace(clouds=CloudConfig(subpixels_per_cell=1))
    with pytest.raises(ValueError, match="sub-cells"):
        DynamicTaskingEnv(cfg)


# ---------------------------------------------------------------------------
# truth_noise: reward and metric come apart (docs/truth-noise-hypothesis.md)
# ---------------------------------------------------------------------------
def test_truth_noise_is_off_by_default():
    """Reward and metric are the same quantity unless truth_noise is set."""
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=200))
    env.reset(seed=3)
    assert env.target_reward_visible is env.target_visible


def test_truth_noise_pays_the_reward_on_a_prop1_draw():
    """The reward truth is Bernoulli(Prop 1); the metric truth stays exact."""
    cfg = make_env_config(32, 150.0, "cadet", episode_length=200, truth_noise=True)
    env = DynamicTaskingEnv(cfg)
    drawn, expected, deterministic = [], [], []
    for seed in range(6):
        env.reset(seed=seed)
        drawn.append(env.target_reward_visible.mean())
        expected.append(env.target_pvis_if_observed.mean())
        deterministic.append(env.target_visible.mean())
    # Averaged over targets, the draw matches the probability it was drawn from.
    assert np.mean(drawn) == pytest.approx(np.mean(expected), abs=0.03)
    # ...and differs from the deterministic test, which is the whole point.
    assert abs(np.mean(drawn) - np.mean(deterministic)) > 0.005
    # The metric truth is untouched, so baselines stay on a common world.
    assert np.mean(deterministic) == pytest.approx(1 / 3, abs=0.05)


def test_truth_noise_leaves_reported_captures_deterministic():
    """captured_targets counts the deterministic test even when reward does not."""
    cfg = make_env_config(32, 1500.0, "cadet", episode_length=300, truth_noise=True)
    env = DynamicTaskingEnv(cfg)
    env.reset(seed=11)
    total = 0.0
    for _ in range(300):
        _, reward, _, _, _ = env.step([MOVE_NOOP, SENSE_PAYLOAD])
        total += reward
    stats = env.statistics()
    captured = np.count_nonzero(env.target_captured & env.target_visible)
    assert stats["captured_targets"] == captured
    # The reward paid out is a different draw, so the two need not agree.
    assert total != stats["captured_targets"]


def test_truth_noise_rejects_a_subpixel_field():
    """Stacking a Prop-1 draw on a sub-pixel field would double-count noise."""
    with pytest.raises(ValueError, match="truth_noise requires"):
        CloudConfig(field_scale="subpixel", truth_noise=True)
