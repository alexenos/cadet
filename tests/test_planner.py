"""Tests for the SSP dynamic program, checked against brute-force enumeration."""

import itertools

import numpy as np
import pytest

from cadet.planner import (
    NOOP,
    ROLL_LEFT,
    ROLL_RIGHT,
    first_command,
    footprint_gain,
    solve_roll_trajectory,
)


def brute_force(gain, start_col, max_slew=1):
    """Exhaustive search over agility-feasible paths."""
    horizon, width = gain.shape
    best = -np.inf
    for path in itertools.product(range(width), repeat=horizon):
        if start_col is not None and path[0] != start_col:
            continue
        if any(abs(path[i + 1] - path[i]) > max_slew for i in range(horizon - 1)):
            continue
        best = max(best, sum(gain[t, path[t]] for t in range(horizon)))
    return best


def test_footprint_gain_sums_the_swath():
    utility = np.array([[1.0, 2.0, 3.0, 4.0]])
    gain = footprint_gain(utility, footprint_width=3)
    np.testing.assert_allclose(gain, [[3.0, 6.0, 9.0, 7.0]])


def test_footprint_gain_width_one_is_identity():
    utility = np.random.default_rng(0).random((5, 6))
    np.testing.assert_allclose(footprint_gain(utility, 1), utility)


def test_footprint_gain_rejects_even_widths():
    with pytest.raises(ValueError):
        footprint_gain(np.zeros((2, 2)), footprint_width=2)


@pytest.mark.parametrize("seed", range(25))
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    horizon = int(rng.integers(1, 7))
    width = int(rng.integers(2, 6))
    utility = rng.integers(0, 3, size=(horizon, width)).astype(float)
    start = int(rng.integers(0, width)) if seed % 2 else None

    columns, value = solve_roll_trajectory(utility, start_col=start)
    gain = footprint_gain(utility, 3)

    assert value == pytest.approx(brute_force(gain, start))
    assert value == pytest.approx(sum(gain[t, columns[t]] for t in range(horizon)))
    assert np.all(np.abs(np.diff(columns)) <= 1)
    if start is not None:
        assert columns[0] == start


def test_trajectory_respects_the_agility_constraint():
    rng = np.random.default_rng(11)
    utility = rng.random((200, 32))
    columns, _ = solve_roll_trajectory(utility, start_col=16)
    assert np.all(np.abs(np.diff(columns)) <= 1)
    assert columns.min() >= 0 and columns.max() < 32


def test_empty_horizon_is_handled():
    columns, value = solve_roll_trajectory(np.zeros((0, 4)))
    assert columns.size == 0 and value == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_first_command_is_value_equivalent_to_full_backtracking(seed):
    """The fast backward DP must start an equally optimal trajectory."""
    rng = np.random.default_rng(100 + seed)
    horizon = int(rng.integers(2, 15))
    width = int(rng.integers(2, 10))
    utility = rng.random((horizon, width)).round(2)
    start = int(rng.integers(0, width))

    gain = footprint_gain(utility, 3)

    def value_of(command):
        delta = {NOOP: 0, ROLL_LEFT: -1, ROLL_RIGHT: 1}[command]
        column = int(np.clip(start + delta, 0, width - 1))
        _, tail = solve_roll_trajectory(utility[1:], start_col=column)
        return gain[0, start] + tail

    fast = first_command(utility, start)
    _, optimal = solve_roll_trajectory(utility, start_col=start)
    assert value_of(fast) == pytest.approx(optimal)


def test_first_command_moves_towards_reward():
    utility = np.zeros((10, 9))
    utility[5:, 8] = 1.0
    assert first_command(utility, start_col=0) == ROLL_RIGHT
    utility = np.zeros((10, 9))
    utility[5:, 0] = 1.0
    assert first_command(utility, start_col=8) == ROLL_LEFT


def test_first_command_holds_when_already_optimal():
    utility = np.zeros((10, 9))
    utility[:, 4] = 1.0
    assert first_command(utility, start_col=4) == NOOP


def test_oracle_utilities_beat_nominal_utilities_on_the_realised_field():
    """Perfect cloud knowledge must never plan a worse realised outcome."""
    rng = np.random.default_rng(3)
    horizon, width = 120, 32
    present = rng.random((horizon, width)) < 0.05
    visible = present & (rng.random((horizon, width)) < 0.34)

    nominal = present.astype(float)
    oracle = visible.astype(float)

    def realised(utility):
        columns, _ = solve_roll_trajectory(utility, start_col=16)
        gain = footprint_gain(oracle, 3)
        return sum(gain[t, columns[t]] for t in range(horizon))

    assert realised(oracle) >= realised(nominal)
