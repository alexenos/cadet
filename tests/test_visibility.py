"""Tests for the conditional visibility model of Proposition 1."""

import numpy as np
import pytest
from scipy.special import ndtr

from cadet.config import CloudConfig
from cadet.visibility import (
    PAPER_SIGMA_A,
    VisibilityModel,
    conditional_visibility,
    logit,
)


def test_logit_is_the_inverse_of_sigmoid():
    y = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-logit(y))), y, rtol=1e-9)


def test_logit_is_guarded_at_the_boundaries():
    assert np.isfinite(logit(0.0))
    assert np.isfinite(logit(1.0))


def test_visibility_is_one_half_at_the_threshold():
    """When the block average sits exactly on tau, visibility is a coin flip."""
    for sigma in (0.3, 0.58, 1.38):
        assert conditional_visibility(0.5, sigma, tau=0.5) == pytest.approx(0.5)


def test_visibility_decreases_with_cloudiness():
    values = conditional_visibility(np.linspace(0.05, 0.95, 25), 1.07)
    assert np.all(np.diff(values) < 0)


def test_visibility_matches_the_closed_form():
    y_a, sigma, tau = 0.3, 0.79, 0.5
    expected = ndtr((logit(tau) - logit(y_a)) / sigma)
    assert conditional_visibility(y_a, sigma, tau) == pytest.approx(expected)


def test_larger_sigma_flattens_the_transition():
    """Coarser lookahead pixels give less decisive visibility estimates."""
    y = np.linspace(0.05, 0.95, 200)
    sharp = conditional_visibility(y, PAPER_SIGMA_A[8])
    blurred = conditional_visibility(y, PAPER_SIGMA_A[64])
    assert np.ptp(sharp) > np.ptp(blurred)
    # Both curves cross 0.5 at the threshold and stay ordered either side of it.
    assert np.all(sharp[y < 0.5] >= blurred[y < 0.5])
    assert np.all(sharp[y > 0.5] <= blurred[y > 0.5])


def test_zero_sigma_degenerates_to_a_hard_threshold():
    values = conditional_visibility(np.array([0.2, 0.8]), 0.0, tau=0.5)
    np.testing.assert_array_equal(values, [1.0, 0.0])


def test_model_selects_sigma_by_footprint_width():
    for width, sigma in PAPER_SIGMA_A.items():
        model = VisibilityModel(width, use_paper_sigma=True)
        assert model.sigma_a == sigma


def test_model_monte_carlo_sigma_is_close_to_the_paper():
    for width, sigma in PAPER_SIGMA_A.items():
        model = VisibilityModel(width)
        assert model.sigma_a == pytest.approx(sigma, rel=0.12)


def test_prior_visibility_is_one_third():
    model = VisibilityModel(32, CloudConfig())
    assert model.prior == pytest.approx(1 / 3, abs=0.02)


def test_sigma_override_takes_precedence():
    model = VisibilityModel(32, sigma_override=0.25)
    assert model.sigma_a == 0.25


def test_calibration_against_a_simulated_field():
    """The model should be calibrated: among targets assigned probability p,
    roughly a fraction p really are cloud free.

    This is the end-to-end check that Proposition 1 is not just monotone but
    numerically right, including the Assumption 1 linearisation.
    """
    from cadet.clouds import sample_cloud_field, sigma_for_lookahead_width

    cfg = CloudConfig()
    rng = np.random.default_rng(7)
    field = sample_cloud_field((600, 32), cfg, rng)

    block = 4  # the n = 32 lookahead pixel
    values = field.values
    h, w = values.shape
    means = values.reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    upsampled = np.repeat(np.repeat(means, block, axis=0), block, axis=1)

    sigma = sigma_for_lookahead_width(32, cfg)
    predicted = conditional_visibility(upsampled.ravel(), sigma, cfg.tau)
    actual = (values.ravel() < cfg.tau).astype(float)

    # Reliability over probability bins.
    edges = np.linspace(0.0, 1.0, 6)
    index = np.clip(np.digitize(predicted, edges) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        mask = index == b
        if mask.sum() < 500:
            continue
        assert actual[mask].mean() == pytest.approx(predicted[mask].mean(), abs=0.12)
