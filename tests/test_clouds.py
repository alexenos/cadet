"""Tests for the cloud field model and the point-to-block discrepancy."""

import numpy as np
import pytest

from cadet.clouds import (
    lookahead_block_side_km,
    matern,
    sample_cloud_field,
    sample_latent_field,
    sigma_for_lookahead_width,
)
from cadet.config import CloudConfig


def test_matern_nu_half_is_exponential():
    r = np.linspace(0.0, 40.0, 50)
    np.testing.assert_allclose(matern(r, 10.0, nu=0.5), np.exp(-r / 10.0), rtol=1e-12)


def test_matern_general_form_matches_exponential_limit():
    """The general Bessel form must agree with the closed form at nu = 0.5."""
    r = np.linspace(0.1, 30.0, 40)
    general = matern(r, 10.0, nu=0.5 + 1e-9)
    np.testing.assert_allclose(general, np.exp(-r / 10.0), rtol=1e-6)


def test_matern_is_unit_at_zero():
    for nu in (0.5, 1.5, 2.5):
        assert matern(0.0, 10.0, nu=nu) == pytest.approx(1.0)


def test_lookahead_block_side_scales_with_fov():
    """A narrower FOV imaged onto the same raster gives finer ground pixels."""
    sides = [lookahead_block_side_km(n) for n in (8, 16, 32, 64)]
    np.testing.assert_allclose(sides, [n * 250 / 1024 for n in (8, 16, 32, 64)])
    assert sides == sorted(sides)


def test_sigma_increases_with_footprint():
    """Larger pixels average over more area, so pointwise inference is noisier."""
    sigmas = [sigma_for_lookahead_width(n) for n in (8, 16, 32, 64)]
    assert sigmas == sorted(sigmas)
    # Within ~10% of the values printed in Figure 3.
    for value, reference in zip(sigmas, (0.58, 0.79, 1.07, 1.38), strict=True):
        assert value == pytest.approx(reference, rel=0.12)


def test_sigma_vanishes_for_a_point_block():
    """With no spatial averaging there is no point-to-block discrepancy."""
    from cadet.clouds import block_discrepancy_sigma

    assert block_discrepancy_sigma(1e-9) == pytest.approx(0.0, abs=1e-3)
    assert block_discrepancy_sigma(0.0) == 0.0


def test_latent_field_has_unit_variance_and_zero_mean():
    cfg = CloudConfig()
    rng = np.random.default_rng(0)
    field = sample_latent_field((256, 128), cfg, rng)
    assert field.mean() == pytest.approx(0.0, abs=0.15)
    assert field.std() == pytest.approx(1.0, rel=0.1)


def test_latent_field_reproduces_the_matern_correlation():
    """Empirical correlation along a row must track the specified kernel."""
    cfg = CloudConfig()
    rng = np.random.default_rng(1)
    field = sample_latent_field((512, 256), cfg, rng)
    field = field - field.mean()
    variance = field.var()
    for lag_cells in (1, 3, 6):
        empirical = float((field[:, :-lag_cells] * field[:, lag_cells:]).mean())
        expected = matern(lag_cells * cfg.subpixel_km, cfg.length_scale_km, cfg.nu)
        assert empirical / variance == pytest.approx(expected, abs=0.08)


def test_cloud_free_fraction_matches_the_design_target():
    """beta = 0.8, alpha = 2.0 puts ~1/3 of the surface below the threshold."""
    cfg = CloudConfig()
    rng = np.random.default_rng(2)
    field = sample_cloud_field((400, 32), cfg, rng)
    fraction = float((field.values < cfg.tau).mean())
    assert fraction == pytest.approx(cfg.prior_visibility, abs=0.05)
    assert cfg.prior_visibility == pytest.approx(0.3446, abs=1e-3)


def test_cloud_values_are_bounded():
    cfg = CloudConfig()
    field = sample_cloud_field((128, 32), cfg, np.random.default_rng(3))
    assert field.values.min() >= 0.0
    assert field.values.max() <= 1.0


def test_block_average_matches_manual_reduction():
    cfg = CloudConfig()
    field = sample_cloud_field((16, 32), cfg, np.random.default_rng(4))
    block = field.block_average(0, 8, 0, 8, block=4)
    assert block.shape == (2, 2)
    assert block[0, 0] == pytest.approx(field.values[0:4, 0:4].mean(), rel=1e-6)


def test_empirical_discrepancy_matches_the_closed_form_variance():
    """Var(logit(Y(p)) - logit(Y_A)) should track sigma_A^2 (Equation 10).

    Checked in the latent space, where the model is exact up to the local
    linearisation of Assumption 1.
    """
    cfg = CloudConfig()
    rng = np.random.default_rng(5)
    latent = sample_latent_field((512, 256), cfg, rng)
    transformed = cfg.alpha * latent + cfg.beta

    block = 4  # 4 sub-cells == a 7.81 km pixel == the n = 32 lookahead
    h, w = transformed.shape
    means = transformed.reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    upsampled = np.repeat(np.repeat(means, block, axis=0), block, axis=1)
    empirical = float((transformed - upsampled).std())

    expected = sigma_for_lookahead_width(32, cfg)
    assert empirical == pytest.approx(expected, rel=0.15)
