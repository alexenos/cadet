"""Spatially correlated cloud fields and the point-to-block discrepancy.

Implements the latent Gaussian model of Appendix A.1:

    Z(p)   ~ stationary isotropic GRF, zero mean, unit variance, Matern kernel
    Zt(p)  = alpha * Z(p) + beta                          (affine transform)
    Y(p)   = sigmoid(Zt(p))                               (observable field)

and the point-to-block discrepancy variance of Equation (10):

    Var(eps_A) = kt(0) - E_{U,V ~ Unif(A)}[ kt(||U - V||) ],   kt(r) = alpha^2 k(r)

Fields are sampled on a regular grid by circulant embedding, which is exact up
to the (empirically negligible) negative eigenvalue mass reported by
:func:`sample_latent_field`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import kv

from .config import (
    AOR_CROSS_TRACK_KM,
    AOR_WIDTH,
    LOOKAHEAD_IMAGE_WIDTH,
    CloudConfig,
)

__all__ = [
    "matern",
    "lookahead_block_side_km",
    "block_discrepancy_sigma",
    "sigma_for_lookahead_width",
    "sample_latent_field",
    "CloudField",
    "sample_cloud_field",
]


# ---------------------------------------------------------------------------
# Covariance kernel
# ---------------------------------------------------------------------------
def matern(r: np.ndarray | float, length_scale: float, nu: float = 0.5) -> np.ndarray:
    """Matern correlation function with unit variance and ``k(0) = 1``.

    ``nu = 0.5`` reduces to the exponential kernel ``exp(-r / length_scale)``,
    the setting used throughout the paper.
    """
    r = np.asarray(r, dtype=np.float64)
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")
    if nu == 0.5:  # exponential; avoids the 0 * inf limit in the general form
        return np.exp(-r / length_scale)

    scaled = np.sqrt(2.0 * nu) * r / length_scale
    out = np.ones_like(scaled)
    nz = scaled > 0
    coeff = (2.0 ** (1.0 - nu)) / gamma_fn(nu)
    out[nz] = coeff * (scaled[nz] ** nu) * kv(nu, scaled[nz])
    return out


# ---------------------------------------------------------------------------
# Lookahead pixel geometry and sigma_A
# ---------------------------------------------------------------------------
def lookahead_block_side_km(
    lookahead_width: int,
    aor_width: int = AOR_WIDTH,
    aor_extent_km: float = AOR_CROSS_TRACK_KM,
    image_width: int = LOOKAHEAD_IMAGE_WIDTH,
) -> float:
    """Ground side length ``L`` of one square lookahead pixel.

    A lookahead sensor images an ``n``-column-wide footprint onto a fixed
    ``image_width``-pixel raster, so a narrower FOV yields a finer ground sample
    distance ("Sensors")::

        L = (n / aor_width) * aor_extent_km / image_width

    For the paper's configuration this is ``n * 250 / 1024`` km, i.e. 1.95, 3.91,
    7.81 and 15.63 km for ``n`` in ``{8, 16, 32, 64}``.
    """
    if lookahead_width <= 0:
        raise ValueError("lookahead_width must be positive.")
    return (lookahead_width / aor_width) * aor_extent_km / image_width


def block_discrepancy_sigma(
    side_km: float,
    cfg: CloudConfig | None = None,
    n_samples: int = 2_000_000,
    seed: int = 0,
) -> float:
    """Monte-Carlo estimate of ``sigma_A`` for a square pixel of side ``side_km``.

    Evaluates Equation (10) by sampling two independent uniform points in the
    block and averaging the transformed kernel::

        sigma_A^2 = alpha^2 * (k(0) - E[k(||U - V||)])
    """
    cfg = cfg or CloudConfig()
    if side_km <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    u = rng.random((n_samples, 2)) * side_km
    v = rng.random((n_samples, 2)) * side_km
    dist = np.linalg.norm(u - v, axis=1)
    mean_k = float(np.mean(matern(dist, cfg.length_scale_km, cfg.nu)))
    var = (cfg.alpha**2) * (1.0 - mean_k)
    return float(np.sqrt(max(var, 0.0)))


@cache
def _sigma_cached(side_km: float, alpha: float, ell: float, nu: float, n: int) -> float:
    cfg = CloudConfig(length_scale_km=ell, nu=nu, alpha=alpha)
    return block_discrepancy_sigma(side_km, cfg, n_samples=n)


def sigma_for_lookahead_width(
    lookahead_width: int,
    cfg: CloudConfig | None = None,
    n_samples: int = 2_000_000,
) -> float:
    """``sigma_A`` for a lookahead sensor with an ``n``-column footprint.

    The paper reports 0.58, 0.79, 1.07 and 1.38 for ``n`` in ``{8, 16, 32, 64}``
    (Figure 3).
    """
    cfg = cfg or CloudConfig()
    side = lookahead_block_side_km(lookahead_width)
    return _sigma_cached(side, cfg.alpha, cfg.length_scale_km, cfg.nu, n_samples)


# ---------------------------------------------------------------------------
# Circulant-embedding GRF sampler
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _sqrt_spectrum(
    rows: int, cols: int, spacing_km: float, cfg: CloudConfig
) -> tuple[np.ndarray, float]:
    """Square-root spectrum of the circulant covariance on a ``rows x cols`` torus.

    Cached because every episode of a given configuration reuses the same grid;
    recomputing the FFT of the covariance each reset would dominate step cost.
    """
    di = np.minimum(np.arange(rows), rows - np.arange(rows))
    dj = np.minimum(np.arange(cols), cols - np.arange(cols))
    dist = np.hypot(di[:, None] * spacing_km, dj[None, :] * spacing_km)
    cov = matern(dist, cfg.length_scale_km, cfg.nu)
    lam = np.fft.fft2(cov).real
    total = lam.sum()
    neg_mass = float(-lam[lam < 0].sum() / total) if total > 0 else 0.0
    np.clip(lam, 0.0, None, out=lam)
    return np.sqrt(lam), neg_mass


def sample_latent_field(
    shape: tuple[int, int],
    cfg: CloudConfig,
    rng: np.random.Generator,
    return_diagnostics: bool = False,
    spacing_km: float | None = None,
):
    """Sample a zero-mean unit-variance GRF on a ``shape`` grid.

    Uses the Dietrich-Newsam circulant embedding: the covariance is embedded on a
    padded torus, diagonalised by a 2-D FFT, and a complex Gaussian is filtered
    by the square-root spectrum.  The real part is returned; negative eigenvalues
    (an artefact of finite padding) are clipped to zero.

    ``spacing_km`` is the physical distance between adjacent grid points; it
    defaults to one sub-cell.  Passing a coarser spacing is how
    :func:`sample_cloud_field` draws the field on the lookahead-pixel grid.

    Returns the field, or ``(field, negative_eigenvalue_fraction)`` when
    ``return_diagnostics`` is set.
    """
    rows, cols = shape
    pad = int(cfg.circulant_pad)
    prows, pcols = rows + pad, cols + pad
    spacing = cfg.subpixel_km if spacing_km is None else float(spacing_km)

    sqrt_lam, neg_mass = _sqrt_spectrum(prows, pcols, spacing, cfg)

    noise = rng.standard_normal((prows, pcols)) + 1j * rng.standard_normal(
        (prows, pcols)
    )
    spectrum = noise * sqrt_lam
    field = np.fft.fft2(spectrum) / np.sqrt(prows * pcols)
    out = np.ascontiguousarray(field.real[:rows, :cols], dtype=np.float32)
    if return_diagnostics:
        return out, neg_mass
    return out


# ---------------------------------------------------------------------------
# Observable cloud field
# ---------------------------------------------------------------------------
@dataclass
class CloudField:
    """A realised cloud field on the sub-cell grid of one episode.

    Attributes
    ----------
    values:
        ``Y(p) in [0, 1]`` on a ``(world_height * s, world_width * s)`` grid,
        where ``s = cfg.subpixels_per_cell``.
    cfg:
        The cloud model that generated the field.
    """

    values: np.ndarray
    cfg: CloudConfig

    @property
    def subpixels_per_cell(self) -> int:
        return self.cfg.subpixels_per_cell

    def visible(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Ground-truth visibility ``Y(p) < tau`` at sub-cell coordinates."""
        return self.values[rows, cols] < self.cfg.tau

    def cell_means(self) -> np.ndarray:
        """Block-average the field down to AoR-cell resolution."""
        s = self.subpixels_per_cell
        h, w = self.values.shape
        return self.values.reshape(h // s, s, w // s, s).mean(axis=(1, 3))

    def block_average(
        self, row0: int, row1: int, col0: int, col1: int, block: int
    ) -> np.ndarray:
        """Average ``Y`` over ``block x block`` sub-cell tiles in a sub-window.

        The window bounds must be exact multiples of ``block`` in extent.
        """
        window = self.values[row0:row1, col0:col1]
        h, w = window.shape
        if h % block or w % block:
            raise ValueError(
                f"Window {h}x{w} is not tileable by {block}x{block} blocks."
            )
        return window.reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def sample_cloud_field(
    world_shape: tuple[int, int],
    cfg: CloudConfig,
    rng: np.random.Generator,
    block_subcells: int = 1,
) -> CloudField:
    """Draw an observable cloud field over a ``(height, width)`` AoR-cell world.

    ``block_subcells`` is the side of one lookahead pixel in sub-cells.  Under
    ``cfg.field_scale == "lookahead"`` the latent field is drawn on that coarser
    grid and replicated across each pixel, so the block average a lookahead
    observation returns *is* the pointwise value and a target's visibility is
    settled by observing it.  The marginal cloud-free fraction is then
    ``Phi(-beta / alpha)`` for every field of view, because ``Y_A`` is itself a
    transformed unit-variance Gaussian rather than an average of several.

    Under ``"subpixel"`` the argument is ignored and the field is drawn at
    sub-cell resolution, leaving genuine within-pixel variability.
    """
    s = cfg.subpixels_per_cell
    rows, cols = world_shape[0] * s, world_shape[1] * s

    block = int(block_subcells) if cfg.field_scale == "lookahead" else 1
    if block < 1:
        raise ValueError(f"block_subcells must be positive; got {block_subcells}.")
    if rows % block or cols % block:
        raise ValueError(
            f"Sub-cell grid {rows}x{cols} is not tileable by {block}x{block} "
            "lookahead pixels."
        )

    latent = sample_latent_field(
        (rows // block, cols // block),
        cfg,
        rng,
        spacing_km=block * cfg.subpixel_km,
    )
    transformed = cfg.alpha * latent + cfg.beta
    values = 1.0 / (1.0 + np.exp(-transformed, dtype=np.float32))
    if block > 1:
        values = np.repeat(np.repeat(values, block, axis=0), block, axis=1)
    return CloudField(values=values.astype(np.float32), cfg=cfg)
