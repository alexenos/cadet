"""Conditional target visibility given a lookahead observation (Proposition 1).

For a point ``p`` inside a lookahead pixel ``A`` with block-averaged measurement
``Y_A``, the probability that the target is cloud free is approximated by

    Pr(Y(p) < tau | Y_A) ~= Phi( (logit(tau) - logit(Y_A)) / sigma_A )

where ``sigma_A^2 = Var(eps_A)`` is the point-to-block discrepancy variance of
Equation (10).  The approximation error is ``O(sigma_A)`` and is smallest near
``Y_A = 1/2`` where visibility is most uncertain (Appendix A.3).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .clouds import sigma_for_lookahead_width
from .config import CloudConfig

__all__ = [
    "logit",
    "conditional_visibility",
    "VisibilityModel",
    "PAPER_SIGMA_A",
]

#: ``sigma_A`` values reported in Figure 3 of the paper, keyed by lookahead
#: footprint width.  Our Monte-Carlo evaluation of Equation (10) yields values
#: ~6% larger (0.62 / 0.85 / 1.13 / 1.45); see ``docs/reproduction-notes.md``.
#: Pass ``use_paper_sigma=True`` to :class:`VisibilityModel` to pin these.
PAPER_SIGMA_A: dict[int, float] = {8: 0.58, 16: 0.79, 32: 1.07, 64: 1.38}

_EPS = 1e-6


def logit(y: np.ndarray | float, eps: float = _EPS) -> np.ndarray:
    """Numerically guarded ``log(y / (1 - y))``."""
    y = np.clip(np.asarray(y, dtype=np.float64), eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def conditional_visibility(
    block_mean: np.ndarray | float,
    sigma_a: float,
    tau: float = 0.5,
) -> np.ndarray:
    """Evaluate Proposition 1.

    Parameters
    ----------
    block_mean:
        Block-averaged cloud measurement ``Y_A`` in ``[0, 1]``.
    sigma_a:
        Point-to-block discrepancy standard deviation for the pixel.
    tau:
        Cloud threshold; a target is visible when ``Y(p) < tau``.

    Returns
    -------
    Probability that the target at ``p`` is cloud free.  With ``sigma_a = 0``
    the model degenerates to the deterministic test ``Y_A < tau``.
    """
    z_tau = logit(tau)
    z_block = logit(block_mean)
    if sigma_a <= 0:
        return (z_block < z_tau).astype(np.float64)
    return ndtr((z_tau - z_block) / sigma_a)


class VisibilityModel:
    """Bundles ``sigma_A`` with the cloud model for one lookahead configuration.

    ``sigma_A`` depends only on the spatial correlation structure of the cloud
    field and the geometry of a lookahead pixel, so it can be precomputed once
    per sensor configuration and reused for every observation.
    """

    def __init__(
        self,
        lookahead_width: int,
        cloud_cfg: CloudConfig | None = None,
        use_paper_sigma: bool = False,
        sigma_override: float | None = None,
    ) -> None:
        self.cloud_cfg = cloud_cfg or CloudConfig()
        self.lookahead_width = int(lookahead_width)
        if sigma_override is not None:
            self.sigma_a = float(sigma_override)
        elif use_paper_sigma and self.lookahead_width in PAPER_SIGMA_A:
            self.sigma_a = PAPER_SIGMA_A[self.lookahead_width]
        else:
            self.sigma_a = sigma_for_lookahead_width(
                self.lookahead_width, self.cloud_cfg
            )
        self.tau = self.cloud_cfg.tau
        self.prior = self.cloud_cfg.prior_visibility

    def __call__(self, block_mean: np.ndarray | float) -> np.ndarray:
        return conditional_visibility(block_mean, self.sigma_a, self.tau)

    def probability(self, block_mean: np.ndarray | float) -> np.ndarray:
        """Alias for :meth:`__call__` with a self-documenting name."""
        return self(block_mean)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"VisibilityModel(lookahead_width={self.lookahead_width}, "
            f"sigma_a={self.sigma_a:.3f}, tau={self.tau})"
        )
