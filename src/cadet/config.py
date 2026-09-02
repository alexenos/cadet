"""Configuration dataclasses for the CADET environment, planner and training.

Default values reproduce the experimental setup described in

    N. Nordlund, T. Upthegrove, L. Tassiulas,
    "Energy-Aware Dynamic Tasking for Earth Observing Satellites with Deep
    Reinforcement Learning", SSC26-IX-06, 40th Annual Small Satellite Conference.

Section names quoted in the docstrings refer to that paper.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Geometry of the simulated area of regard (AoR).
#
# The spacecraft flies at 400 km with a maximum roll of +/-15 deg, which spans
# ~250 km cross-track.  The AoR is rasterised into 32 cross-track columns, so a
# single AoR cell is 250 / 32 = 7.8125 km on a side ("Spacecraft").
# ---------------------------------------------------------------------------
AOR_CROSS_TRACK_KM: float = 250.0
AOR_WIDTH: int = 32
AOR_HEIGHT: int = 64
AOR_PIXEL_KM: float = AOR_CROSS_TRACK_KM / AOR_WIDTH  # 7.8125 km

#: Horizontal resolution of a lookahead image, independent of its FOV.  A
#: narrower FOV therefore yields a finer ground sample distance.
LOOKAHEAD_IMAGE_WIDTH: int = 32

#: Channels in the observation tensor (see :mod:`cadet.observation`).
N_OBS_CHANNELS: int = 8


@dataclass(frozen=True)
class CloudConfig:
    """Latent Gaussian random field model of the cloud field (Appendix A.1).

    The observable cloud field is ``Y(p) = sigmoid(alpha * Z(p) + beta)`` where
    ``Z`` is a zero-mean, unit-variance, stationary and isotropic GRF with a
    Matern covariance kernel.
    """

    #: Matern length scale in kilometres.
    length_scale_km: float = 10.0
    #: Matern smoothness.  ``nu = 0.5`` reduces to the exponential kernel.
    nu: float = 0.5
    #: Contrast of the latent field.
    alpha: float = 2.0
    #: Mean shift.  ``Phi(-beta / alpha) ~ 1/3`` of the surface is cloud free,
    #: matching the ~67% global mean cloud fraction.
    beta: float = 0.8
    #: A target is visible when ``Y(p) < tau``.
    tau: float = 0.5
    #: Sub-cells per AoR cell used to simulate the field.  A lookahead pixel is
    #: ``n / 8`` sub-cells wide for a footprint of ``n`` AoR columns, so a
    #: factor of 4 places every FOV in ``{8, 16, 32, 64}`` on an integer grid.
    subpixels_per_cell: int = 4
    #: Circulant padding (in sub-cells) used by the GRF sampler.
    circulant_pad: int = 48
    #: Scale at which the field is simulated.
    #:
    #: ``"lookahead"`` draws it on the lookahead-pixel grid and holds it
    #: constant within a pixel, so ``Y(p) = Y_A`` and a lookahead observation
    #: settles whether a target is cloud free.  This is the convention the
    #: paper's own implementation uses (the authors set
    #: ``is_cloud_free = is_observed_cloud_free``), and it is the default here.
    #:
    #: ``"subpixel"`` draws it at ``subpixels_per_cell`` resolution and reads
    #: ground truth pointwise, so a capture stays uncertain even after a perfect
    #: observation.  More physically realistic, and the setting under which
    #: Proposition 1 is genuinely calibrated -- but it is not what produced the
    #: published numbers.  See ``docs/shortfall-resolved.md``.
    field_scale: str = "lookahead"
    #: Pay the payload *reward* on a Bernoulli draw from Proposition 1 rather
    #: than on the deterministic test, while every reported metric and both
    #: baselines keep the deterministic one.
    #:
    #: This is the second reading of the author's "we set
    #: ``is_cloud_free = is_observed_cloud_free`` for consistency across the
    #: evaluations": that the Prop-1 noise was dropped from the *metric* so the
    #: 24 cells share a common world, but stayed in the training reward.  It
    #: restores sigma_A to governing something -- under the default the
    #: proposition is a monotone re-encoding of the observation, calibrated to
    #: nothing.  See ``docs/truth-noise-hypothesis.md``.
    truth_noise: bool = False

    def __post_init__(self) -> None:
        if self.field_scale not in ("lookahead", "subpixel"):
            raise ValueError(
                f"field_scale must be 'lookahead' or 'subpixel'; got "
                f"{self.field_scale!r}."
            )
        if self.truth_noise and self.field_scale != "lookahead":
            # A sub-pixel field already carries within-pixel variation, so
            # adding a Prop-1 draw on top would apply the same uncertainty
            # twice.
            raise ValueError(
                "truth_noise requires field_scale='lookahead'; a sub-pixel "
                "field already realises the point-to-block discrepancy."
            )

    @property
    def subpixel_km(self) -> float:
        return AOR_PIXEL_KM / self.subpixels_per_cell

    @property
    def prior_visibility(self) -> float:
        """Unconditional ``Pr(Y(p) < tau)`` before any lookahead observation."""
        from math import erf, sqrt

        return 0.5 * (1.0 + erf((-self.beta / self.alpha) / sqrt(2.0)))


@dataclass(frozen=True)
class PowerConfig:
    """Asymmetric per-action power costs ("Power Model")."""

    roll: float = 36.0
    lookahead: float = 200.0
    payload: float = 750.0
    #: Extra cost charged when the policy delegates to the SSP planner.
    planner: float = 18.0
    #: Long-term average power budget ``P_bar``.
    budget: float = 150.0
    #: Normalise the constraint cost by ``budget`` before accumulating it.  This
    #: keeps the dual step size scale-free across the three budgets and makes
    #: the constraint threshold simply ``1 / (1 - gamma)``.
    normalise_by_budget: bool = True


@dataclass(frozen=True)
class SensorConfig:
    """Body-fixed payload and lookahead sensor geometry ("Sensors")."""

    #: Payload swath in AoR columns (~20 km, Planet Tanager class).
    payload_width: int = 3
    #: Cross-track lookahead footprint width in AoR columns.
    lookahead_width: int = 32
    #: Along-track offset of the lookahead footprint's near edge from nadir.
    lookahead_offset: int = 32
    #: Along-track extent of the lookahead footprint.
    lookahead_height: int = 32

    def __post_init__(self) -> None:
        if self.lookahead_offset + self.lookahead_height != AOR_HEIGHT:
            raise ValueError(
                "The lookahead footprint's far edge must coincide with the top "
                f"of the AoR ({AOR_HEIGHT} rows); got "
                f"{self.lookahead_offset} + {self.lookahead_height}."
            )
        if self.payload_width % 2 != 1:
            raise ValueError("payload_width must be odd so it centres on nadir.")


@dataclass(frozen=True)
class EnvConfig:
    """Full environment specification."""

    episode_length: int = 300
    #: Targets drawn uniformly without replacement over the world grid.
    n_targets: int = 178
    #: Adds the ``delegate`` maneuvering action, turning CADET into CADET-Plan.
    enable_planner_action: bool = False
    #: Horizon of the embedded SSP planner, in decision epochs.
    planner_horizon: int = AOR_HEIGHT
    gamma: float = 0.99
    clouds: CloudConfig = field(default_factory=CloudConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    seed: int | None = None

    @property
    def world_height(self) -> int:
        """Rows of the underlying world grid.

        The world extends one AoR height beyond the episode horizon so that the
        AoR is fully defined at the final decision epoch.
        """
        return self.episode_length + AOR_HEIGHT

    @property
    def world_width(self) -> int:
        return AOR_WIDTH

    @property
    def target_density(self) -> float:
        return self.n_targets / self.world_height

    def replace(self, **changes: Any) -> EnvConfig:
        return dataclasses.replace(self, **changes)

    def evaluation_variant(self, episode_length: int = 3000) -> EnvConfig:
        """Longer-episode copy that preserves the training target density.

        Evaluation episodes are 10x longer than training episodes and the
        target count is scaled to match ("Experimental Design").
        """
        n_targets = int(round(self.target_density * (episode_length + AOR_HEIGHT)))
        return self.replace(episode_length=episode_length, n_targets=n_targets)


@dataclass(frozen=True)
class TrainConfig:
    """PPO and primal-dual hyperparameters (Appendix B, Table 5)."""

    total_timesteps: int = 30_000_000
    n_envs: int = 8

    # --- PPO ---
    learning_rate: float = 1e-4
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    features_dim: int = 256

    # --- Lagrangian dual ascent ---
    lambda_lr: float = 1e-3
    lambda_init: float = 0.0
    #: Budget slack curriculum: hold ``initial_slack`` for ``warmup_steps``
    #: then taper linearly to 1.0 over ``taper_steps``.
    warmup_steps: int = 1_000_000
    taper_steps: int = 10_000_000
    initial_slack: float = 5.0

    seed: int = 0
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Targets per world row.  The paper specifies 178 targets over a 364-row
#: training world and 1532 over a 3064-row evaluation world; 0.5 reproduces the
#: evaluation count exactly and the training count to within one target.
TARGET_DENSITY: float = 0.5

#: The three power regimes and four lookahead FOVs swept in the paper.
POWER_BUDGETS: tuple[float, ...] = (100.0, 150.0, 1500.0)
LOOKAHEAD_WIDTHS: tuple[int, ...] = (8, 16, 32, 64)
CONTROLLERS: tuple[str, ...] = ("cadet", "cadet-plan")


def make_env_config(
    lookahead_width: int = 32,
    budget: float = 150.0,
    controller: str = "cadet",
    episode_length: int = 300,
    n_targets: int | None = None,
    seed: int | None = None,
    truth_noise: bool = False,
) -> EnvConfig:
    """Convenience constructor for one cell of the experimental grid."""
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller {controller!r}; expected {CONTROLLERS}.")
    if n_targets is None:
        n_targets = int(round(TARGET_DENSITY * (episode_length + AOR_HEIGHT)))
    return EnvConfig(
        episode_length=episode_length,
        n_targets=n_targets,
        enable_planner_action=(controller == "cadet-plan"),
        clouds=CloudConfig(truth_noise=truth_noise),
        power=PowerConfig(budget=budget),
        sensors=SensorConfig(lookahead_width=lookahead_width),
        seed=seed,
    )
