"""Gymnasium environment for cloud-aware dynamic tasking (the CPOMDP).

The environment implements the Atari-style simulator of the EVALUATIONS section:
a 64 x 32 area of regard (AoR) scrolls over a taller world grid, one row per
decision epoch, while the spacecraft rolls between 32 discrete cross-track
pointing states and chooses at each epoch whether to fire the payload sensor,
the lookahead sensor, or neither.

The action is a ``MultiDiscrete`` pair ``(a_move, a_sense)`` so that maneuvering
and sensing are selected *jointly* -- the departure from the sequential
lookahead / plan / capture decomposition of prior work.

Only block-averaged cloud measurements are observable; the pointwise values that
determine whether a capture succeeds stay latent, which is what makes the
problem partially observable.  The raw ``reward`` is the number of cloud-free
targets captured; the resource cost is reported separately in ``info["cost"]``
and is turned into an augmented reward by
:class:`cadet.lagrangian.LagrangianRewardWrapper`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - exercised implicitly
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cadet requires gymnasium; install it with `pip install gymnasium`."
    ) from exc

from .clouds import CloudField, sample_cloud_field
from .config import (
    AOR_HEIGHT,
    AOR_WIDTH,
    LOOKAHEAD_IMAGE_WIDTH,
    N_OBS_CHANNELS,
    EnvConfig,
)
from .planner import NOOP, ROLL_LEFT, ROLL_RIGHT, RollPlanner
from .visibility import VisibilityModel

__all__ = [
    "DynamicTaskingEnv",
    "SENSE_NOOP",
    "SENSE_LOOKAHEAD",
    "SENSE_PAYLOAD",
    "MOVE_NOOP",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "MOVE_DELEGATE",
    "CH_FOOTPRINT",
    "CH_TARGETS_TOTAL",
    "CH_TARGETS_UNOBSERVED",
    "CH_TARGETS_CLEAR",
    "CH_TARGETS_OBSCURED",
    "CH_VISIBILITY",
    "CH_CLOUD_VALUE",
    "CH_CLOUD_MASK",
]

# --- sensing actions (A_sense) ---------------------------------------------
SENSE_NOOP = 0
SENSE_LOOKAHEAD = 1
SENSE_PAYLOAD = 2

# --- maneuvering actions (A_move) ------------------------------------------
MOVE_NOOP = NOOP  # 0
MOVE_LEFT = ROLL_LEFT  # 1
MOVE_RIGHT = ROLL_RIGHT  # 2
MOVE_DELEGATE = 3  # CADET-Plan only

# --- observation channels ---------------------------------------------------
CH_FOOTPRINT = 0
CH_TARGETS_TOTAL = 1
CH_TARGETS_UNOBSERVED = 2
CH_TARGETS_CLEAR = 3
CH_TARGETS_OBSCURED = 4
CH_VISIBILITY = 5
CH_CLOUD_VALUE = 6
CH_CLOUD_MASK = 7

#: Encoding of the two footprints in the single sensor-geometry channel.
PAYLOAD_MARK = 1.0
LOOKAHEAD_MARK = 0.5


class DynamicTaskingEnv(gym.Env):
    """Constrained POMDP for energy-aware dynamic tasking.

    Parameters
    ----------
    config:
        Environment specification; see :class:`cadet.config.EnvConfig`.
    use_paper_sigma:
        Pin ``sigma_A`` to the values printed in Figure 3 instead of evaluating
        Equation (10) by Monte Carlo.
    render_mode:
        ``"ansi"`` or ``"rgb_array"``.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: EnvConfig | None = None,
        use_paper_sigma: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        sensors = self.cfg.sensors
        self.visibility = VisibilityModel(
            sensors.lookahead_width,
            self.cfg.clouds,
            use_paper_sigma=use_paper_sigma,
        )

        self.n_move_actions = 4 if self.cfg.enable_planner_action else 3
        self.action_space = spaces.MultiDiscrete([self.n_move_actions, 3])
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_OBS_CHANNELS, AOR_HEIGHT, AOR_WIDTH),
            dtype=np.float32,
        )

        self.planner = RollPlanner(footprint_width=sensors.payload_width)

        # Sub-cell block side of one lookahead pixel, in sub-cells.  The
        # lookahead images an n-column footprint onto a fixed 32-pixel raster,
        # so a pixel is n / LOOKAHEAD_IMAGE_WIDTH AoR cells across.
        subs = self.cfg.clouds.subpixels_per_cell
        block_cells = sensors.lookahead_width / LOOKAHEAD_IMAGE_WIDTH
        block_sub = block_cells * subs
        if abs(block_sub - round(block_sub)) > 1e-9 or round(block_sub) < 1:
            raise ValueError(
                f"A lookahead pixel spans {block_sub} sub-cells; increase "
                "CloudConfig.subpixels_per_cell so that it is a positive integer."
            )
        self._block_sub = int(round(block_sub))

        self._world_shape = (self.cfg.world_height, self.cfg.world_width)
        self._obs_buffer = np.zeros(
            (N_OBS_CHANNELS, AOR_HEIGHT, AOR_WIDTH), dtype=np.float32
        )

        # Episode state, populated by reset().
        self.t: int = 0
        self.roll_col: int = AOR_WIDTH // 2
        self.cloud: CloudField | None = None
        self._rng = np.random.default_rng(self.cfg.seed)
        self.reset(seed=self.cfg.seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.roll_col = AOR_WIDTH // 2

        self.cloud = sample_cloud_field(self._world_shape, self.cfg.clouds, self._rng)
        self._precompute_block_means()
        self._sample_targets()
        self._init_rasters()
        self._reset_statistics()

        return self._observation(), self._info(cost=0.0, reward=0.0)

    def step(
        self, action: np.ndarray | tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a_move, a_sense = int(action[0]), int(action[1])
        if not 0 <= a_move < self.n_move_actions:
            raise ValueError(f"Invalid maneuvering action {a_move}.")
        if not 0 <= a_sense < 3:
            raise ValueError(f"Invalid sensing action {a_sense}.")

        power = self.cfg.power
        cost = 0.0
        reward = 0.0

        # --- Sensing happens at the current attitude ---------------------
        if a_sense == SENSE_LOOKAHEAD:
            cost += power.lookahead
            self._acquire_lookahead()
            self.n_lookahead += 1
        elif a_sense == SENSE_PAYLOAD:
            cost += power.payload
            reward = self._acquire_payload()
            self.n_payload += 1
        else:
            self.n_sense_noop += 1

        # --- Maneuvering; the slew begins during the same epoch ----------
        command = a_move
        if a_move == MOVE_DELEGATE:
            cost += power.planner
            self.n_delegate += 1
            command = self.planner.command(self._planner_utility(), self.roll_col)
        elif a_move in (MOVE_LEFT, MOVE_RIGHT):
            self.n_roll += 1
        else:
            self.n_move_noop += 1

        if command == ROLL_LEFT:
            cost += power.roll
            self.roll_col = max(0, self.roll_col - 1)
        elif command == ROLL_RIGHT:
            cost += power.roll
            self.roll_col = min(AOR_WIDTH - 1, self.roll_col + 1)

        self.t += 1
        self.total_reward += reward
        self.total_power += cost

        truncated = self.t >= self.cfg.episode_length
        observation = self._observation()
        info = self._info(cost, reward, final=truncated)
        return observation, float(reward), False, truncated, info

    def render(self):  # pragma: no cover - visual helper
        from .render import render_ansi, render_rgb_array

        if self.render_mode == "ansi":
            return render_ansi(self)
        if self.render_mode == "rgb_array":
            return render_rgb_array(self)
        return None

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------
    def _precompute_block_means(self) -> None:
        """Block-average the cloud field at lookahead-pixel resolution.

        The measurement a lookahead observation would return is deterministic
        given the field, so the whole map is precomputed once per episode and
        observations become cheap rectangle reads.
        """
        assert self.cloud is not None
        block = self._block_sub
        values = self.cloud.values
        h, w = values.shape
        if h % block or w % block:
            raise ValueError(
                f"Sub-cell grid {h}x{w} is not tileable by {block}x{block} blocks."
            )
        block_means = values.reshape(h // block, block, w // block, block).mean(
            axis=(1, 3)
        )
        self._block_means = block_means

        # AoR-cell resolution view of the same map, for the cloud-value channel.
        subs = self.cfg.clouds.subpixels_per_cell
        fine = np.repeat(np.repeat(block_means, block, axis=0), block, axis=1)
        self._cell_block_mean = (
            fine.reshape(h // subs, subs, w // subs, subs)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )

    def _sample_targets(self) -> None:
        """Draw targets uniformly without replacement over the world grid."""
        assert self.cloud is not None
        height, width = self._world_shape
        n_cells = height * width
        n = min(self.cfg.n_targets, n_cells)
        flat = self._rng.choice(n_cells, size=n, replace=False)
        rows, cols = np.divmod(flat, width)

        subs = self.cfg.clouds.subpixels_per_cell
        # Each target sits at a random sub-cell within its AoR cell; p_i is a
        # point on the surface, not a whole cell.
        sub_rows = rows * subs + self._rng.integers(0, subs, size=n)
        sub_cols = cols * subs + self._rng.integers(0, subs, size=n)

        self.target_row = rows.astype(np.int32)
        self.target_col = cols.astype(np.int32)
        self.target_visible = self.cloud.values[sub_rows, sub_cols] < self.cfg.clouds.tau
        self.target_observed = np.zeros(n, dtype=bool)
        self.target_captured = np.zeros(n, dtype=bool)

        # Belief the target is cloud free once its lookahead pixel is measured.
        block = self._block_sub
        measured = self._block_means[sub_rows // block, sub_cols // block]
        self.target_pvis_if_observed = self.visibility(measured).astype(np.float32)
        self.target_belief = np.full(n, self.visibility.prior, dtype=np.float32)

        # Index of the (at most one) target occupying each world cell.
        self.target_index = np.full(self._world_shape, -1, dtype=np.int32)
        self.target_index[rows, cols] = np.arange(n, dtype=np.int32)

    def _init_rasters(self) -> None:
        """World-sized channel rasters, updated incrementally as beliefs change."""
        shape = self._world_shape
        occupied = self.target_index >= 0
        self.raster_total = occupied.astype(np.float32)
        self.raster_unobserved = occupied.astype(np.float32)
        self.raster_clear = np.zeros(shape, dtype=np.float32)
        self.raster_obscured = np.zeros(shape, dtype=np.float32)
        # Belief map: doubles as the visibility channel and the planner's utility.
        self.raster_belief = np.zeros(shape, dtype=np.float32)
        self.raster_belief[occupied] = self.visibility.prior

        self.obs_value = np.zeros(shape, dtype=np.float32)
        self.obs_mask = np.zeros(shape, dtype=np.float32)

    def _reset_statistics(self) -> None:
        self.total_reward = 0.0
        self.total_power = 0.0
        self.n_lookahead = 0
        self.n_payload = 0
        self.n_sense_noop = 0
        self.n_roll = 0
        self.n_delegate = 0
        self.n_move_noop = 0
        self.n_capture_attempts = 0
        self.n_targets_captured = 0
        self.n_targets_captured_clear = 0

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------
    def payload_columns(self) -> tuple[int, int]:
        half = self.cfg.sensors.payload_width // 2
        return (
            max(0, self.roll_col - half),
            min(AOR_WIDTH, self.roll_col + half + 1),
        )

    def lookahead_window(self) -> tuple[int, int, int, int]:
        """``(row0, row1, col0, col1)`` of the lookahead footprint in world cells."""
        sensors = self.cfg.sensors
        row0 = self.t + sensors.lookahead_offset
        row1 = min(row0 + sensors.lookahead_height, self._world_shape[0])
        start = self.roll_col - sensors.lookahead_width // 2
        col0 = max(0, start)
        col1 = min(AOR_WIDTH, start + sensors.lookahead_width)
        return row0, row1, col0, col1

    def _acquire_lookahead(self) -> None:
        """Measure block-averaged cloud cover and update target beliefs."""
        row0, row1, col0, col1 = self.lookahead_window()
        if row0 >= row1 or col0 >= col1:
            return

        self.obs_value[row0:row1, col0:col1] = self._cell_block_mean[
            row0:row1, col0:col1
        ]
        self.obs_mask[row0:row1, col0:col1] = 1.0

        window = self.target_index[row0:row1, col0:col1]
        ids = window[window >= 0]
        if ids.size == 0:
            return
        fresh = ids[~self.target_observed[ids] & ~self.target_captured[ids]]
        if fresh.size == 0:
            return

        self.target_observed[fresh] = True
        belief = self.target_pvis_if_observed[fresh]
        self.target_belief[fresh] = belief

        rows = self.target_row[fresh]
        cols = self.target_col[fresh]
        self.raster_unobserved[rows, cols] = 0.0
        self.raster_belief[rows, cols] = belief
        clear = belief >= 0.5
        self.raster_clear[rows[clear], cols[clear]] = 1.0
        self.raster_obscured[rows[~clear], cols[~clear]] = 1.0

    def _acquire_payload(self) -> float:
        """Image the nadir row within the payload swath; returns the reward."""
        row = self.t
        col0, col1 = self.payload_columns()
        self.n_capture_attempts += 1

        window = self.target_index[row, col0:col1]
        ids = window[window >= 0]
        if ids.size == 0:
            return 0.0
        ids = ids[~self.target_captured[ids]]
        if ids.size == 0:
            return 0.0

        self.target_captured[ids] = True
        successes = int(np.count_nonzero(self.target_visible[ids]))
        self.n_targets_captured += int(ids.size)
        self.n_targets_captured_clear += successes

        cols = self.target_col[ids]
        rows = self.target_row[ids]
        self.raster_total[rows, cols] = 0.0
        self.raster_unobserved[rows, cols] = 0.0
        self.raster_clear[rows, cols] = 0.0
        self.raster_obscured[rows, cols] = 0.0
        self.raster_belief[rows, cols] = 0.0
        return float(successes)

    # ------------------------------------------------------------------
    # Planner delegation
    # ------------------------------------------------------------------
    def _planner_utility(self) -> np.ndarray:
        """Utility grid over the planning horizon from the agent's beliefs.

        Task utilities are the conditional visibility probabilities of
        Proposition 1 where a lookahead measurement exists and the prior
        ``Phi(-beta / alpha) ~ 1/3`` elsewhere -- exactly the belief raster.
        """
        horizon = min(
            self.cfg.planner_horizon, self._world_shape[0] - self.t
        )
        return self.raster_belief[self.t : self.t + horizon]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        obs = self._obs_buffer
        obs.fill(0.0)

        top = self.t
        bottom = top + AOR_HEIGHT

        # Channel 0: sensor geometry under the current attitude.
        sensors = self.cfg.sensors
        start = self.roll_col - sensors.lookahead_width // 2
        lcol0 = max(0, start)
        lcol1 = min(AOR_WIDTH, start + sensors.lookahead_width)
        if lcol0 < lcol1:
            obs[
                CH_FOOTPRINT,
                sensors.lookahead_offset : sensors.lookahead_offset
                + sensors.lookahead_height,
                lcol0:lcol1,
            ] = LOOKAHEAD_MARK
        pcol0, pcol1 = self.payload_columns()
        obs[CH_FOOTPRINT, 0, pcol0:pcol1] = PAYLOAD_MARK

        # Channels 1-5: target distribution and status.
        obs[CH_TARGETS_TOTAL] = self.raster_total[top:bottom]
        obs[CH_TARGETS_UNOBSERVED] = self.raster_unobserved[top:bottom]
        obs[CH_TARGETS_CLEAR] = self.raster_clear[top:bottom]
        obs[CH_TARGETS_OBSCURED] = self.raster_obscured[top:bottom]
        obs[CH_VISIBILITY] = self.raster_belief[top:bottom]

        # Channels 6-7: partially observed cloud field (value and mask).
        obs[CH_CLOUD_VALUE] = self.obs_value[top:bottom]
        obs[CH_CLOUD_MASK] = self.obs_mask[top:bottom]
        return obs.copy()

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def _info(self, cost: float, reward: float, final: bool = False) -> dict[str, Any]:
        budget = self.cfg.power.budget
        normalised = cost / budget if self.cfg.power.normalise_by_budget else cost
        info: dict[str, Any] = {
            "cost": float(normalised),
            "raw_cost": float(cost),
            "power_budget": budget,
            "reward": float(reward),
            "roll_col": self.roll_col,
            "epoch": self.t,
        }
        # Building the statistics dict on every step is measurable overhead at
        # 30M timesteps, so it is attached only when the episode ends.
        if final:
            info["episode_stats"] = self.statistics()
        return info

    def statistics(self) -> dict[str, float]:
        """Cumulative episode metrics (the quantities reported in the paper)."""
        epochs = max(self.t, 1)
        accuracy = (
            self.n_targets_captured_clear / self.n_targets_captured
            if self.n_targets_captured
            else 0.0
        )
        return {
            "captured_targets": float(self.n_targets_captured_clear),
            "capture_attempts": float(self.n_capture_attempts),
            "targets_imaged": float(self.n_targets_captured),
            "capture_accuracy": float(accuracy),
            "mean_power": float(self.total_power / epochs),
            "normalised_power": float(
                self.total_power / epochs / self.cfg.power.budget
            ),
            "n_lookahead": float(self.n_lookahead),
            "n_payload": float(self.n_payload),
            "n_sense_noop": float(self.n_sense_noop),
            "n_roll": float(self.n_roll),
            "n_delegate": float(self.n_delegate),
            "n_move_noop": float(self.n_move_noop),
        }

    # ------------------------------------------------------------------
    # Introspection helpers used by baselines, rendering and tests
    # ------------------------------------------------------------------
    def oracle_utility(self) -> np.ndarray:
        """Per-cell utility under perfect pointwise cloud knowledge."""
        grid = np.zeros(self._world_shape, dtype=np.float64)
        visible = self.target_visible
        grid[self.target_row[visible], self.target_col[visible]] = 1.0
        return grid

    def nominal_utility(self) -> np.ndarray:
        """Per-cell utility assuming every target is cloud free."""
        grid = np.zeros(self._world_shape, dtype=np.float64)
        grid[self.target_row, self.target_col] = 1.0
        return grid

    @property
    def n_visible_targets(self) -> int:
        return int(np.count_nonzero(self.target_visible))

    @property
    def n_reachable_targets(self) -> int:
        """Targets whose access time falls inside the episode horizon."""
        return int(np.count_nonzero(self.target_row < self.cfg.episode_length))
