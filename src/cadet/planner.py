"""Classical spacecraft scheduling problem (SSP) solver.

For the single-roll-axis spacecraft of the paper the SSP has exact structure: at
each decision epoch the spacecraft occupies one of ``W`` discrete cross-track
pointing states and may slew at most one state per epoch, so an agility-feasible
observation plan is precisely a path through the ``(epoch, pointing state)``
lattice.  Each target has exactly one access time -- the epoch at which its world
row reaches nadir -- so the repetition constraint is satisfied automatically and
the power-unconstrained SSP

    max_{S subset Q} sum_{q in S} c_q   s.t.  S in K_agility ^ K_repetition

is solved exactly by the dynamic program of Lemaitre et al. in ``O(T * W)``.

The same routine serves three roles:

* the ``SSP`` baseline (every target's utility set to 1),
* the ``Oracle`` baseline (utility 1 only for genuinely cloud-free targets),
* the ``delegate`` action of CADET-Plan, where utilities are the conditional
  visibility probabilities of Proposition 1 over the AoR horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "footprint_gain",
    "first_command",
    "solve_roll_trajectory",
    "RollPlan",
    "RollPlanner",
    "NOOP",
    "ROLL_LEFT",
    "ROLL_RIGHT",
]

#: Maneuvering command encoding shared with :mod:`cadet.env`.
NOOP = 0
ROLL_LEFT = 1
ROLL_RIGHT = 2

_NEG_INF = -1e18


def footprint_gain(utility: np.ndarray, footprint_width: int = 3) -> np.ndarray:
    """Utility collected per ``(epoch, pointing state)`` by the payload sensor.

    ``utility`` is a ``(T, W)`` grid of per-cell utilities in the nadir row at
    each epoch.  The payload footprint spans ``footprint_width`` columns centred
    on the pointing state and is clipped at the edges of the AoR.
    """
    if footprint_width % 2 != 1:
        raise ValueError("footprint_width must be odd.")
    half = footprint_width // 2
    if half == 0:
        return np.asarray(utility, dtype=np.float64)

    utility = np.asarray(utility, dtype=np.float64)
    t, w = utility.shape
    padded = np.zeros((t, w + 2 * half), dtype=np.float64)
    padded[:, half : half + w] = utility
    out = padded[:, half : half + w].copy()
    for shift in range(1, half + 1):
        out += padded[:, half - shift : half - shift + w]
        out += padded[:, half + shift : half + shift + w]
    return out


def first_command(
    utility: np.ndarray,
    start_col: int,
    footprint_width: int = 3,
) -> int:
    """First maneuvering command of the optimal trajectory, via a backward DP.

    Equivalent to ``RollPlan(*solve_roll_trajectory(...)).command`` but avoids
    materialising the backtracking table: only the value function one epoch
    ahead is needed to choose the first slew.  This is the hot path of the
    ``delegate`` action, which may fire on the majority of decision epochs.
    """
    utility = np.asarray(utility, dtype=np.float64)
    horizon, width = utility.shape
    if horizon < 2 or width < 2:
        return NOOP

    gain = footprint_gain(utility, footprint_width)
    start_col = int(np.clip(start_col, 0, width - 1))

    value = gain[horizon - 1].copy()
    buffer = np.empty(width + 2, dtype=np.float64)
    buffer[0] = buffer[-1] = _NEG_INF
    for t in range(horizon - 2, 0, -1):
        buffer[1:-1] = value
        np.maximum(buffer[:-2], buffer[1:-1], out=value)
        np.maximum(value, buffer[2:], out=value)
        value += gain[t]

    # value now holds the optimal cost-to-go from epoch 1 at each column.
    best_col, best_value = start_col, value[start_col]
    if start_col > 0 and value[start_col - 1] > best_value:
        best_col, best_value = start_col - 1, value[start_col - 1]
    if start_col < width - 1 and value[start_col + 1] > best_value:
        best_col = start_col + 1
    if best_col < start_col:
        return ROLL_LEFT
    if best_col > start_col:
        return ROLL_RIGHT
    return NOOP


def solve_roll_trajectory(
    utility: np.ndarray,
    start_col: int | None = None,
    footprint_width: int = 3,
    max_slew: int = 1,
) -> tuple[np.ndarray, float]:
    """Exact power-unconstrained SSP over a roll lattice.

    Parameters
    ----------
    utility:
        ``(T, W)`` grid of per-cell utilities, row ``t`` being the nadir row at
        epoch ``t``.
    start_col:
        Fixed pointing state at epoch 0, or ``None`` to let the planner choose.
    footprint_width:
        Payload swath in pointing states.
    max_slew:
        Maximum change in pointing state between consecutive epochs.

    Returns
    -------
    ``(columns, value)`` where ``columns`` is the length-``T`` optimal sequence
    of pointing states and ``value`` its total collected utility.
    """
    utility = np.asarray(utility, dtype=np.float64)
    if utility.ndim != 2:
        raise ValueError("utility must be a (T, W) array.")
    horizon, width = utility.shape
    if horizon == 0:
        return np.zeros(0, dtype=np.int64), 0.0

    gain = footprint_gain(utility, footprint_width)
    offsets = np.arange(-max_slew, max_slew + 1)

    value = np.full(width, _NEG_INF)
    if start_col is None:
        value[:] = gain[0]
    else:
        start_col = int(np.clip(start_col, 0, width - 1))
        value[start_col] = gain[0, start_col]

    back = np.zeros((horizon, width), dtype=np.int8)
    for t in range(1, horizon):
        # candidates[k, c] = value of arriving at c from c - offsets[k]
        candidates = np.full((offsets.size, width), _NEG_INF)
        for k, off in enumerate(offsets):
            # previous column is c - off; valid destinations are shifted
            if off >= 0:
                candidates[k, off:] = value[: width - off] if off else value
            else:
                candidates[k, : width + off] = value[-off:]
        best = np.argmax(candidates, axis=0)
        back[t] = best
        value = candidates[best, np.arange(width)] + gain[t]

    end = int(np.argmax(value))
    total = float(value[end])
    columns = np.empty(horizon, dtype=np.int64)
    columns[-1] = end
    for t in range(horizon - 1, 0, -1):
        columns[t - 1] = columns[t] - offsets[back[t, columns[t]]]
    return columns, total


@dataclass
class RollPlan:
    """A planned roll trajectory and the command that starts it."""

    columns: np.ndarray
    value: float

    @property
    def command(self) -> int:
        """First maneuvering command along the trajectory."""
        if self.columns.size < 2:
            return NOOP
        delta = int(self.columns[1] - self.columns[0])
        if delta < 0:
            return ROLL_LEFT
        if delta > 0:
            return ROLL_RIGHT
        return NOOP


class RollPlanner:
    """Rolling-horizon SSP solver used by the ``delegate`` action of CADET-Plan.

    The planner is invoked on demand with the agent's *current* beliefs, so any
    lookahead observations acquired since the previous invocation are folded in
    before the next command is returned.
    """

    def __init__(self, footprint_width: int = 3, max_slew: int = 1) -> None:
        self.footprint_width = footprint_width
        self.max_slew = max_slew

    def plan(self, utility: np.ndarray, start_col: int) -> RollPlan:
        columns, value = solve_roll_trajectory(
            utility,
            start_col=start_col,
            footprint_width=self.footprint_width,
            max_slew=self.max_slew,
        )
        return RollPlan(columns=columns, value=value)

    def command(self, utility: np.ndarray, start_col: int) -> int:
        """Next roll command given current beliefs (fast path, no backtracking)."""
        if self.max_slew != 1:
            return self.plan(utility, start_col).command
        return first_command(utility, start_col, self.footprint_width)
