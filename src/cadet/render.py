"""Visualisation of the area of regard, in the style of Figure 2.

The AoR is drawn with the spacecraft's nadir cross-track at the *bottom* row;
targets scroll from top to bottom as time advances.  Cloud pixels are tinted red
where the field has not yet been measured and grey once a lookahead observation
has covered them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import AOR_HEIGHT, AOR_WIDTH
from .env import (
    CH_CLOUD_MASK,
    CH_CLOUD_VALUE,
    CH_FOOTPRINT,
    CH_TARGETS_CLEAR,
    CH_TARGETS_OBSCURED,
    CH_TARGETS_UNOBSERVED,
    LOOKAHEAD_MARK,
    PAYLOAD_MARK,
)

if TYPE_CHECKING:  # pragma: no cover
    from .env import DynamicTaskingEnv

__all__ = ["render_ansi", "render_rgb_array", "save_episode_figure"]

_UNOBSERVED = "o"
_CLEAR = "+"
_OBSCURED = "x"


def render_ansi(env: DynamicTaskingEnv) -> str:
    """Compact text view of the current AoR, nadir last."""
    obs = env._observation()
    rows = []
    for r in range(AOR_HEIGHT - 1, -1, -1):
        line = []
        for c in range(AOR_WIDTH):
            if obs[CH_TARGETS_CLEAR, r, c] > 0:
                glyph = _CLEAR
            elif obs[CH_TARGETS_OBSCURED, r, c] > 0:
                glyph = _OBSCURED
            elif obs[CH_TARGETS_UNOBSERVED, r, c] > 0:
                glyph = _UNOBSERVED
            elif obs[CH_FOOTPRINT, r, c] == PAYLOAD_MARK:
                glyph = "#"
            elif obs[CH_FOOTPRINT, r, c] == LOOKAHEAD_MARK:
                glyph = ":"
            elif obs[CH_CLOUD_MASK, r, c] > 0:
                glyph = "-" if obs[CH_CLOUD_VALUE, r, c] >= env.cfg.clouds.tau else " "
            else:
                glyph = "."
            line.append(glyph)
        rows.append("".join(line))
    stats = env.statistics()
    header = (
        f"epoch {env.t:>5}  roll {env.roll_col:>2}  "
        f"captured {stats['captured_targets']:.0f}  "
        f"power/P {stats['normalised_power']:.2f}"
    )
    legend = (
        f"legend: {_UNOBSERVED}=unobserved {_CLEAR}=est.clear "
        f"{_OBSCURED}=est.obscured #=payload :=lookahead"
    )
    return "\n".join([header, *rows, legend])


def _rgb_canvas(env: DynamicTaskingEnv) -> np.ndarray:
    """RGB image of the AoR, nadir at the bottom."""
    obs = env._observation()
    cloud_value = obs[CH_CLOUD_VALUE]
    cloud_mask = obs[CH_CLOUD_MASK]

    image = np.zeros((AOR_HEIGHT, AOR_WIDTH, 3), dtype=np.float32)
    # Unobserved cloud is tinted red; observed cloud is greyscale by density.
    image[..., 0] = 0.45
    image[..., 1] = 0.12
    image[..., 2] = 0.12
    grey = 0.15 + 0.75 * cloud_value
    observed = cloud_mask > 0
    for channel in range(3):
        image[..., channel] = np.where(observed, grey, image[..., channel])

    # Sensor footprints.
    footprint = obs[CH_FOOTPRINT]
    look = footprint == LOOKAHEAD_MARK
    image[look] = 0.6 * image[look] + 0.4 * np.array([0.25, 0.55, 0.95])
    pay = footprint == PAYLOAD_MARK
    image[pay] = 0.4 * image[pay] + 0.6 * np.array([0.95, 0.35, 0.75])

    # Targets on top.
    image[obs[CH_TARGETS_UNOBSERVED] > 0] = (0.95, 0.95, 0.35)
    image[obs[CH_TARGETS_CLEAR] > 0] = (0.30, 0.95, 0.40)
    image[obs[CH_TARGETS_OBSCURED] > 0] = (0.20, 0.20, 0.20)

    return np.flipud(image)


def render_rgb_array(env: DynamicTaskingEnv, scale: int = 8) -> np.ndarray:
    """Upscaled ``uint8`` RGB frame suitable for video recording."""
    image = _rgb_canvas(env)
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def save_episode_figure(
    env: DynamicTaskingEnv,
    path: str,
    title: str | None = None,
) -> str:  # pragma: no cover - plotting helper
    """Write a Figure 2-style snapshot of the current AoR to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.0, 7.0), dpi=150)
    axis.imshow(_rgb_canvas(env), interpolation="nearest", aspect="auto")
    axis.set_xlabel("cross-track pointing state")
    axis.set_ylabel("along-track (epochs ahead of nadir)")
    axis.set_yticks([0, AOR_HEIGHT - 1])
    axis.set_yticklabels([str(AOR_HEIGHT - 1), "nadir"])
    axis.set_title(title or f"AoR at epoch {env.t}")

    handles = [
        mpatches.Patch(color=(0.95, 0.95, 0.35), label="unobserved target"),
        mpatches.Patch(color=(0.30, 0.95, 0.40), label="estimated clear"),
        mpatches.Patch(color=(0.20, 0.20, 0.20), label="estimated obscured"),
        mpatches.Patch(color=(0.95, 0.35, 0.75), label="payload footprint"),
        mpatches.Patch(color=(0.25, 0.55, 0.95), label="lookahead footprint"),
        mpatches.Patch(color=(0.45, 0.12, 0.12), label="cloud: unobserved"),
        mpatches.Patch(color=(0.55, 0.55, 0.55), label="cloud: observed"),
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=7,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path
