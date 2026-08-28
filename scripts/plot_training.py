"""Plot how a training run actually went.

Stable-Baselines3 prints a metrics table every rollout. Those tables are the
only record of the run's trajectory unless CSV or tensorboard logging is on, so
this script reads either source:

  * ``runs/<name>/progress.csv``  -- written by ``cadet.train`` (preferred)
  * a captured stdout log of the SB3 tables (``--log run.log``)

Produces a six-panel figure covering reward, capture behaviour, the power
constraint, the dual variable, policy entropy, and value-function fit, plus a
tidy CSV of every metric so you can inspect the numbers directly.

Usage:
    python scripts/plot_training.py --run runs/cadet-plan_n32_P150
    python scripts/plot_training.py --log train.log --out figures/ --csv history.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A row is "|    key    |   value   |"; a section header is "| name/ |   |".
_ROW = re.compile(r"^\|\s+(?P<key>[^|/]+?)\s+\|\s+(?P<value>[^|]*?)\s+\|$")


def parse_sb3_log(path: Path) -> list[dict[str, float]]:
    """Parse SB3's printed tables into one record per rollout."""
    records: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if set(line.strip()) == {"-"} and line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        m = _ROW.match(line)
        if not m:
            continue
        key, value = m.group("key").strip(), m.group("value").strip()
        if not value:  # section header
            continue
        try:
            current[key] = float(value)
        except ValueError:
            pass
    if current:
        records.append(current)
    # Blocks without a timestep are section fragments, not rollouts.
    return [r for r in records if "total_timesteps" in r]


def read_csv(path: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec: dict[str, float] = {}
            for k, v in row.items():
                if v in (None, ""):
                    continue
                try:
                    rec[k.split("/")[-1]] = float(v)
                except ValueError:
                    pass
            if "total_timesteps" in rec:
                out.append(rec)
    return out


def series(recs: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in recs:
        if key in r:
            xs.append(r["total_timesteps"])
            ys.append(r[key])
    return xs, ys


def _panel(ax, recs, key, title, ylabel, color="C0"):
    x, y = series(recs, key)
    if not x:
        ax.text(0.5, 0.5, f"no {key}", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(x, y, color=color, lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def make_figure(recs: list[dict[str, float]], out: Path, refs: dict[str, float]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    # 1. Reward, against the baselines the run is ultimately scored against.
    # ep_rew_mean is per TRAINING episode (300 epochs); the baselines are per
    # 3000-epoch evaluation episode, so scale them down by 10 to compare.
    ax = _panel(axes[0][0], recs, "ep_rew_mean", "Reward per training episode", "targets")
    for name, val, style in (
        ("SSP/10", refs["ssp"] / 10, "--"),
        ("Oracle/10", refs["oracle"] / 10, ":"),
    ):
        ax.axhline(val, ls=style, color="grey", lw=1)
        ax.annotate(name, (0.01, val), xycoords=("axes fraction", "data"), fontsize=7, color="grey")

    # 2. Capture accuracy -- the mechanism the paper claims.
    ax = _panel(
        axes[0][1], recs, "capture_accuracy", "Capture accuracy", "fraction clear", color="C2"
    )
    ax.axhline(refs["ssp_accuracy"], ls="--", color="grey", lw=1)
    ax.annotate(
        "SSP",
        (0.01, refs["ssp_accuracy"]),
        xycoords=("axes fraction", "data"),
        fontsize=7,
        color="grey",
    )

    # 3. The constraint: discounted cost against its moving threshold.
    ax = axes[1][0]
    x, y = series(recs, "discounted_cost")
    if x:
        ax.plot(x, y, color="C3", lw=1.2, label="discounted cost")
    xt, yt = series(recs, "threshold")
    if xt:
        ax.plot(xt, yt, color="k", ls="--", lw=1, label="threshold (slack x mu)")
    ax.set_title("Power constraint", fontsize=10)
    ax.set_ylabel("discounted cost", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 4. The dual variable and the slack curriculum.
    ax = axes[1][1]
    x, y = series(recs, "lambda")
    if x:
        ax.plot(x, y, color="C4", lw=1.2, label="lambda")
    ax.set_title("Dual variable and slack curriculum", fontsize=10)
    ax.set_ylabel("lambda", fontsize=8)
    ax.grid(alpha=0.3)
    xs, ys = series(recs, "slack")
    if xs:
        ax2 = ax.twinx()
        ax2.plot(xs, ys, color="C1", lw=1, ls=":", label="slack")
        ax2.set_ylabel("slack", fontsize=8, color="C1")
    ax.legend(fontsize=7, loc="upper left")

    # 5. Entropy -- exploration collapse shows up here.
    _panel(axes[2][0], recs, "entropy_loss", "Policy entropy", "entropy loss", color="C5")
    axes[2][0].set_xlabel("timesteps")

    # 6. Value-function fit.
    ax = axes[2][1]
    x, y = series(recs, "explained_variance")
    if x:
        ax.plot(x, y, color="C6", lw=1.2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Value function explained variance", fontsize=10)
    ax.set_ylabel("explained variance", fontsize=8)
    ax.set_xlabel("timesteps")
    ax.grid(alpha=0.3)

    fig.suptitle("CADET training diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def write_csv(recs: list[dict[str, float]], path: Path) -> None:
    keys = sorted({k for r in recs for k in r})
    keys.remove("total_timesteps")
    keys.insert(0, "total_timesteps")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {path}  ({len(recs)} rows x {len(keys)} metrics)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="run directory containing progress.csv")
    src.add_argument("--log", type=Path, help="captured SB3 stdout log")
    ap.add_argument("--out", type=Path, default=Path("figures/training.png"))
    ap.add_argument("--csv", type=Path, default=None, help="also write a tidy CSV")
    ap.add_argument("--ssp", type=float, default=211.2)
    ap.add_argument("--oracle", type=float, default=290.15)
    ap.add_argument("--ssp-accuracy", type=float, default=0.342)
    args = ap.parse_args()

    if args.run:
        csv_path = args.run / "progress.csv"
        if not csv_path.exists():
            raise SystemExit(
                f"{csv_path} not found. Either the run predates CSV logging "
                f"(use --log with a captured stdout log) or training did not finish."
            )
        recs = read_csv(csv_path)
    else:
        recs = parse_sb3_log(args.log)

    if not recs:
        raise SystemExit("no rollout records parsed")
    print(f"parsed {len(recs)} rollouts, {recs[-1]['total_timesteps']:.0f} timesteps")

    if args.csv:
        write_csv(recs, args.csv)
    make_figure(
        recs,
        args.out,
        {"ssp": args.ssp, "oracle": args.oracle, "ssp_accuracy": args.ssp_accuracy},
    )


if __name__ == "__main__":
    main()
