"""Generate a write-up for one training run.

Every run should leave a record that explains what was trained, what came out,
and how the training actually proceeded -- so that claims in
``docs/reproduction-notes.md`` can cite evidence rather than assert it.

This emits the factual half automatically from the run's own artifacts:

  * ``runs/<name>/metadata.json``  -- configuration actually used
  * ``runs/<name>/progress.csv``   -- per-rollout metric history
  * ``results/<file>.json``        -- evaluation against the baselines

The narrative half (the "Analysis" section) is written by hand underneath and
preserved across regeneration, so re-running this after a longer training run
updates the numbers without discarding the interpretation.

Usage:
    python scripts/run_report.py --run runs/cadet-plan_n32_P150 \\
        --results results/quick_cadetplan_n32_P150.json \\
        --out docs/runs/2026-08-28-cadet-plan-n32-P150.md
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

_ANALYSIS_MARKER = "<!-- ANALYSIS -->"


def load_history(path: Path) -> list[dict[str, float]]:
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


def at(recs: list[dict[str, float]], ts: float, key: str) -> float | None:
    """Value of ``key`` at the rollout nearest ``ts``."""
    have = [r for r in recs if key in r]
    if not have:
        return None
    return min(have, key=lambda r: abs(r["total_timesteps"] - ts))[key]


def phase_table(recs: list[dict[str, float]]) -> str:
    """Metric snapshots on a coarse grid, so the shape of the run is visible."""
    end = recs[-1]["total_timesteps"]
    step = 10 ** (len(str(int(end // 20))) - 1) * max(
        1, int(end // 20) // 10 ** (len(str(int(end // 20))) - 1)
    )
    cols = [
        ("timesteps", "total_timesteps", 0),
        ("reward", "ep_rew_mean", 2),
        ("accuracy", "capture_accuracy", 3),
        ("slack", "slack", 1),
        ("threshold", "threshold", 0),
        ("disc. cost", "discounted_cost", 0),
        ("E/P", "normalised_power", 2),
        ("entropy", "entropy_loss", 2),
    ]
    lines = [
        "| " + " | ".join(c[0] for c in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    ts = 0.0
    while ts <= end:
        r = min(recs, key=lambda x: abs(x["total_timesteps"] - ts))
        cells = []
        for _, key, prec in cols:
            v = r.get(key)
            cells.append(f"{v:.{prec}f}" if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
        ts += step
    return "\n".join(lines)


def transitions(recs: list[dict[str, float]]) -> tuple[str, dict[str, float]]:
    """Locate the curriculum boundary and the reward extrema around it."""
    rew = [(r["total_timesteps"], r["ep_rew_mean"]) for r in recs if "ep_rew_mean" in r]
    slack1 = next(
        (r["total_timesteps"] for r in recs if r.get("slack") is not None and r["slack"] <= 1.0),
        None,
    )
    window = slack1 or recs[-1]["total_timesteps"]
    early = [p for p in rew if p[0] <= window]
    peak = max(early, key=lambda t: t[1]) if early else (0.0, 0.0)
    post = [p for p in early if p[0] > peak[0]]
    trough = min(post, key=lambda t: t[1]) if post else peak
    facts = {
        "peak_ts": peak[0],
        "peak_rew": peak[1],
        "trough_ts": trough[0],
        "trough_rew": trough[1],
        "slack1_ts": slack1 or float("nan"),
        "final_ts": rew[-1][0],
        "final_rew": rew[-1][1],
    }
    drop = (1 - trough[1] / peak[1]) * 100 if peak[1] else 0.0
    rows = [
        ("Peak before the squeeze", peak[0], peak[1], at(recs, peak[0], "normalised_power")),
        ("Trough during the taper", trough[0], trough[1], at(recs, trough[0], "normalised_power")),
        (
            "Slack reaches 1",
            slack1 or 0,
            at(recs, slack1 or 0, "ep_rew_mean"),
            at(recs, slack1 or 0, "normalised_power"),
        ),
        ("Final", rew[-1][0], rew[-1][1], at(recs, rew[-1][0], "normalised_power")),
    ]
    md = ["| moment | timesteps | reward | Ē/P̄ |", "|---|---|---|---|"]
    for label, ts, rw, pw in rows:
        md.append(
            f"| {label} | {ts:,.0f} | {rw:.2f} | {pw:.2f} |"
            if pw is not None
            else f"| {label} | {ts:,.0f} | {rw:.2f} | — |"
        )
    md.append("")
    md.append(f"Drop from peak to trough: **{drop:.0f}%**.")
    return "\n".join(md), facts


def render(
    meta: dict, result: dict | None, recs: list[dict[str, float]], run_dir: Path, figure: str | None
) -> str:
    cfg = meta.get("train_config", {})
    trans_md, _ = transitions(recs)
    wall = (recs[-1].get("time_elapsed", 0) or 0) / 3600

    out = [
        f"# {meta.get('controller')} · n={meta.get('lookahead_width')} · "
        f"P̄={meta.get('budget'):.0f}",
        "",
        f"*Generated {date.today().isoformat()} by `scripts/run_report.py`.*",
        "",
        "## Configuration",
        "",
        "| | |",
        "|---|---|",
        f"| Controller | `{meta.get('controller')}` |",
        f"| Lookahead width | {meta.get('lookahead_width')} |",
        f"| Power budget P̄ | {meta.get('budget')} |",
        f"| Timesteps | {cfg.get('total_timesteps'):,} |",
        f"| Parallel envs | {cfg.get('n_envs')} |",
        f"| Seed | {cfg.get('seed')} |",
        f"| Device | {cfg.get('device')} |",
        f"| σ_A | {meta.get('sigma_a'):.4f} |",
        f"| Policy parameters | {meta.get('policy_parameters'):,} |",
        f"| Curriculum | slack {cfg.get('initial_slack')} held to "
        f"{cfg.get('warmup_steps'):,}, tapered over {cfg.get('taper_steps'):,} |",
        f"| Wall clock | {wall:.2f} h |",
        "",
    ]

    if result:
        ssp, orc = result["ssp_captured"], result["oracle_captured"]
        out += [
            "## Result",
            "",
            "| quantity | value | reference |",
            "|---|---|---|",
            f"| Targets captured | **{result['captured_targets']:.1f}** | "
            f"SSP {ssp:.1f} · Oracle {orc:.1f} |",
            f"| Gap closed | **{result['gap_closed_pct']:.1f}%** | paper: 56% avg (CADET-Plan) |",
            f"| Capture accuracy | {result['capture_accuracy']:.3f} | SSP 0.342 |",
            f"| Normalised energy Ē/P̄ | {result['normalised_power']:.2f} | "
            f"{'within budget' if result['normalised_power'] <= 1 else 'OVER BUDGET'} |",
            f"| Evaluation | {result['n_episodes']} episodes × "
            f"{result['episode_length']:,} epochs | |",
            "",
            "Action usage per evaluation episode: "
            f"payload {result.get('n_payload', 0):.0f}, "
            f"lookahead {result.get('n_lookahead', 0):.0f}, "
            f"delegate {result.get('n_delegate', 0):.0f}, "
            f"explicit rolls {result.get('n_roll', 0):.0f}.",
            "",
        ]

    out += ["## Training dynamics", "", trans_md, "", phase_table(recs), ""]
    if figure:
        out += [f"![training diagnostics]({figure})", ""]
    out += [
        "## Artifacts",
        "",
        f"Committed with this write-up: see `assets/`. Metric history is {len(recs):,} rollouts.",
        "",
        f"Not committed (regenerable, and `model.zip` is large): `{run_dir.as_posix()}/`.",
        "",
        "---",
        "",
        _ANALYSIS_MARKER,
        "",
        "## Analysis",
        "",
        "_Written by hand; preserved when this report is regenerated._",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument(
        "--history", type=Path, default=None, help="metric CSV (default: <run>/progress.csv)"
    )
    ap.add_argument("--figure", default=None, help="relative path to the diagnostics image")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    hist = args.history or (args.run / "progress.csv")
    recs = load_history(hist)
    if not recs:
        raise SystemExit(f"no rollouts parsed from {hist}")

    result = None
    if args.results and args.results.exists():
        payload = json.loads(args.results.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        result = next(
            (
                r
                for r in rows
                if r.get("controller") == meta.get("controller")
                and r.get("lookahead_width") == meta.get("lookahead_width")
            ),
            rows[0] if rows else None,
        )

    body = render(meta, result, recs, args.run, args.figure)

    # Preserve hand-written analysis across regeneration.
    if args.out.exists():
        prev = args.out.read_text(encoding="utf-8")
        if _ANALYSIS_MARKER in prev:
            kept = prev.split(_ANALYSIS_MARKER, 1)[1]
            body = body.split(_ANALYSIS_MARKER, 1)[0] + _ANALYSIS_MARKER + kept

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(f"wrote {args.out} ({len(recs):,} rollouts)")


if __name__ == "__main__":
    main()
