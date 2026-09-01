"""Independent verification of the CADET implementation.

Runs checks that do not require a trained policy, so the modelling can be
audited before spending GPU time. Three groups:

  A. Values the paper states, recomputed here from the models alone.
  B. Internal correctness -- claims checked against ground truth we can
     compute independently (brute force, simulated fields).
  C. Negative controls -- orderings and degenerate limits that must hold
     if the pieces are wired together correctly.

Usage:
    python scripts/verify.py            # ~2 minutes
    python scripts/verify.py --quick    # ~30 seconds, fewer episodes
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import time

import numpy as np

from cadet.baselines import evaluate_baselines, run_baseline
from cadet.clouds import CloudConfig, sample_cloud_field, sigma_for_lookahead_width
from cadet.config import AOR_WIDTH, make_env_config
from cadet.env import (
    MOVE_LEFT,
    MOVE_NOOP,
    MOVE_RIGHT,
    SENSE_PAYLOAD,
    DynamicTaskingEnv,
)
from cadet.planner import solve_roll_trajectory
from cadet.visibility import VisibilityModel

# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------

_ROWS: list[tuple[str, str, str, str, bool | None]] = []


def check(group: str, name: str, got, expect, ok: bool | None) -> None:
    _ROWS.append((group, name, str(got), str(expect), ok))
    mark = "    " if ok is None else (" OK " if ok else "FAIL")
    print(f"  [{mark}] {name:<44} got {got!s:<22} expect {expect}")


def banner(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# A. paper values, recomputed
# ---------------------------------------------------------------------------


def group_a(quick: bool) -> None:
    banner("A. Paper values, recomputed from the models (no training)")

    # Cloud-free surface fraction. Paper: ~0.34 (matches ~67% global mean
    # cloud cover). Measured by sampling fields and counting clear subpixels.
    cfg = CloudConfig()
    rng = np.random.default_rng(0)
    n_fields = 4 if quick else 14
    frac = float(
        np.mean(
            [
                np.mean(sample_cloud_field((600, 32), cfg, rng).values < cfg.tau)
                for _ in range(n_fields)
            ]
        )
    )
    # Closed form is Phi(-beta/alpha) = 0.345. The field is strongly correlated,
    # so the effective sample size is far below the pixel count -- allow 0.03.
    check(
        "A", "cloud-free surface fraction", f"{frac:.3f}", "~0.34 (paper)", close(frac, 0.345, 0.03)
    )

    # sigma_A vs Figure 3. These come from a Monte-Carlo integral over the
    # lookahead block, so they are a genuine recomputation of the paper's curve.
    paper_sigma = {8: 0.58, 16: 0.79, 32: 1.07, 64: 1.38}
    n_samples = 200_000 if quick else 2_000_000
    for n, want in paper_sigma.items():
        got = sigma_for_lookahead_width(n, n_samples=n_samples)
        # Known ~6% high; see docs/reproduction-notes.md. Flag >12% as failure.
        check("A", f"sigma_A  n={n:<3}", f"{got:.3f}", f"{want} (Fig 3)", got / want < 1.12)

    # Policy parameter count. Paper says "~2.2M". Count the FULL actor-critic
    # policy as the repo actually builds it -- policy_kwargs() sets net_arch=[],
    # so the heads are single linear layers on the shared encoder. Counting the
    # encoder alone, or letting SB3 apply its default net_arch, both give a
    # number for a network this codebase never instantiates.
    from stable_baselines3 import PPO

    from cadet.policies import count_parameters, policy_kwargs

    env_pp = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet-plan"))
    model_pp = PPO("CnnPolicy", env_pp, policy_kwargs=policy_kwargs(256), device="cpu", verbose=0)
    n_par = count_parameters(model_pp.policy)
    check("A", "policy parameters", f"{n_par:,}", "~2.2M (paper)", 2.0e6 < n_par < 2.4e6)

    # Baselines over the paper's 3000-epoch episodes.
    n_ep = 3 if quick else 10
    ep_len = 800 if quick else 3000
    cfg_env = make_env_config(32, 150.0, "cadet", episode_length=ep_len)
    recs = evaluate_baselines(cfg_env, seeds=range(n_ep))
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(str(r["controller"]), []).append(r)
    for label, paper_val, tol in (("ssp", 194.0, 60.0), ("oracle", 295.0, 60.0)):
        rows = by.get(label)
        if not rows:
            continue
        scaled = [x["captured_targets"] * 3000.0 / ep_len for x in rows]
        got = float(np.mean(scaled))
        check(
            "A",
            f"{label} baseline (per 3000 epochs)",
            f"{got:.1f}",
            f"~{paper_val:.0f} (paper)",
            close(got, paper_val, tol),
        )
    rows = by.get("ssp")
    if rows:
        acc = float(np.mean([x["capture_accuracy"] for x in rows]))
        check("A", "ssp capture accuracy", f"{acc:.3f}", "0.35 (paper)", close(acc, 0.35, 0.06))


# ---------------------------------------------------------------------------
# B. internal correctness against independently computed ground truth
# ---------------------------------------------------------------------------


def group_b(quick: bool) -> None:
    banner("B. Internal correctness (checked against independent ground truth)")

    # The SSP planner claims to be EXACT, not heuristic. Verify by brute-force
    # enumerating every agility-feasible path on small instances.
    rng = np.random.default_rng(1)
    n_inst, T, W = (20, 5, 5) if quick else (60, 6, 6)
    mismatches = 0
    for _ in range(n_inst):
        util = rng.integers(0, 4, size=(T, W)).astype(float)
        _, dp_value = solve_roll_trajectory(util, start_col=W // 2, footprint_width=1, max_slew=1)
        best = -np.inf
        for path in itertools.product((-1, 0, 1), repeat=T - 1):
            col, total = W // 2, util[0, W // 2]
            ok = True
            for t, mv in enumerate(path, start=1):
                col += mv
                if not (0 <= col < W):
                    ok = False
                    break
                total += util[t, col]
            if ok:
                best = max(best, total)
        mismatches += int(abs(best - dp_value) > 1e-9)
    check(
        "B",
        "DP == brute force (exactness)",
        f"{n_inst - mismatches}/{n_inst} exact",
        f"{n_inst}/{n_inst}",
        mismatches == 0,
    )

    # Proposition 1 claims a CALIBRATED probability, not merely a monotone
    # score. Bin predictions and compare to realised frequencies.
    # Proposition 1 is a statement about a *continuous* field: given the block
    # average Y_A, this is the conditional law of the pointwise value.  Checking
    # it therefore requires the sub-pixel field, not the environment's default
    # lookahead-scale one -- under which Y(p) == Y_A and the proposition is
    # degenerate rather than calibrated (see the truth-vs-observed check below).
    cfg = dataclasses.replace(CloudConfig(), field_scale="subpixel")
    rng = np.random.default_rng(2)
    model = VisibilityModel(lookahead_width=32)
    block = 4  # one n=32 lookahead pixel, in sub-cell units
    preds, truths = [], []
    for _ in range(2 if quick else 6):
        values = sample_cloud_field((600, 32), cfg, rng).values
        h, w = values.shape
        means = values.reshape(h // block, block, w // block, block).mean(axis=(1, 3))
        upsampled = np.repeat(np.repeat(means, block, axis=0), block, axis=1)
        preds.append(np.asarray(model.probability(upsampled.ravel())).ravel())
        truths.append((values.ravel() < cfg.tau).astype(float))
    p = np.concatenate(preds)
    y = np.concatenate(truths)
    edges = np.linspace(0, 1, 6)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if m.sum() < 500:
            continue
        worst = max(worst, abs(float(p[m].mean()) - float(y[m].mean())))
    check(
        "B",
        "Prop 1 calibration, sub-pixel field",
        f"{worst:.3f}",
        "< 0.12",
        worst < 0.12,
    )

    # The environment's default convention, which is the paper's: the field is
    # drawn at lookahead-pixel scale, so observing a pixel settles whether a
    # target in it is cloud free.  This must hold exactly, on every target.
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=400))
    worst_agreement = 1.0
    for seed in range(3 if quick else 8):
        env.reset(seed=seed)
        observed_clear = env.target_pvis_if_observed > 0.5
        agreement = float(np.mean(env.target_visible == observed_clear))
        worst_agreement = min(worst_agreement, agreement)
    check(
        "B",
        "truth == observed cloud-free test",
        f"{worst_agreement:.3f}",
        "1.000 (paper convention)",
        worst_agreement == 1.0,
    )

    # The oracle must beat the cloud-blind plan on the field that actually
    # realised -- otherwise cloud knowledge is not being used.
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=400))
    wins = 0
    n_ep = 5 if quick else 15
    for seed in range(n_ep):
        env.reset(seed=seed)
        s = run_baseline(env, "ssp").captured_targets
        o = run_baseline(env, "oracle").captured_targets
        wins += int(o >= s)
    check("B", "oracle >= ssp on realised field", f"{wins}/{n_ep}", f"{n_ep}/{n_ep}", wins == n_ep)

    # The baselines are scored OFFLINE -- run_baseline reads target visibility
    # directly and never calls env.step(). A trained policy is scored ONLINE,
    # by accumulating env.step() rewards. Every gap-closure number compares the
    # two, so the two accountings must agree exactly or the comparison is
    # apples-to-oranges. Replay each baseline's own trajectory through step()
    # and require the rewards to match to the target.
    agree = 0
    n_ep = 5 if quick else 12
    trials = 0
    for name in ("ssp", "oracle"):
        for seed in range(n_ep):
            env.reset(seed=seed)
            res = run_baseline(env, name, start_col=AOR_WIDTH // 2)
            cols = np.asarray(res.columns, dtype=int)

            env.reset(seed=seed)
            env.roll_col = int(cols[0])
            total = 0.0
            for t in range(cols.size):
                nxt = int(cols[t + 1]) if t + 1 < cols.size else int(cols[t])
                d = nxt - int(cols[t])
                mv = MOVE_NOOP if d == 0 else (MOVE_LEFT if d < 0 else MOVE_RIGHT)
                _, r, term, trunc, _ = env.step((mv, SENSE_PAYLOAD))
                total += r
                if term or trunc:
                    break
            trials += 1
            agree += int(abs(total - res.captured_targets) < 1e-9)
    check(
        "B",
        "offline baseline == online env.step reward",
        f"{agree}/{trials}",
        f"{trials}/{trials}",
        agree == trials,
    )


# ---------------------------------------------------------------------------
# C. negative controls and degenerate limits
# ---------------------------------------------------------------------------


def group_c(quick: bool) -> None:
    banner("C. Negative controls (these SHOULD hold if wiring is correct)")

    # A random policy must score well below SSP. If it does not, the task is
    # trivial and the whole comparison is meaningless.
    env = DynamicTaskingEnv(make_env_config(32, 150.0, "cadet", episode_length=400))
    n_ep = 3 if quick else 8
    rand_scores, ssp_scores, oracle_scores = [], [], []
    for seed in range(n_ep):
        env.reset(seed=seed)
        ssp_scores.append(run_baseline(env, "ssp").captured_targets)
        oracle_scores.append(run_baseline(env, "oracle").captured_targets)
        env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            a = env.action_space.sample()
            _, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rand_scores.append(total)
    r_, s_, o_ = (float(np.mean(x)) for x in (rand_scores, ssp_scores, oracle_scores))
    check(
        "C",
        "random < ssp < oracle",
        f"{r_:.1f} < {s_:.1f} < {o_:.1f}",
        "strict ordering",
        r_ < s_ < o_,
    )
    # Expressed as a ratio so the check is invariant to episode length --
    # an absolute target count would depend on how long the episode ran.
    ratio = o_ / s_ if s_ else float("nan")
    check(
        "C", "ssp->oracle gap is non-trivial", f"oracle/ssp = {ratio:.2f}", "> 1.15", ratio > 1.15
    )

    # Degenerate limit: with sigma_A -> 0 the block average is perfectly
    # informative, so visibility must collapse to a hard threshold at tau.
    sharp = VisibilityModel(lookahead_width=32, sigma_override=1e-6)
    lo = float(np.asarray(sharp.probability(0.2)))
    hi = float(np.asarray(sharp.probability(0.8)))
    check(
        "C",
        "sigma_A -> 0 gives hard threshold",
        f"{lo:.3f} / {hi:.3f}",
        "~1.0 / ~0.0",
        lo > 0.99 and hi < 0.01,
    )

    # Prior with no measurement must equal the unconditional clear fraction.
    prior = float(np.asarray(VisibilityModel(lookahead_width=32).probability(0.5)))
    check("C", "no-information prior", f"{prior:.3f}", "0.50 at tau", close(prior, 0.5, 0.01))

    # A wider lookahead must blur more, i.e. sigma_A increases with n.
    sig = [sigma_for_lookahead_width(n, n_samples=200_000) for n in (8, 16, 32, 64)]
    check(
        "C",
        "sigma_A increases with FOV",
        " < ".join(f"{s:.2f}" for s in sig),
        "monotone increasing",
        all(a < b for a, b in zip(sig, sig[1:], strict=False)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer samples, ~30s")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("CADET verification -- no trained policy required")
    print("=" * 78)

    group_a(args.quick)
    group_b(args.quick)
    group_c(args.quick)

    scored = [r for r in _ROWS if r[4] is not None]
    failed = [r for r in scored if not r[4]]
    print("\n" + "=" * 78)
    print(f"{len(scored) - len(failed)}/{len(scored)} checks passed in {time.time() - t0:.0f}s")
    if failed:
        print("\nFAILED:")
        for _, name, got, expect, _ in failed:
            print(f"  - {name}: got {got}, expected {expect}")
    print("=" * 78)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
