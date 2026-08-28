# Reproduction notes

Where the paper leaves a detail unspecified, this records what was chosen and why, so the
gap between the published numbers and this implementation is auditable.

## σ_A is ~6% larger than Figure 3

The paper states that σ_A is estimated by Monte Carlo from Equation (10),

```
σ²_A = k̃(0) − E_{U,V ~ Unif(A)}[ k̃(‖U − V‖) ],   k̃(r) = α² exp(−r/ℓ)
```

with the lookahead pixel side length derived from the sensor geometry:

> Using the lookahead resolution together with the ratio of the n-pixel footprint FOV to
> the 32-pixel-wide AoR, expressed as a fraction of the AoR's 250 km cross-track extent, we
> compute the side length L of a square lookahead pixel.

That reads as `L = (n/32) · 250 / 32 = n · 250/1024` km, which is what `clouds.lookahead_block_side_km`
implements. Evaluating Equation (10) at those side lengths gives:

| n | L (km) | σ_A here | σ_A in Figure 3 | ratio |
|---|---|---|---|---|
| 8 | 1.95 | 0.619 | 0.58 | 1.067 |
| 16 | 3.91 | 0.850 | 0.79 | 1.076 |
| 32 | 7.81 | 1.135 | 1.07 | 1.061 |
| 64 | 15.63 | 1.447 | 1.38 | 1.048 |

Solving for the side length that would reproduce each published value gives
`L = 1.70, 3.32, 6.73, 13.42` km — a near-constant **0.86×** our derivation across all four
FOVs. A uniform scale factor points at a small unstated geometric convention (a disc rather
than a square pixel, an effective rather than nominal cross-track extent, or a discretised
Monte-Carlo estimator that samples block *pixels* including coincident draws) rather than a
different model.

The functional form and the ordering across FOVs — the part that drives the
coverage-versus-fidelity trade-off in the results — are reproduced exactly. Since the
published values are what the paper's agents actually trained against, `--paper-sigma` (or
`VisibilityModel(..., use_paper_sigma=True)`) pins them. Both settings are exercised by the
test suite.

## Constraint cost normalisation

The paper writes the constraint as `J_c(π) ≤ μ P̄` with raw power units and reports a dual
step size of `η = 10⁻³` (Table 5). Taken literally, at `P̄ = 150` a fully active policy has
`Ĵ_c ≈ 75,000` against a threshold of 15,000, so a single dual step moves λ by ~60 — while
the reward is at most ~3 per epoch. The augmented reward would be swamped by the penalty
within one update.

This implementation therefore accumulates `c/P̄` instead of `c`. The constraint becomes
`Σ γ^t (c_t / P̄) ≤ μ`, dual steps land in the right order of magnitude for `η = 10⁻³`, and
the multiplier is comparable across the 15× span of budgets. It is also the same
normalisation the paper uses when *reporting* power (Table 1 gives Ē/P̄).

Set `PowerConfig(normalise_by_budget=False)` to accumulate raw units; `LagrangeState.budget`
then carries `P̄` so the threshold is `μ P̄` as written.

## Cloud field sub-cell resolution

The paper simulates cloud cover as a continuous GRF but rasterises the AoR at 32 columns
(7.8125 km per cell). Lookahead pixels are *finer* than an AoR cell for `n < 32`
(1.95 km at `n = 8`), so simulating the field only at AoR resolution would make the block
average equal to the pointwise value and collapse the very uncertainty Proposition 1 models.

The field is therefore sampled at `subpixels_per_cell = 4`, giving a 1.953 km sub-cell.
Lookahead pixels are then `n/8` sub-cells across — 1, 2, 4 and 8 for the four FOVs, all
integers. This is the coarsest resolution that puts every configuration on exact block
boundaries. Targets are placed at a uniformly random sub-cell within their AoR cell, since
`p_i` is a point on the surface rather than a whole cell.

## Target counts

The paper specifies 178 targets over the 364-row training world ("an expected 0.5 targets
per row" — 178/364 is actually 0.489) and 1,532 over the 3,064-row evaluation world
(exactly 0.5). `TARGET_DENSITY = 0.5` reproduces the evaluation figure exactly and the
training figure to within one target, and keeps density constant when episode length is
changed.

## Unspecified environment details

| Detail | Choice |
|---|---|
| Initial pointing state | Nadir (column 16); baselines start there too, for comparability |
| Roll at the AoR edge | Saturates rather than wrapping |
| Payload footprint at the edge | Clipped, so edge captures see fewer than 3 columns |
| Sensor-geometry channel | Payload encoded as 1.0, lookahead as 0.5, in one channel as the paper specifies |
| Visibility channel (ch. 5) | Belief for *all* live targets — Proposition 1 where measured, the prior elsewhere. This matches what the planner consumes; channels 2–4 already separate observed from unobserved |
| Lookahead pixel alignment | Blocks are aligned to the fixed world sub-grid rather than to the moving footprint. Physically the sensor's grid moves with the spacecraft, but the induced distribution of block averages is the same and the alignment lets the whole map be precomputed once per episode |
| Delegation cost | 18 units for the planner call *plus* 36 if the returned command is a slew, reading "an additional power cost" as additive |
| Episode end | `truncated=True` at the horizon (never `terminated`), the correct Gymnasium signal for a time limit |

## Baseline bounds

The paper plots the SSP and Oracle baselines but does not tabulate them; the values quoted
in the README (~194 and ~295) are back-solved from the gap-closure percentages in Table 2.
This implementation gets 210.6 and 290.0 on 3,000-epoch episodes. The Oracle matches
closely; the SSP baseline is ~8% high, consistent with the back-solved value carrying the
rounding of the Table 2 percentages.

## First trained result (2026-08-28)

One cell trained at the `quick` profile — CADET-Plan, `n = 32`, `P̄ = 150`, 2M timesteps
(6.7% of the paper's 30M), 1.63 h on an RTX 3050, evaluated over 20 episodes of 3,000
epochs:

| quantity | value |
|---|---|
| Targets captured | 225.3 |
| SSP / Oracle | 211.2 / 290.2 |
| **Gap closed** | **17.9%** |
| Capture accuracy | 0.675 (SSP: 0.342) |
| Normalised energy Ē/P̄ | 0.84 |

The paper reports 56% average for CADET-Plan and 38–83% for the moderate and high budgets,
so this is below the published range — expected at 6.7% of the training budget.

Two things worth recording:

**The gap-closure metric is very sensitive to the SSP baseline.** Scored against this
implementation's SSP (211.2) the same policy closes 17.9%; against the back-solved paper
value (194) it closes 31.0%. Since the paper does not tabulate its baselines and 194 is
itself inferred from rounded Table 2 percentages, roughly half the apparent shortfall may
be baseline calibration rather than policy quality.

**The policy delegates on every epoch and never steers itself** — `n_delegate = 3000/3000`,
`n_roll = 0`. It offloads roll planning entirely to the classical SSP solver and spends its
learned capacity on the *sensing* schedule: payload on 10.6% of epochs, lookahead on 4.8%,
converting at 0.675 versus SSP's 0.342. It captures more targets than SSP while firing the
payload one epoch in nine. That is the paper's thesis operating as claimed, but it also
means the learned *maneuvering* contributed nothing in this cell, and predicts a clear
CADET/CADET-Plan split — consistent with the paper's 42% vs 56%.

## What is not reproduced

Tables 1–4 and Figures 4–6 require all 24 configurations at 30M timesteps. One cell at
6.7% scale is a single data point, not a reproduction. `cadet.experiments` runs the grid,
writes incrementally, and regenerates Table 2/3-style summaries from it.
