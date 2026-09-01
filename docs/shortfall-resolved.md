# The shortfall was two modelling conventions, not a weaker policy

*2026-09-01.*

For a week this repo carried an unexplained 8% capture deficit against the paper. Six
hypotheses had been tested and refuted; the working conclusion was a "modest, distributed
shortfall" in the learned policy. That conclusion was wrong.

The paper's first author [replied](author-correspondence.md) to a message describing the
gap and named two places where his implementation differs from this one. Both were tested
directly. Together they account for the entire deficit, and one of them also retires a
"typo" this repo had wrongly attributed to the paper.

**Nothing was retrained to establish this.** Every number below comes from the two existing
30M-timestep checkpoints, re-scored under the author's conventions.

| n = 32, 20 evaluation episodes | P̄ = 150 | P̄ = 1500 |
|---|---|---|
| as reported here | 225.8 — 18.5% of the gap | 237.7 — 33.5% |
| same weights, author's conventions | **267.6 ± 2.9 — 69.3%** | **264.1 ± 2.9 — 65.2%** |
| paper | 255.5 — 56.1% | 258.8 — 60.1% |

---

## 1. Capture accuracy: the denominator is capture *actions*

The paper defines the metric in the *Metrics* section:

> Capture Accuracy: Fraction of payload captures that successfully image cloud-free targets.

The denominator is payload capture **actions**. [`env.py`](../src/cadet/env.py) divides by
targets *imaged* instead. The two differ in both directions: one payload firing can image
two targets in the same swath row, and — as the author points out — an agent with surplus
power fires on epochs where its footprint is empty.

Changing only the denominator, against the same committed 20-episode evaluations:

| n = 32 | clear ÷ imaged (here) | clear ÷ actions (paper) | paper |
|---|---|---|---|
| P̄ = 150 | 0.661 | **0.688** | 0.714 |
| P̄ = 1500 | 0.536 | **0.207** | 0.202 |

The P̄ = 1500 row lands on the published value to within 2%.

### Retraction

This repo previously argued that Table 3's 0.202 could not be reconciled: at
clear-captures ÷ targets-encountered, 258.8 captures implies 1,281 encounters, against a
provable DP upper bound of ~607. The arithmetic was right and the denominator was wrong.
The claim appeared in [`reproduction-notes.md`](reproduction-notes.md), in the
[P̄ = 1500 run write-up](runs/2026-08-30-cadet-n32-P1500-paper.md) and in the message to the
author; it is withdrawn in all three.

### One unresolved piece

Under this definition the `ssp` baseline — which fires the payload every epoch — scores
214 ÷ 3000 = 0.071, not the 35% quoted beside Figure 5. 35% is the cloud-free base rate
(0.345), which is what the ratio returns only if the baseline is charged for *scheduled*
captures rather than all 3,000 epochs:

| `ssp` accuracy | value |
|---|---|
| clear ÷ all epochs | 0.071 |
| clear ÷ targets imaged | 0.351 |
| clear ÷ epochs with a target under the footprint | 0.366 |

Either of the latter two reproduces "35%". This is flagged as an open question in the
[correspondence](author-correspondence.md) rather than guessed at.

## 2. The cloud field is modelled at lookahead-pixel scale

The author's description:

> our implementation instead modeled clouds at the scale of the lookahead sensor (observed
> cloud visibility) and added noise according to the model in Prop. 1 to simulate the true
> visibility. Because this noise changed the ground truth visibility depending on the FOV
> of the lookahead, we set `is_cloud_free = is_observed_cloud_free`.

So in the paper's implementation a lookahead observation is **decisive**: once a pixel is
measured, whether a target in it is cloud free is settled. This implementation instead
samples the field at 1.953 km sub-cells, reads ground truth pointwise, and block-averages
to produce the observation — so every capture stays a weighted coin flip even after a
perfect look. A target believed 79% clear fails 21% of the time.

That is the 8%.

### The obvious shortcut is wrong

`is_cloud_free = is_observed_cloud_free` reads like a one-line change: set
`target_visible = target_pvis_if_observed > 0.5`. It is wrong, and it fails quietly.

Block-averaging a sigmoid-transformed field is not the same as sigmoid-transforming a
block-averaged field. Reading truth off the block means of the existing fine-grained field
drops the cloud-free fraction from 0.345 to 0.315, and the baselines follow it down:

| readout on the existing fine field | cloud-free fraction | SSP | Oracle |
|---|---|---|---|
| pointwise (current) | 0.345 | 211.2 | 290.1 |
| block mean < τ (naive shortcut) | 0.315 | 190.6 | 269.9 |

*(20 episodes, no policy involved.)*

That breaks the calibration this repo trusts most — the Oracle matching the back-solved
paper baseline within 0.5% — and inflates gap closure through a depressed floor rather than
a better policy.

### The faithful version

Draw the latent GRF **on the lookahead-pixel grid** and hold it constant within a pixel.
Then `Y(p) ≡ Y_A` identically, `is_cloud_free = is_observed_cloud_free` holds by
construction rather than by assignment, and the marginal cloud-free fraction stays
`Φ(−β/α) = 0.345` for every field of view — because `Y_A` is itself a sigmoid-transformed
unit-variance Gaussian, not an average of several.

Re-scoring the two existing checkpoints this way, 20 evaluation episodes each:

| n = 32 | P̄ = 150 | P̄ = 1500 |
|---|---|---|
| captured targets | **267.6 ± 2.9** | **264.1 ± 2.9** |
| paper | 255.5 | 258.8 |
| gap closed | **69.3%** | **65.2%** |
| paper | 56.1% | 60.1% |
| capture accuracy (clear ÷ actions) | 0.835 | 0.229 |
| paper | 0.714 | 0.202 |
| SSP / Oracle | 209.3 / 293.3 | 209.3 / 293.3 |
| cloud-free fraction | 0.344 | 0.344 |
| `is_cloud_free == is_observed_cloud_free` | 1.000 | 1.000 |

### Two independent checks that this is the right convention

**The baselines get closer to the paper, not further.** Back-solved from all 24 cells of
Table 2, the paper's baselines are 209.5 / 291.5. This model gives 209.3 / 293.3, against
211.2 / 290.1 for the current one. The SSP accuracy comes out at 0.345 — exactly the base
rate the paper's "35%" refers to.

**The baselines stay independent of the lookahead width.** Under the author's model the
*granularity* of the ground truth changes with `n` (a pixel is 1.95 km at `n = 8` and
15.6 km at `n = 64`), so clear targets clump differently and the baselines could easily
have drifted. The back-solve from Table 2 assumed a single baseline pair across the whole
grid and worked to within a point, so drift would have been evidence against the model.
Measured over 8 episodes per width:

| n | pixel (km) | cloud-free fraction | SSP | Oracle |
|---|---|---|---|---|
| 8 | 1.95 | 0.344 | 213.1 | 290.8 |
| 16 | 3.91 | 0.343 | 207.4 | 290.0 |
| 32 | 7.81 | 0.344 | 207.4 | 293.4 |
| 64 | 15.62 | 0.344 | 216.8 | 292.9 |

Flat within sampling noise, and the base rate is preserved at every width.

## 3. What this does and does not overturn

**Overturned.** The headline conclusion of *Where the remaining shortfall comes from* — a
distributed capability gap in the learned policy — is wrong. So is the Table 3 typo claim.

**Still standing.** Every one of the six refuted hypotheses remains a valid negative
result, and they are the reason the modelling difference was findable at all: because
baselines, σ_A, the constraint machinery, coverage, sensing volume and hyperparameters had
each been eliminated by direct test, the remaining explanation had to be something the
paper did not state. Two specific findings survive intact and are worth keeping:

- Substituting the SSP dynamic program's trajectory *lowers* captures (237.7 → 207.5).
  Raw coverage is not the objective; steering toward believed-clear targets is.
- Forcing a lookahead on every idle epoch changes captures by exactly zero. The agent
  already gathers as much cloud information as it can act on.

**Newly open.** The re-scored numbers *overshoot* the paper by 5–13 points of gap closure,
well outside the ±2.9 sampling error. That is a discrepancy in its own right, not a
result. The likeliest explanation is that these weights were trained under stochastic
outcomes and are being evaluated under deterministic ones: a policy that learned to be
selective about which targets to shoot collects the full value of that selectivity only
when a good look guarantees a good capture. Retraining under the corrected environment is
what would settle it, and the honest expectation is that the numbers move.

**σ_A is now immaterial.** The ~6% discrepancy against Figure 3 was carried as an open
assumption. Under the paper's convention σ_A never touches the ground truth — it only sets
the scale of a monotone belief feature, and monotone rescaling cannot change which targets
a policy prefers. The geometric convention behind the 0.86× factor is still unexplained,
but it can no longer affect any result.

## 4. Recommended changes

| # | change | where | status |
|---|---|---|---|
| 1 | Sample the cloud field at lookahead-pixel resolution; add `CloudConfig.field_scale`, defaulting to the paper's convention | [`clouds.py`](../src/cadet/clouds.py), [`config.py`](../src/cadet/config.py), [`env.py`](../src/cadet/env.py) | **done** |
| 2 | Capture accuracy denominator → `n_capture_attempts` | [`env.py`](../src/cadet/env.py) | **done** |
| 3 | Same denominator for the baselines, counting scheduled captures only | [`baselines.py`](../src/cadet/baselines.py) | **done** |
| 4 | Retract the Table 3 typo claim; correct the shortfall narrative | this repo's docs | **done** |
| 5 | Demote the σ_A discrepancy to a geometry note | [`reproduction-notes.md`](reproduction-notes.md) | **done** |
| 6 | Retrain both cells under the corrected environment | — | not started |

Change 1 needs no edit to target generation: with `Y(p) ≡ Y_A`, the existing
`target_visible` computation *is* the observed-cloud-free test.

### Baselines under the implemented change

Re-measured by `scripts/verify.py` (10 episodes × 3,000 epochs) with the new defaults:

| quantity | before | now | paper |
|---|---|---|---|
| cloud-free surface fraction | 0.343 | 0.347 | ~0.34 |
| SSP baseline | 210.6 | 206.6 | ~194 (209.5 back-solved) |
| Oracle baseline | 290.0 | 289.3 | ~295 (291.5 back-solved) |
| SSP capture accuracy | 0.357 | 0.352 | 0.35 |

### A consequence worth stating plainly

Adopting the paper's convention makes **Proposition 1 non-calibrated by construction**. A
target assigned probability 0.7 is cloud free 100% of the time, because
`Φ((logit τ − logit Y_A)/σ_A) > 0.5` exactly when `Y_A < τ`, which is now the truth
condition. Ranking is untouched, so no policy behaviour changes — but the
`Prop 1 calibration` check in [`scripts/verify.py`](../scripts/verify.py) would fail at
roughly 0.4 if pointed at the environment's default field. It is instead pinned to
`field_scale="subpixel"` — the continuous field the proposition is a statement about, where
it still calibrates to 0.046 — and a new check asserts the environment's actual invariant,
that ground truth equals the observed cloud-free test on every target. Both pass. It also
means the accuracy figures in Table 3 are not measuring the quantity Proposition 1
predicts.

## 5. Reproducing the numbers here

[`scripts/rescore_under_author_model.py`](../scripts/rescore_under_author_model.py)
regenerates both §2 tables that depend on the field model — the re-scored checkpoints and
the sweep across lookahead widths. Since change 1 landed it simply evaluates against the
environment's defaults; `--field-scale subpixel` scores the same checkpoints under the model
they were actually trained on, which is the comparison the §2 tables make. The §1 accuracy
figures come from the committed evaluations under `results/`, and the naive-shortcut table
from a readout applied to the sub-pixel field.

Note that the checkpoint rows remain a *transfer* measurement: those weights were trained
against `field_scale="subpixel"`. Numbers from a policy trained under the current defaults
will differ.

```bash
python scripts/rescore_under_author_model.py --episodes 20
```

The baseline sweep runs without a checkpoint. The two policy rows need the 30M runs under
`runs/`, which are untracked and regenerable; the script skips them when absent.
