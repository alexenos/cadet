# Hypothesis: the Prop-1 noise was dropped from the metric, not from training

*2026-09-02. Written before the run that tests it, so the prediction is on the record.*

> **Confirmed by the author, 2026-09-02, after this was written and committed.**
> [Message 4](author-correspondence.md): *"During training, the Prop-1 noise was included in
> the environment. Payload success was determined by the point visibility Y(p)<τ, while the
> lookahead sensor observed the block-average cloud value Y_A."* Reading B is correct; the
> build committed on 2026-09-01 trained against an environment easier than the paper's.
>
> His generative process — `z̃_A = αZ_A + β`, `ε ~ N(0, σ_A)`, `Y_A = sigmoid(z̃_A)`,
> `Y(p) = sigmoid(z̃_A + ε)` — makes `logit Y(p) = logit Y_A + ε` hold exactly, so
> `Pr(Y(p) < τ | Y_A) = Φ((logit τ − logit Y_A)/σ_A)` with no approximation error. Drawing the
> reward truth as `Bernoulli(Φ(·))`, which is what `truth_noise` does, is distributionally
> identical to drawing ε and thresholding. The implementation is faithful.
>
> **The prediction below is left exactly as written.** Confirming the mechanism is not the
> same as reproducing the numbers, and the run is still the test of that.

Retraining on the paper's conventions closed **82.0%** of the SSP→Oracle gap against a
published **56.1%** ([run write-up](runs/2026-09-02-cadet-n32-P150-lookahead.md)). That
overshoot is larger than the 8% shortfall this whole investigation started from, and it
points the other way. This records what we think causes it, what we predict, and what would
falsify it.

## The ambiguous sentence

From the [author's reply](author-correspondence.md):

> To speed up training, our implementation instead modeled clouds at the scale of the
> lookahead sensor (observed cloud visibility) and added noise according to the model in
> Prop. 1 to simulate the true visibility. Because this noise changed the ground truth
> visibility depending on the FOV of the lookahead, we set
> `is_cloud_free = is_observed_cloud_free` for consistency across the evaluations.

Two clauses, and the second can be read as superseding the first or as scoping to the metric:

- **Reading A** — the noise was removed everywhere. `is_cloud_free ≡ is_observed_cloud_free`
  in the training reward *and* in the reported metrics. This is what is currently
  implemented.
- **Reading B** — the noise stayed in the training reward; the deterministic identity was
  applied only when computing the evaluation metrics, so that the 24 cells of the sweep are
  scored against a common world.

The problem the author describes is real either way. If the reward's ground truth is drawn by
adding Prop-1 noise, the marginal cloud-free fraction depends on `σ_A`, and `σ_A` depends on
the lookahead FOV:

| n | σ_A | cloud-free fraction with noise |
|---|---|---|
| 8 | 0.58 | 0.350 |
| 16 | 0.79 | 0.355 |
| 32 | 1.07 | 0.362 |
| 64 | 1.38 | 0.371 |
| — | — | **0.345** (truth ≡ observed, any n) |

A wider lookahead would make the world itself ~6% easier, confounding "more information" with
"less cloud" across the sweep. Redefining the *metric* fixes that without touching the
training dynamics, and is the smaller intervention.

## Why we think Reading B is right

**1. The accuracy figure is close to diagnostic.** Where a look settles the answer, high
accuracy is nearly free — the policy simply never fires at a pixel it has seen and knows is
cloudy. Our Reading-A run demonstrates this: it reached **0.990** and cut shots at
believed-obscured targets to 10 per episode, from 60. The paper reports **0.714** at the same
cell. No plausible seed or hyperparameter difference drags a competent agent from 0.99 to
0.71 in a world with no residual uncertainty. For their agent to miss 29% of its shots, its
training environment must have had some.

**2. Reading A makes Proposition 1 decorative.** If `is_cloud_free ≡ is_observed_cloud_free`
during training, then `Φ((logit τ − logit Y_A)/σ_A) > 0.5` exactly when `Y_A < τ`, which *is*
the truth condition — so the proposition is a monotone re-encoding of the observation,
calibrated to nothing. `σ_A` parameterises no real quantity, Figure 3 describes nothing the
experiments use, and the coverage-versus-fidelity trade-off across the FOV sweep disappears,
because a wider lookahead would cost nothing in certainty. A paper does not devote a
proposition, a proof and a figure to a quantity its own experiments never exercise. Under
Reading B, σ_A governs the reward, Prop 1 is the calibrated probability it claims to be, and
the trade-off the paper describes is real.

**3. The text scopes it that way.** "Added noise according to the model in Prop. 1" is stated
affirmatively as what the implementation does, not as something tried and abandoned. And
"for consistency across *the evaluations*" attaches the override to the metric. Read this
way the whole passage is coherent: a coarse field for training speed, Prop-1 noise for the
reward, and a deterministic identity so the 24 cells are measured against a common world.

## What has been measured so far

| configuration | reward truth | metric truth | targets | gap | accuracy | Ē/P̄ |
|---|---|---|---|---|---|---|
| **A** — deterministic everywhere *(current build)* | `Y_A < τ` | `Y_A < τ` | 278.2 ± 2.9 | 82.0% | 0.990 | 0.86 |
| **B′** — proxy: old weights, scored deterministically | sub-pixel pointwise | `Y_A < τ` | 267.6 ± 2.9 | 69.3% | 0.835 | 0.93 |
| **C** — noise everywhere *(original build)* | sub-pixel pointwise | sub-pixel pointwise | 225.8 | 18.5% | 0.688 | 0.93 |
| **paper** | | | **255.5** | **56.1%** | **0.714** | **1.03** |

The paper sits between B′ and C on targets and nearest C on accuracy and power. B′ is only a
proxy for Reading B: it was trained on a *sub-pixel* field, whose residuals are spatially
correlated within a pixel, rather than on a pixel-scale field with an independent per-target
draw. The conditional law is identical — `Φ((logit τ − logit Y_A)/σ_A)` either way — but the
spatial structure and the baselines differ, so B has not actually been run.

## The prediction

Reading B implemented properly — pixel-scale field, Prop-1 noise for the reward, deterministic
identity for every reported metric and both baselines — at n = 32, P̄ = 150, seed 0, 30M
timesteps:

| quantity | predicted | Reading A gave | paper |
|---|---|---|---|
| targets captured | **262 – 270** | 278.2 | 255.5 |
| gap closed | **63 – 72%** | 82.0% | 56.1% |
| capture accuracy | **0.80 – 0.87** | 0.990 | 0.714 |
| Ē/P̄ | **~0.92** | 0.86 | 1.03 |

Stated plainly: this is expected to move all three metrics toward the paper simultaneously,
which Reading A does not, but **it is not expected to fully close the gap**. A residual of
roughly 10 targets is anticipated.

### What each outcome means

- **Lands in the predicted band.** Reading B is the better interpretation, the headline
  becomes ~65–70% rather than 82%, and a smaller residual overshoot remains open.
- **Lands at ~255 / 0.714.** Reading B is confirmed outright and the reproduction is exact.
- **Lands near 278 / 0.99 again.** The noise is not reaching the reward — an implementation
  bug, not a finding.
- **Lands below ~240.** The noise is too strong somewhere, most likely σ_A being applied at
  the wrong scale.

The author has been asked directly whether the noise was dropped from the training
environment or only from the metric. His answer supersedes all of this.

## Retracted: the belief-channel hypothesis

The [run write-up](runs/2026-09-02-cadet-n32-P150-lookahead.md) and
[`shortfall-resolved.md`](shortfall-resolved.md) previously proposed that our agent's higher
accuracy might come from a richer observation — that this implementation hands the policy a
belief raster carrying the Proposition 1 probability, and the paper's may not have.

**That is wrong.** The paper specifies its observation tensor explicitly: `C = 8`, with five
target-state channels of which the fifth is *"the probability of point visibility given block
visibility according to Proposition (1)"*, plus a block-averaged cloud value channel and a
binary observation mask. That is this implementation's tensor, channel for channel. The
observation spaces match, so they cannot explain the difference, and the proposed ablation
would have measured nothing. It was checked before any compute was spent on it.
