# Draft email to the author

**Subject:** Reproducing CADET — an LLM-only implementation, and a gap I can't explain

---

Hi Nick,

I really enjoyed the CADET paper — the framing of sensing and maneuvering as one
constrained decision problem, rather than the usual look-then-replan decomposition, is a
genuinely nice idea, and Proposition 1 is an elegant piece of work.

It also struck me as a good test of something I've been curious about: whether an LLM can
implement a research paper and reproduce its results working **only** from the PDF, with no
access to your code. So I had Claude build the whole thing from scratch — environment,
cloud model, both controllers, both baselines, the evaluation protocol. The result is here:

**https://github.com/alexenos/cadet**

I wanted to share what came out, because the modelling reproduces cleanly but the learned
results don't, and I've run out of hypotheses.

## What reproduces

The environment matches your specification closely enough that I think it's right:

| Quantity | Paper | This implementation |
|---|---|---|
| Cloud-free surface fraction | ~0.34 | 0.347 |
| SSP baseline | 209.5 (back-solved from Table 2) | 211.2 |
| Oracle baseline | 291.5 (back-solved) | 290.2 |
| Policy parameters | ~2.2M | 2,157,224 |
| σ_A for n = 8/16/32/64 | 0.58 / 0.79 / 1.07 / 1.38 | 0.62 / 0.85 / 1.14 / 1.45 |

The Oracle matching within 0.5% is the check I trust most, since it exercises target
generation, geometry, agility limits and the dynamic program end to end. I also back-solved
your baselines from all 24 cells of Table 2 — scoring your published target counts against
my baselines reproduces your gap percentages to within about a point across the grid
(56.1 vs 56.1, 64.3 vs 64.1, 83.5 vs 82.5).

There's a repeatable check in the repo — `python scripts/verify.py` runs 18 tests in ~30
seconds, including the SSP dynamic program against brute-force enumeration and Proposition 1
against simulated fields for calibration rather than just monotonicity.

## What doesn't

Two cells trained at full 30M-timestep scale, single seed:

| Cell | Mine | Paper |
|---|---|---|
| CADET, n=32, P̄=150 | 225.8 targets, **18.5%** gap closed | 255.5, **56.1%** |
| CADET, n=32, P̄=1500 | 237.7 targets, **33.5%** gap closed | 258.8, **60.1%** |

Both satisfy the power constraint (Ē/P̄ of 0.93 and 0.24). In target terms I'm at 88% and
92% of your numbers — but because the SSP→Oracle window is only ~79 targets wide, an 8%
capture deficit becomes ~27 points of gap closure.

## What I ruled out

I tested each of these directly rather than reasoning about them:

- **σ_A being 6% high.** It enters Proposition 1 as a monotone scaling inside Φ, so it
  changes calibration but never the *ordering* of targets. Ranking under 1.135 vs 1.07 gives
  identical orderings and identical accuracy at every selection threshold.
- **The primal–dual machinery.** At P̄=1500 the maximum single-epoch spend is 786 units
  (0.52× budget), so the constraint can't bind — λ stayed at exactly 0 for all 30M steps.
  Still short.
- **Trajectory coverage.** My policy covers 455 targets/episode against the DP's 607. But
  substituting the DP's trajectory *lowered* captures to 207.5 — the DP maximises raw target
  count while the learned policy steers toward believed-clear targets, which your Oracle
  confirms is correct (it covers less ground than SSP, 373.7 vs 607, and captures far more).
- **Under-sensing.** Forcing every idle epoch to take a lookahead observation more than
  doubled lookahead actions and changed captures by exactly zero.
- **Under-shooting.** The policy already images 97% of the targets that pass beneath it.
- **Hyperparameters.** All 14 values in Table 5 match exactly.

So the behaviour looks qualitatively right on every axis I can measure — it just captures
about 8% fewer targets.

## Assumptions I had to make

Where the paper left something unspecified, these were my choices, and any of them could be
where I've diverged:

1. **Number of parallel environments.** Table 5 gives "Steps per Update 128" but not how many
   envs. I used 8, so each PPO update sees 1,024 steps. This is my prime suspect.
2. **Constraint normalisation.** I accumulate `c/P̄` rather than raw units. Taken literally,
   at P̄=150 a fully active policy gives Ĵ_c ≈ 75,000 against a threshold of 15,000, so with
   η = 10⁻³ a single dual step moves λ by ~60 while rewards are ~3 per epoch. Normalising
   keeps λ in a sane range (it peaked at 0.14).
3. **Lookahead pixel size.** I read the geometry as L = (n/32)·250/32 km. This gives σ_A
   about 6% above Figure 3 — consistently, across all four widths, suggesting a small
   geometric convention I've guessed wrong.
4. **Delegation cost.** 18 units for the planner call, plus 36 if the returned command is a
   slew, reading "an additional power cost" as additive.
5. `max_grad_norm` = 0.5, CNN feature dimension 256, λ initialised at 0.

## One thing that may be a typo

Table 3 reports capture accuracy 0.202 for CADET at P̄=1500, n=32. Under the definition that
matches your SSP figure (35%, attributed to average cloud coverage — i.e. clear captures ÷
targets encountered), 258.8 captures at 0.202 implies 1,281 targets encountered. But the SSP
dynamic program maximises encounters and reaches only ~607 while imaging on every epoch, so
that would need to be 2.1× a provable upper bound. I suspect the denominator differs there,
or it's a transcription slip — either way I stopped using that row as a target.

## What I'd love your take on

If you have any interest in poking at it, I'd be grateful for two things:

1. **A quick look at the implementation** — particularly the environment and the constraint
   handling. `docs/reproduction-notes.md` records every interpretation I had to make, and
   `docs/runs/` has a write-up per training run with full diagnostics.
2. **Any intuition about the gap.** Does 8% fewer captures suggest something specific to you?
   And do you remember how many parallel environments you trained with?

No obligation at all — I know how these requests go. Mostly I wanted to share it, because
the parts that *did* reproduce came out of the paper alone, which I thought was a decent
result in itself, and says something good about how clearly the paper is written.

Best,
Dax
