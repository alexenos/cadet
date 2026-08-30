# Training runs

One write-up per training run. The factual sections are generated from the run's own
artifacts by [`scripts/run_report.py`](../../scripts/run_report.py); the *Analysis* section
is written by hand and preserved when a report is regenerated.

Claims in [`../reproduction-notes.md`](../reproduction-notes.md) cite these where evidence
exists, rather than asserting behaviour.

| date | run | scale | gap closed | Ē/P̄ | notes |
|---|---|---|---|---|---|
| 2026-08-28 | [cadet-plan n=32 P̄=150](2026-08-28-cadet-plan-n32-P150.md) | quick (2M) | 17.9% | 0.84 | First trained cell. Curriculum collapse and recovery; delegates every epoch. |
| 2026-08-28 | [cadet n=32 P̄=150](2026-08-28-cadet-n32-P150.md) | quick, stopped at 890k | — | 5.24 | **Negative result.** Policy saturated at 207k, gradients exactly zero after; λ diverged to 46. |
| 2026-08-29 | [cadet n=32 P̄=150](2026-08-29-cadet-n32-P150-paper.md) | **paper (30M)** | 18.5% | 0.93 | Trains cleanly — confirms the quick-scale collapse was a curriculum artifact. Short of the paper's 42%. |
| 2026-08-30 | [cadet n=32 P̄=1500](2026-08-30-cadet-n32-P1500-paper.md) | paper (30M) | 33.5% | 0.24 | Diagnostic: constraint cannot bind. Isolates the root cause — learned trajectory covers 75% of the DP optimum. |

Evidence for each write-up — the diagnostics figure, a thinned metric history, and the
evaluation result — is committed alongside it in [`assets/`](assets/). The run directory
under `runs/` (models, full `progress.csv`) is regenerable and stays untracked.

Histories are thinned to 200 rows (~40 KB). A full per-rollout history is ~390 KB for a 2M
run and ~5.7 MB at the paper's 30M, so committing them for a 24-cell sweep would add over
100 MB of poorly-compressing churn. 200 rows preserve the shape of every curve; use
`--csv-rows 0` if you genuinely need the full history for a specific run.

## Adding a run

```bash
python scripts/run_report.py \
    --run runs/<name> \
    --results results/<file>.json \
    --figure assets/<date>-<name>.png \
    --out docs/runs/<date>-<name>.md
```

Then copy the figure, history CSV, and result JSON into `assets/` using the same
`<date>-<name>` stem, write the *Analysis* section, add a row above, and cite it from the
reproduction notes wherever it justifies a choice or explains an observation.

`python scripts/check_docs.py` verifies that every file referenced from `docs/` exists **and
is tracked by git** — a plain existence check passes on untracked files that are missing
from a fresh clone.
