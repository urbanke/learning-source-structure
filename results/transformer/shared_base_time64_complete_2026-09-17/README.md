# Complete shared-base comparison at time 64

All four conditions (alpha=1; L=0,1,4,8) retain 256 independent sequences
of length 1025 and both exponent orders, 16 and 32: **2048 references,
all 44 cohort checks passed**. Bayes uses the full training prior, with
alpha uniform on [0.5,1.5], depth uniform on 0,...,8 and random relabeling.
The reported prediction follows 64 observed tokens.

Expected next-token regret is in bits. Differences below are transformer
minus Bayes, with one paired standard error across the 256 sequences.
Training runs remain separate.

| L | Bayes regret | Run 1 difference | Run 2 difference |
|---|---:|---:|---:|
| 0 | 0.24693 | 0.21672 ± 0.00882 | 0.21989 ± 0.00903 |
| 1 | 0.74371 | 0.11491 ± 0.00988 | 0.12129 ± 0.00932 |
| 4 | 1.64732 | 0.01224 ± 0.01158 | 0.02271 ± 0.01177 |
| 8 | 2.07091 | 0.08925 ± 0.01280 | 0.08224 ± 0.01304 |

`summary.json` is a distinct corrected-audit export. `validation.json`
binds the original auditor, arithmetic correction, plans, effective case
receipts, raw files, source records and frozen model predictions. Export
recomputed all 44 unchanged numerical checks and the paired moments from
all 256 refined (32-node) predictive distributions per depth. It performed
no model inference or new sampling. Numerical precision/mixing checks do
not prove global posterior convergence.

All 2048 references were re-audited with scale-invariant depth Rhat; other
criteria were unchanged. Twenty-two remaining references were extended
from saved chains to fixed targets. Every planned replacement was required.
Original records, including historical failed flags, remain preserved in
`output/shared_base_curve_references_2026-09-15/` and
`output/shared_base_curve_continuation_2026-09-16/`. Corrected audits are in
`output/shared_base_local_revalidation_v2_2026-09-17/`; final continuations,
mandatory overlay and effective receipts are in
`output/shared_base_local_finish_2026-09-17/`. Raw chains remain outside git.

To reproduce the export from these preserved files, use a fresh output
folder; the exporter refuses to overwrite an existing directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/export_shared_base_time64.py --summary output/shared_base_local_finish_2026-09-17/summary.json --out output/shared_base_time64_export_reproduction
```

The active main figure is `../section3_compact/overview.pdf`; the appendix
uses `../shared_base_conditional_curves/figures_t64_complete/`. Both use
Bayesian points at 1/4/16/64 for every depth. Historical morning summaries
and earlier figures remain untouched. Full-cohort references at 256/1024
remain pending. Figure verification and audit tests are recorded in
`results_manifest.json`.
