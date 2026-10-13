# Eight validated complete shared-base cohorts

All four depths now have complete validated Bayesian comparisons at 256 and
1,024 observations. The six previously released groups are unchanged; the
last two groups add depths 4 and 8 at 1,024 observations. Each group retains
all 256 fixed sequences, both integration orders and the original acceptance rules.

| Depth | Context | Mean regret | Sequence SE | Numerical allowance |
| --- | --- | --- | --- | --- |
| 0 | 256 | 0.0832315 | 0.0012273 | 0.0001083 |
| 0 | 1,024 | 0.0273856 | 0.0003956 | 0.0000331 |
| 1 | 256 | 0.4794506 | 0.0126484 | 0.0004426 |
| 1 | 1,024 | 0.3177551 | 0.0111555 | 0.0001466 |
| 4 | 256 | 0.9850131 | 0.0391385 | 0.0005511 |
| 4 | 1,024 | 0.5201099 | 0.0291721 | 0.0002415 |
| 8 | 256 | 1.1612751 | 0.0566844 | 0.0006049 |
| 8 | 1,024 | 0.5788516 | 0.0398744 | 0.0003590 |

All values are in bits. Every planned group passed validation.
The exporter independently replayed source/oracle pairing, each saved
forecast's regret and the whole-cohort numerical gates; the largest score
discrepancy was 1.89e-15 bits. Raw-chain diagnostics retain their cluster
provenance. Paired transformer comparisons remain separate for every model
and seed. No incomplete cohort is averaged.

`reports.json.gz` preserves the exact local `summary.json`, `validation.json`
and `comparisons.tsv`. Their validation receipt binds `paired_scores.npz`.
The figure loader reads this archive directly and checks the saved summary
hash and every complete-cohort gate. Loose generated reports and score
arrays remain local. Reproduce the export from the raw snapshot with:

```bash
python code/scripts/shared_base_full_cohort_results.py --input output/live_compute_check_2026-09-23_afternoon/shared_base/input --campaign-input output/kuma_results_2026-09-20/shared_base_campaign_2026-09-18 --out results/transformer/shared_base_full_cohort_2026-09-23
```

See the [earlier release](../shared_base_full_cohort_2026-09-19/README.md) for
the unchanged validation rules and reference conventions.
