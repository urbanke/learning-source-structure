# Product source with hidden encoding: completed scaling comparison

All four follow-up training and evaluation runs are complete. The transferred
records passed independent local validation, and the results are included in
Section 3.3, Figure 1(c), and the hidden-factor appendix of `mechanisms.tex`.
No further run is needed to close this pilot.

## Comparison

The product source with hidden encoding is 2×8. Its historical internal
identifier remains `hidden_factor` to preserve recorded data paths.
All predictors use the same 256 independent
full-prior sequences of length 1025, the first part of a frozen 4096-sequence
cohort. These 256 sequences motivated the follow-ups and are development
data. Both training seeds are reported separately, at all six declared
observation times: 1, 4, 16, 64, 256 and 1024.

- Original: 2 layers, width 128, 4 heads, 531,968 parameters; 6.4 million
  training sequences.
- More training: continue the original models and optimizer states to 25.6
  million total sequences, holding the final learning rate at 3e-5.
- Larger model: 4 layers, width 256, 8 heads, 3,429,888 parameters; the same
  6.4 million indexed training sequences as the corresponding original seed,
  with the original warmup/cosine schedule.

All checkpoints are the planned final ones, with no selection among seeds.
At 1024 observations, next-token regret in bits is:

| Predictor | Seed 0 | Seed 1 |
|---|---:|---:|
| Original | 0.14651 ± 0.00614 | 0.13506 ± 0.00551 |
| More training | 0.13286 ± 0.00539 | 0.12214 ± 0.00500 |
| Larger model | 0.11406 ± 0.00458 | 0.07831 ± 0.00341 |
| Informed Bayes (common) | 0.03705 ± 0.00143 | 0.03705 ± 0.00143 |

Uncertainty is one standard error across sequences. Paired next-token
improvements over the matching original model are 0.01366 ± 0.00228 and
0.01292 ± 0.00224 bits for continuation, and 0.03246 ± 0.00299 and
0.05675 ± 0.00409 for larger models. Both interventions help in these runs,
with larger gains from the larger architecture, but neither closes the
Bayes gap. This tests one continuation schedule and one larger architecture;
depth, width and head count change together. Two seeds do not establish a
general scaling law. Sequence errors do not estimate variation across seeds.

## Files and verification

- `comparison.json`: checked copy of the cluster summary, with all five
  groups, per-seed results and paired comparisons.
- `validation.json`: independent audit, input hashes and scope limitations.
- `figures/next_token_regret.pdf` and `.png`: separate panels for training
  runs 1 and 2 (saved seeds 0 and 1). Each has four curves: original blue,
  more training orange, larger model green, and Bayes black dashed with
  open circles. Transformer curves are solid without markers. Bands show
  one standard error. Main Figure 1 uses run 1 only for readability; neither
  run is omitted from the complete appendix comparison or the data.
- `figures/receipt.json`: figure inputs, displayed values and artifact hashes.

The audit checks 1024 new prediction records and rechecks 768 original
model/reference records. All five groups and paired means/standard errors
reproduce exactly. It recomputes next-token scores from saved probability
vectors and aggregates saved cumulative scores; it does not repeat full
tokenwise inference. Recorded checkpoint/configuration/ancestry/final-step
identities are checked, but weight binaries were not transferred and are
not rehashed. The original exact-reference audit is retained separately.

Raw data remain outside git:
`output/hidden_factor_scaling_results_2026-09-17/` and
`output/hidden_factor_cluster_pilot_2026-09-16/`.
The frozen plan is
`code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json`.

From the repository root, verify saved raw data without inference:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/scripts/validate_hidden_factor_scaling_results.py --check-corruption
```

Add `--out results/transformer/hidden_factor_scaling_2026-09-17` only to
regenerate the curated comparison and audit. The audit passed 17,329 checks
and rejected six deliberately corrupted inputs.

Reproduce figures from the curated results:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/hidden_factor_scaling_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/hidden_factor_scaling_figures.py --check-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/section3_compact_figure.py --check-only
```

## Separate remaining task

Independent confirmation will use the other 3840 sequences and the same six
saved models. It awaits completion of the full Bayesian references and an
evaluation adapter for this untouched subset. No retraining is needed.
This does not reopen the completed 256-sequence development comparison.
