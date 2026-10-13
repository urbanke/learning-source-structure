# Shared-base LSA: larger models at fixed training exposure

Both larger-model training runs completed 100,000 updates and 6.4 million
sequences on Kuma (job 4254741). Final checkpoints were transferred and
evaluated locally on the existing 256 test sequences per source depth
L=0,1,4,8, at alpha=1. Each sequence has length 1025. The evaluated times
are 1,4,16,64,256,1024 observed tokens. Conditions are hidden from both
predictors; the informed Bayesian reference retains the full training prior.

The original model has 2 layers, width128, 4 heads and 544,256 parameters.
The larger model has 4 layers, width256, 8 heads and 3,454,464 parameters.
Both receive the same number of fresh training sequences from the same prior.
Depth, width and head count change together. Training runs remain separate;
the main figure uses run1 (saved seed0), without selection by performance.
Every comparison uses final weights, not a selected intermediate checkpoint.

At time64 the gaps from Bayes are:

| Depth | Original, run1 | Larger, run1 | Original, run2 | Larger, run2 |
|---|---:|---:|---:|---:|
| 0 | 0.2167 | 0.0395 | 0.2199 | 0.0287 |
| 1 | 0.1149 | 0.0169 | 0.1213 | 0.0103 |
| 4 | 0.0122 | 0.0146 | 0.0227 | 0.0141 |
| 8 | 0.0892 | 0.0049 | 0.0822 | 0.0103 |

Values are next-token loss differences in bits. `comparison.json` includes
paired standard errors and all tested times; the paper's appendix tabulates
the time64 uncertainties. Capacity substantially reduces the larger gaps at
L=0,1,8. At L=4 the original gap was already small. The larger model is not
uniformly better at every context length, and these observations do not
establish convergence to Bayes.

## References and scope

The evaluator reuses existing Bayesian references at times1,4,16,64, with no
new source generation or Bayesian sampling. The complete time64 release is
an explicitly recorded supplement to the frozen training plan. Its 6,144 raw
reference-file hashes, mandatory continuations, complete-cohort numerical
gates and per-sequence scores were checked before model evaluation. Validated
population Bayesian references at times256/1024 remain unavailable. Larger
models improve over the original at time1024 in both runs and all four depths,
but this alone does not determine their remaining gap from Bayes there.

The existing evaluation sequences motivated this capacity experiment. This
is a follow-up comparison, not an independent confirmation sample. Standard
errors describe variation across independent sequences, not across training
runs. Numerical reference uncertainty is assessed separately by the original
Bayesian validation. Neither increasing training exposure nor isolating
transformer depth is tested here.

## Reproduction

The frozen training plan is
`code/transformer/configs/shared_base_scaling_plan_2026-09-17.json`.
Training and transfer instructions are in
`code/transformer/kuma/README_shared_base_scaling.md`.

Evaluation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/transformer/shared_base_scaling_eval.py --plan code/transformer/configs/shared_base_scaling_plan_2026-09-17.json --runs output/shared_base_scaling_download_2026-09-17/seed0 output/shared_base_scaling_download_2026-09-17/seed1 --time64-complete results/transformer/shared_base_time64_complete_2026-09-17/summary.json --out output/shared_base_scaling_evaluation_2026-09-17 --device cpu --batch 4 --torch-threads 4
```

Completed evaluation directories are immutable; use a new output directory
for any reproduction. The comparison retains all 48 depth/time/run records.
The independent validator re-reads saved probabilities, source transition
rows, identities and paired scores. Small per-sequence score archives are
kept with this release; checkpoints and full prediction records stay under
ignored `output/` and in the original cluster run directory.

Presentation uses `code/scripts/shared_base_scaling_figures.py` and
`code/scripts/section3_compact_figure.py`. It performs no inference, sampling,
smoothing or interpolation of estimates. Lines connect the six tested times;
Bayes appears only as points at validated times. Both training runs and both
model sizes are shown separately in the appendix figure.
