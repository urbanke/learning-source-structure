# Section 3: transformer versus informed Bayes

`overview.pdf` and `overview.png` contain the six-panel main figure. All panels
show expected next-token regret in bits after the indicated number of observed
tokens. The main comparison uses fixed **training run 1 (seed 0)**; appendix
figures show the two training runs separately. Runs are never averaged or
selected by performance.

| Panel | Source and displayed comparison |
| --- | --- |
| (a) | Coins: the equal-mass prior on biases 0.2/0.8, tested at 0.2; and one uniform-[0,1] model, tested at 0.1 and 0.5. The legend distinguishes training prior from test bias. |
| (b) | Product chains with known 8×8 encoding, at depth pairs (1,1), (1,8), (4,4), (8,8). Two-layer and four-layer models both use 100k updates. |
| (c) | Product chains with hidden 2×8 encoding: two-layer 100k model (solid blue), four-layer 400k continuation (dashed green), final eight-layer 400k model (dashed purple), informed Bayes (unconnected black open circles), and the Bayesian predictor that ignores the factorization (gray dash-dot curve; legend "Unstructured Bayes"; `flat_dirichlet` in the campaign's `comparison.json`: a 16-state chain with independent rows under a uniform prior). |
| (d) | Uniform-base LSA, depths 0, 1, 4, 8. Two-layer and four-layer models both use 100k updates. No inset. |
| (e) | Shared-base LSA at exponent 1 and depths 0, 1, 4, 8: two-layer 100k model (solid) and final eight-layer 400k model (dashed), with matching colors by source depth. Informed Bayes values are isolated points at the reported contexts. |
| (f) | Two-layer and four-layer pooled transformers trained on three families, both for 300k updates, with curves through 1,024 observations and informed Bayesian mixture points at 1, 4, 16, 64, 256 and 1,024 observations. |

In panels (c)/(e), 100k and 400k updates correspond to 6.4 and 25.6 million
training sequences. The plotted eight-layer models are the final 400k
checkpoints. Model capacity and training exposure change together, so these
comparisons do not isolate a capacity-only effect or establish convergence.
The new four-layer models in (b), (d) and (f) have width 256, eight heads and
3,454,464 parameters. Their smaller models have 544,256 parameters. Training
exposure is unchanged: 6.4M sequences in (b)/(d) and 19.2M in (f). The pooled
models reuse matched indexed streams; the specialists use fresh draws from
the same priors, without claiming identical historical training batches.

## Evidence and validation scope

The hidden-encoding curves use the same **3,840 previously examined sequences**,
indices 256:4096 of the frozen cohort. The first 256 development
sequences are excluded. This is a reused comparison, not fresh independent
confirmation. The appendix shows both runs and also the four-layer 100k model;
all six older models remain in the audited data and endpoint table.

Shared-base comparisons use the same 256 sequences per depth. Bayesian
references are validated at 1, 4, 16 and 64 observations for every depth. The
complete-cohort release supplies **t=256 and t=1024 at all four depths**. Each point includes all 256 predeclared sources, combining the
254 new references with the two separately validated pilot references.
No partial-cohort mean or successful-case subset is plotted. The
appendix retains both runs, the four-layer 100k models and their 400k
continuations; further training and greater capacity do not improve every
condition.

The pooled transformer curves and Bayes at 1/4 average over 1,024 sequences
per family. Estimates at 16/64/256/1024 retain all 64 selected histories per family.
Context16 is accepted; 64/256/1024 remain provisional, recorded in
`../pooled_estimates_2026-09-24/points.json.gz`; RU requested ordinary markers
in the paper and status reporting separately. `pooled_runs.pdf` / `.png` show
both training runs. All circles use posterior family likelihood weights.

`specialist_runs.pdf` / `.png` show both larger product and uniform-base runs
separately against the same two-layer baseline. The
[larger-model release](../larger_models_2026-09-23/README.md) binds the final
checkpoints, frozen cohorts and independently replayed paired statistics.

Shading and error bars represent one pointwise standard error across test
sequences; paired differences are formed before averaging. The pooled mean
uses stratified errors. These errors do not measure training randomness;
shared-base Bayesian numerical allowances are reported separately.

## Audits and reproduction

The plot consumes three independently audited releases. Their READMEs describe
source/probability checks, numerical acceptance and limits of local replay:

- [Hidden-encoding campaign](../hidden_factor_campaign_2026-09-19/README.md).
- [Shared-base campaign](../shared_base_campaign_2026-09-19/README.md).
- [Complete-cohort Bayesian points](../shared_base_full_cohort_2026-09-23/README.md).

Imports perform no inference or sampling. Each release and the overview receipt
bind input/output hashes; raw-chain diagnostics remain cluster-auditor provenance.

With the transferred raw data and earlier frozen inputs present, reproduce
from the repository root in this order:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/hidden_factor_campaign_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/shared_base_campaign_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/release_large_transformer_results.py --check-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/shared_base_full_cohort_results.py --input output/live_compute_check_2026-09-23_afternoon/shared_base/input --campaign-input output/kuma_results_2026-09-20/shared_base_campaign_2026-09-18 --out results/transformer/shared_base_full_cohort_2026-09-23
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/section3_compact_figure.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/section3_compact_figure.py --check-only
```

The main layout is 5.5 inches wide with a minimum 7-point font. Horizontal
axes are base-2; the tick labels are 1, 32 and 1024 (1, 16 and 256 in panel (a),
whose data end at 256 tokens). The label "Training run 1" sits at the upper
right of panels (b) through (f). Figures and this
README are publication assets. Generated JSON/TSV reports and raw transfers
remain local and intentionally ignored for Overleaf; do not force-add them.
