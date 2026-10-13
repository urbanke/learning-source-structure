# Exploratory product-source context requirements

`compact.pdf` shows the two extreme preselected source conditions, coordinate
depths (1,1) and (8,8). `all_four.pdf` retains all four preselected conditions.
Colors match the product-source panel in Section 3. Solid curves are the frozen
transformer; dashed curves and open circles are informed Bayes. Circles/dots
mark six illustrative targets; the whole step curve is derived from the saved
mean regrets.

**Suggested caption.** First tested context length at which mean next-token
regret is at most the target, for the fixed-encoding 8x8 product source.
Each condition averages 256 independent transition-matrix draws and sequences
of length 1025. The tested lengths are 1,2,4,...,1024; each horizontal level is
one of these lengths, with no interpolation between them. The upper band means
the target was not reached at any tested length. Bayes retains the complete
training depth prior. These are exploratory estimates from existing data;
threshold uncertainty and variation across transformer training runs are not
shown. Mean-regret standard errors are retained in the numerical summary.

## What this shows

- At a target of 0.1 bit, both predictors first reach the target at 1024
  observations for (1,1), (1,8), and (4,4), and at 512 for (8,8).
- At 0.5 bit, both give 128 for the first three conditions and 64 for (8,8).
- The agreement is not universal. At 1 bit, (1,1) gives Bayes 16 and transformer
  32. At 0.2 bit, (1,8) gives Bayes 256 and transformer 512.
- The last example illustrates instability near a threshold: at time 256,
  Bayes has mean regret 0.19933 with SE 0.01239, and the transformer 0.21262
  with SE 0.01270. The target 0.2 is very close to both estimated means.
  The displayed factor-two context difference is not an established factor-two
  difference in the population requirement.
- Neither predictor reaches 0.05 bit on the tested grid in any condition.

Equal requirements on this coarse grid do not establish equal exact context
requirements. Conversely, a grid-step difference may exaggerate a small risk
difference. This is a useful picture of the proposed application, not a new
prospective prediction test.

## Inputs, verification and reproduction

Inputs are the validated results under
`results/transformer/conditional_accuracy/product/` and all 3072 original raw
source, Bayesian-reference, and transformer files under
`output/conditional_accuracy/product/L*/{data,references,model}/`.
The new script verifies the existing receipt and all raw-cohort hashes, checks
token/time identities, and recomputes every mean and SE directly from the saved
probability distributions. All 1024 sequences are retained; the old summary
is reproduced exactly. It does not rerun a model or Bayesian integration.

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python
code/scripts/section4_product_accuracy.py`; add `--check-only` to verify the
derived artifacts without modifying them. `summary.json` keeps each risk curve
with its SE, the exploratory requirements, and provenance. `requirements.tsv`
is the compact numerical table. `receipt.json` binds the derived outputs.

The reusable function `draw_required_context(axis, summary, pair_ids=None,
legend=True)` supports a combined main-text figure. `compact_case_ids` contains
the two default conditions; `cases` contains all four. The top plot band is
categorical (nonattainment), not a numerical extrapolation beyond 1024.
