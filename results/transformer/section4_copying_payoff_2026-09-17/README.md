# When copying pays off: saved-data analysis

This analysis reuses the complete, previously audited natural-sequence experiment. It performs no training and no new Bayesian inference.

- 1,024 independent sequences, each of length 1,025, from the uniform-base source on 64 tokens; source depth is uniform on 0,...,8.
- The transformer is the two-layer model trained on 6.4 million sequences. The identified first-layer copying path is removed by zeroing head 2's output. The control removes first-layer head 1. These are zero-based indices `[0,1]` and `[0,0]` in the saved protocol.
- Both predictors receive identical observed sequences. Source depth is used only to group the results; it is withheld from the transformer and Bayesian predictor.
- At time t=1,...,1024, the loss of the next-token distribution is averaged over the true transition row. The paired difference is then averaged across independent source sequences. The cumulative curve sums those differences from time 1 through n. The initial observed token is free.

The main figure `payoff.pdf` shows the 101 depth-one sequences. Blue is expected loss with the copying path disabled minus intact-transformer loss. Orange is expected uniform-predictor loss minus informed Bayesian loss. Positive means that retaining the path (blue), or using Bayes (orange), improves prediction.

The final crossings within the saved horizon are:

| Criterion | Transformer | Informed Bayes |
|---|---:|---:|
| Instantaneous | 305 [297,459] | 108 [74,136] |
| Cumulative | 543 [498,594] | 158 [143,174] |

Brackets are the 2.5% and 97.5% order statistics from 2,000 whole-sequence bootstrap samples. The last crossing of an unsmoothed empirical curve is sensitive to late fluctuations; these are exploratory uncertainty estimates, not simultaneous confidence guarantees. All time points are retained, and all bootstrap draws are retained, including right-censored crossings. Neither a first isolated positive value nor an average of individual sequence crossings is used as the main payoff time. No claim is made about times beyond 1024.

Using uniform prediction as the baseline for the transformer gives crossing times 305 and 544. Thus the large timing discrepancy is not explained by the difference between the disabled transformer and uniform prediction. The latter difference accumulates to only 0.0291 bits at time 1024 in the depth-one group. The data show a copying payoff, but do not show close agreement with Bayesian payoff timing for this trained transformer.

`all_depths.pdf` shows representative depths 0,1,4,8 and includes the independently selected control-path intervention. The additional `payoff_all_depths_*` figures and `curves.npz` retain all nine source depths. Depth zero has no positive payoff through the horizon. Higher depths can have earlier and multiple crossings; their first and last crossings are kept separately in `crossings.tsv` and `summary.json`.

The archived LSA crossover calculation compares an adaptive LSA unigram with a pure row-wise copier. Neither that unigram nor that pure copier has been identified with these two transformer conditions. These measurements therefore do not directly test the archived analytic formula.

Reproduce from the saved raw cohort:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /private/tmp/layered-shared-depth-env/bin/python code/scripts/section4_copying_payoff.py
```

The script verifies each raw-file checksum against the earlier independent audit, recomputes the paired losses from saved probability distributions, checks all earlier means and standard errors, and compares saved cumulative differences with sums of recomputed instantaneous differences. `summary.json` records the protocol/checkpoint/input-manifest hashes and exact statistical definitions. Numerical validation of Bayes is inherited from the saved quadrature audit and retains that audit's stated limitations.

Fast retained-artifact and crossing check (no inference or full raw-data reread):

```bash
/private/tmp/layered-shared-depth-env/bin/python code/scripts/section4_copying_payoff.py --check-only
```
