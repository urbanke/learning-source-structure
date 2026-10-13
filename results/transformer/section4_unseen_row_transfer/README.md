# Prediction before a row has been observed

This exploratory analysis reuses the completed, validated shared-base
experiments. It does not run new sampling, training or transformer inference.

At a fixed time t, retain a sequence exactly when the final observed token
has never appeared earlier. Its transition row therefore has **zero observed
outgoing transitions**. For each retained sequence, compute the predictor's
expected loss over all 64 possible next tokens using the saved true row.
The plotted gain is **6 bits minus that loss**; uniform prediction costs
exactly 6 bits. Positive values mean better prediction than uniform.

The original cohorts contain 256 independent sequences of length 1025 for
each alpha=1, L=0,1,4,8 condition. Both transformers and Bayes retain the
full training prior; the selected alpha and depth are hidden from them.
The figure displays the selected sample count for every condition. Bars
are one standard error across whole sequences, not prediction times.

## What the results show

At t=16:

| True depth | Selected n / 256 | Bayes gain | Transformer seed 0 | Transformer seed 1 | Count baseline |
|---|---:|---:|---:|---:|---:|
| 0 | 136 | .585 ± .014 | .499 ± .014 | .500 ± .014 | .568 ± .012 |
| 1 | 136 | .439 ± .037 | .394 ± .030 | .402 ± .031 | .449 ± .031 |
| 4 | 145 | .233 ± .041 | .219 ± .041 | .195 ± .038 | .219 ± .044 |
| 8 | 161 | .035 ± .032 | .015 ± .034 | .001 ± .031 | −.037 ± .042 |

Values are bits, mean ± one SE. Bayes and both transformers make useful
predictions at L=0,1,4 although that row has not been observed. No clear
gain is established at L=8. L=0 is iid and should not be presented as a
nontrivial row-specific transfer example.

This supports a modest behavioral statement: the transformer uses
information beyond observed continuations of the current token. It does
**not** identify an internal mechanism or establish an advantage that
requires detailed transition structure. The simple counts-only rule
(N_b + 1/2)/(t + 32) accounts for much of the gain. It has a different prior
from informed Bayes, and can outperform it on particular fixed-parameter
subsets without contradicting Bayesian optimality for the full prior.

Agreement is approximate. At t=16 the transformer's extra loss over Bayes,
using paired sequence differences, is .086/.085 bits at L=0,
.044/.036 at L=1, .014/.039 at L=4, and .020/.034 at L=8.

## Controls and limits

Uniform prediction is also exactly the uniform-base source's informed
Bayesian prediction for an unseen row. Here it is evaluated on shared-base
sources as a simple baseline; this is not a new uniform-base transformer
experiment.

Shared-base Bayes already has a nonuniform prediction after one observed
token. The compact records therefore also include a one-token Bayes
baseline, mapped to the final observed label, and paired gains over it.
This is a memory-reset reference; it is not the optimal predictor given
the later time and the unseen-row event. The t=16 transformer gains over
it remain positive at L=0,1,4. It is omitted from the main plot for clarity.

At t=4 the selected counts are 210,214,223,228. The t=4 and t=16 subsets
differ, and their points should not be interpreted as paired changes on
one fixed selected cohort. All eight groups are exploratory subgroups of
previously examined experiments.

To test sensitivity beyond token frequencies, a next experiment should
hold the token histogram and final query fixed while changing transition
arrangements. Freeze those controls and a fresh held-out cohort before
new evaluation. The current data cannot answer that question. Longer
times require completed Bayesian references and a sufficient number of
unseen-row cases; no partial t64 result is used here.

## Validation and reproducibility

The t=4 predictions use the existing deterministic rank-marginalization
catalog and its validated row/exponent refinements. The t=16 predictions
use the completed 16/32-node reference audits. Input hashes, model/source
identities, cached probabilities and loss arithmetic are rechecked.

Crucially, the numerical criterion is **recomputed for each selected t=16
subset**; passing the original 256-sequence average alone is insufficient.
All four selected subsets pass. The largest numerical allowance is
.003231 bits, separately from the whole-sequence error bars. These
sampling diagnostics do not prove absence of an undiscovered mode.
Raw chain sampling and the expensive raw-array audits are not rerun.

- summary.json: all eight groups, paired losses, subset numerical checks,
  model checkpoint hashes and exact input hashes.
- records.csv: one row per selected sequence/time, with all six expected
  predictor losses. Its 1453 records overlap across times; they are never
  pooled as independent observations.
- validation.json: artifact hashes and independent arithmetic replay.
- unseen_row_t16.pdf/png: compact main-panel candidate.
- unseen_row_t4_t16.pdf/png: detailed two-time figure.

Reproduce from the original frozen input directories:

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/section4-unseen-mpl /private/tmp/layered-shared-depth-env/bin/python code/scripts/section4_unseen_row_transfer.py
~~~

Fast independent mean/SE verification from the compact committed records:

~~~bash
OPENBLAS_NUM_THREADS=1 /private/tmp/layered-shared-depth-env/bin/python code/scripts/section4_unseen_row_transfer.py --verify-only
~~~

For combined figures import draw_panel(ax, summary, time=16,
include_counts=True, legend=False). Bayes is an open black diamond,
transformer seeds 0/1 are blue/orange, and the count baseline is gray.
