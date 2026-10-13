# Uniform-base prediction after a token's first occurrence

This control reuses the saved uniform-base model and all eligible histories
from its existing evaluation. There is no new training, transformer
inference, source generation or Bayesian sampling.

## Question and exact Bayesian answer

At time t, retain a sequence when token t has not appeared at times
1 through t-1. Thus no continuation from that token has been observed.
For the uniform-base source, its unobserved transition row is independent
of the observed history conditional on the shared depth. Each row prior
is symmetric in its 64 possible continuations, so its mean is exactly
uniform. Averaging over the posterior depth preserves that mean. The
initial token is uniform as well; the random relabeling changes none of
these statements. At depth zero every row is already uniform.

Consequently, informed Bayes predicts probability 1/64 for every next
token on each selected history. Its conditional expected loss is exactly
6 bits. A prediction Q has excess loss

    - (1/64) sum_b log2 Q(b) - 6 = KL(U64 || Q).

We calculate this quantity directly, integrating out the unknown row
analytically. It is the conditional expected excess loss over Bayes,
not the regret relative to a revealed realized transition row. It has
no noise from a sampled next token or from a randomly realized oracle
row. The mean still has sampling uncertainty because histories vary.

The negative control asks whether the transformer appropriately avoids
learning this new row from token frequencies elsewhere in the history.

## Which model and sequences

The predictor is the existing **uniform-base transformer**, training
run 1 (seed 0): 2 layers, width 128, 4 attention heads, 544,256 parameters,
trained for 100,000 batches of 64 sequences (6.4 million sequences).
It is distinct from the larger shared-base models used in the main
application. Its checkpoint and configuration hashes are recorded in
the summary; source and predictor code bindings are verified.

The original evaluation has 1,024 independent sequences of length
1,025. One depth was drawn uniformly from 0 through 8 per sequence.
We retain depths 0, 1, 4 and 8 and evaluate at 4, 16 and 64 observed
tokens. Selection uses observed tokens only; every eligible history is
retained. Neither predictor receives the test depth. The second
comparator uses smoothed token counts, (N_b + 1/2)/(t + 32), and is
distinct from the paper's LSA unigram estimator.

## Results

Values below are excess loss over informed Bayes, in bits, mean ± one
standard error over independent selected sequences.

| Depth | Observed tokens | Selected / available | Uniform-base transformer | Smoothed token counts |
|---|---:|---:|---:|---:|
| 0 | 4 | 107 / 112 | 0.000516 ± 0.000017 | 0.07148 ± 0.00027 |
| 0 | 16 | 84 / 112 | 0.000454 ± 0.000016 | 0.21165 ± 0.00155 |
| 0 | 64 | 52 / 112 | 0.000354 ± 0.000021 | 0.33004 ± 0.00548 |
| 1 | 4 | 98 / 101 | 0.000517 ± 0.000019 | 0.07127 ± 0.00023 |
| 1 | 16 | 80 / 101 | 0.000483 ± 0.000022 | 0.20855 ± 0.00174 |
| 1 | 64 | 40 / 101 | 0.000365 ± 0.000018 | 0.33689 ± 0.00629 |
| 4 | 4 | 110 / 115 | 0.000513 ± 0.000016 | 0.07123 ± 0.00021 |
| 4 | 16 | 81 / 115 | 0.000469 ± 0.000017 | 0.20809 ± 0.00169 |
| 4 | 64 | 37 / 115 | 0.000365 ± 0.000019 | 0.37257 ± 0.00809 |
| 8 | 4 | 110 / 119 | 0.000507 ± 0.000015 | 0.07123 ± 0.00021 |
| 8 | 16 | 84 / 119 | 0.000499 ± 0.000022 | 0.21954 ± 0.00265 |
| 8 | 64 | 35 / 119 | 0.000499 ± 0.000046 | 0.41118 ± 0.01376 |

The transformer is within 0.00052 bits of the analytic Bayesian
prediction in every condition, whereas token counts are substantially
worse. Together with the shared-base example, this supports the
behavioral distinction between source families where other contexts
provide useful predictive information and this family where they do
not. It does not identify an internal circuit or isolate an effect of
model size: the two families use different trained checkpoints.

These are exploratory subsets of existing evaluation data. Different
times select different histories, and 918 records overlap in sequence
identities across times. Do not pool them as independent observations
or interpret the time curves as changes on one fixed selected subset.
The uncertainty is pointwise; no simultaneous confidence claim is made.

## Validation and reproduction

The script verifies all 1,024 complete raw-file hashes against the
original validated evaluation, as well as each consumed array hash,
source/model identities, code bindings, dimensions, normalization,
source depths, time indices, and complete cohort coverage. It checks
eligibility both as a first occurrence and as zero observed outgoing
transitions. The saved Bayes distributions are **exactly uniform** on
all selected histories (maximum absolute discrepancy 0).

The loss calculation is checked against an independent scalar
cross-entropy calculation; the counts comparator is independently
constructed with a token counter. Means and standard errors are
recomputed from the compact records using Python's statistics module.
The largest arithmetic difference is below 6e-16 bits. The original
source replay and network-fidelity validation are reused through their
immutable raw-file hashes; no network inference is repeated.

Files:

- `summary.json`: all 12 groups, model information, original source
  hashes and numerical checks.
- `records.csv`: one scalar record per selected sequence and time.
- `validation.json`: independent aggregation checks and output hashes.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 /private/tmp/layered-shared-depth-env/bin/python code/scripts/section4_unseen_uniform_control.py
```

Verify saved results against the raw data:

```bash
OPENBLAS_NUM_THREADS=1 /private/tmp/layered-shared-depth-env/bin/python code/scripts/section4_unseen_uniform_control.py --check-only
```
