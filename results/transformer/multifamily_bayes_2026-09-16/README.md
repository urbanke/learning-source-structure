# Pooled transformer versus the informed three-family Bayesian mixture

This supplies the Bayesian reference in Figure 1(f) and the pooled-training appendix.
The two pooled transformers are compared directly with Bayes under their
equal mixture of the product, shared-base LSA and uniform-base source priors.
Each component integrates its own source parameters. Family weights update
using the probability of the observed sequence under each component:

```text
w_f(t) = B_f(x_1,...,x_t) / sum_g B_g(x_1,...,x_t)
p_mix(next | observations) = sum_f w_f(t) p_f(next | observations).
```

The equal weights apply to the original sequence-level mixture. They are
not held fixed when combining next-token predictions after observations.

## What is measured

The existing frozen test set supplies 1,024 independent sequences from each
full training prior, or 3,072 sequences in the equal-family average. Every
sequence has length 1,025. Figure 1(f) shows **expected next-token regret**,
the same quantity as its other panels. The transformer curve uses saved
results at 1, 4, 16, 64, 256 and 1,024 observations. Bayesian-mixture points
are available at **1 and 4 only**, using the validated short-sequence
references. Training run 1 (saved seed 0) is shown in the main text; both
runs appear separately in the appendix. There is no retraining or new
transformer inference.

The next-token data and their checks are in
[next_token_regret.json](next_token_regret.json) and
[next_token_validation.json](next_token_validation.json). These use the
true transition row at the current token and sum the KL divergence over
all 64 possible next tokens. The Bayesian-mixture regrets at 1 and 4
observations are 2.38179 and 2.22485 bits; training run 1 gives 2.38906 and
2.24451 bits. Its regret at 1,024 observations is 0.41466 bits, where no
validated mixture reference is yet available. Family weights use observed
sequence likelihoods. The recovery checks all 3,072 sequence identities,
oracle rows, saved parent and observed-child likelihoods, and full
next-token normalization; the largest normalization error is below 8.1e-12.

The cumulative comparisons below are additional appendix results, rather
than the quantity plotted in Figure 1(f). At time t, cumulative loss scores
tokens 2 through t+1. The first token is observed but unscored, with common
marginal probability 1/64 in all families.

| Prediction time | Training run 1 minus mixture Bayes | Training run 2 minus mixture Bayes |
|---|---:|---:|
| 1 | 0.0048 ± 0.0027 bits | 0.0075 ± 0.0027 bits |
| 4 | 0.0337 ± 0.0066 bits | 0.0360 ± 0.0069 bits |

Errors are sequence-paired standard errors. The equal-family average uses
`sqrt(sum(SE_family**2))/3`; training seeds remain separate. Separate family
results are retained in [short_comparison.json](short_comparison.json).
Individual family averages may be negative: Bayes is optimal for the full
training mixture, not necessarily for a test restricted to one family.

At time 4, the Bayesian mixture costs **0.1768 ± 0.0117 additional bits**
relative to Bayes told the generating family. The universal bound is
log2(3) ≈ 1.585 bits; it need not be attained at a short time. The transformer
comparison above measures the additional approximation error beyond the
mixture itself.

## Reference validation and scope

- [family_evidence_validation.json](family_evidence_validation.json) validates
  product and uniform-base sequence evidence using quadrature refinement,
  probability normalization and an independent sequential implementation.
  Both components are evaluated on all three source-family cohorts, through
  time 1,024, reusing the existing sealed LSA store.
- [shared_base_evidence_validation.json](shared_base_evidence_validation.json)
  validates the shared-base adapter against the established short-prefix
  catalog, its provenance and the independent first-step calculation.
  It provides absolute joint probabilities through five tokens without
  changing or renormalizing the saved numerical values.
- [short_validation.json](short_validation.json) records matched sequence
  identities, input/output hashes and the joint-evidence versus sequential
  posterior-prediction check. The latter agrees within 7.1e-12 bits on
  24 independently selected sequence records and 96 prediction steps.

**The direct mixture comparison at times 16, 64, 256 and 1,024 is not yet
computed.** An experimental annealed SMC implementation estimates the
missing shared-base evidence. Small-alphabet correctness tests pass, but a
bounded pilot did not establish sufficient precision: coarse 1,025-token
replicas differed by 15.24 bits; refined 17-token replicas differed by
0.60 bits. These pilot outputs remain under ignored
`output/shared_base_evidence_smc_experimental_2026-09-16/` and are excluded
from the paper figures and validated comparison. The compact
[pilot archive](experimental_smc_summary.json) preserves the estimates,
timings and raw-report hashes. The subsequent
[controlled calibration](../shared_base_evidence_calibration_2026-09-16/README.md)
also fails its predeclared precision target: both samplers fail all six
initial controls, and a particle-only refinement still has an
error-plus-uncertainty value of 0.0182 bits on AAAAA against a 0.01-bit
target. All 224 estimates and an independent report check are preserved.
This does not validate any additional context lengths. The original design
is in `notes/reviews/multifamily_likelihood_pilot_2026-09-16.md`.

The original pooled-versus-specialist results remain in the adjacent
`multifamily_2026-09-16/` folder as diagnostic records. They are not the
Bayesian reference and no longer supply the manuscript's pooled panel.
Original training and checkpoint provenance is preserved in
`results_manifest.json` and `notes/reviews/multifamily_experiment_2026-09-16.md`.

## Reproduction

Run from the repository root with NumPy/SciPy installed and the original
held-out cohorts and saved model scores under
`output/multifamily_results_2026-09-16/`. The following uses a **fresh output
directory**, since numerical validation receipts are immutable. Set the
existing companion store/source locations for the evaluation host:

```bash
export LSA_STORE="$HOME/Projects/product_model_with_memory/tables/anchors_prod"
export PMM_SRC="$HOME/Projects/product_model_with_memory/src"
export PMM_UNIVERSAL_TABLES="$LSA_STORE"
export PMM_PHI_LADDER_EVERY=1 PMM_PHI_LADDER_DEGREE=11 PMM_PHI_SADDLE_MIN_L=54
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
TF_MIXTURE_REPRO=output/multifamily_bayes_reproduction_2026-09-16
```

Validate the component numerical calculation, then compute each component
on all three frozen full-prior cohorts:

```bash
python code/scripts/validate_multifamily_family_reference.py \
  --heldout output/multifamily_results_2026-09-16/heldout \
  --out "$TF_MIXTURE_REPRO/family_evidence/validation.json"

python code/scripts/multifamily_family_reference.py --family product \
  --heldout output/multifamily_results_2026-09-16/heldout \
  --validation "$TF_MIXTURE_REPRO/family_evidence/validation.json" \
  --out "$TF_MIXTURE_REPRO/family_evidence"

python code/scripts/multifamily_family_reference.py --family uniform_base \
  --heldout output/multifamily_results_2026-09-16/heldout \
  --validation "$TF_MIXTURE_REPRO/family_evidence/validation.json" \
  --out "$TF_MIXTURE_REPRO/family_evidence"
```

Combine these with the validated shared-base catalog and saved transformer
losses. This writes a new summary, validation receipts and per-sequence
arrays under the reproduction directory:

```bash
python code/scripts/multifamily_short_mixture.py \
  --raw output/multifamily_results_2026-09-16 \
  --families "$TF_MIXTURE_REPRO/family_evidence" \
  --out "$TF_MIXTURE_REPRO/short_comparison" \
  --curated "$TF_MIXTURE_REPRO/curated"
```

To reconstruct the next-token quantities from the same frozen scores and
validated short references, with the environment above:

```bash
python code/scripts/pooled_next_token_regret.py
```

This reuses the existing numerical rule and short-prefix catalog. It does
not train or evaluate a transformer again, or run a new posterior sampler.
The paper figure uses the committed curated summaries by default. Its
rebuild and verification commands remain:

```bash
python code/scripts/section3_compact_figure.py
python code/scripts/section3_compact_figure.py --check-only
```

Focused scientific code checks:

```bash
python -m pytest -q code/tests/test_mixture_bayes.py \
  code/tests/test_mixture_family_evidence.py \
  code/tests/test_shared_base_short_evidence.py \
  code/tests/test_shared_base_evidence_smc.py
```

Passing the SMC unit tests verifies its small-case algorithm; it does not
approve the experimental long-context estimates as scientific references.
