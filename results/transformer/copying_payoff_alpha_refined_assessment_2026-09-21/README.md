# Validated copying-payoff times

This directory is the compact publication readback of the approved eight-step
copying-payoff assessment. All 48 Bayesian numerical coordinates (three
depths, eight times, instantaneous and cumulative) pass their predeclared
precision budgets. The source integrity, input-validation, scientific-
precision, and payoff-forecast gates all pass.

The point estimates give a direct comparison. At depth 0, neither informed
Bayes nor either transformer run has a positive gain through observation 8.
At depth 2, the informed-Bayes cumulative gain is positive from observation
1 and remains positive, while the two transformer cumulative gains begin
their final positive runs at observations 4 and 5. At depth 3, the analogous
times are 1 for informed Bayes and 1 and 5 for the two transformer runs.
`payoff_times.tex` reports these and the corresponding next-token times with
paired-bootstrap quantiles; `payoff_times.tsv` is its machine-readable source.

The Bayesian curve uses validated exact short evidence at times 1--4 and the
alpha-32 rule at times 5--8. Its frequency-only comparator is induced by the
source's stationary law and uses the paired 16/32-node refinement. It is a
surrogate for the finite shuffled-training law. The assessment covers only
times 1--8, reuses the development histories, and is not an independent
confirmation. Transformer intervals resample test sequences and exclude
training uncertainty.

Source: `output/copying_payoff_alpha_refined_assessment_2026-09-21/`.
The source `summary.json` SHA-256 is `98d92225d114882f40bb211b363fdc389c70e115b2904b336e1f011e9bbd813f`. `receipt.json` binds the
retained source artifacts, exact validation flags, transitive provenance, and
the three publication files.

The retained assessment was rechecked with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/ruediger/Projects/product_model_with_memory/.venv/bin/python code/scripts/assess_copying_payoff.py --out output/copying_payoff_alpha_refined_assessment_2026-09-21 --check-only
```
