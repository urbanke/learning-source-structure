# Product predictors under mismatch and time variation

Completed fixed cohort: 1,024 independent source draws at each of the two depth pairs (1,1) and (4,4), with four paired conditions per draw.

Frozen plan: `f32337eebf7923d039c6ad705c6453a049b838cc1e39aa4abb70259d7ce1138e`. Transformer checkpoint SHA-256: `29cac698253475a539a0b1bb76a47436a6eee0b0ab3d8c651a61bfe239b9b271`.

`depth1` and `depth4` show next-token KL from the actual current transition row to each predictor. `agreement` shows KL from the training-prior Bayesian prediction to the frozen transformer prediction. Bands are mean ± 1.96 standard errors across independent source draws, pointwise rather than simultaneous. Only one trained transformer seed is used. No probabilities were renormalized.

The dotted gray level in the random-label panel is the independently calculated stationary long-context Bayesian limit from the same source matrices; it is not a fit to these curves or a fixed-window transformer limit.

The transformer weights and Bayesian product prior remain fixed, and both predictors retain the full history. The random label mapping is fixed within each sequence. New matrices first affect the prediction after 256 observed tokens; predictors are neither reset nor given a boundary signal. Predictive agreement outside the training family is an empirical question, not a Bayes optimality guarantee.

Recovery descriptions in the source summary use each predictor's own prechange risk plus 0.05 bits, so the two numerical thresholds can differ. They are descriptive tested-grid crossings, without recovery-time confidence intervals.

The benchmark passed wider-integration and exact depth-one checks on fixed source indices 0 and 3 in every condition. All saved records passed raw normalization and common-prefix causality checks. The test-law construction preserves the exact random label mapping and continues the current state across the parameter boundary.

Source summary: `/Users/ruediger/Projects/layered_architectures/output/product_shift_2026-09-18/summary.json`. `provenance.json` records its checksum and the generated figure checksums.
