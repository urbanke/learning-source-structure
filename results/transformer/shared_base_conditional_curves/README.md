# Shared-base LSA: conditional comparison

Each panel fixes alpha=1 and L=0,1,4,8. Each uses all 256 independent test
sequences of length 1025, with fresh matrices and label permutations.
Both previously trained transformers are evaluated at times
1/4/16/64/256/1024. The selected parameters are hidden from the predictors;
Bayes retains the full training prior. No training was repeated.

Bayesian points are now complete at **1/4/16/64 for every depth**. Times 1
and 4 use deterministic rank marginalization and validated integration.
Times 16 and 64 use independently seeded full-target sampling with 16- and
32-node exponent rules. At time 64, all 2048 effective references and all
11 numerical checks per depth (44 total) pass. Scale-invariant depth Rhat
was corrected uniformly; all other criteria are unchanged. Twenty-two
remaining references were continued from saved chains to fixed targets.
No source sequence or failed case was omitted.

At time 64, training run 1 minus Bayes is 0.2167, 0.1149, 0.0122 and
0.0892 bits for L=0,1,4,8. See
`../shared_base_time64_complete_2026-09-17/summary.json` for both runs and
paired standard errors. Loss is averaged over every possible next token
using the true current transition row; standard errors are across complete
independent sequences. Numerical diagnostics do not prove global posterior
convergence.

The active appendix figure is
`figures_t64_complete/provisional_shared_base.pdf`; its receipt binds the
complete corrected export. It shows both transformer runs and only validated
Bayesian points. Full-cohort Bayes references at 256/1024 remain pending.
Other alpha values have not been established by this comparison.

The original `figures/`, `figures_t16/`, `figures_t64/`, and the morning
L0/L1 audit remain unchanged. Raw source/model records, original chains,
corrected audit receipts and copied-chain continuations remain under
ignored `output/` directories. Reproduction and verification commands are
in `results_manifest.json`; the complete time-64 result has its own README.
