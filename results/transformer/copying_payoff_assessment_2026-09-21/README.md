# Copying-payoff aggregate precision diagnostic

This is a compact receipt for the 21 September 2026 copying-payoff recovery
and assessment. It is **not a publication result** and releases no Bayesian
payoff crossing.

The two previously failing references were continued from 16,384 to 65,536
retained draws per chain without changing their seeds, sampler, runtime or
acceptance criteria. Both pass. Re-merging the complete time-6--8 campaign
retained 974 parent records, substituted exactly those two continuations, and
passed all 976 individual audits and all 488 paired integration-order checks
(64 probability coordinates per pair).

The final eight-step assessment also passes its input and integrity checks,
including 5,327 transitively hashed claims. Its aggregate numerical precision
gate fails, however. The main limitation is the common stationary-frequency
bank and its coarse/fine refinement; the independent comparison between 16-
and 32-node exponent quadrature is also material at times 5--8. Because every
instantaneous and cumulative value
used for a crossing must pass, all Bayesian crossing estimates and bootstrap
intervals are withheld.

`diagnostic_summary.json` records the exact failed coordinates, recovery
diagnostics and hashes of the local source receipts. The large chains, arrays
and transitive hash inventories remain in ignored local output directories:

- `output/copying_payoff_recovery_procinfo_2026-09-21/`
- `output/copying_payoff_recovery_diagnostics_2026-09-21/`
- `output/copying_payoff_assessment_2026-09-21/`

The next calculation should refine the stationary-frequency bank/coarse-fine
comparison at the listed failed coordinates and independently tighten the
comparison between 16- and 32-node exponent quadrature at times 5--8, then
rerun the unchanged aggregate gate.
