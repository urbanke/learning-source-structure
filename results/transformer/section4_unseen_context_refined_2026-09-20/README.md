# Completed unseen-context refinement

All twelve depth/time groups now pass. The two restored groups retain all
66 depth-four and 57 depth-eight histories at time 64, with both integration
rules. Numerical allowances remain below the unchanged 0.003333-bit budget.
One saved-sample repair took 119.22 seconds; original outputs are preserved.

Only 123 Bayesian loss rows changed in the 1699-row table. Neural scores and
all other groups are unchanged. Independent replay checks 432 means/SEs.
Error bars measure sequence variation; fresh confirmation remains outstanding.

Reproduce the data with the frozen local runtime:

```sh
output/unseen_context_refinement_runtime_2026-09-19/bin/python code/scripts/publish_unseen_context_refinement.py --recovery output/unseen_context_recovery_2026-09-20
```

The publisher requires a fresh output directory. Use `--verify-only` to check
an existing release. Figure commands and provenance are in results_manifest.json.
