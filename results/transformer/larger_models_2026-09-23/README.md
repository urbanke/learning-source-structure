# Larger product, uniform-base and pooled transformers

All six final checkpoints completed on Kuma with exit code zero. The models
have four layers, width 256, eight heads and 3,454,464 parameters. Seeds 0 and
1 are reported separately, without selecting the better run.

Product and uniform-base models retain the two-layer models' priors, optimizer,
schedule and 100,000 updates of 64 sequences: 6.4M sequences. Training draws
are fresh; identical historical batches are not asserted. Pooled models use
the same indexed training streams as their two-layer counterparts, with
300,000 updates: 19.2M sequences, balanced equally across the three families.

Every comparison uses the frozen published evaluation cohort. Product has
256 sequences per depth pair; uniform-base retains all 1,024 histories grouped
by source depth. Pooled results average equally across three groups of 1,024.
Differences are formed per sequence before averaging. Their standard errors
preserve pairing and, for pooled results, stratification. No seed averaging
is performed. These errors describe test-sequence variation, not training
randomness.

`curves.npz` contains both runs' means, standard errors and paired differences.
`manifest.json.gz` binds the release files. `provenance.json.gz` preserves the
complete audit, training configurations, final checkpoint hashes, endpoint
table, input hashes and exact validation script. Generated JSON reports stay
local; the compressed provenance retains the evidence for recovery.

The independent audit checks all 46 pooled cohort evaluations and all four
specialist score arrays, complete training counts, source identities, code
bindings, final checkpoint hashes and causal-prefix checks. Every published
mean and standard error is recomputed from the per-sequence scores; the old
small-model and Bayesian summaries are reproduced to 2e-12. It does not run
new model inference or Bayesian calculations.

Reproduce after restoring the small transferred bundle under
`output/kuma_completion_2026-09-23/raw` and the original frozen cohorts:

```bash
python code/scripts/release_large_transformer_results.py --check-only
python code/scripts/section3_compact_figure.py
python code/scripts/section3_compact_figure.py --check-only
```

Remote final checkpoints and raw evaluations remain under Kuma's
`~/Projects/layered_architectures_hidden_factor/output/`, in the
`pooled_transformer_large_2026-09-22` and
`specialist_transformer_large_2026-09-22` directories. Training arrays were
4401572 and 4401633; all six allocations completed successfully.
