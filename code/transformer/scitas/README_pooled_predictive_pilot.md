# Bounded predictive-mixture prototype

This extends the existing shared-base SMC calculation with its full terminal
next-token prediction. It does not change a trained model, the shared-base
prior, the sealed LSA tables, or the existing hash-locked samplers. It is not a
validated population reference and submits no jobs by itself.

The selection is 64 existing independent full-prior sequences from each family:
indices 5..68, 192 total, with observation times 16 and 64. Existing pilot
indices 1..4 in each family are recorded separately for development. All
families and all selected cases must remain in the eventual comparison;
numerically convenient cases must not replace failures.
The transformer curves for these existing cohorts have already been inspected
in aggregate. This is a fixed balanced test subset, not fresh confirmation data.

Freeze the selection once locally or inside a compute allocation (this reads
the full source arrays and hashes their files):

```bash
python code/scripts/pooled_predictive_pilot.py freeze \
  --raw output/multifamily_results_2026-09-16 \
  --out output/pooled_predictive_pilot_2026-09-18/selection.json
```

The cluster equivalent `--raw` is the existing multifamily result root
containing `heldout/` and `eval/pooled_seed{0,1}/`. Use the verified shared-base
dependency bundle's `src/` and `tables/anchors_prod/` directories for the two
kernel arguments. Use the identical fixed indices when freezing on the compute
node, or transfer the frozen selection from the laptop.
Paths can differ between machines; file contents must match their SHA256.

For the prepared Jed checkout, this lightweight launcher submits three
one-hour CPU tasks, one existing development history per family at context 16:

```bash
bash code/transformer/scitas/submit_pooled_predictive_pilot.sh --submit
```

By default it reuses the existing pooled evaluations under
`~/Projects/layered_architectures_multifamily/output/multifamily_2026-09-16`
and the already unpacked `~/Projects/shared_base_dependencies_2026-09-16`.
Use `--raw` and `--dependencies` to override those paths. It creates the fixed
selection inside each compute allocation and runs only development index 1,
with 32 particles, eight temperatures, one move and two independent repeats.
Outputs go to `output/pooled_predictive_pilot_2026-09-18/t16/`.
The log reports total elapsed time including preparation; the result files
separate sampling from terminal-prediction time. This is an execution and
timing test, never an accuracy approval or a population submission.
After reviewing it, `--observed 64 --submit` requests the analogous bounded
context-64 test. Omit `--submit` to inspect the command without making changes.

Alternatively, run one **development** case in a compute-node allocation, beginning with a
small bounded benchmark. For example, after setting the actual paths:

```bash
python code/scripts/pooled_predictive_pilot.py case \
  --raw "$POOLED_RAW" --plan "$POOLED_SELECTION" --out "$POOLED_PILOT_OUT" \
  --family product --index 1 --development --observed 16 \
  --store "$LSA_STORE" --src "$PMM_SRC" \
  --particles 32 --temperatures 8 --moves 1 --replicates 2 \
  --seconds-per-replica 300
```

These settings test execution and timing only. They are not a scientific
accuracy budget. Repeat the bounded development benchmark across the three
families; inspect sampling time separately from terminal-prediction time.
Only then benchmark time 64 and decide validated effort/resources. The CLI
does not loop over the 192 selected test histories. It refuses indices outside
the frozen selection unless `--development` selects the separate existing
pilot histories. No population job has been launched by this implementation.

Each result lives in a configuration-hashed subdirectory with source/model,
code and sealed-store identities, the two inexpensive Bayesian components,
individual SMC repetitions, and an updated result/status JSON. Every completed
repetition is written atomically. Restarting the same command reuses those
repetitions; an interrupted repetition restarts with the same seed. There is
no within-repetition checkpoint in this prototype. Increasing the deadline
preserves its identity; changing the numerical effort creates a separate run.

For a repetition with evidence estimate Z and terminal posterior prediction p,
the aggregator averages Z and Z*p separately. Their ratio supplies the
shared-base prediction; its absolute evidence updates the three family weights.
Existing product/uniform-base predictions are reconstructed on that exact
history. The result reports actual Bayesian-mixture next-token regret, both
saved transformer regrets, their paired differences, and a delta-method
Monte Carlo standard error for the mixture regret that preserves covariance
between Z and Z*p. This diagnostic does not bound missed-mode or quadrature
error and does not replace between-sequence uncertainty.

The alternative observed-next-token log loss needs fewer successor evaluations
but has more sequence noise; it is a different estimator than the expected
regret plotted in Figure 1. The primary prototype retains all 64 probabilities.
Its prediction overhead may be substantial: the existing row routine can
evaluate 64 evidence increments for each distinct terminal latent state.
No population runtime is inferred from the earlier five-token GPU benchmark.

Focused checks (no large posterior calculation):

```bash
python -m pytest -q code/tests/test_shared_base_predictive_smc.py
```

Tests include identical seeded normalizer/move counts against the original
SMC, independent finite-state successor evidence, evidence-weighted pooling,
posterior family weights, all-three-family exact-short fixture reconstruction
at times 1 and 4, invalid-probability rejection, frozen cohort identity and
interruption/restart with preserved completed repetitions. The test restart
fixture is synthetic and is never a scientific result.
