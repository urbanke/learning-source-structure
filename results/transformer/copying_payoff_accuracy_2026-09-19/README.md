# Bounded copying-payoff Bayesian accuracy stage

Completed 19 September: all 42 tasks and 21 integration-order comparisons
passed the independent accuracy diagnostics. The complete-cohort payoff
calculation has not yet run, and no Bayesian onset time is approved.

The cost pilot completed twelve calculations, but two failed numerical checks.
This separate stage keeps all twelve cases and tests the sampler against exact
short-prefix references before planning a dense population comparison.

- Thirty fresh calculations: all fifteen four-token equality patterns, each
  with alpha integration orders 16 and 32, compared to the saved exact catalog.
- Twelve continuations: copies of all pilot checkpoints, extending each chain
  from 4,096 to 16,384 retained draws. Original samples, burn-in, endpoint and
  random-generator states are verified and preserved.
- One worker; 900 seconds per calculation; one attempt; 37,800 seconds maximum
  reservation (10.5 CPU hours). No automatic retry or budget escalation.

The original shared-base prior and numerical sampler are unchanged. Each
result is independently reconstructed from raw chains, with the established
scale-invariant depth diagnostic, mixing/ESS/precision checks and widened row
grid checks. Exact-vector and between-order differences receive a stated
simultaneous normal-MCSE/Bonferroni diagnostic, with deterministic catalog
uncertainty separate. MCMC errors are estimates, so this is not a rigorous
coverage guarantee. The summary requires all 42 tasks and 21 order pairs.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 output/unseen_context_refinement_runtime_2026-09-19/bin/python code/scripts/copying_payoff_accuracy.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 output/unseen_context_refinement_runtime_2026-09-19/bin/python code/scripts/copying_payoff_accuracy.py run
python -m pytest -q code/tests/test_copying_payoff_accuracy.py
```

Use `run --only-task 0 --only-task 34` for the declared operational smoke:
one fresh exact-prefix task and the continuation of a numerically unresolved
pilot task. A subsequent full `run` skips every already attempted task. Raw
outputs and the immutable plan remain under ignored
`output/copying_payoff_accuracy_2026-09-19/`. Neither a successful scheduler
exit nor a complete accuracy diagnostic approves a payoff-time forecast.
Dense complete-cohort predictions, correlated numerical-error accounting for
shared canonical references and independent timing confirmation remain next.
