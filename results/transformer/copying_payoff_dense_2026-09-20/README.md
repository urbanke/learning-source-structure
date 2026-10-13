# Complete-cohort copying-payoff references

This bounded local calculation supplies full informed shared-base Bayesian
predictions on the same 768 ordered histories: 256 each at alpha=1 and depths
0, 2 and 3. Its source prior, sampler, numerical runtime and original histories
are unchanged. It follows the completed 42-task accuracy diagnostic.

The calculations now cover times 5 through 8. Time 5 has 38 distinct ordered
equality patterns at both 16- and 32-node exponent integration orders, hence
76 task records. The parallel continuation adds 162, 318 and 496 records at
times 6, 7 and 8. Alphabet-equivalent histories share predictions without
dropping any sequence or changing the queried context. The source-frequency
comparator and both neural model pairs use the same 768 histories.

Each task uses four chains, 16,384 warmup updates and 16,384 retained draws per
chain. Every original task had one allocation of 900 seconds and no automatic
retry or draw increase. Time 5 used one worker. Times 6 through 8 used a
separate bounded four-worker campaign with a ten-hour overall deadline. That
campaign completed all 976 declared records in 22,520.4 seconds.

From the repository root:

```bash
output/unseen_context_refinement_runtime_2026-09-19/bin/python code/scripts/copying_payoff_dense.py prepare --time 5
output/unseen_context_refinement_runtime_2026-09-19/bin/python code/scripts/copying_payoff_dense.py run --time 5
```

The default outputs are under ignored `output/copying_payoff_dense_2026-09-20/t5`.
`run --only-task N` executes one declared task; later `run` skips every existing
attempt, including a timeout or numerical failure. Frozen parent files are
never modified. `summary` rehashes all saved arrays without sampling.

**Completed task coverage does not approve a population mean or payoff-time
forecast.** The saved plans carry all 768 mappings. Diagnostic mergers combine
the oracle-weighted loss gradients of histories sharing a canonical reference,
rather than treating mapped uses as independent draws. They also check paired
integration orders and retain numerical arrays for the final covariance and
bias assessment. Times 5 and 6 pass their individual and paired-order checks.
Time 7 has one record below the probability-ESS threshold; time 8 has one
record below the ESS threshold and above the probability-MCSE threshold. Those
two saved chains must be continued before the aggregate cross-time covariance,
refinement and payoff-time assessment. No available-case mean is released.

Tests: `python -m pytest -q code/tests/test_copying_payoff_dense.py`.

Local preflight on 2026-09-20 passed all 16 focused tests. One fresh reference
(task 2) completed its full draw budget and raw/grid audits in 68.3 seconds,
with peak resident memory 0.87 GiB. One reused reference (task 0) passed the
same audit in 2.24 seconds at 0.47 GiB, with byte-identical saved chains and no
sampling. Both remain part of the prepared wave; a later full launch skips
their finished attempts. These are individual checks, not aggregate approval.

The separate `code/scripts/merge_copying_payoff_dense.py` is now prepared.
It requires all 76 references, all 768 histories, individual raw-data audits
and the 38 paired-order checks. It combines gradients for shared canonical
references, reports chain/batch covariance and log-bias proxies, and keeps
source-bank and sequence errors separate. It saves diagnostic arrays only
for complete validated input. Exit status is 3 for missing coverage, 2 for
failed numerical audits, and 0 for complete validated diagnostics. It does
not adopt a new loss-precision threshold or infer an onset from one point.
Twenty merger tests and an independent covariance review pass. Run only
after the wave ends:

```bash
output/unseen_context_refinement_runtime_2026-09-19/bin/python code/scripts/merge_copying_payoff_dense.py
```

Current state: time 5 has 76/76 passing records and time 6 has 162/162.
Time 7 has 317 passing records plus one saved-chain continuation target,
`t0007_453a8f0bdcd20ca9731b_a32`. Time 8 has 495 passing records plus target
`t0008_fccb61c7bef875d43ac7_a16`. No local worker remains active, and neither
the population reference nor a payoff forecast is approved.
