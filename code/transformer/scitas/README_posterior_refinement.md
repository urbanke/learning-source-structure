# Targeted shared-base posterior refinement

This follows Jed revalidation 66550732: 31/32 individual references passed,
but four of eight depth/time comparisons still failed numerical checks.
It concerns conditional shared-base Bayesian predictions, not the separate
three-family mixture likelihood calculation or transformer training.

The fixed plan is
`code/transformer/configs/shared_base_posterior_refinement_plan_2026-09-17.json`.
It retains the same two source sequences per depth. The nine tasks are:

| Tasks | Condition | Work |
|---|---|---|
| 0–1 | L4, sequence 1, time 256, orders 16/32 | Continue 131072 → 524288 draws per chain |
| 2 | L8, sequence 0, time 256, order 32 | Continue 131072 → 262144 |
| 3–4 | L8, sequence 1, time 256, orders 16/32 | Continue 65536 → 262144 |
| 5 | L4, sequence 0, time 1024, order 32 | Continue 262144 → 524288 |
| 6–7 | L0, both sequences, time 1024, order 64 | New finer integration rule; 16384–65536 draws |
| 8 | L0, sequence 1, time 1024, order 32 | Independent repeat; 16384–65536 draws |

## Why these calculations

At L4/time256, sequence 1 contributes most of the numerical variance. Its
fourfold extension should reduce the cohort's numerical allowance to about
0.0024–0.0025 bits under usual inverse-draw variance scaling. At L8/time256,
equalizing the shorter chains at 262144 draws should bring it to about
0.0024–0.0027 bits. These forecasts are below the current 0.00333-bit budget;
they are not guarantees. The actual budgets and errors are recomputed.

At L4/time1024 only sequence 0/order32 remains below the probability ESS
threshold of 200 (observed 159.68). Doubling its saved draws is a bounded
attempt to improve that check. At L0/time1024, sampling error is already
small: an integration refinement and independent repeat are more informative
than simply adding samples to all existing calculations.

## Reuse, comparisons and acceptance

The existing numerical worker is unchanged. Each continuation copies its
verified checkpoint, report and moments into a separate directory and
preserves its prior, source, sampling seed, proposal, burn-in and RNG state.
Setting both `--initial-draws` and `--max-draws` to the new target forces the
extension even if the previous reference was ready. Original artifacts stay
unchanged. The existing sealed LSA store and compiled implementation are used.

Each new result receives the full raw audit, the checked cross-platform
probability-loader adapter, and the same corrected depth Rhat arithmetic.
All other numerical checks remain unchanged. A legacy worker exit code 2
can reflect its old depth diagnostic; the full corrected audit determines
readiness. A failed or unfinished calculation is never silently replaced
with its earlier estimate in a merged comparison.

All eight fixed depth/time groups remain in the summary. For L4/L8 the merge
replaces only the six declared continuations; other references are reused.
For L0/time1024, the primary refinement is original order32 versus new
order64, with the original failed order16/32 comparison retained as history.
The independent order32 repeat is a consistency check, never a selectable
replacement for the original. Its agreement and precision must pass before
L0/time1024 can pass. Both original source sequences remain in every group.

The repeat uses the original per-case agreement allowance,
`5 sqrt(var_original + var_repeat) + 2 (bias_original + bias_repeat) + 0.0002`
bits, along with a numerical allowance
`2 sqrt(var_original + var_repeat) + 2 (bias_original + bias_repeat)` below
0.01/3 bits. Both individual corrected audits must also pass. Here the bias
terms are the existing local Jensen diagnostics, not rigorous bias bounds.
All constants are fixed in the plan before the new runs.

L4/time256's integration difference may persist after the noise is reduced.
The batch can expose that problem; it does not promise eight passing groups.
Even a fully passing result is a computational pilot, not a population
comparison with the transformer.

## User-led launch on Jed

After updating the checkout, preview without submission:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior_refinement.sh
```

Submit the nine-task array:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior_refinement.sh --submit
```

Each task reserves up to eight hours, four CPUs and 24 GiB. The numerical
worker uses one thread; the additional CPU allocation provides the memory
needed for raw arrays and audit temporaries at the larger draw counts.
No heavy calculation runs on the login node. The immutable launch binds the
actual completed revalidation and original inputs on Jed. Checkpoint copies
and results go to `output/shared_base_posterior_refinement_2026-09-17`.

After the array finishes, submit the merge to a compute node:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior_refinement.sh --merge --submit
```

Inspect completeness and numerical checks separately from scheduler exit
codes. There is no automatic retry, further budget increase, new training,
or full-cohort production launch.
