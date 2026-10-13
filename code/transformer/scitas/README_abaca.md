# Independent ABACA accuracy check

The completed CPU calibration passed five of six exact five-token controls.
For ABACA, the estimated error was 0.005314 bits and the error including its
approximate 95% uncertainty was 0.010028 bits, just above the fixed 0.01-bit
criterion. This follow-up was chosen after seeing that result.

Run **128 fresh independent repetitions with 16,384 particles**, using the
original CPU sampler, prior, fixed ABACA tokens, temperature/move settings,
quadrature and sealed LSA store. This doubles the particles and repetitions
relative to the original check. The new master seed is 2026091650. Complete
the fixed batch and assess it once, retaining the outcome whether it passes
or fails. There is no automatic retry, budget increase or later stage.

The original six-control plan and results remain unchanged. The new plan is
`code/transformer/configs/shared_base_abaca_plan_2026-09-16.json`; its origin
field binds the original plan. It is a standalone numerical diagnostic,
so it can run independently of the original failed calibration gate. It does
not unlock that plan's confirmation, quadrature or longer-sequence stages.

## Run on Jed, one command at a time

After updating the existing `layered_architectures_mixture_bayes` checkout,
preview the allocation:

```bash
bash code/transformer/scitas/submit_shared_base_abaca.sh
```

Submit only this ABACA batch:

```bash
bash code/transformer/scitas/submit_shared_base_abaca.sh --submit
```

This uses 32 CPU tasks, four repetitions per task, at most 32 concurrent,
with the existing Jed standard/serial allocation: one CPU, 6 GiB RAM and
two hours per task. The time limit is a reservation. All tasks reuse the
prepared numerical environment and sealed store; there is no retraining or
new source sampling. Outputs are under
`output/scitas_shared_base_abaca_2026-09-16/` in the selected checkout.

After every task finishes, merge the full batch:

```bash
bash code/transformer/scitas/submit_shared_base_abaca.sh --merge --submit
```

Then compare it with the original six-control calibration:

```bash
bash code/transformer/scitas/submit_shared_base_abaca.sh --review --submit
```

Merge and review reduce saved records; they perform no numerical sampling.
Omitting `--submit` always prints the command only. Sampling supports
`--only-task INDEX` and `--wall-minutes MINUTES` for a diagnosed interruption.
Existing completed repetitions are verified and reused. Wait for the old
task to stop before resubmitting it; shard boundaries remain four repetitions.

## What the comparison means

The review checks both complete receipts under their own frozen plans and
recomputes their statistics from raw likelihood estimates. It preserves the
original ABACA failure, reports the old and new estimates separately, and
lists the evidence used for each of the six controls: the unchanged original
results for five cases and the new ABACA result. The two particle budgets are
not pooled into one estimate. Changed or incomplete records are rejected.

Both use the same 0.01-bit error-plus-uncertainty criterion and the same
minimum effective repetition count and maximum single-repetition share.
The review reports whether all six controls meet those criteria with the
new ABACA result. Its output is `comparison.json` in the new output directory.
This does not rewrite the old gate or approve production references. Remaining
confirmation and quadrature checks must be assessed separately. These short
controls do not establish accuracy or runtime for longer sequences.
