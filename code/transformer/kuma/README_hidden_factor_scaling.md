# Hidden-factor follow-up: training budget and model size

The 2 × 8 pilot shows a long-context gap from informed Bayes. These two
comparisons test whether additional training or a larger transformer reduces
that gap. The existing Bayesian calculation and test sequences are reused.

| Run | Architecture | Training sequences per seed | Start |
|---|---|---:|---|
| Existing baseline | 2 layers, width 128, 4 heads; 531,968 parameters | 6.4 million | Already complete |
| `longsmall` | Same as baseline | 25.6 million total | Continue the existing model and Adam state; 19.2 million new sequences |
| `large` | 4 layers, width 256, 8 heads; 3,429,888 parameters | 6.4 million | Fresh model; same indexed training batches as baseline |

Both arms use seeds 0 and 1, batch size 64, sequences of length 1025, float32
training and the same source prior. Continuation retains the original final
learning rate, 0.00003, with no restart. The larger model uses the original
warmup/cosine schedule. This is a test of additional low-rate training versus
a larger architecture; a negative continuation result does not exclude other
optimization schedules. The architecture comparison changes depth, width and
head count together.

The primary comparison uses the final planned checkpoints: 400,000 total
updates for `longsmall`, 100,000 for `large`. Intermediate weights at 200,000
and 50,000 respectively are retained for diagnosis, not selected by test loss.
Monitoring every 5,000 updates uses the original separate validation stream.

## Cluster sequence

The user runs one command at a time on Kuma. Update the existing hidden-factor
checkout after this code is pushed; its ignored baseline outputs stay in place.
All follow-up outputs go to a new directory, leaving the original results intact.

First measure the larger model's throughput and GPU memory with 1,000 updates:

```bash
bash code/transformer/kuma/submit_hidden_factor_scaling.sh --benchmark --submit
```

The debug allocation is 30 minutes. Its separate checkpoint is not used for
production training. A successful cap writes
`output/hidden_factor_scaling_2026-09-16/benchmark/large/seed0/benchmark.json`.
Inspect that result before submitting production. Its time projection measures
steady training only; monitoring, checkpoints and evaluation add overhead.

Then submit both arms and both seeds:

```bash
bash code/transformer/kuma/submit_hidden_factor_scaling.sh --submit
```

The array maps tasks 0/1 to continued small-model seeds 0/1, and tasks 2/3 to
larger-model seeds 0/1. Each has one H100, 8 CPU cores, 32 GiB host RAM and a
six-hour limit. The limit is a reservation, not a runtime estimate. Signals
save a resumable checkpoint; no automatic resubmission occurs. Optional
`--arm longsmall` or `--arm large` submits only that pair, reusing completed
work and resuming interrupted runs with the same plan and code.

Each completed training job immediately evaluates its final model on the
**same 256 sequences** as the existing pilot. No Bayesian inference runs here.
Evaluation writes into separate arm/seed directories and resumes saved work.
Only after all four jobs complete, form the paired comparison and figures:

```bash
bash code/transformer/kuma/submit_hidden_factor_scaling.sh --summary --submit
```

This last command reduces saved records; it does not submit a GPU job. It
refuses incomplete comparisons. Omitting `--submit` prints commands without
creating experiment directories or contacting Slurm.

## Paths and checks

Defaults, relative to the selected checkout:

- Frozen plan: `code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json`.
- Read-only baseline: `output/hidden_factor_2026-09-16/2x8/`.
- New outputs: `output/hidden_factor_scaling_2026-09-16/`.
- Override locations with `--baseline-root` and `--out` if using another checkout.

Continuation verifies the original full `latest.pt` against the known evaluated
`model_100000.pt`, including source/configuration, weight identity, optimizer,
random state and runtime. Parent checkpoints are never rewritten. Configurations,
code hashes, parent identities and the frozen plan are retained with each run.

The comparison reports next-token regret and cumulative loss differences from
informed Bayes, plus paired improvement over the same-seed original model. It
keeps both seeds, all six observation times and all four original entropy groups.
Uncertainty is across whole independent sequences.

The 256-sequence pilot motivated this follow-up and is now **development data**.
It gives an efficient paired diagnosis, not independent confirmation of a
model selected using its results. The remaining frozen sequences can provide
confirmation later; no additional Bayesian references are launched here.
