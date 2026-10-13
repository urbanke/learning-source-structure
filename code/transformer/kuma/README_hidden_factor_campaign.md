# Hidden encoding: additional capacity and training

This campaign follows the completed comparison on 3,840 held-out sequences.
It tests two fixed interventions, retaining both training runs separately.

| Arm | Architecture | Total sequences per run | Initialization |
|---|---|---:|---|
| `continue_large` | 4 layers, width 256, 8 heads; 3,429,888 parameters | 25.6 million | Existing 100,000-update larger model, including Adam and random state |
| `xlarge` | 8 layers, width 512, 8 heads; 25,760,768 parameters | 25.6 million | Fresh, same indexed source-batch streams |

Batch size remains 64, context length 1,024, and source dimensions 2 × 8.
Continuation keeps the original learning-rate floor, 0.00003, and adds
300,000 updates. Its saved milestones are 200,000 and 400,000 updates.
The fresh larger architecture uses the original 1,000-update warmup and
cosine schedule over 400,000 updates, saving 100,000, 200,000 and 400,000.
The schedule differs from the earlier 100,000-update schedule: intermediate
comparisons do not isolate model capacity alone.

Every declared milestone is evaluated on the existing 256 exploratory
sequences with their saved exact Bayesian references. These diagnostics help
detect failed learning early. They neither select the better seed nor change
the predeclared final checkpoint. The previously inspected 3,840-sequence
comparison must be described as reused data if evaluated again. It is not a
fresh independent confirmation for this next campaign. New independent
confirmation would require new reference calculations, which this launcher
does not submit.

## Launch and review

Commands print only unless `--submit` is added. Use the existing Kuma checkout.
All checkpoint reads, checksums, inference and training take place on compute
nodes. The login-side launcher only checks paths/small status records and submits.

The existing four-layer continuation can start immediately, independently
of the largest-model benchmark:

```bash
bash code/transformer/kuma/submit_hidden_factor_campaign.sh --arm continue_large --submit
```

Measure the largest model with a separate, uninterrupted 1,000-update run:

```bash
bash code/transformer/kuma/submit_hidden_factor_campaign.sh --benchmark --submit
```

Inspect `output/hidden_factor_campaign_2026-09-18/benchmark/xlarge/seed0/benchmark.json`.
It must report `passed: true`. This requires finite losses, the intended cap,
memory headroom, and a throughput projection no greater than 72 hours for
400,000 updates. The projection excludes queue, checkpoint and evaluation
overhead; the three-day threshold leaves time for results and paper work.
If it fails, review the architecture before production rather than calling
a shorter unplanned run the completed experiment.

After that review, launch the two largest-model runs:

```bash
bash code/transformer/kuma/submit_hidden_factor_campaign.sh --arm xlarge --submit
```

Each production task reserves one H100, eight CPUs, 32 GiB host RAM and up to
24 hours. The first four-layer continuation is roughly 5.6 hours from its
previous measured throughput, excluding overhead. The largest model needs
its own benchmark; parameter count is not an adequate runtime forecast.

Before an allocation ends, the trainer receives a stop signal and saves its
optimizer, random state, exact completed batch index and model. Exit 75
means a saved, incomplete run. Submit the same arm again after its jobs stop
to resume. There is **no automatic resubmission**. Completed milestones and
their verified evaluations are reused. Final completed runs are omitted from
subsequent submissions. `--seed 0` or `--seed 1` limits a retry to that run.

All new outputs go to `output/hidden_factor_campaign_2026-09-18/`.
The original scaling runs and reference data stay read-only. Override paths
with `--out`, `--parent-root` and `--baseline-eval`. The evaluator never runs
Bayesian inference. Existing source/model/training modules are immutable;
new adapters pin their hashes and verify the parent against its evaluated
100,000-update weights before inheriting optimizer state.

Submissions now reject queued/running jobs with the same output, mode, arm and
seed, including jobs launched before this protection was added. Atomic shared
filesystem directory claims cover the full allocation, including every training
milestone and its evaluation; protection does not depend on cross-node `flock`.
Different seeds, arms and the separate benchmark remain independent. Dry runs
do not query Slurm or reserve a claim. The guard never cancels jobs.

Small receipts live under `.campaign_claims/` in the campaign output. Claims
are retained, and a new attempt is allowed only after Slurm confirms the old
allocation is terminal. Missing accounting, scheduler errors and interrupted
submissions without a recorded job ID stop safely and report the claim owner.
Inspect the reported job and receipt before retrying; never delete training
files or clear an uncertain claim to bypass protection. The guard changes no
scientific implementation, checkpoint identity, or USR1 handling.
