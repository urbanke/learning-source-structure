# Shared-base training campaign, 18 September 2026

This campaign addresses the remaining shared-base LSA prediction gap. It uses the
same prior and absolute indexed training batches as the completed 4-layer runs:
64 symbols, 1024 predictions per sequence, fresh alpha uniform on [0.5, 1.5], depth uniform
on 0,...,8, and a fresh rank permutation and source for every sequence.

| Arm | Architecture | Parameters | Updates | Saved comparison updates |
|---|---|---:|---:|---|
| continue_large | 4 layers, width 256, 8 heads |3,454,464|100k→400k|200k,400k|
| xlarge | 8 layers, width 512, 8 heads |25,809,920|0→400k|100k,200k,400k|

Both seeds 0 and 1 are prescribed; report them separately. The original 100k
four-layer weights remain the baseline. At batch 64 the final exposure is
25.6 million sequences per seed. The continuation adds 19.2 million sequences.

Continuation loads `latest.pt` from the previous scaling directory, checks that
its weights match the pinned `model_100000.pt`, verifies the frozen config and
status, and restores the optimizer and every saved RNG. It uses the parent
learning-rate floor, 3e-5, throughout. Fresh larger models use the original
3e-4 peak rate, 1000-update warmup, and cosine decay to 3e-5 at 400k. All arithmetic
and model weights remain float32 with TF32 disabled. No gradient accumulation,
precision reduction, or substituted architecture is implicit.

## Submit the continuation first

On Kuma, from the updated checkout:

```bash
bash code/transformer/kuma/submit_shared_base_campaign.sh --continue --submit
```

This submits two GPU tasks and does **not** wait for a new-model benchmark or
Bayesian calculation. The existing measured throughput implies about 5.6 hours
of additional training per seed, excluding monitoring and checkpoint overhead.
The parent files under `output/shared_base_scaling_2026-09-17/train` are read only.
Every costly parent check occurs on the allocated compute node.

## Benchmark the larger model separately

```bash
bash code/transformer/kuma/submit_shared_base_campaign.sh --benchmark --submit
```

The benchmark uses a separate seed-0 directory and the full 400k schedule capped
at 1000 updates. It must return `passed: true`. Its weights never initialize
production. At least 8% of GPU memory must remain unreserved. A throughput-only projection above 72 hours refuses production and
requires a separately declared smaller architecture; it never shortens the
400k schedule silently. Validation and checkpoint overhead add to the estimate.

After the benchmark passes:

```bash
bash code/transformer/kuma/submit_shared_base_campaign.sh --xlarge --submit
```

Both arms use 24-hour allocations with a 180-second warning. A warning or step cap
saves optimizer, RNG and absolute batch position and returns 75. This means
incomplete production, not a lost checkpoint. Explicitly resume with:

```bash
bash code/transformer/kuma/submit_shared_base_campaign.sh --xlarge --resume --submit
```

Use `--continue --resume` for a stopped continuation, or `--seed 0`/`--seed 1`
to select an unfinished seed. A complete run is validated and skipped. An
unfinished nonempty directory refuses a fresh invocation. Resume rejects changed
configuration, code, parent identity, runtime or monitoring data. Never remove a
checkpoint to get around such a failure. CUDA kernel nondeterminism is possible;
CPU interrupted/resumed updates are independently tested bit for bit.

## Reuse the frozen diagnostic cohorts

The compact export contains the exact 256 existing sequences at each depths 0, 1, 4, 8,
with alpha 1, and true conditional rows at 1, 4, 16, 64, 256, 1024 observed tokens. The
source parameters never enter the transformer. The export is about 3.1 MiB and is
created locally by checking every original source record and its identity:

```bash
python code/transformer/shared_base_campaign_eval.py export --source-root output/shared_base_conditional_curves_2026-09-15 --out output/shared_base_campaign_inputs_2026-09-18
```

Transfer that directory once from the laptop to the same output-relative path
on Kuma. Its manifest is pinned by the campaign plan. Training tasks score each milestone before continuing automatically if this directory has arrived. If not,
training still completes; submit evaluation afterward on a compute node:

```bash
bash code/transformer/kuma/submit_shared_base_campaign.sh --continue --evaluate --submit
bash code/transformer/kuma/submit_shared_base_campaign.sh --xlarge --evaluate --submit
```

`--after-job JOBID` queues an evaluation after successful training. No source or
Bayesian reference is regenerated. Saved arrays contain source IDs, per-source
expected losses, regrets and model probabilities, enabling paired comparisons
with the original 100k model and with validated Bayesian references when those
arrive. Numerical scoring checks include normalized predictions and causal
prefix alignment. Unavailable Bayesian predictions must remain unavailable.

These cohorts motivated the campaign and are **reused diagnostic data**, not an
independent confirmation set. The primary target is a transformer-minus-Bayes
expected next-token loss within 0.01 bits after 1024 observed tokens, checked
separately for both seeds with paired uncertainty and reference precision. The
full context curve and all prescribed milestones also remain visible. Reduced
gaps would not establish asymptotic convergence or isolate depth from width.

All launchers print a dry run unless `--submit` is given. `--out`, `--plan`,
`--parent-root`, and `--cohorts` provide explicit paths. Changing an immutable
plan invalidates old resume/benchmark contracts; create a separate campaign
rather than editing a running experiment.

Duplicate queued/running submissions for the same output, mode, arm and seed
are rejected using Slurm, including the earlier campaign launchers. Shared
filesystem directory claims additionally prevent simultaneous submissions and
batch starts, and remain held across all training milestones and evaluations.
They do not rely on cross-node `flock`. Other seeds, arms and benchmarks remain
independent; a dry run neither queries Slurm nor reserves anything.

Claims are retained under `.campaign_claims/`. Only confirmed terminal Slurm
allocations permit a new attempt. Unavailable accounting, scheduler errors or
an interrupted submission with no recorded job ID fail closed and show the
owner receipt for inspection. No run data are removed and no job is cancelled.
Use `--after-job` for a separate evaluation while training is queued or running:
evaluation is a distinct mode and its claim does not exclude training's own
milestone evaluator. These guards leave scientific code, checkpoint identities
and the trainer's signal/checkpoint handling unchanged.
