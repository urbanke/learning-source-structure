# Shared-base LSA: paired shuffled-training control

Train only the two missing control models. The original larger ordered-source
models are already complete in `output/shared_base_scaling_2026-09-17/train`.
Each new run uses the original seed (0 or 1), architecture (four layers,
eight heads, width 256), model initialization, optimizer, 100000 updates,
batches of 64 length-1025 sequences, and learning-rate schedule. It sees the
same 6.4 million original generated sequences, in the same update and row
positions, after the agreed per-sequence tail-position shuffle.

## What is paired

The unchanged original `train_shared_base_scaling.draw_batch` reconstructs
batch `i` from the original seed and update index. The source prior remains
shared-base LSA, d=64, alpha uniform on [.5,1.5], common source depth uniform on 0..8,
and a fresh alphabet relabeling per sequence. This is not the uniform-base
source, a fixed-depth source, or a new draw merely from the same prior.

After generating the complete original batch of 64, keep each first token
and shuffle its other 1024 positions with an independent RNG indexed by
training seed, update, and row. Shuffle RNGs never touch the source stream or
Torch initialization. Inputs and next-token targets both come from the
shuffled sequence. Every sequence keeps exactly its original token counts.
The finite shuffled tail is exchangeable, not iid from the stationary law.

The original training did not save every token batch. Pairing follows by
reusing its frozen indexed sampler and runtime, and is checked by independent
reconstruction at five full-size update indices for each seed (first two,
1000,49999,99999). Frozen fingerprints in the plan are reconstructed checks,
not hashes recorded at the historical run. Training verifies the original
checkpoint/config/status/runtime hashes, frozen source and training code,
NumPy 2.4.6, Torch 2.14.0+cu126 and CUDA 12.6 before any update. A differing
fingerprint stops the run rather than silently weakening the pairing claim.

This exact pairing is to the **larger** ordered models. The older two-layer
models used worker-dependent historical streams that cannot be reconstructed
in this way. No original model is retrained or overwritten.

## Launch on Kuma, one command at a time

Use the existing `layered_architectures_hidden_factor` checkout, which contains
the completed larger shared-base runs. Update it from GitHub first. The
launcher defaults to print-only; `--submit` is required to send a job.

1. Short GPU benchmark (1000 updates, separate weights):

   ```bash
   bash ~/Projects/layered_architectures_hidden_factor/code/transformer/kuma/submit_shared_base_shuffle.sh --benchmark --submit
   ```

2. Inspect the completed benchmark log. Its final receipt must say
   `passed: true`, confirming the intended 1000-step cap, paired original
   inputs and measured GPU throughput/memory. The trainer's exit 75 at the
   deliberate cap is expected; the Slurm benchmark wrapper converts a
   validated cap into successful completion.

3. Submit both production seeds, after the benchmark passes:

   ```bash
   bash ~/Projects/layered_architectures_hidden_factor/code/transformer/kuma/submit_shared_base_shuffle.sh --submit
   ```

Each task requests one H100, 8 CPU cores, 32 GB RAM, and at most 6 hours. Two tasks
can run concurrently. The earlier original-source run took about 1.9 hours;
use this control's own benchmark for its timing estimate. The 6-hour walltime
is an allocation ceiling, not a measured runtime. No Bayesian integration or
model scoring runs on the login node.

If using another checkout, pass `--ordered-root` with the absolute path to
`shared_base_scaling_2026-09-17/train`. The path does not enter the scientific
identity; file hashes do. `--out` selects a new output directory, which must
not overlap the original training directory. Account and environment can be
set with `SHARED_BASE_SHUFFLE_ACCOUNT` and `SHARED_BASE_SHUFFLE_VENV`.

## Outputs and resumption

```
output/shared_base_shuffle_2026-09-17/
  benchmark/seed0/benchmark.json
  train/seed{0,1}/
    config.json
    status.json
    shuffle_runtime.json
    log.jsonl
    latest.pt
    model_50000.pt
    model_100000.pt
  logs/
  environment/
```

`config.json` includes original-run hashes, source fingerprints, shuffle
contract and implementation hashes. `latest.pt` is an atomic resumable
checkpoint with optimizer/RNG state. Resubmitting the same unfinished output
resumes it; a configuration/code/runtime mismatch is rejected. Each update's
data are indexed, so worker count and prefetching cannot change consumed
examples. CPU interruption/resume is tested bitwise; CUDA kernels can still
be nondeterministic. Benchmark weights never initialize production.

Validation uses the same held-out original sequences as the ordered model,
with an independent fixed validation shuffle. It does not use the scientific
test cohorts. Both seeds and the planned final 100000-update checkpoint are
retained regardless of their scores. The separate evaluation launcher reads
these checkpoints and tests both predictors on the same original ordered
sequences. Before evaluation, transfer the frozen cohort files from the laptop:

```bash
rsync -av ~/Projects/layered_architectures/output/shared_base_shuffle_2026-09-17/cohorts/ urbanke@kuma.hpc.epfl.ch:Projects/layered_architectures_hidden_factor/output/shared_base_shuffle_2026-09-17/cohorts/
```

The destination parent exists after the benchmark is submitted. These files
contain the exact original tokens and source probabilities. The evaluator
validates their checksums and reuses them without regeneration, avoiding
cross-platform floating-point differences in source construction. A local
backup bundle is `output/shared_base_shuffle_cohorts_2026-09-17.tar.gz`;
extracting it into the control output root creates the same `cohorts/` folder.
The cohort manifest is committed with the code. If files are absent, the
compute job can reconstruct them, but requires exact checksum agreement.

The evaluation can be queued immediately after the production array is submitted,
using its returned job ID in place of JOB_ID:

```bash
bash ~/Projects/layered_architectures_hidden_factor/code/transformer/kuma/submit_shared_base_shuffle_eval.sh --after-job JOB_ID --submit
```

Its `afterok` dependency waits for both training seeds to complete. Without
`--after-job`, the launcher requires both final checkpoints already present.
All source verification, any reconstruction, and inference run on the allocated compute node.
The evaluation creates `cohorts/`, per-model files in `eval/`, and
`comparison/summary.json` with `gap_seed0.png`, `gap_seed1.png`,
`regret_seed0.png` and `regret_seed1.png`. The gaps are the expected loss of
the shuffled-training control minus that of its paired ordered-training model;
they do not claim a recovered internal copier weight. The archived original
LSA crossing is not substituted for the missing matched Bayesian curve.

## Local validation without scientific training

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q code/tests/test_shared_base_shuffle_training.py code/tests/test_shared_base_shuffle_benchmark.py code/tests/test_shared_base_shuffle_launchers.py
```

Checks include exact original-batch replay, preserved first token/counts,
independent permutations per sequence, unchanged source/model RNG state,
matched initialization, shifted shuffled targets, and bitwise resume across
worker counts against an independently written short training calculation.
Launcher tests use fake Slurm and never submit GPU jobs. Production plan
validation rejects a different source, schedule, model or implementation.
The separate `code/tests/test_shared_base_shuffle_eval.py` suite checks oracle
row/target indexing, paired cumulative uncertainty, missing/multiple crossings,
checkpoint pairing, exact transferred-cohort reuse, and an end-to-end smoke
with visibly labeled untrained tiny models. Smoke figures are not results.
