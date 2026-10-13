# Shared-base LSA: increase capacity at fixed training exposure

This follow-up tests whether a larger transformer reduces the observed gap
from informed Bayes in the paper's central example. It reuses the existing
source generator, model implementation, test sequences and Bayesian results.

| Setup | Layers | Width | Heads | Parameters | Training sequences per run |
|---|---:|---:|---:|---:|---:|
| Existing baseline | 2 | 128 | 4 | 544,256 | 6.4 million |
| New larger model | 4 | 256 | 8 | 3,454,464 | 6.4 million |

Both setups have two independent training runs. The source alphabet has 64
symbols. Every training sequence has 1025 tokens, a fresh exponent uniformly
drawn from [0.5, 1.5], a fresh depth uniformly drawn from 0 through 8, fresh
transition rows, and a fresh permutation of the labels. Only tokens are
given to the transformer. The initial token is drawn from the common base law.

Training uses 100,000 updates, batches of 64 sequences, float32 parameters,
and the original AdamW settings: peak learning rate 0.0003, 1000 warmup
updates, cosine decay to 0.00003, weight decay 0.1, gradient clipping at 1.
The monitoring samples use the original four fixed source settings and
seeds. Final planned weights are the primary comparison; the saved midpoint
is diagnostic and will not be selected according to test loss.

This changes depth, width and head count together. It does not isolate depth
or test the effect of longer training. Training samples are fresh draws
from the same prior; the historical worker-dependent stream is not reproduced
token for token. New batches are indexed, so interruption and a change in
worker count do not change their identity. Checkpoints save model, optimizer,
random state and completed counts. CPU continuation is tested for bitwise
agreement; GPU kernels can still be nondeterministic.

## User-led launch on Kuma

Use the existing `layered_architectures_hidden_factor` checkout after pulling
the pushed code. New outputs go to its separate
`output/shared_base_scaling_2026-09-17/` directory. Existing experiments
remain separate. The user runs one command at a time; no remote submission
or additional input archive is needed.

First run a 1000-update throughput/memory benchmark in a GPU allocation:

```bash
bash code/transformer/kuma/submit_shared_base_scaling.sh --benchmark --submit
```

It has a 30-minute debug reservation and uses the full production schedule.
Exit 75 from the trainer is the intended capped stop; the batch script then
validates the run and writes `benchmark/seed0/benchmark.json`.
Inspect this receipt before production. The measured projection excludes
some validation and checkpointing overhead. Benchmark weights are not used
to initialize production.

Then submit the two larger-model runs:

```bash
bash code/transformer/kuma/submit_shared_base_scaling.sh --submit
```

Each array task has one H100, 8 CPU cores, 32 GiB host RAM and a 6-hour limit.
That limit is a reservation, not a runtime prediction. The launcher requires
the matching successful benchmark. Omit `--submit` to print commands without
creating outputs or contacting Slurm. An interrupted production run resumes
from its own `latest.pt`; no automatic resubmission occurs. A production run
is complete only when its status records 100,000 updates and 6.4 million
sequences. A Slurm warning requests a checkpoint before termination.

## Evaluation: reuse the existing evidence

After training, transfer each run's `model_100000.pt`, `config.json`,
`status.json`, `scaling_runtime.json` and `log.jsonl` to a separate local
output directory. Each final weight file is approximately 14 MB; the rolling
optimizer checkpoint is not needed for evaluation. It remains on Kuma for
possible later continuation.

The local evaluator uses the same 256 independent sequences per source depth
L=0,1,4,8, all at alpha=1 and length 1025. These fixed conditions are withheld
from both predictors; Bayes retains its full training prior. It saves all
six observation times 1, 4, 16, 64, 256, 1024 for the new transformers.
It compares to Bayes only at released, fully validated conditions:

- Times 1, 4, 16 for all four source depths.
- Time 64 for L=0 and L=1.

The completed September 17 time-64 release can now be added for all four
depths with `--time64-complete`. It is recorded as an additional evaluation
reference, with its own file hashes and explicit role; the frozen training
plan and its hash remain unchanged. The importer requires all 256 original
sequences at every depth and both integration orders, including every
mandatory continuation. It checks the published release, rehashes the raw
reference files (about 19.5 GB), reproduces all four cohort gates, and
recalculates the saved refined predictions' per-sequence regrets and paired
baseline summaries. It performs no Bayesian sampling or new inference.
The original L=0/1 time-64 scores must reproduce within numerical roundoff.
Without the flag, the original reference coverage above is preserved.

References at times 256 and 1024 remain unavailable. Paired
differences compare predictors on identical sequences, with standard errors
across whole sequences. Both training runs remain separate. The current
cohorts motivated this follow-up, so these are diagnostic comparisons,
not untouched confirmation data.

For example, after transferring the two completed run directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/transformer/shared_base_scaling_eval.py \
  --plan code/transformer/configs/shared_base_scaling_plan_2026-09-17.json \
  --runs output/shared_base_scaling_download_2026-09-17/seed0 output/shared_base_scaling_download_2026-09-17/seed1 \
  --time64-complete results/transformer/shared_base_time64_complete_2026-09-17/summary.json \
  --out output/shared_base_scaling_evaluation_2026-09-17 --device mps --batch 4
```

This is local inference on the transferred models, not a login-node command.
Use `--device cpu` if MPS is unavailable. The result is `comparison.json`
with per-sequence predictions and paired scores retained for validation.
Always use a new output directory. A completed earlier evaluation is not
overwritten when adding the newly released reference.

The plan binds the original model and evaluation-file identities in
`code/transformer/configs/shared_base_scaling_plan_2026-09-17.json`.
The final comparison should show original transformer, larger transformer
and Bayes against observed sequence length, separately by source depth;
show the two training runs in separate appendix panels. Report paired
improvements at the validated times, including uncertainty.
