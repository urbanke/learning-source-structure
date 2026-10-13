# Hidden-encoding confirmation on the unused test sequences

The full 2 × 8 Bayesian reference calculation has completed. This job reuses
those references to evaluate all six existing final transformer models on
the **3,840 previously unused sequences**, indices 256 through 4095 of the
frozen 4,096-sequence cohort. The first 256 sequences motivated the scaling
comparison and remain a separate exploratory result.

| Model | Training sequences per run | Final checkpoint |
|---|---:|---|
| Original two-layer model | 6.4 million | `model_100000.pt` |
| Same model with longer training | 25.6 million | `model_400000.pt` |
| Larger four-layer model | 6.4 million | `model_100000.pt` |

Each model has two training runs, seeds 0 and 1. Both are retained, with no
choice based on test performance. The comparison uses the same sequences,
reference predictions and observation times 1, 4, 16, 64, 256 and 1024 for all
six models. It reports next-token regret and paired cumulative loss above
the informed Bayesian predictor, including the existing source-only groups.

## Launch on Kuma

After updating `~/Projects/layered_architectures_hidden_factor`, submit:

```bash
bash ~/Projects/layered_architectures_hidden_factor/code/transformer/kuma/submit_hidden_factor_confirmation.sh --submit
```

Omit `--submit` to print the proposed submission without creating directories
or contacting Slurm. The launcher does only path checks, a few input existence
checks and submission on the login node. It does not load checkpoints, read
large arrays, verify bulk checksums, or summarize numerical records there.
All such work runs in the allocated GPU job.

The job requests one H100, eight CPU cores and 32 GiB host memory, with a
two-hour limit. This is a wall-time allowance, not a measured runtime forecast.
It performs **no training, no new sequence generation and no Bayesian
inference**. It verifies the completed reference inputs and final checkpoints
before evaluating the unused suffix. An incomplete or mismatched reference
is an error; the job does not silently substitute the pilot reference.

Defaults relative to the selected checkout:

- Read-only baseline and full reference: `output/hidden_factor_2026-09-16/2x8/`.
- Read-only scaling checkpoints and pilot evaluation: `output/hidden_factor_scaling_2026-09-16/`.
- Separate confirmation output: `output/hidden_factor_confirmation_2026-09-18/`.

Use `--baseline-root`, `--scaling-root` and `--out` to override these locations.
Output cannot alias, contain or be nested within the input runs. The optional
`--after-job JOB_ID` adds a successful-completion dependency. `--batch-size`
defaults to 16; it changes only inference batching. `HIDDEN_FACTOR_VENV` and
`HIDDEN_FACTOR_ACCOUNT` override the environment and Slurm account.

## Saved results and interruption

The output includes `status.json`, `summary.json`, and compact per-model
arrays and metadata under `models/`. Per-model files distinguish original,
longer-trained and larger models, and both seeds. Scores are retained with
input identities for paired checks and plotting after transfer. Scientific
completion requires all six models; Slurm success alone does not establish
that claim.

The evaluator holds an exclusive output lock. A duplicate active submission
cannot write the same results. If a job stops, repeat the same submission
after it has ended: identity-checked completed model scores are reused, and
unfinished models are evaluated again. There is no automatic resubmission.
All original checkpoints and Bayesian reference files remain read-only.

This confirmation can test whether the previously observed improvement and
remaining gap persist on unused sequences. It does not test a larger model
with the longer training budget, or establish that scaling eventually closes
the gap completely.
