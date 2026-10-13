# Three-family training on Kuma

Completed training: one GPT learns the product, shared-base and uniform-base
64-symbol sources. **Two independent seeds; each receives 6.4 million sequences
per family, 19.2 million total.** Production uses 300,000 updates at batch size 64.
The scientific protocol is
`notes/reviews/multifamily_experiment_2026-09-16.md`; references are in the
adjacent `multifamily_transformer_literature_2026-09-16.md`.

Training array 4233728 and evaluation job 4234451 have completed. Both
pooled seeds received the full per-family budget. The direct informed
three-family Bayes comparison is validated at prediction times 1 and 4;
see `results/transformer/multifamily_bayes_2026-09-16/README.md`. The earlier
specialist comparison remains in `results/transformer/multifamily_2026-09-16/`
as a diagnostic archive. Longer-context mixture-Bayes evidence remains pending. The commands below document reproduction and run
from the repository root on Kuma, using the existing `~/venv-tf` environment.
`MIXTURE_VENV` can select another installed environment. `MIXTURE_ACCOUNT`
can select a Slurm account; otherwise Slurm uses the user's configured default.
The repository's `code/transformer/kuma/setup.sh` prepares the existing
requirements if necessary.

## Check the package and GPU throughput

```bash
source ~/venv-tf/bin/activate
bash code/transformer/kuma/smoke_mixture.sh
bash code/transformer/kuma/submit_mixture.sh --benchmark --dry-run
bash code/transformer/kuma/submit_mixture.sh --benchmark --submit
```

The CPU check uses six tiny training steps and short sequences in a fresh
temporary folder. The GPU benchmark uses the production model and sequence
length for 1,000 updates in a **separate benchmark folder**. It checks CUDA,
generation throughput and checkpointing; it does not count toward the
production sample budget. Inspect its loss, elapsed time and completion
status before selecting resources for production. CUDA throughput is not
established by the local CPU tests.

## Train

```bash
bash code/transformer/kuma/submit_mixture.sh --dry-run
bash code/transformer/kuma/submit_mixture.sh --submit
```

The default output root is `output/multifamily_2026-09-16`. Pass
`--out /absolute/persistent/path` to select another location, and use the
same option for every subsequent launcher. The array runs seeds 0 and 1,
one H100 each, with eight CPU cores, 32 GB RAM and a six-hour allocation.
Do not resubmit while the same runs are active; the trainer also takes an
exclusive run lock. There is no automatic requeue or additional submission.

At a time-limit warning, training saves at a completed update. Exit 75 means
a checkpointed stop, not completion. Repeat the same submit command to resume
from `latest.pt` with the identical configuration and code. Different worker
counts preserve the indexed source stream. Do not alter the total-step
setting to extend a run: it defines the learning-rate schedule and checkpoint
identity. A completed model is `train/seedN/model_300000.pt`, with exactly
6,400,000 recorded examples from each family in `status.json`.

Only the rolling optimizer checkpoint and the declared weights snapshots are
retained. The output directory also records source priors, implementation
hashes, monitoring hashes, family sample counts and Python package versions.
Raw checkpoints and evaluations stay in ignored output storage, not git.

## Evaluate the completed models

The existing specialist final checkpoints must be available at their normal
repository locations under `results/transformer/runs/`: `d64`, `d64_seed1`,
`prod_8x8`, and `lsa_a0_d64`, each `model_100000.pt`.

```bash
bash code/transformer/kuma/submit_mixture_eval.sh --dry-run
bash code/transformer/kuma/submit_mixture_eval.sh --submit
```

This creates/reuses the fixed 8192-sequence evaluation, scores the two pooled
models and the relevant specialists on matched sequences, and writes
`comparison.json`. It reports all 23 conditions separately and paired
differences. All losses use bits and exclude the first token. Evaluation
refuses nonfinal checkpoints, altered cohorts and incompatible cached scores.
The evaluation job requests one H100, eight CPU cores and 32 GiB RAM for
up to four hours; this respects Kuma's memory-per-CPU limit.

The GPU evaluator writes transformer scores only; its original
`comparison.json` therefore retains the historical Bayesian-pending marker.
A separate, now completed short-context calculation combines all three
normalized family sequence probabilities and compares the saved pooled
losses directly with the resulting Bayesian mixture. Family weights update
after observations; fixed averaging of three next-token predictions is a
different predictor. See the direct-comparison README for the current
results and reproducible component/reference commands.

Training needs no LSA numerical store. Bayesian calculations reuse the
existing sealed store on the evaluation host. The direct mixture comparison
is available at times 1 and 4 only; the experimental long-context SMC pilot
has not met reference-precision requirements and is not a production result.

For a local validation of the code (including worker-process resume), run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  code/tests/test_mixture_training.py code/tests/test_mixture_evaluation.py \
  code/tests/test_mixture_bayes.py
```

Preserve run folders when synchronizing results. Raw held-out data,
per-sequence scores and checkpoint records remain under `output/`;
the retained specialist diagnostics are in `results/transformer/multifamily_2026-09-16/`.
The paper now uses `results/transformer/multifamily_bayes_2026-09-16/short_comparison.json`
in panel (f), rendered by `code/scripts/section3_compact_figure.py`. No GPU
re-evaluation is needed to reproduce that panel.

## Cluster documentation checked on 16 September 2026

- [Kuma resources and partitions](https://scitas-doc.epfl.ch/supercomputers/kuma/)
  confirms the `h100` partition and GPU/core allocations.
- [Slurm QoS and partitions](https://scitas-doc.epfl.ch/user-guide/using-clusters/slurm-qos-partitions/)
  documents `normal` QoS and wall-time limits.
- [SCITAS FAQ](https://scitas-doc.epfl.ch/faq/) describes accounts, access and
  submission behavior. The scripts reuse the project's existing Kuma setup;
  no account or credentials are embedded.
