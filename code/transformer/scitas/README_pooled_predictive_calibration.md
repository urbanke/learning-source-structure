# Exact short-prefix calibration of the predictive SMC

This is the next numerical check after the timing smoke. The smoke and its
existing code are unchanged. This new driver actually runs the full shared-base
prior with d=64, depth uniform on 0 through 8, alpha uniform on [0.5,1.5], and
a uniform latent rank permutation. It compares both absolute evidence and all
64 next-token probabilities with the existing validated short-prefix catalog.
It computes no new exact catalog, transformer predictions or population result.

The fixed cases are [0] and all 15 canonical equality patterns of length four.
The existing catalog includes their successor probabilities, including ABAC+A
(the previous ABACA stress case). Its committed provenance and independent
first-step check are verified in every compute task. The existing sealed-table
dependency manifest also verifies all 43 source/table files on the compute node.

The immutable plan specifies:

| Attempt | Particles | Temperatures | Moves | Repetitions per case | Array tasks |
|---|---:|---:|---:|---:|---:|
| initial | 2,048 | 32 | 2 | 32 | 128 |
| stronger | 8,192 | 64 | 4 | 64 | 256 |

Both use 16-point Gauss--Legendre integration for alpha and the existing row
grid [-32,32] with spacing 1/32. Each array task saves four independent
repetitions; at most 64 tasks run concurrently. Each task receives one CPU,
6 GiB and two hours. A repetition has a 30-minute numerical time limit checked
between kernel calls; an individual kernel call may exceed it. The scheduler
allocation is the outer limit. Setup and checkpointing use part of that time.
No automatic stronger attempt or population submission exists.

On Jed, from the updated mixture checkout:

```bash
bash code/transformer/scitas/submit_pooled_predictive_calibration.sh --submit
```

Default output is `output/pooled_predictive_calibration_2026-09-18/initial`.
The launcher only reads the small plan, checks a path and calls the scheduler.
All dependency hashes, array reads, numerical work and merges run on compute
nodes. Omit `--submit` or use `--dry-run` to print the command. An optional
`--only-task 0` selects a first task without changing its numerical settings.
For a bounded cluster preflight, `--only-task 0 --only-task 64` selects the
first four repetitions of [0] and ABAC, respectively, in the initial attempt.
These are unchanged tasks from the full array; a later full submission reuses
their completed records. This preflight checks execution, not the 32-repetition
accuracy gate.

Queue the result merge after the array, replacing JOBID with its actual ID:

```bash
bash code/transformer/scitas/submit_pooled_predictive_calibration.sh --merge --after-job JOBID --submit
```

Alternatively submit `--merge --submit` after completion. The merge writes a
report even if tasks are missing or numerical checks fail; a successful merge
job means the report was produced, not that the scientific checks passed.
Inspect `initial/summary.json` and its `exact_short_passed` field.

The fixed gates require complete coverage, at least 32 repetitions per case,
effective replica count at least 8, and maximum single-replica likelihood
share at most 0.2. Raw predictive normalization must be within 1e-7, with no
renormalization. For absolute evidence loss and **every** predictive log
probability, absolute error plus three Monte Carlo standard errors must be at
most 0.01 bits. Predictive errors use the covariance of Z and Z*p in the ratio;
averaging posterior vectors without their evidence weights is not used.
Successor-joint errors, total variation and KL are also reported. These are
delta-method diagnostics, not a simultaneous-confidence or missed-mode theorem.

If the initial report leaves precision unresolved, the predeclared independent
alternative is explicit:

```bash
bash code/transformer/scitas/submit_pooled_predictive_calibration.sh --attempt stronger --submit
```

Its reports and random seeds are separate; neither attempt overwrites the
other. Numerical tolerances cannot be relaxed by command-line flags. Completed
repetitions resume only with identical identities and verified record hashes.
Resubmit an unfinished task with the same attempt and `--only-task INDEX`.
An atomic directory claim prevents concurrent writers. A killed process may
leave a `.writer` claim: inspect its owner and scheduler state before removing
that claim, and preserve its shard JSON. No stale claim is removed automatically.

Every result retains `scientific_reference_validated: false` and
`population_reference_validated: false`, even when the short check passes.
Still missing afterwards: reconstruct mixtures on the actual pooled histories,
check numerical effort and alpha/row-grid refinement at contexts 16 and 64,
then validate complete coverage and uncertainty for the fixed balanced cohort.

Focused unit tests use synthetic repetitions and the saved catalog; no local
full-prior SMC campaign is run:

```bash
python -m pytest -q code/tests/test_pooled_predictive_calibration.py
```
