# Shared-base likelihood calibration on SCITAS

This campaign checks the Bayesian likelihood calculation needed for the
three-family experiment. It uses the trained transformers and frozen test
sequences already available. It does not train another transformer.

**Nothing is submitted by default.** Each command prints its requested job
and any missing prerequisites. Add `--submit` to execute that one command.
Stages never launch their successors, retry failed tasks, or expand to the
full test cohort automatically. No cluster job was launched while preparing
these files.

## Allocation and environment

Submit CPU stages from **Jed**: partition `standard`, QoS `serial`, one CPU
and 6 GiB per task. Preflight permits one active task; later arrays default
to at most 64 simultaneous tasks. Four complete SMC repetitions share one
task and reuse its caches; preflight has only two repetitions per case.
Each repetition retains its global index and independently derived seed.

The independent GPU benchmark is submitted from **Kuma**: partition `h100`,
QoS `debug`, one GPU, eight CPUs, 32 GiB and 30 minutes. It requires CUDA and
uses float64; there is no CPU fallback. Its output cannot unlock a CPU stage.

These choices follow the official [SCITAS partition and QoS documentation](https://scitas-doc.epfl.ch/user-guide/using-clusters/slurm-qos-partitions/)
and [Jed specifications](https://scitas-doc.epfl.ch/supercomputers/jed/),
checked on 2026-09-16. Jed's serial QoS allows single-node jobs for up to
seven days; this campaign initially requests only 15 minutes for preflight
and two hours per later task. Kuma debug allows up to one hour and is used
here for the independent performance benchmark. Production CPU calibration
uses serial QoS, consistent with [SCITAS guidance on debug jobs](https://scitas-doc.epfl.ch/user-guide/using-clusters/running-jobs/).

The scripts load `gcc` and `python`, then activate `$HOME/venv-tf` (override
with `SHARED_BASE_VENV` or `--venv`). The prepared environment needs NumPy,
SciPy and mpmath; the GPU benchmark also needs CUDA-enabled PyTorch.
No packages are installed by a batch job. BLAS and OpenMP thread counts
are fixed to one to match each CPU allocation. The preflight checks the
actual compute-node environment rather than assuming a login-node check
is sufficient.

## Frozen numerical dependency

The LSA store is too large for git. Transfer the prepared
`output/scitas_shared_base_2026-09-16/dependencies.tar.gz` from the Mac to
the cluster. Its SHA-256 and all 43 member hashes are recorded in
`code/transformer/configs/shared_base_dependency_manifest_2026-09-16.json`.
This archive contains the existing source package and sealed store levels
2 through 8. Nothing generates a replacement store.

From the current project checkout, after transferring the archive to
`$HOME/shared_base_dependencies_2026-09-16.tar.gz`, extract it with:

```bash
~/venv-tf/bin/python code/scripts/prepare_shared_base_cluster.py unpack --archive ~/shared_base_dependencies_2026-09-16.tar.gz --destination ~/Projects
```

Extraction verifies the archive, members and extracted file hashes, and
refuses to overwrite an existing directory. If it was already extracted:

```bash
~/venv-tf/bin/python code/scripts/prepare_shared_base_cluster.py verify --root ~/Projects/shared_base_dependencies_2026-09-16
```

Default paths are:

| Purpose | Path |
| --- | --- |
| Dependency root | `$HOME/Projects/shared_base_dependencies_2026-09-16` |
| Immutable store | `$CLUSTER_DEP_ROOT/tables/anchors_prod` |
| Numerical source | `$CLUSTER_DEP_ROOT/src` |
| Experiment outputs | `output/scitas_shared_base_2026-09-16/` in this checkout |

Set `CLUSTER_DEP_ROOT` to change the root, or use `--store` and `--src` for
an already available matching copy. The GPU benchmark requires both paths
under the same dependency root. Every numerical task verifies the committed
file hashes, the sealed store, read-only access and compiled interpolation
kernel. A missing or changed file stops the task. The Linux kernel binary
is built from the supplied C source by the existing package; no Mac binary
is transferred. Keep the checkout and dependency files unchanged while jobs
are queued or running: each launch binds their source hashes.

Before submitting on each cluster, check that the checkout, dependency and
Python environment paths are visible there. Availability from Kuma does
not by itself verify those paths from Jed.

## Run one stage at a time

Commands below assume the shell is in the current project checkout.
The first read-only preview is:

```bash
bash code/transformer/scitas/submit_shared_base.sh --stage preflight
```

On Jed, submit that stage:

```bash
bash code/transformer/scitas/submit_shared_base.sh --stage preflight --submit
```

After the six tasks finish, inspect their logs and merge the results:

```bash
bash code/transformer/scitas/submit_shared_base.sh --stage preflight --merge --submit
```

`--merge --submit` runs a local JSON reduction in the selected Python
environment; it does not submit a Slurm job or run the sampler. Incomplete
coverage produces an incomplete receipt and exit code 75. A complete
receipt can still report a failed accuracy gate. Inspect the printed gate
before proceeding; the next stage also checks it independently.

For each later stage, use the same two commands with the stage name changed:
first submit its array, then merge only after it has finished.

| Stage | Fixed inputs and budget | Required completed evidence |
| --- | --- | --- |
| `preflight` | Six five-token controls, 256 particles, 2 repetitions | None; checks runtime and cost only |
| `calibrate` | The same controls, 8192 particles, 64 repetitions | Successful runtime preflight |
| `confirm` | The same controls, 16384 particles, 128 fresh repetitions | Calibration accuracy pass |
| `alpha_refine` | Confirmation budget, 32 alpha quadrature points, fresh seeds | Confirmation accuracy pass |
| `row_refine` | Confirmation budget, wider and finer row integral grid, fresh seeds | Confirmation accuracy pass |
| `pilot16` | Twelve new frozen sequences, first 17 tokens, initial 256 particles and 16 repetitions | All four short accuracy stages pass |
| `pilot64` | The same twelve sequences, first 65 tokens | Previous pilot complete and precision pass |
| `pilot256` | The same twelve sequences, first 257 tokens | Previous pilot complete and precision pass |
| `pilot1024` | The same twelve sequences, all 1025 tokens | Previous pilot complete and precision pass |

The twelve sequences comprise four from each family, disjoint from the
three held-out sequences used in short calibration. Initial pilot budgets
measure feasibility; they are not a promise of adequate precision.

Short accuracy gates require every case, at least 16 repetitions,
absolute error plus transformed 95% interval half-width at most 0.01 bits,
effective repetition count at least 8, and maximum likelihood share at
most 0.2. Likelihoods are averaged **before** taking logarithms. Confirmation
and quadrature refinements remain separate tests with fresh seeds.
Longer pilots test Monte Carlo precision (0.01-bit half-width) and stability,
not correctness against an exact answer. Their receipts always retain
`accuracy_approved: false` and `production_approved: false`. Even a complete
successful pilot does not authorize the 3072-sequence comparison.

## Compare with the saved transformer losses

After merging completed pilot stages, create the preliminary comparison:

```bash
~/venv-tf/bin/python code/scripts/compare_shared_base_cluster.py --out output/scitas_shared_base_2026-09-16/comparison --receipts output/scitas_shared_base_2026-09-16/merged/pilot*.json
```

The command writes `comparison.json` and `pilot_comparison.png`. It validates
the raw repetition records and recomputes the likelihood averages,
uncertainty and gates. By default a time is omitted as a whole if any of
the twelve cases is incomplete or insufficiently precise; this avoids
selecting only the easy cases. `--diagnostic` explicitly allows a complete
but imprecise time to appear, with a diagnostic label. Times 1 and 4 use
the already verified exact catalog.

The curves compare the saved pooled-transformer losses with the informed
three-family Bayesian mixture on these same twelve sequences. They can
show whether a larger comparison is worth pursuing, and expose where
the likelihood calculation needs more work. Twelve sequences and narrow
Monte Carlo intervals do not establish population performance or correct
long-context evidence. Read the uncertainty and scope fields with the plot.

## Check progress, resources and incomplete tasks

Each task writes a runtime receipt before sampling and a scientific shard
before its first repetition. Completed repetitions are saved atomically as
they arrive. Logs contain progress, so a partly completed stage is useful
without waiting for every task.

* `logs/`: Slurm standard output, named by stage, job and array task.
* `shards/STAGE/`: scientific records with global indices, seeds and identities.
* `runtime/STAGE/`: per-invocation node, CPU affinity, allocation, versions,
  elapsed time and child-process peak RSS. Scientific shard resources record
  setup time and sealed-store/compiled-kernel checks.
* `merged/STAGE.json`: strict coverage, per-case uncertainty and gate receipt.
* `launches/STAGE/`: immutable plan and source hashes plus exact task mapping.

Use the returned Slurm job ID to inspect status, elapsed time and MaxRSS,
for example with `sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode`.
Review observed preflight time and RSS before larger stages; local-machine
cost projections do not establish cluster runtime or billing.

The numerical deadline leaves two minutes before Slurm's default wall-time
limit. Exit 75 preserves completed repetitions and requires manual review;
there is no automatic requeue. An interrupted repetition is restarted from
its original seed, rather than counted as another independent sample.
To resume only task 3 after diagnosing a timeout, retain the same output
root and shard size and preview, for example:

```bash
bash code/transformer/scitas/submit_shared_base.sh --stage calibrate --only-task 3 --wall-minutes 240
```

Add `--submit` only when ready. Do not change `--replicas-per-shard` within
an existing stage directory: overlapping ranges are intentionally rejected
by the merger. Lower array concurrency with `--concurrency`; the maximum is
64 and preflight is fixed at one. Use `--account` only if the cluster needs
an explicit account override.

## Independent H100 benchmark

From Kuma, preview:

```bash
bash code/transformer/scitas/submit_shared_base.sh --gpu-benchmark
```

Add `--submit` to execute one bounded benchmark. It uses six five-token
controls, 8192 particles and 16 GPU repetitions per control; setup, transfer,
original CPU sampling, matching batched CPU sampling and GPU sampling are
reported separately. Same-state target parity and exact-control accuracy
must be assessed along with speed. A faster GPU implementation is not
adopted automatically. Partial output is retained on timeout; benchmark
reruns need a new `--out` directory. No H100 numerical result is claimed
before this job has actually run.

## Local orchestration tests

```bash
python -m pytest -q code/transformer/scitas/test_launchers.py
```

These tests cover disjoint shard assignment, fixed replica identity during
manual resume, gate refusal, CUDA-only routing, frozen input detection,
timeout preservation and shell syntax. They do not submit jobs or certify
SCITAS runtime, numerical accuracy, or GPU performance.
