# Bounded overnight Bayesian-mixture continuation

This is a **new successor workflow**, preserving the failed original six-case
calibration and the separate ABACA follow-up. It does not relabel either result.
The user submits it once on Jed; the numerical jobs run in batch allocations.

Preview, without writing files or contacting Slurm:

```bash
bash code/transformer/scitas/submit_shared_base_overnight.sh --after-job 66543429
```

Submit the same workflow explicitly:

```bash
bash code/transformer/scitas/submit_shared_base_overnight.sh --after-job 66543429 --submit
```

The default inputs are the existing original receipt at
`output/scitas_shared_base_2026-09-16/merged/calibrate.json` and the ABACA shards
under `output/scitas_shared_base_abaca_2026-09-16/shards/abaca_check/`. Outputs go
to **`output/scitas_shared_base_overnight_2026-09-16/`**, a separate directory.
Dependency-store, numerical-source and Python-environment defaults match the
existing CPU launcher; `--store`, `--src`, `--venv` and `--account` override them.

## What can run

1. Wait for all 32 ABACA tasks to finish successfully. Strictly merge their raw
   records and recompute the existing independent-follow-up review. Five passing
   original controls plus passing new ABACA must meet the unchanged criteria.
2. Freeze `successor_plan.json`, binding that review, both complete receipts and
   the original plans. Run the original all-six-control confirmation budget with
   its fresh seeds: 16,384 particles, 128 repetitions.
3. If confirmation passes, run the original alpha-quadrature and row-quadrature
   refinements concurrently. Both complete numerical gates must pass.
4. Run the 12 frozen sequences (four per family) at 16 observed tokens, with
   **16,384 particles and 32 repetitions per sequence**. This replaces the old
   exploratory 256-particle/16-repetition placeholder; source conditions, sampler,
   seeds, temperature schedule and precision criterion remain unchanged.
5. If all 12 precision checks pass, save the actual paired three-family mixture
   comparison and preliminary figure. **Stop for a cost and precision review.**

No 64/256/1024-token stage or 3,072-sequence population evaluation is submitted.
The old later-stage entries remain in the successor plan solely to retain the
existing comparison schema; they are listed explicitly as `future_not_queued`.
Short-control accuracy and pilot repeatability do not validate long-context
accuracy, missed posterior modes or the full population claim.

## Resources and outcomes

| Stage | Array tasks | Concurrent cores | Repetitions/task | Time limit/task |
|---|---:|---:|---:|---:|
| Confirmation | 192 | 64 | 4 | 2 h |
| Alpha refinement | 192 | 32 | 4 | 2 h |
| Row refinement | 192 | 32 | 4 | 2 h |
| Pilot at 16 tokens | 192 | 64 | 2 | 4 h |

Each numerical task uses the unchanged standard/serial one-core, 6-GiB CPU
launcher. The coordinator reserves one core and 2 GiB for at most 24 hours,
polls once a minute, and exits early on completion or failure. These are
reservation limits, not runtime or cost forecasts. No GPU work is queued here.

`status.json` lists job IDs, scheduler outcomes, strict receipt hashes, the stop
reason, and any saved comparison. `logs/controller-JOBID.out` and the existing
per-stage array logs provide details. A nonzero exit, incomplete coverage,
changed input, failed numerical gate or failed prerequisite stops advancement.
Other already submitted tasks can finish; the controller never cancels them or
retries a failed experiment automatically.

## Safe interruption and resume

If only the coordinator reaches its deadline or is interrupted, its saved child
jobs may still run. After its allocation has ended, attach to the same workflow:

```bash
bash code/transformer/scitas/submit_shared_base_overnight.sh --resume --submit
```

Resume revalidates the bridge and all completed numerical receipts, then attaches
to recorded job IDs. It refuses an active coordinator, a completed or failed
experiment, a changed implementation, or an ambiguous previous submission.
Submission intent is written before `sbatch`; if its response was lost, the
unique job name remains in the receipt for manual scheduler reconciliation.
Do not delete that record and resubmit blindly. Original inputs and merged
receipts are not overwritten by this workflow.

The submitter handles already-completed ABACA arrays without a stale dependency;
otherwise it uses `afterany` so a failed array still produces a stopped workflow
record. All 32 successful task outcomes and complete numerical receipts remain
mandatory. Child submissions remove inherited `SBATCH_*` and `SLURM_*` settings
and explicitly select the checkout and prepared environment.

Slurm supports asynchronous batch submission and array-wide dependency semantics:
[sbatch reference](https://slurm.schedmd.com/sbatch.html),
[job-array dependencies](https://slurm.schedmd.com/job_array.html).
SCITAS documents batch submission and dependencies in its
[running-jobs guide](https://scitas-doc.epfl.ch/user-guide/using-clusters/running-jobs/).
The coordinator uses the standard Slurm clients inside its allocation; if the
site disallows a child submission or the client fails, it records failure and
stops. This orchestration has local mocked-Slurm validation; actual Jed execution
is confirmed only by the resulting scheduler and numerical receipts.

Focused software checks (synthetic likelihood records, no scientific sampling):

```bash
python -m pytest -q code/tests/test_shared_base_overnight.py
```
