# Three shared-base posterior extensions

This separate experiment extends exactly three terminal unresolved records from
the frozen full-cohort deployment:

| Array index | Reference | Parent minimum probability ESS |
|---|---|---|
| 0 | L1_i195_t1024_a32 | 185.69 |
| 1 | L1_i207_t1024_a16 | 120.41 |
| 2 | L1_i232_t1024_a16 | 184.52 |

Each passed the other original gates at 524288 draws per chain. The new fixed
target is 1048576 draws for all four existing chains. No gate, prior, integration
order, burn-in, proposal, random seed or numerical implementation changes. More
draws do not guarantee acceptance. Other integration orders and source records
continue through the original worker queue.

## Submit on Jed

After updating the repository, submit once:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_full_shared_base_extension.sh --submit
```

Then queue the independent merge with the returned array job number:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_full_shared_base_extension.sh --merge --after-job JOB_NUMBER --submit
```

Omitting `--submit` prints the command without starting work. The launcher only
reads its small plan and scheduler status on the login node. Input verification,
large-file copying, sampling, raw-chain auditing and cohort merging run in Slurm
allocations. Defaults are:

- Parent: `/scratch/urbanke/shared_base_posterior_full_2026-09-17/launch.json`.
- New output: `/scratch/urbanke/shared_base_full_extension_2026-09-19`.
- Three concurrent 8-hour allocations, each with 48 GiB; eight allocated CPUs
  are required by Jed standard's 7000 MB/core limit. Numerical libraries remain
  single-threaded. Memory conservatively doubles the established parent envelope
  for the doubled saved-chain arrays and independent audit. The merge requests
  one CPU, 6 GiB and 30 minutes.

SCITAS documents the enforced memory/core bound at
<https://scitas-doc.epfl.ch/accounts/pay-per-use-account/#memory-limit>.

## Isolation, provenance and interrupted work

The plan binds the full-plan and launch identities, the selection snapshot, and
each original corrected audit, report and reference identity. On a compute node
the driver verifies that each parent queue entry and receipt is still terminal
`unresolved`, obtains its existing advisory claim without writing it, verifies
source/runtime/file hashes, and copies its report, chains and moments into the
separate output. Parent inputs, receipts, queue and summary are never modified.
The continued report must retain the original identity and history.

Worker checkpoints save at committed draw blocks, including at least every five
minutes during sampling. The sampler is capped at 6.5 hours per allocation,
reserving time for preparation and an independent one-hour audit. If it reaches
the fixed draw target but numerical checks still fail, status is `unresolved`
and no automatic extra sampling is requested. Exit 2 is a numerical failure;
exit 75 means the bounded allocation stopped with a resumable checkpoint.

An explicit retry uses `--only-task INDEX --attempt retry1 --submit` and the same
output. Repeating an identical submission returns its stored job number.
Different submission attempts are rejected while a matching run is active.
Terminal child results are reused without sampling. If a hard kill leaves a
`claims/TASK.writer/owner.json` directory, first inspect the recorded Slurm job;
there is no unsafe automatic removal or competing writer. Checkpoints are never
silently replaced by fresh chains on retry.

## Reading the result

- `tasks/TASK.json`: bounded execution result and log path.
- `audits/TASK.json`: independent raw-chain audit with unchanged acceptance gates.
- `overlay.json`: only the three replacement records and bound parent identities.
- `parent_snapshot.json`: read-only capture of the terminal original receipts
  used by this merge, including the original unresolved records.
- `summary.json`: separate all-source numerical revalidation of all eight groups,
  using the original two pilot sources plus all 254 remaining source records.

`apply_overlay(snapshot, overlay)` changes only the three pinned audit records in
a copy and clears every old group approval. `revalidate_groups` then reruns the
existing cohort budget and coarse/refined integration comparison on all 256
sources and both orders. Missing, failed and unresolved cases remain visible;
no available-case Bayesian mean is computed. Completion of these three tasks
does not prove that the whole L1/time1024 group, or the other five unfinished
conditions, passes. Publication may use only complete groups with `passed: true`
after the usual local source/score replay. The original publication exporter
does not automatically consume replacements; their provenance must accompany
the new summary when imported.

If other parent tasks finish later, rerun only this small compute-node merge
with a new `--attempt` name. It refreshes the separate summary from current
terminal parent receipts without touching the parent deployment.

## Local checks (no sampling)

```bash
python -m pytest -q code/tests/test_full_shared_base_extension.py
bash code/transformer/scitas/submit_full_shared_base_extension.sh
```
