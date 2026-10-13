# Complete the remaining shared-base posterior cohort

This calculates **4,064 integration-order tasks**, covering all **2,032 missing
predictions**: source indices 2–255 for each of L=0,1,4,8 at observed lengths
256 and 1024. It reuses the original saved 1,025-token sequences and transformer
probabilities. The eight completed pilot sequences (indices 0 and 1 at each
L) are retained in the final 256-sequence-per-depth comparison.

The calculation uses integration orders 32/64 for L=0 at 1024, and 16/32
elsewhere. No source generation or transformer evaluation is repeated.

## Transfer and launch

The prepared archive is
`output/shared_base_posterior_full_2026-09-17/inputs.tar.gz` locally. Transfer it
to `~/shared_base_posterior_full_inputs_2026-09-17.tar.gz` on Jed, update the
`layered_architectures_mixture_bayes` checkout. Unpacking and bulk checksums
belong on a compute node. Within an allocated compute shell, unpack with:

```bash
~/venv-tf/bin/python ~/Projects/layered_architectures_mixture_bayes/code/scripts/prepare_shared_base_posterior_full.py unpack --archive ~/shared_base_posterior_full_inputs_2026-09-17.tar.gz --destination ~/Projects
```

For the current deployment, the existing unpacking was allowed to finish in
place at the user's request. Reuse its completed input directory. From the Jed
login node, submit only this small preparation job:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior_full_prepare.sh --submit
```

This login-side wrapper only creates the log directory and submits one job.
The preparation job requests one CPU, 6 GiB and two hours; it verifies the
existing inputs and automatically submits the 64 workers. It never unpacks
again. Its log is
`/scratch/urbanke/shared_base_posterior_full_preparation_2026-09-18/prepare-JOBID.out`.
The preparation job finishing means the workers were submitted, not that their
calculations finished.

The underlying full launcher must run on a compute node because it verifies
all input and dependency files. Its output directory is explicitly required:
saved chains can exceed the home quota and must live on shared scratch.
Existing sealed dependency inputs
at `~/Projects/shared_base_dependencies_2026-09-16` and `~/venv-tf` are reused.

## Scheduling, continuation and failure reporting

- There are 64 persistent worker slots. Each allocation requests 4 CPUs,
  24 GiB memory and 8 hours. Each slot claims successive tasks from the full
  frozen list under file locks.
- Before sampling, each active slot queues one `afterany` successor. At most
  64 allocations run and 64 wait. Thus the reservation is bounded, instead of
  reserving 4,064 long jobs. At CHF 0.0055 per CPU-hour, 128 full allocations
  correspond to approximately CHF 22.53; actual total cost depends on runtime.
- Each slot has at most 32 generations, and a task has at most four allocation
  attempts. This is a finite scheduling envelope, not a promise of numerical
  convergence. An exhausted envelope is reported explicitly. No fresh task
  starts with less than two hours remaining in an allocation.
- Each task evaluates saved chains at successively doubled budgets,
  4,096 through 524,288 draws per chain. The independent raw audit, canonical
  saved-probability readback and scale-invariant depth diagnostic run after
  every completed budget. A legacy tiny-weight diagnostic cannot force an
  otherwise acceptable task through every budget.
- The numerical wrapper saves chains and their RNG states periodically,
  retaining partial-chain progress across allocation timeouts. Successors
  resume saved chains, rather than generating new samples or selecting a
  favorable repeat. Only interruptions are retried automatically; numerical
  failures at the maximum budget remain `unresolved`.
- Submission intents are saved before contacting Slurm. An uncertain reply is
  reconciled by exact job name through the scheduler; it is never blindly
  resubmitted. An unresolved submission ambiguity stops that slot visibly.
- Frozen inputs are checked once at launch and unpack. Each task then verifies
  only its own seven source/model files. The numerical implementation and
  dependency store remain bound to the frozen plan.

The output directory is
`/scratch/urbanke/shared_base_posterior_full_2026-09-17`. `status.json` gives coverage
counts; `queue.json` records every task; `tasks/`, `audits/`, `attempts/`,
`references/` and `submissions/` retain the calculation and scheduler evidence.
`logs/full-*.out` contains progress. Successors have job names beginning
`sb-full-` and separate job IDs; the initial array's completion alone does not
mean the full computation finished.

The final `summary.json` combines all 256 sequences at each condition with the
established cohort error and integration-refinement checks. A completed queue
is distinct from a validated scientific reference. Missing, unresolved and
failed cases remain in the report; no average silently excludes them. The
collector verifies hashes of the small audits and file-size/modification-time
identity of their raw files, retaining the SHA256 values already checked twice
by each independent audit. It does not reread terabytes of saved chains.

A complete final collection with failed numerical checks exits with code 2;
an explicitly requested incomplete collection exits with code 3. Only a fully
validated collection exits with code 0. Earlier successful worker allocations
mean their assigned work finished, not that the scientific comparison passed.
