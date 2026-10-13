# Pooled predictive comparison: bounded timing and guarded production

This campaign keeps the original 192 histories: indices 5–68 in each of the
product, shared-base and uniform-base families. Both transformer runs and Bayes
are compared on these same histories. The targets are 1, 4, 16, 64, 256 and 1024
observed tokens. Existing exact short-history references supply times 1 and 4;
this campaign prepares the four longer contexts. It never changes trained models,
original inputs, old sampler code or calibration results.

## Start with the timing wave on Jed

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_pooled_predictive_population.sh --stage timing --submit
```

Without `--submit`, the command only prints the intended jobs. It creates no
outputs and makes no scheduler calls. Defaults use the existing multifamily data
checkout and `shared_base_dependencies_2026-09-16` package under `~/Projects`.
All data verification, numerical work and merging run on allocated compute
nodes. The login process reads only the small plan and submission receipts.

The first wave has 36 tasks: three source families, four contexts and three
particle counts (256, 1024 and 8192). Every task uses 64 temperatures, four
moves, alpha order 16 and the declared row integration grid. There is one
independent repetition per task. At most 12 tasks run concurrently. Each has a
20-minute numerical deadline and a 30-minute Slurm limit. The first wave
reserves at most **18 worker CPU hours plus one CPU hour for its gate and merge**.
These are allocation bounds, not estimates of eventual production cost.

Timing may run before the stronger exact-short calibration finishes. It records
that calibration's state but provides **no accuracy approval**. Timeouts remain
visible. They mean the strategy or budget needs reconsideration; they do not
supply a linear extrapolation or an approved population point. The command
queues its own result merge but **never queues a subsequent campaign stage**.

## Subsequent stages are explicit decisions

| Stage | Tasks across all four contexts | Repetitions per task | Worker limit | Concurrency |
|---|---:|---|---:|---:|
| `timing` | 36 | 1 | 30 min | 12 |
| `refinement_timing` | 36 | 1 | 45 min | 12 |
| `refine` | 192 | A: 64; B/C/D: 32 each | 60 min | 64 |
| `population` | 2304 | 64 | 45 min | at most 64 |

The candidate rules are fixed before looking at transformer gaps:

- A: 8192 particles, 64 temperatures, four moves, alpha order 16,
  row grid from -32 to 32 with step 1/32.
- B: 16384 particles and 128 temperatures, otherwise A's integration rules.
- C: B with alpha order 32.
- D: B with row grid from -40 to 40 with step 1/64. The inexpensive product
  and uniform-base component integrals are also independently refined.

`refinement_timing` measures B/C/D on one development history per family and
context. Its measured worst-family times determine a conservative projection:
all 192 production histories, all B/C/D rules, 64 repetitions, all 12 development
histories, two attempts and a factor-two runtime margin. Each context has a
1024-CPU-hour projected budget, summing to 4096. Actual scheduler reservations,
including setup, merging and retries, have a separate 4096-CPU-hour ledger.
The two-attempt worker envelope is 3930 CPU hours; excessive extra overhead is
refused. A complete production context is 576 tasks and reserves 432 worker
CPU hours plus one hour for its gate and merge. These bounds do not promise
that all repetitions will finish in the allocated time.

After reviewing timing, choose contexts explicitly, for example
`--stage refine --observed 16 --submit`. Repeat `--observed` for several
contexts. A slow or unresolved context does not block other contexts. Every
accepted population point still requires **all 192 histories and all three
rules**, never an available-case average. Array indices are local to each
chunk (at most 0–767), with explicit mappings to the original global task IDs.

## Numerical acceptance

Scientific stages require the stronger exact-short report to be complete and
`exact_short_passed: true`; the gate independently checks the underlying frozen
shards. A successful scheduler exit from a calibration merge is insufficient.
Each context also requires complete candidate timing within budget. Population
sampling additionally requires all 12 predeclared development histories to pass
independent effort (A/B), alpha (B/C) and row-grid (B/D) comparisons.

Every repetition retains prefix evidence Z and successor joints Zp. These are
averaged in probability space before forming the predictive law or posterior
family weights. Prefixes contain exactly the declared observed tokens. Raw
normalization, finite absolute evidence, evidence/joint consistency, effective
replica counts and maximum likelihood shares are checked without repairing
invalid kernels. Absolute shared-component evidence and every successor log
probability must meet the declared precision and refinement tolerances; a tiny
mixture weight cannot hide an unreliable shared component.

Publication uses rule B, fixed in advance. C and D are acceptance checks, not
candidates selected for closeness to a transformer. Each rule must pass its own
cohort numerical budget. Aggregate B/C and B/D discrepancies must pass both the
0.01/3-bit resolution limit and stochastic agreement, in addition to per-history
checks. Family-stratified source standard errors, paired transformer-minus-Bayes
errors, numerical Monte Carlo errors and bias allowances remain separate.
Passing these finite checks is numerical evidence, not a proof of convergence.

## Receipts, retries and results

Submission receipts live in `output/pooled_predictive_population_2026-09-19/submissions`.
A repeated identical command returns its existing job IDs. Other waves are
refused while any recorded campaign job is active or its accounting is unknown.
Each accepted scheduler ID is saved immediately; an interrupted or ambiguous
submission fails closed and needs inspection before retrying.

Only after all earlier jobs are terminal may an unfinished task be explicitly
retried using `--retry --only-task GLOBAL_ID` (repeat the latter if needed).
A maximum of two worker allocations per task is enforced across selections.
Completed repetitions are reused with the same frozen seeds and input hashes.
An interrupted repetition has no internal checkpoint and must restart; a
heartbeat every 30 seconds reports that limitation. Exit 75 denotes a numerical
time limit or a handled Slurm signal. Hard termination can leave a writer claim;
inspect terminal scheduler state before any manual claim recovery.

Each stage writes `tasks/task-NNNN.json`, progress files, logs and `summary.json`.
The summary always lists all four contexts and explains withheld points. A merge
returns success only when every explicitly requested context passes; global
coverage remains false until all four pass. A population report contains exact
times 1/4 on the same 192 histories once at least one longer-context point is
accepted. No missing history or integration rule is silently discarded.
