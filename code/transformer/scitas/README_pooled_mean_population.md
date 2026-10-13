# Complete pooled Bayesian mean at context 16

This is the separately approved 20 September protocol. It does not alter the
older componentwise calibration, its failure flags, or any running worker.
The existing 192 histories are indices 5--68 from each of the three families.
Both transformer runs use those same histories. No successful-history subset
can become a published mean.

The shared component uses 32 independent repetitions for each history and
rule. Pool absolute evidence Z and joint evidence Zp before forming the
predictive ratio and the equal-prior family mixture. The four rules are:

| Rule | Particles | Alpha order | Row interval | Row step |
|---|---:|---:|---|---:|
| A | 256 | 16 | [-32,32] | 1/32 |
| B | 1024 | 16 | [-32,32] | 1/32 |
| C | 1024 | 32 | [-32,32] | 1/32 |
| D | 1024 | 32 | [-40,40] | 1/64 |

All use 64 temperatures, four moves, and cache capacity 4096. Seeds are
independent across rules and histories. The population does not reuse
canonical patterns. Short calibration does, with complete label mappings
and covariance for every history sharing a reference. Its 192 histories
contain one context-1 and twelve context-4 patterns, hence 52 tasks.

Acceptance preserves evidence ESS >=8, maximum repetition evidence share
<=0.2, relative numerator/denominator MCSE <=0.1, normalization and
jackknife/curvature checks. One combined numerical allowance includes the
final estimate's 2 MCSE + 2 bias, plus absolute mean shifts and their
uncertainty for A/B, B/C, C/D and the deterministic-family refinement.
It must fit min(0.003333 bits, one quarter of the source SE).
Source SE accounts for numerical covariance before setting this budget.
The reported final rule is D with refined product/uniform integration.
These are numerical diagnostics, not rigorous convergence guarantees.

The new short gate checks all four complete means and the final refined
rule against exact context-1/context-4 results. No componentwise tolerance
is silently relaxed. A failed new short gate prevents population sampling.
The population merge independently replays this gate before acceptance.

The single submission graph is preparation, calibration array, calibration
merge, population array, population merge. Population is submitted with a
dependency on the successful calibration merge; receipts say so explicitly.
There are 768 population tasks, each bounded at four hours, one CPU and
6 GiB, below Jed's 7000 MiB per-CPU partition limit. The launcher checks
the actual memory and CPU directives before contacting the scheduler.
Preparation and merges have one hour each; calibration tasks have
30 minutes. At most 64 workers run concurrently. The full allocation ceiling
is 3101 CPU hours, CHF17.0555 at the recorded CHF0.0055 rate. This is not a
completion guarantee or a reservation for other existing jobs.

From the Jed project checkout, review without submitting:

```bash
bash code/transformer/scitas/submit_pooled_mean_population.sh --dry-run --max-cpu-hours 3101 --max-cost-chf 30 --out output/pooled_mean_population_memory_recovery_2026-09-20
```

The approved bounded submission uses the same command with `--submit`
instead of `--dry-run`. The default raw inputs remain in the multifamily
checkout; sealed dependencies remain in shared_base_dependencies_2026-09-16.
All heavy verification and numerical work run on compute nodes.

The first 20 September submission was rejected before any job ID because its
8 GiB request exceeded the one-CPU memory limit. Keep its default-output
submission receipt unchanged. The corrected launch uses the explicit output
directory output/pooled_mean_population_memory_recovery_2026-09-20, with the
same CPU-hour and CHF limits. No original sampler or accepted job is restarted.

## Run outcome, checked 21 September

Preparation job 66605438 completed. Calibration job 66605439 finished with
37 of 52 tasks complete and 15 tasks stopped by their 30-minute allocation
limit. The saved task records contain 1,523 of the required 1,664 repetitions,
leaving 141. The incomplete task indices are 7, 10, 11, 14, 15, 19, 22, 23,
27, 31, 35, 39, 43, 47 and 51. These are time-limit interruptions, not failed
scientific acceptance tests. Calibration merge 66605440 therefore returned
incomplete. The dependent population job 66605441 was cancelled before any
population sampling, and merge 66605442 has no scientific result.

Do not resubmit the frozen graph. A recovery must be a separate bounded
continuation that preserves the 1,523 completed repetitions and their seeds,
adds only the missing repetitions, and reruns the full calibration gate. The
population may start only after that complete gate passes.

The original launch checks inspected the printed job IDs, the small
output/submission.json receipt, preparation and the first calibration worker
within a minute of startup. Every completed repetition is saved. Interrupted
tasks and claims are preserved; there is no automatic retry or later context.
Never delete a claim or restart a task without checking the live job and saved
work.

## Population-only continuation after a passed recovery

The failed 21 September recovery submission left a receipt in its default
output directory. Submit the 141-repetition recovery into a **new** output
directory after the account cap rises; do not reuse that failed directory.
Only after its `summary.json` says `complete=true`, `passed=true` and
`population_eligible=true`, use the same new path explicitly below. The
population-only launcher checks every recovered result and all 52 calibration
task bindings before scheduling. A compute-node preparation independently
replays the numerical gate, and the final merge replays it again. Neither
stage reruns calibration. The recovery directory records a one-use population
submission claim; a different output directory cannot silently duplicate the
768-task campaign.

```bash
python3 code/scripts/submit_pooled_mean_population_resume.py --dry-run --recovery output/pooled_mean_calibration_recovery_2026-09-22 --out output/pooled_mean_population_resume_2026-09-22 --max-cpu-hours 3074 --max-cost-chf 16.907
```

After reviewing the successful dry run, replace `--dry-run` with `--submit`.
The bound is 768 four-hour population tasks plus one-hour preparation and
merge, at most 64 concurrent workers, one CPU and 6 GiB each. The projected
cost is CHF16.907 at the plan's CHF0.0055/CPU-hour rate. It submits no later
context or automatic retry. The accepted new point requires the separate
population `summary.json` to be complete, passed and scientifically validated.

The 20 September parent output root is
`output/pooled_mean_population_memory_recovery_2026-09-20`; its population
stage did not run. The population-only continuation writes to its explicit
`--out` directory. Its `population/summary.json` must have `passed=true` and
`scientific_reference_validated=true` to supply the accepted context-16 point;
it contains paired rows and separate numerical/source uncertainty. Raw results
stay ignored. A failed or incomplete summary supplies diagnostics only and
cannot change the paper's figure.

Fast verification:

```bash
python -m pytest -q code/tests/test_pooled_mean_validation.py code/tests/test_pooled_mean_population.py code/tests/test_pooled_mean_submission.py
```

The production `prepare --preflight` command checks three actual histories
at contexts 1,4,16 using the frozen inputs and sealed store. Its receipt is
explicitly unusable for scientific workers. Full preparation was also tested
on all 576 history/context pairs locally before submission; no local
population sampler was started by preparation.
