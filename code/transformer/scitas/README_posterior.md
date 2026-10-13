# Shared-base posterior pilot on Jed

This checks the difficult late points of the **original shared-base LSA
comparison**. It is separate from the three-family likelihood calibration.
No transformer is trained and no source or model probabilities are regenerated.

The fixed pilot contains eight already saved sources: sequence indices 0 and 1
at alpha=1 and depths 0,1,4,8. For each, compute the full-family Bayesian
prediction at 256 and 1024 observed tokens, with independent 16/32-node
exponent rules: **32 references**. The predictor retains its full prior;
the selected exponent, depth and source matrix are never given to inference.
Two sequences per condition measure computational feasibility, not a
population learning curve.

The worker, compressed chain runner, numerical store and independent raw
auditor are unchanged. Each task uses four chains, 16,384 warmup moves,
4,096 initial saved draws and at most 262,144 draws per chain. It may stop
early only when the existing worker checks pass. Sampling master seed
2026091670 is fresh; existing source and transformer records remain identical.
Every failure is retained. Passing one case does not remove another case.

## Resources and progress

Jed uses partition standard, QoS serial, eight hours, two CPUs and 12 GiB
per task, at most 32 concurrent. Numerical libraries use **one thread**.
The second CPU supports the larger memory allocation. At the draw cap,
the 512 MiB prediction trace is copied several times by unchanged
long-double variance calculations, on top of the row-integral cache;
6 GiB would risk failing at the final diagnostic stage.

The worker deadline is 7h45m, leaving up to ten minutes for the independent
audit and a final margin. A killed task retains its latest complete atomic
checkpoint; the current unfinished sampling interval may need replay.
There is no automatic requeue, resubmission, larger draw cap or full cohort.
A timed-out task may be resumed explicitly with --only-task INDEX --submit.
The same deployment, code, inputs and numerical runtime are required.
Completed ready or unresolved tasks cannot silently restart.

Progress is in output/shared_base_posterior_cluster_2026-09-16/tasks/.
A receipt is updated every 30 seconds with the last numerical progress event
and checkpoint size. Full progress logs, prior attempt receipts, raw chains,
final reports and audits remain in separate subdirectories. Receipts include
runtime versions, host, elapsed time and peak child RSS.

The compiled kernel, sealed read-only store, all 43 dependency files, selected
inputs, original checkpoints and relevant code hashes are verified. Python
loads through the existing environment.sh (gcc, python, then ~/venv-tf).
No packages or replacement tables are installed by batch jobs.
See the existing [SCITAS allocation documentation](README.md#allocation-and-environment)
for the partition/environment setup.

## Transfer and launch, one command at a time

The prepared input archive is small and ignored by git. Its full manifest
and hash are in
code/transformer/configs/shared_base_posterior_plan_2026-09-16.json.
The original cohort declarations and all selected records are byte-preserved.
Absolute paths embedded in old declarations are metadata only; the new
deployment records actual cluster paths. Fresh chains are created on Jed.
Raw audits run in that same deployment, so no path or runtime check is weakened.

On the Mac, transfer the input archive through the already-used Kuma login:

~~~bash
rsync -av /Users/ruediger/Projects/layered_architectures/output/shared_base_posterior_cluster_2026-09-16/inputs.tar.gz urbanke@kuma.hpc.epfl.ch:~/shared_base_posterior_inputs_2026-09-16.tar.gz
~~~

On Jed, update the isolated mixture_bayes checkout after the changes are pushed.
Use the actual checkout path if different. These additions leave the existing
numerical files bound by earlier running jobs unchanged; keep those files
unchanged until their jobs finish.

~~~bash
git -C ~/Projects/layered_architectures_mixture_bayes pull --ff-only origin main
~~~

Extract from that checkout:

~~~bash
~/venv-tf/bin/python ~/Projects/layered_architectures_mixture_bayes/code/scripts/prepare_shared_base_posterior_cluster.py unpack --archive ~/shared_base_posterior_inputs_2026-09-16.tar.gz --destination ~/Projects
~~~

Extraction rejects wrong hashes, unsafe archive members and existing targets.
It then verifies all eight saved sources and both models with the existing
loader. That loader requires the saved source's NumPy version, 2.4.6; a
mismatch fails here before any job is submitted. Use the prepared venv.
If already extracted, use verify --root ~/Projects/shared_base_posterior_inputs_2026-09-16 instead.
The existing dependency root is ~/Projects/shared_base_dependencies_2026-09-16.
Check that these home-directory paths are visible on Jed; access on Kuma
alone does not prove that.

Optional compute-node preflight (store/kernel and inputs, no posterior sampling):

~~~bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior.sh --preflight --submit
~~~

Submit the fixed pilot:

~~~bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior.sh --submit
~~~

Without --submit these commands only print the requested job, without writing
files. --inputs, --dependency-root, --out, --venv and --concurrency override
deployment paths/resources within the fixed limits. No subsequent stage launches.

## Review all results

After all tasks finish, on the same Jed checkout:

~~~bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior.sh --merge --submit
~~~

This verifies every task/audit/raw-file hash, numerical runtime agreement,
all indices and both exponent rules. The existing cohort audit checks
mixing, row grids, scalar precision and exponent refinement. It reports
missing/failed/unresolved cases explicitly; there is no available-case mean.
The resulting summary.json always labels this as an eight-source
computational pilot, with production_approved=false. Slurm exit 2 means
the bounded numerical task finished but remained unresolved, not successful
Bayesian validation.

The source records contain oracle probabilities for scoring and precision
diagnostics. The unchanged sampler receives only observed tokens; the
unchanged worker keeps the same use of scoring diagnostics for its stopping
rule. No new acceptance threshold is introduced.
