# Finish the two unseen-context Bayesian comparisons

This precision-only follow-up concerns the original first-occurrence subsets
at 64 observed tokens: 66 histories at depth 4 and 57 at depth 8. All 123
histories and both integration orders are retained, giving 246 continuations.
Transformer scores already exist. There is no new training, source generation
or selection of histories according to the transformer–Bayes difference.

The old selected-group numerical allowances are 0.00427–0.00483 bits, above
the unchanged 0.003333-bit limit. Individual checks and integration agreement
passed. Each task continues its four saved chains to exactly four times its
previous draw count, without repeating burn-in. Targets range from 16,384 to
1,048,576 draws per chain. This fixed increase is a planning choice; passing
the final precision checks is not guaranteed.

## Local execution

These checkpoints bind the original Python 3.14.6, NumPy 2.4.6, SciPy 1.18.1
runtime and the Mac's 53-bit accumulator. They cannot be continued unchanged
using Jed's different numerical runtime. The local runtime is isolated under
`output/unseen_context_refinement_runtime_2026-09-19/`; the old environments
and checkpoints remain unchanged.

Check the frozen plan and runtime without sampling:

```bash
bash code/transformer/start_unseen_context_refinement.sh --check
```

Start or continue the pending tasks, followed by the independent group check:

```bash
bash code/transformer/start_unseen_context_refinement.sh --run
```

The dispatcher permits at most four single-thread workers and admits them
against an aggregate 18 GiB memory allowance. Larger tasks reduce concurrency.
It preserves a 50 GB free-disk reserve. Each task has a two-hour bound,
including sampling and auditing; a dispatch lasts at most 12 hours. Completed
tasks are reused. Interrupted tasks require explicit `--retry-incomplete`,
with at most two attempts per task and a cumulative 128 worker-hour bound.
No completed numerical failure triggers further sampling automatically.

Recorded previous elapsed times give a rough 44 worker-hour estimate for the
additional draws. This includes a conservative linear extrapolation of old
burn-in time, but excludes possible nonlinear checkpoint and audit overhead.
It is not a completion-time guarantee. On macOS the launcher requests that
the laptop stay awake during execution; closing the lid can still suspend it.

## Provenance and acceptance

The compact versioned plan binds an ignored manifest with every source,
original report, checkpoint, moments file and saved numerical identity. The
builder reconstructs it from the readiness audit without sampling. Before
each continuation, the worker verifies the original raw checksums and copies
the files into `output/unseen_context_refinement_2026-09-19/references/`.
It retains the original target, proposal, random states, sampling seeds,
burn-in and every previously saved sample. Periodic atomic checkpoints make
interruption resumable without starting fresh chains.

An independent audit checks the continued raw predictions, final chain states,
mixing, stable depth diagnostics, numerical integration and source pairing.
The final merge reruns the original cohort precision test separately on all
66 or all 57 selected histories with both integration orders. A complete
passing group may be released even if the other group is unfinished. Missing
cases never produce an available-case mean, and numerical failures remain
withheld. Precision diagnostics remain conditional on adequate mixing; they
are not mathematical guarantees against missed posterior modes.

The new output contains `context.json`, `status.json`, per-task receipts and
audits, logs, and the final `summary.json`. These stay out of GitHub. Existing
Section 3 and Section 4 results are not overwritten. Fresh-sequence
confirmation is separate from this precision repair.
