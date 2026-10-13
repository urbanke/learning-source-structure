# Bounded pooled timing extension

The first dispatch (preparation66596388, workers66596389, merge66596390)
failed during startup because the wrapper omitted the calibration plan argument.
Both preparation and merge stopped at that call; the worker array was cancelled
at zero elapsed time. No timing sampling started. The original frozen plan and
failed submission receipt are retained.

The corrected launcher defaults to
`pooled_predictive_timing_extension_queue_recovery_plan_2026-09-19.json` and a separate
recovery output directory. All nine settings, original numerical code, histories,
seeds and budgets are unchanged. A real startup regression now calls the actual
calibration loader instead of bypassing the initialization path.

From the updated pooled-comparison checkout on SCITAS, queue the recovery with:

```bash
bash code/transformer/scitas/submit_pooled_predictive_timing_extension.sh --after-job 66595885 --submit
```

Omit `--submit` for a read-only preview. The launcher records submitted job IDs;
repeating the same command reuses that receipt and creates no new jobs. An
ambiguous or interrupted submission stops for inspection rather than retrying.

The second preparation (66596466) passed initialization but stopped because
the targeted live-queue query rejected the expired original job ID. Its worker
array66596467 was cancelled without running; merge66596468 was last reported
running. Both failed plans and submission records remain preserved.

Preparation now starts without a dependency on the expired original array ID.
It queries the user’s live queue and requires every selected original task to
have terminal historical accounting before authorizing workers. An unavailable
queue service, active original task or missing accounting still stops startup.
It checks the
original submission receipt, terminal scheduler accounting, selected task
reports, exact source histories, numerical dependencies and saved runtime.
Selected original reports that completed are reused with their original plan
and report hashes. Only selected reports with the original genuine sampling
deadline error are admitted to new sampling. Every original artifact remains
unchanged.

The fixed global task IDs are **2, 10, 13, 18, 21, 24, 27, 30, 33**. They cover
8192 particles at time 16 for the product source; 1024 particles at time 64 for
product and shared-base sources; and 256 particles at times 256 and 1024 for
each of the three source families. All settings, histories, priors and random
seeds are inherited from the frozen original population plan. Each new
repetition has a **7200-second sampling deadline** inside a **150-minute,
one-CPU, 6 GB allocation**, with at most **three workers concurrently**.
There is one extension attempt per selected task and no automatic retry or
within-repetition checkpoint. A timed-out repetition restarts from its
original seed; it does not continue the discarded partial particle calculation.

The maximum reservation is **23.5 CPU hours**: 22.5 for the nine workers,
0.5 for preparation and 0.5 for the automatic summary. Reused completed cases
exit without sampling. The summary runs after the worker array terminates,
including failures, and retains all nine cases and their missing or stopped
status. The new files are written under
`output/pooled_predictive_timing_extension_queue_recovery_2026-09-19/`; `prepared.json` records
the reuse decisions, `timing/tasks/` holds new worker reports, and `summary.json`
records independently checked completion and measured costs.

This extension measures computational cost only. One repetition cannot
establish numerical accuracy; all summaries explicitly withhold scientific,
accuracy and population approval. Nonselected failures remain untouched, and
the launcher submits no refinement or population campaign.
