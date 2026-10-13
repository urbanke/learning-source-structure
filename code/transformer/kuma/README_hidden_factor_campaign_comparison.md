# Campaign comparison on the existing 3,840 cases

The larger campaign models can be compared with the same sequences, exact
Bayesian predictions, and original 100,000-update four-layer models used in the
current paper. These 3,840 sequences have already been examined. This is a
**reused comparison**, not new independent confirmation.

The evaluator reads immutable milestone weight files, so an eight-layer run may
continue training while its saved 200,000-update model is evaluated. It never
reads `latest.pt` or assumes the training status still describes that milestone.
It checks the checkpoint against its completed 256-case development receipt,
the campaign architecture/source/runtime contract, the full reference records,
and both original models' saved 3,840-case prediction arrays. All expensive
reads, checksums, inference and reductions run on a compute node.

From the updated Kuma checkout, submit the completed four-layer continuations:

```bash
bash code/transformer/kuma/submit_hidden_factor_campaign_comparison.sh --arm continue_large --step 400000 --submit
```

Submit the explicitly intermediate eight-layer models:

```bash
bash code/transformer/kuma/submit_hidden_factor_campaign_comparison.sh --arm xlarge --step 200000 --submit
```

After the eight-layer runs finish, compare their prescribed final checkpoints:

```bash
bash code/transformer/kuma/submit_hidden_factor_campaign_comparison.sh --arm xlarge --step 400000 --submit
```

`--after-job JOBID` can queue the last command after its training array succeeds.
Each command is a separate job, with both training seeds evaluated on one H100.
The requested cap is two hours, with eight CPUs and 32 GiB host memory. This is
a conservative allocation, not a measured runtime forecast. Without `--submit`
the launcher prints the command and creates no files.

Outputs are in
`output/hidden_factor_campaign_comparison_2026-09-19/<arm>/step<step>/`:

- `models/<arm>_seed0.npz` and `seed1.npz`, with matching JSON receipts;
- `summary.json`, retaining both runs separately, the full prior and four entropy
  groups, plus paired differences to informed Bayes and each run's original
  four-layer model;
- `selection.json`, `status.json`, and `logs/comparison-<job>.out`.

Arrays contain sequence indices 256 through 4095, selected next-token
probabilities, KL regret, realized next-token loss, and cumulative realized
loss. Observed length `t` predicts token `t+1`; cumulative losses omit the first
token. The transformer-minus-Bayes KL difference is also its excess conditional
expected loss, since the same source distribution enters both terms. Uncertainty
is one paired standard error across sequences, without averaging the two seeds.
Every result records `fresh_confirmation: false`; 200k eight-layer outputs also
record `final_checkpoint: false`. Different learning-rate schedules mean an
intermediate architecture comparison does not isolate capacity alone.

Repeat the same command after interruption. Completed model arrays are reused
only after identity, checksum, pairing and probability/metric replay checks.
An atomic shared-filesystem claim prevents concurrent writers. A killed owner
must be confirmed terminal by Slurm before a new claim can proceed; unknown
owners fail closed. This guard operates on the compute node, so accidentally
submitting a duplicate may briefly allocate a second job, which then stops
before reading inputs or writing scores. Do not delete claims to bypass checks.
The original training files, reference data, and earlier comparisons remain
read-only. No training or Bayesian inference is launched.
