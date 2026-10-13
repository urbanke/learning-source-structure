# Hidden-factor source: capacity first, then training

The source has two independent coordinate chains with fresh Dirichlet(1) rows
and a fresh arbitrary assignment of coordinate pairs to observed token labels
for **every sequence**. The transformer receives only token labels. Source
generation and exact Bayes are in `../hidden_factor.py`; the GPU reference and
capacity check are in `../hidden_factor_gpu.py`.

Do not choose the alphabet size from the training speed. Exact summation over
hidden encodings is the limiting step. The first job measures it for 2x5, 2x6,
3x4, 2x7, 3x5, 2x8 and 2x9, using full sequences of length 1025 and projecting a 4096-sequence
test cohort. A timed-out case is incomplete; its projected speed is not a
validated measurement. Choose the largest scientific comparison whose complete
reference computation fits the agreed resource budget. No size is preselected
by the launcher, and none of these jobs has been submitted by this preparation.

## First command: exact-reference capacity check

From the cluster checkout, print the command without contacting Slurm:

```bash
bash code/transformer/kuma/submit_hidden_factor.sh --benchmark
```

Add `--submit` to launch it. The report is
`output/hidden_factor_2026-09-16/capacity/benchmark-JOBID.json`; its log is
`output/hidden_factor_2026-09-16/logs/capacity-JOBID.out`.
This is a separate H100 job with a 45-minute limit. All hidden-factor jobs
request eight CPU cores and 32 GiB RAM, within Kuma's 5900 MiB-per-CPU cap.
Each encoding cache is
built once under `output/hidden_factor_2026-09-16/encoding_cache`, with an
8 GiB size limit and a 300-second construction cap. Each candidate then has
two independent sequence timings, capped at 60 seconds each. The 4096-sequence
projection counts cache construction once and uses the slower completed
sequence timing; it screens against a 24-hour reference budget. Candidates
exceeding the cache budget or a time cap remain explicit unsuccessful cases.
This is a practical capacity comparison, not a claim of the mathematically
largest tractable source. It does not train, overwrite earlier reports,
or touch existing jobs.

## After reviewing the reference cost

Select the dimensions explicitly. For example, **if 3x4 is chosen**, print the
1000-update training-speed check:

```bash
bash code/transformer/kuma/submit_hidden_factor.sh --training-benchmark --shape 3x4
```

After that check, print the two production jobs:

```bash
bash code/transformer/kuma/submit_hidden_factor.sh --shape 3x4
```

Each command requires `--submit` to launch. `--shape RxS` accepts any two
dimensions at least 2 whose product is at most 255; this interface limit does
not claim that exact Bayes is feasible for every such shape. The two seed
configuration templates have no default dimensions: the selected dimensions
are inserted into and hashed with the immutable run configuration.
`--out /absolute/path` selects an alternative output root.
`HIDDEN_FACTOR_VENV` defaults to `$HOME/venv-tf`; `HIDDEN_FACTOR_ACCOUNT` can
override the Slurm account. Give the user one command at a time when launching.

## Fixed comparison budget

Every candidate configuration has 100,000 updates with batch size 64:
**6,400,000 fresh training sequences per seed**, each of length 1025. Seeds
0 and 1 are separate repetitions, not a pooled source budget. The model uses
the same two layers, four heads, width 128 and MLP as the earlier runs, in
float32. Only its vocabulary size changes with the source. Parameters and the
hidden encoding are resampled independently between sequences. The 256 fixed
monitoring sequences use a separate random stream and are never used for
gradient updates or final test evidence; no test set is read during training.

The new source and training files are isolated. Existing source/model/mixture
training files are not edited; tested atomic-save and signal helpers are reused.
The new trainer records its source contract, source/model/helper hashes,
configuration hash, validation hashes, Torch/NumPy versions and numerical
runtime. The full learning-rate schedule is retained during a capped check.

## Outputs and continuation

Training outputs are under `ROOT/SHAPE/train/seed0` and `seed1`; the capped speed
check is separately under `ROOT/SHAPE/throughput/seed0`. A completed production
run has `status.json` with `complete: true`, `step: 100000` and
`completed_sequences: 6400000`. Bounded checkpoints consist of atomic
`latest.pt` (optimizer and RNG included) and final `model_100000.pt` (weights).

Repeating a launcher resumes an existing checkpoint with the same configuration,
implementation and runtime. It never silently restarts a nonempty directory.
An exclusive writer lock rejects overlapping trainers. Do not resubmit while
its job is still queued or running. Exit 75 means a safe incomplete stop; no
automatic resubmission or requeue is performed. The training-speed check maps
only its deliberate completed 1000-update cap to Slurm success. A stop signal
or shorter checkpoint stays incomplete.

Source batches are indexed by seed and update number; worker count and
prefetching cannot change them on resume. CPU continuation is tested for exact
agreement. GPU kernels may still be nondeterministic despite identical source
batches and restored random states.

## First inspect a 256-sequence pilot while training runs

The selected source is 2x8. Its exact reference is projected at about
36 GPU-hours for all 4096 test sequences. Begin with the first 256
sequences, expected to take about 2.25 hours. Print its submission command:

```bash
bash code/transformer/kuma/submit_hidden_factor.sh --references --pilot --shape 2x8
```

Add `--submit` to launch. This requests one H100, eight CPU cores, 32 GiB
RAM and a four-hour limit, with no checkpoint dependency. It creates the
full frozen 4096-sequence cohort, then evaluates indices 0 through 255.
Do not substitute `--n-seq 256`: that would draw a different cohort.
References go to `ROOT/2x8/eval`, reusing `ROOT/encoding_cache/2x8.npy`.
The log is `ROOT/logs/references-pilot-2x8-JOBID.out`. Progress and a partial
manifest are saved every 16 sequences (roughly eight minutes).

After the reference calculation, the pilot scores any available final
transformer checkpoints on those same 256 sequences. It writes
`ROOT/2x8/eval/summary_prefix_000256.json` and figures under
`ROOT/2x8/eval/pilot_figures/`. Both figures are explicitly preliminary.
The overview shows next-token regret, the cost of hiding the encoding,
and the transformer's remaining cumulative gap to Bayes. Error bars use
independent sequences; the two training seeds remain separate. The second
figure retains the predeclared source entropy groups and actual counts.
If neither transformer is ready, the plots say that model results are
pending. After training finishes, `--evaluate --pilot --shape 2x8 --submit`
adds both model comparisons and redraws the pilot figures.

The pilot stops after 256 sequences. It does not submit or start the
remaining 3840 automatically. Inspect the curves and checks before
continuing. Reference and comparison manifests remain incomplete for
the full cohort. A small or uncertain pilot gap is preliminary evidence;
it is not a final claim of agreement. If the pilot leads to model tuning,
disclose its use in development rather than calling all 4096 untouched.

## Continue the exact references after reviewing the pilot

```bash
bash code/transformer/kuma/submit_hidden_factor.sh --references --shape 2x8
```

Add `--submit` to launch. This verifies and reuses the pilot's saved values
and continues the remaining sequences, with a 48-hour allocation. Kuma's
normal QoS permits three days:
<https://scitas-doc.epfl.ch/supercomputers/kuma/> (checked 16 September 2026).
The scheduler's cost estimate uses the requested wall time. The reference
manifest has `complete: true` only after all 4096 records are checked.
An interrupted run resumes completed per-sequence records when resubmitted.

Keep the evaluator, source, GPU-reference code, software versions and
chunk size fixed between stages. Both reference-only and full evaluation
hold the same output-directory locks. Submit the full comparison only
after references and both training runs finish; it verifies and reuses
the exact references, then scores the transformers on the same sequences.
The evaluation changes do not modify any Python file used by the running
training jobs. Once the pilot begins, keep the evaluation code fixed too.

## Full comparison after references and both training runs finish

For the selected source, print the evaluation job:

```bash
bash code/transformer/kuma/submit_hidden_factor.sh --evaluate --shape 2x8
```

Add `--submit` to run it. This checks that both final training checkpoints
exist, reuses the validated encoding cache, and compares both transformers
and every reference on the same 4096 independent length-1025 test sequences.
Outputs are under `ROOT/SHAPE/eval`. It requests one H100 for at most six
hours and retains completed per-sequence results. References already made
by `--references` are verified and reused. If the reference stage was not
run first, this command also computes missing references, which can exceed
six hours for 2x8. A later manual submission continues completed records;
there is no automatic resubmission. A writer lock prevents two submitted
evaluators from writing to the same output directory.

Focused smoke verification (no production training):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q code/tests/test_hidden_factor_training.py
```
