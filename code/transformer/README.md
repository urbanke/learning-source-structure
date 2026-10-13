# Transformer experiments

Trains small transformers on fresh source draws and compares their predictions
with Bayesian references and the original unigram/copier comparators.

## One model across three families (prepared experiment)

The new mixture experiment retains the existing 64-symbol product, shared-base
and uniform-base source priors. Each of two training seeds receives 6.4 million
sequences per family, 19.2 million total. The code reuses the existing generators
and GPT; it does not retrain the specialist models.

See [the Kuma instructions](kuma/README_mixture.md) for the local smoke check,
GPU benchmark, training and matched held-out evaluation. The scientific protocol
and annotated references are in `notes/reviews/multifamily_experiment_2026-09-16.md`
and `notes/reviews/multifamily_transformer_literature_2026-09-16.md` at the
repository root. The user submitted Kuma array 4233728 on 16 September 2026;
completion is pending. Bayesian mixture
comparisons still require separately validated sequence marginal likelihoods.

## Shared-depth references (current workflow)

The uniform-base generator draws one depth for the whole chain. The product
generator draws one independent depth per coordinate chain. Their Bayesian
references must combine evidence from all rows before averaging over depth.
The historical refs_famprior.py instead averages depth separately in each row;
its old files remain historical comparators.

- shared_depth.py implements the corrected joint posterior, with exact
  depth-zero/depth-one evidence and numerical higher-depth row evidence.
- refs_shared_depth.py evaluates an explicit NPZ token matrix, uses a separate
  output schema, and checks provenance before reusing or resuming a result.
- eval_shared_depth.py compares the existing uniform-base d=64 and product
  8x8 checkpoints on fresh full-prior draws. It saves paired sequence losses
  and float32 transformer log probabilities at selected prefix lengths, and
  reports predictive KL divergences and standard errors.
- validate_shared_depth.py checks numerical row evidence independently of the
  sequential update, including integration-range and grid refinement.

Set LSA_STORE and PMM_SRC as described in phi_source.py. With the requirements
installed, run from the repository root:

    python -m pytest -q code/tests/test_shared_depth.py
    python code/transformer/validate_shared_depth.py --out output/shared_depth/validation.json
    python code/transformer/eval_shared_depth.py --out output/shared_depth/full_prior_final_2026-09-14 --family both --n-seq 128 --n-ctx 1024 --device cpu --torch-threads 4
    python code/scripts/shared_depth_summary.py --input output/shared_depth/full_prior_final_2026-09-14 --out results/transformer/shared_depth

The tests additionally require pytest. A small evaluation check uses
--n-seq 4 --n-ctx 32 and a separate output directory. Stages sources, refs,
model and summary can run separately. Each source draw has its own seed;
changing the sequence count or inference batch does not change existing
draws. The first token is observed but not scored. Raw caches remain under
output/; commit only the small validated summaries.

Use the wider default integration range [-30,30] for the new references.
It avoids a measurable truncation error of the historical [-19,10] range
in some high-depth product-chain rows. Returned reference probabilities
are checked without silently renormalizing them.

## Shared-base reference on small alphabets

`shared_base.py` retains the original source's unknown exponent, unknown
label ranks and one shared depth. It explicitly sums all permutations and
depths, and integrates the continuous exponent and weighted row laws
numerically. `weighted_lsa.py` supplies ordered row evidence. The initial
token has law q; a fresh transition row generally has a different mean.

With LSA_STORE and PMM_SRC set as above:

    python -m pytest -q code/tests/test_weighted_lsa.py code/tests/test_shared_base.py code/tests/test_shared_depth.py
    python code/transformer/validate_shared_base.py --out results/transformer/shared_base/validation.json
    python code/transformer/refs_shared_base.py --d 3 --n-tokens 32 --seed 20260914 --alpha-order 16 --out results/transformer/shared_base/d3_example.json
    python code/transformer/refs_shared_base.py --d 4 --n-tokens 32 --seed 20260914 --alpha-order 16 --out results/transformer/shared_base/d4_example.json

The validation command also writes a TSV; `--smoke` runs a shorter check.
The prediction command accepts `--sequence 0,1,0,2` instead of generating
tokens. It records predictions and posterior summaries before each token,
excludes the first token from scored losses, and checks raw normalization
and sequential/batch consistency. Recorded latent source parameters are
never supplied to the predictor. For refinement, increase `--alpha-order`
to 32 and halve `--step`, using separate output files.

These are d=3/d=4 reference calculations on short sequences, with independent
checks detailed in notes/reviews/mechanisms_shared_base_reference_2026-09-14.md.
Permutation enumeration is factorial; this is not the d=64 evaluation
method. Longer contexts require fresh integration-accuracy checks.

## Shared-base posterior sampling

`shared_base_mcmc.py` approximates the original Bayesian prediction at a
specified observed prefix. It samples the exponent quadrature node and label
rank assignment, while integrating the weighted rows and one shared depth.
Four independent chains start from different assignments. Unseen labels have
identical predictions by an exact conditional average over their unused ranks.
This computes selected-prefix predictions, not sequential codelengths.

`ranked_lsa.py` is a cache adapter around the existing production evaluator:
`production_coding.configure_production_tables` validates the sealed store,
`phi_source.StoreTables` opens it read-only, and `lsa_seq._lse_rows` performs
the fast integral reduction. No moment tables are built. The cache retains
the zero-count sum and reuses corrections for occupied ranks. It is not a
replacement for the established LSA moment implementation.

With the same LSA_STORE and PMM_SRC settings:

    python -m pytest -q code/tests/test_ranked_lsa.py code/tests/test_shared_base_mcmc.py
    python code/transformer/validate_shared_base_mcmc.py --burn 1024 --draws 2048 --out results/transformer/shared_base_mcmc/validation.json
    python code/transformer/validate_shared_base_mcmc_control.py --burn 4096 --draws 8192 --out results/transformer/shared_base_mcmc/d64_control.json
    python code/transformer/refs_shared_base_mcmc.py --d 64 --n-tokens 128 --check-grid --out output/shared_base_mcmc/d64_n128.json

The prediction command can read an explicit token array with `--tokens-json`;
otherwise it generates a fresh draw from the original training prior. The
output contains raw next-token probabilities, Monte Carlo standard errors,
posterior summaries, chain diagnostics and provenance. `--check-grid` repeats
the final sampled state of each chain with wider bounds and halved spacing.
For exponent refinement, repeat with `--alpha-order 32` and a different seed.
Keep separate outputs for every sampling/refinement configuration.

The retained d=64 pilots use `--burn 8192 --draws 8192 --thin 4`;
the shorter defaults are useful initial diagnostics but can mix too slowly.
The 32-node repetitions use `--term-cache-size 8192` to retain about 1 GiB
of shifted terms; changing cache capacity does not change the target or
numerical calculation. `code/scripts/shared_base_mcmc_summary.py` independently
checks the saved 16/32-node pairs and exports their comparison.

Passing Rhat/ESS/MCSE thresholds is a diagnostic, not a convergence certificate.
Small-alphabet comparisons and the repeated-symbol d=64 control test accuracy
where exact rank sums remain possible; general d=64 prefixes also need
independent-run and exponent-refinement checks before transformer comparisons.

## Original-source full-context pilot

`eval_shared_base_models.py` freezes independent 1025-token source draws
before selecting prefixes, retains their true transition matrices for
evaluation, and evaluates the two existing original-source checkpoints.
The model sees tokens[:1024]; its last output predicts tokens[1024]. Source
seeds are stable by index, so enlarging a cohort preserves previous draws.
The true matrices never enter either predictor.

    python code/transformer/eval_shared_base_models.py --stage sources --out output/shared_base_full_context --n-seq 4 --seed 20260922 --prefixes 1024
    python code/transformer/eval_shared_base_models.py --stage model --out output/shared_base_full_context --n-seq 4 --seed 20260922 --prefixes 1024
    python code/transformer/validate_shared_base_long.py --out results/transformer/shared_base_full_context/long_validation.json
    python code/transformer/eval_shared_base_reference.py --index 0 --alpha-order 16
    python code/transformer/eval_shared_base_reference.py --index 0 --alpha-order 32 --term-cache-size 16384

Repeat both reference commands for indices 1, 2 and 3, with the same
LSA_STORE/PMM_SRC production configuration as above. At most six concurrent
jobs are used on the checked 24 GiB host; each shifted-term cache may use
about 1–2 GiB. The larger cache avoids repeated lookups on the finer
exponent grid. The
new `shared_base_chain_runner.py` reuses the frozen sampler's transitions
and diagnostics. It saves raw samples, RNG states and chain positions in
ignored `output/.../chains/`; rerunning extends exactly the same chains.
Changed inputs, numerical implementations or stores invalidate a checkpoint.

The reference driver doubles draws from 4096 to 65536 when the predeclared
mixing or metric-precision checks fail. It retains failed rounds, exits
nonzero if still unresolved, and supports larger explicit `--max-draws`
extensions. Its small JSON/NPZ outputs contain predictions, covariance
estimates, per-model metrics, diagnostic history and provenance. Exponent
refinement is a separate paired check; a local success alone does not
validate the final transformer comparison.

`full_context_statistics.py` computes KL from the posterior-mean prediction,
not the average conditional KL over latent samples. Covariance projections
give numerical uncertainty in KL and true expected loss difference; the
second-order KL noise scale handles near equality. These numerical errors
are distinct from variation across source draws. Four draws form a pilot,
and the two training seeds do not double the number of source replications.
No cumulative codelength or copying-onset estimate is produced here. See
`notes/reviews/mechanisms_full_context_plan_2026-09-14.md` for the fixed protocol.

The independent exporter reconstructs raw-sample covariance and checks
input, checkpoint, implementation and store provenance before comparing
the exponent rules:

    python code/scripts/shared_base_full_context_summary.py --input output/shared_base_full_context --out output/shared_base_full_context/summary.json

Missing or unresolved source pairs produce a nonzero exit and suppress the
cohort aggregate. Once a full raw audit has passed, `--smoke --raw-validation
PATH_TO_FULL_SUMMARY` checks the retained small files against that receipt.
The full-context cohort is still running; this workflow does not imply that
all ordinary prefixes have passed their diagnostics.

A separate experimental proposal lives in `shared_base_local_rank.py`, with
proposal-aware resume protection in `local_rank_chain_runner.py` and its own
`eval_shared_base_local_rank.py` output directory. It is not adopted for the
production comparison: exact-target checks pass, but the bounded ordinary
prefix benchmark still fails mixing thresholds. Its independent controls
can be repeated with:

    python code/transformer/validate_shared_base_local_rank.py --out output/shared_base_local_rank/validation.json

## Design

* One fresh transition matrix per sequence, exactly as in the paper (`source.py` reproduces
  `chain_sim.draw_table` / `simulate_path`).
* The training stream draws alpha and L_s afresh for every sequence (alpha uniform in [0.5, 1.5], L_s uniform in
  {0, ..., 8}) and relabels the alphabet by a random permutation per sequence. Training can reveal this family
  and its prior; a new context must reveal the particular draw. Controls: `--alpha a a --ls L L`
  (fixed source), `--no_permute` (base law learnable in the weights), `--n_layer 1`, `--no_mlp`.
* Evaluation per fixed setting (d, alpha, L_s): the transformer, the unigram and the copier are run on the same
  held-out sequences (`eval_sets.py` fixes them by a seed; the md5 of the tokens travels with every result). The
  mechanisms are evaluated exactly, per position, with the LSA code (`lsa_seq.py`, which needs the ln phi tables
  of code/scripts/building_block; the tables are not in git).
* `analyze.py` compares the per-position losses, computes the transformer's implied weight on copying (the mixture
  weight that minimises the KL divergence from its prediction to w * copier + (1 - w) * unigram) and the position
  from which it stays above one half, and reports the crossing estimates next to the learner's.

## Files

`source.py` the source; `model.py` the GPT; `train.py` training with checkpoints; `eval_sets.py` the evaluation
sets; `eval_model.py` the transformer on the evaluation sets; `make_refs.py` the mechanisms on the same sets
(needs the tables); `lsa_seq.py` the sequential LSA evaluator; `analyze.py` the comparison; `run_first.sh` the
commands for the first run. `refs_extra.py` the further reference predictors of Section 4 (true depth, joint pairs,
backoff, informed, oracle copiers; the weighted LSA code); `analyze2.py`, `figures_sec4.py` the crossings, the
implied weight and the figures and tables of Section 4; `jobqueue.py` a small job queue for the references.

## The source families (notes/families_plan.md)

`families.py` defines the families with one interface (`--family` in `train.py`, `--family_sets` in
`eval_model.py`): `binary:<prior>` (F1), `expfam:k=<k>[:perm]` (F2), `product:<d1>x<d2>` (F3), and `lsa` (the
source of the paper; F4 is `--alpha 0 0`). Every evaluation set carries `h_true`, the entropy of the true
next-token distribution at every position, so that regrets can be plotted. `refs_family.py` computes the exact
coin references and numerical/comparator predictions for the other families. The grid approximates a continuous
prior; the historical product copier uses row-local depths. Use the shared-depth workflow above for the
product family's Bayesian reference.
`kuma/submit_families.sh` submits the first-round runs, `kuma/eval_families.sh` evaluates them on their own sets
and on the other families' sets (cross-training), `kuma/fetch.sh` copies the results to the laptop.

## Running on a Mac (Apple silicon, PyTorch MPS backend)

    python3 -m venv ~/venv-tf && source ~/venv-tf/bin/activate
    pip install -r code/transformer/requirements.txt
    bash code/transformer/run_first.sh

Outputs go to `results/transformer/runs/<name>/` (checkpoints, `log.jsonl`) and `results/transformer/eval/<name>/`
(the transformer's outputs on the evaluation sets). Git ignores both, except the small files that let another
machine evaluate a checkpoint: `config.json`, `log.jsonl` and the weights-only copies `model_<step>.pt` (about
2 MB each; `train.py` writes one at every checkpoint, `export_weights.py` makes them for older checkpoints).
The evaluation, the references and the analysis are then run where the LSA tables are.

## Running on SCITAS Kuma (SLURM, H100)

Once, on the cluster: clone the repository into the home directory and run `bash code/transformer/kuma/setup.sh`
(creates `~/venv-tf` with the CUDA build of PyTorch). Then `bash code/transformer/kuma/submit_first.sh` submits the
first batch of runs (one GPU each, about an hour per run); `squeue -u $USER` shows the queue, and each run writes
`results/transformer/runs/<name>-<jobid>.out` next to its checkpoints. The weights-only checkpoints
(`model_<step>.pt`, about 2 MB) are copied to the laptop with `scp` into the same path and evaluated there.
