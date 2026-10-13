# learning-source-structure

Code, results and figure scripts for the paper *Learning Source Structure: Bayesian
Prediction with Transformers* by Meir Feder and Ruediger Urbanke.

Transformers are trained on sequences whose source parameters are drawn afresh for every
sequence. Their predictions are compared, in bits, with the informed Bayesian predictor
of the training family. The paper's figures and tables are drawn from the measurements
in `results/`; every one of them can be regenerated from this repository in minutes.
Retraining the models and recomputing the Bayesian references is possible too, and is
described below.

Everything is in bits. The companion code of the layered simplex architecture (LSA) is at
https://github.com/urbanke/layered-simplex-architecture.

## Layout

```
code/transformer/   sources (source.py, families.py), the GPT (model.py), training (train.py),
                    evaluation (eval_model.py), Bayesian references (refs_*.py, shared_*.py),
                    training configurations (configs/) and cluster launchers (kuma/, scitas/)
code/scripts/       one script per campaign, table or figure; building_block/ holds the LSA tables code
code/tests/         the test suite (about 1,750 tests)
code/compat/        adapter that runs the reference code on the public LSA engine (see below)
results/            the measurements behind every number in the paper, with receipts
results_manifest.json   every campaign with its command, smoke variant, inputs, outputs and checks
weights_manifest.json   the trained weights attached to release weights-v1, with SHA-256 and model sizes
docs/MAP.md         figure by figure and table by table: script, inputs, where the inputs are
output/             raw experiment outputs; not in git
```

The README files under `code/` are the working notes of September 2026, kept as they
were: commands, cluster settings and checks of each campaign.

## Setup

Python 3.10 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/transformer/requirements.txt   # numpy, torch, scipy, matplotlib
pip install pytest
```

## Check the setup (seconds)

```sh
python3 code/scripts/smoke_test.py
```

This trains a tiny transformer on coins with the two-point prior (the source of
Figure 1a), evaluates it, computes the exact informed-Bayes reference on the same
sequences, and prints the regret of both at several context lengths (a few seconds on a
laptop). It exercises the three programs the experiments ran. `--steps 2000` brings the
model closer to Bayes.

The test suite runs in about a minute:

```sh
python3 -m pytest -q code/tests
```

Twenty test modules need PyTorch and are skipped at collection without it. A few
tests check the disk-space guard of the campaign drivers and need 5 GB free.

## Regenerate the figures and tables (minutes)

The six figure scripts read saved summaries only; they sample nothing, train nothing and
compute no reference. Each checks the hashes of its inputs and writes a receipt next to
its output. Run from the repository root:

```sh
python3 code/scripts/section3_compact_figure.py          # Figure 1
python3 code/scripts/section4_compact_figure.py          # Figure 2
python3 code/scripts/appendix_source_figures.py          # Figures 3, 4, 6, 7
python3 code/scripts/appendix_specialist_pooled_figures.py   # Figures 5, 8
python3 code/scripts/appendix_application_figures.py     # Figures 9, 10, 11
python3 code/scripts/appendix_copying_figure.py          # Figure 12
```

Outputs: `results/transformer/section3_compact/overview.pdf`,
`results/transformer/section4_compact/overview.pdf` and
`results/transformer/appendix_style_2026-09-24/figure3.pdf` to `figure12.pdf`.
`docs/MAP.md` lists, for every figure and table, the files it is built from, and for
every table the summary its numbers come from. Two tables are generated files
(`results/transformer/product_shift_2026-09-18/endpoint_table.tex` and
`results/transformer/section4_unseen_context_refined_2026-09-20/paired_table.tex`).

## Re-evaluate a trained model (minutes)

The two-layer models of the paper are in `results/transformer/runs/<name>/`: a
`config.json`, the training log and the weights `model_<step>.pt` (about 2 MB each).
For example, the shared-base model of Figure 1e (`d64`), the product model (`prod_8x8`),
the uniform-base model (`lsa_a0_d64`) and the coin models (`bin_twopoint`,
`bin_uniform`). To evaluate one on fresh sequences of its family:

```sh
python3 code/transformer/eval_model.py --run results/transformer/runs/bin_twopoint --ckpt 20000 \
    --settings none --family_sets binary:twopoint --n_eval 256 --out output/check
```

`code/transformer/README.md` describes the evaluation files. The four- and eight-layer
models were trained on the Kuma cluster; their evaluations are in `results/`, and their
final weights are attached to the release
[weights-v1](https://github.com/urbanke/learning-source-structure/releases/tag/weights-v1)
(26 archives, 0.6 GB, two seeds of each of 13 models). Each archive is the run directory
of one model: `config.json`, `log.jsonl` and the final `model_<step>.pt` (the state dict).
`weights_manifest.json` lists every archive with its SHA-256, layers, width and training
steps. To evaluate one, extract it into `output/` and point `eval_model.py` at it, for
example the ordered-training model of the copying experiment:

```sh
curl -LO https://github.com/urbanke/learning-source-structure/releases/download/weights-v1/shared_base_scaling_2026-09-17__train__seed0.tar.gz
tar -C output -xzf shared_base_scaling_2026-09-17__train__seed0.tar.gz
python3 code/transformer/eval_model.py --run output/shared_base_scaling_2026-09-17/train/seed0 --ckpt 100000 ...
```

with the evaluation settings of the experiment, recorded in `results_manifest.json` and in
the README files under `code/transformer/kuma/`.

## Retrain (GPU hours)

`code/transformer/train.py` trains one model; `code/transformer/README.md` explains the
source families, the training stream and the controls. A two-layer model of the paper
is about one GPU hour (100,000 updates of 64 sequences of 1,024 tokens); the larger
models in `code/transformer/configs/*plan*.json` took 400,000 updates. The launchers in
`code/transformer/kuma/` and `code/transformer/scitas/` are written for the EPFL SCITAS
clusters and are kept as a record.

## Recompute the Bayesian references (CPU hours to days)

The references for the coins and for the hidden encoding are exact and need nothing
beyond this repository (`refs_family.py`, `hidden_factor_*.py`). The references for the
product, uniform-base, shared-base and pooled sources use the row integrals of the
layered simplex architecture, served from a store of universal tables. The engine and
the store builder are public:

```sh
git clone https://github.com/urbanke/layered-simplex-architecture.git ../lsa
git -C ../lsa checkout 16d5544
pip install -e ../lsa
LSA_BUILD_EXACT=1 python3 ../lsa/scripts/build_anchor_store.py --out ../lsa_store \
    --factor 0.083071 --levels 2-53 --r-max 200000000 --pad-anchors 8 --dense-below 256 \
    --jobs 8 --go
```

The store is 2.6 GB and costs about four core-hours (the exact builder; the dry run
without `--go` prints the plan). It is the store the paper's references were computed
with. We rebuilt levels 2 to 41 of it with this command on an x86-64 Linux machine and
compared them with the sealed store used for the paper, column by column: every column
has the same length, and the values agree to within 1e-15 of the column's scale. The
files are not byte-identical, because the sealed store was built on Apple Silicon and
the last bit of `exp` and `log` differs between platforms.
`code/scripts/compare_store.py` runs this comparison between two stores. Then, for
every reference command:

```sh
export LSA_STORE=$PWD/../lsa_store
export PMM_SRC=$PWD/code/compat
```

`code/compat` presents the public engine under the name the reference code was written
against; the rows it serves are identical to the development package's. The commands
of the reference computations are in `results_manifest.json` and in the README files
under `code/transformer/`; `docs/MAP.md` says which experiment each figure rests on.
The shared-base references (Monte Carlo over label ranks and exponent) are the
expensive ones: days of CPU for the complete cohorts.

## What is not in the repository

The raw per-sequence outputs of the experiments (`output/`, about 60 GB), the
intermediate checkpoints, and the reference-predictor outputs of the family experiments
(`results/transformer/eval`, `refs_fam`, `refs2048`). The summaries computed from them
are in `results/`; the final weights of the larger models are in the release
`weights-v1` (above). The raw outputs are available from the authors.
