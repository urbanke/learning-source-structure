# Guide for agents and automated tools

This is the code repository of the paper *Learning Source Structure: Bayesian Prediction
with Transformers*. The paper, the lab log (`notes/log.md`) and the standing working
rules are in the paper repository `layered_architectures`, checked out next to this one.
Read its `AGENTS.md` and `notes/log.md` first.

## Start here

- `README.md`: setup, the smoke test, how to regenerate figures, re-evaluate models,
  retrain, and recompute the Bayesian references.
- `docs/MAP.md`: for every figure and table of the paper, the script that makes it, the
  files it reads, and where those files are.
- `results_manifest.json`: every campaign with its command, smoke variant, inputs,
  outputs and checks.
- The README files under `code/` are the working notes of the campaigns, September 2026.

## Rules

- Big data, caches and experiment outputs stay out of git (`output/`,
  `results/transformer/eval`, the reference-predictor outputs). Summaries, receipts,
  final figures and tables go to `results/`.
- A figure the paper includes is copied to `figures/figureN.pdf` in the paper
  repository under its paper number.
- Every figure script checks its inputs' hashes and writes a receipt. Keep that.
- Validation checks must pass before any experiment is trusted. Verify that cluster
  jobs are actually running right after launch.
- Fixed seeds as defaults. Everything in bits.
- The Bayesian reference code runs on the public LSA engine through `code/compat`;
  do not reintroduce a dependency on the private development package.
- Never commit credentials.

## Commit style

Short imperative subject lines. Commits made by an agent carry the standard co-author
trailer identifying the model and session.
