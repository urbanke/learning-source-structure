# Guide for agents and automated tools

This is the code repository of the paper *Learning Source Structure:
Bayesian Prediction with Transformers*. The paper, the lab log
(`notes/log.md`) and the standing working rules are in the paper
repository `layered_architectures`, checked out next to this one.
Read its `AGENTS.md` and `notes/log.md` first.

## Layout

See `README.md`. `results_manifest.json` maps every figure and table to
the command that makes it, with a fast smoke variant.

## Rules

- Big data, caches and experiment outputs stay out of git (`output/`,
  `results/transformer/eval`, the reference-predictor outputs). Small
  final figures and tables go to `results/`.
- A figure the paper includes is copied to `figures/figureN.pdf` in the
  paper repository under its paper number.
- Validation checks must pass before any experiment is trusted. Verify
  that cluster jobs are actually running right after launch.
- Fixed seeds as defaults. Everything in bits.
- Never commit credentials.

## Commit style

Short imperative subject lines. Commits made by an agent carry the
standard co-author trailer identifying the model and session.
