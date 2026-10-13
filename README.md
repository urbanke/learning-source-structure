# learning-source-structure

Code, results and figure scripts for the paper *Learning Source
Structure: Bayesian Prediction with Transformers* (Meir Feder, Yaniv
Fogel, Ruediger Urbanke). The paper itself lives in the repository
`layered_architectures`, which the Overleaf project mirrors.

- `code/src/`: the installable package (source generators, Bayesian
  predictors, LSA row integrals).
- `code/transformer/`: training, evaluation, configurations and cluster
  launchers.
- `code/scripts/`: one script per experiment, table or figure.
- `code/tests/`: the test suite.
- `results/`: the measurements behind every number in the paper and the
  figures it includes (`results/transformer/section3_compact`,
  `section4_compact`, `appendix_style_2026-09-24`).
- `results_manifest.json`: maps every figure and table to the command
  that makes it.
- `output/`: experiment outputs, not in git.

Everything is in bits. The companion code of the LSA paper is at
https://github.com/urbanke/layered-simplex-architecture.
