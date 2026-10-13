# Product source: a fresh test of required context

Complete. The estimated Bayesian and transformer requirements coincide in
six of twelve comparisons. In the other six the transformer needs the next
tested time, a ratio of 1.182--1.203. All twelve simultaneous compatibility
sets overlap. These are estimates for one fixed trained checkpoint and a
discrete time grid; they do not establish equality of population thresholds.
All 4096 sequences and all 12,288 raw records passed the final audit.

This experiment predicts how much context the existing transformer needs to
reach expected next-token regrets of 0.1, 0.2 and 0.5 bits. Four fixed source
conditions are retained: coordinate depths (1,1), (1,8), (4,4), (8,8). Neither
predictor is told the depths. Each condition has 1024 fresh sequences of
length 1025, with independent new transition matrices and coordinate label
permutations. The transformer is the existing `prod_8x8/model_100000.pt`;
no new training or checkpoint selection occurs.

The 26 tested numbers of observed tokens are
1,2,4,8,16,32,39,46,54,64,77,91,108,128,153,182,216,256,305,363,431,512,609,
725,862,1024. This grid was chosen solely from development Bayesian means,
using the rule already in `section4_protocols.tex`. A context requirement is
the first **tested** time meeting a target. No time interpolation, smoothing,
or assumption of permanent subsequent attainment is used.

The frozen plan is in ignored
`output/product_context_confirmation_2026-09-17/plan.json`, with identifier
`baf587e003e964cbfcf148bdaa7739c1eebe6e863348fcb50372c361a42008e5`.
It binds the code, checkpoint, original receipts, numerical settings, sealed
LSA table store, seeds and complete cohort sizes. All 4096 derived generator
seeds were checked for uniqueness and disjointness from the 1024 development
sequences. Benchmarking uses only 32 development sequences, eight per condition.

The complete Bayesian curves and their thresholds must be saved and sealed
before the fresh transformer evaluation starts. The result remains paired:
each predictor sees the same sequence and is compared with the same true
next-token distribution. Numerical checks replay old predictions, use wider
integration bounds, check the depth-one reference against exact Dirichlet
probabilities/evidence, and test causality.

Uncertainty uses 4096 whole-sequence bootstrap resamples. Both predictors and
all 26 times remain together; different source conditions are resampled
independently. One approximate 95% simultaneous band covers all four conditions,
both predictors and all tested times. Its inversion retains all compatible
first crossings, including nonattainment. Bootstrap nonattainment fractions
are diagnostics, not probabilities of infinite population context. This
uncertainty excludes variation across training runs: one fixed checkpoint
is being tested.

Run the stages in order, using the existing Python environment and sealed
store. The benchmark and forecast stages refuse to overwrite completed
receipts. The reference and model stages reuse matching per-sequence files
after interruption.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LSA_STORE=/Users/ruediger/Projects/product_model_with_memory/tables/anchors_prod
export PMM_SRC=/Users/ruediger/Projects/product_model_with_memory/src
python code/scripts/product_context_confirmation.py prepare
python code/scripts/product_context_confirmation.py benchmark
python code/scripts/product_context_confirmation.py references --workers 4
python code/scripts/product_context_confirmation.py forecast
python code/scripts/product_context_confirmation.py model --device cpu --batch 8 --torch-threads 4
python code/scripts/product_context_confirmation.py summary
python code/scripts/product_context_figures.py
python code/scripts/product_context_figures.py --check-only
```

Raw sources, probability vectors, per-sequence risks, original frozen plan,
forecast seal and runtime logs stay in `output/`. The compact summary retains
the complete frozen plan and input hashes. `requirements.tsv` lists all 12
comparisons and their compatible tested lengths. `risk_curves.pdf` now shows
the four target regions in main-text Section4.1, at native5.5-inch width with
readable labels and points marking tested times. Horizontal lines mark the
three targets; the approximate95%simultaneous bands are unchanged. The
appendix retains the threshold table and protocol and refers to this main
figure. The main context forecast is now presented alone in
`../section4_unseen_context_2026-09-17/context_forecast.pdf`; it uses the
same validated summary and all numerical values are unchanged.
`overview.pdf` preserves the earlier combined presentation with the
exploratory unseen-row panel. The plotting audit independently
recomputes regrets and thresholds from all 12,288 raw records.
