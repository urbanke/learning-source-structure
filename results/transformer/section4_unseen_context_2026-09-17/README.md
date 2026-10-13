# Prediction after a token first appears: historical release

Original cohort:256 sequences per L=0,1,4,8, alpha=1. At t=4,16,64 select
histories whose current token has not occurred earlier. All predictors receive
the same selected history and retain their training prior. Scores use the
true next-token row; gain is6 minus expected log loss. Error bars are source
SE, not numerical uncertainty. The counts rule uses a different prior.

This historical release accepts all t=4/16 subsets and t=64 at L=0/1.
The L=4/8 t=64 subsets fail numerical-error allowance and remain withheld
here, despite basic/mixing checks passing. Later refinements are separate.
`validation.json: passed=true` validates export arithmetic and withholding,
not every Bayesian subset. Do not relabel historical failed values accepted.

Reproduce with pinned inputs, without sampling:
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/section4_unseen_context_extension.py
python code/scripts/section4_unseen_context_extension.py --verify-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/layered-section4-mpl python code/scripts/section4_unseen_context_figures.py
python code/scripts/section4_unseen_context_figures.py --check-only
```

The full original README (cohort counts, budgets, inputs, findings, validation
and figure details) is preserved byte-for-byte as
`before/results/transformer/section4_unseen_context_2026-09-17/README.md` in
`notes/archives/overleaf_merge_2026-09-25_1949.tar.gz`.
