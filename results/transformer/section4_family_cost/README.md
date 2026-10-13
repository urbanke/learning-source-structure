# Cost of an unknown source family

Historical paired comparison at one and four predictions: the Bayesian
mixture and two pooled transformer seeds are compared with Bayes given the
family but not its parameters. Each of three families contributes 1,024
sequences; means weight families equally and SEs use paired differences.
Only these two endpoints have validated references in this artifact; it makes
no claim about a longer-context cost curve. Data, figures and the audit remain
in this directory.

Reproduce and audit from the repository root with the saved source arrays:

```bash
python code/scripts/section4_family_cost.py
python code/scripts/section4_family_cost.py --check-only
```

Full definitions, results, data prerequisites and historical discussion are
preserved byte-for-byte in `notes/archives/copying128_preview_2026-09-25.tar.gz`,
member `preview/before/results/transformer/section4_family_cost/README.md`.
