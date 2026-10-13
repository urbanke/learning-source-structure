# Compact application figure

`overview.pdf` is Figure 2 in the main text of `mechanisms.tex`.
It recombines the audited, exploratory application analyses without new
sampling, Bayesian integration, model inference or training.

- Left: inverse learning curves for product depths (1,1) and (8,8).
  The requirement is the first **tested** length meeting the target,
  on the existing powers-of-two grid. The upper band means not reached.
  It is not extrapolation, and no threshold confidence intervals are shown.
- Right: expected next-token loss saved relative to uniform at time 16,
  on shared-base histories whose current token has not occurred earlier.
  All models see the same selected histories. Counts below the axis
  give the independent sequences in each group; bars are one SE.
  The counts-only comparator explains much of the gain and limits its
  interpretation as evidence for a richer mechanism.

The detailed figures and complete methods are in the adjacent
`section4_product_accuracy`, `section4_unseen_row_transfer`, and
`section4_family_cost` directories and in Appendix D. The family-cost
figure remains in the appendix because only two early times are available.

Reproduce after those result sets have been generated:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/layered-section4-mpl python code/scripts/section4_partial_figure.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/layered-section4-mpl python code/scripts/section4_partial_figure.py --check-only
```

The renderer checks the input summary hashes against their receipts and
records every input/output hash. Its 5.5-inch width matches the manuscript;
the minimum displayed font size is 7 points.
