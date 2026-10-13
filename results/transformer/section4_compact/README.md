# Section 4 overview

`overview.pdf` is the paper figure (5.5 by 6.75 inches); `overview.png` is its
240-dpi inspection copy. The builder reads saved results, including the
explicitly provisional 128-token comparison, and applies no smoothing,
probability clipping or inference.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/layered-section4-compact-mpl \
python code/scripts/section4_compact_figure.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/layered-section4-compact-mpl \
python code/scripts/section4_compact_figure.py --check-only
```

Requires NumPy, SciPy and Matplotlib. `--out /tmp/name` builds separately.
The local `receipt.json` binds inputs, replay checks, display choices and outputs.

## Display and inputs

All panels use a base-2 axis of observed tokens. Lines denote transformer
quantities, open markers informed Bayes, and gray dotted triangles token
counts. Matched predictors share a condition color. Blue is the primary
color; orange identifies the second comparison in (c) and copying depth 3.
The legend matches Figure 1: two-layer transformers are solid, larger models
are dashed (the second four-layer run in (a) is dotted). Both copying curves
are dashed; color and Bayesian marker shape distinguish their depths.

- (a) uses the original 1024 depth-(4,4) random-label test histories, the
  two-layer model, both four-layer runs, and the saved stationary projection
  limit. Four-layer curves are dashed/dotted; Bayes markers are displayed at
  powers of 2 without changing any underlying statistics. The independently
  validated larger-model input is `product_mismatch_larger_2026-09-23`.
- (b,c) retain the original product-change data and all saved times; (c)
  compares measured two-matrix excess with twice the one-matrix excess.
- (d,e) use the eight-layer shared-base unseen-context model at depths 1/8;
  (f) uses the two-layer uniform-base control. Selected sample counts at
  times 4/16/64 are 214/136/59, 228/161/57 and 98/80/40 respectively.
- (g,h) use the fixed run 1 ordered/shuffled four-layer pair at depths 2/3,
  over 16,384 histories and all 1,024 prediction times. Approved Bayes points
  through 16 use the original 256 histories; markers at 1, 2, 4, 8 and 16
  appear on the main axes. The accepted `copying_bayes_endpoint32_2026-09-24` and
  `copying_bayes_endpoint64_2026-09-24` releases add t=32 and t=64 only
  in (g); cumulative (h) remains through 16.
  The provisional 128 release uses all 256 original histories at each of
  depths 2 and 3, retaining 32 for the depth-0 control.
  Its one-bank frequency integral uses 288 saved latent draws; its numerical accuracy and posterior checks remain unverified.
  Their gain is relative to the stationary-frequency surrogate, which
  approximates the finite shuffled-training law.

Panels (a--c) use mean plus/minus 1.96 sequence SE; (d--f) use one SE.
Copying transformer bands are pointwise 95% paired whole-source bootstrap
intervals; copying Bayesian bars show one source SE. Training-run variation
is separate. No success-dependent history selection, averaging across training seeds, insets,
or smoothing are used.

## Copying release

`full_copying_curves.npz` contains means, SEs and bootstrap bounds for the
16,384-history cohort. `full_copying_provenance.json` binds it to
`independent_validation.json`, the complete raw scores and immutable plan.
All previous 1,024 histories and scores are retained exactly. The new
independent validator replays source identity, both model protocols,
paired means, SEs and 1,000 bootstrap replicates for the full, retained and
new cohorts, including the depth 0 control. The 256-history Bayesian precision release at `copying_bayes_precision_2026-09-24`
is separately checked by the figure builder. Its complete 11,224-reference
cohort and 256 frequency banks pass independent replay and the unchanged
instantaneous/cumulative numerical budgets. Improved early estimates replace
the previous t=1--8 estimates; source histories and neural scores are unchanged.
This release does not approve a copying-payoff crossing time.

Raw data remain on Kuma in
`~/Projects/layered_architectures_hidden_factor/output/shared_base_shuffle_extension_2026-09-23/`.
Driver, frozen plan, tests, independent exporter and job receipts are preserved
in `notes/archives/figure2_copying_extension_2026-09-23.tar.gz`.
