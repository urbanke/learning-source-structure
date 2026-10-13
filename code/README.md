# Numerical reproduction

The Appendix B commands and fast smoke variants are recorded in
[`results_manifest.json`](../results_manifest.json). Small final results are
committed under `results/mechanism/`; generated tables and caches remain outside
Git. The scripts require Python with NumPy, SciPy, and Matplotlib. The paired
LSA check additionally requires the companion `lsa` package and a pre-existing
sealed anchor store, supplied through `--store` (or `$LA_STORE` in the manifest).

The principal checks are:

- `appendix_b_checks.py`: independently enumerates permutations to validate the
  subset recurrence; compares the uniform-prior measurements with the exact
  finite sum; checks paired code-length differences; recomputes figure coverage.
- `master_curve_validation.py`: compares quadrature resolutions, both tail
  truncations, interpolation widths, and the independent depth-one formula.
- `necessary_examples.py`: regenerates the two-prior measurements with seed 7.
  The Zipf mixture uses a stable subset recurrence for a permanent, reducing
  computation from factorial enumeration to `O(d 2**d)` per count vector. It
  evaluates likelihood ratios directly to avoid cancellation near the limit.
- `mechanism_quantities.py rho`: records both the mixture over depths and the
  generating-depth component, together with the paired excess, when that depth
  is included in the mixture.

Standard errors use sample variance (`ddof=1`). The five legacy 32-draw layered
aggregates were converted once by multiplying their original SEs by
`sqrt(32/31)`; their metadata records the conversion. Repeating the explicit
`--rescale-legacy` operation does not rescale them again. Sample means and source
seeds were unchanged.

The numerical master curve uses a finite table and asymptotic extrapolation.
Its convergence study reports observed differences at specified check points,
not a rigorous uniform accuracy guarantee. The paired validation uses 32 draws
at `d=64`, depth 3, and at most 512 samples. It is a small statistical check,
not a proof of the depth-mixture inequality.

## Appendix C checks

`context_checks_summary.py` recomputes the statistics behind the stored row,
chain, and crossing figures. Its explicit `--rescale-legacy` operation converts
the 32-repetition row data and 8-repetition chain/crossing data to sample standard
errors once, with metadata. The eight-table one-step information data use the
same sample-variance correction and now retain the SE separately from the SD.
The original means are unchanged. In the chain and
crossing experiments, `n` counts transitions; both codes omit the initial token.
The tilted row product is a transition reference under a stationary start,
not the full Bayes mixture for that initialization. Stationary laws are solved
numerically. Future chain runs retain the paired comparison error and its SE;
the old aggregates did not retain the covariance needed to reconstruct it.

`context_concentrated_check.py` studies iid tilted rows for Zipf exponents above
one. At depth one it evaluates the true normalized-row Bayes mixture with
adaptive quadrature, checked against the uniform-prior closed form and a complete
small-alphabet probability sum. It reports the numerical integral error estimate
and an analytic bound on the omitted log-concave tails. The cheaper interpolated
routine remains in the legacy large-chain pipeline; at the recorded uniform
check points its largest difference from the closed form is about 0.000304 nats,
so the previous 0.000001-nat claim is not supported. These checks do not certify
an error bound for all possible count vectors.

The concentrated-row experiment uses the same weight prefixes and sampling
uniforms across alphabet sizes. Its separate normalizer experiment compares
sample variances with the exact finite- and infinite-alphabet variances.
`context_stationary_check.py` uses common matrix prefixes to compare stationary
laws and one-step information at increasing alphabet sizes; it does not generate
sample trajectories. Successive total-variation differences are not distances
to the unknown infinite-alphabet limit. All commands and smoke variants are in
the manifest. The new small results illustrate the mathematical claims and do
not replace their proofs.
