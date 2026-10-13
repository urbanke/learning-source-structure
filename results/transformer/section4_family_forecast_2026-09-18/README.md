# Early Bayesian forecast for an unknown source family

This completes the short-sequence calculation for Section 4.4. It uses the
existing three-family prior and the two already trained pooled transformers.
There is no new training, source sampling, transformer inference or posterior
sampling. The longer-sequence comparison remains unfinished.

## Quantities and result

The first token is supplied. `n` predictions score tokens 2 through n+1.
Both curves subtract the cumulative loss of Bayes told the generating family,
but not the source parameters. The three full family priors have equal weight:
8x8 product, shared-base LSA, and uniform-base control.

The prior-averaged Bayesian costs after 1, 2, 3 and 4 predictions are
0.0116323, 0.0441474, 0.1022745 and 0.1756182 bits. Complete enumeration of
coordinate equality patterns replaces sampling: at most 2704 representatives
cover all 64^5 labelled sequences. Numerical quadrature remains in the component
likelihoods. No probability masses are rescaled.

The existing 1024 test sequences per family provide paired transformer-minus-
mixture losses at n=1 and n=4. Adding their mean to the calculated population
Bayesian cost is a fixed-coefficient (one) control-variate estimate of the
transformer cost. At n=4:

| Training run | Transformer cost, bits | Excess over mixture, bits |
|---|---:|---:|
| 1 | 0.209322 +/- 0.006617 | 0.033703 +/- 0.006617 |
| 2 | 0.211644 +/- 0.006855 | 0.036026 +/- 0.006855 |

Errors are one standard error over independently sampled test sequences,
stratified by family; they exclude variation across training runs and numerical
quadrature error. Direct sample means remain in `summary.json`, alongside the
control-variate estimates. The comparison finds a small resolved excess, not
equality with Bayes. It does not measure internal transformer family weights.

## Outputs

- `forecast.pdf/png`: main text, Bayesian mean n=0..4 and transformer run 1.
- `forecast_runs.pdf/png`: appendix, both runs and separate families at n=4.
- `summary.json`: complete finite-sum metrics, empirical comparisons, provenance.
- `validation.json`: numerical validation and bound hashes.
- `figures_validation.json`: plotted numbers and input/output hashes.
- Raw orbit arrays remain outside Git under
  `output/section4_family_forecast_2026-09-18/orbits.npz`.

## Reproduce

Use the scientific Python environment with NumPy, SciPy, Matplotlib and Pytest.
Set `LSA_STORE` and `PMM_SRC` to the project's verified LSA tables and backend.
The archived raw short-comparison arrays and numerical refinement reports must
also be available; the scripts verify their identities before using them.

```bash
python code/scripts/section4_family_forecast.py
python code/scripts/section4_family_forecast_figures.py
python code/scripts/section4_family_forecast.py --check-only
python code/scripts/section4_family_forecast_figures.py --check-only
python -m pytest -q code/tests/test_section4_family_forecast.py
```

Validation includes exhaustive small-alphabet comparison against direct labelled
enumeration, 123 coordinate-relabeling checks, exact replay of all 18,432 archived
component likelihoods, probability normalization without rescaling, entropy and
posterior identities, and integration refinements. Maximum normalization residual
is 1.46e-11; refined cost differences are below 2.8e-14 bits. Refinement stability
is a numerical check, not a rigorous quadrature error bound.

The appendix also proves a source-only result: the limiting Bayesian cost is
at least log2(3)-16/27 = 0.99237 bits and strictly less than log2(3). Positive-depth
shared-base and uniform-base priors overlap, leaving residual family ambiguity
even when the transition matrix is known. This does not establish the
long-context behavior of the trained transformer.
