# Validated copying comparison through 16 observations

`paired_scores.npz` and `provenance.json.gz` supply the informed-Bayes circles
in Figure 2(g,h). They preserve all 768 original histories (256 each at
depths 0, 2 and 3) and both transformer training seeds. The independent
release validates all 11,224 canonical posterior references and 256 shared
frequency banks. All 96 instantaneous/cumulative mean coordinates satisfy
the unchanged numerical allowance of 0.01/3 bits.

The frequency-only comparator is the source-induced stationary-frequency
surrogate, not exact Bayes for finite shuffling. Its paired quadrature rules
use 32 and 16 alpha nodes. Informed Bayes uses validated deterministic rank
marginalization for times 1–4 and audited posterior sampling with both
integration rules for times 5–16. Earlier sampled values were refined; they
are not required to equal the previous eight-token release.

Source: Jed job `66683678`,
`/scratch/urbanke/copying_bayes_precision_release_evening_2026-09-23`.
Release plan ID:
`6d63f2870406e5fd455bffff2e87e7c2cf48c512a95aed1e8bd8d665e9cc1659`.
The compressed provenance retains the complete source plan, approval,
repair selection, artifact hashes, neural-score identity checks and export
receipt. Full raw-chain bindings remain in the source release.

Reproduce the compact export with `code/scripts/copying_precision_figure_input.py`
using the source release and the original shuffle evaluation directory.
Validate the exported release with:

```
python3 code/scripts/section4_compact_figure.py --check-only
python3 code/tests/test_copying_precision_figure_input.py
```

This release approves instantaneous and cumulative gains through 16 only.
It does not approve a payoff-crossing estimate or longer-context values.
