# Inspect tiny depth-weight diagnostics

The recovered 16 September pilot has two infinite split-Rhat flags for the
conditional depth-2 weight. Saved means are about 6e-17 and 6e-18; token
prediction, exponent and log-likelihood Rhat checks pass. The old arithmetic
uses an absolute variance cutoff, so the flag needs investigation before
spending more on sampling.

This separate diagnostic reads both exponent-integration orders for
L4/index0/time1024 and L8/index0/time256 (tasks18,19,24,25), retaining all nine
depth coordinates. It checks the original and recovered receipts, hashes,
runtime and saved arrays, and reproduces the old means and Rhat values.
It then computes classic split Rhat after scaling each coordinate by its
maximum saved value, using exact zero-variance handling. The mathematical
ratio is scale invariant. A constant coordinate is explicitly uninformative;
chains stuck at different constants remain a failure. Chain and half-chain
weight ranges are included to make the calculation reviewable.

No sample, prediction or acceptance decision changes. This is a diagnostic
of the stored samples and cannot rule out an undiscovered posterior mode.
The separate ESS, precision-budget and quadrature failures remain in force.

On Jed, after updating the existing mixture-Bayes checkout:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_depth_diagnostics.sh --submit
```

Omit `--submit` for a dry run. The launcher uses a 15-minute compute-node
allocation with two CPU cores and 12 GiB RAM. It performs no array processing
on the login node and calls no sampler. Output is separate:
`output/shared_base_depth_diagnostics_2026-09-17/`. The log prints the affected
coordinate and any coordinate triggering the old cutoff. The complete JSON
contains every depth, provenance and both computations. A repeated identical
run verifies the input hashes and reuses the completed report.
