# Larger product models under source mismatch

Both existing four-layer runs evaluated on all 1024 original relabeled
histories at depths 1 and 4. Training family/budget retained; no new training.
Kuma 4405807 completed in 123 seconds. Independent complete-cohort replay passed.

Depth 4 gap at 32 tokens: two-layer 0.20899 bits, larger 0.06165/0.06242.
Largest gaps over the tested grid: 0.20899 versus 0.08720/0.09517.
Both seeds are retained separately. Predictive KL also improves strongly
around 32 tokens; no uniform improvement at every later context is asserted.

curves.npz stores all 27 times, mean/SE/pointwise 95% bounds for regrets,
paired gaps and predictive KL. provenance.json.gz binds the arrays, original
records, both final checkpoint identities, and independent validation.
Raw per-sequence predictions remain in the corresponding output/ directory
on this laptop and Kuma. Code/tests/validator are preserved in
notes/archives/product_mismatch_larger_2026-09-23.tar.gz.
Figure 2 and the manuscript are unchanged pending the presentation decision.
