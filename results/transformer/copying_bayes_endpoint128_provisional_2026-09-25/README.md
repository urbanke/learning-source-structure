# Preliminary copying comparison at 128 tokens

Figure 2(g) displays the paired stationary-frequency minus full informed
Bayesian log loss. L=2,3 each use all 256 original histories; L=0 retains its
fixed first 32 histories. All 544 histories are retained. Only L=2,3 are
plotted. This does not supply a cumulative point for panel (h).

This estimate is NOT numerically accepted. The user authorized provisional
display and the expanded cohort on 25 September; the caption retains the
remark awaiting verification. Frequency integration uses original bank 0,
all 32 exponent nodes, and its first draw at each positive depth. The
depth-zero atom is retained; weights preserve the original prior. Every
required count type is complete. The full Bayesian component uses all 544
fine-order references. All posterior audit failures remain recorded,
including 18 at L=2 and 48 at L=3. Source SE excludes numerical integration
and posterior error. The original 32-source scores and provenance are preserved exactly.

Paired gain in bits (mean +/- one source SE):
L=0: 0.001818231 +/- 0.002407969;
L=2: 0.214753528 +/- 0.019670635;
L=3: 0.406221800 +/- 0.027873972.

`paired_scores.npz` retains both predictors and both paired transformer runs.
`original_paired_scores.npz` preserves the initial 96-source release.
`provenance.json.gz` binds inputs, plans, original bank, complete reduced
integrals and full-Bayes captures. Earlier accepted readers are unchanged.
The separate eight-bank SCITAS outputs are preserved; no further campaign
or retry is authorized by this release.

Original reproduction: `notes/archives/copying128_preview_2026-09-25.tar.gz`.
Expansion and launch: `notes/archives/copying128_expansion_2026-09-25.tar.gz`.
This release and independent review:
`notes/archives/copying128_release_L3_2026-09-26.tar.gz`.
Use `release/derive_full_group.py` then `release/export_expanded.py` in the
expansion root, followed by `code/scripts/section4_compact_figure.py`.
Large inputs remain bound by their original paths and hashes in provenance.
