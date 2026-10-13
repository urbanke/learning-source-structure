# Pooled complete-cohort estimates

Figure1 and the pooled appendix plot show Bayes at16/64/256/1024 with ordinary
markers, as RU requested on24September. Each uses all192 frozen histories,
64 per family. Contexts16/256 use32 repetitions per history; context64 uses the fixed
precision allocation, with32–608 D repetitions per history (9760 total) and
56864 production repetitions across all four arms. Context1024 uses the
fixed first20 from every history. Final32-repetition ABCD acceptance remains incomplete.
Source error bars are stratified across histories, not numerical bounds.

Context16 passes final numerical validation with8192particles and32repetitions
per arm. Independent particle, exponent and row-integration refinements pass
both original and approved cancellation-aware assessments: allowances0.002803
and0.002048bits against0.003333. Replay and rows: context16_validation.json.gz
and context16_rows.json.gz. Contexts64/256/1024 remain provisional; the overall
acceptance flag is false. Contexts1/4 are unchanged.

The1024 snapshot preserves original seeds, evidence-weighted repetitions,
refined product/uniform components and equal family weights. No history was
selected by completion or result. Metadata and rows: context1024_snapshot.json.gz.
Collection/replay sources and proofs: notes/archives/compute_check_2026-09-25_0720.tar.gz.

Context64 uses the complete repaired precision campaign: 1.224270426 ±0.064779180 bits.
Arithmetic replay passes. The v2 total numerical allowance 0.001878926 is below
0.003333333 bits; basic stability checks still fail, so numerical acceptance
remains false. The complete short-calibration replay passes. Allocation was
chosen from the retained first32 outcomes; no fixed-design confidence guarantee
is claimed. Data and full provenance: context64_validation.json.gz and
context64_rows.json.gz. The raw snapshot preservation path and hash are recorded
in those receipts.

Raw first20 snapshot remains in ignored output/compute_check_2026-09-25_0720/pooled1024_first20/snapshot_retry.tar.gz.

The fixed top-up retained every base record and added all 8,096 planned
repetitions. Its complete-cohort replay and source-bound provenance are in
`notes/archives/compute_check_2026-09-25_1851.tar.gz`.
