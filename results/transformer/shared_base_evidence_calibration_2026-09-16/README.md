# Shared-base likelihood calibration

Historical numerical pilot: all 224 estimates completed (16 repetitions per
case/configuration). Both samplers failed all six initial controls; particle
refinement on AAAAA also failed the fixed 0.01-bit error-plus-uncertainty
criterion. This pilot does not validate a production Bayesian reference.
`summary.json` retains all estimates, timings, identities and failed gates;
a successful report audit does not change that status.

From the repository root:

```bash
python code/scripts/validate_shared_base_calibration_report.py --summary results/transformer/shared_base_evidence_calibration_2026-09-16/summary.json
```

The original explanation and full sampling/report commands are preserved
byte-for-byte in `notes/archives/copying128_preview_2026-09-25.tar.gz`, member
`preview/before/results/transformer/shared_base_evidence_calibration_2026-09-16/README.md`.
