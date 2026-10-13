# Revalidate saved depth Rhat arithmetic

Jed diagnostic job 66550618 confirmed that the absolute variance cutoff in
the original split-Rhat implementation caused two false infinity flags for
tiny depth weights. Positive coordinate rescaling gives the same mathematical
statistic, with values 1.000270 and 1.000370 instead. The threshold stays 1.05.

This stage applies that arithmetic to **all 32 fixed pilot references and all
nine depth coordinates**, including previously ready calculations. It can
also expose a failure previously hidden by the cutoff's unit fallback.
It uses saved samples, with no sampling or transformer inference.

## Scope and preserved checks

Only the depth contribution to the split-Rhat gate, its resulting mixing
flag, and derived readiness decisions can change. Token-probability and
exponent/log-likelihood Rhat, effective sample sizes, scalar mixing and
precision, row integration, Jensen checks, and the complete-cohort numerical
budgets and 16-versus-32 integration comparisons retain their original rules.
Losses and probability estimates are unchanged.

An exactly common constant retains the original unit convention and is
explicitly labeled uninformative about mixing. Different constant chains
fail. No coordinate is exempted because its mean is small.

Original plans, workers, sample files, audit failures and recovery reports
remain untouched. New records bind the original inputs and recovered audits
by hashes, and include both old and corrected decisions. Historical
`per_sequence_precision_passed` flags remain historical; they are not the
cohort acceptance criterion. All fixed cases stay in the cohort comparison.
The pilot still has only two source sequences per depth and cannot establish
population prediction accuracy.

## User-led launch on Jed

After updating the checkout, preview without writing outputs or submitting:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_depth_revalidation.sh
```

Submit one compute job (30 minutes, two CPUs, 12 GiB):

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_depth_revalidation.sh --submit
```

Saved arrays and cohort reductions are processed inside the allocation,
not on the login node. The input is the complete recovery directory
`output/shared_base_posterior_audit_recovery_2026-09-17`. Outputs go to the
separate `output/shared_base_depth_revalidation_2026-09-17` directory:

- `launch.json`: fixed scope and code hashes.
- `tasks/*.json`: original provenance, all nine depth diagnostics and
  old/corrected decisions for each reference.
- `summary.json`: complete fixed-cohort comparisons, including failures.
- `logs/revalidation-JOBID.out`: progress and final counts.

Completion of this job means the audit ran, not that its scientific checks
all passed. Inspect `all_numerical_checks_passed` and each group's
`numerical_audit`. Existing effective-sample-size, sampling-precision and
integration failures require their own follow-up. Repeated runs verify and
reuse immutable records rather than replacing them.
