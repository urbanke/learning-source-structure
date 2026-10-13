# Recover the 16 September posterior audits

The 32-task Jed pilot `66543820` saved its calculations but failed the
independent audit at `model predictions differ between loaders`. The two
loaders read the saved probabilities and reconstruct `exp(saved_logp)`,
respectively. On Jed, all 16 transferred model files differ by at most one
floating-point step (maximum absolute difference 5.55e-17). The original
audit incorrectly demanded bitwise equality across platforms.

This recovery changes only that redundant comparison. The independent loader
still verifies the checkpoints, saved logits, checksums and causal alignment.
The adapter checks the reconstruction against the saved vector using both
the original 1e-15 absolute bound and a one-ULP bound, then returns the
unchanged saved probabilities. All posterior, convergence, precision and
quadrature checks remain in force. No sampler is called.

Original plans, workers, auditors, samples and failed receipts remain
unchanged. Recovery receipts go to the separate directory
`output/shared_base_posterior_audit_recovery_2026-09-17/`. Every recovery
receipt binds the old failed audit, original samples, runtime and new code.

## User-led launch on Jed

Update the existing `layered_architectures_mixture_bayes` checkout from
GitHub. Start with one saved task:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior_recovery.sh --only-task 0 --submit
```

The launcher schedules the audit on a compute node (two CPU cores, 12 GiB,
30-minute reservation). It does not perform the numerical audit on the
login node. Review its scheduler result and recovery log before expanding.
A successful first audit should report `ready`; a substantive numerical
failure must be investigated rather than treated as the rounding problem.

After the first audit passes, process all fixed tasks:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior_recovery.sh --submit
```

The previously completed recovery is checked and reused. Once all tasks
finish, schedule the strict merge:

```bash
bash ~/Projects/layered_architectures_mixture_bayes/code/transformer/scitas/submit_shared_base_posterior_recovery.sh --merge --submit
```

The merge requires all 32 saved calculations and both exponent-integration
orders for every declared condition. Unresolved references remain explicit.
Even a passing result is an eight-source numerical pilot, not a population
learning curve or authorization to launch a large sampling campaign.
