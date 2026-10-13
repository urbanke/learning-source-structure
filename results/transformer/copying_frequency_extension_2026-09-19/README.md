# Copying comparator through eight observations

The stationary-frequency surrogate is evaluated at every time 1..8 on all
768 original ordered histories and both shuffled transformer runs. It retains
the original shared-base prior, takes the first token from q and subsequent
independent surrogate tokens from the stationary distribution of T. It does
not use test parameters. This remains an approximation to finite-tail
shuffling, not its exact Bayesian law.

The implementation reuses both frozen source banks from the first-four-token
calibration. A nonnegative polynomial-coefficient calculation groups labels
having equal tail counts. Multiplicity factorials restore distinct labelled
rank assignments. Source-bank quadrature and depth weights are preserved;
joint probabilities are averaged before conditional ratios. This computation
covers 164 required frequency types, including 138 new types. Its first four
predictions reproduce the earlier independent signed-moment calculation.

Both stages completed: approximately 11 seconds for coarse moments/scoring,
28 seconds for refined. Maximum absolute mean loss difference from either
shuffled transformer is **0.00424487 bits**, maximum prior-bank Monte Carlo
standard error **0.00018576 bits**, and maximum coarse/fine change
**0.00045994 bits**. The least balanced posterior bank weights have effective
size 15.14 out of 16 refined banks; this diagnoses only between-bank imbalance,
not the full adequacy of the finite source bank. Paired test-sequence errors
are reported separately. No payoff time or fresh confirmation is established.

Independent replay reconstructs every loss and leave-one-bank-out loss from
the saved moments, then every paired mean, sequence SE and prior jackknife SE.
The largest replay discrepancy is 3.56e-15 bits. Direct small-alphabet rank
enumeration and raw child/parent normalization provide separate kernel checks.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/copying_frequency_extension.py --stage stratified
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/copying_frequency_extension.py --stage refined
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/publish_copying_frequency_extension.py
python -m pytest -q code/tests/test_stationary_frequency_moments.py code/tests/test_copying_frequency_extension.py
```

Each producer stage has an exclusive claim, one attempt and a 1,800-second
allocation covering preparation, moments and scoring. Existing stage outputs
are not overwritten. Raw arrays, moments and frozen plans remain in ignored
`output/copying_frequency_extension_2026-09-19/`. The local summary/validation
JSON under this results directory hashes the inputs and generated arrays;
only this README is versioned. An initial coarse preflight was preserved with
its implementation before launch-lock and hard-deadline safeguards were added.

The extension covers the early neural-payoff region. Full informed Bayesian
predictions at times 5..8 are still missing, so this is a baseline calibration
and does not yet compare the Bayesian and neural payoff times.
