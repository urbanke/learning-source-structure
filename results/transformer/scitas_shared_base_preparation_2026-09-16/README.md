# Prepared SCITAS likelihood experiment

This is a software/deployment verification record, **not a new scientific
comparison**. No cluster job was submitted while preparing it. The paper's
long-context transformer-versus-mixture comparison remains unresolved.

The frozen plan contains six five-token calibration controls and twelve
held-out sequences of length 1025, four per family. The latter reuse saved
transformer scores and existing product/uniform Bayesian likelihoods. Only
the missing shared-base likelihoods require new computation. Generating
parameters are not exported to inference.

`verification.json` records the following local checks:

* The 104 MB dependency archive was extracted and all 43 source/store files
  verified. The existing C kernel compiled and the sealed store opened read-only.
* All six CPU preflight tasks ran through the cluster dispatcher and merged
  successfully: twelve independent repetitions, 105 seconds total local task
  time, maximum resident memory 293 MB. Two repetitions per case test execution,
  not accuracy. Global repetition indices, seeds and likelihood estimates remain
  in this compact record; full diagnostic shards stay in ignored output storage.
* The experimental batched sampler completed eighteen tiny CPU-mode executions
  across the six controls. Same-state targets agree with the original numerical
  kernel to less than 1e-9 in natural log units. This verifies the implemented
  short target; it does not establish Monte Carlo precision or GPU speed.
* The comparison program reconstructs the actual three-family sequence mixture
  and checks paired inputs. Its smoke output contains only the existing exact
  one/four-prediction points. Missing long-context calculations remain pending.

The GPU implementation is restricted to the six short controls. Actual CUDA
execution, SCITAS costs and long-context precision remain to be measured.
Higher-effort and quadrature checks are required before scientific use.

Commands, stage budgets, manual resumptions and prerequisites are documented in
`code/transformer/scitas/README.md`. The frozen protocol and dependency file
hashes are in `code/transformer/configs/shared_base_*_2026-09-16.json`.
The dependency archive remains outside Git at
`output/scitas_shared_base_2026-09-16/dependencies.tar.gz`.
