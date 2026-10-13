# Shared-base long-context numerical pilot

Eight predeclared sequences of length 1025: indices 0 and 1 at each depth 0, 1,
4 and 8, with exponent 1. Each predictor observes 256 or 1024 tokens of the same sequence.
Regret is the expected next-token log loss minus the true transition-row entropy,
in bits. Both original and larger transformers and both training runs are retained.

Bayes uses the full training prior (unknown exponent, depth, and label ranks).
The reported prediction uses exponent order 32, except depth 0 at time 1024 where
the predeclared refined order 64 is used. Declared chain continuations replace
their predecessors. The independent repeat only checks agreement and never
replaces the selected prediction. All eight depth/time numerical gates pass.

`summary.json` retains each case, matched group summaries, model/source identity,
Bayesian report hashes and all local input SHA256 receipts. `records.tsv` is a
compact rendering of the same sixteen comparisons. Bayesian MCSE describes
numerical sampling uncertainty; it is separate from variation across source
sequences. Two sequences per depth do not establish a population learning curve.
The cluster raw-chain audits are reused; local extraction independently checks
the frozen identities, saved probabilities, true-row regrets and cohort means.

Recreate from the transferred JSON summaries and existing local model outputs:

```bash
python code/scripts/shared_base_long_context_results.py
python code/scripts/shared_base_long_context_results.py --check-only
```

No posterior sampling, transformer inference or new experiment is performed.
