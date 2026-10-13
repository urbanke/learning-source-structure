# Product Markov comparisons at fixed source depths

The four depth pairs were chosen before evaluating the curves: (1,1),
(1,8), (4,4), and (8,8). Each panel averages 256 independent draws of the
two transition matrices and their independently relabelled length-1025
sequences. At time t, the predictor observes the first t tokens and predicts
token t+1. The reported times are 1, 2, 4, ..., 1024.

Both predictors retain the training family. The transformer is the existing
`prod_8x8/model_100000.pt` checkpoint. The Bayesian predictor averages over
independent depths uniform on 1 through 8, with each coordinate's depth
shared across its rows. Neither receives the selected test depths.

`curves.pdf` plots expected next-token regret in bits, computed as the
KL divergence from the true transition row to each predictor. Bands show
one standard error across independent matrix/sequence draws. `gaps.pdf`
plots transformer regret minus Bayesian regret, with paired standard errors.
These are descriptive comparisons for one frozen trained model; the
uncertainty bands do not include training-seed variation.

The overall learning curves follow each other across all four settings,
but their differences depend on the source. At time 32, the excess
transformer regret is 0.1238 +/- 0.0089 bits at depths (1,1),
0.0753 +/- 0.0069 at (1,8), 0.0009 +/- 0.0054 at (4,4), and
0.0230 +/- 0.0057 at (8,8). The full-prior average obscures this contrast.
At time 1024 the corresponding excesses are between 0.0040 and 0.0086
bits. A negative conditional difference is possible: the Bayesian rule is
optimal for the full training prior, rather than every fixed-depth subset.

The summaries and validation are machine-readable. `receipt.json` binds the
code, checkpoint, raw records, summaries and figures by SHA256 hashes.
The numerical reference reuses the existing sealed-store LSA implementation.
Validation includes an expanded integration range, exact depth-one Dirichlet
probabilities and evidence, source replay, relabeling and causal time indexing.

Raw source and prediction files remain in `output/conditional_accuracy/product/`.
The receipt records the reproduction commands and table-store settings.
After the numerical driver, run `conditional_product_accuracy_figures.py`
to produce the final figures from the saved summary, then run
`conditional_product_accuracy_receipt.py` to verify and bind the artifacts.
