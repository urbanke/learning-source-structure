# Hidden-encoding source: independent confirmation

All six final models use the same 3,840 unused sequences, indices 256 through 4095 of the frozen cohort. The first 256 sequences were used for exploratory model comparisons and are excluded here. No model is selected using this confirmation; both training runs remain separate.

Expected next-token regret after 1,024 observed tokens, in bits; uncertainty is one standard error across sequences:

| Predictor | Run 1 | Run 2 |
|---|---:|---:|
| Original | 0.14303 ± 0.00142 | 0.13349 ± 0.00134 |
| More training | 0.12834 ± 0.00128 | 0.11991 ± 0.00120 |
| Larger model | 0.10913 ± 0.00111 | 0.07515 ± 0.00084 |
| Informed Bayes | 0.03781 ± 0.00044 | 0.03781 ± 0.00044 |

The original model has two layers and 0.53 million parameters, trained on 6.4 million sequences. More training continues that model to 25.6 million sequences. The larger model has four layers and 3.43 million parameters at the original 6.4-million-sequence budget.

Both changes improve prediction in both runs; a clear Bayesian gap remains. The larger model with the longer training budget has not been tested. These results do not establish convergence or a general scaling law.

The independent consumer checks the six compact model arrays, all 4,096 saved Bayesian records, model and source identities, true transition rows, next-token KL and realized losses, all five source-only groups, and all paired means and standard errors. It aggregates saved cumulative losses without rerunning model inference. The exact Bayesian enumeration is not repeated. Recorded final checkpoint identities match the earlier validated pilot; weight binaries are not transferred.

Figures retain the earlier display convention: solid blue/orange/green transformer curves, dashed black Bayes with open circles, one panel per training run, one-SE bands. Run 1 is the preselected main-text comparison.

`figures/conditional_curves.pdf` separately shows the four source-only entropy groups for the original models. It retains the earlier conditional-figure style: training runs 1 and 2 in solid blue and green, hidden-encoding Bayes dashed orange, encoding-given Bayes dotted purple. All group counts exclude the development sequences; the original-model gap persists in every group.

Reproduce from saved outputs and references:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/hidden_factor_confirmation_results.py
```
