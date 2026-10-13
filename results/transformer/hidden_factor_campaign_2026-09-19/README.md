# Hidden-encoding campaign: reused comparison

All models use the same 3,840 previously examined sequences, excluding the first 256 development cases. The 4-layer model is final at 400,000 updates (25.6 million sequences). The 8-layer model is final at 400,000 updates (25.6 million sequences). This comparison supplies no fresh independent confirmation.

Expected next-token regret after 1,024 observed tokens; mean ± one standard error across sequences:

| Model | Training run 1 | Training run 2 |
|---|---:|---:|
| 2 layers, 6.4M sequences | 0.14303 ± 0.00142 | 0.13349 ± 0.00134 |
| 2 layers, 25.6M sequences | 0.12834 ± 0.00128 | 0.11991 ± 0.00120 |
| 4 layers, 6.4M sequences | 0.10913 ± 0.00111 | 0.07515 ± 0.00084 |
| 4 layers, 25.6M sequences | 0.10669 ± 0.00107 | 0.05629 ± 0.00070 |
| 8 layers, 25.6M sequences | 0.04022 ± 0.00049 | 0.03965 ± 0.00047 |

Informed Bayes: 0.03781 ± 0.00044 bits.

The main panel uses fixed training run 1: two-layer 100,000-update model in blue, four-layer continuation in green, and eight-layer final model in purple. The appendix figure shows both runs separately and adds the four-layer 100,000-update model in gray. The two-layer transformer is solid, all larger transformers use distinct dashed variants, and informed Bayes is shown by unconnected black open circles with error bars. No seed is selected or averaged. All six older models remain in the checked data and table.

The consumer reconstructs true transition rows from the source matrices and encoding, verifies all old model and reference records, checks complete remote status/selection receipts and checkpoint hashes against their development evaluations, replays next-token probabilities/losses, and independently recomputes all means, paired contrasts and standard errors. Cumulative losses are aggregated from saved arrays; weight binaries and exact Bayesian enumeration are not rerun. Different training schedules and joint depth/width changes prevent these observations from identifying a capacity-only effect or establishing convergence.

Reproduce with `python code/scripts/hidden_factor_campaign_results.py --xlarge-step 400000`. Historical comparisons also accept explicit input and development directories.
