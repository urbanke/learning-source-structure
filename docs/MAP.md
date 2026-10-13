# Map: from the paper to the code and the results

Paper: *Learning Source Structure: Bayesian Prediction with Transformers* (`arxiv.tex` in the paper repository `layered_architectures`). This map lists, for every figure and table, the script that makes it and the files it reads, and where those files are. It was built on 13 October 2026 from the receipts the figure scripts wrote (`receipt.json` next to each figure), and every input hash below was checked against its receipt.

Status tags: `git` = in this repository; `disk` = on the research Mac under `results/`, ignored by git until now; `output` = on the research Mac under `output/`, never in git; `ext` = needs the sealed LSA anchor store and the `lsa` package (`LSA_STORE`, `PMM_SRC`), which are not in any repository.

## 1. Figures

All twelve figures are drawn from saved summaries. No sampling, training or Bayesian computation happens in the figure scripts. Each script checks the hashes of its inputs and writes a receipt.

### Figure 1 (fig:bayes-summary)

- Script: `code/scripts/section3_compact_figure.py` -> `results/transformer/section3_compact/overview.pdf`; copied to `figures/` in the paper repository.
- Inputs: 30 files (16 disk, 14 git).

| status | file | size |
|---|---|---|
| git | `results/transformer/larger_models_2026-09-23/manifest.json.gz` | 0.0 MB |
| git | `results/transformer/larger_models_2026-09-23/curves.npz` | 0.0 MB |
| git | `results/transformer/larger_models_2026-09-23/provenance.json.gz` | 0.0 MB |
| disk | `results/transformer/conditional_accuracy/coins/summary.json` | 0.0 MB |
| disk | `results/transformer/conditional_accuracy/coins/receipt.json` | 0.0 MB |
| disk | `results/transformer/conditional_accuracy/product/summary.json` | 0.0 MB |
| disk | `results/transformer/conditional_accuracy/product/receipt.json` | 0.0 MB |
| disk | `results/transformer/hidden_factor_campaign_2026-09-19/comparison.json` | 0.4 MB |
| disk | `results/transformer/hidden_factor_campaign_2026-09-19/validation.json` | 0.0 MB |
| git | `results/transformer/shared_base_conditional_curves/summary.json` | 0.0 MB |
| disk | `results/transformer/shared_base_conditional_curves/validation.json` | 0.0 MB |
| git | `results/transformer/shared_base_conditional_curves/time4_summary.json` | 0.0 MB |
| git | `results/transformer/shared_base_conditional_curves/shortprefix_reference.json` | 0.0 MB |
| git | `results/transformer/shared_base_conditional_curves/time16_summary.json` | 0.3 MB |
| disk | `results/transformer/shared_base_time64_complete_2026-09-17/summary.json` | 0.5 MB |
| disk | `results/transformer/shared_base_time64_complete_2026-09-17/validation.json` | 3.2 MB |
| disk | `results/transformer/shared_base_conditional_curves/figures_t64_complete/receipt.json` | 0.9 MB |
| disk | `results/transformer/shared_base_campaign_2026-09-19/comparison.json` | 0.3 MB |
| disk | `results/transformer/shared_base_campaign_2026-09-19/validation.json` | 0.0 MB |
| git | `results/transformer/shared_base_full_cohort_2026-09-23/reports.json.gz` | 0.1 MB |
| disk | `results/transformer/conditional_accuracy/uniform_base/summary.json` | 0.0 MB |
| disk | `results/transformer/conditional_accuracy/uniform_base/receipt.json` | 0.0 MB |
| disk | `results/transformer/multifamily_bayes_2026-09-16/next_token_regret.json` | 0.0 MB |
| disk | `results/transformer/multifamily_bayes_2026-09-16/next_token_validation.json` | 0.0 MB |
| git | `results/transformer/pooled_estimates_2026-09-24/points.json.gz` | 0.0 MB |

Code and configuration inputs (all in git): `code/scripts/release_large_transformer_results.py`, `code/transformer/configs/hidden_factor_campaign_plan_2026-09-18.json`, `code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json`, `code/transformer/configs/shared_base_campaign_plan_2026-09-18.json`, `code/transformer/configs/shared_base_posterior_full_plan_2026-09-17.json`

### Figure 2 (fig:applications)

- Script: `code/scripts/section4_compact_figure.py` -> `results/transformer/section4_compact/overview.pdf`; copied to `figures/` in the paper repository.
- Inputs: 30 files (8 disk, 14 git, 8 output).

| status | file | size |
|---|---|---|
| output | `output/product_shift_2026-09-18/summary.json` | 0.3 MB |
| output | `output/product_shift_2026-09-18/per_sequence_risks.npz` | 5.9 MB |
| git | `results/transformer/product_mismatch_larger_2026-09-23/curves.npz` | 0.1 MB |
| git | `results/transformer/product_mismatch_larger_2026-09-23/provenance.json.gz` | 0.1 MB |
| output | `output/product_shift_2026-09-18/mismatch_limit/projection_summary.json` | 0.7 MB |
| output | `output/product_shift_2026-09-18/additivity/summary.json` | 0.0 MB |
| disk | `results/transformer/section4_unseen_context_refined_2026-09-20/summary.json` | 3.4 MB |
| disk | `results/transformer/section4_unseen_context_refined_2026-09-20/records.csv` | 0.4 MB |
| disk | `results/transformer/section4_unseen_context_refined_2026-09-20/validation.json` | 0.0 MB |
| disk | `results/transformer/section4_unseen_uniform_control_2026-09-17/summary.json` | 0.4 MB |
| disk | `results/transformer/section4_unseen_uniform_control_2026-09-17/records.csv` | 0.0 MB |
| disk | `results/transformer/section4_unseen_uniform_control_2026-09-17/validation.json` | 0.0 MB |
| disk | `results/transformer/shared_base_shuffle_2026-09-18/summary.json` | 2.3 MB |
| disk | `results/transformer/shared_base_shuffle_2026-09-18/validation.json` | 0.0 MB |
| output | `output/shared_base_shuffle_download_2026-09-18/eval/ordered_seed0/a1_L2.npz` | 5.9 MB |
| output | `output/shared_base_shuffle_download_2026-09-18/eval/shuffled_seed0/a1_L2.npz` | 5.9 MB |
| output | `output/shared_base_shuffle_download_2026-09-18/eval/ordered_seed0/a1_L3.npz` | 5.9 MB |
| output | `output/shared_base_shuffle_download_2026-09-18/eval/shuffled_seed0/a1_L3.npz` | 5.9 MB |
| git | `results/transformer/section4_compact/full_copying_curves.npz` | 0.2 MB |
| git | `results/transformer/section4_compact/full_copying_provenance.json` | 0.0 MB |
| git | `results/transformer/section4_compact/independent_validation.json` | 0.0 MB |
| git | `results/transformer/copying_bayes_precision_2026-09-24/paired_scores.npz` | 0.6 MB |
| git | `results/transformer/copying_bayes_precision_2026-09-24/provenance.json.gz` | 0.1 MB |
| git | `results/transformer/copying_bayes_endpoint32_2026-09-24/paired_scores.npz` | 0.0 MB |
| git | `results/transformer/copying_bayes_endpoint32_2026-09-24/provenance.json.gz` | 0.0 MB |
| git | `results/transformer/copying_bayes_endpoint64_2026-09-24/paired_scores.npz` | 0.0 MB |
| git | `results/transformer/copying_bayes_endpoint64_2026-09-24/provenance.json.gz` | 0.0 MB |
| git | `results/transformer/copying_bayes_endpoint128_provisional_2026-09-25/paired_scores.npz` | 0.0 MB |
| git | `results/transformer/copying_bayes_endpoint128_provisional_2026-09-25/provenance.json.gz` | 0.4 MB |
| git | `results/transformer/copying_bayes_endpoint128_provisional_2026-09-25/original_paired_scores.npz` | 0.0 MB |

### Figures 3, 4, 6, 7 (product gaps; hidden-factor scaling; predictive divergence; shared-base scaling)

- Script: `code/scripts/appendix_source_figures.py` -> `results/transformer/appendix_style_2026-09-24/figure{3,4,6,7}.pdf`; copied to `figures/` in the paper repository.
- Inputs: 20 files (11 disk, 9 git).

| status | file | size |
|---|---|---|
| disk | `results/transformer/conditional_accuracy/product/summary.json` | 0.0 MB |
| disk | `results/transformer/conditional_accuracy/product/validation.json` | 0.0 MB |
| disk | `results/transformer/conditional_accuracy/product/receipt.json` | 0.0 MB |
| disk | `results/transformer/hidden_factor_campaign_2026-09-19/comparison.json` | 0.4 MB |
| disk | `results/transformer/hidden_factor_campaign_2026-09-19/validation.json` | 0.0 MB |
| disk | `results/transformer/shared_depth/summary.json` | 0.0 MB |
| disk | `results/transformer/shared_base_campaign_2026-09-19/comparison.json` | 0.3 MB |
| disk | `results/transformer/shared_base_campaign_2026-09-19/validation.json` | 0.0 MB |
| disk | `results/transformer/shared_base_full_cohort_2026-09-23/summary.json` | 0.2 MB |
| disk | `results/transformer/shared_base_full_cohort_2026-09-23/validation.json` | 0.2 MB |
| disk | `results/transformer/shared_base_full_cohort_2026-09-23/paired_scores.npz` | 0.4 MB |

Code and configuration inputs (all in git): `code/scripts/appendix_source_figures.py`, `code/scripts/appendix_figure_style.py`, `code/scripts/hidden_factor_campaign_results.py`, `code/scripts/shared_base_campaign_results.py`, `code/scripts/bayes_evidence_figure.py`, `code/transformer/configs/hidden_factor_campaign_plan_2026-09-18.json`, `code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json`, `code/transformer/configs/shared_base_campaign_plan_2026-09-18.json`, `code/transformer/configs/shared_base_posterior_full_plan_2026-09-17.json`

### Figures 5, 8 (known-structure scaling; pooled runs)

- Script: `code/scripts/appendix_specialist_pooled_figures.py` -> `results/transformer/appendix_style_2026-09-24/figure{5,8}.pdf`; copied to `figures/` in the paper repository.
- Inputs: 1 files (1 git).

| status | file | size |
|---|---|---|
| git | `results/transformer/section3_compact/receipt.json` | 0.1 MB |

### Figures 9, 10, 11 (product shifts at depth one; additivity; coin shifts)

- Script: `code/scripts/appendix_application_figures.py` -> `results/transformer/appendix_style_2026-09-24/figure{9,10,11}.pdf`; copied to `figures/` in the paper repository.
- Inputs: 4 files (4 output).

| status | file | size |
|---|---|---|
| output | `output/product_shift_2026-09-18/summary.json` | 0.3 MB |
| output | `output/product_shift_2026-09-18/additivity/summary.json` | 0.0 MB |
| output | `output/product_shift_2026-09-18/mismatch_limit/projection_summary.json` | 0.7 MB |
| output | `output/coin_shift_2026-09-18/summary.json` | 0.7 MB |

### Figure 12 (fig:shared-shuffle-early)

- Script: `code/scripts/appendix_copying_figure.py` -> `results/transformer/appendix_style_2026-09-24/figure12.pdf`; copied to `figures/` in the paper repository.
- Inputs: 7 files (7 git).

| status | file | size |
|---|---|---|
| git | `results/transformer/section4_compact/full_copying_curves.npz` | 0.2 MB |
| git | `results/transformer/section4_compact/full_copying_provenance.json` | 0.0 MB |
| git | `results/transformer/section4_compact/independent_validation.json` | 0.0 MB |
| git | `results/transformer/appendix_style_2026-09-24/copying_run2_early16/curves.npz` | 0.0 MB |
| git | `results/transformer/appendix_style_2026-09-24/copying_run2_early16/provenance.json.gz` | 0.0 MB |
| git | `results/transformer/copying_bayes_precision_2026-09-24/paired_scores.npz` | 0.6 MB |
| git | `results/transformer/copying_bayes_precision_2026-09-24/provenance.json.gz` | 0.1 MB |

## 2. Tables

Two tables are generated files; the others were typed from the saved summaries. "Confirmed" means the table's numbers were found in the named file on 13 October 2026.

| table | label | source | status |
|---|---|---|---|
| 1 | `tab:pred` | compiled from the values of Sections 3 and 4 (Figure 1 and Figure 2 receipts) | git |
| 2 | `tab:sources-at-a-glance` | prose, no data | - |
| 3 | `tab:training-settings` | training configurations: `code/transformer/configs/*plan*.json`, `results/transformer/runs/*/config.json` | git |
| 4 | `tab:conditional-cohorts` | cohort sizes recorded in `results/transformer/section3_compact/receipt.json` | disk |
| 5 | `tab:context-requirements` | `results/transformer/product_context_confirmation_2026-09-17/summary.json` (confirmed 2/2) | disk |
| 6 | `tab:hidden-factor-scaling` | `results/transformer/hidden_factor_campaign_2026-09-19/comparison.json` (confirmed 44/52; the rest are rounded differently) | disk |
| 7 | `tab:shared-base-scaling` | `results/transformer/shared_base_campaign_2026-09-19/comparison.json` (confirmed 50/54) | disk |
| 8 | `tab:shared-base-long-context` | `results/transformer/shared_base_full_cohort_2026-09-23/summary.json` and `shared_base_long_context_2026-09-17/summary.json` (to confirm) | disk |
| 9 | `tab:pooled-bayes-loss` | panel f of `results/transformer/section3_compact/receipt.json` (confirmed 12/13); underlying `results/transformer/pooled_estimates_2026-09-24/*.gz` | disk / git |
| 10 | `tab:product-mismatch-larger` | `results/transformer/product_mismatch_larger_2026-09-23/provenance.json.gz` (confirmed 20/22) | git |
| 11 | `tab:product-shift-endpoints` | generated: `results/transformer/product_shift_2026-09-18/endpoint_table.tex` (confirmed 38/38) | git |
| 12 | `tab:coin-shift-endpoints` | `output/coin_shift_2026-09-18/summary.json` (confirmed 3/3) | output |
| 13 | `tab:application-unseen-paired` | generated: `results/transformer/section4_unseen_context_refined_2026-09-20/paired_table.tex` (confirmed 55/55) | git |
| 14 | `tab:shared-shuffle-endpoints` | `results/transformer/shared_base_shuffle_2026-09-18/summary.json` (confirmed 12/12) | disk |
| 15 | `tab:shared-shuffle-bayes-sixteen` | same summary (confirmed 35/36) plus `results/transformer/copying_bayes_precision_2026-09-24/provenance.json.gz` | disk / git |
| 16 | `tab:gap-summary` | compiled from the Figure 1 receipt (confirmed 24/31) and the Figure 2 inputs | disk |

## 3. Behind the summaries: the experiments

The summaries above come from campaign scripts (`code/scripts/*_results.py`, `*_extension.py`, `*_predictions.py`), each with a command, a smoke variant, a unit test and a validation statement in `results_manifest.json`. Their inputs are the raw experiment outputs:

| experiment | trained models | raw evaluation outputs | Bayesian references | campaign script |
|---|---|---|---|---|
| Coins (Fig. 1a, Fig. 11, Table 12) | `results/transformer/runs/bin_twopoint`, `bin_uniform` (git: config, log, weights) | `results/transformer/eval/bin_*` (disk), `output/coin_shift_2026-09-18` (output, 140 MB) | exact count formulas, no store | `conditional_coin_accuracy.py`, `coin_shift_predictions.py` |
| Product, known encoding (Fig. 1b, 3, 5; Tables 5, 10, 11; Fig. 2a-c, 9, 10) | `runs/prod_8x8` and the four-layer runs (git: weights) | `output/conditional_accuracy/product`, `output/product_shift_2026-09-18` (output, 301 MB), Kuma downloads (output) | ext: shared-depth LSA references | `conditional_product_accuracy.py`, `product_shift_*.py`, `mismatch_limits.py`, `product_mismatch_larger.py` |
| Product, hidden encoding (Fig. 1c, 4, 6; Table 6) | Kuma campaign 2026-09-19 (plans in `code/transformer/configs`) | `output/kuma_results_2026-09-20/hidden_factor_campaign*` (output) | exact marginalization over encodings (no store) | `hidden_factor_campaign_results.py` (+ test) |
| Uniform base (Fig. 1d, 5; Fig. 2f) | `runs/lsa_a0_d64` (git: weights) | `output/copier_time/run_2026-09-15`, `output/conditional_uniform_*` (output) | ext | `conditional_uniform_accuracy.py`, `section4_unseen_uniform_control.py` |
| Shared base (Fig. 1e, 7; Tables 7, 8, 13; Fig. 2d-e) | Kuma campaign 2026-09-18 + eight-layer models (plans in configs; weights partly git) | `output/kuma_results_2026-09-20/shared_base_campaign*`, `output/shared_base_posterior_full_download_2026-09-19` (output) | ext: MCMC over ranks and exponent, days of CPU; results in `results/transformer/shared_base_*` (disk) | `shared_base_campaign_results.py`, `shared_base_full_cohort_results.py`, `section4_unseen_context_extension.py` (+ tests) |
| Pooled families (Fig. 1f, 8; Table 9) | Kuma multifamily runs (configs) | `output/multifamily*`, `output/pooled*` (output) | ext: mixture of the three family likelihoods | `pooled_mean_*.py`, `release_large_transformer_results.py` |
| Copying, ordered vs shuffled (Fig. 2g-h, 12; Tables 14, 15) | four-layer ordered and shuffled models (Kuma) | `output/shared_base_shuffle_download_2026-09-18/eval` (output, 87 MB), `output/shared_base_shuffle_extension_recovery_2026-09-22` (95 MB) | ext: `copying_bayes_*` references (git: paired scores and provenance) | `extend_shared_base_shuffle_large.py`, `copying_bayes_*.py` (+ tests) |

## 4. What this means for a reader of the repository

- Today a clone cannot draw a single figure: of the 64 data files the figure scripts read, 26 are in git, 29 are ignored summaries under `results/` and 9 are under `output/`.
- Committing those files (about 50 MB) makes all twelve figures and the generated tables reproducible from a clone in minutes, with no external store.
- Recomputing the summaries from the raw evaluations needs the `output/` folders named above (a few hundred MB per experiment) and, for the Bayesian references, the sealed LSA store (2.8 GB) and the `lsa` package. These belong on Zenodo (or a release asset), not in git.
- Retraining needs a GPU and the Kuma launchers under `code/transformer/kuma`; the configurations are in git.
