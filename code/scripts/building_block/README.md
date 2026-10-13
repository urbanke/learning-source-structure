# Building-block calculation (appendix "Derivation and numerical details of the crossing formula")

Independent numerical implementation of the LSA code and of the
building-block approximation. The moment functions
phi_{L,r}(s) = E[W_L^r e^{-s W_L}] are computed from the density of ln W_L
by saddle-point Mellin inversion (lsa_phi.py) and tabulated on a grid of
u = ln s in [-45, 40] (step 1/64) for depths L <= 26 and counts r <= 2000;
everything else is evaluated from these tables. Two things come from the
anchor store of the production evaluator (product_model_with_memory,
read-only, through its own interpolation): the counts 2001..6000 at every
depth (u in [-10, 40]) and the depths 27..30 needed at d = 512
(store_rows.py). Where the two sources overlap they agree to 1e-8 nats.

Files

- lsa_phi.py          density of ln W_L; base tables of ln phi_{L,r}(e^u)
- build_tables2.py    table blocks r = 0..200, 201..700, 701..1000 on u in [-45, 15]
- build_tables_u.py   extension of those blocks to u in (15, 40]
- merge_tables.py     merges blocks and extension into phitab_L{L}_r0_1000_full.npy
- build_tables_r.py   large-count blocks (r = 1001..1400, 1401..2000) on the full u grid
- store_rows.py       table blocks from the anchor store (r = 0..2000 full, r = 2001..6000
                      on u in [-10, 40]); depth 1 in closed form
- lsa_fast.py         spline-free table evaluator (6-point Lagrange stencil on the u grid,
                      memory-mapped blocks); replaces the loader of bb.py when imported
- uspline.py          memory-light cubic spline on a uniform grid (old evaluator)
- lsa_cost.py         exact LSA cost of a count profile (u integral), Laplace form, profiles
- lsa_cost2.py        gradient / Hessian of the profile cost ((L,u)-measure), control variate
- source_profiles.py  mean profiles of tilted rows (scale integral)
- bb.py               table loader (LSAx), the building block B(k), grids over k
- patch_bb.py         extends a grid file to larger k; patch_small.py recomputes k <= 24
                      with a wide scale window
- single_val2.py      single-row Monte Carlo truths with the exact-centred control variate
- exact_cov_test.py   fluctuation term with the exact pairwise profile covariance (small k)
- info_gain.py        exact mean row entropy E[H(P)] (one-dimensional integrals); its quadrature weights
                      (np.gradient on a log grid) are 0.027 percent too large, see fix_EH.py below
- chain_sim.py        chain simulations with exact costs of both mechanisms + control variates
- table_mc.py         predictions of both mechanisms from sampled matrices (no paths)
- merge_sims.py       merges simulation files with different seeds
- compare.py          paired comparison (control variates), crossings
- diag_copier.py      per-context residuals at the actual visit counts
- archive_extract.py  archived chain runs (output/crossing_finite_count_prediction) ->
                      simulation format with control variates; recomputes the codelengths
- merge_archive.py    merges archived paths with extra paths of chain_sim.py
- archive_compare.py  paired comparison on the archived chains, crossings, tables
                      (TAB_PREFIX selects the prediction files, RK_PREFIX adds a row-known correction)
- bb_exact_small.py   exact B(k) for k <= K0 by the exact law of the count profile (dynamic
                      programme over the symbols; partitions of the integers up to K0)
- merge_exact.py      merges the exact small-k values into a grid file (bb2_*)
- table_rk.py         row-known correction for contexts with a large self-transition probability
                      (tested; not used in the paper)
- diag_bins.py, diag_rowknown.py, diag_klaw.py, single_val_a125.py   diagnostics of the copier error
- diag_runaware.py    heavy contexts: row averaged / row known / runs known, on the paths (results/building_block/diag_runaware_*.json)
- mc_rowknown_check.py  Monte Carlo check of the row-known Laplace prediction
- ext_extract.py      extension runs (output/crossing_lsa_extension, .../refined256, output/crossing_lsa_alpha2) ->
                      simulation format (runs next to the archives; numpy only)
- ext_grid.py         building block of an extension setting: rate window from the rate law (bb.rate_window), formula
                      grid, exact values for k <= 32, merged grid bb2_*
- merge_ext.py        merges archived paths with extra paths of chain_sim.py (sim_AX_*, checkpoints matched by value)
- merge_tabs.py       merges prediction files of one setting (more sampled matrices), covariance term re-centred
- compare_all.py      paired comparison for all thirty settings (raw paired gap; unigram regret with its control
                      variate; copier regret without), crossings with both standard errors -> summary_all.json
- make_table3.py, make_figure3.py, make_gap_figures3.py, make_regret_table3.py, stats3.py   tables, figures, statistics
- assemble.sh         final prediction files tab_M_* (tab_B, plus tab_C merged for the extension settings), comparison
- diag_ctx.py, diag_ctx_arch.py, diag_klaw2.py   per-context residuals and the visit-count law on extra / archived paths
- make_main_table.py, make_main_figure.py, make_archive_figure.py   earlier main-text table and figures (18 settings)
- make_tables.py      LaTeX tables and summary.json for the appendix (own simulations)
- make_figure.py      gap-per-token figure
- export_results.py   reduces the raw run files to results/building_block

Reproduction (from this directory). The saddle-point tables take about
three hours on two cores and use 2.5 GB of disk; the store-derived blocks
another 5 GB. None of them is committed.

    python3 build_tables2.py 1 26 0 200; python3 build_tables2.py 1 26 201 700
    python3 build_tables2.py 1 26 701 1000; python3 build_tables_u.py 1 26 1000
    python3 merge_tables.py 1 26
    python3 build_tables_r.py 1 26 1001 1400; python3 build_tables_r.py 1 26 1401 2000
    python3 store_rows.py 1 30 store_tables <anchors_prod> <product_model_with_memory/src>
    python3 bb.py 128 0.75 2 23 320                 # grid of B(k) for one setting
    python3 patch_small.py bb_d128_a0.75_L2.json 24
    python3 chain_sim.py 128 0.75 2 80 11 700,1100,1450,1800,2250,2900 sim_S1.json
    python3 table_mc.py 128 0.75 2 40 1500 1011 700,1100,1450,1800,2250,2900 bb_d128_a0.75_L2.json tab_S1.json
    python3 compare.py sim_S1.json tab_S1.json S1
    python3 single_val2.py 4000 single_truth3.json 3.0     # single-row truths
    python3 archive_extract.py L2_a1_d128                  # archived chains
    python3 table_mc.py 128 1.0 2 20 600 4128 418,836,1253,1462,1671,1880,2089,2507,3343 bb_d128_a1.0_L2.json tab_A_L2_a1_d128.json
    python3 archive_compare.py

The settings S1..S8, seeds and checkpoints of the own simulations are
listed in run_all.sh; the eighteen archived settings, the grids, the extra
paths at alpha = 1.25 and the predictions at the archive's lengths in
run_archive.sh. All costs are in bits. exact_EH.json and
exact_EH_archive.json hold the exact mean row entropies (bits) used to
centre the covariance term (recomputed by fix_EH.py with exact
quadrature weights; the values first computed by info_gain.py were 0.001
to 0.002 bits too high).

The thirty settings of the paper (eighteen archived settings with source
depths 2 and 3, and the extension runs with depths 6 and 10 and alpha = 2)
are run by run_extension.sh: extraction of the archives (on the machine
that holds output/), grids with the adaptive rate window, exact small-k
values, extra paths, predictions, comparison, tables and figures.

## Inputs without sampling (appendix "Computing the inputs without sampling")

The chain inputs (means and variances of the visit and token counts,
and the entropy covariance) are computed from one random row instead of
sampled matrices: the two-state reduction of the visits to a context
(stay probability = the row's own entry, entry probability from the
balance of the flow), with the law of the own entry from a Fourier
inversion and the flow from the other rows as a shifted gamma with three
exact cumulants. All inputs are integrals of products of the moment
functions phi_r(t) = E[W^r e^{-tW}], r <= 3, and psi(t) = E[W ln W e^{-tW}]
of the source depth.

- onerow.py           the one-row integrals: mean row, E[(x_a/(1-x_c))^j] for j <= 3 (gamma trick with tilted
                      moments), the law of one entry (tilted Fourier inversion, c = 0.3), E[H(P)] and E[H(P)/(1-x_a)]
- sf_predict.py       the sampling-free prediction: components (own entry x own-row-independent flow), two-state
                      moments, copier = log2 d + sum_a E B(K_a) + covariance term, unigram = Laplace cost of the mixed
                      profile - fluctuation term, crossing; table rows beyond r = 6000 by Laplace's method in ln W
                      (LSAext) -> sfp_<tag>.json
- run_sf_all.sh       runs sf_predict.py for the settings of sf_settings.txt (tag d alpha Ls bbfile checkpoints)
- sf_compare.py       sampling-free crossings against the sampled-matrix prediction (tab_M_*, exact Hbar corrected)
                      and the simulation -> sf_compare.json; make_sf_table.py -> sampling_free_table.tex
- fix_EH.py           exact E[H(P)] for all settings with exact log-grid weights (the original info_gain.exact_EHP used
                      np.gradient weights, 0.027 percent too large: E[H(P)] was high by 0.001 to 0.002 bits)
- sf_lib.py, sf_two.py            the same theory with the one-row expectations taken from 10^5 sampled rows
                                  (used to develop and check the integral forms)
- sf_inputs.py, sf_sampled_big.py, sf_means.py, sf_means2.py   per-matrix inputs and the sampled-matrix prediction
                                  with many matrices (6000 at d = 128, 4000 at d = 256; 2 10^5 matrices for the mean
                                  counts), the precise references of the checks
- sf_twostate_givenT.py           the two-state moments from the true (P_a, pi_a) of each sampled matrix against the
                                  exact moments (means to 1e-3, variances to 1 percent; unigram cost to 0.02 bits)
- sf_diag4.py                     the law of the flow term on rare contexts: true (row-sampled) law, gamma with three
                                  cumulants, Gaussian
- sf_numbers.py, sf_check_laplace_rows.py   the numbers quoted in the appendix (from sf_compare.json); the Laplace rows
                                  beyond the tables against the table rows at the same count (7e-5 nats at r = 6000)

    python3 sf_predict.py 128 1.0 2 bb2_d128_a1.0_L2.json 418,836,1253,1462,1671,1880,2089,2507,3343 sfp_L2_a1_d128.json
    ./run_sf_all.sh '^L[23]_a(0.75|1|1.25)_d(128|256)$' A     # all settings with source depths 2 and 3 at d <= 256
    python3 sf_compare.py; python3 make_sf_table.py

## Scaling laws (Section 3.2, Appendix D.4)

- sf_cross.py         sampling-free crossing of one setting without a prior estimate: coarse geometric sweep of the gap
                      (ratio 2 from n = 16) until it is positive twice in a row, then a refined bracket -> sfx_<tag>.json
- run_scaling.sh, scaling_settings.txt   the small-alphabet grid (building block by ext_grid.py if missing, then sf_cross.py)
- sf_gapcurve.py      the predicted gap at given lengths (used to look at d = 32, where no crossing exists)
- scaling_laws.py     power-law fits of n* in d from results/takeover_all.json (the scaling campaign: crossing of the
                      seed-averaged gap, 16 paths), with the predictions overlaid -> scaling_exponents.json/.tex,
                      figures scaling_d.pdf and scaling_L.pdf
- sim_small_d_gaps.json (results)   mean gaps of 300-path simulations at (32, 0.75, 2), (64, 0.75, 2), (64, 1, 2)
                      (chain_sim.py), the checks of the small-alphabet predictions

    python3 sf_cross.py 64 1.0 2 bb2_d64_a1_L2.json sfx_L2_a1_d64.json
    ./run_scaling.sh 0 2 & ./run_scaling.sh 1 2 &     # two workers over scaling_settings.txt
    python3 scaling_laws.py ../../../results/takeover_all.json ../../../results/building_block/summary_all.json

Results (results/building_block): bb_d*_a*_L*.json grids of B(k) (formula),
bbx_*.json exact values for k <= 32, bb2_*.json merged grids (used);
chains_<tag>.json per-path codelengths and control variates (archived paths
plus extra paths) and the per-table predictions at the lengths used, for
all thirty settings; summary_all.json regrets, gaps and crossings per
setting and stats_all.json the regret statistics (the numbers of the paper);
archive_recomputation_check.json recomputed minus stored codelengths;
archive_summary*.json and rowknown_*.json the earlier comparison of the
eighteen settings with fixed-coefficient control variates, with the formula
only and with the row-known correction (tested, not used);
single_truth_*.json Monte Carlo truths of B(k); sfp_<tag>.json the
sampling-free predictions (both costs at the lengths used, the crossing)
and sf_compare.json their comparison with the sampled-matrix prediction
and the simulation; *.tex the tables of the paper.
