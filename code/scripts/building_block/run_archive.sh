#!/bin/bash
# Comparison with the archived chains of output/crossing_finite_count_prediction (as run on 2026-09-10; the
# actual run was split over several driver scripts, this file records the settings). All eighteen settings
# (d, alpha, L_s) in {128, 256, 512} x {0.75, 1, 1.25} x {2, 3}. Requires the saddle-point tables of run_all.sh
# and the store-derived tables (step 0). Costs in bits.
set -e
ARCH=../../../output/crossing_finite_count_prediction     # archived runs: <tag>.npz + <tag>.json
STORE=../../../../product_model_with_memory                # anchor store (tables/anchors_prod) and its reader
# 0. large counts (2001..6000, u in [-10, 40]) for every depth and the full tables for depths 27..30 from the
#    anchor store, through the store's own interpolation (read-only; about 5 GB, not committed). The blocks
#    r0_2000_full for depths 1..26 are the saddle-point tables of run_all.sh, not the store's.
python3 store_rows.py 1 30 store_tables $STORE/tables/anchors_prod $STORE/src
cp store_tables/phitab_L*_r2001_6000_u-10_40.npy .; for L in 27 28 29 30; do cp store_tables/phitab_L${L}_r0_2000_full.npy .; done
# 1. grids of B(k) up to the largest visit count of the archive's lengths used (xhalf: width of the scale window)
grid () { python3 -c "import bb, lsa_fast; bb.bb_grid($1, $2, $3, $4, kgrid=[k for k in bb.KGRID if k <= $5], out='$6', xhalf=$7)"; python3 patch_small.py $6 24; }
#     d    alpha Ls Lmax kmax xhalf
grid 128 0.75 2 23  640  1.6 bb_d128_a0.75_L2.json;  grid 128 1.0 2 23  960  1.6 bb_d128_a1.0_L2.json;  grid 128 1.25 2 23 1440 2.2 bb_d128_a1.25_L2.json
grid 128 0.75 3 23  320  1.6 bb_d128_a0.75_L3.json;  grid 128 1.0 3 23  640  1.6 bb_d128_a1.0_L3.json;  grid 128 1.25 3 23  960 2.8 bb_d128_a1.25_L3.json
grid 256 0.75 2 26 1440  1.6 bb_d256_a0.75_L2.json;  grid 256 1.0 2 26 2000  1.6 bb_d256_a1.0_L2.json;  grid 256 1.25 2 26 2000 2.2 bb_d256_a1.25_L2.json
grid 256 0.75 3 26  700  1.6 bb_d256_a0.75_L3.json;  grid 256 1.0 3 26  960  1.6 bb_d256_a1.0_L3.json;  grid 256 1.25 3 26  960 2.8 bb_d256_a1.25_L3.json
grid 512 0.75 2 30 3000  1.6 bb_d512_a0.75_L2.json;  grid 512 1.0 2 30 4000  1.6 bb_d512_a1.0_L2.json;  grid 512 1.25 2 30 5000 2.2 bb_d512_a1.25_L2.json
grid 512 0.75 3 30 1120  2.2 bb_d512_a0.75_L3.json;  grid 512 1.0 3 30 2000  2.2 bb_d512_a1.0_L3.json;  grid 512 1.25 3 30 2000 2.8 bb_d512_a1.25_L3.json
# 2. archived paths -> simulation format with control variates; codelengths recomputed at d <= 256 (check)
for tag in L2_a0.75_d128 L2_a1_d128 L2_a1.25_d128 L3_a0.75_d128 L3_a1_d128 L3_a1.25_d128 \
           L2_a0.75_d256 L2_a1_d256 L2_a1.25_d256 L3_a0.75_d256 L3_a1_d256 L3_a1.25_d256; do python3 archive_extract.py $tag; done
for tag in L2_a0.75_d512 L2_a1_d512 L2_a1.25_d512 L3_a0.75_d512 L3_a1_d512 L3_a1.25_d512; do python3 archive_extract.py $tag --no-recompute; done
# 3. extra paths at alpha = 1.25 (seed, R) at the archive's lengths, merged with the archived paths
python3 chain_sim.py 128 1.25 2 240 41 322,644,966,1127,1288,1449,1610,1932,2576 sim_X_L2_a1.25_d128.json; python3 merge_archive.py L2_a1.25_d128
python3 chain_sim.py 128 1.25 3 240 42 137,274,411,479,547,616,684,821,1095 sim_X_L3_a1.25_d128.json;    python3 merge_archive.py L3_a1.25_d128
python3 chain_sim.py 256 1.25 3 64 43 352,704,1056,1232,1408,1584,1760,2112,2816 sim_X_L3_a1.25_d256.json; python3 merge_archive.py L3_a1.25_d256
python3 chain_sim.py 256 1.25 2 64 44 1001,2002,3003,3503,4003,4504,5004,6005,8007 sim_X_L2_a1.25_d256.json; python3 merge_archive.py L2_a1.25_d256
# 4. predictions at the archive's lengths (20 unigram tables, 600 copier tables, seed 4d), no paths
tab () { python3 table_mc.py $2 $3 $4 20 600 4$2 $5 $6 tab_A_$1.json; }
tab L2_a0.75_d128 128 0.75 2 454,907,1361,1588,1815,2041,2268,2722,3629 bb_d128_a0.75_L2.json
tab L2_a1_d128 128 1.0 2 418,836,1253,1462,1671,1880,2089,2507,3343 bb_d128_a1.0_L2.json
tab L2_a1.25_d128 128 1.25 2 322,644,966,1127,1288,1449,1610,1932,2576 bb_d128_a1.25_L2.json
tab L3_a0.75_d128 128 0.75 3 130,261,391,457,522,587,652,783,1044 bb_d128_a0.75_L3.json
tab L3_a1_d128 128 1.0 3 142,283,425,495,566,637,708,849,1132 bb_d128_a1.0_L3.json
tab L3_a1.25_d128 128 1.25 3 137,274,411,479,547,616,684,821,1095 bb_d128_a1.25_L3.json
tab L2_a0.75_d256 256 0.75 2 1602,3205,4807,5608,6409,7210,8011,9614,12818 bb_d256_a0.75_L2.json
tab L2_a1_d256 256 1.0 2 1416,2832,4247,4955,5663,6371,7079,8495,11326 bb_d256_a1.0_L2.json
tab L2_a1.25_d256 256 1.25 2 1001,2002,3003,3503,4003,4504,5004,6005,8007 bb_d256_a1.25_L2.json
tab L3_a0.75_d256 256 0.75 3 472,944,1416,1651,1887,2123,2359,2831,3775 bb_d256_a0.75_L3.json
tab L3_a1_d256 256 1.0 3 438,877,1315,1534,1753,1972,2192,2630,3507 bb_d256_a1.0_L3.json
tab L3_a1.25_d256 256 1.25 3 352,704,1056,1232,1408,1584,1760,2112,2816 bb_d256_a1.25_L3.json
tab L3_a0.75_d512 512 0.75 3 1794,3589,5383,6280,7177,8074,8972,10766,14355 bb_d512_a0.75_L3.json
tab L3_a1_d512 512 1.0 3 1511,3022,4533,5288,6044,6799,7555,9065,12087 bb_d512_a1.0_L3.json
tab L3_a1.25_d512 512 1.25 3 908,1816,2725,3179,3633,4087,4541,5449,7266 bb_d512_a1.25_L3.json
tab L2_a0.75_d512 512 0.75 2 5876,11753,17629,20567,23505,26443,29381 bb_d512_a0.75_L2.json
tab L2_a1_d512 512 1.0 2 4391,8781,13172,15367,17562,19757,21953 bb_d512_a1.0_L2.json
tab L2_a1.25_d512 512 1.25 2 3092,6184,9276,10822,12368,13913,15459 bb_d512_a1.25_L2.json
# 5. paired comparison, crossings, tables (tables_out/archive_*.tex, archive_summary.json), export
python3 archive_compare.py; python3 export_results.py
# 6. exact building block for k <= 32 (profile enumeration), merged into the grids (bb2_*), predictions rerun (tab_B_*)
for d in 128 256 512; do for a in 0.75 1.0 1.25; do for L in 2 3; do
  Lmax=$([ $d = 128 ] && echo 23 || ([ $d = 256 ] && echo 26 || echo 30))
  python3 bb_exact_small.py $d $a $L $Lmax 32 bbx_d${d}_a${a}_L${L}.json 0.05
  python3 merge_exact.py bb_d${d}_a${a}_L${L}.json bbx_d${d}_a${a}_L${L}.json bb2_d${d}_a${a}_L${L}.json
done; done; done
#    then step 4 again with bb2_* grids and output files tab_B_*, and
TAB_PREFIX=tab_B OUT_SUFFIX=_B python3 archive_compare.py        # the summary used in the paper (archive_summary_B.json)
# 7. row-known correction for contexts with T(a|a) >= 0.02 (table_rk.py, 200 matrices; tested, not used in the paper):
#    python3 table_rk.py 128 1.25 2 200 5128 322,...,2576 bb2_d128_a1.25_L2.json 0.02 4.016144 tab_RK_L2_a1.25_d128.json
#    TAB_PREFIX=tab_B RK_PREFIX=tab_RK OUT_SUFFIX=_BR python3 archive_compare.py
# 8. diagnostics on the paths: diag_bins.py (residual by self-transition probability and visits), diag_rowknown.py
#    (row known vs row averaged), diag_klaw.py (visit-count law), single_val_a125.py (Monte Carlo of B(k) at small k)
