#!/bin/bash
# The thirty settings of Section 3 (as run on 2026-09-11). Requires the tables of run_all.sh / run_archive.sh (step 0),
# the eighteen archived settings prepared by run_archive.sh (steps 1-6: sim_A_*, bb2_*, tab_B_*), and the extension
# archives output/crossing_lsa_extension (+ refined256) and output/crossing_lsa_alpha2. Costs in bits.
set -e
E=../../../output/crossing_lsa_extension; A=../../../output/crossing_lsa_alpha2
# 1. extension archives -> simulation format (control variates from the stored matrices; codelengths as stored)
for t in L6_a1_d256 L6_a1.5_d256 L10_a1.5_d256 L6_a0.5_d256 L6_a0.5_d512; do python3 ext_extract.py $E $t .; done
for t in L6_a1_d512 L6_a1.5_d512 L10_a1.5_d512; do python3 ext_extract.py $E/refined256 $t .; done
for t in L3_a2_d256 L6_a2_d256 L3_a2_d512 L6_a2_d512; do python3 ext_extract.py $A $t .; done
# 2. building blocks of the twelve extension settings: rate window from the rate law, formula grid, exact k <= 32, merge
./run_ext_grids.sh                       # ext_grid.py d alpha Ls Lmax kmax, two workers
# 3. extra paths: 240 per setting at alpha <= 1, d <= 256 (archived 16), 224 at d = 256 for the extension settings with a
#    crossing (archived 32), 96 at d = 512 for alpha = 2, and 176 more at (256, 1.25, 2) and (256, 1.25, 3) (archived 16 + 64)
./run_extra_paths.sh; ./run_extra_paths2.sh
for t in L2_a0.75_d128 L2_a1_d128 L3_a0.75_d128 L3_a1_d128 L2_a0.75_d256 L2_a1_d256 L3_a0.75_d256 L3_a1_d256 \
         L2_a1.25_d256 L3_a1.25_d256 L6_a1_d256 L6_a1.5_d256 L10_a1.5_d256 L3_a2_d256 L6_a2_d256 L3_a2_d512 L6_a2_d512; do python3 merge_ext.py $t; done
# 4. predictions at the lengths used (20 unigram / 600 copier tables, seed 6d), then more sampled matrices for the
#    extension settings (180 at d = 256, 60 at d = 512, seed 7d), merged into tab_M_* by assemble.sh
./run_ext_tables.sh; ./run_ext_tables2.sh
# 5. comparison of all thirty settings (raw paired gap; both standard errors), tables, figures, statistics, export
./assemble.sh
python3 make_table3.py; python3 make_regret_table3.py; python3 make_figure3.py; python3 make_gap_figures3.py; python3 stats3.py
python3 export_results.py
# 6. diagnostics used in the text: per-context residuals at (512, 1.5, 6) on 32 extra paths and on 64 archived paths
#    (diag_ctx.py, diag_ctx_arch.py) and the visit-count law (diag_klaw2.py)
