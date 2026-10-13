#!/bin/bash
# Full pipeline for the appendix "One building block for both mechanisms" (as run on 2026-09-09; the
# actual run was split over several driver scripts, this file records the settings).
# Requires the tables phitab_L{L}_r*_full.npy (see README).
set -e
# 1. grids of B(k)
python3 bb.py 128 0.75 2 23 320;  python3 bb.py 128 1.0 2 23 450;  python3 patch_bb.py bb_d128_a1.0_L2.json 640
python3 bb.py 128 0.75 3 23 320;  python3 bb.py 128 1.0 3 23 450;  python3 patch_bb.py bb_d128_a1.0_L3.json 640
python3 bb.py 256 0.75 2 26 700;  python3 patch_bb.py bb_d256_a0.75_L2.json 960
python3 bb.py 256 1.0 2 26 700;   python3 patch_bb.py bb_d256_a1.0_L2.json 1440
python3 bb.py 256 0.75 3 26 512;  python3 bb.py 256 1.0 3 26 512
for f in bb_d*_a*_L*.json; do python3 patch_small.py $f 24; done
# 2. chain simulations (seed, R) and table Monte Carlo (NT_pred NT_occ seed); merged extra paths in brackets
run_pair () {   # S d alpha Ls seed checkpoints bbfile R
  if [ $2 = 128 ]; then NO=1500; NP=40; else NO=600; NP=20; fi
  python3 chain_sim.py $2 $3 $4 $8 $5 $6 sim_$1.json
  python3 table_mc.py $2 $3 $4 $NP $NO 10$5 $6 $7 tab_$1.json
}
run_pair S1 128 0.75 2 11 700,1100,1450,1800,2250,2900 bb_d128_a0.75_L2.json 80   # + 160 paths, seed 21
run_pair S2 128 1.0 2 12 650,1000,1350,1700,2100,2700 bb_d128_a1.0_L2.json 80     # + 160 paths, seed 22
run_pair S3 128 0.75 3 13 180,280,380,480,600,780 bb_d128_a0.75_L3.json 80        # + 160 paths, seed 23
run_pair S4 128 1.0 3 14 200,320,420,530,660,850 bb_d128_a1.0_L3.json 80          # + 160 paths, seed 24
run_pair S5 256 0.75 2 15 2500,3900,5100,6400,7300,8200 bb_d256_a0.75_L2.json 32  # + 32 paths, seed 25
run_pair S6 256 1.0 2 16 2300,3500,4700,5800,6800,7800 bb_d256_a1.0_L2.json 32    # + 32 paths, seed 26
run_pair S7 256 0.75 3 17 800,1200,1600,1950,2400,3100 bb_d256_a0.75_L3.json 32
run_pair S8 256 1.0 3 18 750,1150,1500,1900,2350,3000 bb_d256_a1.0_L3.json 32
# e.g. python3 chain_sim.py 128 0.75 2 160 21 700,1100,1450,1800,2250,2900 sim_S1b.json; python3 merge_sims.py sim_S1.json sim_S1a.json sim_S1b.json
# 3. single-row truths, tables, figure, export
python3 single_val2.py 4000 single_truth3.json 3.0; python3 single_val2.py 4000 single_truth4.json 3.0 5000
python3 make_tables.py; python3 make_figure.py; python3 export_results.py
