#!/bin/bash
# Evaluate the family runs on their own test sets and on the sets of the other families with the same alphabet
# (cross-training). Run from the project folder on Kuma after the runs have finished:
#     sbatch --job-name=tf-eval-fam code/transformer/kuma/eval_families.sbatch
# (this file lists the calls; eval_families.sbatch wraps it in a GPU job)
set -e
E=code/transformer/eval_model.py; R=results/transformer/runs; O=results/transformer/eval
BIN="binary:twopoint;binary:uniform;binary:eightpoint;binary:fixed=0.5;binary:fixed=0.2"
for run in bin_twopoint bin_uniform bin_eightpoint; do
  python3 $E --run $R/$run --ckpt latest --settings none --family_sets "$BIN" --n_eval 4096 --n_full 16 --out $O/$run
done
EXP="expfam:k=1;expfam:k=2;expfam:k=1:perm;expfam:k=1:theta=1;expfam:k=2:theta=1,0.1"
for run in expfam_k1 expfam_k2 expfam_k1_perm; do
  python3 $E --run $R/$run --ckpt latest --settings none --family_sets "$EXP" --n_eval 2048 --n_full 32 --out $O/$run
done
# product 8x8 (d = 64) against the d = 64 sets of the paper's source, and the reverse for the d = 64 lsa runs
P8="product:8x8:L=2,2;product:8x8:L=4,4;product:8x8:L=2,6"
python3 $E --run $R/prod_8x8   --ckpt latest --settings "1,2;1,4;0,2;0,4" --family_sets "$P8" --n_eval 2048 --n_full 32 --out $O/prod_8x8
python3 $E --run $R/prod_16x16 --ckpt latest --settings none --family_sets "product:16x16:L=2,2;product:16x16:L=4,4" --n_eval 1024 --n_full 32 --out $O/prod_16x16
# independent rows (alpha = 0) on their own sets and on the alpha = 1 sets of the paper's family
python3 $E --run $R/lsa_a0_d64  --ckpt latest --settings "0,2;0,3;0,4;1,2;1,3;1,4" --family_sets "$P8" --n_eval 2048 --n_full 32 --out $O/lsa_a0_d64
python3 $E --run $R/lsa_a0_d128 --ckpt latest --settings "0,2;0,3;0,4;1,2;1,3;1,4" --n_eval 2048 --n_full 32 --out $O/lsa_a0_d128
# the paper's family (existing runs) on the alpha = 0 sets and on the product sets
python3 $E --run $R/d64_seed1 --ckpt 100000 --settings "0,2;0,3;0,4" --family_sets "$P8" --n_eval 2048 --n_full 32 --out $O/d64_seed1_cross
python3 $E --run $R/d128      --ckpt 100000 --settings "0,2;0,3;0,4" --n_eval 2048 --n_full 32 --out $O/d128_cross
