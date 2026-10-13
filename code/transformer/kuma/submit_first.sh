#!/bin/bash
# The first batch of runs on Kuma: the main setup at d = 32 and 128, and at d = 64 two controls (a fixed source, and no
# relabelling) plus a second seed. Run from the project folder:   bash code/transformer/kuma/submit_first.sh
set -e
mkdir -p results/transformer/runs
S=code/transformer/kuma/train.sbatch
sbatch --job-name=tf-d32        $S --d 32  --out results/transformer/runs/d32        --steps 100000
sbatch --job-name=tf-d128       $S --d 128 --out results/transformer/runs/d128       --steps 100000
sbatch --job-name=tf-d64-fixed  $S --d 64  --out results/transformer/runs/d64_fixed  --steps 100000 --alpha 1 1 --ls 2 2
sbatch --job-name=tf-d64-noperm $S --d 64  --out results/transformer/runs/d64_noperm --steps 100000 --no_permute
sbatch --job-name=tf-d64-seed1  $S --d 64  --out results/transformer/runs/d64_seed1  --steps 100000 --seed 1
squeue -u $USER
