#!/bin/bash
# The family runs of notes/families_plan.md (first round). Run from the project folder on Kuma:
#     bash code/transformer/kuma/submit_families.sh
# F1 binary (three priors, tiny models, one job), F2 exponential family (d = 32; k = 1, k = 2, k = 1 relabelled),
# F3 product chain (8x8 and 16x16), F4 independent rows (the paper's source at alpha = 0; d = 64 and 128).
set -e
mkdir -p results/transformer/runs
S=code/transformer/kuma/train.sbatch
R=results/transformer/runs
# F1: the three binary priors in one job, sequentially (each run is a minute or two on an H100)
sbatch --job-name=tf-binary --time=02:00:00 code/transformer/kuma/binary.sbatch
# F2: exponential family, d = 32, window 1024
sbatch --job-name=tf-expfam-k1     $S --family expfam:k=1      --d 32 --n_ctx 1024 --steps 50000 --out $R/expfam_k1
sbatch --job-name=tf-expfam-k2     $S --family expfam:k=2      --d 32 --n_ctx 1024 --steps 50000 --out $R/expfam_k2
sbatch --job-name=tf-expfam-k1perm $S --family expfam:k=1:perm --d 32 --n_ctx 1024 --steps 50000 --out $R/expfam_k1_perm
# F3: product chains (uniform base laws, depths 1..8 per coordinate)
sbatch --job-name=tf-prod-8x8   $S --family product:8x8   --d 64  --n_ctx 1024 --steps 100000 --ls 1 8 --out $R/prod_8x8
sbatch --job-name=tf-prod-16x16 $S --family product:16x16 --d 256 --n_ctx 1024 --steps 100000 --ls 1 8 --out $R/prod_16x16
# F4: independent rows: the paper's source with alpha = 0
sbatch --job-name=tf-a0-d64  $S --d 64  --alpha 0 0 --ls 0 8 --steps 100000 --out $R/lsa_a0_d64
sbatch --job-name=tf-a0-d128 $S --d 128 --alpha 0 0 --ls 0 8 --steps 100000 --out $R/lsa_a0_d128
squeue -u $USER
