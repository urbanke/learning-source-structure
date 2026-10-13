#!/bin/bash
# F1: the three binary priors, trained one after the other in one GPU job.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/transformer/runs/%x-%j.out
module load gcc python 2>/dev/null || true
source ~/venv-tf/bin/activate
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for prior in twopoint uniform eightpoint; do
  python3 code/transformer/train.py --workers 6 --resume --family binary:$prior --d 2 --n_ctx 256 \
      --n_layer 2 --n_head 4 --d_model 64 --batch 64 --steps 20000 --warmup 500 --val_every 500 --ckpt_every 2500 \
      --out results/transformer/runs/bin_$prior
done
