#!/bin/bash
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/transformer/runs/%x-%j.out
module load gcc python 2>/dev/null || true
source ~/venv-tf/bin/activate
cd "$SLURM_SUBMIT_DIR"
bash code/transformer/kuma/eval_families.sh
