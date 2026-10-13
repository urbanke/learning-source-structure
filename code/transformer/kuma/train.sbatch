#!/bin/bash
# One training run on one H100:   sbatch --job-name=<name> code/transformer/kuma/train.sbatch <train.py arguments>
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/transformer/runs/%x-%j.out
module load gcc python 2>/dev/null || true
source ~/venv-tf/bin/activate
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python3 code/transformer/train.py --workers 6 --resume "$@"
