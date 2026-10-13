#!/bin/bash
# The first run: d = 64, the main setup (alpha and L_s per sequence, relabelling), 2 layers, 4 heads, width 128,
# batches of 64 sequences of 1024 tokens, 100 000 steps, a checkpoint every 5 000 steps. Then the evaluation of the
# last checkpoint on the grid of settings. Run from the repository root:  bash code/transformer/run_first.sh
set -e
cd "$(dirname "$0")/../.."
mkdir -p results/transformer/runs results/transformer/eval
python3 code/transformer/train.py --d 64 --out results/transformer/runs/d64 --steps 100000 --ckpt_every 5000 --workers 2 --resume 2>&1 | tee -a results/transformer/runs/d64.log
python3 code/transformer/eval_model.py --run results/transformer/runs/d64 --ckpt latest --settings grid --out results/transformer/eval/d64 2>&1 | tee -a results/transformer/eval/d64.log
