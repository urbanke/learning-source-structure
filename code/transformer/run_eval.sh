#!/bin/bash
# Evaluate one checkpoint of a run on the grid:  bash code/transformer/run_eval.sh d64 20000   (or 'latest')
set -e
cd "$(dirname "$0")/../.."
python3 code/transformer/eval_model.py --run results/transformer/runs/$1 --ckpt ${2:-latest} --settings grid --out results/transformer/eval/$1 2>&1 | tee -a results/transformer/eval/$1.log
