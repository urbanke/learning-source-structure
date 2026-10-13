#!/bin/bash
# Copy the small files of all runs on Kuma (config.json, log.jsonl, the weights-only checkpoints model_<step>.pt and
# the job outputs) and the transformer's evaluation outputs (results/transformer/eval, git-ignored) into the same
# place in this project folder; only new files are transferred. Run on the laptop, from the project folder:
#     bash code/transformer/kuma/fetch.sh
cd "$(dirname "$0")/../../.."
mkdir -p results/transformer/runs results/transformer/eval
rsync -av -e "ssh -i ~/.ssh/id_ed25519" --include='*/' --include='model_*.pt' --include='config.json' --include='log.jsonl' --include='tf-*.out' --exclude='*' \
    urbanke@kuma.hpc.epfl.ch:Projects/layered_architectures/results/transformer/runs/ results/transformer/runs/
rsync -av -e "ssh -i ~/.ssh/id_ed25519" --include='*/' --include='*.npz' --exclude='*' \
    urbanke@kuma.hpc.epfl.ch:Projects/layered_architectures/results/transformer/eval/ results/transformer/eval/
