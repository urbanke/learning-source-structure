#!/bin/bash
# Commit the small files of a run (config, log, weights-only checkpoints) and push them, so that another machine
# can evaluate the checkpoints:   bash code/transformer/share_run.sh d64
set -e
cd "$(dirname "$0")/../.."
git add results/transformer/runs/$1/config.json results/transformer/runs/$1/log.jsonl results/transformer/runs/$1/model_*.pt
git commit -m "run $1: checkpoints $(ls results/transformer/runs/$1/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tr '\n' ' ')" || true
git push
