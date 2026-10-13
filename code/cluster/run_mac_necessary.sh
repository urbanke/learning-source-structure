#!/bin/bash
# Necessary-curve batch: expected redundancy for one unknown distribution
# drawn from the layered prior (untilted: alpha=0). Five alphabet
# sizes, depth 3, 32 prior draws each, k to 10^6. The evaluated code
# mixes over depths; its mean excess over the generating-depth optimum
# is bounded by log2 Lmax. Current outputs retain both code components.
#
#   cd ~/Projects/layered_architectures
#   nohup bash code/cluster/run_mac_necessary.sh > output/mac_necessary.log 2>&1 &
#
# Uses the sealed anchor store; skips finished cells; caffeinate keeps
# the Mac awake.  Results land in output/necessary/.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LSA="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
VENV="$HOME/venvs/layered-architectures"
JOBS="${JOBS:-12}"
STORE="${LA_STORE:-$HOME/Projects/product_model_with_memory/tables/anchors_prod}"

[ -d "$STORE" ] || { echo "ERROR: anchor store not found: $STORE"; exit 1; }
[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
# the settled anchor-system configuration.  ALL FOUR variables
# together, never a subset (see notes/log.md, 2026-08-27 incident).
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY="$VENV/bin/python"
"$PY" -c 'import lsa' 2>/dev/null || { echo "ERROR: venv missing"; exit 1; }
export PYTHONPATH="$REPO/code/scripts${PYTHONPATH:+:$PYTHONPATH}"
OUT="$REPO/output/necessary"
mkdir -p "$OUT"

for D in 64 256 1024 4096 16384; do
  F="$OUT/nec_d${D}_L3.json"
  if [ -f "$F" ]; then echo "== d=$D done, skipping =="; continue; fi
  echo "== d=$D  ($(date +%H:%M)) =="
  caffeinate -i "$PY" "$REPO/code/scripts/mechanism_quantities.py" rho \
      --d "$D" --alpha 0 --L 3 --rows 32 --k-max 1000000 \
      --jobs "$JOBS" --store "$STORE" --out "$F" \
    || { echo "!! FAILED d=$D (stopping; deterministic bug)"; exit 1; }
done
echo "NECESSARY BATCH ALL DONE"
