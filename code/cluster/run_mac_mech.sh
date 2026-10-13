#!/bin/bash
# Deep maturation-curve measurements on the Mac (mechanism program).
#
#   cd ~/Projects/layered_architectures
#   nohup bash code/cluster/run_mac_mech.sh > output/mac_mech.log 2>&1 &
#
# Measures rho(k) to k = 100000 for d in {256, 1024, 4096, 16384} and
# alpha in {0.75, 1.0, 1.25} (12 cells, 32 rows each) against the
# sealed anchors_prod store, which carries the needed count range.
# Output: output/mech/rho_d{D}_a{A}_L3_deep.json per cell.
# Note: harmless "resource_tracker ... KeyError" lines can appear at
# each cell's end; they are Python teardown noise, not failures.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LSA="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
VENV="$HOME/venvs/layered-architectures"
STORE="${LA_STORE:-$HOME/Projects/product_model_with_memory/tables/anchors_prod}"

[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed store"; exit 1; }
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY="$VENV/bin/python"
export PYTHONPATH="$REPO/code/scripts${PYTHONPATH:+:$PYTHONPATH}"
OUT="$REPO/output/mech"
mkdir -p "$OUT"

for A in 0.75 1.0 1.25; do
  for D in 256 1024 4096 16384; do
    F="$OUT/rho_d${D}_a${A}_L3_deep.json"
    if [ -f "$F" ]; then echo "== skip d=$D a=$A (exists)"; continue; fi
    echo "== rho d=$D alpha=$A  ($(date +%H:%M)) =="
    caffeinate -i "$PY" "$REPO/code/scripts/mechanism_quantities.py" rho \
        --d "$D" --alpha "$A" --L 3 --rows 32 --k-max 100000 \
        --jobs 8 --store "$STORE" --out "$F" \
      || echo "!! FAILED d=$D a=$A (continuing)"
  done
done
echo "MECH BATCH ALL DONE"
