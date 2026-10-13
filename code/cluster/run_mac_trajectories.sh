#!/bin/bash
# Trajectory overlay: individual code lengths l_F(t), l_C(t) and the
# true-model code length l_true(t) on a geometric grid, for a small
# set of cells whose exact crossings the pilot already holds.  One
# forward pass per seed (--traj-only: the crossing search is not
# redone).  Purpose: test the appendix program on the chain itself,
# with random visits and feedback included, and measure how far the
# deployed codes sit from the optimal curves.
#
#   cd ~/Projects/layered_architectures
#   nohup bash code/cluster/run_mac_trajectories.sh > output/mac_traj.log 2>&1 &
#
# Safe next to the exact pilot (separate output files, PAR=4 by
# default so the pilot keeps its cores).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$HOME/venvs/layered-architectures"
PAR="${PAR:-4}"
SEEDS="${SEEDS:-0-7}"
CELLS="${CELLS:-d512_a0.5_L3 d512_a1.0_L3 d1024_a0.5_L3 d1024_a1.0_L3}"
STORE="${LA_STORE:-$HOME/Projects/product_model_with_memory/tables/anchors_prod}"

[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY="$VENV/bin/python"
"$PY" -c 'import lsa' 2>/dev/null || { echo "ERROR: venv missing"; exit 1; }
export PYTHONPATH="$REPO/code/scripts${PYTHONPATH:+:$PYTHONPATH}"
OUT="$REPO/output/exact"
mkdir -p "$OUT/logs"

echo "== trajectory overlay: $CELLS (seeds $SEEDS, $PAR at a time) =="
PIDS=()
for NAME in $CELLS; do
  D=$(echo "$NAME" | sed 's/d\([0-9]*\)_a.*/\1/')
  A=$(echo "$NAME" | sed 's/.*_a\([0-9.]*\)_L.*/\1/')
  L=$(echo "$NAME" | sed 's/.*_L\([0-9]*\)$/\1/')
  SPATH="$REPO/output/takeover/d$D/summary_${NAME}.json"
  [ -f "$SPATH" ] || { echo "skip $NAME (no prior summary)"; continue; }
  OUTF="$OUT/traj_${NAME}.json"
  [ -f "$OUTF" ] && { echo "skip $NAME (done)"; continue; }
  # horizon: 8x the exact median when the pilot already solved the
  # cell (the script falls back to 8x the prior median otherwise)
  TMAX=0
  EXACTF="$OUT/exact_${NAME}.json"
  if [ -f "$EXACTF" ]; then
    TMAX=$("$PY" - "$EXACTF" <<'EOF'
import json, sys
import numpy as np
r = json.load(open(sys.argv[1]))
c = [s["first_pos"] for s in r["seeds"] if s.get("first_pos")]
print(int(min(r.get("n_cap", 10**9), 8 * np.median(c))) if c else 0)
EOF
)
  fi
  (
    if "$PY" "$REPO/code/scripts/exact_crossing.py" \
        --d "$D" --alpha "$A" --L "$L" --seeds "$SEEDS" \
        --prior-summary "$SPATH" --store "$STORE" \
        --out "$OUT/exact_${NAME}.json" \
        --traj-only --traj-max "$TMAX" \
        > "$OUT/logs/traj_${NAME}.log" 2>&1
    then tail -1 "$OUT/logs/traj_${NAME}.log"
    else echo "FAIL $NAME (log: $OUT/logs/traj_${NAME}.log)"; fi
  ) &
  PIDS+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait
echo "TRAJECTORY BATCH ALL DONE"
