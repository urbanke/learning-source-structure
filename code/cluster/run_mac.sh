#!/bin/bash
# Run the small-d part of the discovery grid natively on the Mac.
#
#   cd ~/Projects/layered_architectures
#   nohup bash code/cluster/run_mac.sh > output/mac_grid.log 2>&1 &
#
# Grid file: LA_GRID (default code/cluster/takeover_grid.json).  For
# the 10x10x10 cube use LA_GRID=code/cluster/takeover_cube_mac.json;
# cube files are pre-sorted cheapest first and run in that order.
#
# Runs every grid configuration with d <= MAXD, 16 seeds each, in two
# phases: small cells run PAR at a time (one serial process per cell),
# big cells one at a time with JOBS parallel seed processes.  Moment
# tables come from the permanent universal store (STORE below) --
# nothing is built per run.  Results land in output/takeover/ inside
# the repo; per-cell logs of the small phase in output/takeover/
# cell_logs/.  Safe to interrupt and restart: finished seeds are
# skipped (--resume).  caffeinate keeps the Mac from sleeping.
#
# First run creates a venv at ~/venvs/layered-architectures and
# installs the LSA package from ~/Projects/layered-simplex-architecture.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LSA="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
VENV="$HOME/venvs/layered-architectures"
JOBS="${JOBS:-12}"
MAXD="${MAXD:-32768}"
GRID="${LA_GRID:-$REPO/code/cluster/takeover_grid.json}"
STORE="${LA_STORE:-$HOME/Projects/product_model_with_memory/tables/anchors_prod}"

[ -d "$STORE" ] || { echo "ERROR: anchor store not found: $STORE"; exit 1; }
[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
# the settled anchor-system configuration (pmwm HANDOVER).  ALL FOUR
# variables together, never a subset: a partial set silently points a
# helper (e.g. mech_smoke --validate) at a default cache path — that
# crashed the 2026-08-27 restart until the store path was exported too.
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54

# one BLAS thread per worker process; parallelism comes from the pool
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ ! -d "$VENV" ]; then python3 -m venv "$VENV"; fi
PY="$VENV/bin/python"
"$PY" -c 'import lsa' 2>/dev/null || "$VENV/bin/pip" -q install -e "$LSA"

export PYTHONPATH="$REPO/code/scripts${PYTHONPATH:+:$PYTHONPATH}"
OUT="$REPO/output/takeover"
mkdir -p "$OUT"

echo "== validation (store: $STORE) =="
"$PY" "$REPO/code/scripts/takeover_experiment.py" --validate --store "$STORE"
"$PY" "$REPO/code/scripts/mech_smoke.py" --validate

"$PY" - "$GRID" "$MAXD" <<'EOF' > /tmp/la_tasks.tsv
import json, sys
rows = json.load(open(sys.argv[1]))["tasks"]
rows = [r for r in rows if r["d"] <= int(sys.argv[2])]
if all("task" in r for r in rows):
    rows.sort(key=lambda r: r["task"])      # cube files: cheapest first
else:
    rows.sort(key=lambda r: (r["d"], r["L"], r["alpha"]))
for r in rows:
    print(r["d"], r["alpha"], r["L"], r.get("n_cap", 32 * r["d"]), sep="\t")
EOF
echo "== $(wc -l < /tmp/la_tasks.tsv | tr -d ' ') configurations from $(basename "$GRID"), JOBS=$JOBS =="

# Two phases.  Small cells (n_cap <= 2^19): PAR whole cells at a time,
# each a single serial process -- this is where most of the cube's
# cells (and none of its tokens) live, and per-cell startup would
# otherwise dominate.  Big cells: one at a time, JOBS seed processes.
# All parallelism is separate processes; pools never nest (the lsa
# evaluator pools when jobs > 1, and a pooled seed worker may not
# spawn children -- that crashed the 2026-08-27 cube start).
awk -F'\t' '$4 <= 524288' /tmp/la_tasks.tsv > /tmp/la_small.tsv
awk -F'\t' '$4 >  524288' /tmp/la_tasks.tsv > /tmp/la_big.tsv
PAR="${PAR:-$JOBS}"
CAFF="$(command -v caffeinate || true)"
mkdir -p "$OUT/cell_logs"

cat > /tmp/la_cell.sh <<CELL
#!/bin/bash
# one small cube cell: \$1=d \$2=alpha \$3=L \$4=n_cap (serial process)
LOG="$OUT/cell_logs/d\${1}_a\${2}_L\${3}.log"
if "$PY" "$REPO/code/scripts/takeover_experiment.py" \\
    --d "\$1" --alpha "\$2" --L "\$3" --n-cap "\$4" \\
    --seeds 0-15 --jobs 1 --resume \\
    --store "$STORE" --out "$OUT/d\${1}_a\${2}_L\${3}" > "\$LOG" 2>&1
then echo "ok   d=\$1 a=\$2 L=\$3  (\$(date +%H:%M))"
else echo "FAIL d=\$1 a=\$2 L=\$3  (log: \$LOG)"; exit 1; fi
CELL
chmod +x /tmp/la_cell.sh

FAILED=0
if [ -s /tmp/la_small.tsv ]; then
  echo "== phase A: $(wc -l < /tmp/la_small.tsv | tr -d ' ') small cells, $PAR at a time =="
  ${CAFF:+$CAFF -i} xargs -P "$PAR" -L1 /tmp/la_cell.sh < /tmp/la_small.tsv || FAILED=1
fi

if [ -s /tmp/la_big.tsv ]; then
  echo "== phase B: $(wc -l < /tmp/la_big.tsv | tr -d ' ') big cells, one at a time, $JOBS seed workers =="
  while IFS=$'\t' read -r D A L NCAP; do
    echo "== d=$D alpha=$A L=$L n_cap=$NCAP  ($(date +%H:%M)) =="
    ${CAFF:+$CAFF -i} "$PY" "$REPO/code/scripts/takeover_experiment.py" \
        --d "$D" --alpha "$A" --L "$L" --n-cap "$NCAP" \
        --seeds 0-15 --jobs "$JOBS" --seed-parallel "$JOBS" --resume \
        --store "$STORE" \
        --out "$OUT/d${D}_a${A}_L${L}" \
      || { echo "!! FAILED d=$D alpha=$A L=$L (continuing)"; FAILED=1; }
  done < /tmp/la_big.tsv
fi

if [ "$FAILED" = 1 ]; then echo "DONE WITH FAILURES"; else echo "ALL DONE"; fi
