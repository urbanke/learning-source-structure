#!/bin/bash
# Exact-crossing pilot: every cell with d <= MAXD (default 1024),
# SEEDS seeds each (default 0-255), prior-guided bisection to the
# token.  Cells enumerated from the recorded summaries in
# output/takeover/ (the priors); cheapest (smallest d) first, PAR
# cells in parallel, one core per cell.  Per-cell timing is printed
# so affordability is visible as it runs; safe to stop and restart
# (finished cells are skipped).
#
#   cd ~/Projects/layered_architectures
#   nohup bash code/cluster/run_mac_exact.sh > output/mac_exact.log 2>&1 &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$HOME/venvs/layered-architectures"
PAR="${PAR:-12}"
MAXD="${MAXD:-1024}"
SEEDS="${SEEDS:-0-255}"
STORE="${LA_STORE:-$HOME/Projects/product_model_with_memory/tables/anchors_prod}"

[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "== exact pilot starting: enumerating cells (d <= $MAXD) =="
PY="$VENV/bin/python"
"$PY" -c 'import lsa' 2>/dev/null || { echo "ERROR: venv missing"; exit 1; }
export PYTHONPATH="$REPO/code/scripts${PYTHONPATH:+:$PYTHONPATH}"
OUT="$REPO/output/exact"
mkdir -p "$OUT/logs"

ls "$REPO"/output/takeover/d*/summary_*.json \
  | awk -F'summary_' '{print $2, $0}' | sed 's/\.json / /' \
  | while read -r NAME SPATH; do
      D=$(echo "$NAME" | sed 's/d\([0-9]*\)_a.*/\1/')
      if [ "$D" -le "$MAXD" ]; then echo "$D $NAME $SPATH"; fi
    done | sort -n | awk '{print $2, $3}' > /tmp/la_exact_cells.txt
echo "== $(wc -l < /tmp/la_exact_cells.txt | tr -d ' ') cells (d <= $MAXD), seeds $SEEDS, $PAR at a time =="

cat > /tmp/la_exact_cell.sh <<CELL
#!/bin/bash
NAME="\$1"; SPATH="\$2"
D=\$(echo "\$NAME" | sed 's/d\([0-9]*\)_a.*/\1/')
A=\$(echo "\$NAME" | sed 's/.*_a\([0-9.]*\)_L.*/\1/')
L=\$(echo "\$NAME" | sed 's/.*_L\([0-9]*\)\$/\1/')
OUTF="$OUT/exact_\${NAME}.json"
[ -f "\$OUTF" ] && { echo "skip \$NAME (done)"; exit 0; }
if "$PY" "$REPO/code/scripts/exact_crossing.py" \\
    --d "\$D" --alpha "\$A" --L "\$L" --seeds "$SEEDS" \\
    --prior-summary "\$SPATH" --store "$STORE" --out "\$OUTF" \\
    > "$OUT/logs/\${NAME}.log" 2>&1
then tail -1 "$OUT/logs/\${NAME}.log"
else echo "FAIL \$NAME (log: $OUT/logs/\${NAME}.log)"; exit 1; fi
CELL
chmod +x /tmp/la_exact_cell.sh

FAILED=0
caffeinate -i xargs -P "$PAR" -L1 /tmp/la_exact_cell.sh \
    < /tmp/la_exact_cells.txt || FAILED=1
if [ "$FAILED" = 1 ]; then echo "EXACT PILOT DONE WITH FAILURES"; else echo "EXACT PILOT ALL DONE"; fi
