#!/bin/bash
# Exact-crossing cells missing from the Mac pilot, on SCITAS Jed.
# One array element = one (d, alpha, L) cell from
# code/cluster/exact_missing_jed.json (23 cells: d=1024, alpha>=1.1,
# the expensive high-tilt corner the Mac pilot did not reach).
#
#   sbatch --array=0-22 code/cluster/job_exact_jed.sbatch
#
# This runs the SAME exact-crossing computation as the Mac's
# run_mac_exact.sh (code/scripts/exact_crossing.py), 256 seeds per
# cell, prior-guided bisection to the token.  It writes
# exact_d${d}_a${alpha}_L${L}.json next to the Mac results.
#
# Before the first submission, on jed:
#   1. git pull the repo to $HOME/Projects/layered_architectures
#      (this brings the manifest and the 23 prior summaries under
#      code/cluster/exact_priors_jed/).
#   2. the sealed anchor store must exist at $LA_STORE (as for the
#      takeover job); do NOT let a run build tables.
#   3. venv at $HOME/venvs/layered-architectures with lsa installed.
#   4. mkdir -p /scratch/urbanke/layered_architectures/logs
#
# Results land in $SCRATCH/exact; copy them into the repo's
# output/exact/ (or straight to the Mac) to complete the grid.
# Safe to resubmit: a cell whose output already exists is skipped.
#SBATCH --job-name=la-exact
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=108G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/urbanke/layered_architectures/logs/exact-%A_%a.out

set -euo pipefail

# the venv was built against the module python (login python3 is 3.9,
# too old); load the same stack on the compute node
module load gcc python 2>/dev/null || true

# one BLAS thread per worker process; parallelism comes from the pool
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPOSITORY="${LA_REPO:-$HOME/Projects/layered_architectures}"
LSA_REPO="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
STORE="${LA_STORE:-$HOME/Projects/product_model_with_memory/tables/anchors_prod}"
MANIFEST="${LA_MANIFEST:-$REPOSITORY/code/cluster/exact_missing_jed.json}"
PRIORS="${LA_PRIORS:-$REPOSITORY/code/cluster/exact_priors_jed}"
SCRATCH="/scratch/urbanke/layered_architectures"
PY="${LA_PY:-$HOME/venvs/layered-architectures/bin/python}"
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"

[ -d "$STORE" ] || { echo "ERROR: anchor store not found: $STORE"; exit 1; }
[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
# the settled anchor-system configuration.  ALL FOUR variables
# together, never a subset (see notes/log.md, 2026-08-27 incident).
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
mkdir -p "$SCRATCH/logs" "$SCRATCH/exact"

IFS=$'\t' read -r D ALPHA L < <(
  "$PY" -c 'import json,sys; rows=json.load(open(sys.argv[1]))["tasks"]; m=[r for r in rows if r["task"]==int(sys.argv[2])]; assert len(m)==1, "no such task"; r=m[0]; print(r["d"], r["alpha"], r["L"], sep="\t")' \
    "$MANIFEST" "$TASK"
)

NAME="d${D}_a${ALPHA}_L${L}"
OUTF="$SCRATCH/exact/exact_${NAME}.json"
SPATH="$PRIORS/summary_${NAME}.json"
[ -f "$OUTF" ] && { echo "skip $NAME (done)"; exit 0; }
[ -f "$SPATH" ] || { echo "ERROR: prior summary not found: $SPATH"; exit 1; }

echo "task $TASK: exact $NAME  store=$STORE  prior=$SPATH"
export PYTHONPATH="$REPOSITORY/code/scripts:$LSA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

srun "$PY" "$REPOSITORY/code/scripts/exact_crossing.py" \
    --d "$D" --alpha "$ALPHA" --L "$L" \
    --seeds "${LA_SEEDS:-0-255}" \
    --jobs "$SLURM_CPUS_PER_TASK" \
    --prior-summary "$SPATH" \
    --store "$STORE" \
    --out "$OUTF"
