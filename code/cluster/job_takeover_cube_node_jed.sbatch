#!/bin/bash
# Emergence-time cube, large-d tier, WHOLE-NODE version for SCITAS Jed.
# One array element = one full node (72 cores, ~500G) running one
# stripe of the 192 cells through the pmwm production scheduler
# (checkpoint_scheduler.run_planned_schedule) at seed granularity:
# 16 one-core seed jobs per cell with explicit memory sizes, plus a
# summary job per cell.  A seed launches whenever a core and its
# memory fit, so the node stays full wherever memory allows (72 seeds
# at d=16384; ~47 at d=32768, where memory binds before cores).
#
#   sbatch --array=0 code/cluster/job_takeover_cube_node_jed.sbatch   # test
#   sbatch --array=1-11 code/cluster/job_takeover_cube_node_jed.sbatch
#
# Time request is generous (charged on use).  Safe to resubmit:
# finished seeds are skipped via manifests and --resume.  Results in
# /scratch/urbanke/layered_architectures/takeover, per-seed logs in
# .../logs/cells, scheduler events in the slurm .out file.
#SBATCH --job-name=la-cube-node
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=490G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/urbanke/layered_architectures/logs/cube-node-%A_%a.out

set -euo pipefail
module load gcc python 2>/dev/null || true

NODES_TOTAL=12

# one BLAS thread per process; parallelism is whole seed processes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPOSITORY="${LA_REPO:-$HOME/Projects/layered_architectures}"
LSA_REPO="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
PMWM_REPO="${LA_PMWM_REPO:-$HOME/Projects/product_model_with_memory}"
STORE="${LA_STORE:-$PMWM_REPO/tables/anchors_prod}"
GRID="${LA_GRID:-$REPOSITORY/code/cluster/takeover_cube_scitas.json}"
SCRATCH="/scratch/urbanke/layered_architectures"
PY="${LA_PY:-$HOME/venvs/layered-architectures/bin/python}"
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"

[ -d "$STORE" ] || { echo "ERROR: anchor store not found: $STORE"; exit 1; }
[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
# the settled anchor-system configuration.  ALL FOUR variables
# together, never a subset (see notes/log.md, 2026-08-27 incident).
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
export PYTHONPATH="$REPOSITORY/code/scripts:$LSA_REPO/src:$PMWM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SCRATCH/logs/cells" "$SCRATCH/takeover" "$SCRATCH/manifests"

echo "node stripe $TASK/$NODES_TOTAL  store=$STORE  python=$($PY --version 2>&1)"
srun "$PY" "$REPOSITORY/code/cluster/node_cube_scheduler.py" \
    --grid "$GRID" --stripe "$TASK" --stripes "$NODES_TOTAL" \
    --max-workers "$SLURM_CPUS_PER_TASK" --memory-gib 450 \
    --store "$STORE" --out-root "$SCRATCH/takeover" \
    --manifests "$SCRATCH/manifests" --logs "$SCRATCH/logs/cells" \
    --python "$PY"
