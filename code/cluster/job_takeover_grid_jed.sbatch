#!/bin/bash
# Emergence-time cube, large-d tier (d = 16384 and 32768) on SCITAS Jed.
# One array element = one (d, alpha, L) cell from
# code/cluster/takeover_cube_scitas.json (192 cells, cheapest first).
#
#   sbatch --array=0-191 code/cluster/job_takeover_grid_jed.sbatch
#
# Before the first submission, on jed:
#   1. git clone/pull the repo to $HOME/Projects/layered_architectures
#   2. the sealed anchor store must exist at $LA_STORE (default below;
#      2.8G; it is the pmwm designed store -- anchors.json present).
#      If missing, copy it from the Mac; do NOT let a run build tables.
#   3. venv at $HOME/venvs/layered-architectures with the lsa package
#      installed (pip install -e $HOME/Projects/layered-simplex-architecture)
#   4. mkdir -p /scratch/urbanke/layered_architectures/logs
#
# One python process per cell; parallelism is the LSA worker pool
# (--jobs) times seed processes (--seed-parallel, clamped by memory at
# large d).  Per-cell token caps come from the grid file; the three
# corner cells marked censor_expected report lower bounds, that is
# fine.  Safe to resubmit: finished seeds are skipped (--resume).
#SBATCH --job-name=la-cube
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=108G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/urbanke/layered_architectures/logs/cube-%A_%a.out

set -euo pipefail

# the venv was built against the module python (login python3 is 3.9,
# too old); load the same stack on the compute node
module load gcc python 2>/dev/null || true

# one BLAS thread per worker process; parallelism comes from the pool
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPOSITORY="${LA_REPO:-$HOME/Projects/layered_architectures}"
LSA_REPO="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
STORE="${LA_STORE:-$HOME/Projects/product_model_with_memory/tables/anchors_prod}"
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
# evaluation inside a seed worker is serial (pools never nest), so
# throughput = number of seed processes.  The standard partition caps
# memory at 7G per core (16 cores -> 112G): budget 88G for seed rows,
# which gives 16 workers at d=16384 (34G) and 10 at d=32768 (86G).
# override example (full 16 workers at d=32768, 8 cores idle):
#   sbatch --cpus-per-task=24 --mem=165G --export=ALL,LA_SEED_MEM_GB=150 ...
export LA_SEED_MEM_GB="${LA_SEED_MEM_GB:-88}"
mkdir -p "$SCRATCH/logs" "$SCRATCH/takeover"

IFS=$'\t' read -r D ALPHA L NCAP < <(
  "$PY" -c 'import json,sys; rows=json.load(open(sys.argv[1]))["tasks"]; m=[r for r in rows if r["task"]==int(sys.argv[2])]; assert len(m)==1; r=m[0]; print(r["d"], r["alpha"], r["L"], r["n_cap"], sep="\t")' \
    "$GRID" "$TASK"
)

echo "task $TASK: d=$D alpha=$ALPHA L=$L n_cap=$NCAP  store=$STORE"
export PYTHONPATH="$REPOSITORY/code/scripts:$LSA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

srun "$PY" "$REPOSITORY/code/scripts/takeover_experiment.py" \
    --d "$D" --alpha "$ALPHA" --L "$L" --n-cap "$NCAP" \
    --seeds 0-15 --jobs "$SLURM_CPUS_PER_TASK" --seed-parallel "$SLURM_CPUS_PER_TASK" --resume \
    --store "$STORE" \
    --out "$SCRATCH/takeover/d${D}_a${ALPHA}_L${L}"
