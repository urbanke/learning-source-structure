#!/bin/bash
# Exact-crossing rerun, large-d tiers, whole-node version for jed.
# One array element = one full node running one stripe of the
# per-seed jobs (d 2048-8192 x 128 seeds, 16384 x 64, 32768 x 32;
# censored cells excluded).  Priors: the recorded summaries -- the
# mid tiers from the repo's data branch (materialized below), the
# heavy tiers from this cluster's own scratch results.
#
#   sbatch --array=0 code/cluster/job_exact_node_jed.sbatch     # test
#   sbatch --array=1-11 code/cluster/job_exact_node_jed.sbatch
#
# Resume-safe via manifests; per-seed logs in .../exact_logs.
#SBATCH --job-name=la-exact
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=490G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/urbanke/layered_architectures/logs/exact-node-%A_%a.out

set -euo pipefail
module load gcc python 2>/dev/null || true

STRIPES_TOTAL=12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPOSITORY="${LA_REPO:-$HOME/Projects/layered_architectures}"
LSA_REPO="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
PMWM_REPO="${LA_PMWM_REPO:-$HOME/Projects/product_model_with_memory}"
STORE="${LA_STORE:-$PMWM_REPO/tables/anchors_prod}"
SCRATCH="/scratch/urbanke/layered_architectures"
PY="${LA_PY:-$HOME/venvs/layered-architectures/bin/python}"
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"

[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
export PYTHONPATH="$REPOSITORY/code/scripts:$LSA_REPO/src:$PMWM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# materialize the mid-tier priors from the data branch (idempotent)
if [ ! -d "$SCRATCH/priors_mid" ]; then
  git -C "$REPOSITORY" fetch -q origin data
  mkdir -p "$SCRATCH/priors_mid"
  git -C "$REPOSITORY" archive origin/data \
      results/summaries_cube_2026-08-28 \
    | tar -x -C "$SCRATCH/priors_mid" --strip-components=2
fi
mkdir -p "$SCRATCH/exact" "$SCRATCH/exact_manifests" "$SCRATCH/exact_logs" "$SCRATCH/logs"

echo "exact stripe $TASK/$STRIPES_TOTAL  store=$STORE  python=$($PY --version 2>&1)"
srun "$PY" "$REPOSITORY/code/cluster/node_exact_scheduler.py" \
    --priors-dir "$SCRATCH/priors_mid" --priors-dir "$SCRATCH/takeover" \
    --stripe "$TASK" --stripes "$STRIPES_TOTAL" \
    --max-workers "$SLURM_CPUS_PER_TASK" --memory-gib 450 \
    --store "$STORE" --out-root "$SCRATCH/exact" \
    --manifests "$SCRATCH/exact_manifests" --logs "$SCRATCH/exact_logs" \
    --python "$PY"
