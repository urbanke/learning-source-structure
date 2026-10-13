#!/bin/bash
# One-node patch run: the 4 censored corner cells (d=32768, L=2,
# cap raised to 2^28, 16 seeds) plus the 6 already-measured cells at
# d=16384 the stripe run never enumerated (64 seeds, standard caps).
# Missed cells run first (cheap), corner cells after (memory-bound).
#
#   sbatch code/cluster/job_exact_patch_jed.sbatch
#
# Resume-safe: resubmit and finished seeds are skipped.
#SBATCH --job-name=la-exact-patch
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=490G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/urbanke/layered_architectures/logs/exact-patch-%j.out

set -euo pipefail
module load gcc python 2>/dev/null || true

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

REPOSITORY="${LA_REPO:-$HOME/Projects/layered_architectures}"
LSA_REPO="${LA_LSA_REPO:-$HOME/Projects/layered-simplex-architecture}"
PMWM_REPO="${LA_PMWM_REPO:-$HOME/Projects/product_model_with_memory}"
STORE="${LA_STORE:-$PMWM_REPO/tables/anchors_prod}"
SCRATCH="/scratch/urbanke/layered_architectures"
PY="${LA_PY:-$HOME/venvs/layered-architectures/bin/python}"

[ -f "$STORE/anchors.json" ] || { echo "ERROR: $STORE is not a sealed designed store"; exit 1; }
export LSA_UNIVERSAL_TABLES="$STORE"
export LSA_PHI_LADDER_EVERY=1 LSA_PHI_LADDER_DEGREE=11 LSA_PHI_SADDLE_MIN_L=54
export PYTHONPATH="$REPOSITORY/code/scripts:$LSA_REPO/src:$PMWM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# materialize priors from the data branch (idempotent):
# mid tiers (cube) and the raw campaign (the 6 missed cells)
if [ ! -d "$SCRATCH/priors_mid" ]; then
  git -C "$REPOSITORY" fetch -q origin data
  mkdir -p "$SCRATCH/priors_mid"
  git -C "$REPOSITORY" archive origin/data \
      results/summaries_cube_2026-08-28 \
    | tar -x -C "$SCRATCH/priors_mid" --strip-components=2
fi
if [ ! -d "$SCRATCH/priors_raw" ]; then
  git -C "$REPOSITORY" fetch -q origin data
  mkdir -p "$SCRATCH/priors_raw"
  git -C "$REPOSITORY" archive origin/data \
      results/summaries_raw_2026-08-27 \
    | tar -x -C "$SCRATCH/priors_raw" --strip-components=2
fi
mkdir -p "$SCRATCH/exact" "$SCRATCH/exact_manifests" "$SCRATCH/exact_logs" "$SCRATCH/logs"

echo "exact patch run  store=$STORE  python=$($PY --version 2>&1)"
srun "$PY" "$REPOSITORY/code/cluster/node_exact_patch.py" \
    --priors-dir "$SCRATCH/priors_raw" --priors-dir "$SCRATCH/takeover" \
    --priors-dir "$SCRATCH/priors_mid" \
    --max-workers "$SLURM_CPUS_PER_TASK" --memory-gib 450 \
    --store "$STORE" --out-root "$SCRATCH/exact" \
    --manifests "$SCRATCH/exact_manifests" --logs "$SCRATCH/exact_logs" \
    --python "$PY"
