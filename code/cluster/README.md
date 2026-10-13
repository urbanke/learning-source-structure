# Running the takeover grid

The evaluation mechanism is the LSA package's own:
`depth_averaged_codelength_profiles` over deduplicated count profiles,
moment tables from the permanent universal store (certified values,
computed once ever, grown on demand, shared by every experiment and
machine).  Nothing is built per run.  Parallelism is the package's
worker pool; only the parent process touches the store.

Stores (one per machine, reused forever):
- Mac: `~/Projects/product_model_with_memory/tables/universal_v2`
  (the existing 177 GB store; the two packages share the store format).
- Jed: point `LA_STORE` at the universal_v2 store there (rsync it from
  the Mac if none exists yet, or reuse the one from the
  product_model_with_memory campaigns).

Grid v2 (serious scale, RU 2026-08-26): d up to 32768 on the Mac
(memory cap: one seed's rule rows), d = 65536 and 100000 on SCITAS;
n_cap per task follows the observed n* ~ d^1.9 trend up to 2^30 tokens
at d = 100000.  Data depth L up to 96; alpha 0.5..1.5.  Tasks 0-20 run
on the Mac (the d = 16384 and 32768 configurations are hours each —
the full set is an overnight run), tasks 21-24 on SCITAS.  Hold the
SCITAS submission until the Mac results are read.

## On the Mac (d <= 32768)

From the repo root, after `git pull`:

    nohup bash code/cluster/run_mac.sh > output/mac_grid.log 2>&1 &

First run creates a venv at `~/venvs/layered-architectures` and
installs the LSA package from `~/Projects/layered-simplex-architecture`.
Uses 12 workers (override with JOBS=...), keeps the Mac awake while
computing, writes to `output/takeover/` in the repo, and skips already
finished seeds if restarted.  Watch with `tail -f output/mac_grid.log`.
Stop everything with:

    pkill -f run_mac.sh; pkill -f takeover_experiment; pkill -x caffeinate

## On SCITAS (Jed), d >= 65536

One-time setup on the cluster (from your Mac):

    rsync -av --exclude .venv --exclude output \
        ~/Projects/layered_architectures/ jed:~/Projects/layered_architectures/
    rsync -av --exclude .venv --exclude output --exclude tables \
        ~/Projects/layered-simplex-architecture/ jed:~/Projects/layered-simplex-architecture/

On the cluster, create the venv once:

    python3 -m venv ~/venvs/layered-architectures
    ~/venvs/layered-architectures/bin/pip install -e ~/Projects/layered-simplex-architecture

Submit the large-d part (4 configurations, 16 seeds each):

    mkdir -p /scratch/urbanke/layered_architectures/logs
    cd ~/Projects/layered_architectures
    sbatch --array=21-24 code/cluster/job_takeover_grid_jed.sbatch

(`--array=0-24` would run the full grid there instead.)

Notes.
- The grid is defined in `takeover_grid.json`; the sbatch reads its own
  array index there.  Requested resources (16 cores, 120G, 12h) are
  deliberately generous; retune from the first runs' logs.
- Results: per-seed JSON plus `summary_*.json` per configuration.
  Fetch with rsync and commit curated summaries to `results/` in the
  repo.

## Exact-crossing cells on SCITAS (the missing d=1024, alpha>=1 corner)

The exact-crossing grid (`exact_crossing.py`, the optimal / lower-bound
n*) has been a Mac pipeline (`run_mac_exact.sh`).  The Mac pilot left
23 cells unfinished, all at d=1024 with alpha>=1.1 (the expensive
high-tilt corner).  This job runs exactly those on the cluster.  It is
self-contained: the manifest `exact_missing_jed.json` lists the 23
cells, and their prior summaries travel with the repo in
`exact_priors_jed/`, so no separate rsync of summaries is needed.

    cd ~/Projects/layered_architectures && git pull
    mkdir -p /scratch/urbanke/layered_architectures/logs
    sbatch --array=0-22 code/cluster/job_exact_jed.sbatch

Results land in `/scratch/urbanke/layered_architectures/exact/` as
`exact_d1024_a*_L*.json`, the same schema as the Mac's `output/exact/`;
copy them there (or to the Mac) to complete the 500-cell grid.  Safe to
resubmit: a finished cell is skipped.

- 256 seeds per cell, sequential, each using all 16 cores (`--jobs`).
- The low-depth alpha=1.5 cells (tasks 13-15) are the slowest; if one
  does not finish inside `--time`, raise it, or split the seeds across
  two submissions with `--export=ALL,LA_SEEDS=0-127` and `128-255` into
  separate `--out` files and merge.  exact_crossing writes only on
  completion, so an over-time cell produces nothing.
- Before trusting anything from a changed code state, run
  `python code/scripts/takeover_experiment.py --validate --store <store>`
  (check 1: the profile ledger equals the validated mech_smoke path to
  1e-9 on the same source; check 2, informational: universal store vs
  legacy per-run tables, expected ~1e-2 bits over the small trajectory
  — the store's certified values are the reference) and
  `python code/scripts/mech_smoke.py --validate` (chain rule, collapse,
  sandwich).
