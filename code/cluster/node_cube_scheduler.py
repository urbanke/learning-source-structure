"""Whole-node cube runner on top of the pmwm production scheduler.

Drives ``run_planned_schedule`` from product_model_with_memory
(checkpoint_scheduler.py) -- the same memory-aware executor that ran
the pmwm campaigns -- over one stripe of the cube's large-d cells.

Granularity is the seed: every cell becomes 16 independent one-core
jobs (job type S) with an explicit private-memory size
(d*d*8 rule rows + n_cap*4 chain buffer + 1 GiB interpreter), plus
one summary job (type U) per cell that depends on its 16 seeds and
rewrites the cell summary from the finished seed files.  The
scheduler launches a seed whenever a core AND that much memory are
free, so a 72-core node stays full wherever memory allows (all 72 at
d = 16384; about 47 at d = 32768, where memory binds first).

Failures: the pmwm executor stops everything when a job fails.  This
driver retries the schedule; a job that fails twice is excluded
(with its summary) and reported at the end, so one bad cell cannot
take down a 40-hour node job.  Completed jobs are recognized by
their manifests and never rerun (the seed files double as --resume
state for the experiment itself).

    python node_cube_scheduler.py --grid takeover_cube_scitas.json \
        --stripe 0 --stripes 12 --max-workers 72 --memory-gib 470 \
        --store <anchors_prod> --out-root <scratch>/takeover \
        --manifests <scratch>/manifests --logs <scratch>/logs/cells
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

from product_model_with_memory.checkpoint_scheduler import (
    CheckpointJob,
    run_planned_schedule,
)

GIB = 2 ** 30
EXPERIMENT = Path(__file__).resolve().parents[1] / "scripts" / "takeover_experiment.py"


def wrap(command: list[str], log: Path, mem_bytes: int) -> tuple[str, ...]:
    """Exec the command with its own log file and its declared memory
    enforced (ulimit -d: private memory; the shared store map is not
    counted).  A seed that exceeds its declaration fails alone, with a
    MemoryError in its own log -- it cannot take the node down."""

    kib = max(mem_bytes // 1024, 1)
    return ("/bin/bash", "-c",
            f"ulimit -d {kib}; " +
            'exec "$@" > ' + shlex.quote(str(log)) + " 2>&1",
            "cell-runner", *command)


def cell_jobs(row: dict, args, order: int) -> list[CheckpointJob]:
    d, a, L, ncap = row["d"], row["alpha"], row["L"], row["n_cap"]
    name = f"d{d}_a{a}_L{L}"
    out = Path(args.out_root) / name
    # rule rows + horizon-linear state (chain buffer, successor
    # tables, profile dictionaries) + evaluation transients (batch
    # profile buffers scale with d) + interpreter.  Set from measured
    # peaks: completed d=32768 seeds peaked at 14.47 GiB including
    # transients under caps around 2^25-2^26; the failed full-horizon
    # corner seed wanted ~2.7 GiB more mid-evaluation.  The formula
    # gives 21.8 GiB at (32768, 2^26) and 15.3 GiB at (16384, 2^26);
    # enforcement (ulimit below) makes any remaining shortfall a
    # clean single-seed failure, never a node event.
    # 2026-08-28, final sizing. Measured: completed corner seeds peak
    # at 20.17 GiB at (d=32768, cap 2^26), plus evaluation transients
    # of a few GiB. Declarations cost nothing when unused -- they only
    # cap how many seeds run at once -- so be generous rather than
    # clever: 36 GiB for the biggest cells (12 fit on a node), scaling
    # down with d and cap. Tight margins cost three failed rounds.
    mem = d * d * 8 + ncap * 256 + int(12 * GIB)
    base = [args.python, str(EXPERIMENT), "--d", str(d), "--alpha", str(a),
            "--L", str(L), "--n-cap", str(ncap), "--jobs", "1", "--resume",
            "--store", args.store, "--out", str(out)]
    jobs = []
    for s in range(16):
        jobs.append(CheckpointJob(
            job_id=f"{name}_s{s}", job_type="S", checkpoint=None,
            command=wrap(base + ["--seeds", f"{s}-{s}"],
                         Path(args.logs) / f"{name}_s{s}.log", mem),
            dependencies=(),
            outputs=(str(out / f"{name}_seed{s}.json"),),
            completion_manifest=str(Path(args.manifests) / f"{name}_s{s}.json"),
            workers=1, private_memory_bytes=mem))
    jobs.append(CheckpointJob(
        job_id=f"{name}_summary", job_type="U", checkpoint=None,
        command=wrap(base + ["--seeds", "0-15"],
                     Path(args.logs) / f"{name}_summary.log", 2 * GIB),
        dependencies=tuple(f"{name}_s{s}" for s in range(16)),
        outputs=(str(out / f"summary_{name}.json"),),
        completion_manifest=str(Path(args.manifests) / f"{name}_summary.json"),
        workers=1, private_memory_bytes=2 * GIB))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--stripe", type=int, required=True)
    ap.add_argument("--stripes", type=int, required=True)
    ap.add_argument("--max-workers", type=int, required=True)
    ap.add_argument("--memory-gib", type=float, required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    rows = json.loads(Path(args.grid).read_text())["tasks"]
    rows = [r for r in rows if r["task"] % args.stripes == args.stripe]
    for p in (args.out_root, args.manifests, args.logs):
        Path(p).mkdir(parents=True, exist_ok=True)

    jobs, priorities = [], {}
    for order, row in enumerate(rows):        # grid file is cost-sorted
        block = cell_jobs(row, args, order)
        jobs.extend(block)
        for j in block:
            priorities[j.job_id] = float(order)
    print(f"stripe {args.stripe}/{args.stripes}: {len(rows)} cells, "
          f"{len(jobs)} jobs, {args.max_workers} workers, "
          f"{args.memory_gib:.0f} GiB budget", flush=True)

    def report(event: dict) -> None:
        if event["event"] in ("finished", "failed"):
            print(f"[{event['event']}] {event['job_id']} "
                  f"({event.get('elapsed_seconds', 0):.0f}s, "
                  f"{event.get('assigned_workers', '?')} cores busy)",
                  flush=True)

    # No retries: memory use is deterministic and enforced per seed
    # (ulimit in the wrapper), so a failure is a sizing or code bug.
    # Retrying it would reproduce it -- and the executor's safety rule
    # terminates the stripe's in-flight work each time (that is how
    # the 2026-08-28 under-sizing turned one bug into lost hours).
    # Fail loudly once instead; manifests make the resubmission cheap.
    ones = {j.job_id: 1 for j in jobs}
    try:
        run_planned_schedule(
            tuple(jobs), dict(ones), dict(priorities),
            maximum_workers=args.max_workers,
            maximum_private_memory_bytes=int(args.memory_gib * GIB),
            worker_caps=dict(ones), worker_floors=dict(ones),
            working_directory=args.out_root,
            poll_seconds=0.5, event_callback=report)
    except RuntimeError as err:
        match = re.search(r"planned job (\S+) failed", str(err))
        bad = match.group(1) if match else "unknown"
        print(f"STRIPE {args.stripe} STOPPED: job {bad} failed -- "
              f"deterministic bug, fix the sizing or the code, then "
              f"resubmit (finished work is kept via manifests)",
              flush=True)
        sys.exit(1)
    print(f"stripe {args.stripe}: {len(rows)}/{len(rows)} cells complete",
          flush=True)


if __name__ == "__main__":
    main()
