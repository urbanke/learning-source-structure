"""Whole-node exact-crossing runner (jed): per-seed one-core jobs on
the pmwm production scheduler.

Enumerates cells from summary files (the priors), d in
{2048...32768}, EXCLUDING censored cells (no crossing exists; their
certificate cannot improve).  Seeds per cell, per RU 2026-08-28:
128 at d 2048-8192, 64 at 16384, 32 at 32768.  Each seed is one
one-core job running exact_crossing.py --seeds k-k with the cell's
summary as prior; results land as exact_<cell>_s<k>.json and are
merged at harvest.  Memory declarations are generous (they only cap
concurrency): rows + streamed state at ~2x the prior + evaluation
transients + base.

    python node_exact_scheduler.py --priors-dir <dir> [--priors-dir ..]
        --stripe K --stripes 12 --max-workers 72 --memory-gib 450
        --store <store> --out-root <scratch>/exact
        --manifests <scratch>/exact_manifests --logs <scratch>/exact_logs
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import zlib
from pathlib import Path

GIB = 2 ** 30
EXPERIMENT = Path(__file__).resolve().parents[1] / "scripts" / "exact_crossing.py"


def seeds_for(d: int) -> int:
    return 128 if d <= 8192 else (64 if d == 16384 else 32)


def mem_for(d: int, prior: float) -> int:
    return int(d * d * 8 + 2 * prior * 160 + d * 8 * 16000 + 4 * GIB)


def wrap(command, log: Path, mem_bytes: int):
    kib = max(mem_bytes // 1024, 1)
    return ("/bin/bash", "-c",
            f"ulimit -d {kib}; " +
            'exec "$@" > ' + shlex.quote(str(log)) + " 2>&1",
            "cell-runner", *command)


def main() -> None:
    from product_model_with_memory.checkpoint_scheduler import (
        CheckpointJob, run_planned_schedule)

    ap = argparse.ArgumentParser()
    ap.add_argument("--priors-dir", action="append", required=True)
    ap.add_argument("--stripe", type=int, required=True)
    ap.add_argument("--stripes", type=int, required=True)
    ap.add_argument("--max-workers", type=int, required=True)
    ap.add_argument("--memory-gib", type=float, required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = {}
    for pd in args.priors_dir:
        for f in sorted(Path(pd).glob("**/summary_*.json")):
            s = json.loads(f.read_text())
            d = s["d"]
            if not 2048 <= d <= 32768:
                continue
            ns = [v for v in (s.get("n_star_last") or []) if v]
            if not ns:
                continue                      # censored cell: skip
            name = f"d{d}_a{s['alpha']}_L{s['L']}"
            if name not in cells:
                ns_sorted = sorted(ns)
                cells[name] = (d, s["alpha"], s["L"],
                               float(ns_sorted[len(ns_sorted) // 2]),
                               str(f))

    jobs, priorities = [], {}
    order = 0
    for name, (d, a, L, prior, spath) in sorted(
            cells.items(), key=lambda kv: kv[1][3]):
        mem = mem_for(d, prior)
        for k in range(seeds_for(d)):
            job_id = f"{name}_s{k}"
            # deterministic striping (python's hash() is randomized
            # per process and would scatter differently on each node)
            if zlib.crc32(job_id.encode()) % args.stripes != args.stripe:
                continue
            out = Path(args.out_root) / f"exact_{name}_s{k}.json"
            cmd = [args.python, str(EXPERIMENT), "--d", str(d),
                   "--alpha", str(a), "--L", str(L),
                   "--seeds", f"{k}-{k}", "--prior-summary", spath,
                   "--store", args.store, "--out", str(out)]
            jobs.append(CheckpointJob(
                job_id=job_id, job_type="S", checkpoint=None,
                command=wrap(cmd, Path(args.logs) / f"{name}_s{k}.log",
                             mem),
                dependencies=(), outputs=(str(out),),
                completion_manifest=str(Path(args.manifests)
                                        / f"{name}_s{k}.json"),
                workers=1, private_memory_bytes=mem))
            priorities[job_id] = float(order)
        order += 1

    tot_mem_big = max((j.private_memory_bytes for j in jobs),
                      default=0) / GIB
    print(f"stripe {args.stripe}/{args.stripes}: {len(cells)} cells, "
          f"{len(jobs)} seed jobs this stripe, largest declaration "
          f"{tot_mem_big:.1f} GiB", flush=True)
    if args.dry_run:
        for j in jobs[:5]:
            print("  ", j.job_id,
                  f"{j.private_memory_bytes / GIB:.1f} GiB")
        return
    for p in (args.out_root, args.manifests, args.logs):
        Path(p).mkdir(parents=True, exist_ok=True)

    def report(event):
        if event["event"] in ("finished", "failed"):
            print(f"[{event['event']}] {event['job_id']} "
                  f"({event.get('elapsed_seconds', 0):.0f}s, "
                  f"{event.get('assigned_workers', '?')} cores busy)",
                  flush=True)

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
        m = re.search(r"planned job (\S+) failed", str(err))
        print(f"STRIPE {args.stripe} STOPPED: "
              f"{m.group(1) if m else err} failed -- fix and resubmit "
              f"(finished work kept)", flush=True)
        sys.exit(1)
    print(f"stripe {args.stripe}: all seed jobs complete", flush=True)


if __name__ == "__main__":
    main()
