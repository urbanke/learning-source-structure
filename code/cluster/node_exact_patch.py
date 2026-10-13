"""One-node patch run: the cells the exact rerun could not cover.

Two explicit groups (RU 2026-08-29):

  CORNER -- the 4 fully censored corner cells (d=32768, L=2,
  alpha 0.5-0.75): every grid seed hit the 67M cap with Lambda still
  negative.  Rerun at cap 2^28 = 268M tokens, 16 seeds each, with a
  bracket-start hint of 1e8 (their summaries contain no crossing to
  take a median of).  Expected medians ~1e8; certification to
  ~1.5x fits inside the cap with a 1.8x margin.

  MISSED -- 6 cells at d=16384 dropped from the cube because the raw
  campaign had already measured them; their summaries never entered
  the jed priors, so the stripe run never enumerated them.  Ordinary
  exact runs, 64 seeds, caps from their raw summaries.

MISSED runs first (memory-light, ~2h); CORNER after (memory-bound,
~9 concurrent at the standard declaration).  Resume-safe: resubmit
the same sbatch and finished seeds are skipped via manifests.

    python node_exact_patch.py --priors-dir <dir> [--priors-dir ..]
        --max-workers 72 --memory-gib 450 --store <store>
        --out-root <scratch>/exact --manifests <scratch>/exact_manifests
        --logs <scratch>/exact_logs [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from node_exact_scheduler import GIB, EXPERIMENT, mem_for, wrap

CORNER = [(32768, a, 2, 16, 2 ** 28, 1.0e8) for a in (0.5, 0.6, 0.7, 0.75)]
MISSED = [(16384, 0.5, 3, 64, None, None),
          (16384, 0.75, 3, 64, None, None),
          (16384, 1.0, 3, 64, None, None),
          (16384, 1.0, 12, 64, None, None),
          (16384, 1.0, 48, 64, None, None),
          (16384, 1.25, 3, 64, None, None)]


def main() -> None:
    from product_model_with_memory.checkpoint_scheduler import (
        CheckpointJob, run_planned_schedule)

    ap = argparse.ArgumentParser()
    ap.add_argument("--priors-dir", action="append", required=True)
    ap.add_argument("--max-workers", type=int, required=True)
    ap.add_argument("--memory-gib", type=float, required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def find_summary(d, a, L) -> str:
        fn = f"summary_d{d}_a{a}_L{L}.json"
        for pd in args.priors_dir:
            hits = sorted(Path(pd).glob(f"**/{fn}"))
            if hits:
                return str(hits[0])
        raise SystemExit(f"ERROR: no prior summary {fn} in any priors dir")

    jobs, priorities = [], {}
    for order, (d, a, L, n_seeds, cap, hint) in enumerate(MISSED + CORNER):
        name = f"d{d}_a{a}_L{L}"
        spath = find_summary(d, a, L)
        mem = mem_for(d, hint if hint else 6e6)
        for k in range(n_seeds):
            job_id = f"{name}_s{k}"
            out = Path(args.out_root) / f"exact_{name}_s{k}.json"
            cmd = [args.python, str(EXPERIMENT), "--d", str(d),
                   "--alpha", str(a), "--L", str(L),
                   "--seeds", f"{k}-{k}", "--prior-summary", spath,
                   "--store", args.store, "--out", str(out)]
            if cap:
                cmd += ["--n-cap", str(cap)]
            if hint:
                cmd += ["--prior-hint", str(hint)]
            jobs.append(CheckpointJob(
                job_id=job_id, job_type="S", checkpoint=None,
                command=wrap(cmd, Path(args.logs) / f"{name}_s{k}.log",
                             mem),
                dependencies=(), outputs=(str(out),),
                completion_manifest=str(Path(args.manifests)
                                        / f"{name}_s{k}.json"),
                workers=1, private_memory_bytes=mem))
            priorities[job_id] = float(order)

    print(f"patch run: {len(MISSED)} missed cells (64 seeds) + "
          f"{len(CORNER)} corner cells (16 seeds, cap 2^28), "
          f"{len(jobs)} seed jobs; largest declaration "
          f"{max(j.private_memory_bytes for j in jobs) / GIB:.1f} GiB",
          flush=True)
    if args.dry_run:
        for j in jobs[:3] + jobs[-3:]:
            print("  ", j.job_id,
                  f"{j.private_memory_bytes / GIB:.1f} GiB",
                  " ".join(j.command[-8:]))
        return
    for p in (args.out_root, args.manifests, args.logs):
        Path(p).mkdir(parents=True, exist_ok=True)

    def report(event):
        if event["event"] in ("finished", "failed"):
            print(f"[{event['event']}] {event['job_id']} "
                  f"({event.get('elapsed_seconds', 0):.0f}s)", flush=True)

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
        print(f"PATCH RUN STOPPED: {m.group(1) if m else err} failed "
              f"-- fix and resubmit (finished work kept)", flush=True)
        sys.exit(1)
    print("patch run: all seed jobs complete", flush=True)


if __name__ == "__main__":
    main()
