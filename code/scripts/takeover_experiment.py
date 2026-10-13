"""Measure the takeover time n* of the copier, one (d, alpha, L) config.

Discovery instrument for the scaling of n* (mechanisms.tex, Section 2.4):
runs several seeds, each with an adaptive horizon (doubled until a stable
takeover is seen or the cap is reached), records the exact Lambda(t)
trajectory on a geometric checkpoint grid, and writes per-seed JSON plus
a per-config summary.

Evaluation goes through the LSA package's intended mechanism and nothing
else: per-checkpoint count profiles, deduplicated ACROSS all seeds and
checkpoints of the configuration, evaluated in batches by
``depth_averaged_codelength_profiles``.  Moment tables come from the
permanent universal store (``--store``): certified values, computed once
ever, grown on demand, shared by every experiment and machine.
Parallelism is the package's own worker pool (``--jobs``); only the
parent process touches the store.

Definitions recorded per seed (all exact given the sequence):
  n_star_first   first checkpoint with Lambda > 0
  n_star_last    first checkpoint after the last nonpositive Lambda
                 (the paper's takeover time, up to grid resolution)
  n_star_10      first checkpoint from which Lambda stays >= 10 bits
  dip_bits, dip_t   depth and location of the minimum of Lambda
  censored       no stable takeover within n_cap

Stability rule: a takeover is accepted once the last sign change lies
before 0.7 * horizon and Lambda(horizon) >= 25 bits; otherwise the
horizon is doubled, up to --n-cap (then: censored).

The per-config summary adds the crossing of the seed-averaged Lambda
(computed on the common checkpoint grid, linearly interpolated in
log t), which is the smooth object for exponent fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

# The settled production configuration of the moment-table system
# (product_model_with_memory HANDOVER, "What the evaluator is now"):
# a DESIGNED, sealed anchor store serves anchor columns bit-identically,
# counts between anchors by degree-11 barycentric interpolation in
# ln(r+1), and every level >= 54 by the certified expansion.  All of it
# is engaged by environment variables that must be set BEFORE the lsa
# modules are imported; engage_anchor_mode() below does that and is
# called first thing in main().
ANCHOR_ENV = {
    "LSA_PHI_LADDER_EVERY": "1",
    "LSA_PHI_LADDER_DEGREE": "11",
    "LSA_PHI_SADDLE_MIN_L": "54",
}


def engage_anchor_mode(store: str, allow_cache: bool = False) -> bool:
    """Point the evaluator at ``store`` and switch on the anchor system.

    Refuses an unsealed store: a run pointed at an open cache silently
    builds an exact column for every new count and grows it without
    bound (the 2026-08-27 318 GB incident here; the same failure is
    documented in pmwm as "the job appeared to hang").  Returns whether
    the store is sealed."""

    sealed = (Path(store) / "anchors.json").exists()
    if not sealed and not allow_cache:
        raise SystemExit(
            f"refusing to run: {store} has no anchors.json, so it is an "
            "open cache, not a designed store.  Build one with "
            "scripts/build_anchor_store.py (with LSA_BUILD_EXACT=1) or "
            "pass --allow-cache if growing a cache is really intended.")
    os.environ["LSA_UNIVERSAL_TABLES"] = str(store)
    if sealed:
        for k, v in ANCHOR_ENV.items():
            os.environ.setdefault(k, v)
    return sealed


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def is_stable(lam: list[tuple[int, float]], horizon: int) -> bool:
    neg = [t for (t, v) in lam if v <= 0]
    last_neg = max(neg) if neg else 0
    return lam[-1][1] >= 25.0 and last_neg <= 0.7 * horizon


def make_result(d, alpha, L, seed, l_max, n_cap, horizon, censored,
                losses, seconds) -> dict:
    lam = [(t, lF - lC) for (t, lF, lC) in losses]
    ts = [t for (t, _) in lam]
    vs = [v for (_, v) in lam]
    pos = [t for (t, v) in lam if v > 0]
    neg = [t for (t, v) in lam if v <= 0]
    last_neg = max(neg) if neg else 0
    after = [t for t in ts if t > last_neg]
    n_star_last = min(after) if (after and not censored) else None
    n_star_first = min(pos) if pos else None
    n_star_10 = None
    for i, (t, v) in enumerate(lam):
        if v >= 10.0 and all(w >= 10.0 for (_, w) in lam[i:]):
            n_star_10 = t
            break
    i_dip = int(np.argmin(vs))
    return {
        "d": d, "alpha": alpha, "L": L, "seed": seed,
        "l_max": l_max, "n_cap": n_cap, "horizon": horizon,
        "censored": censored,
        "n_star_first": n_star_first,
        "n_star_last": n_star_last,
        "n_star_10": n_star_10,
        "dip_bits": vs[i_dip], "dip_t": ts[i_dip],
        "lambda_final": vs[-1],
        "checkpoints": ts, "lam": vs,
        "l_F": [a for (_, a, _) in losses],
        "l_C": [c for (_, _, c) in losses],
        "seconds": seconds,
    }


def mean_crossing(results: list[dict]) -> float | None:
    """Zero crossing (last) of the seed-averaged Lambda, interpolated
    linearly in log t, on the grid common to all seeds."""

    grids = [r["checkpoints"] for r in results]
    common = sorted(set(grids[0]).intersection(*map(set, grids[1:]))) \
        if len(grids) > 1 else grids[0]
    if not common:
        return None
    mean = []
    for t in common:
        vals = [r["lam"][r["checkpoints"].index(t)] for r in results]
        mean.append(sum(vals) / len(vals))
    cross = None
    for i in range(1, len(common)):
        if mean[i - 1] <= 0 < mean[i]:
            t0, t1, v0, v1 = common[i - 1], common[i], mean[i - 1], mean[i]
            f = -v0 / (v1 - v0)
            cross = math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
    if mean[-1] <= 0:
        return None
    return cross


def seed_worker(payload):
    """One seed, start to finish (its own evaluated-profiles dict).
    Used by the seed-parallel path; must stay importable (spawn)."""

    (d, alpha, L, n_cap, seed, jobs, store, out_dir) = payload
    ns = argparse.Namespace(d=d, alpha=alpha, L=L, n_cap=n_cap,
                            jobs=jobs, store=store)
    engage_anchor_mode(store)
    return run_config(ns, [seed], [], Path(out_dir), quiet=True)[0]


def seed_workers_for(d: int, requested: int) -> int:
    """Memory-aware seed parallelism: one seed's rule rows take up to
    d*d*8 bytes; keep the sum under LA_SEED_MEM_GB (default 18, sized
    for the 24 GB Mac; the jed sbatch raises it)."""

    budget = float(os.environ.get("LA_SEED_MEM_GB", "18")) * 1e9
    fit = max(1, int(budget / (d * d * 8.0)))
    return max(1, min(requested, fit))


def run_config(args, seeds_todo, done, out_dir, quiet=False) -> list[dict]:
    """Each seed generates its chain only as far as its own adaptive
    horizon grows.  With --seed-parallel > 1 (and memory permitting),
    several seeds run as separate processes, each with a share of the
    evaluation pool; otherwise seeds run one after another and the
    evaluated-profiles dictionary persists across them."""

    from lsa.codelength import (default_l_max,
                                depth_averaged_codelength_profiles)
    import mech_common as mc

    d, alpha, L, n_cap = args.d, args.alpha, args.L, args.n_cap
    workers = seed_workers_for(d, getattr(args, "seed_parallel", 1))
    if workers > 1 and len(seeds_todo) > 1:
        import multiprocessing as mp

        # Seed workers are daemonic pool members and may not spawn
        # children; the lsa evaluator pools whenever jobs > 1.  So all
        # parallelism is ACROSS seeds (and, in run_mac.sh, across
        # cells) -- evaluation inside a seed worker is always serial.
        # jobs > 1 with seed-parallel > 1 crashed every cell of the
        # 2026-08-27 cube start with "daemonic processes are not
        # allowed to have children".
        inner = 1
        print(f"  [{workers} seed workers, serial evaluation each]",
              flush=True)
        work = [(d, alpha, L, n_cap, s, inner, args.store, str(out_dir))
                for s in seeds_todo]
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = list(done) + pool.map(seed_worker, work)
        results.sort(key=lambda r: r["seed"])
        return results

    l_max = default_l_max(d)
    estimator = ("lsa.depth_averaged_codelength_profiles/universal"
                 f"/l_max={l_max}"
                 f"/ladder_every={os.environ.get('LSA_PHI_LADDER_EVERY', '0')}"
                 f"/degree={os.environ.get('LSA_PHI_LADDER_DEGREE', '-')}"
                 f"/saddle_min_L={os.environ.get('LSA_PHI_SADDLE_MIN_L', '-')}")
    t0 = time.time()
    evaluated: dict = {}
    results = list(done)

    for s in seeds_todo:
        rng = np.random.default_rng([s, d, L, int(round(alpha * 100)), 7])
        chain = mc.ChainSampler(d, alpha, L, rng)
        traj = mc.ProfileTrajectory(d)
        horizon = min(2048, n_cap)
        while True:
            x = chain.extend_to(horizon)
            traj.advance(x, mc.geometric_grid(8, horizon))

            need = traj.profiles_needed()
            big = [fp for (_, fp, _) in traj.pending if len(fp) > 1000]
            need -= evaluated.keys()
            if need:
                t1 = time.time()
                evaluated.update(depth_averaged_codelength_profiles(
                    {p: p for p in need}, d=d, l_max=l_max,
                    jobs=args.jobs, tables_source="universal",
                    universal_path=args.store))
                print(f"  [{len(need)} new profiles in "
                      f"{time.time()-t1:.1f}s, {len(evaluated)} held]",
                      flush=True)
            traj.resolve(evaluated)
            for fp in big:  # frequency-strategy profiles never repeat
                evaluated.pop(fp, None)

            stable = is_stable(traj.lam(), horizon)
            if stable or horizon >= n_cap:
                break
            horizon = min(2 * horizon, n_cap)

        del chain  # frees this seed's rule rows and token buffer
        r = make_result(d, alpha, L, s, l_max, n_cap, horizon,
                        not stable, traj.losses,
                        round(time.time() - t0, 1))
        r["sequence_estimator"] = estimator
        r["store"] = str(args.store)
        import resource
        import sys as _sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gib = rss / 2**20 if _sys.platform == "linux" else rss / 2**30
        r["peak_rss_gib"] = round(rss_gib, 2)
        print(f"  d={d} a={alpha} L={L} seed={s}: "
              f"n*={r['n_star_last']} (dip {r['dip_bits']:.0f} bits "
              f"at {r['dip_t']}) censored={r['censored']} "
              f"horizon={horizon} peak_rss={rss_gib:.2f}GiB", flush=True)
        path = out_dir / f"d{d}_a{alpha}_L{L}_seed{s}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(r, indent=1))
        tmp.replace(path)
        results.append(r)

    results.sort(key=lambda r: r["seed"])
    return results


def validate(store: str | None) -> None:
    """Two checks on one small case.

    1. Bookkeeping: the profile ledger must equal the validated
       mech_smoke path at every checkpoint to 1e-9 when BOTH read the
       universal store (same source, so only the ledger logic is
       tested).
    2. Sources: the ledger values are compared once against the legacy
       per-run tables (build_tables_fast).  The two sources are known
       to differ by ~1e-3 bits per profile at small counts (the level
       recursion drifts with depth; the store's certified methods
       replaced it for exactly that reason), so this line is REPORTED,
       not asserted."""

    import os

    from lsa.codelength import (default_l_max,
                                depth_averaged_codelength_profiles)
    import mech_common as mc
    import mech_smoke as ms

    d, alpha, L, n = 64, 1.0, 2, 600
    l_max = default_l_max(d)
    rng = np.random.default_rng([3, d, L, 100, 7])
    chain = mc.ChainSampler(d, alpha, L, rng)
    x = chain.extend_to(n)
    cps = mc.geometric_grid(8, n)

    traj = mc.ProfileTrajectory(d)
    # advance in two stages to exercise the resume path
    traj.advance(x[: n // 2], [c for c in cps if c <= n // 2])
    traj.advance(x, cps)
    t0 = time.time()
    n_prof = len(traj.profiles_needed())
    ev = depth_averaged_codelength_profiles(
        {p: p for p in traj.profiles_needed()}, d=d, l_max=l_max,
        jobs=1, tables_source="universal", universal_path=store)
    t_store = time.time() - t0
    traj.resolve(ev)

    # (1) same source on both paths: mech_smoke with state_tables=None
    # resolves every evaluation through the universal store as well
    if store:
        os.environ["LSA_UNIVERSAL_TABLES"] = str(store)
    ref = ms.trajectory(x, d, cps, state_tables=None, l_max=l_max,
                        verbose=False)
    worst = 0.0
    for (t, lF, lC) in traj.losses:
        rF, rC = ref[t]
        worst = max(worst, abs(lF - rF), abs(lC - rC))
        assert abs(lF - rF) < 1e-9 and abs(lC - rC) < 1e-9, \
            (t, lF, rF, lC, rC)
        mix = -(np.logaddexp(-lF * math.log(2), -lC * math.log(2))
                / math.log(2) - 1.0)
        assert min(lF, lC) <= mix <= min(lF, lC) + 1.0 + 1e-12
    print(f"validate 1: profile ledger == mech_smoke, both on the "
          f"universal store ({n_prof} profiles, "
          f"{t_store:.1f}s): {len(traj.losses)} checkpoints, "
          f"max |diff| = {worst:.1e} bits, sandwich holds  OK")

    # (2) source comparison, informational
    from lsa.fast_tables import build_tables_fast
    tables = build_tables_fast(max_L=l_max, r_values=tuple(range(98)),
                               cache_dir="/tmp/lsa_tables_fast")
    legacy = ms.trajectory(x, d, cps, state_tables=tables, l_max=l_max,
                           verbose=False)
    dev = max(max(abs(lF - legacy[t][0]), abs(lC - legacy[t][1]))
              for (t, lF, lC) in traj.losses)
    print(f"validate 2: universal store vs legacy tables, max |diff| = "
          f"{dev:.1e} bits over the trajectory (per-profile drift ~1e-3, "
          "summed over the active states; the store's certified values "
          "are the reference)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--d", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--L", type=int)
    ap.add_argument("--seeds", type=str, default="0-7")
    ap.add_argument("--n-cap", type=int, default=None,
                    help="default: 32*d")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed-parallel", type=int, default=1,
                    help="run up to N seeds as parallel processes "
                         "(capped by memory: one seed's rule rows are "
                         "d*d*8 bytes)")
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds whose JSON already exists "
                         "(with the same n_cap)")
    ap.add_argument("--store", type=str, required=True,
                    help="designed (sealed) anchor store directory")
    ap.add_argument("--allow-cache", action="store_true",
                    help="permit an unsealed store (grows it; testing "
                         "only)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sealed = engage_anchor_mode(args.store, args.allow_cache)
    print(f"store: {args.store} ({'sealed anchor store' if sealed else 'OPEN CACHE'})",
          flush=True)

    if args.validate:
        validate(args.store)
        return
    if args.d is None or args.alpha is None or args.L is None \
            or args.out is None:
        ap.error("--d, --alpha, --L, --out are required unless --validate")

    args.n_cap = args.n_cap if args.n_cap else 32 * args.d
    seeds = parse_seeds(args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)

    done: list[dict] = []
    todo: list[int] = []
    for s in seeds:
        p = args.out / f"d{args.d}_a{args.alpha}_L{args.L}_seed{s}.json"
        r = None
        if args.resume and p.exists():
            r = json.loads(p.read_text())
            if r.get("n_cap") != args.n_cap:
                r = None  # stale file from a different setup: redo
        if r is None:
            todo.append(s)
        else:
            done.append(r)
    if done:
        print(f"  resume: {len(done)} seed(s) already on disk", flush=True)

    t0 = time.time()
    results = run_config(args, todo, done, args.out)

    ok = [r for r in results if not r["censored"]]
    summary = {
        "d": args.d, "alpha": args.alpha, "L": args.L,
        "seeds": seeds, "n_cap": args.n_cap,
        "n_censored": len(results) - len(ok),
        "n_star_last": sorted(r["n_star_last"] for r in ok),
        "n_star_10": sorted(r["n_star_10"] for r in ok
                            if r["n_star_10"] is not None),
        "dip_bits": sorted(r["dip_bits"] for r in results),
        "mean_lambda_crossing": mean_crossing(results),
        "total_seconds": round(time.time() - t0, 1),
    }
    spath = args.out / (f"summary_d{args.d}_a{args.alpha}"
                        f"_L{args.L}.json")
    spath.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("seeds",)}, indent=1))


if __name__ == "__main__":
    main()
