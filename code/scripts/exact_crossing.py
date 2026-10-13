"""Exact emergence times by prior-guided bisection.

One cell = (d, alpha, L).  The prior is the cell's median crossing M
from the recorded 16-seed campaign (summary_*.json, n_star_last).
Per seed (any seed id; new seeds use the same generator family):

  1. bracket: evaluate Lambda at M/1.5 and 1.5M, widening by x1.5
     until the sign changes across the bracket (replays are cheap:
     the chain is deterministic, counting streams at ~1M tokens/s);
  2. bisect in log t; a backward probe rebuilds counts by replaying
     from zero -- no snapshots, no approximation; resolution: the
     exact token when the prior M < 1e6, else 0.1 percent;
  3. certify the last crossing: forward probes at x1.05 steps up to
     1.5x the candidate; any negative restarts the search there.

Every probe evaluates the SAME ledger as takeover_experiment
(ProfileTrajectory + the sealed store); evaluated profiles are
cached across probes, replays, and seeds of a cell.  Output mirrors
the summary format, with exact per-seed crossings and the evaluated
bracketing points.

    python code/scripts/exact_crossing.py --d 64 --alpha 1.0 --L 3 \
        --seeds 0-15 --prior-summary results/summaries_cube_2026-08-28/summary_d64_a1.0_L3.json \
        --store <store> --out /tmp/exact_d64

Trajectory mode (--trajectories, off by default; --traj-only skips
the crossing search): one forward pass per seed records, on a
geometric grid of t, the individual code lengths l_F(t) and l_C(t)
of the two deployed strategies AND the code length of the true
model on the same path,
l_true(t) = -log2 q(x_1) - sum_{s<t} log2 T(x_{s+1} | x_s).
The difference l_F - l_C is Lambda; the differences to l_true are
the realized redundancies of the two strategies, the objects the
appendix prices.  Results go to a sibling file (exact_* -> traj_*),
so the crossing schema is untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from takeover_experiment import engage_anchor_mode, parse_seeds


class SeedProber:
    """Lambda(t) probes for one seed, forward-incremental with
    replay-from-zero for backward jumps; shared profile cache."""

    def __init__(self, d, alpha, L, seed, jobs, store, cache):
        import mech_common as mc
        self.mc = mc
        self.d, self.alpha, self.L = d, alpha, L
        self.jobs, self.store, self.cache = jobs, store, cache
        rng = np.random.default_rng([seed, d, L, int(round(alpha * 100)), 7])
        self.chain = mc.ChainSampler(d, alpha, L, rng)
        self.traj = None
        self.pos = 0
        self.evals = 0
        self.replays = 0
        self.hist: list[tuple[int, float]] = []

    def lam(self, t: int) -> float:
        from lsa.codelength import (default_l_max,
                                    depth_averaged_codelength_profiles)
        t = int(t)
        if self.traj is None or t < self.pos:
            if self.traj is not None:
                self.replays += 1
            self.traj = self.mc.ProfileTrajectory(self.d)
            self.pos = 0
        x = self.chain.extend_to(t)
        # positions are absolute; truncate at the probe so advance
        # does not consume past it (it always processes to len(x))
        self.traj.advance(x[:t], [t])
        self.pos = t
        need = self.traj.profiles_needed() - self.cache.keys()
        if need:
            self.cache.update(depth_averaged_codelength_profiles(
                {p: p for p in need}, d=self.d,
                l_max=default_l_max(self.d), jobs=self.jobs,
                tables_source="universal", universal_path=self.store))
        self.traj.resolve(self.cache)
        self.evals += 1
        v = float(self.traj.lam()[-1][1])
        self.hist.append((t, v))
        return v


def first_crossing(p, a, b, n_cap):
    """First crossing AFTER the debt era (RU's gap rule: Lambda must
    have been at least 1 bit negative before it counts).  Probe DOWN
    from the last crossing's left end at /1.05 steps, tracking the
    leftmost positive probe, until Lambda <= -25 bits (deep debt: no
    positive wiggle hides between adjacent 5 percent probes at that
    depth) or t <= 8; then bisect to the leftmost sign change.  If no
    probe ever reaches -1 bit, the seed had no debt era: recorded as
    undefined."""

    # One forward sweep from 8 to the last crossing (a single replay
    # pass, all probes incremental).  Small alphabets have jagged
    # Lambda paths -- 10-bit swings within a 27 percent step at d=64,
    # with several separate debt eras -- so the sweep step adapts to
    # scale: fine where paths are jagged, coarse where they are
    # smooth.  first_cross = the first probe-resolution sign change
    # into positive AFTER the first probe at or below -1 bit (the gap
    # rule), bisected to the token.
    step = 1.05 if b < 1e4 else (1.15 if b < 1e6 else 1.3)
    t = 8
    seen_debt = False
    first_bracket = None
    prev_t, prev_v = None, None
    changes = []         # every sign change at sweep resolution
    while t < b:
        v = p.lam(t)
        if prev_v is not None and (v > 0) != (prev_v > 0):
            changes.append([int(prev_t), int(t),
                            "up" if v > 0 else "down"])
            if v > 0 and seen_debt and first_bracket is None:
                first_bracket = (prev_t, t)
        if v <= -1.0:
            seen_debt = True
        prev_t, prev_v = t, v
        if t >= b:
            break
        t = min(int(t * step) + 1, b)
    if not seen_debt:
        return {"first_cross": None, "first_note": "no debt era",
                "sweep_changes": changes}
    if first_bracket is None:
        # no positive probe after debt within the sweep: the first
        # crossing is the last crossing itself
        return {"first_cross": int(b), "sweep_changes": changes,
                "first_note": "first equals last at sweep resolution"}
    x, y = first_bracket
    while y - x > 1:
        m = (x + y) // 2
        if p.lam(m) > 0:
            y = m
        else:
            x = m
    return {"first_cross": int(y), "sweep_changes": changes}


def exact_seed(d, alpha, L, seed, prior_m, n_cap, jobs, store, cache):
    """Return (last_neg, first_pos, evals, replays) or censored."""

    p = SeedProber(d, alpha, L, seed, jobs, store, cache)
    tol_exact = prior_m < 1e6
    a = max(8, int(prior_m / 1.5))
    b = min(n_cap, int(prior_m * 1.5))

    def L_(t):
        return p.lam(t)

    # find a negative left end
    la = L_(a)
    while la > 0 and a > 8:
        a = max(8, int(a / 1.5))
        la = L_(a)
    if la > 0:
        a, la = 8, la  # crossing below 8: report first_pos <= 8
        return {"seed": seed, "last_neg": None, "first_pos": a,
                "evals": p.evals, "replays": p.replays,
                "note": "positive from the start"}
    lb = L_(b)
    while lb <= 0:
        if b >= n_cap:
            return {"seed": seed, "last_neg": b, "first_pos": None,
                    "evals": p.evals, "replays": p.replays,
                    "note": f"censored at cap {n_cap}"}
        b = min(n_cap, int(b * 1.5))
        lb = L_(b)

    while True:
        # bisect [a, b] with Lambda(a)<=0<Lambda(b)
        while True:
            if tol_exact and b - a <= 1:
                break
            if not tol_exact and (b - a) <= 0.001 * b:
                break
            m = int(round(math.exp((math.log(a) + math.log(b)) / 2)))
            m = min(max(m, a + 1), b - 1)
            if L_(m) > 0:
                b = m
            else:
                a = m
        # certification: forward at x1.05 steps until BOTH t >= 1.5*b
        # and Lambda >= 25 bits (the ledger's stability margin: a dip
        # from +25 to zero within one 5 percent step does not happen
        # at these slopes).  Validation demanded this: seed 3 of the
        # d=64 pilot cell crosses at 41, dips again near 130, and a
        # bare 1.5x horizon missed the second dip.
        bad = None
        t = b
        lv = 0.0
        while (t < min(1.5 * b, n_cap) or lv < 25) and t < n_cap:
            t = min(int(t * 1.05) + 1, n_cap)
            lv = L_(t)
            if lv <= 0:
                bad = t
                break
        if bad is None:
            fc = first_crossing(p, a, b, n_cap)
            return {"seed": seed, "last_neg": a, "first_pos": b,
                    **fc, "evals": p.evals, "replays": p.replays}
        a = bad
        b = min(n_cap, int(bad * 1.5))
        while L_(b) <= 0:
            if b >= n_cap:
                return {"seed": seed, "last_neg": b, "first_pos": None,
                        "evals": p.evals, "replays": p.replays,
                        "note": f"censored at cap {n_cap}"}
            b = min(n_cap, int(b * 1.5))


def trajectory_seed(d, alpha, L, seed, t_max, ratio, jobs, store,
                    cache):
    """One forward pass: (t, l_F, l_C, l_true) on a geometric grid.

    Same generator family as the crossing search (identical seeding),
    so trajectories overlay the recorded crossings seed by seed.
    l_F and l_C follow the ProfileTrajectory conventions of the
    deployed codes (l_C charges log2 d for the first token); l_true
    is the generator's own code length on the path.
    """

    import mech_common as mc
    from lsa.codelength import (default_l_max,
                                depth_averaged_codelength_profiles)
    rng = np.random.default_rng([seed, d, L, int(round(alpha * 100)), 7])
    ch = mc.ChainSampler(d, alpha, L, rng)
    t_max = int(t_max)
    x = ch.extend_to(t_max)

    # true-model bits per position: x[0] from q, later positions from
    # the row of their context (rows exist for every visited context)
    lt = np.full(t_max, np.nan)
    lt[0] = -math.log2(float(ch.q[int(x[0])]))
    if t_max > 1:
        ctx = x[: t_max - 1].astype(np.int64)
        nxt = x[1:t_max]
        for a, cum in ch.rows.items():
            sel = np.nonzero(ctx == a)[0]
            if len(sel):
                pmf = np.diff(cum, prepend=0.0)
                lt[sel + 1] = -np.log2(np.maximum(pmf[nxt[sel]],
                                                  1e-300))
    if np.isnan(lt).any():
        raise RuntimeError("uncovered positions in the true model")
    cum_true = np.concatenate([[0.0], np.cumsum(lt)])

    grid = mc.geometric_grid(8, t_max, ratio)
    traj = mc.ProfileTrajectory(d)
    traj.advance(x, grid)
    need = traj.profiles_needed() - cache.keys()
    if need:
        cache.update(depth_averaged_codelength_profiles(
            {p: p for p in need}, d=d, l_max=default_l_max(d),
            jobs=jobs, tables_source="universal",
            universal_path=store))
    traj.resolve(cache)
    rows = traj.losses
    return {"seed": seed,
            "t": [int(t) for (t, _, _) in rows],
            "l_F": [round(f, 4) for (_, f, _) in rows],
            "l_C": [round(c, 4) for (_, _, c) in rows],
            "l_true": [round(float(cum_true[t]), 4)
                       for (t, _, _) in rows],
            "new_profiles": len(need)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seeds", default="0-15")
    ap.add_argument("--prior-summary", type=Path, required=True)
    ap.add_argument("--prior-hint", type=float, default=None,
                    help="override the bracket-start median (for cells "
                         "whose summary is censored: the summary has no "
                         "crossings to take a median of)")
    ap.add_argument("--n-cap", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trajectories", action="store_true",
                    help="after the crossing search, record per-seed "
                         "trajectories (l_F, l_C, l_true) on a "
                         "geometric grid, into a sibling traj_* file")
    ap.add_argument("--traj-only", action="store_true",
                    help="record trajectories without redoing the "
                         "crossing search (t_max from the prior)")
    ap.add_argument("--traj-max", type=int, default=0,
                    help="trajectory horizon; 0 = 8x the cell median "
                         "(exact if available, else prior), capped at "
                         "n_cap")
    ap.add_argument("--traj-ratio", type=float, default=1.12)
    args = ap.parse_args()
    engage_anchor_mode(args.store)

    s = json.loads(args.prior_summary.read_text())
    ns = [v for v in (s.get("n_star_last") or []) if v]
    if args.prior_hint:
        prior_m = float(args.prior_hint)
    else:
        prior_m = float(np.median(ns)) if ns else 4096.0
    n_cap = args.n_cap or s.get("n_cap", 32 * args.d)

    seeds = parse_seeds(args.seeds)
    cache: dict = {}
    t0 = time.time()
    rows = []
    if not args.traj_only:
        for sd in seeds:
            r = exact_seed(args.d, args.alpha, args.L, sd, prior_m,
                           n_cap, args.jobs, args.store, cache)
            rows.append(r)
            print(f"  d={args.d} a={args.alpha} L={args.L} seed={sd}: "
                  f"n*=({r['last_neg']},{r['first_pos']}] first={r.get('first_cross')} "
                  f"evals={r['evals']} replays={r['replays']}"
                  + (f" [{r['note']}]" if r.get("note") else ""),
                  flush=True)
        out = {
            "d": args.d, "alpha": args.alpha, "L": args.L,
            "prior_median": prior_m, "n_cap": n_cap,
            "seconds": round(time.time() - t0, 1),
            "mode": "exact-token below prior 1e6, 0.1 percent above",
            "seeds": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        crossed = [r["first_pos"] for r in rows if r["first_pos"]]
        med = float(np.median(crossed)) if crossed else float("nan")
        print(f"cell done in {out['seconds']}s: median exact n* = {med:.0f} "
              f"({len(rows) - len(crossed)} censored; prior {prior_m:.0f}) "
              f"-> {args.out}")

    if args.trajectories or args.traj_only:
        crossed = [r["first_pos"] for r in rows if r.get("first_pos")]
        basis = float(np.median(crossed)) if crossed else prior_m
        t_max = args.traj_max or min(n_cap,
                                     max(1024, int(8 * basis)))
        t1 = time.time()
        trows = []
        for sd in seeds:
            tr = trajectory_seed(args.d, args.alpha, args.L, sd,
                                 t_max, args.traj_ratio, args.jobs,
                                 args.store, cache)
            trows.append(tr)
            print(f"  traj d={args.d} a={args.alpha} L={args.L} "
                  f"seed={sd}: {len(tr['t'])} points to {t_max}, "
                  f"R_F={tr['l_F'][-1] - tr['l_true'][-1]:.1f} "
                  f"R_C={tr['l_C'][-1] - tr['l_true'][-1]:.1f} bits "
                  f"({tr['new_profiles']} new profiles)", flush=True)
        tout = {
            "d": args.d, "alpha": args.alpha, "L": args.L,
            "t_max": int(t_max), "ratio": args.traj_ratio,
            "seconds": round(time.time() - t1, 1),
            "note": ("l_F, l_C: deployed codes, ProfileTrajectory "
                     "conventions (l_C charges log2 d for the first "
                     "token); l_true: generator code length "
                     "-log2 q(x1) - sum log2 T(x_{s+1}|x_s)"),
            "seeds": trows,
        }
        name = args.out.name
        tpath = (args.out.with_name(name.replace("exact_", "traj_"))
                 if "exact_" in name
                 else args.out.with_suffix(".traj.json"))
        tpath.parent.mkdir(parents=True, exist_ok=True)
        tpath.write_text(json.dumps(tout, indent=1) + "\n")
        print(f"trajectories done in {tout['seconds']}s -> {tpath}")


if __name__ == "__main__":
    main()
