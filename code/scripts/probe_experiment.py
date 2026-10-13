"""Probe loss versus total loss: the two faces of the takeover.

For one configuration and seed, replay the chain and compute the
LEARNER's exact per-position loss  m_t = M(t) - M(t-1)  with
M(t) = -log2( (1/2) 2^{-l_F(t)} + (1/2) 2^{-l_C(t)} ),
maintaining both strategies' cumulative bits incrementally (their
sequential predictions telescope, so per-position values are exact).
A position t >= 2 is a PROBE position if its context x_{t-1} has
already been seen as a context (the copier's table is nonempty).
The paper's claim (mechanisms.tex 2.4): the loss on probe positions
falls sharply around the takeover while the average over all
positions stays smooth.

Writes per-seed JSON with per-window averages of total / probe /
non-probe loss and the posterior weight, on a geometric grid.

    python code/scripts/probe_experiment.py --d 1024 --alpha 1.0 --L 3 \
        --seed 0 --n 40960 --store <sealed store> --out output/probe/
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from takeover_experiment import engage_anchor_mode


class ProfileTracker:
    """Canonical count profile of one growing count table, maintained
    incrementally: O(distinct count values) per update."""

    def __init__(self):
        self.of_counts: Counter = Counter()   # count value -> #symbols

    def bump(self, old: int) -> tuple:
        """Symbol goes old -> old+1; returns the new profile tuple."""
        if old:
            self.of_counts[old] -= 1
            if not self.of_counts[old]:
                del self.of_counts[old]
        self.of_counts[old + 1] += 1
        return self.key()

    def key(self) -> tuple:
        out = []
        for v in sorted(self.of_counts, reverse=True):
            out.extend([v] * self.of_counts[v])
        return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--store", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    engage_anchor_mode(args.store)

    import mech_common as mc
    from lsa.codelength import (default_l_max,
                                depth_averaged_codelength_profiles)

    d, n = args.d, args.n
    l_max = default_l_max(d)
    rng = np.random.default_rng(
        [args.seed, d, args.L, int(round(args.alpha * 100)), 7])
    chain = mc.ChainSampler(d, args.alpha, args.L, rng)
    x = chain.extend_to(n)
    chain.release_rules()

    # pass 1: record, per position, the profile keys whose difference
    # gives that position's F and C loss, plus the probe flag
    f_track = ProfileTracker()
    c_track: dict[int, ProfileTracker] = {}
    c_counts: dict[int, Counter] = {}
    fkeys = [f_track.bump(0)]           # position 0: first token
    ckeys: list[tuple | None] = [None]  # first token: uniform, log2 d
    probe = [False]
    counts = np.zeros(d, dtype=np.int64)
    counts[x[0]] = 1
    for t in range(1, n):
        tok, a = int(x[t]), int(x[t - 1])
        fkeys.append(f_track.bump(int(counts[tok])))
        counts[tok] += 1
        tr = c_track.get(a)
        if tr is None:
            tr = c_track[a] = ProfileTracker()
            c_counts[a] = Counter()
        probe.append(bool(c_counts[a]))
        ckeys.append(tr.bump(c_counts[a][tok]))
        c_counts[a][tok] += 1

    need = set(fkeys) | {k for k in ckeys if k is not None}
    print(f"seed {args.seed}: {n} positions, {len(need)} distinct "
          f"profiles", flush=True)
    ev = depth_averaged_codelength_profiles(
        {p: p for p in need}, d=d, l_max=l_max, jobs=args.jobs,
        tables_source="universal", universal_path=args.store)
    bits = {p: -ev[p].log2_q_avg for p in need}
    bits[()] = 0.0

    # pass 2: exact per-position mixture losses
    LN2 = math.log(2.0)
    lF = lC = 0.0
    M_prev = 0.0
    prev_f: tuple = ()
    prev_c: dict[int, tuple] = {}
    cps = mc.geometric_grid(64, n)
    ci = 0
    win = {"t0": 0, "tot": 0.0, "pr": 0.0, "npr": 0.0, "n_pr": 0}
    rows = []
    for t in range(n):
        lF += bits[fkeys[t]] - bits[prev_f]
        prev_f = fkeys[t]
        if t == 0:
            lC += math.log2(d)
        else:
            a = int(x[t - 1])
            lC += bits[ckeys[t]] - bits[prev_c.get(a, ())]
            prev_c[a] = ckeys[t]
        M = -(np.logaddexp(-lF * LN2, -lC * LN2) / LN2 - 1.0)
        m_t = M - M_prev
        M_prev = M
        win["tot"] += m_t
        if probe[t]:
            win["pr"] += m_t
            win["n_pr"] += 1
        else:
            win["npr"] += m_t
        if ci < len(cps) and t + 1 == cps[ci]:
            span = t + 1 - win["t0"]
            n_pr = win["n_pr"]
            rows.append({
                "t": t + 1,
                "loss_total": win["tot"] / span,
                "loss_probe": (win["pr"] / n_pr) if n_pr else None,
                "loss_nonprobe": (win["npr"] / (span - n_pr))
                                 if span > n_pr else None,
                "probe_fraction": n_pr / span,
                "w_C": (float(1.0 / (1.0 + 2.0 ** (lF - lC)))
                        if abs(lF - lC) < 50 else
                        (0.0 if lF > lC else 1.0)),
                "Lambda": lF - lC,
            })
            win = {"t0": t + 1, "tot": 0.0, "pr": 0.0, "npr": 0.0,
                   "n_pr": 0}
            ci += 1

    # consistency: totals must match the ledger path to 1e-9
    traj = mc.ProfileTrajectory(d)
    traj.advance(x, [n])
    missing = traj.profiles_needed() - set(bits)
    if missing:
        ref = depth_averaged_codelength_profiles(
            {p: p for p in missing}, d=d, l_max=l_max, jobs=args.jobs,
            tables_source="universal", universal_path=args.store)
        bits.update({p: -ref[p].log2_q_avg for p in ref})
    traj.resolve(_wrap(bits))
    (tt, rF, rC) = traj.losses[-1]
    # 1e-6 bits over ~1e5 incremental additions (float accumulation)
    assert abs(rF - lF) < 1e-6 and abs(rC - lC) < 1e-6, (rF, lF, rC, lC)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "d": d, "alpha": args.alpha, "L": args.L, "seed": args.seed,
        "n": n, "l_max": l_max, "windows": rows,
        "final": {"l_F": lF, "l_C": lC, "Lambda": lF - lC},
    }
    f = args.out / (f"probe_d{d}_a{args.alpha}_L{args.L}"
                    f"_seed{args.seed}.json")
    f.write_text(json.dumps(payload, indent=1))
    print(f"consistency vs ledger OK (1e-9); wrote {f}", flush=True)


def _wrap(bits):
    class _E:
        __slots__ = ("log2_q_avg",)

        def __init__(self, v):
            self.log2_q_avg = v

    return {p: _E(-b) for p, b in bits.items()}


if __name__ == "__main__":
    main()
