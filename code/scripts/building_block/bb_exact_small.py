"""Exact building block B(k) for small k, by the exact law of the count profile.

By the Poisson representation, k draws from a random row have the profile law of independent counts
N_b ~ p_{Ls}(.; s q_b) conditioned on S = k, mixed over the rate s with density g_k(s) = k P_s(k)/s.
For a fixed rate the profile law (the multiset of positive counts) is computed exactly by a dynamic
programme over the symbols; the states are the partitions of the integers <= K0.  With the exact LSA cost
of every profile this gives E[c | S = k, s] exactly, and the rate integral (trapezoid in ln s) gives B(k).
Usage: python3 bb_exact_small.py d alpha Ls Lmax K0 out.json [xstep [mlo mup]]   (rate grid ln s in [-mlo, ln K0 + mup])
"""
import sys, json, time
import numpy as np
import lsa_cost as C
import bb
import lsa_fast  # noqa: table evaluator


def partitions_upto(K0):
    """all partitions (as descending tuples) of the integers 0..K0"""
    parts = [()]
    for k in range(1, K0 + 1):
        def gen(n, maxpart):
            if n == 0:
                yield ()
                return
            for p in range(min(n, maxpart), 0, -1):
                for rest in gen(n - p, p):
                    yield (p,) + rest
        parts += list(gen(k, k))
    return parts


def transition_maps(parts, K0):
    """for each r = 1..K0: arrays (from, to) with to = index of the partition from + {r} when the sum stays <= K0"""
    index = {p: i for i, p in enumerate(parts)}
    sums = np.array([sum(p) for p in parts])
    maps = {}
    for r in range(1, K0 + 1):
        fr = []; to = []
        for i, p in enumerate(parts):
            if sums[i] + r <= K0:
                newp = tuple(sorted(p + (r,), reverse=True))
                fr.append(i); to.append(index[newp])
        maps[r] = (np.array(fr, dtype=np.int64), np.array(to, dtype=np.int64))
    return index, sums, maps


def profile_law(pm, parts, sums, maps, K0):
    """pm[b, r] = law of N_b at r = 0..K0 (independent symbols); returns Pr(multiset of positive counts = p) per partition p"""
    dist = np.zeros(len(parts)); dist[0] = 1.0
    for b in range(pm.shape[0]):
        new = dist * pm[b, 0]
        for r in range(1, K0 + 1):
            fr, to = maps[r]
            new[to] += dist[fr] * pm[b, r]
        dist = new
    return dist


def exact_costs(lsa, parts, sums, d):
    """exact LSA cost (bits) of every partition, as the count profile of d symbols"""
    costs = np.zeros(len(parts))
    for i, p in enumerate(parts):
        k = sums[i]
        if k == 0:
            continue
        prof = {}
        for r in p:
            prof[r] = prof.get(r, 0) + 1
        prof[0] = d - len(p)
        costs[i] = lsa.cost_bits(prof, int(k))
    return costs


def run(d, alpha, Ls, Lmax, K0, out, xstep=0.05, mlo=4.0, mup=4.0):
    t0 = time.time()
    q = C.zipf(d, alpha); lsa = bb.LSAx(Lmax)
    parts = partitions_upto(K0); index, sums, maps = transition_maps(parts, K0)
    print(f"d={d} alpha={alpha} Ls={Ls} K0={K0}: {len(parts)} partitions; maps built {time.time()-t0:.0f}s", flush=True)
    import os
    cfile = f"bbx_costs_d{d}_L{Lmax}_K{K0}.npy"          # the costs depend on the code (d, Lmax) only, not on the source
    if os.path.exists(cfile):
        costs = np.load(cfile)
    else:
        costs = exact_costs(lsa, parts, sums, d); np.save(cfile, costs)
    print(f"exact costs of all profiles: {time.time()-t0:.0f}s", flush=True)
    # absolute rate grid in ln s covering the windows of every k <= K0 (s from e^{-mlo} to K0 e^{mup}; margins 4 by default,
    # larger for deep sources and strong tilts, where the rate law given k is wide and shifted to s > k)
    lo = np.log(1.0) - mlo; hi = np.log(K0) + mup
    lns = np.arange(lo, hi + 1e-9, xstep)
    ks = np.arange(1, K0 + 1)
    num = np.zeros(K0 + 1); den = np.zeros(K0 + 1)
    for j, u in enumerate(lns):
        s = np.exp(u)
        pm = bb.pmfs_at_scale(lsa, q, Ls, s, K0)          # d x (K0+1)
        dist = profile_law(pm, parts, sums, maps, K0)
        for k in ks:
            sel = sums == k
            Pk = dist[sel].sum()                              # P_s(k)
            if Pk <= 0:
                continue
            w = k * Pk * xstep                                # g_k(s) ds = k P_s(k) d ln s
            num[k] += w * float(dist[sel] @ costs[sel]) / Pk
            den[k] += w
        if j % 20 == 0:
            print(f"  rate {j+1}/{len(lns)}: ln s = {u:.2f}; {time.time()-t0:.0f}s", flush=True)
    res = dict(d=d, alpha=alpha, Ls=Ls, Lmax=Lmax, K0=K0, xstep=xstep, margins=[mlo, mup], rows=[])
    for k in ks:
        res["rows"].append(dict(k=int(k), cost=float(num[k] / den[k]), mass=float(den[k])))
        print(f"k={k:3d}: B(k) = {num[k]/den[k]:10.4f}  (rate mass {den[k]:.5f})", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")
    return res


if __name__ == "__main__":
    d, alpha, Ls, Lmax, K0 = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = sys.argv[6]; xstep = float(sys.argv[7]) if len(sys.argv) > 7 else 0.05
    mlo = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0; mup = float(sys.argv[9]) if len(sys.argv) > 9 else 4.0
    run(d, alpha, Ls, Lmax, K0, out, xstep, mlo, mup)
