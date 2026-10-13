"""Diagnostics for the copier: per-context regret at the actual visits vs the building block.

residual_a = [cost_a - K_a H2(T_a)] - [BB(K_a) - K_a * hbar]   with hbar = E H2(row) (bits),
summed over contexts; its mean tests the building block + (row, visit) independence, not the visit law.
Also: bin by K_a.
"""
import sys, json
import numpy as np
from table_mc import BBInterp


def run(simfile, bbfile, hbar_bits=None):
    s = json.load(open(simfile)); BB = BBInterp(bbfile)
    ns = s["checkpoints"]; R = s["R"]; d = s["d"]
    K = np.array(s["K"], dtype=float)          # R x J x d
    costA = np.array(s["costA"])               # R x J x d
    rowH = np.array(s["rowH"])                 # R x d
    if hbar_bits is None:
        hbar_bits = rowH.mean()
    print(f"{simfile}: d={d} alpha={s['alpha']} Ls={s['Ls']} R={R}; mean row entropy {hbar_bits:.4f} bits")
    for j, n in enumerate(ns):
        Kj = K[:, j, :]; cj = costA[:, j, :]
        regret = cj - Kj * rowH[:, None, :][:, 0, :]          # R x d, measured per-context regret
        pred = BB(Kj) - Kj * hbar_bits                         # R x d, predicted regret at the actual visits
        # note: predicted uses hbar (average row entropy), measured uses the actual row entropy;
        # to compare fairly, add back (rowH - hbar) K: residual = cost - BB(K) - K*(rowH - rowH) ... use plain difference of costs:
        resid = cj - BB(Kj)                                    # cost minus building block at actual visits
        tot = resid.sum(axis=1)
        print(f"  n={n:5d}: sum_a [cost_a - BB(K_a)] mean {tot.mean():+8.2f} bits (SE {tot.std(ddof=1)/np.sqrt(R):5.2f});  "
              f"total copier regret meas {(cj.sum(1) - (Kj * rowH).sum(1)).mean():8.1f}")
        # bins by K
        edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1e9]
        line = "     by K bins: "
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (Kj >= lo) & (Kj < hi)
            if m.sum() > 0:
                line += f"[{int(lo)},{int(hi) if hi < 1e9 else 'inf'}): {resid[m].mean():+.3f} (n={m.sum()}, per-path {resid[m].sum()/R:+.2f}) "
        print(line)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
