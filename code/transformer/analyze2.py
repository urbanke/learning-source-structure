"""The crossings: the paper's n*_d (copier against unigram), the informed crossing n*_product (product-model copier
against unigram), both measured on the evaluation sequences as the last zero of the mean cumulative gap, and the
transformer's switch, the position from which its implied weight on the product-model copier (the mixture of the
unigram's and the product-model copier's predictions closest to its own, in KL) stays above one half. Also the
mean losses of everything. Usage: python3 analyze2.py --out tf_analysis2   (reads tf_refs, tf_refs2, tf_eval; see the
layout in figures_sec4.py)"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = np.log(2.0)
RUNS = {32: ("d32", 100000), 64: ("d64_seed1", 100000), 128: ("d128", 100000)}


def last_zero(gap_mean):
    """1-based position from which the (mean cumulative) gap stays positive; None if not by the end."""
    if gap_mean[-1] <= 0:
        return None
    i = len(gap_mean) - 1
    while i > 0 and gap_mean[i - 1] > 0:
        i -= 1
    return i + 1


def implied_weights(pT, pA, pB, grid=201):
    """For arrays (N, P, d): at each position the w minimising mean_N KL(pT || w pB + (1-w) pA)."""
    ws = np.linspace(0, 1, grid); P = pT.shape[1]; out = np.empty(P)
    ent = (pT * np.log(np.clip(pT, 1e-300, None))).sum(-1)                       # (N, P)
    for j in range(P):
        best = None
        for w in ws:
            mix = w * pB[:, j] + (1 - w) * pA[:, j]
            kl = (ent[:, j] - (pT[:, j] * np.log(np.clip(mix, 1e-300, None))).sum(-1)).mean()
            if best is None or kl < best[1]:
                best = (w, kl)
        out[j] = best[0]
    return out


def crossing_of_weight(w, positions, smooth=5):
    """1-based position from which the (smoothed) weight stays above one half; None if never."""
    if smooth > 1:
        k = np.ones(smooth) / smooth; w = np.convolve(w, k, mode="same")
    if w[-1] <= 0.5:
        return None
    i = len(w) - 1
    while i > 0 and w[i - 1] > 0.5:
        i -= 1
    return int(positions[i] + 1)


def analyse(d, a, L):
    run, step = RUNS[d]; tag = f"d{d}_a{a:g}_L{L}"
    fr = f"tf_refs/d{d}/{tag}.npz"; fx = f"tf_refs2/d{d}/extra_{tag}.npz"; fe = f"tf_eval/{run}/{tag}_step{step}.npz"
    if not (os.path.exists(fr) and os.path.exists(fx) and os.path.exists(fe)):
        return None
    R = np.load(fr); X = np.load(fx); E = np.load(fe)
    assert str(R["md5"]) == str(X["md5"]) == str(E["md5"])
    N = int(X["n_seq"]); NF = int(X["n_full"])
    lF = R["lF"][:N].astype(float); lC = R["lC"][:N].astype(float); lZ = X["lZ"].astype(float); lO = X["lO"].astype(float)
    lT = -E["logp_true"][:N].astype(float) / LN2
    gapC = np.cumsum(lF - lC, axis=1).mean(0); gapZ = np.cumsum(lF - lZ, axis=1).mean(0)
    # the learner with the product model: its loss and its mean weight
    g = np.cumsum(lF, axis=1) - np.cumsum(lZ, axis=1); w = np.empty_like(g); w[:, 0] = 0.5; w[:, 1:] = 1 / (1 + 2.0 ** (-g[:, :-1]))
    lQZ = -np.log2(w * 2.0 ** (-lZ) + (1 - w) * 2.0 ** (-lF))
    pos = R["positions"]; assert np.array_equal(pos, E["positions"]) and np.array_equal(pos, X["positions"])
    nf = min(NF, E["logp_full"].shape[0], R["pF"].shape[0])
    pT = np.exp(E["logp_full"][:nf].astype(float)); pT /= pT.sum(-1, keepdims=True)
    pF = R["pF"][:nf].astype(float); pZ = X["pZ"][:nf].astype(float); pC = R["pC"][:nf].astype(float)
    wZ = implied_weights(pT, pF, pZ); wCw = implied_weights(pT, pF, pC)
    res = dict(d=d, alpha=a, ls=L, n_seq=N, n_full=nf,
               bits=dict(transformer=float(lT.mean()), unigram=float(lF.mean()), copier=float(lC.mean()), product=float(lZ.mean()), oracle=float(lO.mean()), learner_product=float(lQZ.mean())),
               n_star_paper=last_zero(gapC), n_star_product=last_zero(gapZ),
               transformer_switch_vs_product=crossing_of_weight(wZ, pos), transformer_switch_vs_copier=crossing_of_weight(wCw, pos),
               learner_product_switch=crossing_of_weight(w.mean(0)[pos], pos, smooth=1),
               positions=(pos + 1).tolist(), implied_weight_product=wZ.tolist(), implied_weight_copier=wCw.tolist(), learner_product_weight=w.mean(0)[pos].tolist())
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="tf_analysis2"); args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for d in RUNS:
        for a in (0.75, 1.0, 1.25):
            for L in (2, 3, 4):
                r = analyse(d, a, L)
                if r:
                    rows.append(r)
                    b = r["bits"]
                    print(f"d={d:3d} a={a:<4} L={L}: bits T {b['transformer']:.3f} F {b['unigram']:.3f} C {b['copier']:.3f} Z {b['product']:.3f} O {b['oracle']:.3f} QZ {b['learner_product']:.3f} | "
                          f"n*_paper {r['n_star_paper']}  n*_product {r['n_star_product']}  learner(Z) switch {r['learner_product_switch']}  transformer switch vs Z {r['transformer_switch_vs_product']} (vs C {r['transformer_switch_vs_copier']})")
    json.dump(rows, open(os.path.join(args.out, "crossings.json"), "w"), indent=1)
    # figure: crossings against d, one panel per L_s, colour per alpha
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    cols = {0.75: "C1", 1.0: "C2", 1.25: "C3"}
    for ax, L in zip(axes, (2, 3, 4)):
        for a in (0.75, 1.0, 1.25):
            sel = [r for r in rows if r["ls"] == L and r["alpha"] == a]
            for key, mk, lab in (("n_star_paper", "o-", "$n^*_d$ (paper)"), ("n_star_product", "s--", "$n^*$ product model"), ("transformer_switch_vs_product", "^:", "transformer switch")):
                xs = [r["d"] for r in sel if r[key]]; ys = [r[key] for r in sel if r[key]]
                if xs:
                    ax.plot(xs, ys, mk, color=cols[a], ms=5, lw=1, label=f"{lab}, $\\alpha={a:g}$" if L == 2 else None)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(f"$L_s={L}$"); ax.set_xlabel("$d$"); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("position"); axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "crossings.png"), dpi=120)


if __name__ == "__main__":
    main()
