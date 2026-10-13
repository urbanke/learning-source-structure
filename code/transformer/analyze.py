"""Compare a transformer with the two mechanisms on the evaluation sets.

    python3 analyze.py --refs refs --eval eval/d64 --out analysis/d64
For every setting with both a reference file and a transformer file: the mean loss per position of the transformer,
the unigram, the copier and the learner (windowed on a geometric grid of positions); the transformer's implied
weight on copying, the w in [0, 1] that minimises the KL divergence from the transformer's prediction to the mixture
w * copier + (1 - w) * unigram, averaged over sequences at each stored position; and the crossing estimates: the
position from which the transformer's windowed loss stays below the unigram's, the position from which its
implied weight stays above one half, and the same for the learner's weight. Writes summary.json and one figure
per setting.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = np.log(2.0)


def windows(n, per_decade=12):
    """Geometric position windows [lo, hi) covering 1..n (1-based positions), at least one position each."""
    edges = np.unique(np.round(np.logspace(0, np.log10(n), int(per_decade * np.log10(n)) + 1)).astype(int))
    edges = np.append(edges, n + 1)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def windowed(curve, wins):
    """curve: (n,) per-position mean (0-based index = 1-based position - 1)."""
    return np.array([curve[lo - 1:hi - 1].mean() for lo, hi in wins])


def implied_weight(pT, pF, pC, grid=401):
    """The w minimising mean_seq KL(pT || w pC + (1-w) pF), for arrays (N, d) at one position."""
    ws = np.linspace(0, 1, grid)
    pT = pT.astype(np.float64); pF = pF.astype(np.float64); pC = pC.astype(np.float64)
    ent = (pT * np.log(np.clip(pT, 1e-300, None))).sum(1)                     # sum p log p
    best = None
    for w in ws:
        mix = w * pC + (1 - w) * pF
        kl = (ent - (pT * np.log(np.clip(mix, 1e-300, None))).sum(1)).mean() / LN2
        if best is None or kl < best[1]:
            best = (w, kl)
    return best


def last_crossing(values, threshold, above=True):
    """1-based index of the first element from which values stay > threshold (or < if not above); None if never."""
    ok = values > threshold if above else values < threshold
    if not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return i + 1


def analyse_one(ref, ev, out_png, per_decade=12):
    assert str(ref["md5"]) == str(ev["md5"]), "the transformer and the references were evaluated on different sequences"
    lT = -ev["logp_true"].astype(np.float64) / LN2                              # (N, n) bits
    lF = ref["lF"].astype(np.float64); lC = ref["lC"].astype(np.float64); wC = ref["wC"].astype(np.float64)
    lQ = -np.log2(wC * 2.0 ** (-lC) + (1 - wC) * 2.0 ** (-lF))                 # the learner's loss
    n = lT.shape[1]; wins = windows(n, per_decade); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins])
    curves = {k: windowed(v.mean(0), wins) for k, v in dict(transformer=lT, unigram=lF, copier=lC, learner=lQ).items()}
    se = {k: windowed(v.mean(0), wins) * 0 + np.array([v[:, lo - 1:hi - 1].mean(1).std() / np.sqrt(v.shape[0]) for lo, hi in wins])
          for k, v in dict(transformer=lT, unigram=lF, copier=lC).items()}
    # implied weight on copying at the stored positions
    pos = ref["positions"]; assert np.array_equal(pos, ev["positions"])
    pT = np.exp(ev["logp_full"].astype(np.float64)); pT /= pT.sum(-1, keepdims=True)
    pF = ref["pF"].astype(np.float64); pC = ref["pC"].astype(np.float64)
    nf = min(pT.shape[0], pF.shape[0])
    wI = np.array([implied_weight(pT[:nf, j], pF[:nf, j], pC[:nf, j])[0] for j in range(len(pos))])
    klF = np.array([((pT[:nf, j] * (np.log(np.clip(pT[:nf, j], 1e-300, None)) - np.log(np.clip(pF[:nf, j], 1e-300, None)))).sum(1)).mean() / LN2 for j in range(len(pos))])
    klC = np.array([((pT[:nf, j] * (np.log(np.clip(pT[:nf, j], 1e-300, None)) - np.log(np.clip(pC[:nf, j], 1e-300, None)))).sum(1)).mean() / LN2 for j in range(len(pos))])
    wL = wC.mean(0)[pos]                                                       # the learner's mean weight at the same positions
    # crossings (1-based positions)
    gap_win = curves["unigram"] - curves["transformer"]
    i = last_crossing(gap_win, 0.0, above=True)
    cross_loss = float(np.sqrt(wins[i - 1][0] * (wins[i - 1][1] - 1))) if i else None
    j = last_crossing(wI, 0.5, above=True); cross_w = float(pos[j - 1] + 1) if j else None
    k = last_crossing(wL, 0.5, above=True); cross_learner = float(pos[k - 1] + 1) if k else None
    gapQ = (lF - lC).mean(0)                                                   # the expected gap on these sequences
    k2 = last_crossing(gapQ, 0.0, above=True); cross_gap = float(k2) if k2 else None
    summary = dict(alpha=float(ref["alpha"]), ls=int(ref["ls"]), d=int(ref["d"]), step=int(ev["step"]), n_seq=int(lT.shape[0]), n_full=int(nf),
                   mean_bits=dict(transformer=float(lT.mean()), unigram=float(lF.mean()), copier=float(lC.mean()), learner=float(lQ.mean())),
                   crossing=dict(transformer_loss_below_unigram=cross_loss, transformer_weight_above_half=cross_w,
                                 learner_weight_above_half=cross_learner, expected_gap_positive=cross_gap),
                   windows=[[lo, hi - 1] for lo, hi in wins], curves={k: v.tolist() for k, v in curves.items()},
                   positions=(pos + 1).tolist(), implied_weight=wI.tolist(), learner_weight=wL.tolist(), kl_to_unigram=klF.tolist(), kl_to_copier=klC.tolist())
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    for k, c in (("unigram", "C0"), ("copier", "C3"), ("learner", "0.4"), ("transformer", "C2")):
        ax.plot(mid, curves[k], "-" if k != "learner" else "--", color=c, label=k, lw=1.2 if k == "transformer" else 1)
    ax.set_xscale("log"); ax.set_xlabel("position $t$"); ax.set_ylabel("loss [bits]"); ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"$d={summary['d']}$, $\\alpha={summary['alpha']:g}$, $L_s={summary['ls']}$", fontsize=9)
    ax = axes[1]
    ax.plot(pos + 1, wI, "-", color="C2", label="transformer, implied")
    ax.plot(pos + 1, wL, "--", color="0.4", label="learner $w_C$")
    ax.axhline(0.5, color="0.7", lw=0.8); ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("position $t$"); ax.set_ylabel("weight on copying"); ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    ax.plot(pos + 1, klF, "-", color="C0", label="KL to unigram"); ax.plot(pos + 1, klC, "-", color="C3", label="KL to copier")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("position $t$"); ax.set_ylabel("KL [bits]"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = []
    for f in sorted(glob.glob(os.path.join(args.eval, "*_step*.npz"))):
        tag = re.sub(r"_step\d+\.npz$", "", os.path.basename(f))
        rf = os.path.join(args.refs, tag + ".npz")
        if not os.path.exists(rf):
            print("no references for", tag); continue
        ev = np.load(f); ref = np.load(rf)
        s = analyse_one(ref, ev, os.path.join(args.out, os.path.basename(f).replace(".npz", ".png")))
        results.append(s)
        c = s["crossing"]
        print(f"{tag} step {s['step']}: bits/token T {s['mean_bits']['transformer']:.3f} F {s['mean_bits']['unigram']:.3f} C {s['mean_bits']['copier']:.3f} Q {s['mean_bits']['learner']:.3f}; "
              f"crossings: loss {c['transformer_loss_below_unigram']}, implied weight {c['transformer_weight_above_half']}, learner weight {c['learner_weight_above_half']}, expected gap {c['expected_gap_positive']}")
    json.dump(results, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
