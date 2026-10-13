"""The transformer against the two mechanisms and the four extra references, per position, on the same sequences.
    python3 plot_extra.py --refs refs/d32 --eval eval/d32 --tag d32_a1_L2 --step 100000 --out fig.png"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import windows, windowed

LN2 = np.log(2.0)
NAMES = [("unigram", "C0"), ("copier", "C3"), ("joint pairs", "C1"), ("true depth", "C4"), ("backoff", "C6"), ("product model, Zipf estimate", "C8"), ("learner with product model", "C9"), ("oracle base law", "C5"), ("transformer", "C2")]


def load(refs, ev, tag, step):
    R = np.load(os.path.join(refs, f"{tag}.npz")); X = np.load(os.path.join(refs, f"extra_{tag}.npz")); E = np.load(os.path.join(ev, f"{tag}_step{step}.npz"))
    assert str(R["md5"]) == str(E["md5"]) == str(X["md5"])
    N = int(X["n_seq"])
    L = dict(unigram=R["lF"][:N], copier=R["lC"][:N])
    L["joint pairs"] = X["lJ"]; L["true depth"] = X["lD"]; L["backoff"] = X["lB"]; L["oracle base law"] = X["lO"]
    L["product model, Zipf estimate"] = X["lZ"] if "lZ" in X.files else X["lB"] * np.nan
    L["transformer"] = -E["logp_true"][:N] / LN2
    L = {k: v.astype(np.float64) for k, v in L.items()}
    # the learner of the paper with the product-model copier in place of the copier: equal prior weights, posterior
    # weights from the cumulative losses, loss of the mixture from the two probabilities of the true token
    lF, lZ = L["unigram"], L["product model, Zipf estimate"]
    gap = np.cumsum(lF, axis=1) - np.cumsum(lZ, axis=1)
    w = np.empty_like(gap); w[:, 0] = 0.5; w[:, 1:] = 1.0 / (1.0 + 2.0 ** (-gap[:, :-1]))
    L["learner with product model"] = -np.log2(w * 2.0 ** (-lZ) + (1 - w) * 2.0 ** (-lF))
    return L, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--step", type=int, required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L, N = load(args.refs, args.eval, args.tag, args.step)
    n = L["transformer"].shape[1]; wins = windows(n, 8); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins])
    cur = {k: windowed(v.mean(0), wins) for k, v in L.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for k, c in NAMES:
        ax.plot(mid, cur[k], "-", color=c, lw=1.6 if k == "transformer" else 1.0, label=k)
    ax.set_xscale("log"); ax.set_xlabel("position $t$"); ax.set_ylabel("loss [bits]"); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{args.tag}, step {args.step}, {N} sequences", fontsize=9)
    ax = axes[1]
    for k, c in NAMES:
        if k != "transformer":
            ax.plot(mid, cur[k] - cur["transformer"], "-", color=c, lw=1.0, label=k)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("position $t$"); ax.set_ylabel("loss minus the transformer's [bits]")
    ax.set_ylim(-0.15, 0.8); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print("mean bits/token over the window:", {k: round(float(v.mean()), 3) for k, v in L.items()})
    for lo, hi in ((1, 8), (8, 32), (32, 128), (128, 512), (512, 1025)):
        print(f"  positions {lo}-{hi-1}: " + "  ".join(f"{k} {L[k][:, lo-1:hi-1].mean():.3f}" for k, _ in NAMES))


if __name__ == "__main__":
    main()
