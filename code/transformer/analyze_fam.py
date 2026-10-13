"""Figures and tables for the source families (notes/families_plan.md): regret curves against the position for the
trained models and the exact reference predictors, per test set and per training family (cross-training), the
loss per position and the crossings for the chain families. Robust to missing files (skips what is not there).

Layout (all relative to the working directory):
    fam_eval/<run>/<tag>_step<step>.npz     eval_model.py outputs (tag = family spec with ':' -> '_', '=' removed, ',' -> '-')
    refs_fam/<tag>.npz                       refs_family.py outputs for the family sets
    tf_refs_2048/d<d>/<tag>.npz              make_refs.py outputs for the paper's source (2048 sequences)
Writes fam_figs/*.pdf|png and fam_res/*.json|tex.   Usage: python3 analyze_fam.py [--eval fam_eval]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 6.5})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import windows as _windows, windowed
LN2 = np.log(2.0)
OUT_FIG = "fam_figs"; OUT_RES = "fam_res"; REFS = "refs_fam"; REFS2048 = "tf_refs_2048"


def windows(n, per_decade=4, min_last=8):
    w = _windows(n, per_decade)
    if len(w) > 1 and w[-1][1] - w[-1][0] < min_last:
        w = w[:-2] + [(w[-2][0], w[-1][1])]
    return w


def load_eval(evaldir, run, tag):
    fs = sorted(glob.glob(f"{evaldir}/{run}/{tag}_step*.npz"), key=lambda f: int(f.rsplit("step", 1)[1].split(".")[0]))
    return np.load(fs[-1]) if fs else None


def cum_regret(loss, h):
    """Mean over sequences of the cumulative regret at every position: (n,)."""
    return np.cumsum(loss.astype(float) - h.astype(float), axis=1).mean(0)


# ---- F1: binary --------------------------------------------------------------------------------------------------------
def binary(evaldir):
    runs = [("bin_twopoint", "trained on the two-point prior", "C3"), ("bin_uniform", "trained on the uniform prior", "C0"), ("bin_eightpoint", "trained on the eight-point prior", "C2")]
    sets = [("binary_twopoint", "test: two-point prior", "twopoint"), ("binary_uniform", "test: uniform prior", "laplace"), ("binary_eightpoint", "test: eight-point prior", "eightpoint"),
            ("binary_fixed0.5", "test: $\\theta=0.5$", None), ("binary_fixed0.2", "test: $\\theta=0.2$", None)]
    refs = {"twopoint": ("two-point Bayes", "C3"), "laplace": ("add-one (uniform Bayes)", "C0"), "eightpoint": ("eight-point Bayes", "C2"), "kt": ("KT (add-half)", "0.5")}
    avail = [(tag, title, bayes) for tag, title, bayes in sets if os.path.exists(f"{REFS}/{tag}.npz")]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(2.9 * len(avail), 2.7), sharey=False); table = {}
    for ax, (tag, title, bayes) in zip(np.atleast_1d(axes), avail):
        R = np.load(f"{REFS}/{tag}.npz"); h = R["h_true"]; n = h.shape[1]; pos = np.arange(1, n + 1); table[tag] = {}
        for key, (lab, c) in refs.items():
            if key in R.files:
                cr = cum_regret(R[key], h); table[tag][lab] = float(cr[-1])
                ax.plot(pos, cr, "--" if key != bayes else "-", color=c, lw=1.6 if key == bayes else 0.8, alpha=1.0 if key == bayes else 0.7, label=lab + (" (Bayes for this set)" if key == bayes else ""))
        for run, lab, c in runs:
            E = load_eval(evaldir, run, tag)
            if E is None or str(E["md5"]) != str(R["md5"]):
                continue
            lT = -E["logp_true"].astype(float) / LN2; cr = cum_regret(lT, h); table[tag][lab] = float(cr[-1])
            ax.plot(pos, cr, "-", color=c, lw=1.2, marker="o", ms=2.5, markevery=[3, 7, 15, 31, 63, 127, 255][: int(np.log2(n))], label="transformer " + lab)
        ax.set_xscale("log"); ax.set_xlabel("position $n$"); ax.set_title(title); ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("cumulative regret [bits]"); np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/fam_binary.pdf"); fig.savefig(f"{OUT_FIG}/fam_binary.png", dpi=130); plt.close(fig)
    json.dump(table, open(f"{OUT_RES}/fam_binary.json", "w"), indent=1)
    print("binary:", json.dumps({k: {kk: round(v, 2) for kk, v in vv.items()} for k, vv in table.items()}, indent=1))


# ---- F2: exponential family ----------------------------------------------------------------------------------------------
def expfam(evaldir):
    runs = [("expfam_k1", "trained on $k=1$", "C3"), ("expfam_k2", "trained on $k=2$", "C2"), ("expfam_k1_perm", "trained on $k=1$, relabelled", "C1")]
    sets = [("expfam_k1", "test: $k=1$, labels known", "bayes_k1"), ("expfam_k2", "test: $k=2$, labels known", "bayes_k2"), ("expfam_k1_perm", "test: $k=1$, relabelled", None)]
    refs = {"bayes_k1": ("Bayes, $k=1$, labels known", "C3"), "bayes_k2": ("Bayes, $k=2$, labels known", "C2"), "zipf_plugin": ("Zipf plug-in (labels unknown)", "C1"), "lsa_unigram": ("LSA unigram predictor", "C0"), "laplace_d": ("add-one over $d$ symbols", "0.5")}
    avail = [(tag, title, bayes) for tag, title, bayes in sets if os.path.exists(f"{REFS}/{tag}.npz")]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(3.0 * len(avail), 2.7)); table = {}
    for ax, (tag, title, bayes) in zip(np.atleast_1d(axes), avail):
        R = np.load(f"{REFS}/{tag}.npz"); h = R["h_true"]; n = h.shape[1]; pos = np.arange(1, n + 1); table[tag] = {}
        for key, (lab, c) in refs.items():
            if key in R.files:
                cr = cum_regret(R[key], h); table[tag][lab] = float(cr[-1])
                ax.plot(pos, cr, "-" if key == bayes else "--", color=c, lw=1.6 if key == bayes else 0.8, label=lab + (" (Bayes for this set)" if key == bayes else ""))
        for run, lab, c in runs:
            E = load_eval(evaldir, run, tag)
            if E is None or str(E["md5"]) != str(R["md5"]):
                continue
            lT = -E["logp_true"].astype(float) / LN2; cr = cum_regret(lT, h); table[tag][lab] = float(cr[-1])
            ax.plot(pos, cr, "-", color=c, lw=1.2, marker="o", ms=2.5, markevery=[7, 31, 127, 511, 1023][: int(np.log2(n)) // 2], label="transformer " + lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("position $n$"); ax.set_title(title); ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("cumulative regret [bits]"); np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/fam_expfam.pdf"); fig.savefig(f"{OUT_FIG}/fam_expfam.png", dpi=130); plt.close(fig)
    json.dump(table, open(f"{OUT_RES}/fam_expfam.json", "w"), indent=1)
    print("expfam:", json.dumps({k: {kk: round(v, 1) for kk, v in vv.items()} for k, vv in table.items()}, indent=1))


# ---- F3: the product chain ---------------------------------------------------------------------------------------------
def last_zero(g):
    if g[-1] <= 0:
        return None
    i = len(g) - 1
    while i > 0 and g[i - 1] > 0:
        i -= 1
    return i + 1


def product(evaldir):
    sets = [("product_8x8_L2-2", "$8\\times8$, $L_{\\mathrm{s}}=2,2$"), ("product_8x8_L4-4", "$8\\times8$, $L_{\\mathrm{s}}=4,4$"), ("product_8x8_L2-6", "$8\\times8$, $L_{\\mathrm{s}}=2,6$"),
            ("product_16x16_L2-2", "$16\\times16$, $L_{\\mathrm{s}}=2,2$"), ("product_16x16_L4-4", "$16\\times16$, $L_{\\mathrm{s}}=4,4$")]
    runs = [("prod_8x8", "transformer, trained on product chains", "C2"), ("prod_16x16", "transformer, trained on product chains", "C2"),
            ("lsa_a0_d64", "transformer, trained on independent rows", "C1"), ("d64_seed1_cross", "transformer, trained on the paper's family", "C4")]
    refs = [("unigram", "unigram predictor", "C0"), ("copier", "copier", "C3"), ("prod_unigram", "product unigram", "C0"), ("prod_copier", "product copier, paper's depth prior", "C5")]
    avail = [(tag, title) for tag, title in sets if os.path.exists(f"{REFS}/{tag}.npz")]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(3.0 * len(avail), 2.7)); table = {}
    for ax, (tag, title) in zip(np.atleast_1d(axes), avail):
        R = np.load(f"{REFS}/{tag}.npz"); h = R["h_true"]; n = h.shape[1]; wins = windows(n, 4); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins]); table[tag] = {}
        ax.plot(mid, windowed(h.mean(0), wins), ":", color="0.3", lw=0.8, label="entropy of the source")
        R = dict(R)
        fp = f"{REFS}/{tag}_famprior.npz"
        if os.path.exists(fp):
            P = np.load(fp)
            if str(P["md5"]) == str(R["md5"]):
                R["prod_copier_fam"] = P["prod_copier_fam"]
        for key, lab, c in refs + [("prod_copier_fam", "product copier, family's depth prior (Bayes)", "C9")]:
            if key not in R:
                continue
            v = R[key].astype(float); table[tag][lab] = dict(bits=float(v.mean()), regret=float((v - h).sum(1).mean()))
            ax.plot(mid, windowed(v.mean(0), wins), "--" if key.startswith("prod") else "-", color=c, lw=1.4 if key == "prod_copier_fam" else 1.0, label=lab)
        gF = np.cumsum(R["unigram"].astype(float) - R["copier"].astype(float), 1).mean(0); gP = np.cumsum(R["unigram"].astype(float) - R["prod_copier"].astype(float), 1).mean(0)
        table[tag]["crossing copier vs unigram"] = last_zero(gF); table[tag]["crossing product copier vs unigram"] = last_zero(gP)
        for run, lab, c in runs:
            E = load_eval(evaldir, run, tag)
            if E is None or str(E["md5"]) != str(R["md5"]):
                continue
            lT = -E["logp_true"].astype(float) / LN2; table[tag]["transformer " + run] = dict(bits=float(lT.mean()), regret=float((lT - h).sum(1).mean()))
            ax.plot(mid, windowed(lT.mean(0), wins), "-", color=c, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("position $t$"); ax.set_title(title); ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("loss [bits]"); np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/fam_product.pdf"); fig.savefig(f"{OUT_FIG}/fam_product.png", dpi=130); plt.close(fig)
    json.dump(table, open(f"{OUT_RES}/fam_product.json", "w"), indent=1)
    print("product:", json.dumps(table, indent=1))


# ---- F4: independent rows (alpha = 0) and the cross-training with the paper's family ----------------------------------
def alpha0(evaldir):
    """Per d: the alpha = 0 sets (L_s = 2, 3, 4) and the alpha = 1 sets, each with the unigram, the copier, the model
    trained at alpha = 0 and the model trained on the paper's family."""
    for d, a0run, prun in ((64, "lsa_a0_d64", "d64_seed1_cross"), (128, "lsa_a0_d128", "d128_cross")):
        sets = [(f"d{d}_a{a}_L{L}", f"$d={d}$, $\\alpha={a}$, $L_{{\\mathrm{{s}}}}={L}$") for a in (0, 1) for L in (2, 3, 4)]
        avail = [(tag, title) for tag, title in sets if os.path.exists(f"{REFS2048}/d{d}/{tag}.npz")]
        if not avail:
            continue
        fig, axes = plt.subplots(1, len(avail), figsize=(2.6 * len(avail), 2.6)); table = {}
        for ax, (tag, title) in zip(np.atleast_1d(axes), avail):
            R = np.load(f"{REFS2048}/d{d}/{tag}.npz"); lF = R["lF"].astype(float); lC = R["lC"].astype(float); n = lF.shape[1]
            wins = windows(n, 4); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins]); table[tag] = {}
            ax.plot(mid, windowed(lF.mean(0), wins), "-", color="C0", lw=1.0, label="unigram predictor"); ax.plot(mid, windowed(lC.mean(0), wins), "-", color="C3", lw=1.0, label="copier, paper's depth prior")
            table[tag]["unigram"] = float(lF.mean()); table[tag]["copier"] = float(lC.mean()); table[tag]["n_star"] = last_zero(np.cumsum(lF - lC, 1).mean(0))
            fp = f"{REFS2048}/d{d}/{tag}_famprior.npz"
            if os.path.exists(fp):
                P = np.load(fp)
                if str(P["md5"]) == str(R["md5"]):
                    lCf = P["copier_fam"].astype(float); table[tag]["copier_fam"] = float(lCf.mean()); table[tag]["n_star_fam"] = last_zero(np.cumsum(lF - lCf, 1).mean(0))
                    ax.plot(mid, windowed(lCf.mean(0), wins), "--", color="C9", lw=1.4, label="copier, family's depth prior (Bayes at $\\alpha=0$)")
            for run, lab, c in ((a0run, "transformer, trained at $\\alpha=0$", "C1"), (prun, "transformer, trained on the paper's family", "C2")):
                E = load_eval(evaldir, run, tag)
                if E is None or str(E["md5"]) != str(R["md5"]):
                    continue
                lT = -E["logp_true"].astype(float) / LN2; table[tag][run] = float(lT.mean())
                ax.plot(mid, windowed(lT.mean(0), wins), "-", color=c, lw=1.6, label=lab)
            ax.set_xscale("log"); ax.set_xlabel("position $t$"); ax.set_title(title); ax.grid(True, alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("loss [bits]"); np.atleast_1d(axes)[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(f"{OUT_FIG}/fam_alpha0_d{d}.pdf"); fig.savefig(f"{OUT_FIG}/fam_alpha0_d{d}.png", dpi=130); plt.close(fig)
        json.dump(table, open(f"{OUT_RES}/fam_alpha0_d{d}.json", "w"), indent=1)
        print(f"alpha0 d={d}:", json.dumps(table, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eval", default="fam_eval"); ap.add_argument("--refs", default="refs_fam")
    ap.add_argument("--refs2048", default="tf_refs_2048"); ap.add_argument("--out_fig", default="fam_figs"); ap.add_argument("--out_res", default="fam_res")
    args = ap.parse_args()
    REFS = args.refs; REFS2048 = args.refs2048; OUT_FIG = args.out_fig; OUT_RES = args.out_res
    os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_RES, exist_ok=True)
    binary(args.eval); expfam(args.eval); product(args.eval); alpha0(args.eval)
