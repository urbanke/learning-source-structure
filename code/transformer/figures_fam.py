"""The figures of Section 4 (the five source families): fig_coins, fig_expfam, fig_indep, fig_product, fig_paper.
Run from the project folder:   python3 code/transformer/figures_fam.py
Reads the transformer evaluations (eval_model.py) and the reference predictors (refs_family.py, make_refs.py,
refs_famprior.py, refs_extra.py); writes figures/fig_*.pdf|png and results/transformer/fig_*.json. The directories
below are the repository layout; each can be overridden by the environment variable named next to it (the cloud
scratchpad used other names). fig_paper needs the 512-sequence sets of the paper's source; it is skipped when they
are absent. The true entropies of the paper's-source sets are regenerated from their seeds (h_true_lsa.py).

One visual grammar for all five figures: the Bayes predictor of a training family is a thick pale line, the
transformer trained on that family a thin dark line of the same colour drawn on top of it, so that the two are
compared where the dark line runs inside the pale one; predictors that know only d are grey. The legend labels of a
pair are parallel: "Bayes predictor of <family>" and "transformer trained on <family>".
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 6.8, "lines.linewidth": 1.1})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_fam import windows, windowed, cum_regret, load_eval, last_zero
LN2 = np.log(2.0)
env = os.environ.get
EV = env("FAM_EVAL", "results/transformer/eval")               # transformer evaluations: <run>/<tag>_step<n>.npz
RF = env("FAM_REFS", "results/transformer/refs_fam")           # references on the family sets (refs_family.py, refs_famprior.py)
R2 = env("FAM_REFS2048", "results/transformer/refs2048")       # references on the 2048-sequence sets of the paper's source (make_refs.py)
REFS512 = env("FAM_REFS512", "results/transformer/refs")       # unigram and copier on the 512-sequence sets (make_refs.py, first round)
EXTRA512 = env("FAM_EXTRA512", "results/transformer/refs2")    # the informed copier on those sets (refs_extra.py)
FAM512 = env("FAM_FAMPRIOR512", "results/transformer/refs_famprior")  # the copier with the family's depth prior on those sets
EV512 = env("FAM_EVAL512", EV)                                 # the transformers on those sets
OUT = env("FAM_FIGS", "figures"); RES = env("FAM_RES", "results/transformer")
os.makedirs(OUT, exist_ok=True); os.makedirs(RES, exist_ok=True)

# colours by training family; grey for the predictors that know only d
C_TWO, C_UNI, C_EIGHT = "C3", "C0", "C1"          # the coin priors; also the one-term, two-term and relabelled exponential families
C_INDEP, C_PAPER, C_PROD = "C2", "C4", "C9"       # independent contexts, the paper's source, product chains
GREY, GREY2 = "0.35", "0.6"
REF_LW, REF_ALPHA, TF_LW = 4.2, 0.4, 1.3


def ref_line(ax, x, y, color, label=None):
    """The Bayes predictor of a family (or the plug-in stand-in): thick and pale, drawn first."""
    return ax.plot(x, y, "-", color=color, lw=REF_LW, alpha=REF_ALPHA, solid_capstyle="round", label=label, zorder=2)[0]


def tf_line(ax, x, y, color, label=None):
    """The transformer trained on that family: thin and dark, drawn on top."""
    return ax.plot(x, y, "-", color=color, lw=TF_LW, label=label, zorder=3)[0]


def grey_line(ax, x, y, ls="-", color=GREY, label=None, lw=1.0):
    return ax.plot(x, y, ls, color=color, lw=lw, label=label, zorder=1)[0]


def h_ref(color):
    return Line2D([], [], color=color, lw=REF_LW, alpha=REF_ALPHA, solid_capstyle="round")


def h_tf(color):
    return Line2D([], [], color=color, lw=TF_LW)


def h_grey(ls="-", color=GREY, lw=1.0):
    return Line2D([], [], color=color, lw=lw, ls=ls)


def legend_below(fig, handles, labels, ncol, bottom):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handlelength=2.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, bottom, 1, 1))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(f"{OUT}/{name}.png", dpi=140, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


def tf_regret(run, tag, R):
    E = load_eval(EV, run, tag); assert E is not None, f"no evaluation of {run} on {tag}"; assert str(E["md5"]) == str(R["md5"])
    return cum_regret(-E["logp_true"].astype(float) / LN2, R["h_true"])


def tf_loss(evdir, run, tag, md5):
    E = load_eval(evdir, run, tag); assert E is not None, f"no evaluation of {run} on {tag}"; assert str(E["md5"]) == md5
    return -E["logp_true"].astype(float) / LN2


# ---- Figure: coins -----------------------------------------------------------------------------------------------------
def fig_coins():
    """One panel per prior. On sequences from that prior: the expected cumulative regret against the position, for the
    transformer trained on that prior (line) and the Bayes predictor of that prior (circles), with the limit of the
    Bayes regret dotted (1 bit, 3 bits, and (1/2) log2 n + const). The json also holds the sets with theta fixed and
    the mismatched transformers, which the text quotes."""
    fams = [("binary_twopoint", "twopoint", "bin_twopoint", "two-point prior", "two-point Bayes", "transformer trained on the two-point prior"),
            ("binary_eightpoint", "eightpoint", "bin_eightpoint", "eight-point prior", "eight-point Bayes", "transformer trained on the eight-point prior"),
            ("binary_uniform", "laplace", "bin_uniform", "uniform prior", "add-one (uniform Bayes)", "transformer trained on the uniform prior")]
    tab = {}
    for tag in ("binary_twopoint", "binary_eightpoint", "binary_uniform", "binary_fixed0.5", "binary_fixed0.2"):
        R = np.load(f"{RF}/{tag}.npz"); tab[tag] = {}
        for _, key, run, _, lab_b, lab_t in fams:
            tab[tag][lab_b] = float(cum_regret(R[key], R["h_true"])[-1]); tab[tag][lab_t] = float(tf_regret(run, tag, R)[-1])
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.3))
    marks = np.unique(np.round(np.logspace(0, np.log10(256), 17)).astype(int))          # positions of the circles
    for ax, (tag, key, run, name, _, _) in zip(axes, fams):
        R = np.load(f"{RF}/{tag}.npz"); n = R["h_true"].shape[1]; pos = np.arange(1, n + 1)
        cb = cum_regret(R[key], R["h_true"]); ct = tf_regret(run, tag, R)
        if key == "twopoint":
            ax.axhline(1.0, ls=":", color="k", lw=0.8); ax.text(1.15, 1.0, "1 bit", va="bottom", ha="left", fontsize=7); ax.set_ylim(0, 1.25)
        elif key == "eightpoint":
            ax.axhline(3.0, ls=":", color="k", lw=0.8); ax.text(1.15, 3.0, "3 bits", va="bottom", ha="left", fontsize=7); ax.set_ylim(0, 3.6)
        else:
            c = float(np.mean(cb[63:] - 0.5 * np.log2(pos[63:])))
            ax.plot(pos, 0.5 * np.log2(pos) + c, ":", color="k", lw=0.8); ax.text(14, 0.5 * np.log2(14) + c - 0.35, "$\\frac{1}{2}\\log_2 n$ + const", va="top", ha="left", fontsize=7); ax.set_ylim(0, 4.2)
        ax.plot(pos, ct, "-", color="C0", lw=1.4, label="transformer trained on the prior", zorder=3)
        ax.plot(marks, cb[marks - 1], "o", mfc="none", mec="k", ms=4.5, mew=0.9, label="Bayes predictor of the prior", zorder=4)
        ax.set_xscale("log"); ax.set_xlim(0.85, 300); ax.set_xlabel("position $n$"); ax.set_title(name); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("expected cumulative\nregret [bits]")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7.5, handlelength=2.2, columnspacing=2.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, "fig_coins")
    json.dump(tab, open(f"{RES}/fig_coins.json", "w"), indent=1); return tab


# ---- Figure: the Zipf family --------------------------------------------------------------------------------------------
def fig_expfam():
    """One panel per version of the Zipf family, on its own sequences: the expected cumulative regret of the transformer
    trained on that version (line) and of the reference that knows the form (circles: the Bayes predictor with the
    labels known; squares in the relabelled panel: the Zipf plug-in, a stand-in). The relabelled panel also shows the
    Bayes predictor with the labels known (dotted) and the LSA unigram predictor (dash-dotted); in the other two the
    unigram predictor is off the scale and quoted in a note."""
    fams = [("expfam_k1", "bayes_k1", "expfam_k1", "one parameter"), ("expfam_k2", "bayes_k2", "expfam_k2", "two parameters"),
            ("expfam_k1_perm", "zipf_plugin", "expfam_k1_perm", "one parameter, symbols relabelled")]
    runs = [("expfam_k1", "transformer trained on one term"), ("expfam_k2", "transformer trained on two terms"), ("expfam_k1_perm", "transformer trained on one term, relabelled")]
    tab = {}
    for tag, _, _, _ in fams:
        R = np.load(f"{RF}/{tag}.npz"); h = R["h_true"]; tab[tag] = {}
        for key, lab in (("bayes_k1", "Bayes, one term, labels known"), ("bayes_k2", "Bayes, two terms, labels known"), ("zipf_plugin", "Zipf plug-in, labels unknown"), ("lsa_unigram", "LSA unigram predictor")):
            tab[tag][lab] = float(cum_regret(R[key], h)[-1])
        for run, lab in runs:
            tab[tag][lab] = float(tf_regret(run, tag, R)[-1])
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.3))
    marks = np.unique(np.round(np.logspace(0, np.log10(1024), 19)).astype(int))
    for ax, (tag, key, run, title) in zip(axes, fams):
        R = np.load(f"{RF}/{tag}.npz"); h = R["h_true"]; n = h.shape[1]; pos = np.arange(1, n + 1)
        cb = cum_regret(R[key], h); ct = tf_regret(run, tag, R); cu = cum_regret(R["lsa_unigram"], h)
        ax.plot(pos, ct, "-", color="C0", lw=1.4, label="transformer trained on the family", zorder=3)
        if key == "zipf_plugin":
            ax.plot(marks, cb[marks - 1], "s", mfc="none", mec="k", ms=4.2, mew=0.9, label="Zipf plug-in, labels unknown (a stand-in)", zorder=4)
            ax.plot(pos, cum_regret(R["bayes_k1"], h), ":", color="k", lw=0.9, label="Bayes predictor, labels known", zorder=2)
            ax.plot(pos, cu, "-.", color="0.45", lw=0.9, label="LSA unigram predictor (form unknown)", zorder=2)
            ax.set_ylim(0, 80)
        else:
            ax.plot(marks, cb[marks - 1], "o", mfc="none", mec="k", ms=4.5, mew=0.9, label="Bayes predictor of the family", zorder=4)
            ax.set_ylim(0, 3.2 if key == "bayes_k1" else 7.5)
            ax.text(0.05, 0.95, f"LSA unigram predictor:\n{cu[-1]:.0f} bits at $n=1024$\n(off the scale)", transform=ax.transAxes, va="top", ha="left", fontsize=6.3, color="0.3")
        ax.set_xscale("log"); ax.set_xlim(0.85, 1200); ax.set_xlabel("position $n$"); ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("expected regret\nper position [bits]")
    H, L = [], []
    for ax in (axes[0], axes[2]):
        h_, l_ = ax.get_legend_handles_labels()
        for a, b in zip(h_, l_):
            if b not in L:
                H.append(a); L.append(b)
    fig.legend(H, L, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7, handlelength=2.0, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.16, 1, 1)); save(fig, "fig_expfam")
    json.dump(tab, open(f"{RES}/fig_expfam.json", "w"), indent=1); return tab


def _marks(n, k=19):
    return np.unique(np.round(np.logspace(0, np.log10(n), k)).astype(int))


def _chain_panel(ax, pos, tf, bayes, copier, unigram, marker="o", bayes_label="Bayes predictor of the family", oracle=None):
    """The common panel of the chain figures: expected regret per position (the mean loss minus the entropy of the true
    next-token distribution, averaged over windows of positions equally spaced on a log scale) of the transformer (line),
    of the Bayes predictor or its stand-in (markers), of the copier of the paper (dashed grey) and of the unigram
    predictor (dash-dotted grey). All inputs are per-position means over the sequences, (n,)."""
    n = len(pos); wins = windows(n, 4); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins]); W = lambda v: windowed(v, wins)
    ax.plot(mid, W(unigram), "-.", color="0.45", lw=0.9, label="unigram predictor", zorder=1)
    ax.plot(mid, W(copier), "--", color="0.45", lw=0.9, label="copier of the paper", zorder=1)
    if oracle is not None:
        ax.plot(mid, W(oracle), ":", color="k", lw=0.9, label="oracle copier (given the base law)", zorder=2)
    ax.plot(mid, W(tf), "-", color="C0", lw=1.4, label="transformer trained on the family", zorder=3)
    ax.plot(mid, W(bayes), marker, mfc="none", mec="k", ms=4.3, mew=0.9, ls="none", label=bayes_label, zorder=4)
    ax.set_xscale("log"); ax.set_xlim(0.85, 1200); ax.set_xlabel("position $t$"); ax.grid(True, alpha=0.3); ax.set_ylim(0, None)


def _finish(fig, axes, name, bottom=0.2, ncol=3):
    axes[0].set_ylabel("expected regret\nper position [bits]")
    H, L = [], []
    for ax in axes:
        for a, b in zip(*ax.get_legend_handles_labels()):
            if b not in L:
                H.append(a); L.append(b)
    order = [i for i, b in enumerate(L) if b.startswith("transformer")] + [i for i, b in enumerate(L) if not b.startswith("transformer")]
    fig.legend([H[i] for i in order], [L[i] for i in order], loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7, handlelength=2.0, columnspacing=1.6)
    fig.tight_layout(rect=(0, bottom, 1, 1)); save(fig, name)


# ---- Figure: independent contexts (alpha = 0) ---------------------------------------------------------------------------
def fig_indep(d=128):
    """d = 128, alpha = 0, L_s = 2 and 4 (2048 sequences): the transformer trained on independent contexts (line), the
    copier with the family's depth prior 0..8, its Bayes predictor (circles), the copier of the paper and the unigram
    predictor. The true entropies are regenerated from the set's seed (h_true_lsa)."""
    from h_true_lsa import h_true_lsa
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4)); tab = {}
    for ax, L in zip(axes, (2, 4)):
        tag = f"d{d}_a0_L{L}"; R = np.load(f"{R2}/d{d}/{tag}.npz"); P = np.load(f"{R2}/d{d}/{tag}_famprior.npz"); md5 = str(R["md5"]); assert str(P["md5"]) == md5
        h = h_true_lsa(d, 0.0, L, R["lF"].shape[1], R["lF"].shape[0], md5); n = h.shape[1]; pos = np.arange(1, n + 1)
        lF = R["lF"].astype(float); lC = R["lC"].astype(float); lCf = P["copier_fam"].astype(float); tI = tf_loss(EV, f"lsa_a0_d{d}", tag, md5)
        cur = (("transformer trained on independent contexts", tI), ("copier, depths 0 to 8 (the family's)", lCf), ("copier, depths 1 to $L_{\\max}$ (the paper's)", lC), ("unigram predictor", lF))
        tab[tag] = {k: dict(bits=float(v.mean()), regret=float(cum_regret(v, h)[-1])) for k, v in cur}
        cv = {k: (v.astype(float) - h).mean(0) for k, v in cur}
        tab[tag]["n_star"] = last_zero(np.cumsum(lF - lC, 1).mean(0)); tab[tag]["n_star_fam"] = last_zero(np.cumsum(lF - lCf, 1).mean(0))
        _chain_panel(ax, pos, cv["transformer trained on independent contexts"], cv["copier, depths 0 to 8 (the family's)"], cv["copier, depths 1 to $L_{\\max}$ (the paper's)"], cv["unigram predictor"],
                     bayes_label="Bayes predictor of the family (copier, depths 0 to 8)")
        ax.set_title(f"$d={d}$, $\\alpha=0$, $L_{{\\mathrm{{s}}}}={L}$")
    _finish(fig, axes, "fig_indep", bottom=0.22, ncol=2)
    json.dump(tab, open(f"{RES}/fig_indep.json", "w"), indent=1); return tab


# ---- Figure: the graphical model ------------------------------------------------------------------------------------------
def fig_product():
    """8x8 with L_s = 2,2 (2048 sequences) and 16x16 with L_s = 4,4 (1024): the transformer trained on product chains
    (line), the product copier, its Bayes predictor (circles), the copier of the paper and the unigram predictor."""
    sets = [("product_8x8_L2-2", "$8\\times8$, $L_{\\mathrm{s}}=2,2$", "prod_8x8"), ("product_16x16_L4-4", "$16\\times16$, $L_{\\mathrm{s}}=4,4$", "prod_16x16")]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4)); tab = {}
    for ax, (tag, title, run) in zip(axes, sets):
        R = np.load(f"{RF}/{tag}.npz"); P = np.load(f"{RF}/{tag}_famprior.npz"); md5 = str(R["md5"]); assert str(P["md5"]) == md5
        h = R["h_true"]; n = h.shape[1]; pos = np.arange(1, n + 1)
        curves = [("unigram predictor", R["unigram"].astype(float)), ("copier of the paper", R["copier"].astype(float)),
                  ("product copier (two copiers, depths 1 to 8)", P["prod_copier_fam"].astype(float)), ("transformer trained on product chains", tf_loss(EV, run, tag, md5))]
        if run == "prod_8x8":
            curves += [("transformer trained on independent contexts", tf_loss(EV, "lsa_a0_d64", tag, md5)), ("transformer trained on the paper's source", tf_loss(EV, "d64_seed1_cross", tag, md5))]
        tab[tag] = {lab: dict(bits=float(v.mean()), regret=float((v - h).sum(1).mean())) for lab, v in curves}
        cv = {lab: (v.astype(float) - h).mean(0) for lab, v in curves}
        _chain_panel(ax, pos, cv["transformer trained on product chains"], cv["product copier (two copiers, depths 1 to 8)"], cv["copier of the paper"], cv["unigram predictor"],
                     bayes_label="Bayes predictor of the family (the product copier)")
        ax.set_title(title)
    _finish(fig, axes, "fig_product", bottom=0.22, ncol=2)
    json.dump(tab, open(f"{RES}/fig_product.json", "w"), indent=1); return tab


# ---- Figure: the paper's source ---------------------------------------------------------------------------------------------
def fig_paper():
    """The paper's source at (64,1,2) and (128,1,2), on the 48 and 24 sequences per set on which the informed copier was
    run: the transformer trained on the paper's source (line), the informed copier (squares, a stand-in for the Bayes
    predictor), the oracle copier (dotted), the copier of the paper and the unigram predictor."""
    from h_true_lsa import h_true_lsa
    sets = [(64, 1.0, 2, "d64_seed1"), (128, 1.0, 2, "d128")]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4)); tab = {}
    for ax, (d, a, L, prun) in zip(axes, sets):
        tag = f"d{d}_a{a:g}_L{L}"; R = np.load(f"{REFS512}/d{d}/{tag}.npz"); X = np.load(f"{EXTRA512}/d{d}/extra_{tag}.npz"); md5 = str(R["md5"]); assert str(X["md5"]) == md5
        N = int(X["n_seq"]); h = h_true_lsa(d, a, L, R["lF"].shape[1], R["lF"].shape[0], md5)[:N]; n = h.shape[1]; pos = np.arange(1, n + 1)
        tP = tf_loss(EV512, prun, tag, md5)[:N]
        curves = [("transformer trained on the paper's source", tP), ("informed copier", X["lZ"].astype(float)), ("oracle copier", X["lO"].astype(float)),
                  ("copier of the paper", R["lC"].astype(float)[:N]), ("unigram predictor", R["lF"].astype(float)[:N])]
        tab[tag] = {lab: dict(bits=float(v.mean()), regret=float(cum_regret(v, h)[-1])) for lab, v in curves}; tab[tag]["n_seq"] = N
        cv = {lab: (v.astype(float) - h).mean(0) for lab, v in curves}
        _chain_panel(ax, pos, cv["transformer trained on the paper's source"], cv["informed copier"], cv["copier of the paper"], cv["unigram predictor"],
                     marker="s", bayes_label="informed copier (a stand-in for the Bayes predictor)", oracle=cv["oracle copier"])
        ax.set_title(f"$d={d}$, $\\alpha={a:g}$, $L_{{\\mathrm{{s}}}}={L}$ ({N} sequences)")
    _finish(fig, axes, "fig_paper", bottom=0.26, ncol=2)
    json.dump(tab, open(f"{RES}/fig_paper.json", "w"), indent=1); return tab


if __name__ == "__main__":
    for f in (fig_coins, fig_expfam, fig_indep, fig_product, fig_paper):
        if f is fig_paper and not os.path.exists(f"{EXTRA512}/d64/extra_d64_a1_L2.npz"):
            print("fig_paper skipped: the 512-sequence sets of the paper's source are not here"); continue
        t = f(); print(f.__name__, json.dumps(t, indent=1)[:600])
