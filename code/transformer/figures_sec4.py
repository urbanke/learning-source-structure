"""The figures and tables of Section 4 (working version). Run in a directory that holds
    tf_refs/d<d>/<tag>.npz          the mechanism references (make_refs.py), 512 sequences per setting
    tf_refs2/d<d>/extra_<tag>.npz   the informed and oracle copiers on the grid (refs_extra.py --which ZO)
    tf_refs3/d<d>/extra_<tag>.npz   the full ladder at (1,2) and (1,4) (refs_extra.py --which JDBZO)
    tf_eval/<run>/<tag>_step<step>.npz   the transformer evaluations (eval_model.py)
    tf_runs/<run>/log.jsonl         the training logs (results/transformer/runs/<run>/log.jsonl in the repository)
    tf_analysis2/crossings.json     the output of analyze2.py
and writes sec4_figs/tf_*.pdf|png (copied to figures/) and sec4_res/*.tex|json (copied to results/transformer/).
Usage: python3 figures_sec4.py"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import windows as _windows, windowed
LN2 = np.log(2.0)


def windows(n, per_decade=4, min_last=8):
    """Geometric position windows; a last window with fewer than min_last positions is merged into the one before."""
    w = _windows(n, per_decade)
    if len(w) > 1 and w[-1][1] - w[-1][0] < min_last:
        w = w[:-2] + [(w[-2][0], w[-1][1])]
    return w
RUNS = {32: ("d32", 100000), 64: ("d64_seed1", 100000), 128: ("d128", 100000)}
OUT_FIG = "sec4_figs"; OUT_RES = "sec4_res"
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_RES, exist_ok=True)


def load_setting(d, a, L, ladder=False, all_seq=False):
    """Per-position losses (sequences x positions) of the predictors at one setting, on the same sequences: the first
    n_seq of the set (those on which the informed/oracle copiers were run), or all 512 if all_seq (then only the
    unigram, the copier, the learner and the transformer)."""
    run, step = RUNS[d]; tag = f"d{d}_a{a:g}_L{L}"
    R = np.load(f"tf_refs/d{d}/{tag}.npz"); E = np.load(f"tf_eval/{run}/{tag}_step{step}.npz")
    assert str(R["md5"]) == str(E["md5"])
    if all_seq:
        N = R["lF"].shape[0]
        L_ = dict(unigram=R["lF"], copier=R["lC"], transformer=-E["logp_true"][:N] / LN2)
        L_["learner"] = -np.log2(R["wC"] * 2.0 ** (-L_["copier"]) + (1 - R["wC"]) * 2.0 ** (-L_["unigram"]))
        return {k: v.astype(np.float64) for k, v in L_.items()}, N
    X = np.load(f"tf_refs3/d{d}/extra_{tag}.npz") if ladder else np.load(f"tf_refs2/d{d}/extra_{tag}.npz")
    assert str(R["md5"]) == str(X["md5"])
    N = int(X["n_seq"])
    L_ = dict(unigram=R["lF"][:N], copier=R["lC"][:N], transformer=-E["logp_true"][:N] / LN2)
    L_["learner"] = -np.log2(R["wC"][:N] * 2.0 ** (-L_["copier"]) + (1 - R["wC"][:N]) * 2.0 ** (-L_["unigram"]))
    for key, name in (("lD", "true depth"), ("lJ", "joint pairs"), ("lB", "backoff"), ("lZ", "informed"), ("lO", "oracle")):
        if key in X.files:
            L_[name] = X[key]
    L_ = {k: v.astype(np.float64) for k, v in L_.items()}
    lF, lZ = L_["unigram"], L_["informed"]
    g = np.cumsum(lF, 1) - np.cumsum(lZ, 1); w = np.empty_like(g); w[:, 0] = 0.5; w[:, 1:] = 1 / (1 + 2.0 ** (-g[:, :-1]))
    L_["learner, informed"] = -np.log2(w * 2.0 ** (-lZ) + (1 - w) * 2.0 ** (-lF))
    return L_, N


# ---- Figure A: the transformer against the two mechanisms, two settings --------------------------------------------
def figure_A(settings=((64, 1.0, 2), (128, 1.0, 2))):
    fig, axes = plt.subplots(1, len(settings), figsize=(3.2 * len(settings), 2.6))
    for ax, (d, a, L) in zip(np.atleast_1d(axes), settings):
        Ld, N = load_setting(d, a, L, all_seq=True); n = Ld["transformer"].shape[1]; wins = windows(n, 4); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins])
        for k, c, ls in (("unigram", "C0", "-"), ("copier", "C3", "-"), ("learner", "0.4", "--"), ("transformer", "C2", "-")):
            ax.plot(mid, windowed(Ld[k].mean(0), wins), ls, color=c, lw=1.6 if k == "transformer" else 1.0, label="unigram predictor" if k == "unigram" else k)
        ax.set_xscale("log"); ax.set_xlabel("position $t$"); ax.set_title(f"$d={d}$, $\\alpha={a:g}$, $L_{{\\mathrm{{s}}}}={L}$")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("loss [bits]"); np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/tf_mechanisms.pdf"); fig.savefig(f"{OUT_FIG}/tf_mechanisms.png", dpi=130); plt.close(fig)


# ---- Figure B: the ladder, two settings: loss minus the transformer's --------------------------------------------------
def figure_B(settings=((64, 1.0, 2), (128, 1.0, 2))):
    names = [("unigram", "C0"), ("copier", "C3"), ("true depth", "C4"), ("joint pairs", "C1"), ("backoff", "C6"), ("informed", "C8"), ("learner, informed", "C9"), ("oracle", "C5")]
    fig, axes = plt.subplots(1, len(settings), figsize=(3.4 * len(settings), 2.8), sharey=True)
    for ax, (d, a, L) in zip(np.atleast_1d(axes), settings):
        Ld, N = load_setting(d, a, L, ladder=True); n = Ld["transformer"].shape[1]; wins = windows(n, 4); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins])
        base = windowed(Ld["transformer"].mean(0), wins)
        for k, c in names:
            if k in Ld:
                ax.plot(mid, windowed(Ld[k].mean(0), wins) - base, "-", color=c, lw=1.4 if k == "informed" else 1.0, label=k)
        ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("position $t$"); ax.set_title(f"$d={d}$, $\\alpha={a:g}$, $L_{{\\mathrm{{s}}}}={L}$ ({N} sequences)")
        ax.set_ylim(-0.2, 0.9); ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("loss minus the transformer's [bits]"); np.atleast_1d(axes)[-1].legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/tf_ladder.pdf"); fig.savefig(f"{OUT_FIG}/tf_ladder.png", dpi=130); plt.close(fig)


# ---- Table: bits per token over the grid -------------------------------------------------------------------------------
def table_grid():
    rows = []; lines = ["\\begin{tabular}{rrr|rrrr|rrr}", "\\toprule",
                        "$d$ & $\\alpha$ & $L_{\\mathrm{s}}$ & \\unigram & copier & learner & transformer & informed & learner, inf. & oracle\\\\", "\\midrule"]
    for d in (32, 64, 128):
        for a in (0.75, 1.0, 1.25):
            for L in (2, 3, 4):
                try:
                    Ld, N = load_setting(d, a, L)
                except (FileNotFoundError, AssertionError):
                    continue
                m = {k: float(v.mean()) for k, v in Ld.items()}; m.update(d=d, alpha=a, ls=L, n_seq=N); rows.append(m)
                lines.append(f"{d} & {a:g} & {L} & {m['unigram']:.2f} & {m['copier']:.2f} & {m['learner']:.2f} & {m['transformer']:.2f} & {m['informed']:.2f} & {m['learner, informed']:.2f} & {m['oracle']:.2f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT_RES}/tf_grid_table.tex", "w").write("\n".join(lines) + "\n"); json.dump(rows, open(f"{OUT_RES}/tf_grid.json", "w"), indent=1)
    return rows


# ---- Figure C: the crossings against d --------------------------------------------------------------------------------
def figure_C():
    rows = json.load(open("tf_analysis2/crossings.json"))
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True)
    cols = {0.75: "C1", 1.0: "C2", 1.25: "C3"}
    for ax, L in zip(axes, (2, 3, 4)):
        for a in (0.75, 1.0, 1.25):
            sel = sorted([r for r in rows if r["ls"] == L and r["alpha"] == a], key=lambda r: r["d"])
            for key, mk, lab in (("n_star_paper", "o-", "$n^*_d$, the copier of the paper"), ("n_star_product", "s--", "the informed copier")):
                xs = [r["d"] for r in sel if r[key]]; ys = [r[key] for r in sel if r[key]]
                if xs:
                    ax.plot(xs, ys, mk, color=cols[a], ms=4, lw=1, label=(f"{lab}" if (L == 2 and a == 1.0) else None))
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(f"$L_{{\\mathrm{{s}}}}={L}$"); ax.set_xlabel("$d$"); ax.grid(True, alpha=0.3)
        ax.axhline(1024, color="0.7", lw=0.8)
    axes[0].set_ylabel("crossing [tokens]")
    h = [plt.Line2D([], [], color=cols[a], lw=1, label=f"$\\alpha={a:g}$") for a in cols] + [plt.Line2D([], [], color="0.3", marker="o", lw=1, label="$n^*_d$ (copier of the paper)"), plt.Line2D([], [], color="0.3", marker="s", ls="--", lw=1, label="informed copier")]
    axes[0].legend(handles=h, frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/tf_crossings.pdf"); fig.savefig(f"{OUT_FIG}/tf_crossings.png", dpi=130); plt.close(fig)


# ---- Figure D: the transition during training -------------------------------------------------------------------------
def figure_D():
    runs = [("d32", "$d=32$", "C0"), ("d64_seed1", "$d=64$", "C2"), ("d64", "$d=64$, second seed", "C2"), ("d128", "$d=128$", "C3"), ("d64_fixed", "$d=64$, fixed source", "C4"), ("d64_noperm", "$d=64$, no relabelling", "C1")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6)); out = {}
    for run, lab, c in runs:
        f = f"tf_runs/{run}/log.jsonl"
        if not os.path.exists(f):
            continue
        rec = [json.loads(l) for l in open(f)]; vals = [(r["step"], r["val"]["a1_L4"]["513-1024"]) for r in rec if "val" in r]
        steps = np.array([s for s, v in vals]); late = np.array([v for s, v in vals])
        plateau = late[(steps >= 1000) & (steps <= 3000)].mean(); final = late[-5:].mean()
        half = int(steps[np.argmax(late < (plateau + final) / 2)]); out[run] = dict(plateau=float(plateau), final=float(final), half_step=half)
        ax.plot(steps / 1000, late, color=c, lw=1.4 if "seed1" in run or run in ("d32", "d128") else 0.9, ls="--" if run in ("d64", "d64_fixed") else (":" if run == "d64_noperm" else "-"), label=lab)
    ax.set_xlabel("training step [thousands]"); ax.set_ylabel("loss, positions 513 to 1024 [bits]"); ax.set_xlim(0, 60); ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=6)
    ax.set_title("validation setting $\\alpha=1$, $L_{\\mathrm{s}}=4$")
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/tf_training.pdf"); fig.savefig(f"{OUT_FIG}/tf_training.png", dpi=130); plt.close(fig)
    json.dump(out, open(f"{OUT_RES}/tf_training_transition.json", "w"), indent=1)
    return out


# ---- Figure E: the weight on copying along the sequence ---------------------------------------------------------------
def figure_E(settings=((64, 1.0, 2), (128, 1.0, 2)), per_decade=4):
    """Window means (log-spaced windows of stored positions) of: the paper's learner weight w_C, the informed learner's
    weight, and the transformer's implied weight on the informed copier (against the unigram)."""
    rows = {(r["d"], r["alpha"], r["ls"]): r for r in json.load(open("tf_analysis2/crossings.json"))}
    fig, axes = plt.subplots(1, len(settings), figsize=(3.2 * len(settings), 2.6), sharey=True); out = {}
    for ax, (d, a, L) in zip(np.atleast_1d(axes), settings):
        r = rows[(d, a, L)]; pos = np.array(r["positions"])
        R = np.load(f"tf_refs/d{d}/d{d}_a{a:g}_L{L}.npz"); wC = R["wC"].astype(float).mean(0); n = len(wC)   # all 512 sequences
        wins = windows(n, per_decade); mid = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in wins])
        def wm(vals, at):
            return np.array([vals[(at >= lo) & (at < hi)].mean() for lo, hi in wins])
        curves = {"learner of the paper ($w_C$)": (wm(wC, np.arange(1, n + 1)), "0.4", "--"),
                  "learner with the informed copier": (wm(np.array(r["learner_product_weight"]), pos), "C9", "-"),
                  "transformer, implied weight": (wm(np.array(r["implied_weight_product"]), pos), "C2", "-")}
        for k, (v, c, ls) in curves.items():
            ax.plot(mid, v, ls, color=c, lw=1.6 if "transformer" in k else 1.0, label=k)
        ax.axhline(0.5, color="0.6", lw=0.8); ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("position $t$"); ax.grid(True, alpha=0.3)
        ax.set_title(f"$d={d}$, $\\alpha={a:g}$, $L_{{\\mathrm{{s}}}}={L}$")
        out[f"d{d}_a{a:g}_L{L}"] = {k: v.round(3).tolist() for k, (v, _, _) in curves.items()} | {"window_mid": mid.round(1).tolist()}
    np.atleast_1d(axes)[0].set_ylabel("weight on copying"); np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/tf_weights.pdf"); fig.savefig(f"{OUT_FIG}/tf_weights.png", dpi=130); plt.close(fig)
    json.dump(out, open(f"{OUT_RES}/tf_weights.json", "w"), indent=1)


def robust_switches(per_decade=4):
    """For every grid setting: the first window from which the window means of the weight stay above one half, for the
    transformer's implied weight on the informed copier and for the informed learner's weight. Windows of stored positions."""
    out = []
    for r in json.load(open("tf_analysis2/crossings.json")):
        pos = np.array(r["positions"]); n = 1024; wins = windows(n, per_decade)
        def sw(vals):
            m = np.array([vals[(pos >= lo) & (pos < hi)].mean() for lo, hi in wins])
            if m[-1] <= 0.5:
                return None
            i = len(m) - 1
            while i > 0 and m[i - 1] > 0.5:
                i -= 1
            return wins[i][0]
        out.append(dict(d=r["d"], alpha=r["alpha"], ls=r["ls"], n_star_paper=r["n_star_paper"], n_star_product=r["n_star_product"],
                        transformer_switch=sw(np.array(r["implied_weight_product"])), learner_informed_switch=sw(np.array(r["learner_product_weight"]))))
    json.dump(out, open(f"{OUT_RES}/tf_switches.json", "w"), indent=1)
    return out


# ---- Table: the ladder at (1,2) and (1,4), loss minus the transformer's with paired standard errors ------------------
def table_ladder():
    order = [("unigram", "\\unigram"), ("copier", "copier"), ("true depth", "copier at the true depth"), ("joint pairs", "joint-pair copier"),
             ("backoff", "backoff copier"), ("informed", "informed copier"), ("learner, informed", "learner with the informed copier"), ("oracle", "oracle copier")]
    res = {}
    for L in (2, 4):
        for d in (32, 64, 128):
            Ld, N = load_setting(d, 1.0, L, ladder=True); T = Ld["transformer"].mean(1)
            res[(d, L)] = {"N": N, "T": float(Ld["transformer"].mean())}
            for k, _ in order:
                diff = Ld[k].mean(1) - T; res[(d, L)][k] = (float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(N)))
    lines = ["\\begin{tabular}{l|rrr|rrr}", "\\toprule", " & \\multicolumn{3}{c|}{$L_{\\mathrm{s}}=2$} & \\multicolumn{3}{c}{$L_{\\mathrm{s}}=4$}\\\\",
             "predictor & $d=32$ & $d=64$ & $d=128$ & $d=32$ & $d=64$ & $d=128$\\\\", "\\midrule"]
    for k, name in order:
        lines.append(f"{name} & " + " & ".join(f"${res[(d, L)][k][0]:+.3f}$" for L in (2, 4) for d in (32, 64, 128)) + "\\\\")
    lines.append("\\midrule")
    lines.append("transformer, bits per token & " + " & ".join(f"${res[(d, L)]['T']:.3f}$" for L in (2, 4) for d in (32, 64, 128)) + "\\\\")
    lines.append("sequences & " + " & ".join(f"${res[(d, L)]['N']}$" for L in (2, 4) for d in (32, 64, 128)) + "\\\\")
    semax = max(res[(d, L)][k][1] for L in (2, 4) for d in (32, 64, 128) for k, _ in order if k != "unigram")
    lines.append("standard error, at most & " + " & ".join(f"${max(res[(d, L)][k][1] for k, _ in order if k != 'unigram'):.3f}$" for L in (2, 4) for d in (32, 64, 128)) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT_RES}/tf_ladder_table.tex", "w").write("\n".join(lines) + "\n")
    json.dump({f"d{d}_L{L}": v for (d, L), v in res.items()}, open(f"{OUT_RES}/tf_ladder.json", "w"), indent=1)
    return res


# ---- Table: the crossings n*_d (512 sequences) and n*_inf (the informed copier's sequences), with bootstrap intervals --
def last_zero(g):
    """1-based position from which the mean cumulative gap stays positive; None if it is not positive at the end."""
    if g[-1] <= 0:
        return None
    i = len(g) - 1
    while i > 0 and g[i - 1] > 0:
        i -= 1
    return i + 1


def bootstrap_crossing(lF, lX, B=300, seed=0):
    """16th and 84th percentiles of the crossing over B resamples of the sequences (1025 stands for no crossing)."""
    rng = np.random.default_rng(seed); N = lF.shape[0]; D = np.cumsum(lF - lX, 1); vals = []
    for _ in range(B):
        idx = rng.integers(0, N, N); v = last_zero(D[idx].mean(0)); vals.append(1025 if v is None else v)
    return np.percentile(np.array(vals), [16, 84])


def table_crossings():
    def fmt(v, ci):
        if v is None:
            return "$>1024$"
        lo, hi = int(ci[0]), int(ci[1])
        if v == 1 and hi == 1:
            return "$1$"
        hi_s = "{>}1024" if hi >= 1025 else str(hi)
        return f"${v}$ {{\\scriptsize$[{lo},{hi_s}]$}}"
    lines = ["\\begin{tabular}{rr|ll|ll|ll}", "\\toprule", " & & \\multicolumn{2}{c|}{$d=32$} & \\multicolumn{2}{c|}{$d=64$} & \\multicolumn{2}{c}{$d=128$}\\\\",
             "$\\alpha$ & $L_{\\mathrm{s}}$ & $n^*_d$ & $n^*_{\\mathrm{inf}}$ & $n^*_d$ & $n^*_{\\mathrm{inf}}$ & $n^*_d$ & $n^*_{\\mathrm{inf}}$\\\\", "\\midrule"]
    out = []
    for L in (2, 3, 4):
        for a in (0.75, 1.0, 1.25):
            cells = []; rec = dict(alpha=a, ls=L)
            for d in (32, 64, 128):
                tag = f"d{d}_a{a:g}_L{L}"; R = np.load(f"tf_refs/d{d}/{tag}.npz"); X = np.load(f"tf_refs2/d{d}/extra_{tag}.npz")
                lF = R["lF"].astype(float); lC = R["lC"].astype(float); N = int(X["n_seq"]); lZ = X["lZ"].astype(float)
                nd = last_zero(np.cumsum(lF - lC, 1).mean(0)); ci_d = bootstrap_crossing(lF, lC)
                ni = last_zero(np.cumsum(lF[:N] - lZ, 1).mean(0)); ci_i = bootstrap_crossing(lF[:N], lZ)
                cells += [fmt(nd, ci_d), fmt(ni, ci_i)]
                rec[f"d{d}"] = dict(n_star_d=nd, ci_d=[float(c) for c in ci_d], n_seq_d=int(lF.shape[0]), n_star_inf=ni, ci_inf=[float(c) for c in ci_i], n_seq_inf=N)
            lines.append(f"{a:g} & {L} & " + " & ".join(cells) + "\\\\"); out.append(rec)
            if L in (2, 3) and a == 1.25:
                lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT_RES}/tf_crossings_table.tex", "w").write("\n".join(lines) + "\n"); json.dump(out, open(f"{OUT_RES}/tf_crossings.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    figure_A(); print("A done")
    try:
        figure_B(); print("B done")
    except Exception as e:
        print("B skipped:", e)
    rows = table_grid(); print("table:", len(rows), "rows")
    figure_C(); print("C done")
    print(figure_D())
    figure_E(); print("E done")
    for s in robust_switches():
        print(s)
    table_ladder(); print("ladder table done")
    table_crossings(); print("crossings table done")
