"""LaTeX tables for the validation subsection: single-distribution checks, chain regrets, crossings.
Writes bb_tables.tex (input by the appendix) and bb_summary.json (numbers quoted in the text)."""
import json, os, sys
import numpy as np
import io, contextlib
from compare import report

SETTINGS = [("S1", 128, 0.75, 2), ("S2", 128, 1.0, 2), ("S3", 128, 0.75, 3), ("S4", 128, 1.0, 3),
            ("S5", 256, 0.75, 2), ("S6", 256, 1.0, 2), ("S7", 256, 0.75, 3), ("S8", 256, 1.0, 3)]


def fmt_alpha(a):
    return f"{a:g}"


def single_table():
    """Pools the independent truth runs single_truth3.json, single_truth4.json (same centring)."""
    runs = [json.load(open(f)) for f in ["single_truth3.json", "single_truth4.json", "single_truth5.json"] if os.path.exists(f)]
    pooled = {}
    for run in runs:
        for r in run:
            key = (r["d"], r["alpha"], r["Ls"], r["k"])
            p = pooled.setdefault(key, dict(r, n=0, sum=0.0, var=0.0))
            n = r["ndraw"]; p["n"] += n; p["sum"] += n * r["mc"]; p["var"] += n * n * r["se"] ** 2
    rows = []
    for key, p in pooled.items():
        mc = p["sum"] / p["n"]; se = np.sqrt(p["var"]) / p["n"]
        rows.append(dict(d=key[0], alpha=key[1], Ls=key[2], k=key[3], mc=mc, se=se, pred=p["pred"], U=p["U"], eta=p["eta"],
                         err=p["pred"] - mc, ndraw=p["n"]))
    rows.sort(key=lambda r: (r["d"], r["alpha"], r["Ls"], r["k"]))
    out = ["\\begin{tabular}{rrrrrrr}", "\\toprule",
           "$d$ & $\\alpha$ & $L_{\\mathrm{s}}$ & $k$ & simulation & formula & error \\\\", "\\midrule"]
    errs = []
    for r in rows:
        out.append(f"{r['d']} & {fmt_alpha(r['alpha'])} & {r['Ls']} & {r['k']} & "
                   f"{r['mc']:.2f} $\\pm$ {r['se']:.2f} & {r['pred']:.2f} & {r['err']:+.2f} \\\\")
        errs.append(r["err"])
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out), rows


def chain_tables():
    regret_rows = []
    cross_rows = []
    summary = {}
    excl = os.environ.get("EXCLUDE", "").split(",")
    for S, d, a, Ls in SETTINGS:
        if S in excl or not (os.path.exists(f"sim_{S}.json") and os.path.exists(f"tab_{S}.json")):
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = report(f"sim_{S}.json", f"tab_{S}.json", S)
        s = json.load(open(f"sim_{S}.json")); t = json.load(open(f"tab_{S}.json"))
        ns = s["checkpoints"]
        lF = np.array(s["lF"]); lC = np.array(s["lC"]); cvF = np.array(s["cvF"]); cvC = np.array(s["cvC"])
        R = lF.shape[0]
        mF = lF - cvF; mC = lC - cvC; mG = mF - mC
        pU = np.array(t["predF_U"]); pe = np.array(t["predF_eta"]); EcvF = np.array(t["EcvF"])
        pC = np.array(t["predC"]); covC = np.array(t.get("covC", np.zeros_like(pC))); EcvC_occ = np.array(t["EcvC_occ"]); EcvF_occ = np.array(t["EcvF_occ"])
        if os.path.exists("exact_EH.json") and S in json.load(open("exact_EH.json")):
            covC = covC + (np.array(ns, float)[None, :] - 1.0) * (t["Hbar"] - json.load(open("exact_EH.json"))[S])
        pF = pU - pe - EcvF; pCr = pC + covC - EcvC_occ
        NT = pF.shape[0]; NC = pCr.shape[0]
        w1 = np.array(t["w1"])
        dF = []; dC = []
        for j, n in enumerate(ns):
            a_, ase = mF[:, j].mean(), mF[:, j].std(ddof=1) / np.sqrt(R)
            b_, bse = pF[:, j].mean(), pF[:, j].std(ddof=1) / np.sqrt(NT)
            c_, cse = mC[:, j].mean(), mC[:, j].std(ddof=1) / np.sqrt(R)
            e_, ese = pCr[:, j].mean(), pCr[:, j].std(ddof=1) / np.sqrt(NC)
            g_, gse = mG[:, j].mean(), mG[:, j].std(ddof=1) / np.sqrt(R)
            EF = EcvF_occ[:, j].mean(); EC = EcvC_occ[:, j].mean()
            gm = g_ + EF - EC; gp = (b_ - e_) + EF - EC
            regret_rows.append((S, d, a, Ls, n, a_, ase, b_, bse, c_, cse, e_, ese, gm, gse, gp, covC[:, j].mean(), w1[:, j].mean()))
            dF.append(b_ - a_); dC.append(e_ - c_)
        # crossing and its statistical error: slope of the mean gap around the crossing
        xm, xp = r["x_meas"], r["x_pred"]
        gm_arr = np.array(r["gap_meas"]); ns_arr = np.array(ns, float)
        xse = None
        if xm is not None:
            i = max(j for j in range(len(ns) - 1) if ns_arr[j] <= xm)
            slope = (gm_arr[i + 1] - gm_arr[i]) / (ns_arr[i + 1] - ns_arr[i])
            gse_local = 0.5 * (mG[:, i].std(ddof=1) + mG[:, i + 1].std(ddof=1)) / np.sqrt(R)
            xse = gse_local / abs(slope)
        cross_rows.append((S, d, a, Ls, R, xm, xse, xp))
        summary[S] = dict(d=d, alpha=a, Ls=Ls, R=R, NT=int(NT), NC=int(NC), ns=ns, x_meas=xm, x_se=xse, x_pred=xp,
                          dF=dF, dC=dC, w1=w1.mean(axis=0).tolist(), covC=covC.mean(axis=0).tolist(),
                          regretF=[row[5] for row in regret_rows if row[0] == S],
                          regretC=[row[9] for row in regret_rows if row[0] == S])
    # regret table
    out = ["\\begin{tabular}{rrrr rr rr rrr}", "\\toprule",
           " & & & & \\multicolumn{2}{c}{\\unigram regret} & \\multicolumn{2}{c}{copier regret} & \\multicolumn{3}{c}{gap $\\ell_F-\\ell_C$} \\\\",
           "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}",
           "$d$ & $\\alpha$ & $L_{\\mathrm{s}}$ & $n$ & sim. & formula & sim. & formula & sim. & SE & formula \\\\", "\\midrule"]
    last = None
    for row in regret_rows:
        S, d, a, Ls, n, a_, ase, b_, bse, c_, cse, e_, ese, gm, gse, gp, cov, w1m = row
        if last is not None and last != S:
            out.append("\\addlinespace")
        last = S
        out.append(f"{d} & {fmt_alpha(a)} & {Ls} & {n} & {a_:.1f} & {b_:.1f} & {c_:.1f} & {e_:.1f} & {gm:.0f} & {gse:.0f} & {gp:.0f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    regret_tex = "\n".join(out)
    # crossings of Section 3 (results/mechanism/crossing_count_comparison_table.tex): prediction, new chains
    SEC3 = {(2, 0.75, 128): (1795, 1780), (2, 0.75, 256): (6392, 6420), (2, 1.0, 128): (1452, 1699), (2, 1.0, 256): (5874, 5928),
            (3, 0.75, 128): (415, 414), (3, 0.75, 256): (1964, 1965), (3, 1.0, 128): (568, 491), (3, 1.0, 256): (2011, 1985)}
    out = ["\\begin{tabular}{rrrrrrrr}", "\\toprule",
           "$d$ & $\\alpha$ & $L_{\\mathrm{s}}$ & paths & simulation & building block & error & Section~\\ref{sec:crossing-prediction} \\\\", "\\midrule"]
    for S, d, a, Ls, R, xm, xse, xp in cross_rows:
        sm = f"{xm:.0f} $\\pm$ {xse:.0f}" if xm is not None else "--"
        sp = f"{xp:.0f}" if xp is not None else "--"
        er = f"{100 * (xp - xm) / xm:+.1f}\\%" if (xm is not None and xp is not None) else "--"
        s3 = SEC3.get((Ls, a, d), (None, None))[0]
        s3s = f"{s3}" if s3 is not None else "--"
        out.append(f"{d} & {fmt_alpha(a)} & {Ls} & {R} & {sm} & {sp} & {er} & {s3s} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    cross_tex = "\n".join(out)
    return regret_tex, cross_tex, summary


if __name__ == "__main__":
    single_tex, single_rows = single_table()
    regret_tex, cross_tex, summary = chain_tables()
    os.makedirs("tables_out", exist_ok=True)
    for name, tex in [("single_table.tex", single_tex), ("chain_regret_table.tex", regret_tex), ("crossing_table.tex", cross_tex)]:
        with open(os.path.join("tables_out", name), "w") as f:
            f.write("% generated by make_tables.py\n" + tex + "\n")
    summary["single"] = single_rows
    json.dump(summary, open("bb_summary.json", "w"), indent=1)
    print(single_tex); print(); print(regret_tex); print(); print(cross_tex)
    for S, v in summary.items():
        if S == "single":
            continue
        print(S, "unigram diff (formula - sim):", np.round(v["dF"], 1), " copier diff:", np.round(v["dC"], 1),
              " w1:", np.round(v["w1"], 4), " covC:", np.round(v["covC"], 1), " crossing:", v["x_meas"], v["x_se"], v["x_pred"])
