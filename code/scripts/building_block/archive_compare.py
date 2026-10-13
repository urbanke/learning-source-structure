"""Compare the building-block predictions (tab_A_<tag>.json, table Monte Carlo at the archive's checkpoints)
with the archived chain runs (sim_A_<tag>.json from archive_extract.py). Same paired comparison as compare.py,
restricted to the common checkpoints; the covariance term is centred at the exact mean row entropy.
Writes archive_summary.json and results tables (tables_out/archive_*.tex)."""
import json, os, sys
import numpy as np
import lsa_cost as C
from compare import crossing
from info_gain import exact_EHP

TAGS = ["L2_a0.75_d128", "L2_a1_d128", "L2_a1.25_d128", "L3_a0.75_d128", "L3_a1_d128", "L3_a1.25_d128",
        "L2_a0.75_d256", "L2_a1_d256", "L2_a1.25_d256", "L3_a0.75_d256", "L3_a1_d256", "L3_a1.25_d256",
        "L2_a0.75_d512", "L2_a1_d512", "L2_a1.25_d512", "L3_a0.75_d512", "L3_a1_d512", "L3_a1.25_d512"]
ARCH = "/mnt/user-data/uploads/layered_architectures/output/crossing_finite_count_prediction"
_EHFILE = "exact_EH_archive.json"   # cache of the exact mean row entropies (bits), keyed "d,alpha,Ls"
_EH = json.load(open(_EHFILE)) if os.path.exists(_EHFILE) else {}


def exact_EH(d, alpha, Ls):
    key = f"{d},{alpha:g},{Ls}"
    if key not in _EH:
        _EH[key] = exact_EHP(C.zipf(d, alpha), Ls) / np.log(2.0)
        json.dump(_EH, open(_EHFILE, "w"), indent=1)
    return _EH[key]


def _s(x, nd=1):
    """signed number in math mode without negative zero"""
    v = round(float(x), nd) + 0.0
    return f"${v:+.{nd}f}$"


def _rng(a, b, nd=1):
    """error range as an interval [a, b] in math mode"""
    a = round(float(a), nd) + 0.0; b = round(float(b), nd) + 0.0
    return f"$[{a:+.{nd}f},{b:+.{nd}f}]$"


TAB = os.environ.get("TAB_PREFIX", "tab_A")      # prediction files to compare (tab_A: formula grids; tab_B: exact small-k blocks)
OUT = os.environ.get("OUT_SUFFIX", "")


def report(tag, verbose=True):
    simfile = f"sim_AX_{tag}.json" if os.path.exists(f"sim_AX_{tag}.json") else f"sim_A_{tag}.json"
    s = json.load(open(simfile)); t = json.load(open(f"{TAB}_{tag}.json"))
    ns_t = t["checkpoints"]; ns_s = s["checkpoints"]
    idx = [ns_s.index(n) for n in ns_t]          # archive columns used
    ns = ns_t
    lF = np.array(s["lF"])[:, idx]; lC = np.array(s["lC"])[:, idx]; cvF = np.array(s["cvF"])[:, idx]; cvC = np.array(s["cvC"])[:, idx]
    R = lF.shape[0]
    mF = lF - cvF; mC = lC - cvC; mG = mF - mC
    pU = np.array(t["predF_U"]); pe = np.array(t["predF_eta"]); pC = np.array(t["predC"])
    EcvF = np.array(t["EcvF"]); EcvC_occ = np.array(t["EcvC_occ"]); EcvF_occ = np.array(t["EcvF_occ"])
    covC = np.array(t["covC"]) + (np.array(ns, float)[None, :] - 1.0) * (t["Hbar"] - exact_EH(s["d"], s["alpha"], s["Ls"]))
    pF = pU - pe - EcvF; pCr = pC + covC - EcvC_occ
    RK = os.environ.get("RK_PREFIX")                # optional row-known correction files (table_rk.py), added to the copier
    if RK and os.path.exists(f"{RK}_{tag}.json"):
        rk = json.load(open(f"{RK}_{tag}.json")); rkC = np.array(rk["rkC"])
        assert rk["checkpoints"] == list(ns), "checkpoints of the correction differ"
        pCr = pCr + rkC.mean(axis=0)[None, :]         # mean correction (its own SE is small; not added to the table's SE)
    NT, NC = pF.shape[0], pCr.shape[0]
    rows = []; gm = []; gp = []
    for j, n in enumerate(ns):
        a, ase = mF[:, j].mean(), mF[:, j].std(ddof=1) / np.sqrt(R)
        b, bse = pF[:, j].mean(), pF[:, j].std(ddof=1) / np.sqrt(NT)
        c, cse = mC[:, j].mean(), mC[:, j].std(ddof=1) / np.sqrt(R)
        e, ese = pCr[:, j].mean(), pCr[:, j].std(ddof=1) / np.sqrt(NC)
        g, gse = mG[:, j].mean(), mG[:, j].std(ddof=1) / np.sqrt(R)
        EF = EcvF_occ[:, j].mean(); EC = EcvC_occ[:, j].mean()
        gm.append(g + EF - EC); gp.append((b - e) + EF - EC)
        rows.append(dict(n=n, F_meas=a, F_se=ase, F_pred=b, F_pse=bse, C_meas=c, C_se=cse, C_pred=e, C_pse=ese,
                         gap_meas=g + EF - EC, gap_se=gse, gap_pred=(b - e) + EF - EC, covC=float(covC[:, j].mean())))
    xm, xp = crossing(ns, gm), crossing(ns, gp)
    xse = None
    if xm is not None:
        ns_arr = np.array(ns, float); i = max(j for j in range(len(ns) - 1) if ns_arr[j] <= xm)
        slope = (gm[i + 1] - gm[i]) / (ns_arr[i + 1] - ns_arr[i])
        xse = 0.5 * (mG[:, i].std(ddof=1) + mG[:, i + 1].std(ddof=1)) / np.sqrt(R) / abs(slope)
    # the archive's own numbers: paired chain crossing (raw means), Section 3 count prediction, earlier archived chains
    j = json.load(open(f"{ARCH}/{tag}.json"))
    x_chain = j["chain"]["mean_curve_crossing"]["value"]; x_count = j["count"]["mean_curve_crossing"]["value"]; x_arch = j["archived_mean_curve_crossing"]
    out = dict(tag=tag, d=s["d"], alpha=s["alpha"], Ls=s["Ls"], R=R, R_archive=s.get("R_archive", R), NT=int(NT), NC=int(NC), ns=ns, rows=rows,
               x_meas=xm, x_se=xse, x_pred=xp, x_chain_archive=x_chain, x_count_archive=x_count, x_earlier_archive=x_arch)
    if verbose:
        print(f"=== {tag}: d={s['d']} alpha={s['alpha']} Ls={s['Ls']}  paths R={R} (archive {s.get('R_archive', R)}); prediction tables {NT}/{NC}")
        print("     n | unigram regret meas(SE) pred(SE) diff | copier regret meas(SE) pred(SE) diff | gap meas(SE) pred diff | covC")
        for r in rows:
            print(f"{r['n']:6d} | {r['F_meas']:8.1f}({r['F_se']:4.1f}) {r['F_pred']:8.1f}({r['F_pse']:3.1f}) {r['F_pred']-r['F_meas']:+6.1f} | "
                  f"{r['C_meas']:8.1f}({r['C_se']:5.1f}) {r['C_pred']:8.1f}({r['C_pse']:4.1f}) {r['C_pred']-r['C_meas']:+7.1f} | "
                  f"{r['gap_meas']:8.1f}({r['gap_se']:5.1f}) {r['gap_pred']:8.1f} {r['gap_pred']-r['gap_meas']:+7.1f} | {r['covC']:7.1f}")
        print(f"   crossing: archive chains (CV) {xm:.0f} +- {xse:.0f}; building block {xp:.0f} ({100*(xp-xm)/xm:+.1f}%); "
              f"archive's own chain crossing {x_chain:.0f}; Section 3 count prediction {x_count:.0f} ({100*(x_count-xm)/xm:+.1f}%); earlier archive {x_arch:.0f}"
              if (xm and xp) else f"   crossing: measured {xm}, predicted {xp}")
    return out


if __name__ == "__main__":
    tags = [a for a in sys.argv[1:]] or [t for t in TAGS if os.path.exists(f"sim_A_{t}.json") and os.path.exists(f"{TAB}_{t}.json")]
    summ = {}
    for tag in tags:
        summ[tag] = report(tag)
    json.dump(summ, open(f"archive_summary{OUT}.json", "w"), indent=1)
    # crossing table for the appendix
    os.makedirs("tables_out", exist_ok=True)
    lines = ["\\begin{tabular}{rrrrrrrrrr}", "\\toprule",
             "$d$ & $\\alpha$ & $L_{\\mathrm{s}}$ & paths & chains & raw & block & error & count & error \\\\", "\\midrule"]
    for tag in TAGS:
        if tag not in summ: continue
        v = summ[tag]
        if v["x_meas"] is None or v["x_pred"] is None: continue
        lines.append(f"{v['d']} & {v['alpha']:g} & {v['Ls']} & {v['R']} & {v['x_meas']:.0f} $\\pm$ {v['x_se']:.0f} & {v['x_chain_archive']:.0f} & {v['x_pred']:.0f} & {_s(100*(v['x_pred']-v['x_meas'])/v['x_meas'])}\\% & "
                     f"{v['x_count_archive']:.0f} & {_s(100*(v['x_count_archive']-v['x_meas'])/v['x_meas'])}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"tables_out/archive_crossing_table{OUT}.tex", "w").write("% generated by archive_compare.py\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    # regret summary table: one line per setting (errors over all lengths of the archive grid)
    lines = ["\\begin{tabular}{rrrrrrrrrr}", "\\toprule",
             "$d$ & $\\alpha$ & $L_{\\mathrm{s}}$ & \\unigram regret & error & SE & copier regret & error & SE & error/regret \\\\", "\\midrule"]
    stats = {}
    for tag in TAGS:
        if tag not in summ: continue
        v = summ[tag]; rows = v["rows"]
        Ferr = np.array([r["F_pred"] - r["F_meas"] for r in rows]); Cerr = np.array([r["C_pred"] - r["C_meas"] for r in rows])
        Cpct = 100 * Cerr / np.array([r["C_meas"] for r in rows])
        Fse = max(r["F_se"] for r in rows); Cse = max(r["C_se"] for r in rows)
        stats[tag] = dict(Ferr_max=float(np.abs(Ferr).max()), Fse_max=float(Fse), Cerr_min=float(Cerr.min()), Cerr_max=float(Cerr.max()),
                          Cpct_min=float(Cpct.min()), Cpct_max=float(Cpct.max()), Cse_max=float(Cse),
                          F_reg=(rows[0]["F_meas"], rows[-1]["F_meas"]), C_reg=(rows[0]["C_meas"], rows[-1]["C_meas"]))
        lines.append(f"{v['d']} & {v['alpha']:g} & {v['Ls']} & "
                     f"{rows[0]['F_meas']:.0f}--{rows[-1]['F_meas']:.0f} & {_rng(Ferr.min(), Ferr.max())} & {Fse:.1f} & "
                     f"{rows[0]['C_meas']:.0f}--{rows[-1]['C_meas']:.0f} & {_rng(Cerr.min(), Cerr.max(), 0)} & {Cse:.0f} & {_rng(Cpct.min(), Cpct.max())}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"tables_out/archive_regret_table{OUT}.tex", "w").write("% generated by archive_compare.py\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump(stats, open(f"archive_stats{OUT}.json", "w"), indent=1)
