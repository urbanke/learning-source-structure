"""Paired comparison of the building-block predictions with the simulated chains, for all settings: the eighteen
archived settings of output/crossing_finite_count_prediction and the extension runs of output/crossing_lsa_extension
and output/crossing_lsa_alpha2 (Appendix E), plus the extra paths of chain_sim.py where present (sim_AX_*).

Estimators. Gap: the raw paired gap l_F - l_C per path (GAP=raw, default), or with the fixed-coefficient control
variates of archive_compare.py (GAP=1), or with two regression coefficients (GAP=reg). Regrets: cost minus the exact
conditional mean of the entropy term, with the control variate applied with coefficient CV_F (default 1: the unigram
control variate reduces the variance several-fold) and CV_C (default 0: the copier control variate does not help;
"reg" fits the coefficient). The crossing is the last zero of the mean gap by linear interpolation; its standard error
is the standard error of the gap at the bracketing lengths divided by the slope.
Inputs: sim_A(X)_<tag>.json, tab_<TAB_PREFIX>_<tag>.json (default tab_B), exact_EH_archive.json.
Output: summary_all<OUT_SUFFIX>.json. Usage: python3 compare_all.py [tag ...]"""
import json, os, sys
import numpy as np
import lsa_cost as C
from compare import crossing
from info_gain import exact_EHP

ARCHIVE_TAGS = ["L2_a0.75_d128", "L2_a1_d128", "L2_a1.25_d128", "L3_a0.75_d128", "L3_a1_d128", "L3_a1.25_d128",
                "L2_a0.75_d256", "L2_a1_d256", "L2_a1.25_d256", "L3_a0.75_d256", "L3_a1_d256", "L3_a1.25_d256",
                "L2_a0.75_d512", "L2_a1_d512", "L2_a1.25_d512", "L3_a0.75_d512", "L3_a1_d512", "L3_a1.25_d512"]
EXT_TAGS = ["L6_a0.5_d256", "L6_a1_d256", "L6_a1.5_d256", "L6_a2_d256", "L3_a2_d256", "L10_a1.5_d256",
            "L6_a0.5_d512", "L6_a1_d512", "L6_a1.5_d512", "L6_a2_d512", "L3_a2_d512", "L10_a1.5_d512"]
TAGS = ARCHIVE_TAGS + EXT_TAGS
_EHFILE = "exact_EH_archive.json"
_EH = json.load(open(_EHFILE)) if os.path.exists(_EHFILE) else {}
TAB = os.environ.get("TAB_PREFIX", "tab_B"); OUT = os.environ.get("OUT_SUFFIX", "")
CV_F = os.environ.get("CV_F", "1"); CV_C = os.environ.get("CV_C", "0"); GAP = os.environ.get("GAP", "raw")


def exact_EH(d, alpha, Ls):
    key = f"{d},{alpha:g},{Ls}"
    if key not in _EH:
        _EH[key] = exact_EHP(C.zipf(d, alpha), Ls) / np.log(2.0)
        json.dump(_EH, open(_EHFILE, "w"), indent=1)
    return _EH[key]


def adjusted(cost, cv, Ecv, mode):
    """cost - beta (cv - E[cv|T]) per length; beta = float(mode), or fitted for mode "reg". Returns values and betas."""
    R, J = cost.shape
    out = np.empty_like(cost); betas = np.zeros(J)
    for j in range(J):
        x = cv[:, j] - Ecv[:, j]; y = cost[:, j]
        if mode == "reg":
            vx = np.var(x, ddof=1)
            b = float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
        else:
            b = float(mode)
        out[:, j] = y - b * x; betas[j] = b
    return out, betas


def gap_paths(lF, lC, cvF, cvC, EcvF, EcvC):
    """paired gap per path and length, minus the same E[cv_F - cv_C | T] (so that the sampled-matrix mean can be added)"""
    g = lF - lC - (EcvF - EcvC)
    if GAP == "raw":
        return g
    xF = cvF - EcvF; xC = cvC - EcvC
    if GAP == "1":
        return g - xF + xC
    out = np.empty_like(g)
    for j in range(g.shape[1]):
        X = np.column_stack([xF[:, j], xC[:, j]]); Xc = X - X.mean(axis=0); gc = g[:, j] - g[:, j].mean()
        beta = np.linalg.lstsq(Xc, gc, rcond=None)[0]; out[:, j] = g[:, j] - X @ beta
    return out


def report(tag, verbose=True):
    simfile = f"sim_AX_{tag}.json" if os.path.exists(f"sim_AX_{tag}.json") else f"sim_A_{tag}.json"
    s = json.load(open(simfile)); t = json.load(open(f"{TAB}_{tag}.json"))
    ns_t = t["checkpoints"]; ns_s = s["checkpoints"]
    idx = [ns_s.index(n) for n in ns_t]; ns = ns_t
    lF = np.array(s["lF"])[:, idx]; lC = np.array(s["lC"])[:, idx]
    cvF = np.array(s["cvF"])[:, idx]; cvC = np.array(s["cvC"])[:, idx]; EcvF_s = np.array(s["EcvF"])[:, idx]; EcvC_s = np.array(s["EcvC"])[:, idx]
    R = lF.shape[0]
    # regrets per path: cost - E[cv|T] - beta (cv - E[cv|T]); with beta = 1 this is cost - cv as in archive_compare.py.
    # The gaps of both sides add the same E[cv_F - cv_C] over the prediction's sampled matrices (paired comparison).
    mF, bF = adjusted(lF - EcvF_s, cvF, EcvF_s, CV_F); mC, bC = adjusted(lC - EcvC_s, cvC, EcvC_s, CV_C)
    pU = np.array(t["predF_U"]); pe = np.array(t["predF_eta"]); pC = np.array(t["predC"])
    EcvF = np.array(t["EcvF"]); EcvC_occ = np.array(t["EcvC_occ"]); EcvF_occ = np.array(t["EcvF_occ"])
    covC = np.array(t["covC"]) + (np.array(ns, float)[None, :] - 1.0) * (t["Hbar"] - exact_EH(s["d"], s["alpha"], s["Ls"]))
    pF = pU - pe - EcvF; pCr = pC + covC - EcvC_occ
    NT, NC = pF.shape[0], pCr.shape[0]
    rows = []; gm = []; gp = []
    mG = gap_paths(lF, lC, cvF, cvC, EcvF_s, EcvC_s)   # paired gap per path (minus E[cv_F - cv_C | T])
    for j, n in enumerate(ns):
        a, ase = mF[:, j].mean(), mF[:, j].std(ddof=1) / np.sqrt(R)
        b, bse = pF[:, j].mean(), pF[:, j].std(ddof=1) / np.sqrt(NT)
        c, cse = mC[:, j].mean(), mC[:, j].std(ddof=1) / np.sqrt(R)
        e, ese = pCr[:, j].mean(), pCr[:, j].std(ddof=1) / np.sqrt(NC)
        g, gse = mG[:, j].mean(), mG[:, j].std(ddof=1) / np.sqrt(R)
        EF = EcvF_occ[:, j].mean(); EC = EcvC_occ[:, j].mean()
        gm.append(g + EF - EC); gp.append((b - e) + EF - EC)   # both gaps with the same E[cv_F - cv_C] over the sampled matrices
        rows.append(dict(n=n, F_meas=a, F_se=ase, F_pred=b, F_pse=bse, C_meas=c, C_se=cse, C_pred=e, C_pse=ese,
                         gap_meas=g + EF - EC, gap_se=gse, gap_pred=(b - e) + EF - EC, covC=float(covC[:, j].mean()),
                         betaF=float(bF[j]), betaC=float(bC[j])))
    xm, xp = crossing(ns, gm), crossing(ns, gp)
    xse = xpse = None
    ns_arr = np.array(ns, float)
    if xm is not None:
        i = max(j for j in range(len(ns) - 1) if ns_arr[j] <= xm)
        slope = (gm[i + 1] - gm[i]) / (ns_arr[i + 1] - ns_arr[i])
        xse = 0.5 * (mG[:, i].std(ddof=1) + mG[:, i + 1].std(ddof=1)) / np.sqrt(R) / abs(slope)
    if xp is not None:
        # statistical uncertainty of the predicted crossing from the sampled matrices (unigram and copier tables)
        i = max(j for j in range(len(ns) - 1) if ns_arr[j] <= xp)
        slope = (gp[i + 1] - gp[i]) / (ns_arr[i + 1] - ns_arr[i])
        gse = [np.sqrt(rows[j]["F_pse"] ** 2 + rows[j]["C_pse"] ** 2) for j in (i, i + 1)]
        xpse = 0.5 * (gse[0] + gse[1]) / abs(slope)
    out = dict(tag=tag, d=s["d"], alpha=s["alpha"], Ls=s["Ls"], R=R, R_archive=s.get("R_archive", R), NT=int(NT), NC=int(NC), ns=ns, rows=rows,
               x_meas=xm, x_se=xse, x_pred=xp, x_pse=xpse, archive=s.get("archive"), estimators=dict(CV_F=CV_F, CV_C=CV_C, GAP=GAP), tab=TAB)
    if verbose:
        print(f"=== {tag}: d={s['d']} alpha={s['alpha']} Ls={s['Ls']}  paths R={R} (archive {s.get('R_archive', R)}); tables {NT}/{NC}; CV_F {CV_F} CV_C {CV_C} GAP {GAP}")
        print("     n | unigram regret meas(SE) pred(SE) diff | copier regret meas(SE) pred(SE) diff | gap meas(SE) pred diff | betaF betaC")
        for r in rows:
            print(f"{r['n']:6d} | {r['F_meas']:8.1f}({r['F_se']:4.1f}) {r['F_pred']:8.1f}({r['F_pse']:3.1f}) {r['F_pred']-r['F_meas']:+6.1f} | "
                  f"{r['C_meas']:8.1f}({r['C_se']:5.1f}) {r['C_pred']:8.1f}({r['C_pse']:4.1f}) {r['C_pred']-r['C_meas']:+7.1f} | "
                  f"{r['gap_meas']:8.1f}({r['gap_se']:5.1f}) {r['gap_pred']:8.1f} {r['gap_pred']-r['gap_meas']:+7.1f} | {r['betaF']:5.2f} {r['betaC']:5.2f}")
        if xm and xp:
            print(f"   crossing: simulation {xm:.0f} +- {xse:.0f}; prediction {xp:.0f} +- {xpse:.0f} ({100*(xp-xm)/xm:+.1f}%)")
        else:
            print(f"   crossing: simulation {xm}, prediction {xp}")
    return out


if __name__ == "__main__":
    tags = [a for a in sys.argv[1:]] or [t for t in TAGS if os.path.exists(f"sim_A_{t}.json") and os.path.exists(f"{TAB}_{t}.json")]
    summ = {}
    for tag in tags:
        summ[tag] = report(tag)
    json.dump(summ, open(f"summary_all{OUT}.json", "w"), indent=1)
    print("written", f"summary_all{OUT}.json", len(summ), "settings")
