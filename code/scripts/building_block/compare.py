"""Paired comparison: simulated costs vs building-block predictions, with control variates."""
import sys, json
import numpy as np


def load(path):
    return json.load(open(path))


def crossing(ns, gap):
    """Last sign change of gap(n) by linear interpolation in n; None if no crossing."""
    ns = np.asarray(ns, float); gap = np.asarray(gap, float)
    idx = [i for i in range(len(ns) - 1) if gap[i] <= 0 < gap[i + 1] or gap[i] < 0 <= gap[i + 1]]
    if not idx:
        return None
    i = idx[-1]
    return ns[i] + (ns[i + 1] - ns[i]) * (-gap[i]) / (gap[i + 1] - gap[i])


def report(simfile, tabfile, label=""):
    s = load(simfile); t = load(tabfile)
    ns = s["checkpoints"]
    assert ns == t["checkpoints"], (ns, t["checkpoints"])
    lF = np.array(s["lF"]); lC = np.array(s["lC"]); cvF = np.array(s["cvF"]); cvC = np.array(s["cvC"])
    R = lF.shape[0]
    # measured regret-like quantities (paths), with SE
    mF = lF - cvF; mC = lC - cvC; mG = mF - mC
    # predicted (tables)
    pU = np.array(t["predF_U"]); pe = np.array(t["predF_eta"]); pC = np.array(t["predC"])
    EcvF = np.array(t["EcvF"]); EcvC = np.array(t["EcvC"])
    EcvF_occ = np.array(t["EcvF_occ"]); EcvC_occ = np.array(t["EcvC_occ"])
    NO = EcvF_occ.shape[0]
    pC = pC[:NO]
    covC = np.array(t.get("covC", np.zeros_like(pC)))[:pC.shape[0]]
    # covC was computed with a Monte Carlo estimate of the mean row entropy; re-centre at the exact value
    # (sum_a E[K_a|T] = n-1 exactly, so the correction is (n-1)(Hbar_mc - H_exact) per checkpoint)
    import os
    if os.path.exists("exact_EH.json") and label in json.load(open("exact_EH.json")):
        H_exact = json.load(open("exact_EH.json"))[label]
        covC = covC + (np.array(ns, float)[None, :] - 1.0) * (t["Hbar"] - H_exact)
    COVFAC = float(os.environ.get("COVFAC", "1.0"))
    pC = pC + COVFAC * covC
    pF = pU - pe - EcvF; pCr = pC - EcvC_occ[:pC.shape[0]]
    NT = pU.shape[0]
    NC = pCr.shape[0]
    pG = None
    print(f"=== {label}  d={s['d']} alpha={s['alpha']} Ls={s['Ls']} Lmax={s['Lmax']}  paths R={R}, unigram prediction tables={NT}, copier prediction tables={NC}, occupation tables={NO}")
    print("   n | unigram regret: meas(SE)  pred(SE)  diff | copier regret: meas(SE) pred(SE) diff | gap: meas(SE) pred(SE) diff | cost F meas pred | cost C meas pred")
    gm = []; gp = []
    for j, n in enumerate(ns):
        a, ase = mF[:, j].mean(), mF[:, j].std(ddof=1) / np.sqrt(R)
        b, bse = pF[:, j].mean(), pF[:, j].std(ddof=1) / np.sqrt(NT)
        c, cse = mC[:, j].mean(), mC[:, j].std(ddof=1) / np.sqrt(R)
        e, ese = pCr[:, j].mean(), pCr[:, j].std(ddof=1) / np.sqrt(NC)
        g, gse = mG[:, j].mean(), mG[:, j].std(ddof=1) / np.sqrt(R)
        h, hse = b - e, np.sqrt(bse**2 + ese**2)
        EF = EcvF_occ[:, j].mean(); EC = EcvC_occ[:, j].mean()
        gm.append(g + EF - EC); gp.append(h + EF - EC)
        print(f"{n:6d} | {a:9.1f}({ase:4.1f}) {b:9.1f}({bse:4.1f}) {b-a:+7.1f} | {c:9.1f}({cse:4.1f}) {e:9.1f}({ese:4.1f}) {e-c:+7.1f} | {g+EF-EC:9.1f}({gse:4.1f}) {h+EF-EC:9.1f}({hse:4.1f}) {h-g:+7.1f} | {a+EF:9.1f} {b+EF:9.1f} | {c+EC:9.1f} {e+EC:9.1f}")
    xm = crossing(ns, gm); xp = crossing(ns, gp)
    print(f"   crossing (linear interpolation of the mean gap): measured {xm}, predicted {xp}" + (f", relative error {100*(xp-xm)/xm:+.1f}%" if xm and xp else ""))
    return dict(ns=ns, gap_meas=gm, gap_pred=gp, x_meas=xm, x_pred=xp)


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
