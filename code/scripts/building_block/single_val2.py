"""Single-distribution truths, recomputed: Monte Carlo of E c(m) (row ~ tilted Zipf, m ~ Mult(k, row))
with a control variate that is linear in the count profile and centred at the EXACT mean profile
(the scale integral), so that the estimator is unbiased.  Compares with the building block."""
import sys, json, time, numpy as np
import bb, lsa_cost as C, lsa_cost2 as C2
import lsa_fast  # noqa: spline-free evaluator (replaces bb.LSAx)

LN2 = np.log(2.0)


def exact_mean_profile(lsa, q, Ls, k, xhalf, xstep=0.05):
    xgrid = np.arange(-xhalf, xhalf + 1e-9, xstep)
    acc = np.zeros(k + 1); tot = 0.0
    for x in xgrid:
        s = k * np.exp(x)
        pm = bb.pmfs_at_scale(lsa, q, Ls, s, k)
        ptil, PS = bb.conditioned_profile(pm, k)
        w = k * PS * xstep
        if w <= 0:
            continue
        acc += w * ptil.sum(axis=0); tot += w
    return acc / tot, tot


def truth(d, alpha, Ls, k, Lmax, ndraw, seed, xhalf):
    rng = np.random.default_rng(seed)
    q = C.zipf(d, alpha); lsa = bb.LSAx(Lmax, rmax=k + 5)
    t0 = time.time()
    nu, mass = exact_mean_profile(lsa, q, Ls, k, xhalf)
    rs = [r for r in range(k + 1) if nu[r] > 1e-12]
    nu_d = {r: float(nu[r]) for r in rs}
    g, _ = C2.grad_hess(lsa, nu_d, k, rs)          # E_mu[l_r] at the exact mean profile (nats)
    gmap = dict(zip(rs, g))
    t1 = time.time()
    vals = np.empty(ndraw); raw = np.empty(ndraw)
    for i in range(ndraw):
        p = C.sample_source(q, Ls, rng)
        m = rng.multinomial(k, p)
        prof = C.profile_of_counts(m)
        c = lsa.cost_bits(prof, k)
        corr = sum((prof.get(r, 0) - nu_d.get(r, 0.0)) * gmap.get(r, 0.0) for r in set(prof) | set(rs)) / LN2
        vals[i] = c + corr; raw[i] = c
    return dict(d=d, alpha=alpha, Ls=Ls, k=k, Lmax=Lmax, ndraw=ndraw, seed=seed, mass=mass,
                mc=float(vals.mean()), se=float(vals.std(ddof=1) / np.sqrt(ndraw)),
                mc_raw=float(raw.mean()), se_raw=float(raw.std(ddof=1) / np.sqrt(ndraw)),
                t_profile=t1 - t0, t_mc=time.time() - t1)


if __name__ == "__main__":
    cases = [(256, 0.75, 2, 16, 26), (256, 0.75, 2, 64, 26), (256, 0.75, 2, 256, 26), (256, 1.0, 2, 64, 26),
             (256, 0.5, 2, 64, 26), (256, 0.75, 3, 64, 26), (64, 0.5, 2, 128, 20), (256, 1.5, 6, 64, 26),
             (128, 0.75, 2, 8, 23), (128, 0.75, 2, 32, 23), (128, 0.75, 2, 128, 23)]
    ndraw = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = sys.argv[2] if len(sys.argv) > 2 else "single_truth2.json"
    pred = {(r["d"], r["alpha"], r["Ls"], r["k"]): r for r in json.load(open("single_val.json"))}
    rows = []
    xhalf_arg = float(sys.argv[3]) if len(sys.argv) > 3 else None
    for (d, a, Ls, k, Lmax) in cases:
        xhalf = (xhalf_arg if Ls <= 3 else 4.5) if xhalf_arg else (1.6 if Ls <= 2 else 2.4 if Ls <= 3 else 3.5)
        seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
        r = truth(d, a, Ls, k, Lmax, ndraw, seed=seed0 + k + int(100 * a) + Ls, xhalf=xhalf)
        pr = pred[(d, a, Ls, k)]
        if k <= 32 or Ls >= 6:
            # recompute the formula with the same (wide) scale window; cheap at small k
            q = C.zipf(d, a); lsa = bb.LSAx(Lmax, rmax=k + 5)
            rr = bb.building_block(lsa, q, Ls, k, xhalf=xhalf)
            pr = dict(pred=rr["cost"], U=rr["U"], eta=rr["eta"])
        r.update(pred=pr["pred"], U=pr["U"], eta=pr["eta"], err=pr["pred"] - r["mc"])
        rows.append(r)
        print(f"d={d} a={a} Ls={Ls} k={k}: MC {r['mc']:.3f}+-{r['se']:.3f} (raw {r['mc_raw']:.2f}+-{r['se_raw']:.2f}, mass {r['mass']:.5f}) "
              f"pred {pr['pred']:.3f} err {r['err']:+.3f} [{r['t_profile']:.0f}s + {r['t_mc']:.0f}s]", flush=True)
        json.dump(rows, open(out, "w"), indent=1)
