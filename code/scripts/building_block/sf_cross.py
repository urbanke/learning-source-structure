"""Sampling-free crossing of one setting without a prior estimate of n*: a coarse geometric sweep of the gap
(ratio 2 from nmin) until the gap is positive at two consecutive lengths, then a refinement of the last bracket
(three interior points, ratio 2^(1/4)) and linear interpolation.  Writes sfx_<tag>.json.
Usage: python3 sf_cross.py d alpha Ls bbfile out.json [nmin=16] [nmax=131072]"""
import sys, time, json
import numpy as np
from sf_predict import SFPredict

d, alpha, Ls = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
bbfile, out = sys.argv[4], sys.argv[5]
nmin = int(sys.argv[6]) if len(sys.argv) > 6 else 16
nmax = int(sys.argv[7]) if len(sys.argv) > 7 else 131072
t0 = time.time()
coarse = [n for n in (nmin * 2 ** j for j in range(40)) if n <= nmax]
sf = SFPredict(d, alpha, Ls, bbfile, [nmax], verbose=False)
print(f"setup {time.time()-t0:.0f}s, Hbar {sf.Hbar/np.log(2):.4f} bits", flush=True)
rows = {}


def gap_at(n):
    if n in rows:
        return rows[n]["gap"]
    t1 = time.time()
    C, cov, EK, VA = sf.copier(n)
    F = sf.unigram(n)
    rows[n] = dict(n=n, C=C + cov, C_bb=C, cov=cov, F=F, gap=F - C - cov)
    print(f"n={n:7d}: copier {C+cov:10.1f} unigram {F:10.1f} gap {F-C-cov:+9.1f}  [{time.time()-t1:.0f}s]", flush=True)
    return rows[n]["gap"]


# coarse sweep
seq = []
for n in coarse:
    seq.append((n, gap_at(n)))
    if len(seq) >= 3 and seq[-1][1] > 0 and seq[-2][1] > 0 and any(g <= 0 for _, g in seq[:-2]):
        break
    if len(seq) >= 2 and seq[-1][1] > 0 and seq[-2][1] > 0 and n >= 64 and not any(g <= 0 for _, g in seq):
        break   # positive from the start: no crossing above nmin
neg = [i for i, (n, g) in enumerate(seq) if g <= 0]
x = None
if neg and neg[-1] + 1 < len(seq):
    i = neg[-1]; lo, hi = seq[i][0], seq[i + 1][0]
    fine = sorted(set([lo, hi] + [int(round(lo * (hi / lo) ** (j / 4))) for j in (1, 2, 3)]))
    for n in fine:
        gap_at(n)
    pts = sorted(rows)
    gs = [rows[n]["gap"] for n in pts]
    j = max(k for k in range(len(pts)) if gs[k] <= 0)
    if j + 1 < len(pts):
        x = pts[j] + (0 - gs[j]) * (pts[j + 1] - pts[j]) / (gs[j + 1] - gs[j])
        slope = (gs[j + 1] - gs[j]) / (pts[j + 1] - pts[j])
    else:
        slope = None
else:
    slope = None
res = dict(d=d, alpha=alpha, Ls=Ls, bbfile=bbfile, nmin=nmin, nmax=nmax, x_sf=x, slope=slope, Hbar_bits=sf.Hbar / np.log(2),
           rows=[rows[n] for n in sorted(rows)], seconds=time.time() - t0)
json.dump(res, open(out, "w"), indent=1)
print(f"crossing (sampling-free): {x}   total {time.time()-t0:.0f}s", flush=True)
