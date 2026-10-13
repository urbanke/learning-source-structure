"""Building block for one of the extension settings: rate window measured at k = 64 (wide window), grid of the
formula with the window centred there, exact values for k <= 32 with matching margins, merged grid bb2_*.
Usage: python3 ext_grid.py d alpha Ls Lmax kmax"""
import sys, json, time, subprocess, os
import numpy as np
import lsa_cost as C, bb, lsa_fast  # noqa

d, alpha, Ls, Lmax, kmax = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
tag = f"d{d}_a{alpha:g}_L{Ls}"
q = C.zipf(d, alpha)
t0 = time.time()
# 1. the rate window from the law of x = ln(s/k) (tails below 1e-5 on each side) at k = 64 and k = kmax,
#    and for the exact small-k values at k = 2 and 8
lsa = bb.LSAx(Lmax, rmax=max(kmax, 70) + 5)
win = {k: bb.rate_window(lsa, q, Ls, k) for k in (2, 8, 64, min(kmax, 2000))}
for k, (lo, hi, m) in win.items():
    print(f"{tag}: rate window k={k}: [{lo:+.1f}, {hi:+.1f}] (mass on the wide grid {m:.5f})", flush=True)
lo = min(win[64][0], win[min(kmax, 2000)][0]) - 0.3; hi = max(win[64][1], win[min(kmax, 2000)][1]) + 0.3
xc = round(0.5 * (lo + hi), 2); xh = round(0.5 * (hi - lo), 2)
mlo = float(np.ceil(-min(win[2][0], win[8][0]) + 0.5)); mup = float(np.ceil(max(win[2][1], win[8][1]) + 0.5))
print(f"{tag}: window centre {xc:+.2f} half-width {xh:.2f} ({int(2*xh/0.1)+1} points); exact margins {mlo:.0f} {mup:.0f}; {time.time()-t0:.0f}s", flush=True)
# 2. formula grid
kgrid = [k for k in bb.KGRID if k <= kmax]
bb.bb_grid(d, alpha, Ls, Lmax, kgrid=kgrid, out=f"bb_{tag}.json", xhalf=xh, xcenter=xc)
# 3. exact small k and merge
subprocess.run(["python3", "bb_exact_small.py", str(d), str(alpha), str(Ls), str(Lmax), "32", f"bbx_{tag}.json", "0.05", str(mlo), str(mup)], check=True)
subprocess.run(["python3", "merge_exact.py", f"bb_{tag}.json", f"bbx_{tag}.json", f"bb2_{tag}.json"], check=True)
print(f"{tag}: all done in {time.time()-t0:.0f}s", flush=True)
