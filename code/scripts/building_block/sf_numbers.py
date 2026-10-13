"""Numbers for the accuracy text of the sampling-free appendix (from sf_compare.json)."""
import json, numpy as np
S = json.load(open("sf_compare.json"))
main = [t for t in S if S[t]["Ls"] <= 3 and S[t]["alpha"] <= 1.25 and S[t]["x_sf"] is not None]
deep = [t for t in S if t not in main]
def pct(a, b): return 100 * (a - b) / b
E_s = np.array([pct(S[t]["x_sf"], S[t]["x_sampled"]) for t in main]); E_m = np.array([pct(S[t]["x_sf"], S[t]["x_meas"]) for t in main])
E_ps = np.array([pct(S[t]["x_sampled"], S[t]["x_meas"]) for t in main])
print(f"{len(main)} main settings")
print(f"  sf vs sampled: max |.| {np.abs(E_s).max():.1f}%, median {np.median(np.abs(E_s)):.1f}%, mean {E_s.mean():+.1f}%, lower in {(E_s < 0).sum()} settings; range {E_s.min():+.1f} .. {E_s.max():+.1f}")
print(f"  sf vs sim:     max |.| {np.abs(E_m).max():.1f}%, median {np.median(np.abs(E_m)):.1f}%, mean {E_m.mean():+.1f}%")
print(f"  sampled vs sim (same settings): max |.| {np.abs(E_ps).max():.1f}%, median {np.median(np.abs(E_ps)):.1f}%, mean {E_ps.mean():+.1f}%")
SE = np.array([100 * np.sqrt(S[t]["x_se"] ** 2 + S[t]["x_pse"] ** 2) / S[t]["x_meas"] for t in main])
print(f"  sf within combined SE of sim: {(np.abs(E_m) <= SE).sum()}; sampled within: {(np.abs(E_ps) <= SE).sum()}")
for t in main:
    v = S[t]; i = max(j for j in range(len(v["ns"]) - 1) if v["ns"][j] <= v["x_sf"])
    C = 0.5 * (v["C_sampled"][i] + v["C_sampled"][i + 1]); F = 0.5 * (v["F_sampled"][i] + v["F_sampled"][i + 1])
    v["dCpct"] = 100 * v["dC"] / C; v["dFpct"] = 100 * v["dF"] / F
dC = np.array([S[t]["dC"] for t in main]); dF = np.array([S[t]["dF"] for t in main])
print(f"  copier at the crossing: sf - sampled from {dC.min():+.0f} to {dC.max():+.0f} bits ({min(S[t]['dCpct'] for t in main):+.3f} .. {max(S[t]['dCpct'] for t in main):+.3f} percent)")
print(f"  unigram at the crossing: sf - sampled from {dF.min():+.0f} to {dF.max():+.0f} bits ({min(S[t]['dFpct'] for t in main):+.3f} .. {max(S[t]['dFpct'] for t in main):+.3f} percent)")
secs = sorted(S[t]["seconds"] / 60 for t in main); print(f"  minutes per setting: {secs[0]:.0f} .. {secs[-1]:.0f}; by d:", {d: [round(S[t]["seconds"] / 60) for t in main if S[t]["d"] == d] for d in (128, 256, 512)})
print("deep / strong tilt:")
for t in deep:
    v = S[t]
    if v["x_sf"] is None: print("  ", t, "no crossing"); continue
    print(f"   {t:14s} sf {v['x_sf']:6.0f} sampled {v['x_sampled']:6.0f} sim {v['x_meas']:6.0f}: vs sampled {pct(v['x_sf'], v['x_sampled']):+.1f}%, vs sim {pct(v['x_sf'], v['x_meas']):+.1f}%; copier {v['dC']:+.0f} bits, unigram {v['dF']:+.0f} bits, slope {v['slope']:.3f}")
