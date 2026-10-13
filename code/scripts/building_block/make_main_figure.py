"""Main-text figure: gap per token against n at d = 256, rows L_s = 2, 3, columns alpha = 0.75, 1, 1.25;
simulation (points, two standard errors) and prediction (lines). Reads archive_summary.json."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

d = int(sys.argv[1]) if len(sys.argv) > 1 else 256
S = json.load(open(__import__("os").environ.get("SUMMARY", "archive_summary.json")))
alphas = [0.75, 1.0, 1.25]; Lss = [2, 3]
fig, axes = plt.subplots(len(Lss), len(alphas), figsize=(5.5, 3.4), squeeze=False)
for i, Ls in enumerate(Lss):
    for j, a in enumerate(alphas):
        ax = axes[i, j]; v = S[f"L{Ls}_a{a:g}_d{d}"]
        ns = np.array([r["n"] for r in v["rows"]], float)
        gm = np.array([r["gap_meas"] for r in v["rows"]]); se = np.array([r["gap_se"] for r in v["rows"]])
        gp = np.array([r["gap_pred"] for r in v["rows"]])
        ax.axhline(0, color="0.6", lw=0.8)
        ax.errorbar(ns, gm / ns, yerr=2 * se / ns, fmt="o", ms=2.5, lw=0.8, color="C0", label="simulation")
        ax.plot(ns, gp / ns, "-", lw=1.0, color="C3", label="prediction")
        ax.set_title(f"$\\alpha={a:g}$, $L_{{\\mathrm{{s}}}}={Ls}$, {v['R']} paths", fontsize=7.5)
        ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        if i == len(Lss) - 1:
            ax.set_xlabel("$n$", fontsize=7.5, labelpad=1)
        if j == 0:
            ax.set_ylabel("gap per token [bits]", fontsize=7, labelpad=1)
axes[0, 0].legend(fontsize=6.5, loc="upper left", frameon=False)
plt.tight_layout(h_pad=0.5, w_pad=0.5)
plt.savefig(f"crossing_gap_d{d}.pdf"); plt.savefig(f"crossing_gap_d{d}.png", dpi=110)
print("done")
