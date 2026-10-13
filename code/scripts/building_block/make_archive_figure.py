"""Figure: gap per token on the archived chains (with the extra paths at alpha = 1.25) against the building
block, all eighteen settings. Reads archive_summary.json written by archive_compare.py."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open(__import__("os").environ.get("SUMMARY", "archive_summary.json")))
ds = [128, 256, 512]; alphas = [0.75, 1.0, 1.25]; Lss = [2, 3]
rows = [(d, Ls) for d in ds for Ls in Lss]
fig, axes = plt.subplots(len(rows), len(alphas), figsize=(5.5, 8.6), squeeze=False)
from matplotlib.ticker import MaxNLocator
for i, (d, Ls) in enumerate(rows):
    for j, a in enumerate(alphas):
        ax = axes[i, j]
        tag = f"L{Ls}_a{a:g}_d{d}"
        v = S[tag]
        ns = np.array([r["n"] for r in v["rows"]], float)
        gm = np.array([r["gap_meas"] for r in v["rows"]]); se = np.array([r["gap_se"] for r in v["rows"]])
        gp = np.array([r["gap_pred"] for r in v["rows"]])
        ax.axhline(0, color="0.6", lw=0.8)
        ax.errorbar(ns, gm / ns, yerr=2 * se / ns, fmt="o", ms=2.2, lw=0.8, color="C0", label="archived chains")
        ax.plot(ns, gp / ns, "-", lw=1.0, color="C3", label="building block")
        ax.set_title(f"$d={d}$, $\\alpha={a:g}$, $L_{{\\mathrm{{s}}}}={Ls}$, {v['R']} paths", fontsize=6.5)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        if i == len(rows) - 1:
            ax.set_xlabel("$n$", fontsize=6.5, labelpad=1)
        if j == 0:
            ax.set_ylabel("gap per token [bits]", fontsize=6, labelpad=1)
axes[0, 0].legend(fontsize=5.5, loc="upper left", frameon=False)
plt.tight_layout(h_pad=0.4, w_pad=0.4)
plt.savefig("bb_gap_archive_figure.pdf"); plt.savefig("bb_gap_archive_figure.png", dpi=110)
print("done")
