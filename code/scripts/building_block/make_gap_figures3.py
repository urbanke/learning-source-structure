"""Appendix figures: gap per token against n, simulation (points, two standard errors) and prediction (lines), for
all settings. Figure 1: the eighteen settings with source depths 2 and 3 (rows d x L_s, columns alpha). Figure 2: the
twelve settings with source depths 6 and 10 and alpha = 2 (rows d, columns (L_s, alpha)). Reads summary_all.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

S = json.load(open(os.environ.get("SUMMARY", "summary_all.json")))


def panel(ax, tag, last_row, first_col):
    v = S[tag]
    ns = np.array([r["n"] for r in v["rows"]], float)
    gm = np.array([r["gap_meas"] for r in v["rows"]]); se = np.array([r["gap_se"] for r in v["rows"]]); gp = np.array([r["gap_pred"] for r in v["rows"]])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.errorbar(ns, gm / ns, yerr=2 * se / ns, fmt="o", ms=2.2, lw=0.8, color="C0", label="simulation")
    ax.plot(ns, gp / ns, "-", lw=1.0, color="C3", label="prediction")
    ax.set_title(f"$d={v['d']}$, $\\alpha={v['alpha']:g}$, $L_{{\\mathrm{{s}}}}={v['Ls']}$\n{v['R']} paths", fontsize=6, linespacing=1.0)
    ax.tick_params(labelsize=5.5, length=2, pad=1.5)
    if ns.max() / ns.min() > 12:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if last_row: ax.set_xlabel("$n$", fontsize=6.5, labelpad=1)
    if first_col: ax.set_ylabel("gap per token [bits]", fontsize=6, labelpad=1)


# figure 1: source depths 2 and 3
ds = [128, 256, 512]; alphas = [0.75, 1.0, 1.25]; rows = [(d, Ls) for d in ds for Ls in (2, 3)]
fig, axes = plt.subplots(len(rows), 3, figsize=(5.5, 8.6), squeeze=False)
for i, (d, Ls) in enumerate(rows):
    for j, a in enumerate(alphas):
        panel(axes[i, j], f"L{Ls}_a{a:g}_d{d}", i == len(rows) - 1, j == 0)
axes[0, 0].legend(fontsize=5.5, loc="upper left", frameon=False)
plt.tight_layout(h_pad=0.4, w_pad=0.4); plt.savefig("gap_figure_depth23.pdf"); plt.savefig("gap_figure_depth23.png", dpi=100)
# figure 2: source depths 6 and 10, and alpha = 2 (3 rows x 4 columns)
order = [(256, 6, 0.5), (256, 6, 1.0), (256, 6, 1.5), (256, 6, 2.0), (256, 3, 2.0), (256, 10, 1.5),
         (512, 6, 0.5), (512, 6, 1.0), (512, 6, 1.5), (512, 6, 2.0), (512, 3, 2.0), (512, 10, 1.5)]
fig, axes = plt.subplots(3, 4, figsize=(5.5, 5.2), squeeze=False)
for k, (d, Ls, a) in enumerate(order):
    i, j = divmod(k, 4); tag = f"L{Ls}_a{a:g}_d{d}"
    if tag in S: panel(axes[i, j], tag, i == 2, j == 0)
    else: axes[i, j].set_visible(False)
axes[0, 0].legend(fontsize=5.5, loc="upper left", frameon=False)
plt.tight_layout(h_pad=0.5, w_pad=0.4); plt.savefig("gap_figure_deep.pdf"); plt.savefig("gap_figure_deep.png", dpi=100)
print("done")
