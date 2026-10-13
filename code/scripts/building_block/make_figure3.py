"""Main-text figure. Top: crossing length against alphabet size, simulation (markers, two standard errors) and
prediction (lines), for source depths 2 and 3 (left) and 6 and 10 (right). Bottom: gap per token against n for three
settings at d = 512. Reads summary_all.json; writes crossing_figure.pdf/png."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = json.load(open(os.environ.get("SUMMARY", "summary_all.json")))
COL = {0.75: "C0", 1.0: "C1", 1.25: "C2", 1.5: "C4", 2.0: "C3"}
LS = {2: "-", 3: "--", 6: "-", 10: "--"}; MK = {2: "o", 3: "s", 6: "o", 10: "s"}
BOTTOM = [("L2_a1_d512", "$L_{\\mathrm{s}}=2$, $\\alpha=1$"), ("L6_a1.5_d512", "$L_{\\mathrm{s}}=6$, $\\alpha=1.5$"),
          ("L3_a2_d512", "$L_{\\mathrm{s}}=3$, $\\alpha=2$")]

fig = plt.figure(figsize=(5.5, 4.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0], hspace=0.55, wspace=1.2)
axA = fig.add_subplot(gs[0, 0:3]); axB = fig.add_subplot(gs[0, 3:6])
axs = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]


def series(ax, Lss, legend_loc="upper left"):
    ok = lambda v: v["x_meas"] is not None and v["x_pred"] is not None
    combos = sorted({(v["Ls"], v["alpha"]) for v in S.values() if v["Ls"] in Lss and ok(v)})
    for Ls, a in combos:
        vs = sorted([v for v in S.values() if v["Ls"] == Ls and v["alpha"] == a and ok(v)], key=lambda v: v["d"])
        ds = np.array([v["d"] for v in vs], float); xm = np.array([v["x_meas"] for v in vs]); se = np.array([v["x_se"] for v in vs])
        xp = np.array([v["x_pred"] for v in vs])
        ax.plot(ds, xp, LS[Ls], color=COL[a], lw=1.0)
        ax.errorbar(ds, xm, yerr=2 * se, fmt=MK[Ls], ms=3, color=COL[a], mfc="white", mew=0.9, lw=0.8, capsize=1.5)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([128, 256, 512]); ax.set_xticklabels(["128", "256", "512"]); ax.minorticks_off()
    ax.set_xlabel("$d$", fontsize=7.5, labelpad=1); ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    alphas = sorted({a for _, a in combos})
    h = [Line2D([], [], color=COL[a], lw=1.0, label=f"$\\alpha={a:g}$") for a in alphas]
    h += [Line2D([], [], color="0.3", ls=LS[Ls], marker=MK[Ls], ms=3, mfc="white", lw=1.0, label=f"$L_{{\\mathrm{{s}}}}={Ls}$") for Ls in sorted({L for L, _ in combos})]
    ax.legend(handles=h, fontsize=6, frameon=False, loc=legend_loc, ncol=2, handlelength=1.8, columnspacing=0.8, labelspacing=0.3)


series(axA, {2, 3}); series(axB, {6, 10}, legend_loc="lower left")
axA.set_ylabel("crossing $n^*_d$ [tokens]", fontsize=7.5, labelpad=1)
axA.set_title("source depths 2 and 3", fontsize=7.5); axB.set_title("source depths 6 and 10", fontsize=7.5)
for ax, (tag, title) in zip(axs, BOTTOM):
    v = S[tag]; ns = np.array([r["n"] for r in v["rows"]], float)
    gm = np.array([r["gap_meas"] for r in v["rows"]]); se = np.array([r["gap_se"] for r in v["rows"]]); gp = np.array([r["gap_pred"] for r in v["rows"]])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.errorbar(ns, gm / ns, yerr=2 * se / ns, fmt="o", ms=2.5, lw=0.8, color="C0", label="simulation")
    ax.plot(ns, gp / ns, "-", lw=1.0, color="C3", label="prediction")
    ax.set_title(f"{title}, {v['R']} paths", fontsize=7); ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax.set_xlabel("$n$", fontsize=7.5, labelpad=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axs[0].set_ylabel("gap per token [bits]", fontsize=7, labelpad=1)
axs[0].legend(fontsize=6, loc="upper left", frameon=False)
plt.savefig("crossing_figure.pdf", bbox_inches="tight"); plt.savefig("crossing_figure.png", dpi=120, bbox_inches="tight")
print("done")
