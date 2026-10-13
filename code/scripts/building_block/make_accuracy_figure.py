"""Main-text Figure 1, the prediction against the simulation. Left: predicted against simulated crossing for every
setting with a located crossing (error bars: two standard errors), colour by alpha, marker by source depth, with the
diagonal. Right: the gap per token along the sequence for one family (d = 256, L_s = 2, alpha = 0.75, 1, 1.25), simulation
(points, two standard errors) against prediction (line). Reads summary_all.json; writes accuracy_figure.pdf/png."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = json.load(open(os.environ.get("SUMMARY", "summary_all.json")))
COL = {0.75: "C0", 1.0: "C1", 1.25: "C2", 1.5: "C4", 2.0: "C3"}
MK = {2: "o", 3: "s", 6: "^", 10: "D"}
FAMILY = [("L2_a0.75_d256", "$\\alpha=0.75$"), ("L2_a1_d256", "$\\alpha=1$"), ("L2_a1.25_d256", "$\\alpha=1.25$")]

fig = plt.figure(figsize=(5.6, 2.5))
gs = fig.add_gridspec(1, 21, wspace=3.2)
axA = fig.add_subplot(gs[0, 0:8]); axs = [fig.add_subplot(gs[0, 9:13]), fig.add_subplot(gs[0, 13:17]), fig.add_subplot(gs[0, 17:21])]

# --- left: predicted against simulated crossing
rows = [v for v in S.values() if v["x_meas"] is not None and v["x_pred"] is not None]
lo = 0.7 * min(min(v["x_meas"], v["x_pred"]) for v in rows); hi = 1.4 * max(max(v["x_meas"], v["x_pred"]) for v in rows)
axA.plot([lo, hi], [lo, hi], "-", color="0.6", lw=0.8, zorder=1)
for v in rows:
    axA.errorbar(v["x_meas"], v["x_pred"], xerr=2 * v["x_se"], yerr=2 * v["x_pse"], fmt=MK[v["Ls"]], ms=3.2, color=COL[v["alpha"]],
                 mfc="white", mew=0.9, lw=0.7, capsize=1.2, zorder=2)
axA.set_xscale("log"); axA.set_yscale("log"); axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.set_aspect("equal")
axA.set_xlabel("simulated crossing [tokens]", fontsize=7.5, labelpad=1); axA.set_ylabel("predicted crossing [tokens]", fontsize=7.5, labelpad=1)
axA.tick_params(labelsize=6.5, length=2, pad=1.5)
h1 = [Line2D([], [], color=COL[a], lw=0, marker="o", ms=3, mfc="white", mew=0.9, label=f"$\\alpha={a:g}$") for a in sorted({v["alpha"] for v in rows})]
h2 = [Line2D([], [], color="0.3", lw=0, marker=MK[L], ms=3, mfc="white", mew=0.9, label=f"$L_{{\\mathrm{{s}}}}={L}$") for L in sorted({v["Ls"] for v in rows})]
leg1 = axA.legend(handles=h1, fontsize=6, frameon=False, loc="lower right", handlelength=1.0, labelspacing=0.25, borderaxespad=0.3)
axA.add_artist(leg1)
axA.legend(handles=h2, fontsize=6, frameon=False, loc="upper left", handlelength=1.0, labelspacing=0.25, borderaxespad=0.3)

# --- right: the gap per token along the sequence, one family
for ax, (tag, title) in zip(axs, FAMILY):
    v = S[tag]; ns = np.array([r["n"] for r in v["rows"]], float)
    gm = np.array([r["gap_meas"] for r in v["rows"]]); se = np.array([r["gap_se"] for r in v["rows"]]); gp = np.array([r["gap_pred"] for r in v["rows"]])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.errorbar(ns, gm / ns, yerr=2 * se / ns, fmt="o", ms=2.5, lw=0.8, color="C0", label="simulation")
    ax.plot(ns, gp / ns, "-", lw=1.0, color="C3", label="prediction")
    ax.set_title(title, fontsize=7.5, pad=3); ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax.set_xlabel("$n$", fontsize=7.5, labelpad=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3)); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"))
axs[0].set_ylabel("gap per token [bits]", fontsize=7, labelpad=1)
hh = [Line2D([], [], color="C0", marker="o", ms=2.5, lw=0.8, label="simulation"), Line2D([], [], color="C3", lw=1.0, label="prediction")]
fig.legend(handles=hh, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.86, 1.06), ncol=2, handlelength=1.5, columnspacing=1.0)
fig.text(0.56, 1.0, "$d=256$, $L_{\\mathrm{s}}=2$, $256$ paths", ha="center", va="top", fontsize=7.5)
plt.savefig("accuracy_figure.pdf", bbox_inches="tight"); plt.savefig("accuracy_figure.png", dpi=130, bbox_inches="tight")
print("done", len(rows), "settings")
