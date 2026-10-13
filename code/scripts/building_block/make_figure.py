"""Figure: measured vs predicted gap per token for the chain settings that have been run."""
import json, os, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare import report

settings = [("S1", 128, 0.75, 2), ("S2", 128, 1.0, 2), ("S3", 128, 0.75, 3), ("S4", 128, 1.0, 3),
            ("S5", 256, 0.75, 2), ("S6", 256, 1.0, 2), ("S7", 256, 0.75, 3), ("S8", 256, 1.0, 3)]
excl = os.environ.get("EXCLUDE", "").split(",")
avail = [s for s in settings if s[0] not in excl and os.path.exists(f"sim_{s[0]}.json") and os.path.exists(f"tab_{s[0]}.json")]
ncol = 4 if len(avail) > 6 else 3 if len(avail) > 4 else 2
nrow = int(np.ceil(len(avail) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow), squeeze=False)
for ax in axes.ravel():
    ax.set_visible(False)
for ax, (S, d, a, Ls) in zip(axes.ravel(), avail):
    ax.set_visible(True)
    with contextlib.redirect_stdout(io.StringIO()):
        r = report(f"sim_{S}.json", f"tab_{S}.json", S)
    ns = np.array(r["ns"]); gm = np.array(r["gap_meas"]); gp = np.array(r["gap_pred"])
    s = json.load(open(f"sim_{S}.json"))
    lF = np.array(s["lF"]); lC = np.array(s["lC"]); cvF = np.array(s["cvF"]); cvC = np.array(s["cvC"])
    mG = (lF - cvF) - (lC - cvC)
    se = mG.std(axis=0, ddof=1) / np.sqrt(mG.shape[0])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.errorbar(ns, gm / ns, yerr=2 * se / ns, fmt="o", ms=4, color="C0", label="simulation")
    ax.plot(ns, gp / ns, "-", color="C3", label="building block")
    ax.set_title(f"$d={d}$, $\\alpha={a:g}$, $L_{{\\mathrm{{s}}}}={Ls}$", fontsize=10)
    ax.set_xlabel("$n$")
    ax.set_ylabel("gap per token [bits]")
    ax.tick_params(labelsize=8)
axes[0, 0].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("bb_gap_figure.pdf"); plt.savefig("bb_gap_figure.png", dpi=120)
print("figure with", len(avail), "panels")
