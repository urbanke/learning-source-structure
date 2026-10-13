"""Show finite crossing predictions, actual code costs, and alphabet dependence.

MPLCONFIGDIR=/tmp/la-mpl python code/scripts/crossing_count_figures.py

Reads the complete numerical experiment and its independent bootstrap audit.
Use --costs-only to inspect the single completed example during the campaign.
No fit is used in either figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/mechanism/crossing_finite_count_prediction.json"
AUDIT = ROOT / "results/mechanism/crossing_count_prediction_audit.json"
DEPTHS, ALPHAS, DS = (2, 3), (.75, 1., 1.25), (128, 256, 512)


def plot_costs(cells):
    cell = next(c for c in cells if (c["L"], c["alpha"], c["d"]) == (2, 1., 256))
    n = np.asarray(cell["tokens"])
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for key, color in [("frequency_codelength", "#2369a0"), ("context_codelength", "#bd6023")]:
        mean = np.asarray(cell["count"][key]["mean"])/n
        se = np.asarray(cell["count"][key]["SE"])/n
        axes[0].plot(n, mean, color=color, lw=1.4)
        axes[0].fill_between(n, mean-2*se, mean+2*se, color=color, alpha=.12, lw=0)
        axes[0].errorbar(n, np.asarray(cell["chain"][key]["mean"])/n,
            yerr=2*np.asarray(cell["chain"][key]["SE"])/n, fmt="o", mfc="white",
            mec=color, color=color, ms=3.2, elinewidth=.6, capsize=1.5)
    axes[0].set_ylabel("Cumulative prediction cost\n(bits per token)")
    pred = np.asarray(cell["count"]["gap"]["mean"])/n
    pse = np.asarray(cell["count"]["gap"]["SE"])/n
    actual = np.asarray(cell["chain"]["gap"]["mean"])/n
    ase = np.asarray(cell["chain"]["gap"]["SE"])/n
    axes[1].plot(n, pred, color="#704a9e", lw=1.4)
    axes[1].fill_between(n, pred-2*pse, pred+2*pse, color="#704a9e", alpha=.13, lw=0)
    axes[1].errorbar(n, actual, yerr=2*ase, fmt="o", color=".15", mfc="white",
                     ms=3.2, elinewidth=.7, capsize=1.5)
    lower = min(float(min(pred-2*pse)), float(min(actual-2*ase)))-.04
    upper = max(float(max(pred+2*pse)), float(max(actual+2*ase)))+.04
    axes[1].set_ylim(lower, upper)
    axes[1].axhspan(0, upper, color="#bd6023", alpha=.045, lw=0)
    axes[1].axhspan(lower, 0, color="#2369a0", alpha=.045, lw=0)
    axes[1].axhline(0, color=".35", lw=.65)
    axes[1].text(.04, .96, "Copying is cheaper", transform=axes[1].transAxes, va="top", fontsize=7)
    axes[1].text(.04, .035, "Frequency is cheaper", transform=axes[1].transAxes, va="bottom", fontsize=7)
    axes[1].set_ylabel("Cost difference per token\n(frequency minus copying; bits)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlim(1250, 25500)
        ax.set_xticks([2000, 4000, 8000, 16000], ["2000", "4000", "8000", "16000"])
        ax.set_xlabel("Sequence length $n$ (tokens)")
        ax.grid(alpha=.16, lw=.5)
    fig.legend(handles=[Line2D([], [], color="#2369a0", label="Frequency prediction"),
                        Line2D([], [], color="#bd6023", label="Copying"),
                        Line2D([], [], color="#704a9e", label="Predicted difference"),
                        Line2D([], [], color=".15", marker="o", mfc="white", ls="none",
                               label="Measured means")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.52, 1.015),
               columnspacing=1.5, handlelength=2)
    fig.subplots_adjust(left=.09, right=.985, bottom=.19, top=.79, wspace=.4)
    fig.savefig(ROOT / "figures/crossing_count_costs.pdf")
    plt.close(fig)
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--costs-only", action="store_true")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42})
    data = json.loads(SOURCE.read_text())
    example = plot_costs(data["cells"])
    if args.costs_only:
        print("Wrote the finite-cost example.")
        return
    assert data["complete"] and len(data["cells"]) == 18
    audit = json.loads(AUDIT.read_text())
    assert audit["complete"] and len(audit["cells"]) == 18
    assert audit["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    cells = {(c["L"], c["alpha"], c["d"]): c for c in data["cells"]}
    checked = {(c["L"], c["alpha"], c["d"]): c for c in audit["cells"]}
    assert set(cells) == set(checked) == {(L, a, d) for L in DEPTHS for a in ALPHAS for d in DS}
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.9), sharex=True)
    rows = []
    for i, L in enumerate(DEPTHS):
        for j, alpha in enumerate(ALPHAS):
            ax = axs[i, j]
            for name, color, marker, shift in [("count", "#2369a0", None, 1.),
                                                ("chain", ".1", "o", .977),
                                                ("archive", "#b35b26", "x", 1.023)]:
                y, lo, hi, show_interval = [], [], [], []
                for d in DS:
                    record = checked[L, alpha, d]
                    value = record["mean_curve_crossings"][name]
                    # The independent audit stores the original mean-curve root.
                    if isinstance(value, dict):
                        value = value["value"]
                    quantiles = record["bootstrap"][name]["conditional_on_located_quantiles"]
                    assert value is not None and quantiles is not None
                    y.append(value); lo.append(quantiles[0]); hi.append(quantiles[-1])
                    bootstrap = record["bootstrap"][name]
                    show_interval.append(name != "archive" or bootstrap["null"]/bootstrap["draws"] <= .05)
                x = np.asarray(DS)
                if name == "count":
                    ax.plot(x, y, color=color, lw=1.3)
                    ax.fill_between(x, lo, hi, color=color, alpha=.13, lw=0)
                else:
                    # Whiskers show the percentile interval itself even when a
                    # nonlinear last-root estimate lies outside that interval.
                    valid = np.asarray(show_interval)
                    ax.vlines((x*shift)[valid], np.asarray(lo)[valid], np.asarray(hi)[valid], color=color, lw=.65)
                    ax.plot(x*shift, y, marker=marker, mfc="white", mec=color,
                            color=color, ls="none", ms=3.8)
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlim(105, 620)
            ax.set_xticks(DS, [str(d) for d in DS])
            ax.yaxis.set_major_locator(FixedLocator([300, 500, 1000, 2000, 5000, 10000, 20000, 50000]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.16, lw=.5)
            if i == 0: ax.set_title(f"$\\alpha={alpha:g}$")
            if j == 0: ax.set_ylabel(f"$L_{{\\mathrm{{s}}}}={L}$\nCrossing length (tokens)")
            if i == 1: ax.set_xlabel("Alphabet size $d$")
    fig.legend(handles=[Line2D([], [], color="#2369a0", label="Finite prediction"),
                        Line2D([], [], color=".1", marker="o", mfc="white", ls="none", label="New chains: same tables"),
                        Line2D([], [], color="#b35b26", marker="x", ls="none", label="Earlier independent chains")],
               loc="upper center", bbox_to_anchor=(.54, 1.01), ncol=2, frameon=False,
               handlelength=2, columnspacing=1.4)
    fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.81, wspace=.43, hspace=.3)
    fig.savefig(ROOT / "figures/crossing_count_scaling.pdf")
    plt.close(fig)
    for key in sorted(cells):
        c, a = cells[key], checked[key]
        roots = a["mean_curve_crossings"]
        roots = {name: (value["value"] if isinstance(value, dict) else value) for name, value in roots.items()}
        rows.append(dict(L=key[0], alpha=key[1], d=key[2], **roots,
            paired_relative_error=roots["count"]/roots["chain"]-1,
            archive_relative_error=roots["count"]/roots["archive"]-1,
            paired_error_interval=a["bootstrap"]["count_over_chain_minus_one"]["conditional_on_located_quantiles"]))
    output = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(AUDIT.read_bytes()).hexdigest(), rows=rows,
        paired_median_absolute_relative_error=float(np.median([abs(r["paired_relative_error"]) for r in rows])),
        paired_maximum_absolute_relative_error=max(abs(r["paired_relative_error"]) for r in rows),
        archive_median_absolute_relative_error=float(np.median([abs(r["archive_relative_error"]) for r in rows])),
        archive_maximum_absolute_relative_error=max(abs(r["archive_relative_error"]) for r in rows),
        example={k:example[k] for k in ("L", "alpha", "d", "tokens")})
    (ROOT / "results/mechanism/crossing_count_figures.json").write_text(json.dumps(output, indent=2)+"\n")
    lines = [r"\begin{tabular}{ccrrrrr}", r"\toprule",
             r"$L_{\mathrm{s}}$ & $\alpha$ & $d$ & Prediction & New chains & Earlier chains & Paired difference (\%)\\", r"\midrule"]
    for r in rows:
        interval = r["paired_error_interval"]
        lines.append(f"{r['L']} & {r['alpha']:g} & {r['d']} & {r['count']:.0f} & {r['chain']:.0f} & {r['archive']:.0f} & "
                     f"${100*r['paired_relative_error']:+.1f}\\ [{100*interval[0]:+.1f},{100*interval[-1]:+.1f}]$"+r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "results/mechanism/crossing_count_comparison_table.tex").write_text("\n".join(lines)+"\n")
    print("Wrote both figures, all 18 crossing comparisons, and the appendix table.")


if __name__ == "__main__":
    main()
