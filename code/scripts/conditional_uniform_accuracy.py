"""Regroup existing validated uniform-base predictions by source depth.

No new source draws, model inference or Bayesian calculation. The selected
depth is an evaluation label only: Bayes retains its full training prior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from context_accuracy_audit import (
    Audit, ROOT, TIMES, checked_protocol, evaluate, panel, record, require,
    sha_file,
)

SOURCE = ROOT / "output/copier_time/run_2026-09-15"
AUDITED = ROOT / "results/transformer/context_accuracy_audit"
DEFAULT_OUT = ROOT / "results/transformer/conditional_accuracy/uniform_base"
SELECTED = (0, 1, 4, 8)
COUNTS = (112, 101, 127, 111, 115, 96, 118, 125, 119)


def collect():
    audit = Audit()
    receipt = audit.json(AUDITED / "receipt.json")
    require(receipt["passed"], "Existing accuracy audit failed")
    require(sha_file(AUDITED / "summary.json") == receipt["outputs"]["summary.json"],
            "Audited summary changed")
    old = next(p for p in audit.json(AUDITED / "summary.json")["panels"]
               if p["case"] == "uniform_base_64")
    protocol = checked_protocol(SOURCE, audit, ("source.py",))
    require(protocol["source_depths"] == list(range(9)) and protocol["alpha"] == [0., 0.],
            "Unexpected source prior")
    numerical = audit.json(ROOT / "results/transformer/copier_time/numerical_validation.json")
    require(numerical["passed"], "Existing Bayesian numerical audit failed")
    historical = audit.json(ROOT / "results/transformer/copier_time/summary.json")
    require(historical["passed"] and historical["n_sequences"] == 1024,
            "Historical depth summaries are incomplete")
    keys = ("tokens", "prefix_lengths", "source_depth", "transition_probs",
            "intact_probs", "bayes_probs", "metric_oracle_regret_intact",
            "metric_oracle_regret_bayes")
    digest, rows, depths = hashlib.sha256(), [], []
    for i in range(1024):
        data, meta = record(SOURCE / f"seq_{i:06d}.npz", protocol, keys, digest)
        require(meta["network_checks"]["passed"], "Saved model checks failed")
        ix = TIMES - 1
        true = data["transition_probs"][data["tokens"][ix]]
        values = evaluate(true, data["intact_probs"][ix], data["bayes_probs"][ix], audit)
        for name, value in zip(("intact", "bayes"), values[:2]):
            audit.error("saved_regret_identity_bits", np.max(np.abs(
                value - data["metric_oracle_regret_" + name][ix])), 3e-13)
        rows.append(values)
        depths.append(int(data["source_depth"]))
        if (i + 1) % 256 == 0:
            print(f"Checked and grouped {i+1}/1024 existing sequences", flush=True)
    require(digest.hexdigest() == receipt["source_cohorts"]["uniform_base_64"]["consumed_arrays_sha256"],
            "Source cohort differs from the existing full-prior figure")
    values, depths = np.asarray(rows).transpose(1, 0, 2), np.asarray(depths)
    require(tuple(np.bincount(depths, minlength=9)) == COUNTS, "Changed depth groups")
    panels = []
    for depth in range(9):
        mask = depths == depth
        current = panel(f"depth_{depth}", f"$L={depth}$", 1025, TIMES,
                        *(v[mask] for v in values), details={"source_depth": depth})
        for predictor, name in (("transformer", "intact"), ("informed_bayes", "bayes")):
            for stat in ("mean", "se"):
                expected = historical["groups"][f"depth_{depth}"]["final_prefix"][
                    "oracle_regret_" + name][stat]
                audit.error("historical_depth_endpoint_identity_bits", abs(
                    current[predictor][stat][-1] - expected), 3e-13)
        panels.append(current)
    for k, name in enumerate(("transformer", "informed_bayes", "transformer_minus_bayes")):
        for stat, result in (("mean", values[k].mean(0)),
                             ("se", values[k].std(0, ddof=1) / np.sqrt(1024))):
            audit.error("full_prior_figure_identity_bits", np.max(np.abs(
                result - old[name][stat])), 3e-13)
        combined = sum(p["n_sequences"] * np.asarray(p[name]["mean"]) for p in panels) / 1024
        audit.error("weighted_groups_recover_full_prior_bits", np.max(np.abs(
            combined - old[name]["mean"])), 3e-13)
    return {
        "schema": "conditional-uniform-accuracy-v1", "passed": True,
        "units": "bits per next-token prediction", "selected_depths": list(SELECTED),
        "time": "At time t observe tokens 1 through t and predict token t+1.",
        "quantity": "KL of the true current transition row to the predicted distribution, averaged across independent source/sequence draws at each fixed depth.",
        "bayes_prior": "One depth uniform on 0..8, shared across rows; uniform base. The selected source depth is withheld.",
        "averaging": "Each depth group averages independent transition matrices, label permutations and generated sequences; depth is held fixed.",
        "uncertainty": "Pointwise one standard error across independent whole sequences; paired errors for transformer minus Bayes.",
        "new_source_draws": False, "checkpoint_inference": False, "reference_computation": False,
        "inputs": audit.files, "raw_arrays_sha256": digest.hexdigest(),
        "checks": audit.maxima, "panels": panels,
    }


def plot(report, depths, path):
    colors = ["#1764AB", "#B85914", "#25836A", "#8650A1"] if len(depths) == 4 else plt.cm.viridis(np.linspace(.05, .9, 9))
    ncol = 2 if len(depths) == 4 else 3
    nrow = len(depths) // ncol
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 4.7 if nrow == 2 else 7.0))
        fig.subplots_adjust(left=.105, right=.975, bottom=.11, top=.86, wspace=.29, hspace=.48)
        for ax, depth, color in zip(axes.flat, depths, colors):
            p = report["panels"][depth]
            ax.set_xscale("log", base=2)
            ax.set_xlim(.8, 1280)
            ticks = [1, 4, 16, 64, 256, 1024]
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.tick_params(labelsize=7)
            ax.minorticks_off()
            ax.set_title(f"$L={depth}$  ({p['n_sequences']} sequences)", loc="left", fontsize=9)
            upper = 0.
            for name, style, marker in (("transformer", "-", "D"), ("informed_bayes", "--", "o")):
                mean, se = (np.asarray(p[name][key]) for key in ("mean", "se"))
                upper = max(upper, float(np.max(mean + se)))
                ax.errorbar(TIMES, mean, yerr=se, color=color, linestyle=style,
                            marker=marker, markersize=3 if marker == "D" else 5,
                            markerfacecolor="none", linewidth=1.4, elinewidth=.7, capsize=1.8)
            ax.set_ylim(0, upper * 1.14)
            ax.grid(axis="y", color="#D8DEE5", linewidth=.5)
            ax.set_xlabel("Observed tokens, $t$", fontsize=8)
        fig.supylabel("Extra loss (bits per prediction)", x=.015, fontsize=9)
        handles = [Line2D([], [], color="#333333", linestyle=ls, marker=marker,
                          markerfacecolor="none", markersize=4, linewidth=1.4)
                   for ls, marker in (("-", "D"), ("--", "o"))]
        fig.legend(handles, ["Transformer", "Bayes with the training prior"],
                   loc="upper center", bbox_to_anchor=(.54, .995), ncol=2,
                   frameon=False, fontsize=9, handlelength=3.2)
        fig.text(.54, .92, "Uniform-base LSA, $d=64$ — source depth fixed in each panel",
                 ha="center", fontsize=9)
        fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(path.with_suffix(".png"), dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = collect()
    report["script_sha256"] = sha_file(__file__)
    if args.check_only:
        saved = json.loads((args.out / "summary.json").read_text())
        require(saved == report, "Saved conditional summary changed")
        receipt = json.loads((args.out / "receipt.json").read_text())
        require(receipt["passed"] and receipt["script_sha256"] == sha_file(__file__), "Changed script receipt")
        require(all(sha_file(args.out / name) == digest for name, digest in receipt["outputs"].items()),
                "Figure or table changed")
        print("All source groups, pooled identities, historical endpoints and artifact hashes pass.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.out / "summary.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["depth", "n_sequences", "observed_tokens", "predictor", "mean_bits", "sequence_se_bits"])
        for p in report["panels"]:
            for name in ("transformer", "informed_bayes", "transformer_minus_bayes"):
                for i, t in enumerate(TIMES):
                    writer.writerow([p["details"]["source_depth"], p["n_sequences"], t, name,
                                     p[name]["mean"][i], p[name]["se"][i]])
    plot(report, SELECTED, args.out / "selected_depths")
    plot(report, list(range(9)), args.out / "all_depths")
    outputs = ["summary.json", "summary.tsv", "selected_depths.pdf", "selected_depths.png", "all_depths.pdf", "all_depths.png"]
    (args.out / "receipt.json").write_text(json.dumps({
        "passed": True, "script_sha256": sha_file(__file__),
        "outputs": {name: sha_file(args.out / name) for name in outputs},
    }, indent=2) + "\n")
    print(f"Wrote depth-specific loss curves to {args.out}")
    for depth in SELECTED:
        p = report["panels"][depth]
        print(f"L={depth}, N={p['n_sequences']}: t=1024 TF={p['transformer']['mean'][-1]:.6f}, Bayes={p['informed_bayes']['mean'][-1]:.6f}, gap={p['transformer_minus_bayes']['mean'][-1]:.6f} +/- {p['transformer_minus_bayes']['se'][-1]:.6f}")


if __name__ == "__main__":
    main()
