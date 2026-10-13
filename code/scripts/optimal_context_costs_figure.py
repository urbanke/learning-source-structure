"""Plot the matched Bayes cost and its independent scalar approximation.

python code/scripts/optimal_context_costs_figure.py
Smoke: python code/scripts/optimal_context_costs_figure.py --out /tmp/optimal_context_costs.pdf --summary /tmp/optimal_context_costs.json
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
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/mechanism/optimal_context_figure_check.json"
AUDIT = ROOT / "results/mechanism/optimal_context_figure_audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/optimal_context_costs.pdf")
    parser.add_argument("--summary", type=Path, default=ROOT / "results/mechanism/optimal_context_costs_figure.json")
    args = parser.parse_args()
    data = json.loads(SOURCE.read_text())
    audit = json.loads(AUDIT.read_text())
    assert data["complete"] and audit["status"] == "passed"
    assert audit["source_sha256"] == sha(SOURCE)
    assert audit["raw_sha256"] == data["raw_sha256"]
    assert (data["d"], data["alpha"], data["L"], data["repetitions"]) == (256, 1., 2, 16)
    n = np.asarray(data["tokens"])
    costs = data["costs"]
    source_tables = ROOT / data["source"]
    assert sha(source_tables) == data["source_sha256"]
    with np.load(source_tables, allow_pickle=False) as source:
        tables = source["transition_matrices"]
        stationary = source["stationary_laws"]
    assert tables.shape == (data["repetitions"], data["d"], data["d"])
    assert np.allclose(stationary.sum(axis=1), 1, rtol=0, atol=1e-12)
    assert np.allclose(np.einsum("ra,rab->rb", stationary, tables),
                       stationary, rtol=0, atol=1e-12)
    log_tables = np.zeros_like(tables)
    np.log2(tables, out=log_tables, where=tables > 0)
    row_entropies = -(tables * log_tables).sum(axis=2)
    entropy_rates = (stationary * row_entropies).sum(axis=1)
    cost_after_learning = float(entropy_rates.mean())
    plt.rcParams.update({"font.size": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    displayed = {}
    for ax, prediction, measurement, color in [
        (axes[0], "predicted_finite_full_cost", "bayes_full_cost", "#2369a0"),
        (axes[1], "J_finite_occupation", "bayes_redundancy", "#704a9e"),
    ]:
        mean = np.asarray(costs[prediction]["mean"]) / n
        se = np.asarray(costs[prediction]["SE"]) / n
        obs = np.asarray(costs[measurement]["mean"]) / n
        obs_se = np.asarray(costs[measurement]["SE"]) / n
        ax.plot(n, mean, color=color, lw=1.4)
        ax.fill_between(n, mean - 2*se, mean + 2*se, color=color, alpha=.12, lw=0)
        point_color = ".15" if ax is axes[1] else color
        ax.errorbar(n, obs, yerr=2*obs_se, fmt="o", mfc="white",
                    mec=point_color, color=point_color, ms=3.2,
                    elinewidth=.6, capsize=1.5)
        displayed[prediction] = dict(mean_per_token=mean.tolist(), SE_per_token=se.tolist())
        displayed[measurement] = dict(mean_per_token=obs.tolist(), SE_per_token=obs_se.tolist())
    axes[0].axhline(cost_after_learning, color="#bd6023", lw=1.4, ls="--")
    axes[0].set_ylabel("Prediction cost\n(bits per token)")
    axes[1].set_ylabel("Learning cost\n(bits per token)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlim(1250, 25500)
        ax.set_xticks([2000, 4000, 8000, 16000], ["2000", "4000", "8000", "16000"])
        ax.set_xlabel("Sequence length $n$ (tokens)")
        ax.grid(alpha=.16, lw=.5)
    fig.legend(handles=[
        Line2D([], [], color="#2369a0", label="Prediction cost from approximation"),
        Line2D([], [], color="#bd6023", ls="--", label="Cost after learning"),
        Line2D([], [], color="#704a9e", label="Learning cost from approximation"),
        Line2D([], [], color=".15", marker="o", mfc="white", ls="none",
               label="Measured mean cost")],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.52, 1.015),
        columnspacing=1.5, handlelength=2)
    fig.subplots_adjust(left=.09, right=.985, bottom=.19, top=.80, wspace=.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    total_relative = np.asarray(costs["bayes_full_cost"]["mean"]) / costs["predicted_finite_full_cost"]["mean"] - 1
    learning_relative = np.asarray(costs["bayes_redundancy"]["mean"]) / costs["J_finite_occupation"]["mean"] - 1
    j = data["tokens"].index(5663)
    summary = dict(source_sha256=sha(SOURCE), audit_sha256=sha(AUDIT),
        d=data["d"], alpha=data["alpha"], L=data["L"], repetitions=data["repetitions"],
        tokens=data["tokens"], displayed=displayed,
        stationary_reference=dict(
            source=data["source"], source_sha256=data["source_sha256"],
            definition="For each table, sum_a pi_T(a) H(T(.|a)); average over the same sampled tables.",
            bits_per_token_by_table=entropy_rates.tolist(),
            mean_bits_per_token=cost_after_learning,
            SE_bits_per_token=float(entropy_rates.std(ddof=1) / np.sqrt(len(entropy_rates))),
            finite_known_cost_minus_reference=(np.asarray(costs["known_table_expected_full_cost"]["mean"]) / n - cost_after_learning).tolist()),
        learning_cost_change=dict(
            sequence_length_factor=float(n[-1] / n[0]),
            measured_bits_per_token_at_endpoints=[displayed["bayes_redundancy"]["mean_per_token"][i] for i in (0, -1)],
            predicted_bits_per_token_at_endpoints=[displayed["J_finite_occupation"]["mean_per_token"][i] for i in (0, -1)],
            measured_decrease_per_doubling=float((displayed["bayes_redundancy"]["mean_per_token"][0] - displayed["bayes_redundancy"]["mean_per_token"][-1]) / np.log2(n[-1] / n[0])),
            predicted_decrease_per_doubling=float((displayed["J_finite_occupation"]["mean_per_token"][0] - displayed["J_finite_occupation"]["mean_per_token"][-1]) / np.log2(n[-1] / n[0])),
            qualification="Endpoint secants over the plotted finite range; not asymptotic scaling laws."),
        measured_total_over_prediction_minus_one=total_relative.tolist(),
        measured_learning_over_prediction_minus_one=learning_relative.tolist(),
        maximum_absolute_total_relative_difference=float(abs(total_relative).max()),
        measured_learning_shortfall_range=(-float(learning_relative.max()), -float(learning_relative.min())),
        comparison_at_5663_tokens=dict(
            bayes_bits_per_token=costs["bayes_full_cost"]["mean"][j]/n[j],
            lsa_context_bits_per_token=costs["lsa_context_full_cost"]["mean"][j]/n[j],
            lsa_context_minus_bayes_bits=data["paired_differences"]["lsa_context_minus_bayes"]["mean"][j],
            paired_SE_bits=data["paired_differences"]["lsa_context_minus_bayes"]["SE"][j],
            qualification="The optimal code knows q and true L; the symmetric LSA code does not. The difference includes this information advantage."))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("maximum_absolute_total_relative_difference", "measured_learning_shortfall_range", "comparison_at_5663_tokens")}, indent=2))


if __name__ == "__main__":
    main()
