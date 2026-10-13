"""Audit the fixed-input routing follow-up and make one simple probability plot.

All 1,024 independent probe families and every intervention/control are
retained. The displayed successor probabilities average the two opposite
A-query prompts within each family before computing uncertainty across
families. No network inference is performed by this consumer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

import copier_circuit_figure as audit

ROOT = Path(__file__).resolve().parents[2]
N = 1024
GROUPS = ("clean_A", "counterfactual_A", "clean_D", "counterfactual_D", "clean_E", "counterfactual_E")
CONDITIONS = ("intact", "zero_candidate", "swap_successors", "restore_successors",
              "restore_contexts", "swap_contexts", "sham", "swap_L1H1", "swap_L1H3", "swap_L1H4")


def ci_statistics(values):
    from scipy.stats import t as student_t

    result = audit.stats(values)
    critical = float(student_t.ppf(.975, result["n"] - 1))
    result.update(ci95_lower=result["mean"] - critical * result["se"],
                  ci95_upper=result["mean"] + critical * result["se"])
    return result


def compute(arrays, probes):
    """Recompute all saved outcomes directly, including paired differences."""
    conditions, _ = audit.recompute_conditions(arrays, probes, CONDITIONS)
    b, c = probes["target_b"], probes["target_c"]
    rows, groups = np.arange(N)[:, None], np.arange(6)[None, :]
    r = np.arange(N)
    odds, original, other = {}, {}, {}
    for name in CONDITIONS:
        logits = arrays["logits_" + name]
        odds[name] = (logits[rows, groups, b[:, None]] - logits[rows, groups, c[:, None]]) / np.log(2)
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        original[name] = (probabilities[r, 0, b] + probabilities[r, 1, c]) / 2
        other[name] = (probabilities[r, 0, c] + probabilities[r, 1, b]) / 2
    zero = odds["zero_candidate"][:, 0] - odds["zero_candidate"][:, 1]
    zero -= odds["zero_candidate"][:, 4] - odds["zero_candidate"][:, 5]
    for name in CONDITIONS:
        query_response = odds[name][:, 0] - odds[name][:, 1] - odds[name][:, 4] + odds[name][:, 5]
        conditions[name]["recovery_from_zero"] = audit.stats(query_response - zero)
        conditions[name]["log_odds_B_over_C"] = {group: audit.stats(odds[name][:, g]) for g, group in enumerate(GROUPS)}
        conditions[name]["log_odds_change_from_intact"] = {
            group: audit.stats(odds[name][:, g] - odds["intact"][:, g]) for g, group in enumerate(GROUPS)}
    probabilities = {name: dict(original_successor=ci_statistics(original[name]),
                               other_successor=ci_statistics(other[name]),
                               paired_preference=ci_statistics(original[name] - other[name]),
                               original_change_from_intact=ci_statistics(original[name] - original["intact"]),
                               other_change_from_intact=ci_statistics(other[name] - other["intact"]))
                     for name in CONDITIONS}
    attention = {}
    original_attention = arrays["attention_sites_intact"]
    for name in CONDITIONS:
        values = arrays["attention_sites_" + name]
        attention[name] = {}
        for g, group in enumerate(GROUPS):
            attention[name][group] = {}
            for head in range(4):
                difference = values[:, g, head, 0] - values[:, g, head, 1]
                baseline = original_attention[:, g, head, 0] - original_attention[:, g, head, 1]
                attention[name][group][f"L2H{head+1}"] = dict(
                    earlier_A_successor_site=audit.stats(values[:, g, head, 0]),
                    earlier_D_successor_site=audit.stats(values[:, g, head, 1]),
                    site_attention_difference=audit.stats(difference),
                    difference_change_from_intact=audit.stats(difference - baseline))
    return conditions, probabilities, attention


def build(directory):
    directory = Path(directory)
    paths = {name: directory / name for name in ("summary.json", "protocol.json", "records.npz")}
    source = json.loads(paths["summary.json"].read_text())
    protocol = json.loads(paths["protocol.json"].read_text())
    if (source["protocol"] != protocol or protocol["schema"] != "copier-key-path-v1"
            or protocol["n_probes"] != N or protocol["seed"] != 20261003
            or protocol["controlled_length"] != 64 or protocol["candidate"] != [0, 1]
            or protocol["group_order"] != list(GROUPS)):
        raise ValueError("The complete declared 1,024-family follow-up is required")
    for path, digest in protocol["provenance"].items():
        if audit.sha(ROOT / path) != digest:
            raise ValueError(f"Changed recorded producer, selection, or checkpoint: {path}")
    selection_path = ROOT / "output/copier_circuit/discovery_2026-09-15/selection.json"
    selection = json.loads(selection_path.read_text())
    if (protocol["selection"] != selection or selection["stage"] != "discovery"
            or selection["n_probes"] != 256 or selection["discovery_seed"] != 20260930
            or not selection["selection_valid"] or selection["selected_heads"][0] != [0, 1]):
        raise ValueError("Changed independent discovery selection")
    configuration = json.loads((ROOT / "results/transformer/runs/lsa_a0_d64/config.json").read_text())
    expected = dict(family="lsa", alpha=[0., 0.], ls=[0, 8], seed=0, d=64, n_ctx=1024, no_permute=False)
    if any(configuration.get(key) != value for key, value in expected.items()):
        raise ValueError("Wrong fixed uniform-base checkpoint configuration")
    if audit.sha(paths["records.npz"]) != source["records_sha256"]:
        raise ValueError("Raw record checksum mismatch")
    with np.load(paths["records.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected_arrays = {"probe_" + key for key in audit.PROBE_FIELDS}
    expected_arrays |= {"logits_" + name for name in CONDITIONS} | {"attention_sites_" + name for name in CONDITIONS}
    expected_arrays |= {"attention_final_intact", "attention_final_swap_successors"}
    if set(arrays) != expected_arrays or set(source["conditions"]) != set(CONDITIONS):
        raise ValueError("A required control/condition/array is missing or unexpected")
    probes = audit.audit_probes(arrays, N)
    sys.path.insert(0, str(ROOT / "code/transformer"))
    from copier_probes import make_probes
    expected_probes = make_probes(N, length=64, vocab=64, seed=20261003)
    if any(not np.array_equal(value, expected_probes[key]) for key, value in probes.items()):
        raise ValueError("Raw probes differ from their declared independent seed")
    for name in CONDITIONS:
        logits, attention = arrays["logits_" + name], arrays["attention_sites_" + name]
        if logits.shape != (N, 6, 64) or logits.dtype != np.float64 or not np.isfinite(logits).all():
            raise ValueError(f"Invalid complete-cohort logits: {name}")
        if (attention.shape != (N, 6, 4, 2) or not np.isfinite(attention).all()
                or np.any(attention < 0) or np.any(attention > 1)):
            raise ValueError(f"Invalid complete-cohort site attention: {name}")
    maximum_attention_sum_error, maximum_site_error = 0.0, 0.0
    for name in ("intact", "swap_successors"):
        full = arrays["attention_final_" + name]
        if full.shape != (N, 6, 4, 64) or not np.isfinite(full).all() or np.any(full < 0):
            raise ValueError("Invalid full final-layer attention")
        maximum_attention_sum_error = max(maximum_attention_sum_error, float(np.max(np.abs(full.sum(-1) - 1))))
        a_site = np.take_along_axis(full, probes["source_b_pos"][:, None, None, None], axis=-1)[..., 0]
        d_site = np.take_along_axis(full, probes["source_c_pos"][:, None, None, None], axis=-1)[..., 0]
        maximum_site_error = max(maximum_site_error, float(np.max(np.abs(
            np.stack((a_site, d_site), axis=-1) - arrays["attention_sites_" + name]))))
    sham_error = float(np.max(np.abs(arrays["logits_sham"] - arrays["logits_intact"])))
    if maximum_attention_sum_error > 1e-10 or maximum_site_error > 1e-12 or sham_error > 1e-10:
        raise ValueError("A required attention indexing/normalization or sham check failed")
    for key, value in (("sham_max_logit_error", sham_error), ("full_attention_sum_error", maximum_attention_sum_error)):
        if not np.isfinite(source["checks"][key]) or abs(source["checks"][key] - value) > 1e-12:
            raise ValueError(f"Stored check disagrees with raw records: {key}")
    conditions, probabilities, attention = compute(arrays, probes)
    errors = dict(conditions=audit.compare(conditions, source["conditions"], "conditions"),
                  attention=audit.compare(attention, source["final_layer_attention"], "final-layer attention"),
                  maximum_attention_sum_error=maximum_attention_sum_error, maximum_site_index_error=maximum_site_error,
                  self_patch_logit_max_abs=sham_error)
    return dict(schema="copier-routing-figure-v1", passed=True, n_independent_probe_families=N,
                protocol=protocol, conditions=conditions, successor_probabilities=probabilities,
                final_layer_attention=attention, validation=errors,
                displayed_conditions=["intact", "zero_candidate", "swap_successors"],
                averaging="Within each family, average probability of B after clean A and of C after counterfactual A; exchange these labels for the other successor. Then average all 1,024 independent family values equally.",
                uncertainty="95% Student-t intervals for the mean across 1,024 independent probe families (1023 degrees of freedom). Each family's paired prompts stay together. Paired intervention changes are also reported.",
                intervention="Visible input is unchanged within each intervention. Swap the L1H2 pre-projection output vectors at the two demonstrated successor positions; preserve all other head outputs.",
                restoration_note="Restoring only the two successor-site outputs while zeroing this head elsewhere overshoots the intact response. Its full result is retained, without interpreting it as a recovered fraction.",
                scope="Causal evidence for a copying-related computation on these controlled prompts in one fixed transformer. The intervention can change both downstream keys and values; it does not isolate a key-only circuit or establish natural-source prevalence or sequence-time emergence.",
                inputs={name: dict(path=audit.relative(path), sha256=audit.sha(path)) for name, path in paths.items()},
                consumers={audit.relative(path): audit.sha(path) for path in (Path(__file__), Path(audit.__file__))})


def plot(report, pdf, png):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/copier_circuit/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "axes.linewidth": .6,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.5, 2.75))
    colors = ("#2468A0", "#BF611C")
    names = report["displayed_conditions"]
    labels = ("Intact", "Head removed", "Outputs exchanged")
    width = .30
    for offset, (quantity, color, label) in enumerate(zip(
            ("original_successor", "other_successor"), colors,
            ("Demonstrated successor", "Alternative successor"))):
        values = [report["successor_probabilities"][name][quantity] for name in names]
        means = np.array([value["mean"] for value in values]) * 100
        errors = np.array([value["ci95_upper"] - value["mean"] for value in values]) * 100
        x = np.arange(3) + (offset - .5) * width
        ax.bar(x, means, width, color=color, label=label, zorder=2)
        ax.errorbar(x, means, yerr=errors, fmt="none", color="#24333C",
                    capsize=2.5, linewidth=.75, zorder=3)
        for position, mean, error in zip(x, means, errors):
            ax.text(position, mean + error + .10, f"{mean:.2f}%", ha="center",
                    va="bottom", fontsize=7.5, color=color)
    ax.set_xticks(range(3), labels)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_ylim(0, 7.2)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel("Next-token probability (%)")
    ax.axhline(100 / 64, color="#777777", linewidth=.8, linestyle=":", zorder=1)
    ax.annotate("Uniform $1/64$", (.99, 100 / 64), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                fontsize=7, color="#666666")
    ax.grid(axis="y", color="#D8DEE5", linewidth=.45, zorder=0)
    ax.tick_params(axis="y", length=2.5, pad=2)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.53, .98),
               frameon=False, ncol=2, columnspacing=1.2, handletextpad=.5)
    fig.subplots_adjust(left=.10, right=.985, top=.82, bottom=.19)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=200)
    plt.close(fig)


def save_table(report, path):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["category", "condition", "query", "head", "quantity", "mean", "standard_error", "n_families", "ci95_lower", "ci95_upper"])
        for name, values in report["successor_probabilities"].items():
            for quantity, value in values.items():
                writer.writerow(["successor_probability", name, "paired_A", "", quantity, value["mean"], value["se"], value["n"], value["ci95_lower"], value["ci95_upper"]])
        for name, values in report["conditions"].items():
            for quantity, value in values.items():
                if "mean" in value:
                    writer.writerow(["condition", name, "", "", quantity, value["mean"], value["se"], value["n"], "NA", "NA"])
                else:
                    for query, statistic in value.items():
                        writer.writerow(["condition", name, query, "", quantity, statistic["mean"], statistic["se"], statistic["n"], "NA", "NA"])
        for name, groups in report["final_layer_attention"].items():
            for query, heads in groups.items():
                for head, values in heads.items():
                    for quantity, value in values.items():
                        writer.writerow(["attention", name, query, head, quantity, value["mean"], value["se"], value["n"], "NA", "NA"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/copier_circuit/key_path_2026-09-15")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/copier_circuit")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    report = build(args.input)
    if not args.check_only:
        args.out.mkdir(parents=True, exist_ok=True)
        pdf, png, table = args.out / "routing.pdf", args.out / "routing.png", args.out / "routing.tsv"
        plot(report, pdf, png)
        save_table(report, table)
        report["artifacts"] = {name: dict(path=audit.relative(path), sha256=audit.sha(path)) for name, path in (("pdf", pdf), ("png", png), ("table", table))}
        (args.out / "routing_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=True, n_probe_families=N, validation=report["validation"],
                          probabilities={name: report["successor_probabilities"][name] for name in report["displayed_conditions"]},
                          output=audit.relative(args.out))))


if __name__ == "__main__":
    main()
