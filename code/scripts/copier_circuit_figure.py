"""Audit saved copier interventions and plot the independent confirmation.

This consumer performs no network evaluation and changes no source data.
All eight individual head removals and every saved control are retained.
Means and standard errors are recomputed from complete independent probe
pairs; the six related conditions within a pair are never counted as six
independent observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADS = tuple(f"L{layer}H{head}" for layer in (1, 2) for head in range(1, 5))
PROBE_FIELDS = ("clean", "counterfactual", "neutral_clean", "neutral_counterfactual",
                "target_b", "target_c", "source_b_pos", "source_c_pos", "query_pos",
                "context_a", "context_d", "control_e")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite independent probe-pair values")
    return dict(mean=float(values.mean()), se=float(values.std(ddof=1) / np.sqrt(len(values))), n=len(values))


def compare(actual, expected, name):
    """Compare recursively with saved summaries without accepting extra fields."""
    if set(actual) != set(expected):
        raise ValueError(f"Different saved/computed fields: {name}")
    maximum = 0.0
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, dict):
            maximum = max(maximum, compare(value, other, name + "/" + key))
        elif key == "n":
            if value != other:
                raise ValueError(f"Wrong number of probe pairs: {name}")
        else:
            if not np.isfinite(value) or not np.isfinite(other):
                raise ValueError(f"Nonfinite statistic: {name}/{key}")
            maximum = max(maximum, abs(value - other))
    if maximum > 2e-12:
        raise ValueError(f"Raw observations disagree with saved statistics: {name}, {maximum}")
    return maximum


def audit_probes(arrays, n):
    probes = {key: arrays["probe_" + key] for key in PROBE_FIELDS}
    for key, value in probes.items():
        expected = (n, 64) if key in PROBE_FIELDS[:4] else (n,)
        if value.shape != expected or value.dtype.kind not in "iu":
            raise ValueError(f"Wrong complete-cohort probe shape/type: {key}")
    clean, changed, neutral, neutral_changed = (probes[key] for key in PROBE_FIELDS[:4])
    if np.any(clean < 0) or np.any(clean >= 64):
        raise ValueError("Invalid probe tokens")
    rows = np.arange(n)
    a, d, b, c, e = (probes[key] for key in ("context_a", "context_d", "target_b", "target_c", "control_e"))
    labels = np.stack((a, d, b, c, e), axis=1)
    if np.any(np.diff(np.sort(labels, axis=1), axis=1) == 0):
        raise ValueError("Probe role labels are not distinct")
    bpos, cpos = probes["source_b_pos"], probes["source_c_pos"]
    if (np.any(bpos < 2) or np.any(cpos < 2) or np.any(bpos > 61) or np.any(cpos > 61)
            or np.any(np.abs(bpos.astype(int) - cpos.astype(int)) < 3)
            or not np.array_equal(probes["query_pos"], np.full(n, 63))):
        raise ValueError("Invalid or overlapping motif positions")
    for position, label in ((bpos-1, a), (bpos, b), (cpos-1, d), (cpos, c)):
        if not np.array_equal(clean[rows, position], label):
            raise ValueError("Raw motif differs from its declared position and label")
    if not np.array_equal(clean[:, -1], a) or np.any(clean == e[:, None]):
        raise ValueError("Wrong final query or non-novel control token")
    if np.any(np.diff(np.sort(clean[:, :-1], axis=1), axis=1) == 0):
        raise ValueError("A token repeats before the final query")
    expected_changed = clean.copy()
    expected_changed[rows, bpos], expected_changed[rows, cpos] = c, b
    if not np.array_equal(changed, expected_changed):
        raise ValueError("Counterfactual changes more than the two successor tokens")
    if not np.array_equal(np.sort(clean, axis=1), np.sort(changed, axis=1)):
        raise ValueError("Successor swap does not preserve token counts")
    for observed, base in ((neutral, clean), (neutral_changed, changed)):
        expected = base.copy()
        expected[:, -1] = e
        if not np.array_equal(observed, expected):
            raise ValueError("Unrelated-query control changes more than the final token")
    return probes


def recompute_conditions(arrays, probes, names):
    rows = np.arange(len(probes["target_b"]))[:, None]
    groups = np.arange(6)[None, :]
    b, c = probes["target_b"], probes["target_c"]
    raw, result = {}, {}
    for name in names:
        logits = arrays["logits_" + name]
        if logits.shape != (len(b), 6, 64) or not np.isfinite(logits).all():
            raise ValueError(f"Wrong complete-cohort logits: {name}")
        log_odds = (logits[rows, groups, b[:, None]] - logits[rows, groups, c[:, None]]) / np.log(2)
        differences = log_odds[:, [0, 2, 4]] - log_odds[:, [1, 3, 5]]
        raw[name] = dict(A=differences[:, 0], D=differences[:, 1], E=differences[:, 2],
                         query_specific=differences[:, 0] - differences[:, 2])
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        r = np.arange(len(b))
        result[name] = {key: stats(value) for key, value in raw[name].items()}
        result[name]["probabilities"] = dict(
            demonstrated_successor=stats((probabilities[r, 0, b] + probabilities[r, 1, c]) / 2),
            other_successor=stats((probabilities[r, 0, c] + probabilities[r, 1, b]) / 2))
    for name in names:
        result[name]["reduction_from_intact"] = stats(raw["intact"]["query_specific"] - raw[name]["query_specific"])
    return result, raw


def attention_stats(values, positions, n):
    if values.shape != (n, 6, 2, 4, 64) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Wrong complete-cohort attention cache")
    if np.max(np.abs(values.sum(axis=-1) - 1)) > 1e-12:
        raise ValueError("Unnormalized saved attention")
    rows = np.arange(n)
    return {f"L{layer+1}H{head+1}": dict(
        successor=stats(values[rows, 0, layer, head, positions]),
        context=stats(values[rows, 0, layer, head, positions - 1]))
        for layer in range(2) for head in range(4)}


def read_stage(directory, stage, n, seed, selection=None):
    directory = Path(directory)
    paths = {name: directory / name for name in ("summary.json", "protocol.json", "records.npz")}
    summary = json.loads(paths["summary.json"].read_text())
    protocol = json.loads(paths["protocol.json"].read_text())
    if (summary["protocol"] != protocol or protocol["stage"] != stage or protocol["n_probes"] != n
            or protocol["controlled_length"] != 64 or protocol["seed"] != seed
            or protocol["selection"] != selection):
        raise ValueError("Wrong cohort, protocol, selection, or random seed")
    if protocol["precision"] != "float64 forward with unchanged float32-trained parameter values":
        raise ValueError("Unrecognized evaluation precision")
    for path, digest in protocol["provenance"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Current producer/checkpoint differs from recorded provenance: {path}")
    fidelity = protocol["fidelity"]
    if (fidelity["double_manual_logit_max_abs"] > 1e-10 or fidelity["double_causal_logit_max_abs"] > 1e-10
            or fidelity["original_float32_probability_max_abs"] > 1e-5
            or not all(np.isfinite(fidelity[key]) for key in (
                "double_manual_logit_max_abs", "double_causal_logit_max_abs", "original_float32_probability_max_abs"))):
        raise ValueError("Recorded forward-fidelity check failed")
    if sha(paths["records.npz"]) != summary["records_sha256"]:
        raise ValueError("Raw record checksum differs from saved summary")
    with np.load(paths["records.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    probes = audit_probes(arrays, n)
    # Reproduce the declared seeded source independently of its saved arrays.
    import sys
    sys.path.insert(0, str(ROOT / "code/transformer"))
    from copier_probes import make_probes
    expected_probes = make_probes(n, length=64, vocab=64, seed=seed)
    if any(not np.array_equal(value, expected_probes[key]) for key, value in probes.items()):
        raise ValueError("Raw probes do not match their declared random seed")
    names = ["intact"] + ["ablate_" + head for head in HEADS]
    if selection:
        selected = [f"L{layer+1}H{head+1}" for layer, head in selection["selected_heads"]]
        controls = [f"L{layer+1}H{head+1}" for layer, head in selection["control_heads"]]
        names += ["ablate_selected", "ablate_controls"]
        names += ["patch_" + head for head in selected + controls]
        names += ["patch_selected", "patch_sham", "patch_irrelevant_sites"]
    expected_arrays = {"probe_" + key for key in PROBE_FIELDS} | {"logits_" + name for name in names} | {"attention_final"}
    if stage == "discovery":
        expected_arrays |= {"natural_" + key for key in ("tokens", "depths", "previous", "induction", "chance")}
    else:
        expected_arrays.add("attention_after_upstream_ablation")
    if set(arrays) != expected_arrays or set(summary["conditions"]) != set(names):
        raise ValueError("Missing or unexpected condition, control, or raw array")
    conditions, raw = recompute_conditions(arrays, probes, names)
    error = compare(conditions, summary["conditions"], stage + "/conditions")
    attention = attention_stats(arrays["attention_final"], probes["source_b_pos"], n)
    error = max(error, compare(attention, summary["head_attention"], stage + "/attention"))
    extra = {}
    if stage == "discovery":
        if protocol["natural_discovery_n"] != 256 or arrays["natural_tokens"].shape != (256, 129):
            raise ValueError("Wrong natural-discovery cohort")
        if arrays["natural_depths"].shape != (256,) or np.any(arrays["natural_depths"] < 0) or np.any(arrays["natural_depths"] > 8):
            raise ValueError("Wrong natural-discovery source depths")
        previous, induction, chance = (arrays["natural_" + key] for key in ("previous", "induction", "chance"))
        if previous.shape != (256, 2, 4) or induction.shape != previous.shape or chance.shape != (256,):
            raise ValueError("Wrong natural-discovery attention observations")
        natural = {f"L{layer+1}H{head+1}": dict(
            previous_token_attention=stats(previous[:, layer, head]),
            matching_successor_attention=stats(induction[:, layer, head]),
            matching_successor_uniform_attention=stats(chance)) for layer in range(2) for head in range(4)}
        error = max(error, compare(natural, summary["natural_discovery"], "natural discovery"))
        extra["natural_discovery"] = natural
    else:
        upstream = attention_stats(arrays["attention_after_upstream_ablation"], probes["source_b_pos"], n)
        error = max(error, compare(upstream, summary["head_attention_after_upstream_ablation"], "upstream intervention"))
        sham_error = float(np.max(np.abs(arrays["logits_patch_sham"] - arrays["logits_intact"])))
        if sham_error > 1e-10:
            raise ValueError("Self-patching changed the logits")
        extra.update(head_attention_after_upstream_ablation=upstream, self_patch_logit_max_abs=sham_error)
        # Paired specificity contrast; individual-condition SEs cannot be added.
        selected_second = next(key for key in selected if key.startswith("L2"))
        control_second = next(key for key in controls if key.startswith("L2"))
        extra["selected_vs_control_second_layer_patch"] = stats(
            raw["patch_" + selected_second]["query_specific"] - raw["patch_" + control_second]["query_specific"])
    return dict(protocol=protocol, conditions=conditions, head_attention=attention,
                maximum_summary_recomputation_error=error, **extra,
                inputs={name: dict(path=relative(path), sha256=sha(path)) for name, path in paths.items()}), summary


def build(discovery_directory, confirmation_directory):
    discovery, source_summary = read_stage(discovery_directory, "discovery", 256, 20260930)
    selection_path = Path(discovery_directory) / "selection.json"
    selection = json.loads(selection_path.read_text())
    if (selection != source_summary["selection"] or selection["n_probes"] != 256
            or selection["discovery_seed"] != 20260930 or not selection["selection_valid"]
            or selection["unresolved_layers"] or selection["provenance"] != discovery["protocol"]["provenance"]):
        raise ValueError("Invalid or changed discovery selection")
    for layer in range(2):
        candidates = list(range(4))
        scores = [discovery["conditions"][f"ablate_L{layer+1}H{head+1}"]["reduction_from_intact"]["mean"] for head in candidates]
        if (scores[np.argmax(scores)] <= 0 or selection["selected_heads"][layer] != [layer, int(np.argmax(scores))]
                or selection["control_heads"][layer] != [layer, int(np.argmin(scores))]):
            raise ValueError("Candidates or controls differ from the discovery-only selection rule")
    confirmation, _ = read_stage(confirmation_directory, "confirm", 1024, 20261001, selection)
    if discovery["protocol"]["provenance"] != confirmation["protocol"]["provenance"]:
        raise ValueError("Discovery and confirmation use different checkpoints or producers")
    return dict(schema="copier-circuit-identification-consumer-v1", passed=True,
                discovery=discovery, confirmation=confirmation, selection=selection,
                selection_file=dict(path=relative(selection_path), sha256=sha(selection_path)),
                metric="Change in log2[P(B)/P(C)] between the paired successor-swapped prompts, minus the same change for unseen query E.",
                uncertainty="One standard error across independent matched probe pairs. All six outputs within a pair remain together.",
                scope="Controlled circuit identification in one fixed uniform-base transformer. No sequence-time transition or unique second-layer copier is established.",
                controls_note="Every head ablation and every patch/control condition is retained, including response increases and effective control-head patches.",
                script=dict(path=relative(__file__), sha256=sha(__file__)))


def plot(report, pdf, png):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/copier_circuit/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    selected = {f"L{layer+1}H{head+1}" for layer, head in report["selection"]["selected_heads"]}
    control_second = next(f"L{layer+1}H{head+1}" for layer, head in report["selection"]["control_heads"] if layer == 1)
    selected_second = next(head for head in selected if head.startswith("L2"))
    conditions = report["confirmation"]["conditions"]
    blue, orange, grey = "#166B94", "#CF5B24", "#75838D"
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.7), gridspec_kw={"wspace": .77})
    panels = [(["intact"] + ["ablate_" + head for head in HEADS],
               ["Intact network"] + ["Remove " + head for head in HEADS], "Remove one head at a time"),
              (["intact", "patch_" + selected_second, "patch_" + control_second, "patch_sham", "patch_irrelevant_sites"],
               ["Intact network", "Swap " + selected_second + " output\n(selected head)",
                "Swap " + control_second + " output\n(control head)", "Reuse original outputs\n(self-patch control)",
                "Swap at unrelated positions"], "Exchange internal outputs between the pair")]
    for ax, (names, labels, title) in zip(axes, panels):
        for index, name in enumerate(names):
            value = conditions[name]["query_specific"]
            head = name.split("_", 1)[-1]
            color = blue if name == "intact" else orange if head in selected else grey
            ax.errorbar(value["mean"], index, xerr=value["se"], fmt="o", color=color,
                        markersize=6, capsize=3, elinewidth=1.5, zorder=3)
            displayed = 0.0 if abs(value["mean"]) < .005 else value["mean"]
            ax.text(value["mean"] + .12, index, f"{displayed:.2f}", va="center", fontsize=9, color=color)
        ax.axvline(0, color="#A8B3BA", linewidth=.8)
        ax.axvline(conditions["intact"]["query_specific"]["mean"], color=blue, linewidth=.8, linestyle=":", alpha=.6)
        ax.set_yticks(range(len(names)), labels)
        ax.set_ylim(len(names) - .5, -.7)
        ax.set_xlim(-.18, 4.95)
        ax.set_xticks(range(5))
        ax.set_xlabel("Successor-change response (bits)", labelpad=10)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.grid(axis="x", color="#E5E9EC", linewidth=.6)
        ax.set_title(title, fontsize=11, pad=15)
    fig.suptitle("Testing copying inside the trained transformer", fontsize=16, x=.52, y=.973)
    fig.text(.52, .928, "... A B ... D C ... A     versus     ... A C ... D B ... A",
             ha="center", fontsize=13, family="DejaVu Sans Mono")
    fig.text(.52, .891, "The earlier successors are exchanged; token counts stay fixed. Both motif orders are tested.",
             ha="center", fontsize=10)
    fig.legend(handles=[Line2D([0], [0], marker="o", color=blue, ls="", label="Intact network"),
                        Line2D([0], [0], marker="o", color=orange, ls="", label="Selected on separate discovery data"),
                        Line2D([0], [0], marker="o", color=grey, ls="", label="Other heads and controls")],
               loc="upper center", bbox_to_anchor=(.52, .863), ncol=3, frameon=False, fontsize=9)
    fig.text(.52, .112, "Shown: all 1,024 independent confirmation pairs. Candidates were fixed using 256 separate discovery pairs.",
             ha="center", fontsize=9)
    fig.text(.52, .077, "Response: change in log₂[P(B)/P(C)] after the successor swap, minus that change for an unseen final query E.",
             ha="center", fontsize=8.5)
    fig.text(.52, .046, "Positive values mean predictions follow A’s earlier successor. Error bars: one standard error across matched pairs.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.13, right=.97, top=.762, bottom=.20)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=190)
    plt.close(fig)


def save_table(report, path):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["stage", "category", "condition", "quantity", "mean", "standard_error", "n_independent_pairs_or_sequences"])
        for stage in ("discovery", "confirmation"):
            stage_report = report[stage]
            for condition, observations in stage_report["conditions"].items():
                for quantity, value in observations.items():
                    if quantity == "probabilities":
                        for name, statistic in value.items():
                            writer.writerow([stage, "condition", condition, "probability_" + name,
                                             statistic["mean"], statistic["se"], statistic["n"]])
                    else:
                        writer.writerow([stage, "condition", condition, quantity, value["mean"], value["se"], value["n"]])
            for category in ("head_attention", "head_attention_after_upstream_ablation", "natural_discovery"):
                for head, observations in stage_report.get(category, {}).items():
                    for quantity, value in observations.items():
                        writer.writerow([stage, category, head, quantity, value["mean"], value["se"], value["n"]])
        value = report["confirmation"]["selected_vs_control_second_layer_patch"]
        writer.writerow(["confirmation", "paired_patch_difference", "selected_minus_control", "query_specific",
                         value["mean"], value["se"], value["n"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=ROOT / "output/copier_circuit/discovery_2026-09-15")
    parser.add_argument("--confirmation", type=Path, default=ROOT / "output/copier_circuit/confirmation_2026-09-15")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/copier_circuit")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    report = build(args.discovery, args.confirmation)
    if not args.check_only:
        args.out.mkdir(parents=True, exist_ok=True)
        pdf, png, table = args.out / "identification.pdf", args.out / "identification.png", args.out / "small.tsv"
        plot(report, pdf, png)
        save_table(report, table)
        report["artifacts"] = {name: dict(path=relative(path), sha256=sha(path)) for name, path in
                               (("pdf", pdf), ("png", png), ("table", table))}
        (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=True, discovery_pairs=256, confirmation_pairs=1024,
                          maximum_recomputation_error=max(report[key]["maximum_summary_recomputation_error"] for key in ("discovery", "confirmation")),
                          selected_vs_control_second_layer_patch=report["confirmation"]["selected_vs_control_second_layer_patch"],
                          output=relative(args.out))))


if __name__ == "__main__":
    main()
