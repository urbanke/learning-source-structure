"""Layer eight targeted alpha continuations onto the complete copying cohort.

The original t5--8 campaigns and the prior two-record convergence recovery
remain immutable.  This merger validates both layers, substitutes exactly the
eight declared all-distinct 16/32 records, reruns every paired-order check and
rebuilds all four complete-cohort diagnostic arrays for the unchanged aggregate
assessment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

import copying_payoff_alpha_refinement as alpha
import copying_payoff_alpha_memory_recovery as alpha_memory
import copying_payoff_alpha_audit_only_salvage as alpha_audit_only
import copying_payoff_alpha_sampling_memory_recovery as alpha_sampling_memory
import copying_payoff_alpha_t6_audit_only_salvage as alpha_t6_audit_only
import copying_payoff_alpha_t7_t8_single_worker_recovery as alpha_t7_t8_recovery
import copying_payoff_alpha_t8_audit_only_salvage as alpha_t8_audit_only
import combine_copying_payoff_alpha_audit_recovery as alpha_audit_combined
import merge_copying_payoff_recovery as recovered


parent = recovered.parent
old = recovered.old
dense = recovered.dense
ROOT, require, read, sha = recovered.ROOT, recovered.require, recovered.read, recovered.sha
SCHEMA = "copying-payoff-alpha-refinement-merge-v1"
TIMES = (5, 6, 7, 8)
T5_PARENT = alpha.T5_PARENT
CAMPAIGN_PARENT = alpha.CAMPAIGN_PARENT
PRIOR_RECOVERY_OVERLAY = ROOT / "output/copying_payoff_recovery_procinfo_2026-09-21/overlay.json"
EXPECTED_IDS = {item[2] for item in alpha.TARGETS}
EXPECTED_PAIR_COUNT = 526
COMBINED_PRODUCERS = (alpha_audit_combined,)


def producer_for_overlay(path):
    path = Path(path)
    saved = read(path) if path.is_file() else {}
    for combined in COMBINED_PRODUCERS:
        if saved.get("schema") == combined.SCHEMA:
            return combined
    plan = read(path.parent / "plan.json")
    producers = {module.SCHEMA: module for module in
                 (alpha, alpha_memory, alpha_audit_only, alpha_sampling_memory,
                  alpha_t6_audit_only, alpha_t7_t8_recovery, alpha_t8_audit_only)}
    require(plan.get("schema") in producers,
            "Unknown alpha-refinement producer schema")
    return producers[plan["schema"]]


def producer_code_paths(producers):
    """Bind every allocation wrapper and the frozen sampling engine."""
    return ({Path(__file__), Path(alpha.__file__), Path(recovered.__file__)}
            | {Path(producer.__file__) for producer in producers})


def load_alpha_overlay(path):
    path = Path(path)
    producer = producer_for_overlay(path)
    validated = producer.overlay(path.parent, write_manifest=False)
    result = validated[0] if producer in COMBINED_PRODUCERS else validated
    require(path.resolve() == (path.parent / "overlay.json").resolve()
            and read(path) == result and result["schema"] == producer.OVERLAY_SCHEMA
            and result["complete"] is True
            and result["baseline_draws"] == producer.BASELINE_DRAWS
            and result["target_draws"] == producer.TARGET_DRAWS,
            "Alpha refinement overlay changed")
    require({item["task_id"] for item in result["replacements"]} == EXPECTED_IDS
            and len(result["replacements"]) == 8,
            "Alpha refinement selected or omitted tasks")
    return producer, result


def replacement_records(overlay_path):
    producer, overlay = load_alpha_overlay(overlay_path)
    if producer in COMBINED_PRODUCERS:
        _, entries = producer.collect()
        producers = {entry["producer"] for entry in entries.values()} | {producer}
    else:
        plan = producer.load_plan(Path(overlay_path).parent)
        specs = {spec["task"]["id"]: spec for spec in plan["tasks"]}
        entries = {}
        for task_id, spec in specs.items():
            checked = producer.validate_replacement(plan, spec, require_pass=True)
            item = next(item for item in overlay["replacements"] if item["task_id"] == task_id)
            entries[task_id] = dict(producer=producer, plan=plan, spec=spec,
                                    checked=checked, item=item)
        producers = {producer}
    records, bindings = {}, {
        str(Path(overlay_path).resolve()): sha(overlay_path),
    }
    for component in overlay.get("components", []):
        for name in ("plan", "allocation"):
            bindings[component[name]["path"]] = component[name]["sha256"]
    unused_plans = list(overlay.get("unused_plans", []))
    if overlay.get("unused_prepared_tail") is not None:
        unused_plans.append(overlay["unused_prepared_tail"])
    for unused in unused_plans:
        bindings[unused["plan"]["path"]] = unused["plan"]["sha256"]
    if producer not in COMBINED_PRODUCERS:
        plan_path = Path(overlay_path).parent / "plan.json"
        bindings[str(plan_path.resolve())] = sha(plan_path)
    for item in overlay["replacements"]:
        task_id = item["task_id"]
        require(task_id in entries and item["task"] == entries[task_id]["spec"]["task"],
                "Alpha overlay task differs from its plan")
        entry = entries[task_id]
        task_producer, checked = entry["producer"], entry["checked"]
        require(item["sample_prefix"] == checked["prefix"]
                and item["rng_continuity"] == checked["rng"]
                and item["audit_passed"] is True,
                "Alpha overlay continuation proof changed")
        p = checked["paths"]
        with np.load(p["moments"], allow_pickle=False) as archive:
            moments = {key: archive[key].copy() for key in archive}
        recovered.moments_checked(moments, checked["report"]["estimate"],
                                  len(set(item["task"]["tokens"])), task_producer.TARGET_DRAWS)
        records[task_id] = dict(item=item, checked=checked, moments=moments)
        for name in (*task_producer.PARENT_FILES, "receipt", "prepared"):
            bindings[str(p[name].resolve())] = sha(p[name])
    require(set(records) == EXPECTED_IDS, "Alpha replacement records are incomplete")
    return producers, overlay, records, bindings


def base_waves():
    plan5, result5, refs5, bindings5 = old.load_references(T5_PARENT)
    require(result5["complete"] is True and result5["individual_audits_passed"] is True
            and result5["paired_orders_passed"] is True
            and result5["diagnostic_arrays_validated"] is False,
            "Frozen t5 reference family no longer passes its raw gates")
    _, later, bindings_later, paired = recovered.load_campaign_references(
        CAMPAIGN_PARENT, PRIOR_RECOVERY_OVERLAY)
    require(paired is True and set(later) == {6, 7, 8},
            "Prior complete t6--8 recovery family no longer passes")
    waves = {5: dict(plan=plan5, refs=refs5, result=result5)}
    waves.update(later)
    bindings5.update(bindings_later)
    return waves, bindings5


def paired_check(time, left, right):
    if time == 5:
        return dense.accuracy.paired_order_check(
            left["mean"], left["mcse_probs"], right["mean"], right["mcse_probs"])
    return parent.paired_order_check(left, right)


def apply_replacements(waves, records):
    by_time = {time: [] for time in TIMES}
    for task_id, record in records.items():
        task = record["item"]["task"]
        time, order, key = task["time"], task["order"], task["reference_key"]
        require(time in waves and key in waves[time]["refs"][order],
                "Alpha replacement is outside the complete base cohort")
        waves[time]["refs"][order][key] = record["moments"]
        by_time[time].append(task_id)
    require(all(len(by_time[time]) == 2 for time in TIMES),
            "Each time must replace exactly its 16/32 all-distinct pair")
    all_pairs = []
    for time in TIMES:
        refs = waves[time]["refs"]
        require(set(refs[16]) == set(refs[32]) and len(refs[16]) == dense.COUNTS[time],
                "Refined wave lost canonical-reference coverage")
        pairs = [dict(time=time, reference_key=key,
                      **paired_check(time, refs[16][key], refs[32][key]))
                 for key in sorted(refs[16])]
        require(all(item["passed"] for item in pairs),
                "A refined paired-order diagnostic failed")
        all_pairs.extend(pairs)
        result = deepcopy(waves[time]["result"])
        result.update(time=time, complete=True, original_histories=768,
            individual_audits_passed=True, paired_orders_passed=True,
            paired_order_checks=pairs, alpha_refined_task_ids=sorted(by_time[time]),
            alpha_refinement_target_draws=alpha.TARGET_DRAWS,
            diagnostic_arrays_validated=False, aggregate_precision_checked=False,
            aggregate_precision_passed=False, population_reference_approved=False,
            payoff_forecast_approved=False, finite_shuffle_exact=False,
            published_mean_gain_bits=None)
        waves[time]["result"] = result
    require(len(all_pairs) == sum(dense.COUNTS.values()) == EXPECTED_PAIR_COUNT,
            "Full t5--8 paired comparison family changed")
    return all_pairs


def diagnostic_wave(plan, result, refs, bindings):
    """Recompute all 768 histories with shared-reference covariance."""
    observed = plan["observed"]
    require(observed in TIMES, "Alpha diagnostics require t5--8")
    arrays, groups = {}, []
    extra, _ = old.frozen_inputs(plan)
    bindings.update(extra)
    tasks = {task["reference_key"]: task for task in plan["tasks"]}
    require(set(plan["source_bindings"]) == {"a1_L0", "a1_L2", "a1_L3"},
            "Incomplete source conditions")
    for name, source in plan["source_bindings"].items():
        require(sha(source["path"]) == source["sha256"], "Original source binding changed")
        bindings[source["path"]] = source["sha256"]
        with np.load(source["path"], allow_pickle=False) as archive:
            tokens, transition = archive["tokens"], archive["transition"]
        require(tokens.shape == (256, 1025) and transition.shape == (256, 64, 64),
                "Incomplete source cohort")
        uses = sorted((use for use in plan["uses"] if use["condition"] == name),
                      key=lambda use: use["index"])
        require([use["index"] for use in uses] == list(range(256)),
                "Selected or duplicate histories")
        if observed == 5:
            mapped = [old.map_history(tokens[index], transition[index], use,
                                      tasks[use["reference_key"]])
                      for index, use in enumerate(uses)]
        else:
            mapped = [parent.map_history(tokens[index], transition[index], use,
                                         tasks[use["reference_key"]], observed)
                      for index, use in enumerate(uses)]
        oracles = np.stack([value[1] for value in mapped])
        keys = [use["reference_key"] for use in uses]
        with np.load(old.FREQ / "refined" / (name + ".npz"), allow_pickle=False) as archive:
            require(archive["stationary_loss"].shape == (256, 8)
                    and archive["stationary_leave_one_bank_out_loss"].shape == (16, 256, 8),
                    "Stationary coverage changed")
            stationary = archive["stationary_loss"][:, observed - 1]
            leave = archive["stationary_leave_one_bank_out_loss"][:, :, observed - 1]
        with np.load(old.FREQ / "stratified" / (name + ".npz"), allow_pickle=False) as archive:
            require(archive["stationary_loss"].shape == (256, 8),
                    "Coarse stationary coverage changed")
            coarse = archive["stationary_loss"][:, observed - 1]
        require(all(np.isfinite(value).all() for value in (stationary, leave, coarse)),
                "Invalid stationary losses")
        bank = old.bank_error(stationary, leave)
        bank["coarse_refined_mean_shift_bits"] = float((stationary - coarse).mean())
        group = dict(condition=name, n_sequences=256, stationary_numerics=bank, rules={})
        arrays[name + "_stationary_loss"] = stationary
        for order in (16, 32):
            prediction = np.stack([refs[order][key]["mean"] for key in keys])
            loss = -np.sum(oracles * np.log2(prediction), axis=1)
            error = old.shared_error(oracles, keys, refs[order])
            gain = stationary - loss
            row = dict(bayes_loss=old.stats(loss), stationary_minus_bayes=old.stats(gain), **error)
            row["stationary_minus_bayes"].update(
                combined_numerical_mcse_bits=float(np.hypot(
                    error["numerical_mcse_bits"], bank["prior_bank_mcse_bits"])),
                combined_log_bias_proxy_bits=error["log_bias_proxy_bits"]
                    + bank["prior_bank_jackknife_bias_proxy_bits"])
            group["rules"][str(order)] = row
            arrays[name + f"_bayes_loss_a{order}"] = loss
            original = np.empty_like(prediction)
            for index, (mapping, _) in enumerate(mapped):
                original[index, mapping] = prediction[index]
            arrays[name + f"_original_label_probs_a{order}"] = original
        difference = arrays[name + "_bayes_loss_a16"] - arrays[name + "_bayes_loss_a32"]
        group["order_loss_difference"] = dict(**old.stats(difference),
            numerical_mcse_bits=float(np.hypot(*[
                group["rules"][str(order)]["numerical_mcse_bits"] for order in (16, 32)])))
        group["neural"] = {}
        for seed in (0, 1):
            if observed == 5:
                ordered, shuffled = [old.model_loss(name, seed, role, source, bindings)
                                     for role in ("ordered", "shuffled")]
            else:
                ordered, shuffled = [parent.model_loss(name, seed, role, source, bindings, observed)
                                     for role in ("ordered", "shuffled")]
            arrays[name + f"_ordered_seed{seed}"] = ordered
            arrays[name + f"_shuffled_seed{seed}"] = shuffled
            group["neural"][str(seed)] = dict(
                gain=old.stats(shuffled - ordered),
                ordered_minus_bayes=old.stats(ordered - arrays[name + "_bayes_loss_a32"]),
                shuffled_minus_stationary=old.stats(shuffled - stationary))
        groups.append(group)
    result.update(diagnostic_arrays_validated=True, diagnostic_groups=groups,
        interpretation="Complete 768-history diagnostic after targeted exponent-rule refinement; payoff timing requires the unchanged aggregate precision assessment.",
        comparator="Source-induced stationary-frequency surrogate, not exact finite-shuffle Bayes.",
        informed_bayes="Original full shared-base prior; source parameters used only for scoring.",
        uncertainty="Shared-prefix MCMC covariance, local log-bias proxies and common-bank jackknife errors; sequence variation remains separate.")
    return arrays


def merge(overlay_path):
    producers, overlay, records, bindings = replacement_records(overlay_path)
    waves, base_bindings = base_waves()
    bindings.update(base_bindings)
    pairs = apply_replacements(waves, records)
    for path in producer_code_paths(producers):
        bindings[str(path.resolve())] = sha(path)
    arrays = {}
    for observed in TIMES:
        wave = waves[observed]
        arrays[observed] = diagnostic_wave(wave["plan"], wave["result"], wave["refs"], bindings)
    result = dict(schema=SCHEMA, complete=True, times=list(TIMES), original_histories=768,
        retained_original_records=1052 - 2 - 8, retained_prior_recovery_records=2,
        substituted_alpha_refinement_records=8,
        alpha_refined_task_ids=sorted(records), individual_audits_passed=True,
        paired_orders_passed=all(item["passed"] for item in pairs),
        paired_order_checks=pairs, paired_reference_count=len(pairs),
        diagnostic_arrays_validated=all(waves[t]["result"]["diagnostic_arrays_validated"] for t in TIMES),
        aggregate_precision_checked=False, aggregate_precision_passed=False,
        population_reference_approved=False, payoff_forecast_approved=False,
        prior_recovery_overlay_sha256=sha(PRIOR_RECOVERY_OVERLAY),
        alpha_refinement_overlay_sha256=sha(overlay_path), input_sha256=bindings,
        waves={str(time): waves[time]["result"] for time in TIMES})
    require(result["paired_orders_passed"] and result["diagnostic_arrays_validated"],
            "Refined cohort did not pass its complete diagnostic reconstruction")
    return result, arrays


def write_output(overlay_path, out):
    out = Path(out)
    require(not out.exists(), "Use a new output folder; preserve prior diagnostics")
    result, arrays = merge(overlay_path)
    out.mkdir()
    for observed in TIMES:
        folder = out / f"t{observed}"
        folder.mkdir()
        np.savez_compressed(folder / "arrays.npz", **arrays[observed])
        result["waves"][str(observed)]["arrays_sha256"] = sha(folder / "arrays.npz")
    # Bind the final, common input family after every diagnostic reader has run.
    for observed in TIMES:
        wave = result["waves"][str(observed)]
        wave["input_sha256"] = result["input_sha256"]
        wave["merger_sha256"] = sha(__file__)
        dense.write(out / f"t{observed}" / "summary.json", wave, immutable=True)
    dense.write(out / "summary.json", result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = write_output(args.overlay, args.out)
    print({key: result[key] for key in ("complete", "individual_audits_passed",
        "paired_orders_passed", "diagnostic_arrays_validated", "payoff_forecast_approved")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
