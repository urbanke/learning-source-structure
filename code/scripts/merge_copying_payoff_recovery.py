"""Validate a two-record recovery overlay and remerge the full copying cohort.

The parent campaign is immutable.  Exactly two continued chains may replace
their declared failed parent records.  Every other t6--8 record is reread from
the frozen campaign, and no diagnostic array is released unless all 976
individual records and all 488 paired integration-order comparisons pass.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import merge_copying_payoff_parallel as parent


dense = parent.dense
old = parent.old
ROOT, require, read, sha = parent.ROOT, parent.require, parent.read, parent.sha
TIMES = parent.TIMES
BASELINE_DRAWS = 16384
TARGET_DRAWS = 65536
EXPECTED_REPLACEMENTS = {
    7: "t0007_453a8f0bdcd20ca9731b_a32",
    8: "t0008_fccb61c7bef875d43ac7_a16",
}
PARENT_FILES = ("report", "identity", "audit", "checkpoint", "moments")
REPLACEMENT_FILES = PARENT_FILES + ("receipt",)


def verify_bound(item, expected=None):
    require(isinstance(item, dict) and set(("path", "sha256")) <= set(item),
            "Malformed bound file")
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            "Bound file missing or linked: " + str(path))
    if expected is not None:
        require(path.resolve() == Path(expected).resolve(),
                "Bound file has an unexpected path: " + str(path))
    if "size_bytes" in item:
        require(path.stat().st_size == item["size_bytes"], "Bound file size changed")
    require(sha(path) == item["sha256"], "Bound file hash changed: " + str(path))
    return path


def validate_recovery_plan(path):
    """Use the recovery driver's own frozen-plan validator."""
    import copying_payoff_recovery as recovery
    path = Path(path)
    plan = recovery.load_plan(path.parent)
    require(path.resolve() == (Path(plan["output"]) / "plan.json").resolve(),
            "Overlay does not bind the validated recovery plan")
    return plan


def moments_checked(moments, estimate, n_seen, expected_draws):
    """The frozen moment checks, parameterized only by the continued draw count."""
    p, chain = moments["mean"], moments["chain_means"]
    require(p.shape == (64,) and chain.shape == (4, 64)
            and int(moments["chains"]) == 4
            and int(moments["draws"]) == expected_draws,
            "Predictive moment shape or draw count changed")
    require(all(np.isfinite(value).all() for value in moments.values())
            and p.min() > 0 and chain.min() > 0
            and abs(p.sum() - 1) < 1e-7
            and np.max(abs(chain.sum(1) - 1)) < 1e-7,
            "Invalid predictive moments")
    require(np.allclose(p, estimate["predictive_probs"], rtol=0, atol=2e-14)
            and np.allclose(p, chain.mean(0), rtol=0, atol=2e-14),
            "Predictive readbacks disagree")
    require(np.max(abs(p[n_seen:] - p[n_seen])) < 2e-14
            and np.max(abs(chain[:, n_seen:] - chain[:, n_seen, None])) < 2e-14,
            "Unseen labels must share equal mass")
    for key in ("cov_batch", "cov_chain"):
        covariance = moments[key]
        require(covariance.shape == (64, 64)
                and np.max(abs(covariance - covariance.T)) < 1e-14
                and np.linalg.eigvalsh(covariance).min() > -1e-12,
                "Invalid predictive covariance")
        require(np.max(abs(covariance[:, n_seen:] - covariance[:, n_seen, None])) < 2e-14,
                "Unequal unseen covariance")
    expected = np.sqrt(np.maximum(0, np.maximum(
        np.diag(moments["cov_batch"]), np.diag(moments["cov_chain"]))))
    require(np.allclose(expected, moments["mcse_probs"], rtol=1e-10, atol=2e-14),
            "Moment MCSE mismatch")


def _check_prefix_provenance(item, parent_checkpoint, child_checkpoint):
    prefix = item["sample_prefix"]
    require(prefix["verified"] is True
            and prefix["parent_draw_counts"] == [BASELINE_DRAWS] * 4
            and prefix["child_draw_counts"] == [TARGET_DRAWS] * 4,
            "Recovery did not attest the fixed sample prefix")
    continuity = item["rng_continuity"]
    require(continuity["passed"] is True
            and continuity["parent_rng_states_sha256"] == continuity["loaded_rng_states_sha256"]
            and continuity["parent_terminal_states_sha256"] == continuity["loaded_terminal_states_sha256"],
            "Recovery did not restore the parent RNG and terminal states")
    for value in (*prefix["array_prefix_sha256"].values(), prefix["burn_done_sha256"],
                  continuity["parent_rng_states_sha256"],
                  continuity["parent_terminal_states_sha256"]):
        require(isinstance(value, str) and len(value) == 64,
                "Malformed recovery provenance digest")

    from unseen_context_refinement import checkpoint_metadata, verify_preserved_samples
    spec = dict(baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS)
    counts = verify_preserved_samples(parent_checkpoint, child_checkpoint, spec, complete=True)
    require(counts == prefix["child_draw_counts"], "Recovery draw counts disagree")
    metadata = checkpoint_metadata(parent_checkpoint)
    expected_names = {f"{kind}_{chain}" for chain in range(4)
                      for kind in ("predictions", "depths", "alpha_indices", "scalars")}
    require(set(prefix["array_prefix_sha256"]) == expected_names,
            "Recovery prefix proof has incomplete arrays")
    require(all(prefix["array_prefix_sha256"][name] == metadata["array_hashes"][name]
                for name in expected_names)
            and prefix["burn_done_sha256"] == metadata["array_hashes"]["burn_done"],
            "Recovery prefix hashes differ from the parent checkpoint")


def load_overlay(path, campaign):
    """Bind the immutable recovery declaration before reading any replacement."""
    path, campaign = Path(path), Path(campaign)
    overlay = read(path)
    require(overlay.get("schema") == "copying-payoff-recovery-overlay-v1"
            and overlay.get("complete") is True
            and overlay.get("original_acceptance_criteria_unchanged") is True
            and overlay.get("baseline_draws") == BASELINE_DRAWS
            and overlay.get("target_draws") == TARGET_DRAWS,
            "Incomplete or changed recovery overlay")
    bindings = {str(path.resolve()): sha(path)}
    campaign_file = verify_bound(overlay["campaign"], campaign / "campaign.json")
    bindings[str(campaign_file)] = overlay["campaign"]["sha256"]
    recovery_plan_path = verify_bound(overlay["recovery_plan"])
    bindings[str(recovery_plan_path)] = overlay["recovery_plan"]["sha256"]
    recovery_plan = validate_recovery_plan(recovery_plan_path)
    recovery_root = recovery_plan_path.parent
    require(Path(overlay["parent_output"]).resolve() == campaign.resolve()
            and Path(recovery_plan["parent_output"]).resolve() == campaign.resolve()
            and recovery_plan["parent_campaign"] == overlay["campaign"]
            and recovery_plan["baseline_draws_per_chain"] == BASELINE_DRAWS
            and recovery_plan["target_draws_per_chain"] == TARGET_DRAWS
            and recovery_plan["original_acceptance_criteria_unchanged"] is True,
            "Overlay differs from its validated recovery plan")

    replacements = overlay.get("replacements")
    require(isinstance(replacements, list) and len(replacements) == 2,
            "Exactly two declared recoveries are required")
    by_time = {}
    for item in replacements:
        time = item.get("time")
        require(time in EXPECTED_REPLACEMENTS
                and item.get("task_id") == EXPECTED_REPLACEMENTS[time]
                and item.get("order") in (16, 32)
                and item.get("baseline_draws") == BASELINE_DRAWS
                and item.get("target_draws") == TARGET_DRAWS
                and time not in by_time,
                "Recovery task set or fixed draw budget changed")
        wave = campaign / f"t{time}"
        plan = dense.load_plan(wave)
        _, campaign_bindings = parent.campaign_binding(campaign, wave)
        bindings.update(campaign_bindings)
        task = next((task for task in plan["tasks"] if task["id"] == item["task_id"]), None)
        require(task is not None and item.get("task") == task and item["order"] == task["order"],
                "Recovered task differs from the frozen task")
        plan_path = verify_bound(item["parent_plan"], wave / "plan.json")
        require(item["parent_plan"]["sha256"] == sha(wave / "plan.json"),
                "Recovered parent plan differs")
        bindings[str(plan_path)] = item["parent_plan"]["sha256"]

        original_paths = dense.paths(wave, task)
        require(set(item["parent_files"]) == set(PARENT_FILES),
                "Recovery omitted a parent artifact")
        require(set(item["replacement_files"]) == set(REPLACEMENT_FILES),
                "Recovery omitted a replacement artifact")
        for name in PARENT_FILES:
            artifact = verify_bound(item["parent_files"][name], original_paths[name])
            bindings[str(artifact)] = item["parent_files"][name]["sha256"]
        replacement_dir = recovery_root / "tasks" / task["id"]
        expected_names = dict(report="report.json", identity="identity.json",
            audit="raw_audit.json", checkpoint="chains.npz", moments="moments.npz",
            receipt="receipt.json")
        for name in REPLACEMENT_FILES:
            artifact = verify_bound(item["replacement_files"][name],
                                    replacement_dir / expected_names[name])
            bindings[str(artifact)] = item["replacement_files"][name]["sha256"]
        require(set(Path(v["path"]).resolve() for v in item["parent_files"].values()).isdisjoint(
                    Path(v["path"]).resolve() for v in item["replacement_files"].values()),
                "Recovery would overwrite a parent artifact")

        parent_report = read(original_paths["report"])
        parent_identity = read(original_paths["identity"])
        parent_audit = read(original_paths["audit"])
        require(parent_report["complete"] is True and parent_report["audit_passed"] is False
                and parent_report["task"] == parent_identity["task"] == task
                and parent_report["plan_id"] == parent_identity["plan_id"] == plan["plan_id"],
                "Declared recovery does not replace its failed parent record")
        require(parent_report["audit_passed"] == (parent_audit["passed"] and all(
            check["passed"] for check in parent_audit["final_state_grid_checks"])),
            "Parent audit acceptance changed")
        for name in ("identity", "audit", "checkpoint", "moments"):
            require(parent_report[name + "_sha256"] == item["parent_files"][name]["sha256"],
                    "Parent report binding differs from the overlay")

        receipt = read(item["replacement_files"]["receipt"]["path"])
        require(receipt.get("task") == task
                and receipt.get("sample_prefix") == item["sample_prefix"]
                and receipt.get("rng_continuity") == item["rng_continuity"]
                and receipt.get("plan_id") == recovery_plan["plan_id"]
                and receipt.get("parent_plan_id") == plan["plan_id"]
                and receipt.get("baseline_draws_per_chain") == BASELINE_DRAWS
                and receipt.get("target_draws_per_chain") == TARGET_DRAWS
                and receipt.get("sampling_complete") is True
                and receipt.get("audit_passed") is True
                and receipt.get("status") == "ready"
                and receipt.get("original_acceptance_criteria_unchanged") is True
                and receipt.get("parent_plan") == item["parent_plan"]
                and receipt.get("parent_files") == item["parent_files"]
                and receipt.get("replacement_files") == {
                    name: item["replacement_files"][name] for name in PARENT_FILES},
                "Recovery receipt differs from the overlay")
        prepared = replacement_dir / "prepared.json"
        require(prepared.is_file() and not prepared.is_symlink()
                and sha(prepared) == receipt["prepared_sha256"],
                "Recovery preparation receipt changed")
        bindings[str(prepared)] = receipt["prepared_sha256"]
        _check_prefix_provenance(item, original_paths["checkpoint"],
                                 Path(item["replacement_files"]["checkpoint"]["path"]))
        by_time[time] = dict(item=item, plan=plan, task=task,
                             recovery_plan=recovery_plan)
    require(set(by_time) == set(EXPECTED_REPLACEMENTS),
            "Recovery overlay must contain the fixed t7 and t8 tasks")
    return overlay, by_time, bindings


def _load_record(plan, task, paths, expected_draws, replacement=None):
    report, audit, identity = (read(paths[name]) for name in ("report", "audit", "identity"))
    require(report["task"] == identity["task"] == task and report["complete"] is True,
            "Reference identity changed")
    if replacement is None:
        require(report["plan_id"] == identity["plan_id"] == plan["plan_id"],
                "Parent reference plan changed")
    else:
        recovery_plan = replacement["recovery_plan"]
        require(report["plan_id"] == identity["plan_id"] == recovery_plan["plan_id"]
                and report["parent_plan_id"] == identity["parent_plan_id"] == plan["plan_id"]
                and report["baseline_draws_per_chain"] == BASELINE_DRAWS
                and report["target_draws_per_chain"] == TARGET_DRAWS
                and report["parent_samples_preserved"] is True
                and report["exact_rng_restored_before_extension"] is True,
                "Replacement provenance differs")
        original_identity = read(replacement["item"]["parent_files"]["identity"]["path"])
        require(identity["target"] == original_identity["target"]
                and identity["proposal"] == original_identity["proposal"]
                and identity["runtime"] == original_identity["runtime"],
                "Replacement target, proposal or runtime changed")
    for name in ("audit", "identity", "checkpoint", "moments"):
        require(report[name + "_sha256"] == sha(paths[name]),
                "Reference artifact binding changed")
    passed = audit["passed"] and all(check["passed"] for check in audit["final_state_grid_checks"])
    require(report["audit_passed"] == passed, "Audit acceptance changed")
    with np.load(paths["moments"], allow_pickle=False) as archive:
        moments = {key: archive[key].copy() for key in archive}
    moments_checked(moments, report["estimate"], len(set(task["tokens"])), expected_draws)
    return report, moments


def load_campaign_references(campaign, overlay_path):
    """Rerun every individual gate and all 488 family-wide paired checks."""
    overlay, replacements, bindings = load_overlay(overlay_path, campaign)
    waves, all_pairs = {}, []
    for time in TIMES:
        folder = Path(campaign) / f"t{time}"
        plan = dense.load_plan(folder)
        _, campaign_bindings = parent.campaign_binding(campaign, folder)
        bindings.update(campaign_bindings)
        bindings[str((folder / "plan.json").resolve())] = sha(folder / "plan.json")
        require(len(plan["tasks"]) == 2 * dense.COUNTS[time], "Wrong full wave scope")
        refs = {16: {}, 32: {}}
        retained, substituted = 0, []
        for task in plan["tasks"]:
            replacement = replacements.get(time) if task["id"] == EXPECTED_REPLACEMENTS.get(time) else None
            if replacement is None:
                paths = dense.paths(folder, task)
                report, moments = _load_record(plan, task, paths, BASELINE_DRAWS)
                require(report["audit_passed"] is True,
                        "An undeclared parent failure remains in the full cohort")
                retained += 1
                for name in PARENT_FILES:
                    bindings[str(paths[name])] = sha(paths[name])
            else:
                item = replacement["item"]
                paths = {name: Path(item["replacement_files"][name]["path"])
                         for name in PARENT_FILES}
                report, moments = _load_record(plan, task, paths, TARGET_DRAWS, replacement)
                require(report["audit_passed"] is True and item["audit_passed"] is True,
                        "Recovered record failed an individual gate")
                substituted.append(task["id"])
            require(task["reference_key"] not in refs[task["order"]],
                    "Duplicate canonical reference")
            refs[task["order"]][task["reference_key"]] = moments
        expected_substitutions = [] if time == 6 else [EXPECTED_REPLACEMENTS[time]]
        require(substituted == expected_substitutions
                and retained + len(substituted) == len(plan["tasks"]),
                "Overlay selected or omitted parent records")
        require(set(refs[16]) == set(refs[32]) and len(refs[16]) == dense.COUNTS[time],
                "Missing paired integration orders")
        pairs = [dict(time=time, reference_key=key,
                      **parent.paired_order_check(refs[16][key], refs[32][key]))
                 for key in sorted(refs[16])]
        all_pairs.extend(pairs)
        waves[time] = dict(plan=plan, refs=refs, result=dict(
            time=time, complete=True, expected_tasks=len(plan["tasks"]),
            original_histories=768, retained_parent_records=retained,
            substituted_task_ids=substituted, individual_audits_passed=True,
            paired_order_checks=pairs,
            paired_orders_passed=all(pair["passed"] for pair in pairs),
            diagnostic_arrays_validated=False, aggregate_precision_checked=False,
            aggregate_precision_passed=False, population_reference_approved=False,
            payoff_forecast_approved=False, published_mean_gain_bits=None))
    require(len(all_pairs) == parent.FAMILY_PAIRS == 488,
            "Full paired comparison family changed")
    paired_passed = all(pair["passed"] for pair in all_pairs)
    for wave in waves.values():
        wave["result"]["family_pairs"] = parent.FAMILY_PAIRS
        wave["result"]["family_comparisons"] = parent.FAMILY_COMPARISONS
        wave["result"]["family_alpha"] = parent.FAMILY_ALPHA
        wave["result"]["z_multiplier"] = parent.Z_MULTIPLIER
        wave["result"]["all_family_paired_orders_passed"] = paired_passed
    return overlay, waves, bindings, paired_passed


def diagnostic_wave(plan, result, refs, bindings):
    """Recompute all 768 mapped histories with shared-reference covariance."""
    time = plan["observed"]
    require(time in (7, 8), "Recovery diagnostics are released only for t7 and t8")
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
                "Incomplete cohort")
        uses = sorted((use for use in plan["uses"] if use["condition"] == name),
                      key=lambda use: use["index"])
        require([use["index"] for use in uses] == list(range(256)),
                "Selected or duplicate histories")
        mapped = [parent.map_history(tokens[index], transition[index], use,
                                     tasks[use["reference_key"]], time)
                  for index, use in enumerate(uses)]
        oracles = np.stack([value[1] for value in mapped])
        keys = [use["reference_key"] for use in uses]
        with np.load(old.FREQ / "refined" / (name + ".npz"), allow_pickle=False) as archive:
            require(archive["stationary_loss"].shape == (256, 8)
                    and archive["stationary_leave_one_bank_out_loss"].shape == (16, 256, 8),
                    "Stationary coverage changed")
            stationary = archive["stationary_loss"][:, time - 1]
            leave = archive["stationary_leave_one_bank_out_loss"][:, :, time - 1]
        with np.load(old.FREQ / "stratified" / (name + ".npz"), allow_pickle=False) as archive:
            require(archive["stationary_loss"].shape == (256, 8),
                    "Coarse stationary coverage changed")
            coarse = archive["stationary_loss"][:, time - 1]
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
            ordered, shuffled = [parent.model_loss(name, seed, role, source, bindings, time)
                                 for role in ("ordered", "shuffled")]
            arrays[name + f"_ordered_seed{seed}"] = ordered
            arrays[name + f"_shuffled_seed{seed}"] = shuffled
            group["neural"][str(seed)] = dict(
                gain=old.stats(shuffled - ordered),
                ordered_minus_bayes=old.stats(ordered - arrays[name + "_bayes_loss_a32"]),
                shuffled_minus_stationary=old.stats(shuffled - stationary))
        groups.append(group)
    result.update(diagnostic_arrays_validated=True, diagnostic_groups=groups,
                  interpretation="Complete 768-history recovery diagnostic; payoff timing requires "
                                 "the separate aggregate precision assessment.",
                  comparator="Source-induced stationary-frequency surrogate, not exact finite-shuffle Bayes.",
                  uncertainty="Shared-prefix MCMC covariance, local log-bias proxies and common-bank "
                              "jackknife errors; sequence variation remains separate.")
    return arrays


def merge(campaign, overlay_path):
    overlay, waves, bindings, paired_passed = load_campaign_references(campaign, overlay_path)
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    all_individual = all(wave["result"]["individual_audits_passed"] for wave in waves.values())
    arrays = {}
    if all_individual and paired_passed:
        for time in (7, 8):
            wave = waves[time]
            arrays[time] = diagnostic_wave(wave["plan"], wave["result"], wave["refs"], bindings)
    result = dict(schema="copying-payoff-recovery-merge-v1",
        overlay_sha256=sha(overlay_path), campaign_sha256=overlay["campaign"]["sha256"],
        complete=True, expected_tasks=sum(2 * dense.COUNTS[time] for time in TIMES),
        retained_parent_records=sum(wave["result"]["retained_parent_records"] for wave in waves.values()),
        substituted_task_ids=[EXPECTED_REPLACEMENTS[7], EXPECTED_REPLACEMENTS[8]],
        individual_audits_passed=all_individual, paired_orders_passed=paired_passed,
        paired_order_family=dict(pairs=parent.FAMILY_PAIRS,
            coordinates=parent.FAMILY_COMPARISONS, alpha=parent.FAMILY_ALPHA,
            z_multiplier=parent.Z_MULTIPLIER),
        complete_cohorts_passed=all_individual and paired_passed,
        diagnostic_arrays_validated=set(arrays) == {7, 8},
        aggregate_precision_checked=False, aggregate_precision_passed=False,
        population_reference_approved=False, payoff_forecast_approved=False,
        waves={str(time): wave["result"] for time, wave in waves.items()},
        input_sha256=bindings)
    return result, arrays


def exit_status(result):
    if not result["complete"]:
        return 3
    if not result["individual_audits_passed"] or not result["paired_orders_passed"]:
        return 2
    require(result["diagnostic_arrays_validated"],
            "Passing full cohorts did not produce both diagnostic arrays")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Use a new output folder; preserve prior diagnostics")
    result, arrays = merge(args.campaign, args.overlay)
    args.out.mkdir(parents=True)
    if result["complete_cohorts_passed"]:
        require(set(arrays) == {7, 8}, "Both complete cohorts are required before array release")
        for time in (7, 8):
            folder = args.out / f"t{time}"
            folder.mkdir()
            np.savez_compressed(folder / "arrays.npz", **arrays[time])
            result["waves"][str(time)]["arrays_sha256"] = sha(folder / "arrays.npz")
    dense.write(args.out / "summary.json", result, immutable=True)
    print({key: result[key] for key in ("complete", "individual_audits_passed",
        "paired_orders_passed", "diagnostic_arrays_validated", "payoff_forecast_approved")})
    return exit_status(result)


if __name__ == "__main__":
    raise SystemExit(main())
