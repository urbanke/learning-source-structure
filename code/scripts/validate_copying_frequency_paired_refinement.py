"""Independent readback of the paired stationary-frequency refinement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import copying_frequency_paired_refinement as producer


ASSESSMENT = ROOT / "output/copying_payoff_assessment_2026-09-21/summary.json"


def component_accounting(row, fine_mcse, fine_bias, comparison_shift,
                         comparison_mcse, comparison_bias):
    final = 2 * np.hypot(row["mcmc_mcse_bits"], fine_mcse) + 2 * (
        row["mcmc_log_bias_proxy_bits"] + fine_bias)
    refinement = abs(comparison_shift) + 3 * comparison_mcse + comparison_bias
    return float(final), float(refinement), float(final + refinement)


def validate(out):
    out = producer.output_path(out); plan = producer.load_plan(out)
    allocation = producer.read(out / "allocation.json")
    summary = producer.read(out / "summary.json")
    producer.require(allocation["status"] == "complete"
        and allocation["plan_id"] == plan["plan_id"]
        and allocation["summary_sha256"] == producer.sha(out / "summary.json")
        and len(allocation["attempts"]) == producer.BANKS
        and {item["bank"] for item in allocation["attempts"]} == set(range(producer.BANKS))
        and all(item["status"] == "ready" and item["exit_code"] == 0
                and item["elapsed_seconds"] <= producer.TASK_WALL_SECONDS
                and item["observed_peak_rss_bytes"] <= producer.MAX_RSS_BYTES
                for item in allocation["attempts"]), "Allocation did not pass")
    producer.require(summary["complete"] is True and summary["plan_id"] == plan["plan_id"]
        and summary["banks"] == producer.BANKS and summary["times"] == list(range(1, 9)),
        "Summary identity changed")

    artifacts = {"plan.json": producer.sha(out / "plan.json"),
                 "allocation.json": producer.sha(out / "allocation.json"),
                 "summary.json": producer.sha(out / "summary.json")}
    fine_joint = []; coarse_joint = []
    for index in range(producer.BANKS):
        arrays, report_path, log = producer.task_paths(out, index)
        report = producer.read(report_path)
        producer.require(report["complete"] is True and report["bank"] == index
            and report["plan_id"] == plan["plan_id"]
            and report["arrays_sha256"] == producer.sha(arrays)
            and report["peak_rss_bytes"] <= producer.MAX_RSS_BYTES
            and report["elapsed_seconds"] <= producer.TASK_WALL_SECONDS,
            "Bank receipt failed")
        with np.load(arrays, allow_pickle=False) as saved:
            producer.require(saved.files == ["fine_joint", "coarse_joint"]
                and saved["fine_joint"].shape == saved["coarse_joint"].shape
                == (len(plan["frequency_types"]),), "Bank arrays changed")
            fine_joint.append(saved["fine_joint"].copy())
            coarse_joint.append(saved["coarse_joint"].copy())
        for path in (arrays, report_path, log):
            artifacts[str(path.relative_to(out))] = producer.sha(path)

    groups, _ = producer.load_groups()
    kinds = [(json.loads(value)[0], tuple(json.loads(value)[1]))
             for value in plan["frequency_types"]]
    replay_fine = producer.score(np.asarray(fine_joint), groups, kinds)
    replay_coarse = producer.score(np.asarray(coarse_joint), groups, kinds)
    assessment = producer.read(ASSESSMENT); assessment_groups = {
        item["condition"]: item for item in assessment["groups"]}
    saved_groups = {item["condition"]: item for item in summary["groups"]}
    producer.require(set(saved_groups) == set(assessment_groups)
                     == {"a1_L0", "a1_L2", "a1_L3"}, "Condition set changed")
    checks = []; worst_ratio = 0.; all_passed = True
    for condition in saved_groups:
        path = out / (condition + ".npz"); artifacts[path.name] = producer.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            fl = saved["fine_stationary_loss"].copy()
            fj = saved["fine_stationary_leave_one_bank_out_loss"].copy()
            cl = saved["coarse_stationary_loss"].copy()
            cj = saved["coarse_stationary_leave_one_bank_out_loss"].copy()
            producer.require(np.array_equal(saved["paired_difference_leave_one_bank_out_loss"], fj-cj),
                             "Paired leave-one difference changed")
        producer.require(np.array_equal(fl, replay_fine[condition][0])
            and np.array_equal(fj, replay_fine[condition][1])
            and np.array_equal(cl, replay_coarse[condition][0])
            and np.array_equal(cj, replay_coarse[condition][1]), "Scoring replay changed")
        saved_group = saved_groups[condition]
        for cumulative, label in ((False, "instantaneous"), (True, "cumulative")):
            vf, vfj, vc, vcj = fl, fj, cl, cj
            if cumulative:
                vf = np.cumsum(vf, axis=1); vfj = np.cumsum(vfj, axis=2)
                vc = np.cumsum(vc, axis=1); vcj = np.cumsum(vcj, axis=2)
            fm = producer._delete_metrics(vf, vfj)
            dm = producer._delete_metrics(vf-vc, vfj-vcj)
            shift = (vf-vc).mean(0)
            if not cumulative:
                producer.require(np.allclose(shift, saved_group["mean_difference_bits"], rtol=0, atol=1e-15)
                    and np.allclose(fm["mcse"], saved_group["fine_bank_mcse_bits"], rtol=0, atol=1e-15)
                    and np.allclose(fm["bias"], saved_group["fine_bank_bias_bits"], rtol=0, atol=1e-15)
                    and np.allclose(dm["mcse"], saved_group["paired_comparison_mcse_bits"], rtol=0, atol=1e-15)
                    and np.allclose(dm["bias"], saved_group["paired_comparison_bias_bits"], rtol=0, atol=1e-15),
                    "Saved stationary metrics changed")
            for index, row in enumerate(assessment_groups[condition][label]):
                final, refinement, total = component_accounting(row, fm["mcse"][index],
                    fm["bias"][index], shift[index], dm["mcse"][index], dm["bias"][index])
                ratio = total / row["numerical_budget_bits"]; worst_ratio = max(worst_ratio, ratio)
                passed = total <= row["numerical_budget_bits"]; all_passed &= passed
                checks.append(dict(condition=condition, curve=label, time=index+1,
                    final_estimate_allowance_bits=final,
                    stationary_refinement_allowance_bits=refinement,
                    total_excluding_separate_alpha_refinement_bits=total,
                    numerical_budget_bits=row["numerical_budget_bits"], passed=bool(passed)))
    producer.require(all_passed, "Paired stationary component still fails the aggregate budget")
    report = dict(schema="copying-frequency-paired-refinement-validation-v1",
        passed=True, input_integrity_passed=True, bank_tasks_passed=producer.BANKS,
        stationary_component_precision_passed=True,
        separate_bayesian_alpha_refinement_included=False,
        payoff_forecast_approved=False, worst_budget_fraction=worst_ratio,
        assessment_policy_sha256=producer.sha(ASSESSMENT), checks=checks,
        artifact_sha256=dict(sorted(artifacts.items())), validator_sha256=producer.sha(__file__),
        scientific_timing_validated=False)
    producer.write(out / "validation.json", report)
    print(json.dumps(dict(passed=True, tasks=producer.BANKS,
        worst_budget_fraction=worst_ratio, validation=str(out / "validation.json"))))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=producer.DEFAULT_OUT)
    validate(parser.parse_args().out)
