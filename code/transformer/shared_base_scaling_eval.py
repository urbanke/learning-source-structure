"""Evaluate larger shared-base transformers on the existing Section 3 sequences.

This local adapter loads frozen source files and previously released Bayesian
scores. It neither generates test sequences nor computes a Bayesian reference.
All six transformer times are retained. The original training plan binds Bayes
at times 1,4,16 for every depth and time64 for depths0/1. --time64-complete adds
the separately audited time64 release for all four depths without changing that
plan. Run after copying both final weights, config.json and status.json from
the completed training directories.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from eval_shared_base_models import (atomic_json, atomic_npz, fingerprint, models,
    sha_array, sha_file)
from shared_base_conditional_cohort import load_source, moments

N = 256
DEPTHS = (0, 1, 4, 8)
TIMES = (1, 4, 16, 64, 256, 1024)
BASELINES = ("d64", "d64_seed1")
PRIOR = dict(alpha=[.5, 1.5], depths=list(range(9)), ranks="uniform permutation")
RELEASES = {
    "original": "results/transformer/shared_base_conditional_curves/summary.json",
    "time4": "results/transformer/shared_base_conditional_curves/time4_summary.json",
    "time16": "results/transformer/shared_base_conditional_curves/time16_summary.json",
    "time64": "results/transformer/shared_base_morning_audit_2026-09-17/summary.json",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def check_moments(values, saved):
    actual = moments(values)
    require(saved["n"] == actual["n"] and all(
        np.isclose(actual[key], saved[key], rtol=1e-10, atol=2e-12)
        for key in ("mean", "standard_error")), "Saved paired summary does not reproduce")


def one(rows, **fields):
    selected = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    require(len(selected) == 1, f"Expected exactly one saved record: {fields}")
    return selected[0]


def read_plan(path):
    plan = read_json(path)
    require(plan["schema"] == "shared-base-scaling-plan-v1" and plan["seeds"] == [0, 1],
            "Unexpected scaling plan or training repeats")
    require(plan["source"] == dict(d=64, n_ctx=1024, alpha=[.5, 1.5], ls=[0, 8], permute=True),
            "The shared-base training prior differs")
    frozen = plan["evaluation_files_sha256"]
    require(set(RELEASES.values()).issubset(frozen), "Missing frozen evaluation releases")
    for relative, digest in frozen.items():
        require(sha_file(ROOT / relative) == digest, f"Frozen evaluation input changed: {relative}")
    return plan


def load_sources(cohorts, plan):
    """Read all declared sources; no available-case subset or source regeneration."""
    result = {}
    for depth in DEPTHS:
        folder = Path(cohorts) / f"a1_L{depth}"
        for name in ("cohort.json", "protocol.json"):
            key = f"output/shared_base_conditional_curves_2026-09-15/a1_L{depth}/{name}"
            require(sha_file(folder / name) == plan["evaluation_files_sha256"][key],
                    f"Frozen {name} changed at depth{depth}")
        cohort, protocol = read_json(folder / "cohort.json"), read_json(folder / "protocol.json")
        require(cohort["n_sequences"] == N and cohort["n_tokens_per_sequence"] == 1025
                and cohort["alpha"] == 1. and cohort["depth"] == depth
                and cohort["prefixes"] == list(TIMES)
                and [r["index"] for r in cohort["records"]] == list(range(N)),
                "The complete declared source cohort differs")
        require(protocol["reference_prior"] == PRIOR
                and protocol["conditions_hidden_from_predictors"] is True,
                "Source conditions must remain hidden from both predictors")
        records = []
        for index, entry in enumerate(cohort["records"]):
            record = load_source(folder, index)
            records.append(dict(path=folder / "data" / f"seq_{index:06d}.json",
                record=record, file_sha256=entry["file_sha256"]))
        result[depth] = records
    return result


def audited_values(path, digest, depth, time, cohort_digest):
    """Reuse scalar values from the hash-bound, independently audited release."""
    require(sha_file(path) == digest, "Released raw audit changed")
    audit = read_json(path)
    require(audit["schema"] == "shared-base-curve-audit-v1"
            and audit["all_requested_times_validated"] is True
            and audit["depth"] == depth and audit["alpha"] == 1.
            and audit["times"] == [time] and audit["orders"] == [16, 32]
            and audit["n_sequences"] == N
            and audit["provenance"]["cohort_sha256"] == cohort_digest,
            "Bayesian audit has the wrong scope or has not passed")
    row = one(audit["rows"], time=time)
    require(row["cohort_validation"]["passed"] is True, "Bayesian cohort gate failed")
    cases = row["case_audits"]
    require(len(cases) == 2*N and {(c["index"], c["order"]) for c in cases}
            == {(i, order) for i in range(N) for order in (16, 32)}, "Incomplete Bayesian audit")
    require(all(c["ready"] is True and c["status"] == "ready"
                and c["recomputed_final_states"] is True and c["recomputed_finer_grid"] is True
                and c["gates"] and all(c["gates"].values()) for c in cases),
            "A released Bayesian case is unvalidated")
    values = np.asarray([one(cases, index=i, order=32)["record"]["precision"]["bayes_regret_bits"]
                         for i in range(N)])
    require(values.shape == (N,) and np.isfinite(values).all(), "Invalid saved Bayesian scores")
    return values


def load_saved_scores(cohorts):
    """Load and cross-check per-sequence baseline and released Bayes scores."""
    release = {k: read_json(ROOT / v) for k, v in RELEASES.items()}
    require(all(release["original"]["checks"].values()), "Original evaluation validation failed")
    require(release["time4"]["reference_validation_passed"] is True
            and release["time16"]["reference_validation_passed"] is True
            and release["time16"]["prior"] == PRIOR
            and release["time64"]["all_requested_cohorts_passed"] is True,
            "A Bayesian release has not passed validation")
    baseline, bayes = {}, {}
    for depth in DEPTHS:
        folder = Path(cohorts) / f"a1_L{depth}"
        digest = sha_file(folder / "cohort.json")
        receipt = one(release["original"]["provenance"]["cohorts"], depth=depth)
        require(receipt["cohort_sha256"] == digest, "Baseline source cohort changed")
        archive = folder / "transformer_regrets.npz"
        require(sha_file(archive) == receipt["array_sha256"], "Saved baseline scores changed")
        with np.load(archive, allow_pickle=False) as data:
            require(np.array_equal(data["times"], TIMES)
                    and data["models"].tolist() == list(BASELINES)
                    and np.array_equal(data["source_indices"], np.arange(N)), "Baseline indices differ")
            baseline[depth] = data["regret_bits"].copy()
            bayes[depth, 1] = data["bayes_regret_t1_bits"].copy()
        require(baseline[depth].shape == (N, len(TIMES), 2)
                and np.isfinite(baseline[depth]).all(), "Invalid baseline scores")
        for j, time in enumerate(TIMES):
            for seed, name in enumerate(BASELINES):
                check_moments(baseline[depth][:, j, seed], one(release["original"]["rows"],
                              depth=depth, time=time, model=name))
        receipt = one(release["time4"]["provenance"]["cohorts"], depth=depth)
        archive = folder / "shortprefix_t4_predictions.npz"
        require(receipt["cohort_sha256"] == digest and sha_file(archive) == receipt["array_sha256"],
                "Saved time4 predictions changed")
        with np.load(archive, allow_pickle=False) as data:
            require(np.array_equal(data["source_indices"], np.arange(N)), "Time4 indices differ")
            bayes[depth, 4] = data["bayes_regret_bits"].copy()
        receipt = one(release["time16"]["provenance"]["cohorts"], depth=depth)
        bayes[depth, 16] = audited_values(Path(receipt["audit_path"]), receipt["audit_sha256"],
                                        depth, 16, digest)
        for time in (1, 4, 16):
            rows = release["original"]["first_step_paired"] if time == 1 else release[f"time{time}"]["rows"]
            for seed, name in enumerate(BASELINES):
                row = one(rows, depth=depth, time=time, model=name)
                check_moments(bayes[depth, time], row["bayes_regret"])
                check_moments(baseline[depth][:, TIMES.index(time), seed] - bayes[depth, time],
                              row["transformer_minus_bayes"])
        if depth in (0, 1):
            receipt = one(release["time64"]["cohorts"], depth=depth, time=64)
            require(receipt["full_cohort_passed"] is True, "Time64 cohort is not complete")
            bayes[depth, 64] = audited_values(ROOT / receipt["report"], receipt["report_sha256"],
                                             depth, 64, digest)
            for seed, name in enumerate(BASELINES):
                row = receipt["matched_comparisons"][name]
                check_moments(bayes[depth, 64], row["bayes_regret"])
                check_moments(baseline[depth][:, TIMES.index(64), seed]-bayes[depth, 64],
                              row["transformer_minus_bayes"])
    return baseline, bayes


def load_complete_time64(path, cohorts, sources, baseline, original_bayes):
    """Import the complete corrected release, including mandatory replacements.

    The exporter checks the full release and original model identities. Its
    source reader additionally rehashes the bound raw reports, chain states and
    moments once; no chain is sampled or numerical audit rerun. Recompute KL
    from each refined prediction against the same frozen source oracle so all
    subsequent model comparisons remain paired on the original sequence index.
    """
    import export_shared_base_time64 as exported
    from shared_base_curve_audit import aggregate_time, same
    from validate_shared_base_mcmc import json_safe
    path = Path(path).resolve()
    files, release = exported.validate_export(path, read_json(ROOT / RELEASES["original"]))
    validation = read_json(files[1])
    environment = dict(os.environ)
    try:
        origin, source_plan, receipts, bound = exported.source_rows(release["source_summary"])
    finally:
        # The continuation checker sets its numerical runtime environment.
        # Keep those audit settings from leaking into later model evaluation.
        for name in set(os.environ)-set(environment):
            del os.environ[name]
        os.environ.update(environment)
    require(origin["schema"] == release["source_summary_schema"]
            and source_plan["plan_id"] == release["source_plan_id"],
            "Complete time64 source summary/plan differs from the release")
    require(all(validation["inputs"].get(name) == digest for name, digest in bound.items()),
            "Complete time64 source inputs differ from the released hashes")
    raw_files = {name: digest for row in receipts for name, digest in row["task"]["file_sha256"].items()}
    require(raw_files == validation["raw_file_sha256"]
            and validation["raw_files_rehashed_at_export"] is True,
            "Complete time64 raw-file coverage differs from the release")
    by_case = {(row["task"]["depth"], row["task"]["index"], row["task"]["order"]): row
               for row in receipts}
    expected = {(depth, index, order) for depth in DEPTHS for index in range(N) for order in (16, 32)}
    require(len(receipts) == len(expected) and set(by_case) == expected,
            "Complete time64 reference population differs")
    result, unchanged = {}, {}
    for depth in DEPTHS:
        cohort = one(release["cohorts"], depth=depth)
        selected = [row for (d, _, _), row in by_case.items() if d == depth]
        numerical_audit = json_safe(aggregate_time(
            [row["corrected"]["audit"] for row in selected], list(range(N)), [16, 32]))
        exported.checked_cohort_gate(numerical_audit)
        require(same(numerical_audit, cohort["numerical_audit"]),
                "Complete time64 aggregate numerical gates differ from the release")
        folder = Path(cohorts) / f"a1_L{depth}"
        require(len(sources[depth]) == N and all(
            sha_file(folder / name) == cohort["provenance"][key]
            for name, key in (("cohort.json", "cohort_sha256"), ("protocol.json", "protocol_sha256"))),
            "Complete time64 source cohort differs")
        expected_receipts = {row["task"]["key"]: origin["references"][row["task"]["key"]]["case_receipt_sha256"]
                             for (d, _, _), row in by_case.items() if d == depth}
        require(cohort["case_receipts"] == expected_receipts,
                "Complete time64 case receipts differ from the released population")
        values = np.empty(N)
        for index, item in enumerate(sources[depth]):
            row = by_case[depth, index, 32]
            task = row["task"]
            require(task["time"] == 64 and Path(task["source_folder"]).resolve() == folder.resolve(),
                    "Complete time64 case uses another source/time")
            reference = read_json(task["reference"])
            identity = reference["identity"]["inputs"]
            source = item["record"]
            tokens = np.asarray(source["tokens"])
            require(identity["index"] == index and identity["prefix"] == 64
                    and identity["source_id"] == source["source_id"]
                    and identity["source_file_sha256"] == item["file_sha256"]
                    and identity["protocol_sha256"] == cohort["provenance"]["protocol_sha256"]
                    and identity["tokens_sha256"] == source["tokens_sha256"]
                    and identity["prefix_tokens_sha256"] == sha_array(tokens[:64], '<i2'),
                    "Complete time64 reference/source pairing differs")
            precision = row["corrected"]["audit"]["record"]["precision"]
            prediction = np.asarray(precision["predictive_probs"], dtype=float)
            require(prediction.shape == (64,) and np.isfinite(prediction).all()
                    and np.all(prediction > 0) and abs(prediction.sum()-1) < 1e-10,
                    "Invalid complete time64 refined prediction")
            oracle = np.asarray(source["oracle"]["transition_probs"])[tokens[63]]
            values[index] = np.sum(rel_entr(oracle, prediction)) / np.log(2)
            require(np.isclose(values[index], precision["bayes_regret_bits"], atol=1e-12, rtol=1e-12),
                    "Complete time64 KL differs from the audited probability vector")
        for seed, name in enumerate(BASELINES):
            saved = cohort["matched_comparisons"][name]
            check_moments(values, saved["bayes_regret"])
            check_moments(baseline[depth][:, TIMES.index(64), seed], saved["transformer_regret"])
            check_moments(baseline[depth][:, TIMES.index(64), seed]-values, saved["transformer_minus_bayes"])
        if (depth, 64) in original_bayes:
            difference = float(np.max(np.abs(values-original_bayes[depth, 64])))
            require(difference < 2e-12, "Complete time64 changed an originally released sequence score")
            unchanged[str(depth)] = difference
        result[depth, 64] = values
    bound_files = {str(Path(p).resolve()): sha_file(p) for p in [*files, release["source_summary"], exported.__file__]}
    provenance = dict(schema="shared-base-scaling-time64-supplement-v1",
        role="Completed time64 evaluation reference added after the frozen training plan; no training-plan change or checkpoint selection.",
        files_sha256=bound_files, source_plan_id=release["source_plan_id"],
        depths=list(DEPTHS), times=[64], n_sequences_per_depth=N, reported_integration_order=32,
        mandatory_replacements_checked=True, raw_files_rehashed_before_evaluation=True,
        raw_file_count=len(raw_files), raw_files_manifest_sha256=fingerprint(raw_files),
        release_inputs_manifest_sha256=fingerprint(validation["inputs"]),
        per_sequence_values_sha256=fingerprint({str(d): result[d, 64].tolist() for d in DEPTHS}),
        original_time64_max_absolute_difference_bits=unchanged,
        source_sampling=False, posterior_sampling=False, model_inference=False)
    return result, provenance


def check_supplement_files(provenance):
    for name, digest in provenance["files_sha256"].items():
        require(sha_file(name) == digest, "Complete time64 release changed during evaluation: " + name)


def validate_final_checkpoint(plan_path, run):
    import torch
    from train_shared_base_scaling import (read_plan as training_plan, planned_config,
        implementation_hashes, SOURCE_CONTRACT, digest_json)
    plan, plan_hash = training_plan(plan_path)
    run = Path(run)
    cfg, status = read_json(run / "config.json"), read_json(run / "status.json")
    seed = cfg["seed"]
    expected = planned_config(plan, seed)
    expected_hash = digest_json(expected)
    require(cfg["config"] == expected and cfg["config_hash"] == expected_hash
            and cfg["plan_sha256"] == plan_hash
            and cfg["implementation_hashes"] == implementation_hashes()
            and cfg["source_contract"] == SOURCE_CONTRACT
            and cfg["source_contract_hash"] == digest_json(SOURCE_CONTRACT)
            and cfg["n_params"] == plan["n_params"] and cfg["precision"] == "float32",
            "Training configuration/provenance differs from the fixed plan")
    for key, value in dict(family="lsa", d=64, n_ctx=1024, alpha=[.5,1.5],
                           ls=[0,8], no_permute=False).items():
        require(cfg[key] == value, f"Training source field differs: {key}")
    architecture = dict(vocab=64, n_ctx=1024, n_layer=expected["n_layer"],
        n_head=expected["n_head"], d_model=expected["d_model"], mlp=True, dropout=0.)
    require(cfg["model"] == architecture, "Training architecture differs")
    require(status["schema"] == "shared-base-scaling-status-v1"
            and status["complete"] is True and status["reason"] == "completed"
            and status["step"] == status["production_steps"] == expected["steps"]
            and status["completed_sequences"] == expected["steps"] * expected["batch"]
            and status["config_hash"] == expected_hash and status["plan_sha256"] == plan_hash,
            "Training is incomplete or status metadata differs")
    path = run / f"model_{expected['steps']}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint["schema"] == "shared-base-scaling-weights-v1"
            and checkpoint["cfg"] == cfg and checkpoint["step"] == expected["steps"]
            and checkpoint["completed_batches"] == expected["steps"]
            and checkpoint["completed_sequences"] == expected["steps"] * expected["batch"]
            and checkpoint["config_hash"] == expected_hash and checkpoint["plan_sha256"] == plan_hash
            and checkpoint["implementation_hashes"] == implementation_hashes()
            and checkpoint["source_contract_hash"] == digest_json(SOURCE_CONTRACT),
            "Final checkpoint metadata differs or planned exposure is incomplete")
    require(checkpoint["model"] and all(torch.isfinite(v).all().item()
            for v in checkpoint["model"].values()), "Invalid checkpoint tensors")
    return dict(seed=seed, step=expected["steps"], completed_sequences=status["completed_sequences"],
        checkpoint_sha256=sha_file(path), config_sha256=sha_file(run / "config.json"),
        status_sha256=sha_file(run / "status.json"), n_params=cfg["n_params"], plan_sha256=plan_hash)


def validate_runs(plan_path, plan, runs):
    require(len(runs) == 2 and len({Path(p).resolve() for p in runs}) == 2,
            "Require two distinct completed training directories")
    validated = [validate_final_checkpoint(plan_path, Path(run)) for run in runs]
    require(sorted(v["seed"] for v in validated) == [0, 1], "Require exactly one model per training seed")
    return sorted(zip(runs, validated), key=lambda pair: pair[1]["seed"])


def read_model_scores(output, records, run_name):
    values = np.empty((N, len(TIMES)))
    identities = []
    for i, item in enumerate(records):
        path = Path(output) / "models" / run_name / f"seq_{i:06d}.json"
        report = read_json(path)
        array_path = path.with_suffix(".npz")
        require(report["cache_id"] == fingerprint(report["identity"])
                and report["identity"]["source_file_sha256"] == item["file_sha256"]
                and report["source_id"] == item["record"]["source_id"]
                and report["prefixes"] == list(TIMES)
                and report["target_indices"] == list(TIMES)
                and report["checks"]["alignment"]["passed"] is True
                and sha_file(array_path) == report["arrays_sha256"], "New model record failed readback")
        tokens = np.asarray(item["record"]["tokens"])
        oracle = np.asarray(item["record"]["oracle"]["transition_probs"])[tokens[np.asarray(TIMES)-1]]
        with np.load(array_path, allow_pickle=False) as data:
            p = data["predictive_probs"]
            require(np.array_equal(data["oracle_probs"], oracle)
                    and np.array_equal(data["targets"], tokens[np.asarray(TIMES)])
                    and np.array_equal(p, report["predictive_probs"])
                    and np.isfinite(p).all() and np.all(p > 0)
                    and np.max(np.abs(p.sum(axis=1)-1)) < 1e-12
                    and np.max(np.abs(np.exp(data["logp"])-p)) < 1e-15,
                    "New probability/oracle/target readback failed")
            values[i] = np.sum(rel_entr(oracle, p), axis=1) / np.log(2)
        identities.append(dict(record_sha256=sha_file(path), arrays_sha256=report["arrays_sha256"]))
    return values, fingerprint(identities)


def comparison_rows(depth, new_values, old_values, bayes):
    rows = []
    for seed in (0, 1):
        for j, time in enumerate(TIMES):
            new, old = new_values[:, j, seed], old_values[:, j, seed]
            row = dict(depth=depth, alpha=1., time=time, seed=seed,
                baseline_regret=moments(old), larger_regret=moments(new),
                improvement_over_baseline=moments(old-new), bayes_available=(depth, time) in bayes)
            if row["bayes_available"]:
                b = bayes[depth, time]
                row.update(bayes_regret=moments(b), baseline_minus_bayes=moments(old-b),
                           larger_minus_bayes=moments(new-b))
            rows.append(row)
    return rows


def evaluate(plan_path, runs, cohorts, out, *, device="cpu", batch=4, torch_threads=1,
             time64_complete=None):
    plan_path, out = Path(plan_path), Path(out)
    plan = read_plan(plan_path)
    ordered = validate_runs(plan_path, plan, runs)
    require(len({Path(run).name for run, _ in ordered}) == 2, "Run directory names must differ")
    sources = load_sources(cohorts, plan)
    baseline, bayes = load_saved_scores(cohorts)
    require(not out.exists(), "Use a new output directory; completed evaluations are immutable")
    require(batch >= 1 and torch_threads >= 1, "Batch and thread counts must be positive")
    supplements = []
    if time64_complete is not None:
        additional, receipt = load_complete_time64(time64_complete, cohorts, sources, baseline, bayes)
        bayes.update(additional)
        supplements.append(receipt)
    rows, provenance = [], []
    for depth in DEPTHS:
        destination = out / f"a1_L{depth}"
        models(destination, sources[depth], TIMES, [run for run, _ in ordered],
               step=plan["config"]["steps"], device=device, batch=batch, torch_threads=torch_threads)
        new = np.empty((N, len(TIMES), 2))
        receipts = []
        for seed, (run, verification) in enumerate(ordered):
            new[:, :, seed], digest = read_model_scores(destination, sources[depth], Path(run).name)
            receipts.append(dict(seed=seed, prediction_digest=digest, training=verification))
        rows.extend(comparison_rows(depth, new, baseline[depth], bayes))
        array_path = destination / "paired_scores.npz"
        available_times = [t for t in TIMES if (depth, t) in bayes]
        atomic_npz(array_path, source_indices=np.arange(N), times=np.asarray(TIMES),
            larger_regret_bits=new, baseline_regret_bits=baseline[depth],
            bayes_times=np.asarray(available_times),
            bayes_regret_bits=np.stack([bayes[depth, t] for t in available_times], axis=1))
        provenance.append(dict(depth=depth, models=receipts, scores_path=str(array_path.resolve()),
                               scores_sha256=sha_file(array_path)))
    # Recheck frozen inputs and final checkpoint metadata before publishing.
    read_plan(plan_path)
    require(validate_runs(plan_path, plan, runs) == ordered, "Training inputs changed during evaluation")
    for supplement in supplements:
        check_supplement_files(supplement)
    summary = dict(schema="shared-base-scaling-evaluation-v1", plan_sha256=sha_file(plan_path),
        n_sequences_per_depth=N, n_tokens_per_sequence=1025, depths=list(DEPTHS), times=list(TIMES),
        training_prior=PRIOR, evaluated_conditions_hidden=True, seed_order=[0, 1],
        rows=rows, provenance=provenance, evaluation_files_sha256=plan["evaluation_files_sha256"],
        additional_evaluation_releases=supplements,
        evaluator_sha256=sha_file(__file__),
        statistic="Mean KL(true next-token transition row || predictor), bits. Standard errors over independent sequences; all model differences paired per sequence.",
        scope="Existing256 sequences per depth; no test regeneration or Bayesian inference. Missing Bayesian times remain absent. Two separate training repeats.",
        limitation="These cohorts motivated the capacity follow-up; this is a development comparison. Numerical Bayesian diagnostics were validated separately and do not prove global convergence.")
    atomic_json(out / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--cohorts", type=Path, default=ROOT / "output/shared_base_conditional_curves_2026-09-15")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--time64-complete", type=Path,
        help="Optional validated complete time64 summary for all four depths; recorded separately from the training plan")
    args = parser.parse_args()
    summary = evaluate(args.plan, args.runs, args.cohorts, args.out, device=args.device,
                       batch=args.batch, torch_threads=args.torch_threads, time64_complete=args.time64_complete)
    print(json.dumps(dict(output=str(args.out / "comparison.json"), comparisons=len(summary["rows"]),
                          n_sequences_per_depth=N, training_repeats=2)))


if __name__ == "__main__":
    main()
