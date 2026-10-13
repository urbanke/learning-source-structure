"""Release the fixed eight-sequence shared-base long-context comparison.

Read already audited cluster JSON summaries and existing local probabilities.
This performs no posterior sampling or transformer inference. Cluster raw-chain
audits are retained as provenance; they are not rerun by this extraction.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from shared_base_precision_cohort import validate_cohort

OUT = ROOT / "results/transformer/shared_base_long_context_2026-09-17"
BASE = ROOT / "output/shared_base_refinement_base_2026-09-17.json"
REFINEMENT = ROOT / "output/shared_base_refinement_completed_2026-09-17.json"
SOURCE_PLAN = ROOT / "code/transformer/configs/shared_base_posterior_plan_2026-09-16.json"
REFINE_PLAN = ROOT / "code/transformer/configs/shared_base_posterior_refinement_plan_2026-09-17.json"
COHORTS = ROOT / "output/shared_base_conditional_curves_2026-09-15"
LARGER = ROOT / "output/shared_base_scaling_evaluation_2026-09-17"
DEPTHS, TIMES, INDICES = (0, 1, 4, 8), (256, 1024), (0, 1)
MODELS = {
    "original_seed0": (COHORTS, "d64", ROOT / "results/transformer/runs/d64"),
    "original_seed1": (COHORTS, "d64_seed1", ROOT / "results/transformer/runs/d64_seed1"),
    "larger_seed0": (LARGER, "seed0", ROOT / "output/shared_base_scaling_download_2026-09-17/seed0"),
    "larger_seed1": (LARGER, "seed1", ROOT / "output/shared_base_scaling_download_2026-09-17/seed1"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if a is None or isinstance(a, (bool, str)):
        return type(a) is type(b) and a == b
    return isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-10, abs_tol=2e-12)


def law(value):
    p = np.asarray(value, dtype=float)
    require(p.shape == (64,) and np.isfinite(p).all() and np.all(p > 0)
            and abs(p.sum() - 1) < 1e-7, "Invalid 64-token probability vector")
    return p


def regret(oracle, prediction):
    # Independent true-row expectation; never use the one realized next token.
    return float(np.sum(oracle * np.log2(oracle / prediction)))


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), standard_error=float(a.std(ddof=1) / math.sqrt(len(a))))


def select_references(base, refinement, source_plan, refine_plan):
    """Overlay only predeclared continuations; repeats never replace estimates."""
    require(base["schema"] == "shared-base-depth-revalidation-v1" and base["complete"] is True,
            "Missing complete base revalidation")
    require(refinement["schema"] == "shared-base-posterior-refinement-summary-v1"
            and refinement["complete"] is True and refinement["all_numerical_checks_passed"] is True
            and refinement["missing_tasks"] == [] and refinement["expected_new_tasks"] == 9
            and refinement["original_reference_count"] == 32,
            "The complete fixed refinement batch has not passed")
    require([r["task"] for r in base["tasks"]] == source_plan["tasks"]
            and len(base["tasks"]) == 32, "Base task identities differ from frozen plan")
    tasks = refine_plan["tasks"]
    require(set(refinement["new_audits"]) == {t["id"] for t in tasks}
            and [r["task"] for r in refinement["tasks"]] == tasks
            and all(r["status"] == "ready" for r in refinement["tasks"]),
            "Refinement task scope/status changed")
    selected = {(r["task"]["depth"], r["task"]["index"], r["task"]["time"], r["task"]["order"]):
                deepcopy(r["corrected"]["audit"]) for r in base["tasks"]}
    repeat = None
    for task in tasks:
        row = refinement["new_audits"][task["id"]]
        require(row["task"] == task, "New audit differs from declared task")
        parent = source_plan["tasks"][task["base_task_index"]]
        require(all(parent[k] == task[k] for k in ("setting", "index", "time")), "Refinement changed source")
        a = row["corrected"]["audit"]
        require((a["index"], a["time"], a["order"]) == (task["index"], task["time"], task["order"]),
                "Refined audit case/order changed")
        require(row["base_row_id"] == base["tasks"][task["base_task_index"]]["row_id"],
                "New audit is not bound to original row")
        if task["kind"] == "repeat":
            require(repeat is None, "Unexpected repeated diagnostic")
            repeat = a
        else:
            require(task["kind"] in ("continuation", "quadrature"), "Unknown replacement kind")
            key = (parent["depth"], task["index"], task["time"], task["order"])
            require((key in selected) == (task["kind"] == "continuation"), "Wrong replacement role")
            selected[key] = a
    require(repeat is not None and refinement["repeat_check"]["passed"] is True,
            "Independent repeat has not passed")
    # Replay its agreement/precision decision without using it as a predictor.
    a = selected[0, 1, 1024, 32]["record"]
    b = repeat["record"]
    require(a["sampling_seed"] != b["sampling_seed"] and repeat["ready"] is True
            and repeat["status"] == "ready" and all(repeat["gates"].values())
            and all(b[k] is True for k in ("full_prior_valid", "provenance_valid", "mixing_passed", "row_grid_passed")),
            "Repeat is not independent/ready")
    p, q = a["precision"], b["precision"]
    gate = refine_plan["repeat_gate"]
    sigma = math.sqrt(p["mc_variance_bits2"] + q["mc_variance_bits2"])
    bias = gate["bias_multiplier"] * (p["jensen_proxy_bits"] + q["jensen_proxy_bits"])
    difference = abs(p["bayes_regret_bits"] - q["bayes_regret_bits"])
    allowance = gate["agreement_mc_multiplier"] * sigma + bias + gate["agreement_floor_bits"]
    numerical = gate["precision_mc_multiplier"] * sigma + bias
    require(difference <= allowance and numerical <= gate["precision_budget_bits"], "Repeat check no longer passes")
    for depth in DEPTHS:
        for time in TIMES:
            orders = (32, 64) if (depth, time) == (0, 1024) else (16, 32)
            for index in INDICES:
                for order in orders:
                    a = selected[depth, index, time, order]
                    require(a["ready"] is True and a["status"] == "ready" and all(a["gates"].values())
                            and a["recomputed_final_states"] is True and a["recomputed_finer_grid"] is True,
                            "A selected reference did not pass its raw audit")
    return selected


def build(base_path=BASE, refinement_path=REFINEMENT):
    receipt = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        digest = sha(path)
        require(expected is None or digest == expected, f"Input hash mismatch: {key}")
        require(key not in receipt or receipt[key] == digest, f"Input changed: {key}")
        receipt[key] = digest
        return digest

    def read(path, expected=None):
        bind(path, expected)
        def invalid(x):
            raise ValueError(f"Nonfinite JSON: {x}")
        return json.loads(Path(path).read_text(), parse_constant=invalid)

    refinement = read(refinement_path)
    base = read(base_path, refinement["base"]["summary_sha256"])
    require(base["launch_sha256"] == refinement["base"]["launch_sha256"], "Base launch hash differs")
    source_plan, refine_plan = read(SOURCE_PLAN), read(REFINE_PLAN)
    require(source_plan["conditions"] == dict(alpha=1., depths=list(DEPTHS), indices=list(INDICES),
            observed_tokens=list(TIMES), exponent_orders=[16, 32]), "Source pilot scope changed")
    selected = select_references(base, refinement, source_plan, refine_plan)
    # Bind all original source/old-model inputs to the exact cluster bundle.
    for relative, metadata in source_plan["inputs"]["files"].items():
        bind(COHORTS / relative, metadata["sha256"])
    model_metadata = {}
    for name, (_, _, run) in MODELS.items():
        model_metadata[name] = dict(checkpoint_sha256=bind(run / "model_100000.pt"),
            config_sha256=bind(run / "config.json"))
    records = []
    source_ids = {r["source_id"] for r in source_plan["source_sequences"]}
    require(len(source_ids) == 8, "Need eight independent source identities")
    for depth in DEPTHS:
        setting = f"a1_L{depth}"
        for index in INDICES:
            source_path = COHORTS / setting / "data" / f"seq_{index:06d}.json"
            source = read(source_path)
            planned = next(r for r in source_plan["source_sequences"] if r["setting"] == setting and r["index"] == index)
            tokens = np.asarray(source["tokens"], dtype=int)
            require(tokens.shape == (1025,) and source["source_id"] == planned["source_id"]
                    and fingerprint(source["identity"]) == source["source_id"]
                    and source["tokens_sha256"] == planned["tokens_sha256"] == array_sha(tokens, "<i2"),
                    "Source identity or token sequence differs")
            transition = np.asarray(source["oracle"]["transition_probs"], dtype=float)
            require(transition.shape == (64, 64) and array_sha(transition, "<f8") ==
                    source["oracle"]["transition_probs_sha256"], "Oracle transition matrix changed")
            model_rows = {}
            for name, (root, saved_name, _) in MODELS.items():
                path = root / setting / "models" / saved_name / f"seq_{index:06d}.json"
                report = read(path)
                ident = report["identity"]
                require(report["cache_id"] == fingerprint(ident) and report["source_id"] == source["source_id"]
                        and ident["source_file_sha256"] == sha(source_path)
                        and all(ident["model"][k] == v for k, v in model_metadata[name].items())
                        and report["checks"]["alignment"]["passed"] is True,
                        "Transformer/source/checkpoint identity mismatch")
                bind(path.with_suffix(".npz"), report["arrays_sha256"])
                with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                    for time in TIMES:
                        j = report["prefixes"].index(time)
                        oracle = law(transition[tokens[time - 1]])
                        prediction = law(data["predictive_probs"][j])
                        require(report["target_indices"][j] == time and report["targets"][j] == tokens[time]
                                and data["targets"][j] == tokens[time]
                                and report["prefix_tokens_sha256"][str(time)] == array_sha(tokens[:time], "<i2")
                                and np.array_equal(oracle, data["oracle_probs"][j])
                                and np.array_equal(prediction, report["predictive_probs"][j])
                                and np.max(np.abs(np.exp(data["logp"][j]) - prediction)) < 1e-15,
                                "Transformer probability/oracle/prefix readback failed")
                        model_rows[name, time] = dict(regret_bits=regret(oracle, prediction),
                            source_file_sha256=ident["source_file_sha256"], model_record_sha256=sha(path),
                            model_array_sha256=report["arrays_sha256"])
            for time in TIMES:
                order = 64 if (depth, time) == (0, 1024) else 32
                audited = selected[depth, index, time, order]
                precision = audited["record"]["precision"]
                oracle = law(transition[tokens[time - 1]])
                p = law(precision["predictive_probs"])
                r = regret(oracle, p)
                entropy = float(-np.sum(oracle * np.log2(oracle)))
                require(same(r, precision["bayes_regret_bits"])
                        and same(entropy, precision["oracle_entropy_bits"])
                        and same(entropy + r, precision["bayes_cross_entropy_bits"]),
                        "Bayesian regret fails independent true-row reconstruction")
                predictors = {}
                for name in MODELS:
                    predictors[name] = {**model_rows[name, time],
                        "excess_over_bayes_bits": model_rows[name, time]["regret_bits"] - r}
                records.append(dict(depth=depth, index=index, time=time, alpha=1.,
                    source_id=source["source_id"], tokens_sha256=source["tokens_sha256"],
                    reference_order=order, oracle_entropy_bits=entropy,
                    bayes=dict(regret_bits=r, mcse_bits=precision["mcse_bits"],
                        jensen_proxy_bits=precision["jensen_proxy_bits"],
                        numerical_allowance_bits=2 * precision["mcse_bits"] + 2 * precision["jensen_proxy_bits"],
                        predictive_probs=p.tolist(), report_sha256=audited["report_sha256"],
                        reference_id=audited["reference_id"], draws_per_chain=audited["draws_per_chain"]),
                    predictors=predictors))
    groups = []
    require(len(refinement["groups"]) == 8, "Wrong reported group count")
    for depth in DEPTHS:
        for time in TIMES:
            orders = (32, 64) if (depth, time) == (0, 1024) else (16, 32)
            replay = validate_cohort(*[[selected[depth, i, time, o]["record"] for i in INDICES] for o in orders], list(INDICES))
            saved_group = next(g for g in refinement["groups"] if (g["depth"], g["time"]) == (depth, time))
            saved = saved_group["numerical_audit"]
            require(saved_group["orders"] == list(orders) and saved["complete"] is True and saved["passed"] is True
                    and all(c["passed"] for c in saved["checks"]) and replay["passed"] is True,
                    "Cohort gate failed on independent replay")
            require(all(same(replay[k], saved[k]) for k in ("rules", "refinement", "reported_mean_bayes_regret_bits")),
                    "Saved cohort numbers differ from audited per-case records")
            rows = [r for r in records if (r["depth"], r["time"]) == (depth, time)]
            require(len(rows) == 2, "Incomplete matched source pair")
            b = stats([r["bayes"]["regret_bits"] for r in rows])
            require(same(b["mean"], saved["reported_mean_bayes_regret_bits"]), "Bayesian group mean differs")
            fine = replay["rules"][1]
            predictors = {}
            for name in MODELS:
                s = stats([r["predictors"][name]["regret_bits"] for r in rows])
                delta = stats([r["predictors"][name]["excess_over_bayes_bits"] for r in rows])
                predictors[name] = dict(mean_regret_bits=s["mean"], sequence_standard_error_bits=s["standard_error"],
                    mean_excess_over_bayes_bits=delta["mean"], paired_sequence_standard_error_bits=delta["standard_error"])
            groups.append(dict(depth=depth, time=time, n_sequences=2, orders=list(orders),
                bayes=dict(mean_regret_bits=b["mean"], sequence_standard_error_bits=b["standard_error"],
                    numerical_mcse_bits=fine["numerical_mcse_bits"],
                    numerical_allowance_bits=fine["numerical_error_allowance_bits"]), predictors=predictors))
    bind(Path(__file__))
    bind(ROOT / "code/scripts/shared_base_precision_cohort.py")
    for path, digest in receipt.items():
        require(sha(ROOT / path) == digest, "Input changed during extraction")
    return dict(schema="shared-base-long-context-results-v1", scientific_n=8, n_sequences=8,
        n_sequences_per_depth=2, n_scored_sequence_times=16, n_tokens_per_sequence=1025,
        depths=list(DEPTHS), indices=list(INDICES), times=list(TIMES), alpha=1.,
        reference_prior=source_plan["reference_prior"], source_ids=sorted(source_ids),
        models=model_metadata, records=records, groups=groups, all_numerical_checks_passed=True,
        independent_true_row_reconstruction_passed=True, source_and_model_identity_passed=True,
        cluster_raw_audit_reused=True, raw_chains_reaudited_locally=False,
        full_population_claim=False, input_sha256=receipt,
        interpretation="Fixed computational pilot: two predeclared independent sequences per depth, reused at both times. Numerical Monte Carlo uncertainty is conditional on reaching the posterior target; passing diagnostics does not prove convergence. Sequence standard errors describe variation across only two source sequences and do not establish population agreement. No new training, inference or sampling.")


def tsv(summary):
    stream = io.StringIO()
    columns = ["depth", "index", "time", "source_id", "reference_order", "bayes_regret_bits",
               "bayes_mcse_bits", "bayes_jensen_proxy_bits"]
    columns += [f"{name}_regret_bits" for name in MODELS]
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for r in summary["records"]:
        row = {k: r[k] for k in columns[:5]}
        row.update(bayes_regret_bits=r["bayes"]["regret_bits"], bayes_mcse_bits=r["bayes"]["mcse_bits"],
                   bayes_jensen_proxy_bits=r["bayes"]["jensen_proxy_bits"])
        row.update({f"{name}_regret_bits": r["predictors"][name]["regret_bits"] for name in MODELS})
        writer.writerow(row)
    return stream.getvalue()


README = """# Shared-base long-context numerical pilot

Eight predeclared sequences of length 1025: indices 0 and 1 at each depth 0, 1,
4 and 8, with exponent 1. Each predictor observes 256 or 1024 tokens of the same sequence.
Regret is the expected next-token log loss minus the true transition-row entropy,
in bits. Both original and larger transformers and both training runs are retained.

Bayes uses the full training prior (unknown exponent, depth, and label ranks).
The reported prediction uses exponent order 32, except depth 0 at time 1024 where
the predeclared refined order 64 is used. Declared chain continuations replace
their predecessors. The independent repeat only checks agreement and never
replaces the selected prediction. All eight depth/time numerical gates pass.

`summary.json` retains each case, matched group summaries, model/source identity,
Bayesian report hashes and all local input SHA256 receipts. `records.tsv` is a
compact rendering of the same sixteen comparisons. Bayesian MCSE describes
numerical sampling uncertainty; it is separate from variation across source
sequences. Two sequences per depth do not establish a population learning curve.
The cluster raw-chain audits are reused; local extraction independently checks
the frozen identities, saved probabilities, true-row regrets and cohort means.

Recreate from the transferred JSON summaries and existing local model outputs:

```bash
python code/scripts/shared_base_long_context_results.py
python code/scripts/shared_base_long_context_results.py --check-only
```

No posterior sampling, transformer inference or new experiment is performed.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--refinement", type=Path, default=REFINEMENT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    result = build(args.base, args.refinement)
    files = {"summary.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
             "records.tsv": tsv(result), "README.md": README}
    if args.check_only:
        for name, content in files.items():
            require((args.out / name).read_text() == content, f"Released output differs: {name}")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            temporary = args.out / (name + ".tmp")
            temporary.write_text(content)
            temporary.replace(args.out / name)
    print(json.dumps(dict(checked=args.check_only, n_sequences=8, scored_sequence_times=16,
                         groups=8, numerical_gates_passed=True, out=str(args.out))))


if __name__ == "__main__":
    main()
