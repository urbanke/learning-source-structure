"""Export complete, independently audited time16 shared-base comparisons.

This is a release gate, not an inference step. All four declared cohorts and
both exponent rules must pass together. An incomplete or stale audit leaves
any existing output untouched; no available-case mean is exported.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np

from shared_base_precision_cohort import validate_cohort

ROOT = Path(__file__).resolve().parents[2]
DEPTHS = [0, 1, 4, 8]
ORDERS = [16, 32]
MODELS = ["d64", "d64_seed1"]
TIME = 16
N = 256
PRIOR = dict(alpha=[.5, 1.5], depths=list(range(9)), ranks="uniform permutation")
SCHEMA = "shared-base-curve-paired-summary-v1"
AUDIT_SCHEMA = "shared-base-curve-audit-v1"
SCORING_SCRIPTS = ["shared_base_curve_reference.py", "shared_base_conditional_cohort.py",
                   "shared_base_precision_cohort.py"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def native_values(value):
    """Normalize NumPy scalar flags returned by the precision validator."""
    if isinstance(value, dict):
        return {key: native_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native_values(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value


def same(a, b):
    """Compare complete structures, allowing only floating-point roundoff."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if a is None or isinstance(a, (str, bool)):
        return type(a) is type(b) and a == b
    return (type(b) in (int, float) and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-11, abs_tol=2e-13))


def checked_moments(value):
    require(set(value) == {"n", "mean", "standard_error"} and value["n"] == N,
            "moment schema or cohort size differs")
    require(type(value["mean"]) in (int, float) and math.isfinite(value["mean"])
            and type(value["standard_error"]) in (int, float)
            and math.isfinite(value["standard_error"]) and value["standard_error"] >= 0,
            "invalid mean or standard error")
    return deepcopy(value)


def audit_one(path, depth, script_hashes):
    report = read_json(path)
    require(report["schema"] == AUDIT_SCHEMA, "audit schema mismatch")
    require((report["setting"], report["alpha"], report["depth"], report["n_sequences"],
             report["n_tokens_per_sequence"], report["times"], report["orders"]) ==
            (f"a1_L{depth}", 1., depth, N, 1025, [TIME], ORDERS),
            "audit source, cohort, time or quadrature differs")
    require(report["all_requested_times_validated"] is True, "audit has not passed")
    require(len(report["rows"]) == 1 and report["rows"][0]["time"] == TIME,
            "missing, duplicated or unexpected audit rows")
    provenance = report["provenance"]
    require(provenance["audit_sha256"] == script_hashes["shared_base_curve_audit.py"],
            "audit implementation changed")
    cohort_file = Path(provenance["cohort_file"])
    protocol_file = cohort_file.with_name("protocol.json")
    require(sha_file(cohort_file) == provenance["cohort_sha256"], "source cohort changed")
    require(sha_file(protocol_file) == provenance["protocol_sha256"], "source protocol changed")
    cohort, protocol = read_json(cohort_file), read_json(protocol_file)
    require(cohort["schema"] == protocol["schema"] == "shared-base-conditional-cohort-v1",
            "source schema mismatch")
    require(cohort["protocol_sha256"] == provenance["protocol_sha256"], "cohort protocol mismatch")
    for description in (cohort, protocol):
        require(description["alpha"] == 1. and description["depth"] == depth,
                "declared source parameters differ")
    require(cohort["n_sequences"] == N and cohort["n_tokens_per_sequence"] == 1025
            and [r["index"] for r in cohort["records"]] == list(range(N)),
            "incomplete source declaration")
    require(protocol["d"] == 64 and protocol["n_tokens"] == 1025
            and TIME in protocol["times"] and TIME in cohort["prefixes"]
            and protocol["reference_prior"] == PRIOR
            and protocol["conditions_hidden_from_predictors"] is True,
            "source dimensions, hidden conditions or full training prior differ")
    row = report["rows"][0]
    expected_counts = dict(missing=0, running=0, invalid=0, needs_more_draws=0, ready=2*N)
    require(row["status_counts"] == expected_counts, "not every reference is ready")
    cases = row["case_audits"]
    keys = [(c["index"], c["order"]) for c in cases]
    require(len(keys) == len(set(keys)) == 2*N
            and set(keys) == {(i, o) for i in range(N) for o in ORDERS},
            "incomplete or duplicate audited case/order")
    for case in cases:
        require(case["time"] == TIME and case["status"] == "ready" and case["ready"] is True
                and case["recomputed_final_states"] is True and case["recomputed_finer_grid"] is True
                and type(case["raw_checks"]) is int and case["raw_checks"] > 0,
                "case was not independently validated")
        require(case["gates"] and all(v is True for v in case["gates"].values()), "failed case gate")
        require(case["record"]["case_id"] == case["index"], "record index differs")
        reference_path = Path(case["report"])
        require(sha_file(reference_path) == case["report_sha256"], "reference changed after audit")
        reference = read_json(reference_path)
        require(reference["schema"] == "shared-base-curve-reference-v1"
                and reference["reference_id"] == case["reference_id"]
                and (reference["setting"], reference["index"], reference["time"], reference["order"])
                    == (report["setting"], case["index"], TIME, case["order"]),
                "audited reference identity differs")
        require(reference["identity"]["scoring_code"] ==
                {name: script_hashes[name] for name in SCORING_SCRIPTS}, "scoring implementation changed")
        require(reference["ready_for_cohort_check"] is True
                and same(reference["record"], case["record"]), "audited precision record differs")
    by_key = dict(zip(keys, cases))
    aggregate = native_values(validate_cohort(*[[by_key[i, o]["record"] for i in range(N)] for o in ORDERS], list(range(N))))
    aggregate["complete"] = True
    require(aggregate["passed"] is True, "recomputed cohort validation failed")
    require(same(aggregate, row["cohort_validation"]), "saved cohort validation differs from recomputation")
    require(set(row["matched_comparisons"]) == set(MODELS), "missing or unexpected transformer model")
    bayes = [by_key[i, ORDERS[-1]]["record"]["precision"]["bayes_regret_bits"] for i in range(N)]
    bayes_moments = dict(n=N, mean=statistics.mean(bayes), standard_error=statistics.stdev(bayes)/math.sqrt(N))
    rows = []
    for model in MODELS:
        matched = row["matched_comparisons"][model]
        require(set(matched) == {"bayes_regret", "transformer_regret", "transformer_minus_bayes"},
                "matched comparison schema mismatch")
        checked = {key: checked_moments(value) for key, value in matched.items()}
        require(same(checked["bayes_regret"], bayes_moments), "Bayesian mean differs from audited records")
        require(same(checked["transformer_regret"], row["full_cohort_transformer_only"][model]),
                "transformer cohort summaries differ")
        require(same(checked["transformer_minus_bayes"]["mean"],
                     checked["transformer_regret"]["mean"] - checked["bayes_regret"]["mean"]),
                "paired difference mean is inconsistent")
        rows.append(dict(alpha=1., depth=depth, time=TIME, model=model, n_sequences=N, **checked))
    receipt = dict(alpha=1., depth=depth, n_sequences=N,
        cohort_sha256=provenance["cohort_sha256"], protocol_sha256=provenance["protocol_sha256"],
        audit_path=str(Path(path).resolve()), audit_sha256=sha_file(path),
        raw_check_counts=dict(references=len(cases), total=sum(c["raw_checks"] for c in cases),
            minimum_per_reference=min(c["raw_checks"] for c in cases),
            maximum_per_reference=max(c["raw_checks"] for c in cases)))
    validation = dict(depth=depth, orders=ORDERS, cohort_validation=aggregate, status_counts=expected_counts)
    return rows, receipt, validation


def summarize(audits, out):
    scripts = SCORING_SCRIPTS + ["shared_base_curve_audit.py", "shared_base_curve_summary.py"]
    script_hashes = {name: sha_file(Path(__file__).with_name(name)) for name in scripts}
    rows, receipts, validations = [], [], []
    # Validate the complete release before opening or replacing its output.
    for depth in DEPTHS:
        new_rows, receipt, validation = audit_one(Path(audits)/f"a1_L{depth}_t16.json", depth, script_hashes)
        rows.extend(new_rows)
        receipts.append(receipt)
        validations.append(validation)
    result = dict(schema=SCHEMA, time=TIME, prior=PRIOR, rows=rows,
        reference="Full original training prior; selected alpha/depth are hidden. Refined 32-node exponent quadrature, checked against 16 nodes.",
        statistic="Mean KL(true next-token transition row || predictor), in bits; standard errors over independent whole sequences.",
        every_declared_sequence_retained=True, reference_validation_passed=True,
        validations=validations, provenance=dict(script_sha256=script_hashes, cohorts=receipts),
        numerical_uncertainty="Cohort Monte Carlo error and local Jensen bias diagnostics are reported separately from sampling standard errors; allowances are not rigorous bounds.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=out.parent, prefix=out.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, out)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, default=ROOT/"output/shared_base_curve_audits_2026-09-16")
    parser.add_argument("--out", type=Path, default=ROOT/"results/transformer/shared_base_conditional_curves/time16_summary.json")
    args = parser.parse_args()
    result = summarize(args.audits, args.out)
    print(json.dumps(dict(out=str(args.out), rows=result["rows"], validated=True)))


if __name__ == "__main__":
    main()
