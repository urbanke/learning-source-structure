"""The release exporter must not turn partial, stale or failed audits into data."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import shared_base_curve_summary as summary
from shared_base_precision_cohort import validate_cohort


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def script_hashes():
    names = summary.SCORING_SCRIPTS + ["shared_base_curve_audit.py", "shared_base_curve_summary.py"]
    return {name: summary.sha_file(Path(summary.__file__).with_name(name)) for name in names}


def fixture_audit(folder, depth=0):
    hashes = script_hashes()
    source = folder/f"source/a1_L{depth}"
    protocol = dict(schema="shared-base-conditional-cohort-v1", alpha=1., depth=depth,
        d=64, n_tokens=1025, times=[16], reference_prior=summary.PRIOR,
        conditions_hidden_from_predictors=True)
    write(source/"protocol.json", protocol)
    cohort = dict(schema="shared-base-conditional-cohort-v1", alpha=1., depth=depth,
        n_sequences=256, n_tokens_per_sequence=1025, prefixes=[16],
        protocol_sha256=summary.sha_file(source/"protocol.json"), records=[dict(index=i) for i in range(256)])
    write(source/"cohort.json", cohort)
    cases = []
    for index in range(256):
        for order in [16, 32]:
            precision = dict(bayes_regret_bits=.5+(index % 2), mc_variance_bits2=.01**2,
                jensen_proxy_bits=.0002, deterministic=False,
                scalar_mixing_passed=True, jensen_local_check_passed=True)
            record = dict(case_id=index, sampling_seed=index+10000*order, precision=precision,
                full_prior_valid=True, provenance_valid=True, mixing_passed=True, row_grid_passed=True,
                per_sequence_precision_passed=False)
            ref = dict(schema="shared-base-curve-reference-v1", setting=f"a1_L{depth}",
                index=index, time=16, order=order, reference_id=f"{depth}-{index}-{order}",
                identity=dict(scoring_code={name: hashes[name] for name in summary.SCORING_SCRIPTS}),
                ready_for_cohort_check=True, record=record)
            path = folder/f"references/L{depth}_{index}_{order}.json"
            write(path, ref)
            cases.append(dict(index=index, order=order, time=16, status="ready", ready=True,
                recomputed_final_states=True, recomputed_finer_grid=True, raw_checks=69,
                gates=dict(mixing=True), record=record, report=str(path),
                report_sha256=summary.sha_file(path), reference_id=ref["reference_id"]))
    aggregate = summary.native_values(validate_cohort([c["record"] for c in cases if c["order"] == 16],
                               [c["record"] for c in cases if c["order"] == 32], list(range(256))))
    assert aggregate["passed"]
    aggregate["complete"] = True
    b = dict(n=256, mean=1., standard_error=aggregate["rules"][1]["observed_sequence_se_bits"])
    matched = {m: dict(bayes_regret=b, transformer_regret=dict(b, mean=1.02),
                       transformer_minus_bayes=dict(n=256, mean=.02, standard_error=.002)) for m in summary.MODELS}
    report = dict(schema=summary.AUDIT_SCHEMA, setting=f"a1_L{depth}", alpha=1., depth=depth,
        n_sequences=256, n_tokens_per_sequence=1025, times=[16], orders=[16, 32],
        all_requested_times_validated=True,
        provenance=dict(cohort_file=str(source/"cohort.json"), cohort_sha256=summary.sha_file(source/"cohort.json"),
            protocol_sha256=summary.sha_file(source/"protocol.json"), audit_sha256=hashes["shared_base_curve_audit.py"]),
        rows=[dict(time=16, cohort_validation=aggregate, matched_comparisons=matched,
            full_cohort_transformer_only={m: v["transformer_regret"] for m, v in matched.items()},
            case_audits=cases, status_counts=dict(missing=0, running=0, invalid=0, needs_more_draws=0, ready=512))])
    path = folder/f"audits/a1_L{depth}_t16.json"
    write(path, report)
    return path, report


def test_success_retains_all_conditions_and_individual_precision_flags(tmp_path):
    for depth in summary.DEPTHS:
        fixture_audit(tmp_path, depth)
    out = tmp_path/"result/time16_summary.json"
    result = summary.summarize(tmp_path/"audits", out)
    assert json.loads(out.read_text()) == result
    assert len(result["rows"]) == 8
    assert result["prior"] == summary.PRIOR
    assert result["every_declared_sequence_retained"] is True
    assert result["reference_validation_passed"] is True
    assert all(r["n_sequences"] == 256 for r in result["rows"])
    assert all(c["raw_check_counts"]["total"] == 512*69 for c in result["provenance"]["cohorts"])
    # Whole-cohort precision is a separate decision; old false flags are kept.
    assert all(not flag for v in result["validations"] for rule in v["cohort_validation"]["rules"]
               for flag in rule["per_sequence_precision_status"].values())
    assert "case_audits" not in out.read_text()
    assert "predictive_probs" not in out.read_text()


@pytest.mark.parametrize("change", ["schema", "failed", "missing_case", "duplicate_case", "wrong_order",
    "partial_status", "false_complete", "bad_moment_size", "inconsistent_mean", "tampered_budget"])
def test_malformed_or_incomplete_audit_never_replaces_output(tmp_path, change):
    path, report = fixture_audit(tmp_path)
    row = report["rows"][0]
    if change == "schema":
        report["schema"] = "wrong-v1"
    elif change == "failed":
        report["all_requested_times_validated"] = False
    elif change == "missing_case":
        row["case_audits"].pop()
    elif change == "duplicate_case":
        row["case_audits"][-1] = deepcopy(row["case_audits"][0])
    elif change == "wrong_order":
        report["orders"] = [8, 16]
    elif change == "partial_status":
        row["status_counts"]["ready"] = 511
        row["status_counts"]["needs_more_draws"] = 1
    elif change == "false_complete":
        row["cohort_validation"]["complete"] = False
    elif change == "bad_moment_size":
        row["matched_comparisons"]["d64"]["transformer_minus_bayes"]["n"] = 255
    elif change == "inconsistent_mean":
        row["matched_comparisons"]["d64"]["transformer_minus_bayes"]["mean"] = 1.
    elif change == "tampered_budget":
        row["cohort_validation"]["rules"][0]["numerical_budget_bits"] = 1.
    write(path, report)
    out = tmp_path/"time16_summary.json"
    out.write_text("previous valid release\n")
    with pytest.raises(ValueError):
        summary.summarize(tmp_path/"audits", out)
    assert out.read_text() == "previous valid release\n"


@pytest.mark.parametrize("artifact", ["cohort", "protocol", "reference", "audit_code", "scoring_code"])
def test_stale_or_changed_inputs_are_rejected(tmp_path, artifact):
    path, report = fixture_audit(tmp_path)
    hashes = script_hashes()
    if artifact in ("cohort", "protocol"):
        source = Path(report["provenance"]["cohort_file"])
        changed = source if artifact == "cohort" else source.with_name("protocol.json")
        changed.write_text(changed.read_text()+"\n")
    elif artifact == "reference":
        changed = Path(report["rows"][0]["case_audits"][0]["report"])
        changed.write_text(changed.read_text()+"\n")
    elif artifact == "audit_code":
        hashes["shared_base_curve_audit.py"] = "changed"
    elif artifact == "scoring_code":
        hashes["shared_base_curve_reference.py"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        summary.audit_one(path, 0, hashes)


def test_missing_later_depth_cannot_produce_partial_output(tmp_path):
    fixture_audit(tmp_path, 0)
    out = tmp_path/"result.json"
    with pytest.raises(FileNotFoundError):
        summary.summarize(tmp_path/"audits", out)
    assert not out.exists()


def test_failed_raw_record_cannot_be_hidden_behind_successful_aggregate(tmp_path):
    path, report = fixture_audit(tmp_path)
    case = report["rows"][0]["case_audits"][0]
    case["record"]["mixing_passed"] = False
    reference_path = Path(case["report"])
    reference = json.loads(reference_path.read_text())
    reference["record"] = case["record"]
    write(reference_path, reference)
    case["report_sha256"] = summary.sha_file(reference_path)
    write(path, report)
    with pytest.raises(ValueError, match="recomputed cohort validation failed"):
        summary.audit_one(path, 0, script_hashes())
