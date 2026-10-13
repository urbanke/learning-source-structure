"""Completeness and publication barriers for the new curve reference audit."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from shared_base_curve_audit import aggregate_time, same, audit_reference


def complete_audits(n=256):
    rows=[]
    for index in range(n):
        for order in (16,32):
            precision=dict(bayes_regret_bits=.5+(index%2),mc_variance_bits2=.0145**2,
                jensen_proxy_bits=.0005,deterministic=False,
                scalar_mixing_passed=True,jensen_local_check_passed=True)
            record=dict(case_id=index,sampling_seed=index+10000*order,precision=precision,
                full_prior_valid=True,provenance_valid=True,mixing_passed=True,row_grid_passed=True,
                per_sequence_precision_passed=False)
            rows.append(dict(index=index,order=order,status="ready",record=record))
    return rows


@pytest.mark.parametrize("status",["missing","running","invalid"])
def test_one_missing_or_invalid_case_blocks_all_bayes_means(status):
    rows=complete_audits()
    rows[51]=dict(index=25,order=32,status=status)
    output=aggregate_time(rows,list(range(256)),[16,32])
    assert not output["complete"] and not output["passed"]
    assert output["reported_mean_bayes_regret_bits"] is None
    assert "candidate_mean_bayes_regret_bits" not in output
    assert output["missing_or_invalid"] == [dict(index=25,order=32)]


def test_completely_absent_case_is_not_silently_removed():
    output=aggregate_time(complete_audits()[:-2],list(range(256)),[16,32])
    assert output["missing_or_invalid"] == [dict(index=255,order=16),dict(index=255,order=32)]
    assert output["reported_mean_bayes_regret_bits"] is None


def test_mixing_failure_retains_complete_cohort_but_blocks_publication():
    rows=complete_audits()
    rows[0]["record"]["mixing_passed"]=False
    rows[0]["status"]="needs_more_draws"
    output=aggregate_time(rows,list(range(256)),[16,32])
    assert output["complete"] and not output["passed"]
    assert output["n_sequences"]==256
    assert output["reported_mean_bayes_regret_bits"] is None


def test_complete_cohort_can_pass_without_relabeling_individual_precision():
    output=aggregate_time(complete_audits(),list(range(256)),[16,32])
    assert output["passed"] and output["complete"]
    assert output["reported_mean_bayes_regret_bits"]==1.
    assert all(not flag for rule in output["rules"] for flag in rule["per_sequence_precision_status"].values())


def test_duplicate_or_unrequested_case_is_rejected():
    rows=complete_audits()
    with pytest.raises(ValueError,match="duplicate"):
        aggregate_time(rows+[rows[0]],list(range(256)),[16,32])
    extra=deepcopy(rows[0]);extra["index"]=256
    with pytest.raises(ValueError,match="undeclared"):
        aggregate_time(rows+[extra],list(range(256)),[16,32])


def test_recursive_readback_does_not_ignore_fields_flags_or_dimensions():
    expected=dict(a=[1.,2.],b=True,c=None,d="curve")
    assert same(dict(a=np.array([1.,2.+1e-13]),b=True,c=None,d="curve"),expected)
    assert not same(dict(expected,extra=1),expected)
    assert not same(dict(expected,b=1),expected)
    assert not same(dict(expected,a=[1.]),expected)
    assert not same(dict(expected,a=[1.,2.1]),expected)


def test_running_worker_is_reported_even_before_first_json(tmp_path):
    import fcntl
    folder=tmp_path/"refs"/"a1_L4"/"alpha16"
    folder.mkdir(parents=True)
    path=folder/"seq_000001_t0016.lock"
    with path.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=audit_reference(tmp_path/"a1_L4",tmp_path/"refs",1,16,16)
        assert result["status"]=="running" and not result["ready"]
    result=audit_reference(tmp_path/"a1_L4",tmp_path/"refs",1,16,16)
    assert result["status"]=="missing"
