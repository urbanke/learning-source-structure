"""Independent score/coverage checks for the transferred full Bayes snapshot.

Synthetic laws isolate the release logic from posterior sampling.  The tests
retain old individual precision failures because the frozen cohort protocol
deliberately judges accuracy of the complete mean separately.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import shared_base_full_cohort_results as results
from shared_base_precision_cohort import validate_cohort


def make_audit(index, order, oracle, prediction, time=256):
    positive = oracle > 0
    # Scalar log2 expectations, independent of the exporter's implementation.
    entropy = -sum(float(p)*np.log2(float(p)) for p in oracle[positive])
    cross_entropy = -sum(float(p)*np.log2(float(q)) for p, q in zip(oracle, prediction))
    precision = dict(predictive_probs=prediction.tolist(),
        oracle_entropy_bits=entropy, bayes_cross_entropy_bits=cross_entropy,
        bayes_regret_bits=cross_entropy-entropy, mc_variance_bits2=1e-12,
        mcse_bits=1e-6, variance_by_estimator_bits2={"between": 1e-12, "batch_64": 5e-13},
        jensen_curvature_scale_bits=1e-10, chain_jackknife_bias_bits=8e-11,
        jensen_proxy_bits=1e-10, max_relative_probability_mcse=.001,
        scalar_split_rhat=1.001, scalar_ess=900., scalar_mixing_passed=True,
        jensen_local_check_passed=True, chains=4, draws_per_chain=4096,
        deterministic=False)
    record = dict(case_id=index, sampling_seed=1000*order+index, precision=precision,
        full_prior_valid=True, provenance_valid=True, mixing_passed=True,
        row_grid_passed=True, per_sequence_precision_passed=False)
    return dict(index=index, time=time, order=order, status="ready", ready=True,
        record=record, raw_checks=69, report=f"/scratch/references/{index}-{order}.json",
        report_sha256="a"*64, reference_id="b"*64, draws_per_chain=4096,
        scalar_ess=900., minimum_probability_ess=800.,
        gates=dict(split_rhat_below_1_05=True, probability_ess_at_least_200=True,
                   maximum_probability_mcse_at_most_0_005=True, raw_normalization=True),
        recomputed_final_states=True, recomputed_finer_grid=True)


@pytest.fixture
def cohort():
    ids = [0, 1, 2, 3]
    orders = [16, 32]
    oracles = []
    audits = []
    for index in ids:
        oracle = np.arange(1, 65, dtype=float)
        oracle[::2] = 0.
        oracle /= oracle.sum()
        oracles.append(oracle)
        prediction = (1+index)*np.ones(64) + np.arange(64)
        prediction /= prediction.sum()
        for order in orders:
            audits.append(make_audit(index, order, oracle, prediction))
    lookup = {(a["index"], a["order"]): a for a in audits}
    aggregate = validate_cohort(*[[lookup[i, o]["record"] for i in ids] for o in orders], ids)
    aggregate["complete"] = True
    assert aggregate["passed"] is True
    return audits, np.asarray(oracles), orders, aggregate, ids


def run_group(cohort):
    audits, oracle, orders, saved, ids = cohort
    return results.validate_group(audits, oracle, 256, orders, saved, expected_ids=ids)


def test_recompute_true_row_with_zero_support(cohort):
    audits, oracle, _, _, _ = cohort
    source = audits[0]
    fixed, risk, checks = results.recompute_audit(source, oracle[0], 0, 256, 16,
                                                expected_seed=16000)
    assert risk == pytest.approx(source["record"]["precision"]["bayes_regret_bits"], abs=2e-13)
    assert fixed["record"]["per_sequence_precision_passed"] is False
    assert isinstance(checks, dict)


def test_full_paired_cohort_and_individual_precision_flags(cohort):
    aggregate, regrets, checks = run_group(cohort)
    assert aggregate["passed"] and aggregate["complete"]
    assert aggregate["n_sequences"] == 4
    assert aggregate["reported_mean_bayes_regret_bits"] == pytest.approx(
        cohort[3]["reported_mean_bayes_regret_bits"], abs=2e-13)
    assert isinstance(checks, dict)


@pytest.mark.parametrize("field", ["bayes_regret_bits", "oracle_entropy_bits", "bayes_cross_entropy_bits"])
def test_rejects_changed_saved_scores(cohort, field):
    audit = deepcopy(cohort[0][0])
    audit["record"]["precision"][field] += .01
    with pytest.raises(ValueError):
        results.recompute_audit(audit, cohort[1][0], 0, 256, 16)


@pytest.mark.parametrize("field,value", [
    ("mcse_bits", .001), ("mc_variance_bits2", 1e-9), ("jensen_proxy_bits", 1e-8),
    ("scalar_mixing_passed", False), ("jensen_local_check_passed", False),
])
def test_rejects_inconsistent_or_failed_precision(cohort, field, value):
    audit = deepcopy(cohort[0][0])
    audit["record"]["precision"][field] = value
    with pytest.raises(ValueError):
        results.recompute_audit(audit, cohort[1][0], 0, 256, 16)


def test_rejects_bad_normalization(cohort):
    audit = deepcopy(cohort[0][0])
    audit["record"]["precision"]["predictive_probs"] = (
        np.asarray(audit["record"]["precision"]["predictive_probs"])*1.001).tolist()
    with pytest.raises(ValueError):
        results.recompute_audit(audit, cohort[1][0], 0, 256, 16)


def test_preserves_small_raw_normalization_error(cohort):
    # The frozen Bayes contract is 1e-7, and actual accepted laws differ from
    # unity by up to 1e-11. They must not be silently renormalized on import.
    original = cohort[0][0]
    prediction = np.asarray(original["record"]["precision"]["predictive_probs"])*(1+2e-11)
    audit = make_audit(0, 16, cohort[1][0], prediction)
    fixed, _, _ = results.recompute_audit(audit, cohort[1][0], 0, 256, 16)
    np.testing.assert_array_equal(fixed["record"]["precision"]["predictive_probs"], prediction)


def test_rejects_wrong_seed(cohort):
    with pytest.raises(ValueError):
        results.recompute_audit(cohort[0][0], cohort[1][0], 0, 256, 16, expected_seed=123)


@pytest.mark.parametrize("field,value", [("index", 1), ("time", 1024), ("order", 64)])
def test_rejects_case_coordinate_mismatch(cohort, field, value):
    audit = deepcopy(cohort[0][0])
    audit[field] = value
    with pytest.raises(ValueError):
        results.recompute_audit(audit, cohort[1][0], 0, 256, 16)


def test_rejects_missing_sequence_order_instead_of_available_case_mean(cohort):
    cohort[0].pop()
    with pytest.raises(ValueError):
        run_group(cohort)


def test_rejects_duplicate_sequence_order(cohort):
    cohort[0].append(deepcopy(cohort[0][0]))
    with pytest.raises(ValueError):
        run_group(cohort)


def test_rejects_shared_sampling_seed(cohort):
    cohort[0][1]["record"]["sampling_seed"] = cohort[0][0]["record"]["sampling_seed"]
    with pytest.raises(ValueError):
        run_group(cohort)


def test_rejects_changed_aggregate_report(cohort):
    cohort[3]["reported_mean_bayes_regret_bits"] += .01
    with pytest.raises(ValueError):
        run_group(cohort)


def test_rejects_false_standard_error(cohort):
    cohort[3]["rules"][1]["observed_sequence_se_bits"] *= .5
    with pytest.raises(ValueError):
        run_group(cohort)


def test_rejects_incomplete_report_even_when_all_rows_exist(cohort):
    cohort[3]["complete"] = False
    with pytest.raises(ValueError):
        run_group(cohort)


def test_rejects_cohort_precision_budget_failure(cohort):
    audits, _, orders, saved, ids = cohort
    for audit in audits:
        p = audit["record"]["precision"]
        p["mc_variance_bits2"] = 1.
        p["mcse_bits"] = 1.
        p["variance_by_estimator_bits2"] = {"between": 1.}
    lookup = {(a["index"], a["order"]): a for a in audits}
    failed = validate_cohort(*[[lookup[i, o]["record"] for i in ids] for o in orders], ids)
    failed["complete"] = True
    assert failed["passed"] is False
    saved.clear()
    saved.update(failed)
    with pytest.raises(ValueError):
        run_group(cohort)
