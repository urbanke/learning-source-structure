"""Independent statistical checks for the frozen pooled-population experiment."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
sys.path.insert(0, str(ROOT / "code/transformer"))
import pooled_predictive_population as population


def record(z, p):
    p = np.asarray(p, dtype=float)
    return dict(log_evidence=math.log(z), predictive_probs=p.tolist(),
                log_joint_next=(math.log(z) + np.log(p)).tolist())


def law(tilt=0.):
    p = np.ones(64) / 64
    p[0] += tilt
    p[1:] -= tilt / 63
    return p


def sample_records(n=64):
    phase = np.linspace(-1, 1, n)
    return [record(.02 * (1 + .12 * x), law(.003 * x)) for x in phase]


def fixed_components():
    return [record(.01, law(.01)), record(.03, law(-.005))]


def independent_risk(z, joint, fixed, oracle):
    fz = np.exp([r["log_evidence"] for r in fixed])
    fp = np.array([r["predictive_probs"] for r in fixed])
    p = (fz @ fp + joint) / (fz.sum() + z)
    return float(np.dot(oracle, np.log2(oracle / p)))


def test_actual_mixture_uncertainty_matches_independent_directional_derivative():
    records, fixed, oracle = sample_records(), fixed_components(), law(.006)
    answer = population.mixture_uncertainty(records, fixed, oracle)
    z = np.exp([r["log_evidence"] for r in records])
    joint = z[:, None] * np.array([r["predictive_probs"] for r in records])
    mz, mj = z.mean(), joint.mean(axis=0)
    eps = 1e-5
    derivative = []
    for zi, ji in zip(z, joint):
        plus = independent_risk(mz + eps * (zi - mz), mj + eps * (ji - mj), fixed, oracle)
        minus = independent_risk(mz - eps * (zi - mz), mj - eps * (ji - mj), fixed, oracle)
        derivative.append((plus - minus) / (2 * eps))
    expected = np.std(derivative, ddof=1) / math.sqrt(len(records))
    assert answer["mcse_bits"] == pytest.approx(expected, rel=2e-7, abs=1e-12)


def test_curvature_and_jackknife_recomputed_from_probability_space():
    records, fixed, oracle = sample_records(), fixed_components(), law(.006)
    answer = population.mixture_uncertainty(records, fixed, oracle)
    z = np.exp([r["log_evidence"] for r in records])
    joint = z[:, None] * np.array([r["predictive_probs"] for r in records])
    fz = np.exp([r["log_evidence"] for r in fixed])
    fp = np.array([r["predictive_probs"] for r in fixed])
    a, d = fz @ fp + joint.mean(0), fz.sum() + z.mean()
    n = len(z)
    expected_curvature = (oracle @ (joint.var(0, ddof=1) / n / a**2)
                          + z.var(ddof=1) / n / d**2) / (2 * math.log(2))
    leave = [independent_risk(np.delete(z, i).mean(),
                             np.delete(joint, i, axis=0).mean(0), fixed, oracle)
             for i in range(n)]
    expected_jack = abs((n - 1) * (np.mean(leave) - independent_risk(z.mean(), joint.mean(0), fixed, oracle)))
    assert answer["curvature_bias_bits"] == pytest.approx(expected_curvature, rel=1e-12)
    assert answer["jackknife_bias_bits"] == pytest.approx(expected_jack, abs=2e-13)
    assert answer["bias_allowance_bits"] == max(answer["curvature_bias_bits"], answer["jackknife_bias_bits"])
    assert answer["passed"]


def test_common_predictive_law_cancels_evidence_noise_in_actual_mixture():
    p = law(.01)
    records = [record(z, p) for z in np.linspace(.01, .03, 64)]
    fixed = [record(.005, p), record(.04, p)]
    result = population.mixture_uncertainty(records, fixed, law())
    assert result["mcse_bits"] < 1e-15
    assert result["jackknife_bias_bits"] < 1e-12
    # The conservative absolute curvature bound need not share this cancellation.
    assert result["curvature_bias_bits"] > 0


def test_probability_space_scaling_does_not_change_uncertainty():
    records, fixed = sample_records(), fixed_components()
    a = population.mixture_uncertainty(records, fixed, law(.006))
    shifted = deepcopy(records + fixed)
    for r in shifted:
        r["log_evidence"] -= 1000
        r["log_joint_next"] = (np.asarray(r["log_joint_next"]) - 1000).tolist()
    b = population.mixture_uncertainty(shifted[:-2], shifted[-2:], law(.006))
    for key in ("mcse_bits", "curvature_bias_bits", "relative_mcse_max"):
        assert a[key] == pytest.approx(b[key], rel=1e-10, abs=1e-15)
    assert a["jackknife_bias_bits"] == pytest.approx(b["jackknife_bias_bits"], abs=3e-13)


def test_dominating_replica_fails_ratio_precision_guard():
    records = [record(1e-9, law()) for _ in range(63)] + [record(.1, law(.2))]
    fixed = [record(1e-10, law()), record(1e-10, law())]
    result = population.mixture_uncertainty(records, fixed, law(.05))
    assert result["relative_mcse_max"] > .1
    assert not result["passed"]


def test_refinement_checks_absolute_evidence_even_when_predictions_identical():
    left = [record(.01, law()) for _ in range(64)]
    right = [record(.02, law()) for _ in range(64)]
    result = population.refinement_pair(left, right)
    assert result["evidence_difference_bits"] == pytest.approx(1)
    assert max(result["predictive_allowance_bits"]) < 1e-12
    assert not result["passed"]


def test_refinement_checks_every_successor_without_oracle_weighting():
    left = [record(.01, law()) for _ in range(64)]
    right = [record(.01, law(.001)) for _ in range(64)]
    result = population.refinement_pair(left, right)
    assert result["evidence_allowance_bits"] < 1e-12
    assert result["predictive_allowance_bits"][0] > .01
    assert not result["passed"]


def test_refinement_includes_independent_sampling_uncertainty():
    # Equal estimates do not pass if their independent error bars are too wide.
    records = [record(z, law()) for z in np.linspace(.01, .03, 64)]
    result = population.refinement_pair(records, records)
    assert result["evidence_difference_bits"] == pytest.approx(0)
    assert result["evidence_allowance_bits"] > .01
    assert not result["passed"]


def test_identical_precise_refinement_passes():
    records = [record(.01, law()) for _ in range(64)]
    assert population.refinement_pair(records, records)["passed"]


@pytest.mark.parametrize("bad", [np.r_[.2, np.full(63, .01)], np.r_[0., np.full(63, 1/63)],
                                  np.r_[np.nan, np.full(63, 1/64)]])
def test_invalid_raw_prediction_is_not_silently_normalized(bad):
    records = sample_records()
    records[0]["predictive_probs"] = bad.tolist()
    with pytest.raises(ValueError):
        population.mixture_uncertainty(records, fixed_components(), law())


def paired_rows(*, spread=.001, numerical=1e-6, bias=1e-7):
    rows = []
    for f, family in enumerate(population.FAMILIES):
        for index in range(5, 69):
            x = index - 36.5
            bayes = .1 * (f + 1) + spread * x
            rows.append(dict(family=family, index=index, bayes_regret_bits=bayes,
                transformer_regret_bits=dict(pooled_seed0=bayes + .01 + .0002*x,
                                              pooled_seed1=bayes + .02 - .0003*x),
                mcse_bits=numerical, bias_allowance_bits=bias))
    return rows


def test_balanced_strata_and_paired_errors_use_sources_not_training_seeds():
    rows = paired_rows()
    result = population.balanced_summary(list(reversed(rows)))
    x = np.arange(64) - 31.5
    expected_se = (.001*x).std(ddof=1) / math.sqrt(192)
    assert result["n_total"] == 192 and result["n_per_family"] == 64
    assert result["bayes"]["mean"] == pytest.approx(.2)
    assert result["bayes"]["se"] == pytest.approx(expected_se)
    np.testing.assert_allclose(result["bayes"]["family_means"], [.1, .2, .3], atol=1e-14)
    for name, mean, scale in (("pooled_seed0", .01, .0002), ("pooled_seed1", .02, .0003)):
        assert result["paired"][name]["mean"] == pytest.approx(mean)
        assert result["paired"][name]["se"] == pytest.approx((scale*x).std(ddof=1)/math.sqrt(192))
    assert result["precision_passed"]


def test_between_family_mean_differences_do_not_inflate_stratified_se():
    result = population.balanced_summary(paired_rows(spread=0, numerical=0, bias=0))
    assert result["bayes"]["se"] < 1e-16
    assert result["bayes"]["mean"] == pytest.approx(.2)


def test_numerical_mcse_and_bias_are_separate_from_source_se():
    rows = paired_rows()
    result = population.balanced_summary(rows)
    numerical = 1e-6/math.sqrt(192)
    assert result["numerical_mcse_bits"] == pytest.approx(numerical)
    assert result["bias_allowance_bits"] == pytest.approx(1e-7)
    assert result["numerical_allowance_bits"] == pytest.approx(2*numerical+2e-7)
    assert result["denoised_source_se_bits"] == pytest.approx(math.sqrt(result["bayes"]["se"]**2-numerical**2))
    assert result["numerical_budget_bits"] == pytest.approx(min(.01/3, .25*result["denoised_source_se_bits"]))


def test_denoising_clips_each_fixed_stratum_before_combining():
    rows = paired_rows()
    for r in rows:
        if r["family"] == population.FAMILIES[0]:
            r["bayes_regret_bits"] = .1
            r["mcse_bits"] = .03
        else:
            r["mcse_bits"] = 0.
    result = population.balanced_summary(rows)
    within = (.001*(np.arange(64)-31.5)).var(ddof=1)
    expected = math.sqrt(2*within/64)/3
    assert result["denoised_source_se_bits"] == pytest.approx(expected)


def test_cohort_precision_failure_is_not_erased_by_source_variation():
    result = population.balanced_summary(paired_rows(numerical=.01, bias=.002))
    assert result["numerical_allowance_bits"] > result["numerical_budget_bits"]
    assert not result["precision_passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "outside", "wrong_family", "missing_seed", "nonfinite", "negative_error"])
def test_population_summary_rejects_incomplete_or_unpaired_coverage(mutation):
    rows = paired_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "outside":
        rows[0]["index"] = 0
    elif mutation == "wrong_family":
        rows[0]["family"] = "other"
    elif mutation == "missing_seed":
        del rows[0]["transformer_regret_bits"]["pooled_seed1"]
    elif mutation == "nonfinite":
        rows[0]["bayes_regret_bits"] = float("nan")
    elif mutation == "negative_error":
        rows[0]["mcse_bits"] = -.1
    with pytest.raises(ValueError):
        population.balanced_summary(rows)


def context_gates(tmp_path):
    for stage in ("refinement_timing", "refine"):
        contexts = {str(t): dict(complete=False, passed=False, task_sha256={})
                    for t in population.TIMES}
        contexts["16"].update(complete=True, passed=True,
            resource_projection=dict(within_budget=True),
            all_component_and_mixture_refinements_passed=True)
        folder = tmp_path / stage
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps(dict(plan_sha256="frozen",
            driver_sha256=population.sha(population.__file__), by_context=contexts)))


def test_one_validated_context_does_not_wait_for_unresolved_longest_context(tmp_path, monkeypatch):
    context_gates(tmp_path)
    monkeypatch.setattr(population, "calibration_decision", lambda *a, **k: dict(exact_short_passed=True))
    result = population.scientific_gate({}, "frozen", tmp_path, "population", tmp_path, 16)
    assert result["calibration"]["exact_short_passed"]
    with pytest.raises(ValueError, match="context has not passed"):
        population.scientific_gate({}, "frozen", tmp_path, "population", tmp_path, 1024)


def test_resumable_context_binding_survives_unrelated_context_progress(tmp_path, monkeypatch):
    context_gates(tmp_path)
    monkeypatch.setattr(population, "calibration_decision", lambda *a, **k: dict(exact_short_passed=True))
    before = population.scientific_gate({}, "frozen", tmp_path, "population", tmp_path, 16)
    for stage in ("refinement_timing", "refine"):
        path = tmp_path / stage / "summary.json"
        report = json.loads(path.read_text())
        report["by_context"]["1024"]["progress_note"] = "Another context has made progress"
        path.write_text(json.dumps(report))
    after = population.scientific_gate({}, "frozen", tmp_path, "population", tmp_path, 16)
    assert after == before


def timing_reports(observed=16, seconds=1.):
    return {i: dict(complete=True, task=dict(observed=observed, family=f, arm=a),
                    records=[dict(elapsed_seconds=seconds)])
            for i, (f, a) in enumerate((f, a) for f in population.FAMILIES for a in ("B", "C", "D"))}


def test_context_resource_bound_uses_all_three_families_and_all_rules():
    plan = dict(resource_limits=dict(timing_safety_factor=2.))
    reports = timing_reports()
    result = population.resource_projection(plan, reports, 16)
    assert result["within_budget"] and result["maximum_cpu_hours"] == 1024
    reports.pop(8)
    assert not population.resource_projection(plan, reports, 16)["within_budget"]
    assert not population.resource_projection(plan, reports, 1024)["within_budget"]


def test_context_resource_bound_refuses_costly_calculation():
    plan = dict(resource_limits=dict(timing_safety_factor=2.))
    result = population.resource_projection(plan, timing_reports(seconds=1200), 16)
    assert result["projected_total_cpu_hours"] > 1024
    assert not result["within_budget"]


def rule_summaries():
    baseline = population.balanced_summary(paired_rows())
    return {arm: deepcopy(baseline) for arm in ("B", "C", "D")}


def test_cohort_refinement_accepts_precise_matching_rules():
    result = population.cohort_refinement(rule_summaries())
    assert result["passed"]
    assert set(result["pairs"]) == {"B_C", "B_D"}
    assert all(pair["passed"] for pair in result["pairs"].values())


def test_coherent_shift_cannot_pass_through_individual_point_tolerance():
    summaries = rule_summaries()
    # .009 is below the individual .01-bit threshold but not the mean threshold.
    summaries["C"]["bayes"]["mean"] += .009
    result = population.cohort_refinement(summaries)
    pair = result["pairs"]["B_C"]
    assert pair["absolute_difference_bits"] == pytest.approx(.009)
    assert pair["absolute_difference_bits"] > pair["resolution_limit_bits"]
    assert not pair["passed"] and not result["passed"]


def test_cohort_refinement_requires_stochastic_agreement_as_well_as_resolution():
    summaries = rule_summaries()
    summaries["D"]["bayes"]["mean"] += .0001
    result = population.cohort_refinement(summaries)
    pair = result["pairs"]["B_D"]
    assert pair["absolute_difference_bits"] < pair["resolution_limit_bits"]
    assert pair["absolute_difference_bits"] > pair["stochastic_agreement_limit_bits"]
    assert not pair["passed"] and not result["passed"]


def test_cohort_refinement_requires_every_rules_precision_budget():
    summaries = rule_summaries()
    summaries["C"]["precision_passed"] = False
    assert not population.cohort_refinement(summaries)["passed"]


def test_cohort_refinement_rejects_missing_integration_rule():
    summaries = rule_summaries()
    del summaries["D"]
    with pytest.raises(ValueError):
        population.cohort_refinement(summaries)
