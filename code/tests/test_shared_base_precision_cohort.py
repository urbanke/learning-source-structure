"""Checks for the distinct cohort-mean numerical estimand and log-loss bias."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shared_base_precision_cohort import case_precision, deterministic_precision, validate_cohort
from shared_base_precision_diagnostic import self_test


def records(n=256, variance=.0145**2, bias=.0005, shift=0., seed_offset=0):
    result = []
    for index in range(n):
        precision = dict(bayes_regret_bits=.5+(index % 2)+shift,
            mc_variance_bits2=variance, jensen_proxy_bits=bias,
            deterministic=False, scalar_mixing_passed=True, jensen_local_check_passed=True)
        result.append(dict(case_id=index, sampling_seed=seed_offset+index,
            precision=precision, full_prior_valid=True, provenance_valid=True,
            mixing_passed=True, row_grid_passed=True, per_sequence_precision_passed=False))
    return result


def test_aggregate_random_error_averages_but_old_individual_status_is_preserved():
    out = validate_cohort(records(), records(seed_offset=1000), list(range(256)))
    assert out["passed"]
    assert out["reported_mean_bayes_regret_bits"] == 1.
    assert out["rules"][0]["numerical_mcse_bits"] == pytest.approx(.0145/16)
    assert out["rules"][0]["numerical_error_allowance_bits"] == pytest.approx(2*.0145/16+.001)
    assert not any(out["rules"][0]["per_sequence_precision_status"].values())
    small = validate_cohort(records(8), records(8, seed_offset=1000), list(range(8)))
    assert not small["passed"] and small["reported_mean_bayes_regret_bits"] is None


def test_jensen_bias_does_not_shrink_with_cohort_size():
    for n in (256, 1024):
        out = validate_cohort(records(n, bias=.003), records(n, bias=.003, seed_offset=10000), list(range(n)))
        assert not out["passed"]
        assert out["rules"][0]["mean_jensen_proxy_bits"] == pytest.approx(.003)


def test_missing_cases_duplicates_and_shared_random_seeds_are_rejected():
    coarse, refined = records(8), records(8, seed_offset=1000)
    with pytest.raises(ValueError, match="every declared case"):
        validate_cohort(coarse[:-1], refined, list(range(8)))
    with pytest.raises(ValueError, match="every declared case"):
        validate_cohort(coarse+[coarse[0]], refined, list(range(8)))
    with pytest.raises(ValueError, match="independent"):
        validate_cohort(coarse, deepcopy(coarse), list(range(8)))


@pytest.mark.parametrize("flag", ["full_prior_valid", "provenance_valid", "mixing_passed", "row_grid_passed"])
def test_one_failed_required_case_cannot_be_dropped(flag):
    coarse, refined = records(), records(seed_offset=1000)
    coarse[117][flag] = False
    out = validate_cohort(coarse, refined, list(range(256)))
    assert not out["passed"] and out["reported_mean_bayes_regret_bits"] is None
    assert out["n_sequences"] == 256


def test_large_pointwise_quadrature_errors_cannot_cancel_in_the_mean():
    coarse, refined = records(), records(seed_offset=1000)
    coarse[0]["precision"]["bayes_regret_bits"] += 1
    coarse[1]["precision"]["bayes_regret_bits"] -= 1
    out = validate_cohort(coarse, refined, list(range(256)))
    assert out["refinement"]["absolute_mean_difference_bits"] == 0
    assert not out["passed"]
    assert sum(not row["passed"] for row in out["refinement"]["per_case"]) == 2


def test_quadrature_bias_cannot_pass_only_because_each_point_is_noisy():
    out = validate_cohort(records(), records(shift=.005, seed_offset=1000), list(range(256)))
    assert not out["passed"]
    assert not next(c["passed"] for c in out["checks"] if c["name"] == "aggregate_refinement_resolution")


def test_mc_noise_cannot_inflate_the_relative_sequence_budget():
    coarse, refined = records(), records(seed_offset=1000)
    for group in (coarse, refined):
        for index, r in enumerate(group):
            r["precision"]["bayes_regret_bits"] = 1+.005*(index % 2)
    out = validate_cohort(coarse, refined, list(range(256)))
    assert out["rules"][0]["observed_sequence_se_bits"] > 0
    assert out["rules"][0]["denoised_sequence_se_bits"] == 0
    assert not out["passed"]


def test_true_loss_curvature_weights_rare_bayes_tokens_by_oracle_mass():
    rng = np.random.default_rng(431)
    first = .02+.004*rng.choice([-1, 1], size=(4,4096))
    samples = np.stack([first, 1-first], axis=2)
    oracle = np.array([.8,.2])
    out = case_precision(samples, oracle)
    # For two tokens C has diagonal (v,v); independently reconstruct the
    # curvature from scalar variances at each batching scale and across chains.
    v = [first.mean(axis=1).var(ddof=1)/4]
    for length in (64,128,256,512):
        batch = first.reshape(-1,length).mean(axis=1)
        v.append(batch.var(ddof=1)/len(batch))
    p = np.array([first.mean(), 1-first.mean()])
    correct = .5*max(v)*np.sum(oracle/p**2)/np.log(2)
    incorrect_kl_noise = .5*max(v)*np.sum(1/p)/np.log(2)
    assert out["jensen_curvature_scale_bits"] == pytest.approx(correct, rel=1e-9)
    assert out["jensen_curvature_scale_bits"] > 30*incorrect_kl_noise
    assert out["scalar_mixing_passed"] and out["jensen_local_check_passed"]


def test_deterministic_curve_has_zero_mc_budget_and_retains_oracle_scoring():
    oracle = np.array([.8,.2]); prediction = np.array([.6,.4])
    exact = deterministic_precision(prediction, oracle)
    assert exact["mcse_bits"] == exact["jensen_proxy_bits"] == 0.
    assert exact["bayes_regret_bits"] == pytest.approx(oracle@np.log2(oracle/prediction))
    assert self_test()["passed"]


def test_nonmixing_scalar_trace_is_rejected_even_when_other_flags_pass():
    first = np.vstack([np.full((2,512),.2), np.full((2,512),.8)])
    samples = np.stack([first,1-first], axis=2)
    out = case_precision(samples, [.9,.1])
    assert not out["scalar_mixing_passed"]
    assert not out["jensen_local_check_passed"]


def test_long_chain_prediction_matches_stable_reference_reduction():
    from stable_chain_statistics import stable_predictive_moments
    samples = np.broadcast_to(np.array([.1,.9]), (4,262144,2)).copy()
    stable = stable_predictive_moments(samples)
    out = case_precision(samples, [.8,.2])
    assert np.array_equal(out["predictive_probs"], stable["mean"])
    assert np.allclose(out["predictive_probs"], [.1,.9], rtol=0, atol=2e-16)
    assert np.max(abs(samples.mean(axis=(0,1))-stable["mean"])) > 2e-14
