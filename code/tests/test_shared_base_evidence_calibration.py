"""Statistical and resume-identity checks for the bounded calibration runner."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_shared_base_evidence import (SCHEMA, check_resume, fingerprint,
    normalizer_summary, numerical_environment, paired_budget_comparison, replicate_seed)


@pytest.mark.parametrize("name,value", [("PMM_PHI_BIAS", ".01"), ("PMM_PHI_WAVE", "-1e-12"), ("PMM_PHI_BIAS", "nan")])
def test_perturbed_backend_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="perturbation"):
        numerical_environment()


def test_averages_likelihood_not_log_likelihood():
    result = normalizer_summary(np.log([.1, .4]), np.log(.25))
    assert result["loss_of_mean_evidence_bits"] == pytest.approx(2.)
    assert result["estimated_minus_catalog_loss_bits"] == pytest.approx(0.)
    assert result["arithmetic_mean_replicate_loss_bits_diagnostic_only"] != pytest.approx(2.)
    assert result["effective_replica_count"] == pytest.approx(.5**2 / (.1**2 + .4**2))
    assert result["maximum_single_replica_likelihood_share"] == pytest.approx(.8)


def test_stable_for_tiny_probabilities_and_constant_replicates():
    result = normalizer_summary(np.full(16, -10000.), -10000.)
    assert result["log_mean_evidence_nats"] == -10000.
    assert result["loss_mc_se_bits_delta_method"] == 0.
    assert result["absolute_error_plus_ci95_max_halfwidth_bits"] == 0.
    assert result["effective_replica_count"] == 16.
    assert result["maximum_single_replica_likelihood_share"] == 1 / 16


def test_partial_and_unbounded_intervals_are_explicit():
    single = normalizer_summary([-10.])
    assert single["relative_mc_se"] is None
    assert single["approximate_loss_ci95_bits"] is None
    noisy = normalizer_summary([-100., 0.])
    assert noisy["unbounded_upper_loss_ci"] is True
    assert noisy["approximate_loss_ci95_bits"][1] is None
    assert noisy["approximate_loss_ci95_max_halfwidth_bits"] is None


def test_replica_seed_independence_and_repeatability():
    assert replicate_seed("case-a", 0) == replicate_seed("case-a", 0)
    assert len({replicate_seed("case-a", 0), replicate_seed("case-a", 1),
                replicate_seed("case-b", 0), replicate_seed("case-a", 0, 99)}) == 4


def fixture_report(scale=1.):
    identity = dict(cases=[dict(id="case-a")], replicates=4, master_seed=2026091603,
                    algorithm=dict(particles=32))
    records = [dict(sequence_id="case-a", replica=i, seed=replicate_seed("case-a", i),
                    estimate=dict(log_evidence=np.log(scale * value)))
               for i, value in enumerate((.1, .2, .3, .4))]
    return dict(schema=SCHEMA, identity=identity, identity_id=fingerprint(identity), records=records)


def test_resume_rejects_duplicate_or_changed_repetition():
    report = fixture_report()
    assert len(check_resume(report, report["identity"])) == 4
    report["records"].append(report["records"][0])
    with pytest.raises(ValueError):
        check_resume(report, report["identity"])
    report = fixture_report()
    report["identity"]["master_seed"] = 5
    with pytest.raises(ValueError):
        check_resume(report, report["identity"])


def test_paired_budget_comparison_retains_covariance():
    original, changed = fixture_report(), fixture_report(2.)
    result = paired_budget_comparison(changed, original)["by_fixed_sequence"]["case-a"]
    assert result["current_minus_previous_loss_of_mean_evidence_bits"] == pytest.approx(-1.)
    assert result["paired_mc_se_bits_delta_method"] < 1e-14
