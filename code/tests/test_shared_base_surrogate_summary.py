"""The new exporter must detect stale, tampered or incomplete raw runs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base_full_context_summary import Audit
from shared_base_surrogate import SharedBaseSurrogateMCMC
from shared_base_surrogate_summary import raw_checkpoint
from surrogate_chain_runner import SurrogateChainRun


def saved_case(tmp_path):
    sampler = SharedBaseSurrogateMCMC(3, RankedLSARowEvidence(3, [.75, 1.25], [0]),
                                     [.5, .5], surrogate_steps=4)
    config = dict(chains=4, burn=11, thin=1, seed=439)
    run = SurrogateChainRun(sampler, [0, 1, 0, 2, 0, 0, 1, 0], **config)
    estimate = run.extend(64, progress=lambda _: None)
    path = tmp_path / "chain.npz"
    run.save(path)
    report = dict(identity=dict(numerical_problem=run.fingerprint,
                                proposal_descriptor=run.proposal_descriptor),
                  settings=config, estimate=estimate)
    return path, report, run.samples


def test_complete_raw_readback_recovers_current_report(tmp_path):
    path, report, samples = saved_case(tmp_path)
    audit = Audit()
    moments, fresh = raw_checkpoint(audit, "reference", path, report)
    assert all(check["passed"] for check in audit.checks)
    np.testing.assert_allclose(moments["mean"], samples["predictive_probs"].mean(axis=(0, 1)), atol=2e-14)
    np.testing.assert_array_equal(fresh["predictive_probs"], report["estimate"]["predictive_probs"])


def test_stale_report_prediction_is_rejected(tmp_path):
    path, report, _ = saved_case(tmp_path)
    report = deepcopy(report)
    report["estimate"]["predictive_probs"][0] += .001
    with pytest.raises(ValueError, match="raw_summary:predictive_probs"):
        raw_checkpoint(Audit(), "reference", path, report)


def test_modified_raw_sample_is_rejected_before_it_can_affect_summary(tmp_path):
    path, report, _ = saved_case(tmp_path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    arrays["predictions_0"][0, 0] += .001
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="array_checksum:predictions_0"):
        raw_checkpoint(Audit(), "reference", path, report)


def test_incomplete_raw_array_inventory_is_rejected(tmp_path):
    path, report, _ = saved_case(tmp_path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files if key != "predictions_3"}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="complete_raw_arrays"):
        raw_checkpoint(Audit(), "reference", path, report)


def test_attempt_counters_are_checked_against_retained_draws(tmp_path):
    path, report, _ = saved_case(tmp_path)
    report["estimate"]["diagnostics"]["move_attempts"][0][0] += 1
    with pytest.raises(ValueError, match="transition_counters"):
        raw_checkpoint(Audit(), "reference", path, report)
