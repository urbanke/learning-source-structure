"""Resumption is bit-identical to the independently exercised core sampler."""
from pathlib import Path
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC
from shared_base_chain_runner import ChainRun


TOKENS = np.array([0, 2, 0, 1, 0, 2, 0, 0, 2])


def sampler(nodes=(0.5, 1.5), weights=(0.3, 0.7), step=1 / 32):
    return SharedBaseMCMC(3, RankedLSARowEvidence(3, nodes, [0], step=step), weights)


def quiet(_):
    pass


def assert_same_reports(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "elapsed_seconds":
            continue
        if isinstance(actual[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


def test_chunked_run_matches_core_samples_and_diagnostics_bit_for_bit():
    config = dict(chains=4, burn=19, thin=3, seed=42)
    expected = sampler().sample(TOKENS, draws=48, retain_samples=True, **config)
    original = expected.pop("samples")
    run = ChainRun(sampler(), TOKENS, **config)
    run.extend(16, block_draws=7, progress=quiet)
    actual = run.extend(48, block_draws=13, progress=quiet)
    assert_same_reports(actual, expected)
    for key, core_key in (("predictive_probs", "predictive_probs"),
                          ("posterior_depth", "posterior_depth"),
                          ("alpha_indices", "alpha_index"),
                          ("alpha_and_loglik", "alpha_and_loglik")):
        np.testing.assert_array_equal(run.samples[key], original[core_key])


def test_save_resume_matches_uninterrupted_and_burn_is_not_repeated(tmp_path):
    config = dict(chains=2, burn=13, thin=2, seed=73)
    expected_run = ChainRun(sampler(), TOKENS, **config)
    expected = expected_run.extend(40, progress=quiet)
    run = ChainRun(sampler(), TOKENS, **config)
    run.extend(16, progress=quiet)
    path = tmp_path / "checkpoint.npz"
    run.save(path)
    resumed = ChainRun.load(sampler(), TOKENS, path)
    assert_same_reports(resumed.extend(40, block_draws=5, progress=quiet), expected)
    for key in run.samples:
        np.testing.assert_array_equal(resumed.samples[key], expected_run.samples[key])
    assert np.all(resumed._burn_done == config["burn"])


def test_resume_partially_completed_chain_blocks(tmp_path):
    config = dict(chains=2, burn=7, thin=2, seed=11)
    reference = ChainRun(sampler(), TOKENS, **config)
    reference.extend(32, progress=quiet)
    interrupted = ChainRun(sampler(), TOKENS, **config)
    path = tmp_path / "partial.npz"

    class StopAtCheckpoint(Exception):
        pass

    def save_and_stop(event):
        if event["phase"] == "draws":
            interrupted.save(path)
            raise StopAtCheckpoint()

    with pytest.raises(StopAtCheckpoint):
        interrupted.extend(32, block_draws=5, progress=save_and_stop)
    np.testing.assert_array_equal(interrupted.draw_counts, [5, 0])
    with pytest.raises(ValueError, match="equal retained"):
        interrupted.report()
    resumed = ChainRun.load(sampler(), TOKENS, path)
    assert_same_reports(resumed.extend(32, block_draws=3, progress=quiet), reference.report())
    for key in resumed.samples:
        np.testing.assert_array_equal(resumed.samples[key], reference.samples[key])


@pytest.mark.parametrize("change", ["tokens", "nodes", "weights", "grid"])
def test_checkpoint_rejects_changed_numerical_problem(tmp_path, change):
    run = ChainRun(sampler(), TOKENS, chains=2, burn=0, thin=1, seed=4)
    path = tmp_path / "state.npz"
    run.save(path)
    tokens = TOKENS.copy()
    changed = sampler()
    if change == "tokens":
        tokens[-1] = 1
    elif change == "nodes":
        changed = sampler(nodes=(0.6, 1.5))
    elif change == "weights":
        changed = sampler(weights=(0.4, 0.6))
    else:
        changed = sampler(step=1 / 64)
    with pytest.raises(ValueError, match="fingerprint"):
        ChainRun.load(changed, tokens, path)


def test_extension_cannot_discard_draws_or_mutate_returned_samples():
    run = ChainRun(sampler(), TOKENS, chains=2, burn=0, thin=1, seed=5)
    run.extend(32, progress=quiet)
    sample = run.samples["predictive_probs"]
    sample[:] = 99
    assert np.max(run.samples["predictive_probs"]) < 1
    with pytest.raises(ValueError, match="discard"):
        run.extend(16, progress=quiet)


def test_weighted_depth_resume_reconstructs_state_exactly(tmp_path):
    def weighted():
        return SharedBaseMCMC(3, RankedLSARowEvidence(3, [0.5, 1.5], [0, 1]), [0.3, 0.7])
    config = dict(chains=2, burn=9, thin=2, seed=84)
    expected = weighted().sample(TOKENS, draws=32, **config)
    run = ChainRun(weighted(), TOKENS, **config)
    run.extend(16, progress=quiet)
    path = tmp_path / "weighted.npz"
    run.save(path)
    resumed = ChainRun.load(weighted(), TOKENS, path)
    assert_same_reports(resumed.extend(32, progress=quiet), expected)


def test_failed_partial_block_rolls_back_rng_and_state(monkeypatch):
    config = dict(chains=2, burn=5, thin=2, seed=109)
    reference = ChainRun(sampler(), TOKENS, **config)
    reference.extend(16, progress=quiet)
    run = ChainRun(sampler(), TOKENS, **config)
    original = run.target.predict
    calls = 0

    def fail_second_prediction(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("simulated interrupted block")
        return original(state)

    monkeypatch.setattr(run.target, "predict", fail_second_prediction)
    with pytest.raises(FloatingPointError, match="interrupted block"):
        run.extend(16, block_draws=5, progress=quiet)
    np.testing.assert_array_equal(run.draw_counts, [0, 0])
    monkeypatch.setattr(run.target, "predict", original)
    assert_same_reports(run.extend(16, block_draws=5, progress=quiet), reference.report())


@pytest.mark.skipif(not (os.environ.get("LSA_STORE") and os.environ.get("PMM_SRC")),
                    reason="all-depth resumption requires the read-only production store")
def test_all_depth_state_reconstruction_after_save(tmp_path):
    def full():
        return SharedBaseMCMC(3, RankedLSARowEvidence(3, [0.7, 1.3]), [0.5, 0.5])
    config = dict(chains=2, burn=7, thin=2, seed=59)
    expected = full().sample(TOKENS, draws=32, **config)
    run = ChainRun(full(), TOKENS, **config)
    run.extend(16, progress=quiet)
    path = tmp_path / "all_depths.npz"
    run.save(path)
    resumed = ChainRun.load(full(), TOKENS, path)
    assert_same_reports(resumed.extend(32, progress=quiet), expected)
