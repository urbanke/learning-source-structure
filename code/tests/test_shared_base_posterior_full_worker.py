"""Checkpoint scheduling must preserve samples and exact interrupted restart."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shared_base_posterior_full_worker as worker
from ranked_lsa import RankedLSARowEvidence
from shared_base_curve_runner import CurveChainRun
from shared_base_surrogate import SharedBaseSurrogateMCMC


TOKENS = np.asarray([0, 2, 0, 1, 0, 2, 0, 0, 2])


def make_run():
    kernel = RankedLSARowEvidence(3, [.5, 1.5], [0, 1])
    sampler = SharedBaseSurrogateMCMC(3, kernel, [.3, .7], surrogate_steps=4)
    return CurveChainRun(sampler, TOKENS, chains=4, burn=7, thin=1, seed=701)


def ticking():
    n = [0]
    def clock():
        n[0] += 301
        return n[0]
    return clock


def test_periodic_saves_preserve_exact_samples_and_restore_method(tmp_path):
    baseline = make_run()
    baseline.extend(16, block_draws=4, progress=lambda event: None)
    candidate = make_run()
    path = tmp_path / 'checkpoint.npz'
    method = CurveChainRun.extend
    with worker.periodic_checkpoints(path, clock=ticking()):
        candidate.extend(16, block_draws=4, progress=lambda event: None)
    assert path.exists()
    assert CurveChainRun.extend is method
    for name in baseline.samples:
        np.testing.assert_array_equal(candidate.samples[name], baseline.samples[name])


def test_resume_after_periodic_mid_chain_save_is_bit_exact(tmp_path, monkeypatch):
    expected = make_run()
    expected.extend(16, block_draws=4, progress=lambda event: None)
    interrupted = make_run()
    path = tmp_path / 'checkpoint.npz'
    real_save = CurveChainRun.save
    class Stop(Exception):
        pass
    def save_then_stop(self, destination):
        real_save(self, destination)
        raise Stop()
    monkeypatch.setattr(CurveChainRun, 'save', save_then_stop)
    with pytest.raises(Stop), worker.periodic_checkpoints(path, clock=ticking()):
        interrupted.extend(16, block_draws=4, progress=lambda event: None)
    monkeypatch.setattr(CurveChainRun, 'save', real_save)
    resumed = CurveChainRun.load(make_run().sampler, TOKENS, path)
    assert 0 < resumed.draw_counts[0] < 16
    resumed.extend(16, block_draws=3, progress=lambda event: None)
    for name in expected.samples:
        np.testing.assert_array_equal(resumed.samples[name], expected.samples[name])


def test_positive_checkpoint_interval_required(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        with worker.periodic_checkpoints(tmp_path / 'bad.npz', interval_seconds=0):
            pass
