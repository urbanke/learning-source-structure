"""Bit-exact restart and lossless-storage checks for conditional curve runs."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transformer'))
from ranked_lsa import RankedLSARowEvidence
from shared_base_chain_runner import ChainRun
from shared_base_curve_runner import CurveChainRun, descriptor
from shared_base_efficiency import SharedBaseEfficiencyMCMC, SharedBaseEfficiencyRowMCMC
from shared_base_surrogate import SharedBaseSurrogateMCMC
from stable_chain_statistics import stable_predictive_moments
from surrogate_chain_runner import SurrogateChainRun


TOKENS = np.array([0, 2, 0, 1, 0, 2, 0, 0, 2])


def sampler(kind='iid', steps=64, power=.5, concentration=1.):
    kernel = RankedLSARowEvidence(3, [.5, 1.5], [0, 1])
    kwargs = dict(surrogate_steps=steps)
    if kind == 'tempered':
        return SharedBaseEfficiencyMCMC(3, kernel, [.3, .7], auxiliary_power=power, **kwargs)
    if kind == 'row':
        return SharedBaseEfficiencyRowMCMC(3, kernel, [.3, .7], concentration=concentration, **kwargs)
    return SharedBaseSurrogateMCMC(3, kernel, [.3, .7], **kwargs)


def assert_same_reports(a, b):
    assert a.keys() == b.keys()
    for name in a:
        if name == 'elapsed_seconds':
            continue
        if isinstance(a[name], np.ndarray):
            np.testing.assert_array_equal(a[name], b[name])
        else:
            assert a[name] == b[name]


def quiet(event):
    pass


def test_iid64_samples_and_stable_report_match_existing_runner_bit_for_bit():
    config = dict(chains=4, burn=19, thin=2, seed=20261006)
    original = SurrogateChainRun(sampler(), TOKENS, **config)
    expected = original.extend(48, block_draws=11, progress=quiet)
    compressed = CurveChainRun(sampler(), TOKENS, **config)
    actual = compressed.extend(48, block_draws=7, progress=quiet)
    for name in original.samples:
        np.testing.assert_array_equal(compressed.samples[name], original.samples[name])
    assert_same_reports(actual, expected)
    moments = stable_predictive_moments(compressed.samples['predictive_probs'])
    np.testing.assert_allclose(moments['mean'], actual['predictive_probs'], atol=2e-14, rtol=0)
    np.testing.assert_allclose(moments['mcse_probs'], actual['mcse_probs'], atol=1e-15, rtol=1e-11)


@pytest.mark.parametrize('kind', ['iid', 'tempered', 'row'])
@pytest.mark.parametrize('stop_phase', ['burn', 'draws'])
def test_resume_unequal_counts_is_exact_and_preserves_burn(tmp_path, kind, stop_phase):
    config = dict(chains=4, burn=11, thin=2, seed=502)
    reference = CurveChainRun(sampler(kind, steps=4), TOKENS, **config)
    expected = reference.extend(32, block_draws=8, progress=quiet)
    partial = CurveChainRun(sampler(kind, steps=4), TOKENS, **config)
    path = tmp_path / 'partial.npz'

    class Interrupted(Exception):
        pass

    def stop(event):
        if event['phase'] == stop_phase:
            partial.save(path)
            raise Interrupted()

    with pytest.raises(Interrupted):
        partial.extend(32, block_draws=5, progress=stop)
    expected_counts = [5, 0, 0, 0] if stop_phase == 'draws' else [0, 0, 0, 0]
    np.testing.assert_array_equal(partial.draw_counts, expected_counts)
    resumed = CurveChainRun.load(sampler(kind, steps=4), TOKENS, path)
    np.testing.assert_array_equal(resumed.draw_counts, expected_counts)
    np.testing.assert_array_equal(resumed._burn_done, partial._burn_done)
    actual = resumed.extend(32, block_draws=3, progress=quiet)
    assert_same_reports(actual, expected)
    for name in reference.samples:
        np.testing.assert_array_equal(resumed.samples[name], reference.samples[name])
    np.testing.assert_array_equal(resumed._burn_done, [11]*4)


def test_zip_compression_preserves_every_array_payload_and_checksum(tmp_path):
    run = CurveChainRun(sampler(steps=4), TOKENS, chains=2, burn=7, thin=1, seed=6)
    run.extend(32, progress=quiet)
    raw, compressed = tmp_path/'raw.npz', tmp_path/'compressed.npz'
    ChainRun.save(run, raw)
    run.save(compressed)
    with zipfile.ZipFile(raw) as source, zipfile.ZipFile(compressed) as destination:
        assert set(source.namelist()) == set(destination.namelist())
        assert all(entry.compress_type == zipfile.ZIP_DEFLATED for entry in destination.infolist())
        for name in source.namelist():
            if name not in ('metadata.npy', 'metadata_sha256.npy'):
                assert source.read(name) == destination.read(name)
    with np.load(raw, allow_pickle=False) as source, np.load(compressed, allow_pickle=False) as destination:
        original = json.loads(str(source['metadata'].item()))
        serialized = str(destination['metadata'].item())
        changed = json.loads(serialized)
        assert changed.pop('proposal_descriptor') == descriptor(run.sampler)
        assert changed == original
        assert hashlib.sha256(serialized.encode()).hexdigest() == str(destination['metadata_sha256'].item())
    assert compressed.stat().st_size < raw.stat().st_size
    assert not list(tmp_path.glob('.curve-checkpoint-*'))


@pytest.mark.parametrize('change', ['steps', 'kind', 'power', 'concentration'])
def test_checkpoint_rejects_changed_proposal(tmp_path, change):
    initial = sampler('tempered', steps=4) if change == 'power' else sampler('row', steps=4) if change == 'concentration' else sampler(steps=4)
    changed = (sampler(steps=8) if change == 'steps' else sampler('row', steps=4)
               if change == 'kind' else sampler('tempered', steps=4, power=.75)
               if change == 'power' else sampler('row', steps=4, concentration=2.))
    path = tmp_path/'run.npz'
    CurveChainRun(initial, TOKENS, chains=2, burn=0, thin=1, seed=4).save(path)
    with pytest.raises(ValueError, match='proposal or implementation'):
        CurveChainRun.load(changed, TOKENS, path)


@pytest.mark.parametrize('kind,attribute,value', [
    ('iid', 'surrogate_steps', 8),
    ('tempered', 'auxiliary_power', .75),
    ('row', 'concentration', 2.),
])
def test_mutated_live_setting_is_rejected_without_overwriting_checkpoint(
        tmp_path, kind, attribute, value):
    mc = sampler(kind, steps=4)
    run = CurveChainRun(mc, TOKENS, chains=2, burn=0, thin=1, seed=4)
    path = tmp_path/'original.npz'
    run.save(path)
    original_bytes = path.read_bytes()
    setattr(mc, attribute, value)
    for operation in (lambda: descriptor(mc), lambda: run.save(path),
                      lambda: CurveChainRun.load(mc, TOKENS, path)):
        with pytest.raises(ValueError, match=f'live proposal setting.*{attribute}'):
            operation()
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize('corrupt', ['metadata', 'array'])
def test_checksum_failure_is_detected_before_resuming(tmp_path, corrupt):
    path = tmp_path/'run.npz'
    run = CurveChainRun(sampler(steps=4), TOKENS, chains=2, burn=3, thin=1, seed=6)
    run.extend(16, progress=quiet)
    run.save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if corrupt == 'metadata':
        arrays['metadata'] = np.asarray(str(arrays['metadata'].item())+' ')
    else:
        arrays['predictions_0'][0, 0] += 1e-5
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='checksum'):
        CurveChainRun.load(sampler(steps=4), TOKENS, path)


def test_untagged_original_checkpoint_is_rejected(tmp_path):
    path = tmp_path/'original.npz'
    run = CurveChainRun(sampler(steps=4), TOKENS, chains=2, burn=0, thin=1, seed=6)
    ChainRun.save(run, path)
    with pytest.raises(ValueError, match='proposal or implementation'):
        CurveChainRun.load(sampler(steps=4), TOKENS, path)


@pytest.mark.parametrize('failed_gate', [None, 'old_mixing', 'scalar_mixing', 'jensen', 'row_grid'])
def test_worker_requires_every_mixing_and_local_error_gate(tmp_path, monkeypatch, failed_gate):
    """Exercise orchestration with deterministic probabilities, not MCMC cost."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import shared_base_curve_reference as worker
    probabilities = np.full(64, 1/64)
    source = tmp_path/'a1_L4'
    source.mkdir()
    args = SimpleNamespace(input=source, index=0, prefix=16, alpha_order=16,
                           steps=64, proposal='iid', power=.5, concentration=1.,
                           seed=5, out=tmp_path/'references', burn=0,
                           initial_draws=16, max_draws=16)
    monkeypatch.setattr(worker, 'inputs', lambda *args: (
        np.array([0]*16), probabilities,
        {'d64': {'prediction': probabilities}}, {'source_id': 'fixed-input'}))
    monkeypatch.setattr(worker, 'RankedLSARowEvidence', lambda *args, **kwargs: object())
    monkeypatch.setattr(worker, 'SharedBaseSurrogateMCMC', lambda *args, **kwargs: object())
    monkeypatch.setattr(worker, 'diagnostic_gates', lambda estimate: {
        'old_mixing': failed_gate != 'old_mixing'})
    monkeypatch.setattr(worker, 'selected_grid_checks', lambda *args: [
        {'passed': failed_gate != 'row_grid'}])
    monkeypatch.setattr(worker, 'case_precision', lambda *args: dict(
        predictive_probs=probabilities, scalar_mixing_passed=failed_gate != 'scalar_mixing',
        jensen_local_check_passed=failed_gate != 'jensen', mcse_bits=0.,
        jensen_proxy_bits=0., scalar_ess=64.))

    class FakeRun:
        def __init__(self, sampler, tokens, *, chains, burn, thin, seed):
            self.chains, self.burn, self.thin, self.seed = chains, burn, thin, seed
            self.fingerprint, self.proposal_descriptor = {}, {}
            self.draw_counts = np.zeros(chains, dtype=int)
            self.elapsed_seconds = 0.
            self.samples = {'predictive_probs': np.broadcast_to(probabilities, (4, 16, 64)).copy()}

        def save(self, path):
            path.write_bytes(b'fake-checkpoint-not-a-production-chain')

        def extend(self, budget, *, block_draws, progress):
            self.draw_counts[:] = budget
            progress(dict(phase='draws', draws=budget, requested_draws=budget))
            return {'predictive_probs': probabilities}

    monkeypatch.setattr(worker, 'CurveChainRun', FakeRun)
    report = worker.run_job(args)
    assert report['ready_for_cohort_check'] is (failed_gate is None)
    assert report['history'][-1]['ready_for_cohort_check'] is (failed_gate is None)
    assert report['record']['mixing_passed'] is (failed_gate not in ('old_mixing', 'scalar_mixing'))
    # The old scalar status remains explicit and separate from cohort eligibility.
    assert report['record']['per_sequence_precision_passed'] is (failed_gate not in ('old_mixing', 'row_grid'))
