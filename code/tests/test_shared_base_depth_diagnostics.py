"""Read-only checkpoint inspection and scheduler-only launch boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/scripts'), str(ROOT/'code/transformer')]
import inspect_shared_base_depth_diagnostics as diagnostic
from shared_base_chain_runner import _hash_array
from stable_chain_statistics import stable_chain_summary


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(321)
    samples = rng.dirichlet(np.ones(9), size=(4, 64))
    arrays = {f'depths_{i}':samples[i] for i in range(4)}
    metadata = dict(format='shared_base_chain_run_v1', config=dict(chains=4),
                    array_hashes={k:_hash_array(v) for k,v in arrays.items()})
    encoded = json.dumps(metadata)
    arrays.update(metadata=np.asarray(encoded),
                  metadata_sha256=np.asarray(hashlib.sha256(encoded.encode()).hexdigest()))
    path = tmp_path/'chains.npz'
    np.savez(path, **arrays)
    summary = stable_chain_summary(samples)
    estimate = dict(posterior_depth=summary['mean'].tolist(), mcse_depth=summary['mcse'].tolist(),
        diagnostics=dict(chains=4, draws=64,
            rhat_depth=[float(v) if np.isfinite(v) else None for v in summary['rhat']]))
    return path, arrays, estimate


def test_saved_depth_readback_preserves_all_bytes_and_reports_all_coordinates(checkpoint):
    path, _, estimate = checkpoint
    before = path.read_bytes()
    rows = diagnostic.summarize_depths(path, estimate)
    assert path.read_bytes() == before
    assert [row['depth'] for row in rows] == list(range(9))
    assert all(row['rescaled']['rhat'] is not None for row in rows)
    assert all('passed' not in row for row in rows)
    json.dumps(rows, allow_nan=False)


def test_corrupt_depth_samples_are_rejected(checkpoint):
    path, arrays, estimate = checkpoint
    arrays['depths_0'][0, 0] += 1e-4
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='checksum mismatch'):
        diagnostic.summarize_depths(path, estimate)


def test_changed_mean_or_legacy_diagnostic_is_rejected(checkpoint):
    path, _, estimate = checkpoint
    estimate['posterior_depth'][0] *= 1.1
    with pytest.raises(ValueError, match='depth mean differs'):
        diagnostic.summarize_depths(path, estimate)
    estimate['posterior_depth'][0] /= 1.1
    estimate['diagnostics']['rhat_depth'][0] += .1
    with pytest.raises(ValueError, match='Legacy Rhat'):
        diagnostic.summarize_depths(path, estimate)


def test_dry_run_only_prepares_compute_node_command(tmp_path, monkeypatch):
    source = tmp_path/'recovery-launch.json'
    source.write_text('{}')
    old = dict(campaign=dict(output=str(tmp_path/'original'), inputs=str(tmp_path/'inputs')))
    recovered = dict(context=dict(output=str(tmp_path/'recovery')), venv=str(tmp_path/'venv'))
    monkeypatch.setattr(diagnostic, 'locate_launch', lambda _:source)
    monkeypatch.setattr(diagnostic.recovery, 'read_recovery', lambda _:(recovered, old, {}))
    monkeypatch.setattr(diagnostic.subprocess, 'run', lambda *a, **k:pytest.fail('Dry run submitted'))
    args = argparse.Namespace(recovery=tmp_path/'recovery', out=tmp_path/'diagnostics', submit=False)
    diagnostic.submit(args)
    assert not args.out.exists()
    spec, _, command = diagnostic.build_launch(args)
    assert spec['task_indices'] == [18, 19, 24, 25]
    assert spec['no_sampling'] and not spec['changes_acceptance']
    assert command[0] == 'sbatch'


def test_original_output_cannot_be_used_for_diagnostics(tmp_path, monkeypatch):
    source = tmp_path/'recovery-launch.json'
    source.write_text('{}')
    old = dict(campaign=dict(output=str(tmp_path/'original'), inputs=str(tmp_path/'inputs')))
    recovered = dict(context=dict(output=str(tmp_path/'recovery')), venv=str(tmp_path/'venv'))
    monkeypatch.setattr(diagnostic, 'locate_launch', lambda _:source)
    monkeypatch.setattr(diagnostic.recovery, 'read_recovery', lambda _:(recovered, old, {}))
    for output in (tmp_path, tmp_path/'original', tmp_path/'original'/'new'):
        with pytest.raises(ValueError, match='separate'):
            diagnostic.build_launch(argparse.Namespace(recovery=None, out=output))
