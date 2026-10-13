"""Reporting must preserve pilot scope, replicate identities and exact snapshots."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from calibrate_shared_base_evidence import SCHEMA, fingerprint, replicate_seed
from shared_base_calibration_report import collect


def write_protocol(tmp_path):
    protocol = dict(frozen_cases=dict(cohorts=['product_prior', 'shared_base_prior', 'uniform_base_prior'],
                    sequence_index=0, additional_exact_controls=[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 2, 0]]),
                    short_gate=dict(minimum_completed_replicates=16, tolerance_bits=.01,
                                    minimum_effective_replica_count=8,
                                    maximum_single_replica_likelihood_share=.2))
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    return path


def cases():
    names = ['product_prior_0', 'shared_base_prior_0', 'uniform_base_prior_0',
             'pattern_0_0_0_0_0', 'pattern_0_1_0_1_0', 'pattern_0_1_0_2_0']
    return [dict(name=name, id='case-'+name, tokens=[0, 1, 2, 3, 4], catalog_log_evidence=-10.)
            for name in names]


def write_run(tmp_path, protocol, subset=None, *, particles=256, filename='run.json', noisy=False):
    selected = cases() if subset is None else [cases()[i] for i in subset]
    identity = dict(cases=selected, replicates=16, master_seed=111, n_tokens=5,
                    algorithm=dict(sampler='original', particles=particles),
                    predeclared_protocol=dict(sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()))
    records = []
    for case in selected:
        for i in range(16):
            records.append(dict(sequence_id=case['id'], replica=i, seed=replicate_seed(case['id'], i, 111),
                                estimate=dict(log_evidence=-10.+(np.log(.9 if i%2 else 1.1) if noisy else 0.),
                                              elapsed_seconds=.25, initial_particle_ancestry_counts=[particles, 50],
                                              effective_sample_sizes=[particles, particles/2])))
    payload = dict(schema=SCHEMA, identity=identity, identity_id=fingerprint(identity), records=records)
    path = tmp_path / filename
    path.write_text(json.dumps(payload))
    return path


def test_complete_six_control_scope_and_compact_estimates(tmp_path):
    protocol = write_protocol(tmp_path)
    path = write_run(tmp_path, protocol)
    report = collect([path], protocol)
    coverage, = report['configuration_coverage']
    assert coverage['all_predeclared_cases_complete']
    assert coverage['all_short_controls_passed']
    assert coverage['missing_cases'] == []
    assert len(coverage['expected_cases']) == 6
    raw = json.loads(path.read_text())
    row = report['rows'][0]
    assert row['tokens'] == raw['identity']['cases'][0]['tokens']
    assert row['input_identity_id'] == raw['identity_id']
    assert row['catalog_log_evidence'] == -10.
    assert len(row['repetition_estimates']) == 16
    assert row['repetition_estimates'][0] == dict(replica=0,
        seed=raw['records'][0]['seed'], log_evidence=-10., elapsed_seconds=.25)


def test_subset_refinement_cannot_inherit_baseline_coverage(tmp_path):
    protocol = write_protocol(tmp_path)
    full = write_run(tmp_path, protocol)
    subset = write_run(tmp_path, protocol, [3], particles=1024, filename='refinement.json')
    report = collect([full, subset], protocol)
    baseline, refinement = report['configuration_coverage']
    assert baseline['all_short_controls_passed']
    assert not refinement['all_predeclared_cases_complete']
    assert not refinement['all_short_controls_passed']
    assert len(refinement['missing_cases']) == 5
    assert refinement['status'] == 'partial predeclared coverage'
    assert report['rows'][-1]['short_control_passed']


def test_complete_cases_can_fail_accuracy_gate(tmp_path):
    protocol = write_protocol(tmp_path)
    report = collect([write_run(tmp_path, protocol, noisy=True)], protocol)
    coverage, = report['configuration_coverage']
    assert coverage['all_predeclared_cases_complete']
    assert not coverage['all_short_controls_passed']
    row = report['rows'][0]
    assert row['statistics']['loss_of_mean_evidence_bits'] == pytest.approx(10/np.log(2))
    assert row['error_plus_mc_halfwidth_bits'] > .01
    assert not row['checks']['exact_control_error_and_precision']


@pytest.mark.parametrize('damage', ['seed', 'identity', 'duplicate'])
def test_report_rejects_invalid_saved_run_identity(tmp_path, damage):
    protocol = write_protocol(tmp_path)
    path = write_run(tmp_path, protocol)
    data = json.loads(path.read_text())
    if damage == 'seed':
        data['records'][0]['seed'] += 1
    elif damage == 'identity':
        data['identity']['master_seed'] += 1
    else:
        data['records'].append(data['records'][0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        collect([path], protocol)


def test_hashes_match_parsed_bytes_when_live_file_changes(tmp_path, monkeypatch):
    protocol = write_protocol(tmp_path)
    path = write_run(tmp_path, protocol)
    original = path.read_bytes()
    read_bytes = Path.read_bytes
    def replaced_after_read(self):
        raw = read_bytes(self)
        if self == path:
            self.write_text('{"later_version": true}')
        return raw
    monkeypatch.setattr(Path, 'read_bytes', replaced_after_read)
    report = collect([path], protocol)
    assert report['input_sha256'][str(path)] == hashlib.sha256(original).hexdigest()
    assert report['configuration_coverage'][0]['all_short_controls_passed']
    assert json.loads(read_bytes(path)) == {'later_version': True}


def test_incomplete_repetitions_are_explicit(tmp_path):
    protocol = write_protocol(tmp_path)
    path = write_run(tmp_path, protocol)
    data = json.loads(path.read_text())
    data['records'].pop()
    path.write_text(json.dumps(data))
    report = collect([path], protocol)
    coverage, = report['configuration_coverage']
    assert not coverage['all_predeclared_cases_complete']
    assert coverage['missing_cases'] == []
    assert coverage['incomplete_cases'] == ['pattern_0_1_0_2_0']
    assert coverage['status'] == 'incomplete repetitions'
