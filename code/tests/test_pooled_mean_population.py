"""Production orchestration replay without live sampling or remote dependencies."""
from contextlib import nullcontext
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('mean_population', ROOT / 'code/scripts/pooled_mean_population.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def component(tokens, probabilities=None):
    probabilities = np.full(64, 1/64) if probabilities is None else np.asarray(probabilities)
    logz = -len(tokens)*math.log(64)
    return dict(log_evidence=logz, predictive_probs=probabilities.tolist(),
                log_joint_next=(logz+np.log(probabilities)).tolist())


def cases():
    missing = {'n4_0011', 'n4_0101', 'n4_0110'}
    patterns = [c['tokens'] for c in m.cal.cases() if len(c['tokens']) == 4 and c['id'] not in missing]
    result = []
    for f, family in enumerate(m.old.FAMILIES):
        for index in range(5, 69):
            # Original labels change; t4 spans the fixed cohort's 12 patterns.
            pattern = patterns[(index-5) % len(patterns)]
            labels = [(11 + 9*f + index + 7*k) % 64 for k in range(4)]
            for observed in (1, 4, 16):
                tokens = [labels[0]] if observed == 1 else [labels[i] for i in pattern]
                if observed == 16:
                    tokens = (tokens*4)[:16]
                q = .03 + .7*(index-5)/63
                oracle = np.full(64, (1-q)/63)
                oracle[labels[0]] = q
                result.append(dict(family=family, index=index, observed=observed,
                    tokens=tokens, oracle_probs=oracle.tolist(),
                    transformer_regret_bits={'pooled_seed0': .5 + index/100,
                                             'pooled_seed1': .6 + index/100}))
    return result


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    plan_path = tmp_path / 'plan.json'
    plan = dict(repetitions=32, term_cache_size=4096, worker_margin_seconds=180,
                seconds_per_repetition=1800,
                scheduler=dict(calibration=dict(tasks=52, allocation_minutes=30),
                               population=dict(tasks=768, allocation_minutes=240)))
    plan_path.write_text(json.dumps(plan))
    plan_hash = m.sha(plan_path)
    raw = tmp_path / 'raw'
    raw.mkdir()
    source = raw / 'source.txt'
    source.write_text('Frozen synthetic original-label source identity\n')
    hashes = {'source.txt': m.sha(source)}
    dep = tmp_path / 'dependencies'
    dep.mkdir()
    dependency = dict(manifest_sha256='synthetic', files={}, environment={})
    monkeypatch.setattr(m, 'read_plan', lambda path: (plan, plan_hash, {}))
    monkeypatch.setattr(m, 'dependency_state', lambda dependencies, full: dependency)
    monkeypatch.setattr(m, 'load_sources', lambda raw, parent, indices=None:
        ([c for c in cases() if indices is None or c['index'] in indices], hashes))
    monkeypatch.setattr(m, 'FamilyEvidence', lambda *args, **kwargs: object())
    monkeypatch.setattr(m, 'SharedBaseShortEvidence', lambda: object())
    monkeypatch.setattr(m, 'component_prediction', lambda engine, tokens: component(tokens))
    monkeypatch.setattr(m.cal, 'exact_reference', lambda catalog, tokens: component(tokens))
    out = tmp_path / 'run'
    report = m.prepare(plan_path, raw, dep, out)
    return dict(plan=plan, plan_path=plan_path, plan_hash=plan_hash, raw=raw,
                dep=dep, out=out, prepared=report, prepared_hash=m.sha(out / 'prepared.json'))


def record(fixture, spec, repetition):
    settings = m.ARMS[spec['arm']]
    return dict(component(spec['tokens']), schema='shared-base-predictive-smc-repetition-v1',
        seed=m.seed_for(fixture['plan_hash'], spec, repetition), n_observed=spec['observed'],
        resample_fraction=.75, particles=settings['particles'], temperatures=settings['temperatures'],
        moves=settings['moves'], maximum_conditional_normalization_error=0.,
        effective_sample_sizes=[settings['particles']]*settings['temperatures'])


def save_stage(fixture, stage):
    gate = (m.sha(fixture['out'] / 'calibration/summary.json') if stage == 'population' else None)
    specs = m.tasks(fixture['prepared'], stage)
    for spec in specs:
        records = [record(fixture, spec, j) for j in range(32)]
        report = dict(schema='pooled-mean-task-v1', plan_sha256=fixture['plan_hash'],
            prepared_sha256=fixture['prepared_hash'], task=spec,
            calibration_summary_sha256=gate, records=records, records_sha256=m.digest(records),
            complete=True, status='complete', scientific_reference_validated=False)
        m.save(m.task_path(fixture['out'], stage, spec['task_index']), report)
    return specs


def merge(fixture, stage):
    return m.merge(fixture['plan_path'], stage, fixture['raw'], fixture['dep'], fixture['out'])


def test_production_imports_and_help_startup():
    result = subprocess.run([sys.executable, str(ROOT / 'code/scripts/pooled_mean_population.py'), '--help'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'Pooled mean protocol' in result.stdout
    assert '{prepare,calibration,population,merge}' in result.stdout


def test_full_prepare_calibration_and_population_replay(prepared):
    assert [len(prepared['prepared']['groups'][str(t)]['histories']) for t in (1, 4, 16)] == [192]*3
    assert len(m.tasks(prepared['prepared'], 'calibration')) == 52
    assert len(m.tasks(prepared['prepared'], 'population')) == 768
    save_stage(prepared, 'calibration')
    calibration = merge(prepared, 'calibration')
    assert calibration['complete'] and calibration['passed']
    assert not calibration['scientific_reference_validated']
    assert set(calibration['by_context']) == {'1', '4'}
    for value in calibration['by_context'].values():
        assert 'D_cheap_refined' in value['checks']['exact_short']['arms']
    save_stage(prepared, 'population')
    population = merge(prepared, 'population')
    assert population['complete'] and population['passed'] and population['scientific_reference_validated']
    assert set(population['by_context']) == {'16'}
    published = population['by_context']['16']['publication']
    assert len(published['rows']) == 192
    # Independent truth: every predictor is uniform, so KL is 6 - entropy(q).
    expected = []
    for history in prepared['prepared']['groups']['16']['histories']:
        q = np.asarray(history['oracle_probs'])
        expected.append(6 + float(q @ np.log2(q)))
    assert published['summary']['bayes']['mean'] == pytest.approx(np.mean(expected), abs=1e-13)


def test_missing_tasks_never_form_a_successful_subset(prepared):
    save_stage(prepared, 'calibration')
    m.task_path(prepared['out'], 'calibration', 17).unlink()
    report = merge(prepared, 'calibration')
    assert not report['complete'] and not report['passed']
    assert report['missing_tasks'] == [17]
    assert report['by_context'] == {}


def test_tampered_record_and_source_are_rejected(prepared):
    save_stage(prepared, 'calibration')
    path = m.task_path(prepared['out'], 'calibration', 0)
    item = json.loads(path.read_text())
    item['records'][0]['seed'] += 1
    m.save(path, item)
    with pytest.raises(ValueError, match='Records changed'):
        merge(prepared, 'calibration')
    (prepared['raw'] / 'source.txt').write_text('changed')
    with pytest.raises(ValueError, match='Publication inputs changed'):
        merge(prepared, 'calibration')


def test_saved_calibration_decision_must_replay(prepared):
    save_stage(prepared, 'calibration')
    report = merge(prepared, 'calibration')
    report['by_context']['4']['arms']['A']['bayes']['mean'] += .01
    m.save(prepared['out'] / 'calibration/summary.json', report)
    with pytest.raises(ValueError, match='independently replay'):
        merge(prepared, 'population')


def test_canonical_mapping_preserves_original_label_mixture_and_oracle(monkeypatch):
    probabilities = np.arange(1, 65, dtype=float)
    probabilities /= probabilities.sum()
    fake = component([0], probabilities)
    monkeypatch.setattr(m, 'FamilyEvidence', lambda *args, **kwargs: object())
    monkeypatch.setattr(m, 'SharedBaseShortEvidence', lambda: object())
    monkeypatch.setattr(m, 'component_prediction', lambda engine, tokens: fake)
    monkeypatch.setattr(m.cal, 'exact_reference', lambda catalog, tokens: component(tokens))
    case = cases()[1]
    groups = m.build_prepared([case])
    history = groups['4']['histories'][0]
    order = history['canonical_to_original']
    original_tokens = np.asarray(case['tokens'])
    canonical = groups['4']['references'][history['reference_key']]['tokens']
    np.testing.assert_array_equal(np.asarray(order)[canonical], original_tokens)
    np.testing.assert_allclose(history['fixed_components'][0]['predictive_probs'], probabilities[order])
    np.testing.assert_allclose(history['oracle_probs'], np.asarray(case['oracle_probs'])[order])
    assert sorted(order) == list(range(64))


def test_worker_saves_each_repetition_and_refuses_restart(prepared, monkeypatch):
    spec = m.tasks(prepared['prepared'], 'calibration')[0]
    monkeypatch.setattr(m, 'RankedLSARowEvidence', lambda *args, **kwargs: object())
    monkeypatch.setattr(m, 'SharedBaseMCMC', lambda *args, **kwargs: object())
    monkeypatch.setattr(m.old, 'heartbeat', lambda *args, **kwargs: nullcontext())
    seen = []
    def estimate(sampler, tokens, **kwargs):
        j = len(seen)
        saved = json.loads(m.task_path(prepared['out'], 'calibration', 0).read_text())
        assert len(saved['records']) == j
        assert kwargs['seed'] == m.seed_for(prepared['plan_hash'], spec, j)
        seen.append(j)
        return record(prepared, spec, j)
    monkeypatch.setattr(m, 'estimate', estimate)
    assert m.run_task(prepared['plan_path'], 'calibration', 0, prepared['dep'], prepared['out']) == 0
    assert len(seen) == 32
    saved = json.loads(m.task_path(prepared['out'], 'calibration', 0).read_text())
    assert saved['complete'] and len(saved['records']) == 32
    with pytest.raises(ValueError, match='already attempted'):
        m.run_task(prepared['plan_path'], 'calibration', 0, prepared['dep'], prepared['out'])


def test_timeout_preserves_completed_repetitions(prepared, monkeypatch):
    spec = m.tasks(prepared['prepared'], 'calibration')[0]
    monkeypatch.setattr(m, 'RankedLSARowEvidence', lambda *args, **kwargs: object())
    monkeypatch.setattr(m, 'SharedBaseMCMC', lambda *args, **kwargs: object())
    monkeypatch.setattr(m.old, 'heartbeat', lambda *args, **kwargs: nullcontext())
    seen = []
    def estimate(sampler, tokens, **kwargs):
        if len(seen) == 2:
            raise TimeoutError('synthetic numerical deadline')
        j = len(seen)
        seen.append(j)
        return record(prepared, spec, j)
    monkeypatch.setattr(m, 'estimate', estimate)
    assert m.run_task(prepared['plan_path'], 'calibration', 0, prepared['dep'], prepared['out']) == 75
    saved = json.loads(m.task_path(prepared['out'], 'calibration', 0).read_text())
    assert len(saved['records']) == 2
    assert saved['status'] == 'incomplete' and saved['complete'] is False
    assert saved['records_sha256'] == m.digest(saved['records'])


def test_population_worker_requires_real_calibration_gate(prepared):
    with pytest.raises(FileNotFoundError):
        m.run_task(prepared['plan_path'], 'population', 0, prepared['dep'], prepared['out'])
    assert not m.task_path(prepared['out'], 'population', 0).exists()


def test_dependency_read_plan_uses_explicit_path(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    dep = tmp_path / 'dependency'
    (dep / 'tables/anchors_prod').mkdir(parents=True)
    (dep / 'tables/anchors_prod/anchors.json').write_text('{}')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(dict(files={})))
    called = []
    def read_plan(path):
        called.append(path)
        return dict(dependencies=dict(manifest='manifest.json', sha256=m.sha(manifest))), 'test'
    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(m.cal, 'read_plan', read_plan)
    monkeypatch.setattr(m.cal, 'numerical_environment', lambda *args: {})
    monkeypatch.setattr(m.cal, 'dependencies', lambda *args: {})
    result = m.dependency_state(dep, full=True)
    assert called == [m.cal.PLAN]
    assert result['manifest_sha256'] == m.sha(manifest)
