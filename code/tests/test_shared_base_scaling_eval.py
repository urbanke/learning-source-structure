"""Paired scoring and provenance guards, including actual tiny trained models."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transformer'))
import shared_base_scaling_eval as e


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return e.sha_file(path)


def test_paired_scoring_preserves_seeds_and_never_invents_long_bayes():
    new = np.zeros((4, len(e.TIMES), 2))
    old = np.zeros_like(new)
    new[:, :, 0] = np.array([1., 2., 3., 4.])[:, None]
    new[:, :, 1] = 2 * new[:, :, 0]
    old[:, :, 0] = new[:, :, 0] + np.array([1., -1., 2., -2.])[:, None]
    old[:, :, 1] = new[:, :, 1] + 3
    bayes = {(4, 16): np.array([.2, .1, .4, .3])}
    rows = e.comparison_rows(4, new, old, bayes)
    first = e.one(rows, seed=0, time=16)
    expected = np.std([1., -1., 2., -2.], ddof=1) / 2
    assert first['improvement_over_baseline']['mean'] == 0
    assert first['improvement_over_baseline']['standard_error'] == pytest.approx(expected)
    assert e.one(rows, seed=1, time=16)['improvement_over_baseline']['mean'] == 3
    assert first['larger_minus_bayes']['mean'] == pytest.approx(2.25)
    for row in rows:
        if row['time'] != 16:
            assert row['bayes_available'] is False
            assert 'bayes_regret' not in row and 'larger_minus_bayes' not in row


@pytest.fixture
def audited(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'N', 2)
    cases = [dict(index=i, order=o, ready=True, status='ready',
                 recomputed_final_states=True, recomputed_finer_grid=True,
                 gates={'ess': True, 'mixing': True},
                 record={'precision': {'bayes_regret_bits': i + o / 100.}})
             for i in range(2) for o in (16, 32)]
    report = dict(schema='shared-base-curve-audit-v1', all_requested_times_validated=True,
        depth=0, alpha=1., times=[64], orders=[16,32], n_sequences=2,
        provenance={'cohort_sha256': 'cohort'},
        rows=[dict(time=64, cohort_validation={'passed':True}, case_audits=cases)])
    path = tmp_path / 'audit.json'
    return path, report


def test_audited_scores_use_refined_rule_and_ordered_original_indices(audited):
    path, report = audited
    report['rows'][0]['case_audits'].reverse()
    digest = write(path, report)
    actual = e.audited_values(path, digest, 0, 64, 'cohort')
    np.testing.assert_array_equal(actual, [.32, 1.32])


@pytest.mark.parametrize('change', ['hash', 'duplicate', 'missing', 'gate', 'cohort', 'time'])
def test_changed_or_partial_audit_is_rejected(audited, change):
    path, report = audited
    original_hash = write(path, report)
    if change == 'hash':
        report['rows'][0]['case_audits'][0]['record']['precision']['bayes_regret_bits'] += 1
    elif change == 'duplicate':
        report['rows'][0]['case_audits'][-1] = report['rows'][0]['case_audits'][0]
    elif change == 'missing':
        report['rows'][0]['case_audits'].pop()
    elif change == 'gate':
        report['rows'][0]['case_audits'][0]['gates']['ess'] = False
    elif change == 'cohort':
        report['provenance']['cohort_sha256'] = 'changed'
    else:
        report['times'] = [256]
    digest = write(path, report)
    with pytest.raises(ValueError):
        e.audited_values(path, original_hash if change == 'hash' else digest, 0, 64, 'cohort')


@pytest.fixture
def trained_pair(tmp_path, monkeypatch):
    import torch
    import train_shared_base_scaling as trainer
    from model import GPT
    torch.set_num_threads(1)
    cfg = dict(trainer.PRODUCTION, batch=1, steps=1, warmup=0, n_layer=1,
        n_head=1, d_model=8, val_n=1, val_chunk=1, val_every=1, log_every=1,
        ckpt_every=1, weight_steps=[1])
    plan = dict(schema='shared-base-scaling-plan-v1', seeds=[0,1],
        source=trainer.SOURCE, config=cfg, n_params=GPT(64,1024,1,1,8).n_params(),
        evaluation_files_sha256={})
    path = tmp_path / 'tiny_plan.json'
    write(path, plan)
    digest = e.sha_file(path)
    # The numerical engine really trains both models; only the production-size
    # plan admission is replaced by this explicit tiny test plan.
    monkeypatch.setattr(trainer, 'read_plan', lambda p: (deepcopy(plan), digest))
    runs = []
    for seed in (0,1):
        out = tmp_path / f'seed{seed}'
        out.mkdir()
        actual = trainer.planned_config(plan, seed)
        assert trainer._train_locked(actual, digest, out, resume=False, device='cpu',
                                     workers=0, max_steps=None) == 0
        runs.append(out)
    return path, plan, runs


def test_actual_trainer_final_weights_are_accepted(trained_pair):
    path, plan, runs = trained_pair
    values = e.validate_runs(path, plan, list(reversed(runs)))
    assert [value['seed'] for _, value in values] == [0, 1]
    assert all(value['completed_sequences'] == 1 for _, value in values)


@pytest.mark.parametrize('change', ['incomplete', 'wrong_prior', 'stale_config', 'wrong_weights', 'same_seed'])
def test_final_checkpoint_guards(trained_pair, change):
    import torch
    path, plan, runs = trained_pair
    if change == 'same_seed':
        with pytest.raises(ValueError, match='distinct'):
            e.validate_runs(path, plan, [runs[0], runs[0]])
        return
    run = runs[0]
    if change == 'incomplete':
        target = run / 'status.json'; record = e.read_json(target); record['complete'] = False
    elif change == 'wrong_prior':
        target = run / 'config.json'; record = e.read_json(target); record['alpha'] = [0,0]
    elif change == 'stale_config':
        target = run / 'config.json'; record = e.read_json(target); record['config']['steps'] = 3
    else:
        target = run / 'model_1.pt'
        record = torch.load(target, weights_only=True)
        record['completed_sequences'] = 0
        torch.save(record, target)
        with pytest.raises(ValueError, match='checkpoint'):
            e.validate_final_checkpoint(path, run)
        return
    write(target, record)
    with pytest.raises(ValueError):
        e.validate_final_checkpoint(path, run)


@pytest.mark.parametrize('complete_time64', [False, True])
def test_tiny_actual_training_to_evaluation_pipeline(trained_pair, tmp_path, monkeypatch, complete_time64):
    from eval_shared_base_models import sources
    path, plan, runs = trained_pair
    monkeypatch.setattr(e, 'N', 2)
    monkeypatch.setattr(e, 'read_plan', lambda p: deepcopy(plan))
    records = sources(tmp_path / 'test_sources', 2, 2026091777, create=True)
    frozen = {depth: records for depth in e.DEPTHS}
    baseline = {depth: np.full((2,len(e.TIMES),2), 2.) for depth in e.DEPTHS}
    bayes = {(depth,t): np.array([.1,.2]) for depth in e.DEPTHS for t in (1,4,16)}
    bayes.update({(depth,64): np.array([.1,.2]) for depth in (0,1)})
    monkeypatch.setattr(e, 'load_sources', lambda *args: frozen)
    monkeypatch.setattr(e, 'load_saved_scores', lambda *args: (baseline, bayes))
    release = tmp_path / 'time64.json'
    release_hash = write(release, {'complete': True})
    receipt = {'files_sha256': {str(release): release_hash}, 'role': 'additional evaluation reference'}
    monkeypatch.setattr(e, 'load_complete_time64', lambda *args: (
        {(depth,64): np.array([.1,.2]) for depth in e.DEPTHS}, receipt))
    out = tmp_path / 'evaluation'
    plan_hash = e.sha_file(path)
    result = e.evaluate(path, list(reversed(runs)), tmp_path, out, device='cpu', batch=2,
                        time64_complete=release if complete_time64 else None)
    assert result['seed_order'] == [0,1] and len(result['rows']) == 48
    assert len([r for r in result['rows'] if r['bayes_available']]) == (32 if complete_time64 else 28)
    assert result['additional_evaluation_releases'] == ([receipt] if complete_time64 else [])
    assert result['plan_sha256'] == e.sha_file(path) == plan_hash
    assert result['evaluation_files_sha256'] == plan['evaluation_files_sha256']
    assert (out / 'comparison.json').is_file()
    for depth in e.DEPTHS:
        with np.load(out / f'a1_L{depth}' / 'paired_scores.npz') as a:
            assert a['larger_regret_bits'].shape == (2,6,2)
            assert np.isfinite(a['larger_regret_bits']).all()
            assert 1024 not in a['bayes_times']
            assert (64 in a['bayes_times']) == (complete_time64 or depth in (0,1))
    with pytest.raises(ValueError, match='new output'):
        e.evaluate(path, runs, tmp_path, out, device='cpu', batch=2)


@pytest.fixture
def time64_release(tmp_path, monkeypatch):
    """Small independent vectors exercise import/pairing; exporter gates have their own tests."""
    import export_shared_base_time64 as exported
    import shared_base_curve_audit as audit
    monkeypatch.setattr(e, 'N', 2)
    original = tmp_path / 'original.json'
    write(original, {})
    monkeypatch.setattr(e, 'RELEASES', dict(e.RELEASES, original=str(original)))
    sources, baseline, bayes = {}, {}, {}
    receipts, references, cohorts, raw = [], {}, [], {}
    for depth in e.DEPTHS:
        folder = tmp_path / f'a1_L{depth}'
        cohort_hash = write(folder / 'cohort.json', {'depth': depth})
        protocol_hash = write(folder / 'protocol.json', {'depth': depth, 'prior': e.PRIOR})
        sources[depth] = []
        values = []
        case_hashes = {}
        for index in range(2):
            tokens = np.full(1025, index, dtype=np.int16)
            oracle = np.full((64,64), 1/64)
            prediction = np.full(64, (1-(.05+.005*depth+.01*index))/63)
            prediction[0] = .05+.005*depth+.01*index
            regret = float(np.sum(e.rel_entr(oracle[0],prediction))/np.log(2))
            values.append(regret)
            source = dict(source_id=f'{depth}-{index}', tokens=tokens.tolist(),
                tokens_sha256=e.sha_array(tokens,'<i2'), oracle={'transition_probs':oracle.tolist()})
            source_hash = write(folder / 'data' / f'seq_{index:06d}.json',source)
            sources[depth].append({'record':source, 'file_sha256':source_hash})
            identity = dict(index=index,prefix=64,source_id=source['source_id'],
                source_file_sha256=source_hash,protocol_sha256=protocol_hash,
                tokens_sha256=source['tokens_sha256'],prefix_tokens_sha256=e.sha_array(tokens[:64],'<i2'))
            for order in (16,32):
                key = f'a1_L{depth}/seq_{index:06d}/t0064/alpha{order}'
                path = folder / 'references' / f'seq{index}_order{order}.json'
                digest = write(path,{'identity':{'inputs':identity}})
                raw[str(path)] = digest
                task = dict(key=key,depth=depth,index=index,order=order,time=64,
                    source_folder=str(folder),reference=str(path),file_sha256={str(path):digest})
                case_hashes[key] = f'case-{depth}-{index}-{order}'
                references[key] = {'case_receipt_sha256':case_hashes[key]}
                receipts.append(dict(task=task,corrected={'audit':dict(depth=depth,
                    record={'precision':dict(predictive_probs=prediction.tolist(),bayes_regret_bits=regret)})}))
        bayes[depth,64] = np.asarray(values)
        baseline[depth] = np.empty((2,len(e.TIMES),2))
        baseline[depth][:,:,0] = (bayes[depth,64]+np.array([.1,.3]))[:,None]
        baseline[depth][:,:,1] = (bayes[depth,64]+np.array([.4,.2]))[:,None]
        matched = {}
        for seed,name in enumerate(e.BASELINES):
            b = baseline[depth][:,e.TIMES.index(64),seed]
            matched[name] = dict(transformer_regret=e.moments(b),bayes_regret=e.moments(values),
                transformer_minus_bayes=e.moments(b-values))
        cohorts.append(dict(depth=depth,provenance={'cohort_sha256':cohort_hash,'protocol_sha256':protocol_hash},
            case_receipts=case_hashes,matched_comparisons=matched,numerical_audit={'depth':depth,'passed':True}))
    origin = dict(schema='corrected-source',references=references)
    source_path = tmp_path / 'source_summary.json';source_digest = write(source_path,origin)
    bound = {str(source_path):source_digest}
    release = dict(source_summary=str(source_path),source_summary_schema='corrected-source',
        source_plan_id='source-plan',cohorts=cohorts)
    path = tmp_path / 'complete.json';write(path,release)
    validation = dict(inputs=bound,raw_file_sha256=raw,raw_files_rehashed_at_export=True)
    validation_path=tmp_path/'validation.json';write(validation_path,validation)
    monkeypatch.setattr(exported,'validate_export',lambda *args:([path,validation_path],release))
    def read_sources(*args):
        os.environ['TIME64_IMPORT_TEST_SETTING'] = 'temporary-audit-setting'
        return origin, {'plan_id':'source-plan'}, receipts, bound
    monkeypatch.setattr(exported,'source_rows',read_sources)
    def aggregate(rows,indices,orders):
        assert len(rows)==4 and indices==[0,1] and orders==[16,32]
        return {'depth':rows[0]['depth'],'passed':True}
    monkeypatch.setattr(audit,'aggregate_time',aggregate)
    monkeypatch.setattr(exported,'checked_cohort_gate',lambda value: e.require(value['passed'],'Failed cohort'))
    return dict(path=path,root=tmp_path,sources=sources,baseline=baseline,bayes=bayes,
                release=release,validation=validation,validation_path=validation_path,
                origin=origin,receipts=receipts)


def import_time64(fixture):
    f=fixture
    return e.load_complete_time64(f['path'],f['root'],f['sources'],f['baseline'],
                                 {key:value for key,value in f['bayes'].items() if key[0] in (0,1)})


def test_complete_time64_uses_ordered_refined_scores_and_restores_environment(time64_release,monkeypatch):
    f=time64_release
    f['receipts'].reverse()
    monkeypatch.setenv('TIME64_IMPORT_TEST_SETTING','original-setting')
    result,receipt=import_time64(f)
    assert set(result)=={(depth,64) for depth in e.DEPTHS}
    for key,value in result.items():
        np.testing.assert_array_equal(value,f['bayes'][key])
    assert os.environ['TIME64_IMPORT_TEST_SETTING']=='original-setting'
    assert receipt['mandatory_replacements_checked'] is True
    assert receipt['raw_files_rehashed_before_evaluation'] is True
    assert receipt['raw_file_count']==16
    assert set(receipt['original_time64_max_absolute_difference_bits'])=={'0','1'}
    e.check_supplement_files(receipt)
    f['path'].write_text('{}')
    with pytest.raises(ValueError,match='changed during evaluation'):
        e.check_supplement_files(receipt)


@pytest.mark.parametrize('change',['missing','duplicate','case_hash','raw_hash_map','source_identity',
    'prediction','kl','paired_summary','aggregate','old_score','source_plan'])
def test_complete_time64_rejects_partial_unpaired_or_changed_inputs(time64_release,change):
    f=time64_release
    if change=='missing':f['receipts'].pop()
    elif change=='duplicate':f['receipts'][-1]=deepcopy(f['receipts'][0])
    elif change=='case_hash':f['release']['cohorts'][0]['case_receipts'].popitem()
    elif change=='raw_hash_map':
        f['validation']['raw_file_sha256'].popitem();write(f['validation_path'],f['validation'])
    elif change=='source_identity':f['sources'][0][0]['file_sha256']='changed'
    elif change=='prediction':f['receipts'][1]['corrected']['audit']['record']['precision']['predictive_probs']=[1.]
    elif change=='kl':f['receipts'][1]['corrected']['audit']['record']['precision']['bayes_regret_bits']+=.1
    elif change=='paired_summary':f['release']['cohorts'][0]['matched_comparisons']['d64']['transformer_minus_bayes']['mean']+=.1
    elif change=='aggregate':f['release']['cohorts'][0]['numerical_audit']['passed']=False
    elif change=='old_score':f['bayes'][0,64][0]+=.1
    else:f['release']['source_plan_id']='another-plan'
    with pytest.raises(ValueError):import_time64(f)
