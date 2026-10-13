"""Rounding-only readback repair, immutable provenance and complete pilot gates."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import recover_shared_base_posterior_audit as recovery
from prepare_shared_base_posterior_cluster import fingerprint, sha, atomic_json, task_list


def law():
    return np.full(64,1/64,dtype=np.float64)


def test_rounding_allowance_is_ulp_and_absolute_bounded():
    saved=law(); rounded=np.nextafter(saved,np.inf)
    detail=recovery.discrepancy(rounded,saved)
    assert detail['max_ulp']==1 and detail['unequal_coordinates']==64
    assert recovery.discrepancy(saved,saved)['max_abs']==0
    five=saved.copy()
    for _ in range(5):
        five=np.nextafter(five,np.inf)
    with pytest.raises(ValueError,match='rounding-only'):
        recovery.discrepancy(five,saved)
    bad=saved.copy();bad[0]+=2e-15;bad[1]-=2e-15
    with pytest.raises(ValueError,match='rounding-only'):
        recovery.discrepancy(bad,saved)
    with pytest.raises(ValueError,match='shape or dtype'):
        recovery.discrepancy(saved[:-1],saved[:-1])
    with pytest.raises(ValueError,match='Unnormalized'):
        recovery.discrepancy(saved*2,saved*2)


@pytest.fixture
def models(tmp_path):
    for name in ('d64','d64_seed1'):
        folder=tmp_path/'models'/name;folder.mkdir(parents=True)
        arrays=folder/'seq_000000.npz'
        np.savez(arrays,predictive_probs=law()[None,:])
        atomic_json(folder/'seq_000000.json',dict(arrays_file=arrays.name,arrays_sha256=sha(arrays),
                     prefixes=[256],predictive_probs=[law().tolist()]))
    return tmp_path


def test_adapter_runs_independent_loader_then_returns_canonical_and_restores(models):
    calls=[]
    def independent(*args):
        calls.append(args)
        return {name:np.nextafter(law(),np.inf) for name in ('d64','d64_seed1')}
    auditor=SimpleNamespace(load_models=independent)
    before={str(p):sha(p) for p in models.rglob('*') if p.is_file()}
    observations=[]
    with recovery.canonical_model_loader(auditor,observations):
        result=auditor.load_models(None,models,None,0,256)
        assert all(np.array_equal(p,law()) for p in result.values())
    assert len(calls)==1 and auditor.load_models is independent
    assert len(observations)==2 and all(d['max_ulp']==1 for d in observations)
    assert before=={str(p):sha(p) for p in models.rglob('*') if p.is_file()}


def test_adapter_never_bypasses_independent_hash_or_causal_failure(models):
    def failed(*args):
        raise ValueError('causal_alignment_checks')
    auditor=SimpleNamespace(load_models=failed)
    with pytest.raises(ValueError,match='causal_alignment'):
        with recovery.canonical_model_loader(auditor,[]):
            auditor.load_models(None,models,None,0,256)
    assert auditor.load_models is failed


def test_adapter_rejects_corrupt_saved_file_even_if_loader_is_stubbed(models):
    array=models/'models/d64/seq_000000.npz'
    np.savez(array,predictive_probs=(law()+1e-18)[None,:],unexpected=np.array([1]))
    auditor=SimpleNamespace(load_models=lambda *a:{n:law() for n in ('d64','d64_seed1')})
    original=auditor.load_models
    with pytest.raises(ValueError,match='Saved prediction file changed'):
        with recovery.canonical_model_loader(auditor,[]):
            auditor.load_models(None,models,None,0,256)
    assert auditor.load_models is original


def test_adapter_rejects_more_than_roundoff_after_independent_loader(models):
    p=law();p[0]+=1e-14;p[1]-=1e-14
    auditor=SimpleNamespace(load_models=lambda *a:{n:p for n in ('d64','d64_seed1')})
    with pytest.raises(ValueError,match='rounding-only'):
        with recovery.canonical_model_loader(auditor,[]):
            auditor.load_models(None,models,None,0,256)


@pytest.fixture
def original_files(tmp_path):
    task=task_list()[0]
    campaign=dict(plan_sha256='plan',output=str(tmp_path/'original'),inputs=str(tmp_path/'inputs'))
    old=dict(campaign=campaign)
    plan=dict(tasks=task_list(),code_sha256={'code/scripts/shared_base_curve_audit.py':'audit-code'})
    paths=recovery.original.task_paths(campaign,task)
    for key in ('reference','checkpoint','moments'):
        paths[key].parent.mkdir(parents=True,exist_ok=True);paths[key].write_bytes(key.encode())
    audit=dict(task=task,plan_sha256='plan',audit_code_sha256='audit-code',
        result=dict(index=0,time=256,order=16,status='invalid',error=recovery.KNOWN_ERROR),
        raw_sha256={str(paths[key]):sha(paths[key]) for key in ('reference','checkpoint','moments')})
    atomic_json(paths['audit'],audit)
    receipt=dict(identity=dict(plan_sha256='plan',task=task,deployment_sha256=fingerprint(campaign)),
        sampling_complete=True,worker_exit_code=0,status='audit_failed',audit_sha256=sha(paths['audit']),runtime={'python':'test'})
    atomic_json(paths['receipt'],receipt)
    return old,plan,task,paths


def test_original_snapshot_rejects_changed_raw_and_task_identity(original_files):
    old,plan,task,paths=original_files
    before={str(p):sha(p) for p in paths.values() if p.is_file()}
    snapshot=recovery.original_snapshot(old,plan,task)
    assert snapshot['files']==before
    receipt=json.loads(paths['receipt'].read_text());receipt['identity']['task']=dict(task,time=1024)
    atomic_json(paths['receipt'],receipt)
    with pytest.raises(ValueError,match='task identity'):
        recovery.original_snapshot(old,plan,task)
    receipt['identity']['task']=task;atomic_json(paths['receipt'],receipt)
    paths['checkpoint'].write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='raw reference changed'):
        recovery.original_snapshot(old,plan,task)


def test_dryrun_from_other_cwd_never_submits_or_writes(tmp_path,monkeypatch):
    old_launch=tmp_path/'old-launch.json';old_launch.write_text('{}')
    old=dict(mode='pilot',selected_tasks=list(range(32)),campaign=dict(output=str(tmp_path/'original'),
             inputs=str(tmp_path/'inputs'),plan_sha256='plan'))
    monkeypatch.setattr(recovery.original,'read_launch',lambda p:(old,{}))
    monkeypatch.setattr(recovery.subprocess,'run',lambda *a,**k:pytest.fail('Submission in dry run'))
    monkeypatch.chdir(tmp_path)
    args=argparse.Namespace(original_launch=old_launch,original_out=tmp_path/'original',out=tmp_path/'recovery',
            only_task=[0],merge=False,concurrency=32,venv=tmp_path/'venv',submit=False)
    recovery.submit(args)
    assert not args.out.exists()
    launch,path,command=recovery.build_launch(args)
    assert '--array=0%32' in command and 'recovery.sbatch' in command[-3]
    assert all('shared_base_curve_reference.py' not in item for item in command)
    assert launch['context']['no_sampling'] is True


def test_missing_pilot_never_releases_mean(tmp_path,monkeypatch):
    context=dict(output=str(tmp_path))
    launch=dict(context=context)
    monkeypatch.setattr(recovery,'read_recovery',lambda p:(launch,{},dict(tasks=task_list())))
    result=recovery.merge('fake')
    assert result['complete'] is False and len(result['rows'])==32
    assert all(g['numerical_audit']['reported_mean_bayes_regret_bits'] is None for g in result['groups'])
    assert result['all_numerical_checks_passed'] is False


def test_original_mixing_gate_not_relaxed_by_recovery_merge(tmp_path,monkeypatch):
    context=dict(output=str(tmp_path))
    tasks=task_list()
    monkeypatch.setattr(recovery,'read_recovery',lambda p:(dict(context=context),{},dict(tasks=tasks)))
    for task in tasks:
        atomic_json(tmp_path/'tasks'/f"{task['id']}.json",dict(status='unresolved',result={},
                    model_readback=[],runtime={}))
    # Exercise the unchanged aggregator using genuine precision record shape.
    sys.path.insert(0,str(recovery.ROOT/'code/transformer'))
    from shared_base_precision_cohort import deterministic_precision
    from shared_base_curve_audit import aggregate_time
    record=dict(case_id=0,sampling_seed=None,precision=deterministic_precision(law(),law()),
                full_prior_valid=True,provenance_valid=True,mixing_passed=False,row_grid_passed=True,
                per_sequence_precision_passed=False)
    def validate(report,context,old,plan,task):
        return dict(index=task['index'],time=task['time'],order=task['order'],status='needs_more_draws',
                    record=dict(record,case_id=task['index']))
    monkeypatch.setattr(recovery,'validate_recovered',validate)
    result=recovery.merge('fake')
    assert result['complete'] is True and result['all_numerical_checks_passed'] is False
    assert all(not group['numerical_audit']['passed'] for group in result['groups'])
    assert all(group['numerical_audit']['reported_mean_bayes_regret_bits'] is None for group in result['groups'])


def test_batch_script_is_audit_only():
    text=(recovery.ROOT/'code/transformer/scitas/shared_base_posterior_recovery.sbatch').read_text()
    assert 'recover_shared_base_posterior_audit.py run' in text
    assert 'shared_base_curve_reference.py' not in text
    assert '#SBATCH --no-requeue' in text
