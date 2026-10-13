"""Depth-only gate correction cannot change scores or unrelated acceptance gates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import revalidate_shared_base_depth as correction
from prepare_shared_base_posterior_cluster import task_list
from shared_base_curve_audit import diagnostic_gates
from shared_base_precision_cohort import deterministic_precision


def example():
    d=dict(chains=4,draws=64,rhat_probs=[1.]*64,rhat_depth=[1.]*9,
           rhat_alpha_and_loglik=[1.,1.],ess_probs=[1024.]*64,max_conditional_probability_sum_error=0.)
    estimate=dict(diagnostics=d,predictive_probs=[1/64]*64,mcse_probs=[0.]*64,
                  posterior_depth=[1/9]*9,mcse_depth=[0.]*9)
    record=dict(case_id=0,sampling_seed=None,precision=deterministic_precision(np.ones(64)/64,np.ones(64)/64),
                full_prior_valid=True,provenance_valid=True,mixing_passed=True,row_grid_passed=True,
                per_sequence_precision_passed=False)
    reference=dict(estimate=estimate,record=record)
    audit=dict(index=0,time=256,order=16,status='ready',ready=True,record=deepcopy(record),gates=diagnostic_gates(estimate))
    rows=[dict(depth=i,chains=4,draws=64,mean=1/9,min=.01,max=.9,
               original=dict(rhat=1.),rescaled=dict(status='finite',rhat=1.001)) for i in range(9)]
    return reference,audit,rows


def synchronize(reference,audit):
    audit['gates']=diagnostic_gates(reference['estimate'])
    audit['record']=deepcopy(reference['record'])


def test_cutoff_correction_preserves_every_numerical_score_and_historical_flag():
    ref,audit,rows=example()
    ref['estimate']['diagnostics']['rhat_depth'][2]=None;rows[2]['original']['rhat']=None
    ref['record']['mixing_passed']=False;synchronize(ref,audit)
    audit.update(ready=False,status='needs_more_draws')
    old=deepcopy((ref,audit,rows));result=correction.adjust_depth_gate(ref,audit,rows)
    assert result['audit']['ready'] and result['audit']['record']['mixing_passed']
    assert result['audit']['record']['per_sequence_precision_passed'] is False
    assert result['audit']['record']['precision']==ref['record']['precision']
    assert (ref,audit,rows)==old
    allowed={'mixing_passed'}
    assert {k for k in ref['record'] if result['audit']['record'][k]!=ref['record'][k]}<=allowed
    assert result['change']['historical_per_sequence_precision_passed'] is False


@pytest.mark.parametrize('kind',['probability_rhat','alpha_rhat','ess','mcse','scalar','grid','jensen'])
def test_non_depth_failure_persists(kind):
    ref,audit,rows=example();d=ref['estimate']['diagnostics']
    if kind=='probability_rhat': d['rhat_probs'][0]=1.06
    if kind=='alpha_rhat': d['rhat_alpha_and_loglik'][0]=None
    if kind=='ess': d['ess_probs'][0]=199.
    if kind=='mcse': ref['estimate']['mcse_probs'][0]=.006
    if kind=='scalar': ref['record']['precision']['scalar_mixing_passed']=False
    if kind=='grid': ref['record']['row_grid_passed']=False
    if kind=='jensen': ref['record']['precision']['jensen_local_check_passed']=False
    synchronize(ref,audit)
    result=correction.adjust_depth_gate(ref,audit,rows)
    assert result['audit']['ready'] is False and result['audit']['status']=='needs_more_draws'
    assert result['audit']['record']['precision']==ref['record']['precision']


def test_rescaling_can_reject_a_previously_passing_reference():
    ref,audit,rows=example();rows[2]['rescaled']['rhat']=1.08
    result=correction.adjust_depth_gate(ref,audit,rows)
    assert result['change']['old_status']=='ready' and result['change']['new_status']=='needs_more_draws'
    assert not result['audit']['record']['mixing_passed']


@pytest.mark.parametrize('status,same_value,passes',[
    ('constant_uninformative',True,True),('constant_uninformative',False,False),
    ('different_constant_chains',False,False),('numerically_unresolved',False,False)])
def test_constant_fallback_requires_actual_exact_common_constant(status,same_value,passes):
    ref,audit,rows=example()
    rows[2]['rescaled']=dict(status=status,rhat=None)
    if same_value: rows[2]['min']=rows[2]['max']=1/9
    result=correction.adjust_depth_gate(ref,audit,rows)
    assert result['audit']['ready'] is passes
    assert result['change']['corrected_depth_rhat'][2]==(1. if passes else None)


def test_incomplete_or_reordered_depths_rejected():
    ref,audit,rows=example()
    with pytest.raises(ValueError,match='nine ordered depths'):
        correction.adjust_depth_gate(ref,audit,rows[:-1])
    with pytest.raises(ValueError,match='nine ordered depths'):
        correction.adjust_depth_gate(ref,audit,rows[::-1])


def make_aggregate_rows():
    rows=[]
    for task in task_list():
        ref,audit,depths=example()
        ref['record']['case_id']=task['index'];synchronize(ref,audit)
        audit.update(index=task['index'],time=task['time'],order=task['order'])
        rows.append(dict(task=task,corrected=correction.adjust_depth_gate(ref,audit,depths)))
    return rows


def test_complete_aggregation_uses_original_gate_and_missing_order_blocks_release():
    rows=make_aggregate_rows();tasks=task_list()
    complete=correction.aggregate_rows(rows,tasks)
    assert complete['complete'] and complete['all_numerical_checks_passed']
    assert len(complete['groups'])==8
    missing=correction.aggregate_rows(rows[:-1],tasks)
    assert missing['complete'] is False and missing['all_numerical_checks_passed'] is False
    assert all(g['numerical_audit']['reported_mean_bayes_regret_bits'] is None for g in missing['groups'])
    rows[-1]['corrected']['audit']['record']['mixing_passed']=False
    rows[-1]['corrected']['audit'].update(ready=False,status='needs_more_draws')
    failed=correction.aggregate_rows(rows,tasks)
    assert failed['complete'] and not failed['all_numerical_checks_passed']
    assert failed['groups'][-1]['numerical_audit']['reported_mean_bayes_regret_bits'] is None
    with pytest.raises(ValueError,match='Duplicate or undeclared'):
        correction.aggregate_rows(rows+[rows[0]],tasks)


def test_launcher_dryrun_never_submits_or_writes(tmp_path,monkeypatch):
    launch=tmp_path/'original.json';launch.write_text('{}')
    recovered=dict(context=dict(output=str(tmp_path/'recovery')),venv=str(tmp_path/'venv'))
    old=dict(campaign=dict(output=str(tmp_path/'original'),inputs=str(tmp_path/'inputs')))
    monkeypatch.setattr(correction.diagnostic,'locate_launch',lambda p:launch)
    monkeypatch.setattr(correction.recovery,'read_recovery',lambda p:(recovered,old,{}))
    monkeypatch.setattr(correction.subprocess,'run',lambda *a,**k:pytest.fail('Unexpected submission'))
    monkeypatch.chdir(tmp_path)
    args=argparse.Namespace(recovery=tmp_path/'recovery',out=tmp_path/'new',submit=False)
    correction.submit(args)
    assert not args.out.exists()
    spec,_,command=correction.build_launch(args)
    assert spec['task_indices']==list(range(32)) and spec['depths']==list(range(9))
    assert spec['no_sampling'] is True and not any('--array' in item for item in command)
    batch=(correction.ROOT/correction.OWN_FILES[-1]).read_text()
    assert '#SBATCH --time=00:30:00' in batch and '#SBATCH --mem=12G' in batch
    assert 'shared_base_curve_reference.py' not in batch


def test_continuation_advice_is_historical_only():
    ref,audit,rows=example()
    audit.update(todo='Extend this same reference',suggested_next_draws=131072,draw_deficit=65536,
                 estimated_minimum_draws_for_ess=100000)
    result=correction.adjust_depth_gate(ref,audit,rows)
    assert 'todo' not in result['audit'] and 'suggested_next_draws' not in result['audit']
    assert result['change']['historical_continuation_guidance']['todo']=='Extend this same reference'


@pytest.fixture
def replay_fixture(tmp_path,monkeypatch):
    from shared_base_chain_runner import _hash_array
    import hashlib
    from prepare_shared_base_posterior_cluster import atomic_json,sha,fingerprint
    ref,audit,_=example();task=task_list()[0]
    root=tmp_path/'recovery';receipt_path=root/'tasks'/f"{task['id']}.json"
    reference=tmp_path/'reference.json';checkpoint=tmp_path/'checkpoint.npz'
    arrays={f'depths_{i}':np.full((64,9),1/9,dtype=np.float64) for i in range(4)}
    metadata=json.dumps(dict(format='shared_base_chain_run_v1',config=dict(chains=4),
        array_hashes={name:_hash_array(v) for name,v in arrays.items()}))
    np.savez(checkpoint,**arrays,metadata=np.asarray(metadata),
             metadata_sha256=np.asarray(hashlib.sha256(metadata.encode()).hexdigest()))
    atomic_json(reference,ref)
    paths=dict(reference=reference,checkpoint=checkpoint)
    snapshot=lambda *a:dict(files={str(p):sha(p) for p in paths.values()})
    runtime={'test_runtime':True}
    receipt=dict(original=snapshot(),runtime=runtime)
    atomic_json(receipt_path,receipt)
    launch=dict(context=dict(output=str(root)))
    monkeypatch.setattr(correction.recovery.original,'task_paths',lambda *a:paths)
    monkeypatch.setattr(correction.recovery,'original_snapshot',snapshot)
    monkeypatch.setattr(correction.recovery,'validate_recovered',lambda *a:audit)
    rows=correction.diagnostic.summarize_depths(checkpoint,ref['estimate'])
    row=dict(task=task,launch_sha256='spec',runtime=runtime,recovery_receipt_sha256=sha(receipt_path),
        original_file_sha256=receipt['original']['files'],depth_diagnostics=rows,
        corrected=correction.adjust_depth_gate(ref,audit,rows),elapsed_seconds=1.)
    from validate_shared_base_mcmc import json_safe
    row=json_safe(row);row['row_id']=fingerprint(row)
    return dict(row=row,task=task,runtime=runtime,launch=launch,reference=reference,
                checkpoint=checkpoint,receipt=receipt_path)


def replay(data):
    correction.check_row(data['row'],data['task'],'spec',data['runtime'],data['launch'],{'campaign':{}}, {})


def test_replay_is_idempotent_and_preserves_raw_artifacts(replay_fixture):
    data=replay_fixture
    paths=[data[k] for k in ('reference','checkpoint','receipt')]
    before={str(p):correction.sha(p) for p in paths};row=deepcopy(data['row'])
    replay(data);replay(data)
    assert before=={str(p):correction.sha(p) for p in paths} and data['row']==row


def test_replay_rejects_changed_raw_file(replay_fixture):
    data=replay_fixture
    data['checkpoint'].write_bytes(b'not a saved chain')
    with pytest.raises((ValueError,OSError,KeyError)):
        replay(data)


def test_replay_rejects_changed_recovery_receipt(replay_fixture):
    data=replay_fixture
    value=json.loads(data['receipt'].read_text());value['unexpected']=True
    data['receipt'].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='inputs changed'):
        replay(data)


def test_replay_recomputes_even_if_tampered_row_self_hash_is_updated(replay_fixture):
    data=replay_fixture
    data['row']['depth_diagnostics'][2]['rescaled']=dict(status='finite',rhat=1.01)
    data['row']['row_id']=correction.fingerprint({k:v for k,v in data['row'].items() if k!='row_id'})
    with pytest.raises(ValueError,match='raw depth arrays'):
        replay(data)
