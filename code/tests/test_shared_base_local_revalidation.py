"""The local arithmetic replay retains the complete, fixed scientific cohort."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import shared_base_local_revalidation as local
import shared_base_curve_audit as auditor
import inspect_shared_base_depth_diagnostics as diagnostic
from revalidate_shared_base_depth import adjust_depth_gate
from test_shared_base_depth_revalidation import example, synchronize


def test_fixed_scope_includes_all_cases_depths_and_both_rules():
    keys=local.expected_keys()
    assert len(keys)==2048
    assert all(sum(k.startswith(f'a1_L{d}/') for k in keys)==512 for d in (0,1,4,8))
    assert all(sum(k.endswith(f'alpha{o}') for k in keys)==1024 for o in (16,32))


def make_rows(tmp_path):
    rows=[];tasks=[]
    for key in sorted(local.expected_keys()):
        setting,index,_,order=key.split('/')
        task=dict(key=key,setting=setting,depth=int(setting[4:]),index=int(index[4:]),time=64,order=int(order[5:]))
        audit=dict(index=task['index'],time=64,order=task['order'],status='ready',ready=True,
                   draws_per_chain=4096,minimum_probability_ess=999.,scalar_ess=999.)
        row=dict(task=task,original_audit=audit,corrected=dict(audit=audit,change=dict(old_status='ready',new_status='ready')))
        local.write_json(local.case_path(tmp_path,task),row)
        tasks.append(task);rows.append(row)
    return dict(cases=tasks,output=str(tmp_path),plan_id='test'),rows


def test_incomplete_fixed_population_never_releases_any_mean(tmp_path,monkeypatch):
    plan,rows=make_rows(tmp_path);calls=[]
    def aggregate(values,ids,orders):
        calls.append((values,ids,orders))
        return dict(complete=len(values)==512,passed=len(values)==512,reported_mean_bayes_regret_bits=1.)
    monkeypatch.setattr(auditor,'aggregate_time',aggregate)
    result=local.summarize_rows(plan,rows[:-1])
    assert not result['complete'] and not result['all_cohorts_passed']
    assert len(result['missing_keys'])==1
    assert all(g['numerical_audit']['reported_mean_bayes_regret_bits'] is None for g in result['groups'])
    assert all(ids==list(range(256)) and orders==[16,32] for _,ids,orders in calls)


def test_summary_keeps_unresolved_and_uses_corrected_readiness(tmp_path,monkeypatch):
    plan,rows=make_rows(tmp_path)
    rows[0]['corrected']['audit']=dict(rows[0]['original_audit'],ready=False,status='needs_more_draws')
    rows[0]['corrected']['change']['new_status']='needs_more_draws'
    monkeypatch.setattr(auditor,'aggregate_time',lambda values,ids,orders:
        dict(complete=True,passed=all(v['ready'] for v in values),reported_mean_bayes_regret_bits=None))
    result=local.summarize_rows(plan,rows)
    assert result['complete'] and not result['all_cohorts_passed']
    assert result['counts']=={'needs_more_draws':1,'ready':2047}
    assert result['changed_cases']==[rows[0]['task']['key']]
    with pytest.raises(ValueError,match='Repeated or undeclared'):
        local.summarize_rows(plan,rows+[rows[0]])


def test_changed_raw_file_is_rejected(tmp_path):
    path=tmp_path/'checkpoint';path.write_bytes(b'original')
    task=dict(key='example',file_sha256={str(path):local.sha(path)})
    local.check_task_files(task)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Frozen reference files changed'):
        local.check_task_files(task)


def test_resume_replays_raw_depths_and_rejects_changed_acceptance(tmp_path,monkeypatch):
    reference,audit,depths=example();audit.update(draws_per_chain=64)
    reference['estimate']['diagnostics']['rhat_depth'][0]=None
    reference['record']['mixing_passed']=False;depths[0]['original']['rhat']=None
    synchronize(reference,audit);audit.update(status='needs_more_draws',ready=False)
    path=tmp_path/'reference.json';local.write_json(path,reference)
    task=dict(key='example',reference=str(path),checkpoint='unused',file_sha256={str(path):local.sha(path)})
    row=dict(task=task,launch_sha256='launch',no_sampling=True,original_audit=audit,
             depth_diagnostics=depths,corrected=adjust_depth_gate(reference,audit,depths))
    row['row_id']=local.fingerprint(row)
    monkeypatch.setattr(diagnostic,'summarize_depths',lambda path,estimate:deepcopy(depths))
    local.validate_row(row,task,'launch')
    bad=deepcopy(depths);bad[0]['rescaled']['rhat']=1.051
    monkeypatch.setattr(diagnostic,'summarize_depths',lambda path,estimate:bad)
    with pytest.raises(ValueError,match='Saved correction differs'):
        local.validate_row(row,task,'launch')


def test_runtime_rejects_numerical_perturbation(monkeypatch):
    monkeypatch.setenv('PMM_PHI_BIAS','0.001')
    with pytest.raises(ValueError,match='Numerical perturbation'):
        local.configure_runtime(dict(environment={}))


def test_pool_completes_beyond_previous_four_times_sixteen_recycling_boundary(monkeypatch):
    monkeypatch.delenv('PMM_PHI_BIAS',raising=False)
    monkeypatch.delenv('PMM_PHI_WAVE',raising=False)
    with local.audit_pool(4,dict(environment={})) as pool:
        jobs=[pool.submit(abs,-i) for i in range(80)]
        assert [job.result(timeout=15) for job in jobs]==list(range(80))
