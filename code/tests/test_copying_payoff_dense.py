"""Coverage, identity and bounded dispatch checks independent of the sampler."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/scripts'))
import copying_payoff_dense as dense
from prepare_copying_payoff_references import inventory_tokens,canonicalize,alphabet_map
from shared_base_short_evidence import _patterns


@pytest.fixture
def inventory():
    patterns=list(_patterns(5))[:38]
    cohorts={name:np.array([patterns[(i+shift)%38] for i in range(256)])
             for shift,name in enumerate(('a1_L0','a1_L2','a1_L3'))}
    refs,uses=inventory_tokens(cohorts,times=[5])
    return dict(histories_total=768,histories_per_condition=256,depths=[0,2,3],
                references=list(refs.values()),uses=uses)


def test_complete_t5_wave_preserves_every_history_and_two_orders(inventory):
    tasks,uses=dense.wave_tasks(inventory,5,{})
    assert len(tasks)==76 and len(uses)==768
    assert {t['order'] for t in tasks}=={16,32}
    assert len({t['seed'] for t in tasks})==76
    assert {u['condition'] for u in uses}=={'a1_L0','a1_L2','a1_L3'}
    assert tasks==dense.wave_tasks(inventory,5,{})[0]


@pytest.mark.parametrize('mutation',['missing','duplicate','context','labelmap','order','prefix','refcount'])
def test_no_selected_history_or_order_collapse(inventory,mutation):
    if mutation=='missing':inventory['uses'].pop()
    elif mutation=='duplicate':inventory['uses'][0]=deepcopy(inventory['uses'][1])
    elif mutation=='context':inventory['uses'][0]['queried_canonical_label']=99
    elif mutation=='labelmap':inventory['uses'][0]['canonical_to_original_seen']=[]
    elif mutation=='order':inventory['references'][3]['tokens'].reverse()
    elif mutation=='prefix':inventory['references'][0]['reference_key']='wrong'
    else:inventory['references'].pop()
    with pytest.raises((ValueError,KeyError)):dense.wave_tasks(inventory,5,{})


def test_reuse_preserves_seed_target_and_draw_budget(inventory):
    tasks,_=dense.wave_tasks(inventory,5,{})
    old=deepcopy(tasks[0]);old['seed']=987654321
    parent={old['id']:dict(task=old)}
    updated,_=dense.wave_tasks(inventory,5,parent)
    assert updated[0]['kind']=='reuse' and updated[0]['seed']==987654321
    assert updated[1:]==tasks[1:]
    parent[old['id']]['task']['target_draws']=4096
    with pytest.raises(ValueError,match='target'):dense.wave_tasks(inventory,5,parent)


def test_canonical_use_preserves_true_query_row_under_relabelling():
    # Distinct histories with equal counts may have different query contexts.
    a,la=canonicalize([19,41,19,41,7]);b,lb=canonicalize([19,41,7,41,19])
    assert sorted(a)==sorted(b) and a!=b
    oracle=np.arange(64*64).reshape(64,64)
    for tokens,labels in (([19,41,19,41,7],la),([19,41,7,41,19],lb)):
        mapped=oracle[tokens[-1],alphabet_map(labels)]
        assert mapped[0]==oracle[tokens[-1],19]


def test_completion_never_approves_population_precision_or_forecast(inventory):
    tasks,_=dense.wave_tasks(inventory,5,{})
    plan=dict(tasks=tasks,plan_id='frozen',observed=5)
    reports=[dict(task=t,plan_id='frozen',complete=True,audit_passed=True) for t in tasks]
    result=dense.coverage(plan,reports)
    assert result['complete'] and result['all_individual_audits_passed']
    assert not any(result[k] for k in ('aggregate_precision_checked','aggregate_precision_passed',
                                     'population_reference_approved','payoff_forecast_approved'))
    partial=dense.coverage(plan,reports[:-1])
    assert not partial['complete'] and len(partial['missing_task_ids'])==1
    with pytest.raises(ValueError):dense.coverage(plan,reports+[reports[0]])


def test_bound_parent_rejects_changed_file(tmp_path):
    p=tmp_path/'input.json';p.write_text('unchanged')
    bound=dense.bound(p);dense.verify_bound(bound)
    p.write_text('changed!')
    with pytest.raises(ValueError):dense.verify_bound(bound)


@pytest.fixture
def dispatch_setup(tmp_path,monkeypatch):
    task=dict(task_index=0,id='one',kind='fresh')
    plan=dict(plan_id='bound',tasks=[task],python=sys.executable,environment={},
              resources=dict(total_worker_seconds=900,minimum_free_disk_bytes=1))
    monkeypatch.setattr(dense,'load_plan',lambda out:plan)
    monkeypatch.setattr(dense,'lock',lambda path:nullcontext())
    monkeypatch.setattr(dense.pilot,'environment',lambda plan:{})
    monkeypatch.setattr(dense,'summary',lambda out,verify_arrays=True:dict(done=True))
    return tmp_path,plan,dense.paths(tmp_path,task)


def test_attempt_is_never_retried_even_if_previous_timeout(dispatch_setup,monkeypatch):
    out,plan,p=dispatch_setup
    dense.write(p['attempt'],dict(reserved_seconds=900,status='time_limit'))
    monkeypatch.setattr(dense.subprocess,'Popen',lambda *a,**k:pytest.fail('Silent retry'))
    assert dense.dispatch(out)==dict(done=True)


def test_dispatch_kills_at_fixed_bound_and_records_timeout(dispatch_setup,monkeypatch):
    out,plan,p=dispatch_setup;waits=[];signals=[]
    class Process:
        pid=12345
        def wait(self,timeout):
            waits.append(timeout)
            if timeout==900:raise subprocess.TimeoutExpired('worker',timeout)
            return -15
        def poll(self):return -15
    monkeypatch.setattr(dense.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(dense.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    dense.dispatch(out)
    receipt=dense.read(p['attempt'])
    assert waits==[900,5] and signals==[(12345,dense.signal.SIGTERM)]
    assert receipt['status']=='time_limit' and receipt['exit_code']==75 and receipt['reserved_seconds']==900


def test_dispatch_keeps_numerical_failure_unresolved_not_ready(dispatch_setup,monkeypatch):
    out,plan,p=dispatch_setup
    class Process:
        pid=12345
        def wait(self,timeout):
            dense.write(p['report'],dict(audit_passed=False));return 0
        def poll(self):return 0
    monkeypatch.setattr(dense.subprocess,'Popen',lambda *a,**k:Process())
    dense.dispatch(out)
    assert dense.read(p['attempt'])['status']=='unresolved'
    assert dense.read(p['attempt'])['exit_code']==0


def test_disk_reserve_stops_before_sampling(dispatch_setup,monkeypatch):
    out,plan,p=dispatch_setup
    monkeypatch.setattr(dense.shutil,'disk_usage',lambda path:type('Usage',(),{'free':0})())
    monkeypatch.setattr(dense.subprocess,'Popen',lambda *a,**k:pytest.fail('Started without reserve'))
    with pytest.raises(ValueError,match='disk reserve'):dense.dispatch(out)
    assert not p['attempt'].exists()
