"""Frozen pairing, fail-closed stages, bounded allocations and saved-work reuse."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import shlex
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'code/scripts/pooled_predictive_population.py'
spec=importlib.util.spec_from_file_location('population_driver',SCRIPT)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_declared_contexts_cases_and_separate_replica_seeds():
    plan,_=p.read_plan()
    assert [len(p.task_specs(plan,s)) for s in p.STAGES]==[36,36,192,2304]
    timing=p.task_specs(plan,'timing')
    assert {x['settings']['particles'] for x in timing}=={256,1024,8192}
    assert all(x['settings']['temperatures']==64 and x['settings']['moves']==4 for x in timing)
    population=p.task_specs(plan,'population')
    for t in p.TIMES:
        rows=[x for x in population if x['observed']==t]
        assert len(rows)==576
        assert {(x['family'],x['index'],x['arm']) for x in rows}=={(f,i,a) for f in p.FAMILIES for i in range(5,69) for a in ('B','C','D')}
    assert p.seed_for(plan,population[0],0)!=p.seed_for(plan,population[1],0)


def test_timing_is_allowed_without_calibration_but_science_fails_closed(tmp_path):
    plan,h=p.read_plan()
    assert not p.scientific_gate(plan,h,tmp_path,'timing',tmp_path/'absent',16)['calibration']['available']
    with pytest.raises(ValueError,match='missing'):
        p.scientific_gate(plan,h,tmp_path,'refine',tmp_path/'absent',16)
    folder=tmp_path/'calibration';folder.mkdir()
    (folder/'summary.json').write_text(json.dumps(dict(schema='pooled-predictive-exact-short-summary-v1',
        complete=True,exact_short_passed=False,identity={'attempt':'stronger'})))
    assert p.calibration_decision(folder)['complete']
    with pytest.raises(ValueError,match='has not passed'):p.calibration_decision(folder,require_pass=True)


def test_model_family_pairing_uses_prefix_not_successor():
    plan,_=p.read_plan();raw=ROOT/'output/multifamily_results_2026-09-16'
    if not raw.exists():pytest.skip('Local frozen paired inputs absent')
    x,q,scores,_=p.load_case(raw,plan,dict(family='product',index=5,observed=1024,development=False))
    assert x.shape==(1024,) and q.shape==(64,) and set(scores)=={'pooled_seed0','pooled_seed1'}
    with pytest.raises(ValueError,match='outside'):
        p.load_case(raw,plan,dict(family='product',index=1,observed=1024,development=False))


def test_context_projection_independent_of_other_timeouts():
    plan,_=p.read_plan();reports={}
    for task in p.task_specs(plan,'refinement_timing'):
        if task['observed']==16:
            reports[task['task_index']]=dict(task=task,complete=True,records=[{'elapsed_seconds':1.}])
    assert p.resource_projection(plan,reports,16)['within_budget']
    assert not p.resource_projection(plan,reports,1024)['within_budget']
    reports.pop(next(iter(reports)))
    assert not p.resource_projection(plan,reports,16)['within_budget']


def test_projection_rejects_infeasible_measured_work():
    plan,_=p.read_plan();reports={}
    for t in p.task_specs(plan,'refinement_timing'):
        reports[t['task_index']]=dict(task=t,complete=True,records=[{'elapsed_seconds':1000.}])
    assert all(not p.resource_projection(plan,reports,t)['within_budget'] for t in p.TIMES)


def test_population_launcher_splits_preserving_all_global_task_ids(tmp_path):
    command=['bash',str(ROOT/'code/transformer/scitas/submit_pooled_predictive_population.sh'),
        '--stage','population','--out',str(tmp_path/'not-created')]
    r=subprocess.run(command,text=True,capture_output=True,check=True)
    commands=[shlex.split(line) for line in r.stdout.splitlines() if line.startswith('sbatch') and '--array=' in line]
    assert len(commands)==3 and all('--array=0-767%21' in c for c in commands)
    mapping=[int(i) for c in commands for i in c[-1].split(',')]
    assert mapping==list(range(2304))
    assert 'afterok:DRY_RUN_JOB' in r.stdout and not (tmp_path/'not-created').exists()
    r=subprocess.run(command+['--observed','16'],text=True,capture_output=True,check=True)
    assert '--array=0-575%64' in r.stdout and '--array=576-' not in r.stdout
    r=subprocess.run(command+['--observed','1024'],text=True,capture_output=True,check=True)
    c=next(shlex.split(line) for line in r.stdout.splitlines() if line.startswith('sbatch') and '--array=' in line)
    assert '--array=0-575%64' in c and list(map(int,c[-1].split(',')))==list(range(1728,2304))


def test_timing_launcher_declares_cost_and_no_population_submission(tmp_path):
    command=['bash',str(ROOT/'code/transformer/scitas/submit_pooled_predictive_population.sh'),
        '--stage','timing','--out',str(tmp_path/'not-created')]
    r=subprocess.run(command,text=True,capture_output=True,check=True)
    report=json.loads(r.stdout.splitlines()[0])
    assert report['tasks']==36 and report['concurrency']==12 and report['maximum_first_attempt_worker_cpu_hours']==18
    assert '--time=00:30:00' in r.stdout and '--array=0-35%12' in r.stdout
    assert not report['automatic_successor_submission']
    assert not (tmp_path/'not-created').exists()


def test_invalid_array_context_rejected_before_any_submit():
    command=['bash',str(ROOT/'code/transformer/scitas/submit_pooled_predictive_population.sh'),
        '--stage','timing','--observed','16','--only-task','35']
    r=subprocess.run(command,text=True,capture_output=True)
    assert r.returncode and 'outside requested contexts' in r.stderr


def test_kernel_wrong_normalization_and_joint_are_not_repaired():
    c=dict(log_evidence=-3.,predictive_probs=[1/64]*64,log_joint_next=[-3.-np.log(64)]*64)
    p.checked_component(c)
    changed=copy.deepcopy(c);changed['predictive_probs'][0]+=.001
    with pytest.raises(ValueError,match='probabilities'):p.checked_component(changed)
    changed=copy.deepcopy(c);changed['log_joint_next'][0]+=.01
    with pytest.raises(ValueError,match='joint/evidence'):p.checked_component(changed)


def test_resource_hard_worker_envelope_with_two_attempts():
    plan,_=p.read_plan()
    hours=sum(len(p.task_specs(plan,s))*plan['stages'][s]['allocation_minutes']/60*2 for s in p.STAGES)
    assert hours==3930 and hours<plan['resource_limits']['maximum_total_projected_cpu_hours']


@pytest.fixture
def fake_scheduler(tmp_path):
    import os
    folder=tmp_path/'bin';folder.mkdir()
    executable=folder/'scheduler'
    executable.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
store=Path(os.environ['POPULATION_FAKE_STATE'])
s=json.loads(store.read_text()) if store.exists() else {'counter':0,'active':True,'calls':[]}
name=Path(sys.argv[0]).name
if name=='sbatch':
 s['counter']+=1;s['calls'].append(sys.argv[1:]);store.write_text(json.dumps(s))
 if s.get('ambiguous'): print('accepted but response unknown')
 else: print(70000+s['counter'])
elif name=='squeue':
 if s['active']: print('70001|RUNNING')
elif name=='sacct':
 assert '--format=JobID,State' in sys.argv, 'Array accounting must use logical JobID, not JobIDRaw'
 if not s.get('missing_accounting'):
  for job in sys.argv[sys.argv.index('--jobs')+1].split(','):
   print(job+'_0|COMPLETED|');print(job+'_1|COMPLETED|')
''')
    executable.chmod(0o755)
    for name in ('sbatch','squeue','sacct'):(folder/name).symlink_to(executable)
    raw=tmp_path/'raw';(raw/'heldout').mkdir(parents=True);(raw/'heldout/manifest.json').write_text('{}')
    dep=tmp_path/'dep';(dep/'tables/anchors_prod').mkdir(parents=True);(dep/'tables/anchors_prod/manifest.json').write_text('{}')
    state=tmp_path/'scheduler.json'
    env=dict(os.environ,PATH=str(folder)+os.pathsep+os.environ['PATH'],POPULATION_FAKE_STATE=str(state))
    command=['bash',str(ROOT/'code/transformer/scitas/submit_pooled_predictive_population.sh'),
        '--stage','timing','--raw',str(raw),'--dependencies',str(dep),'--out',str(tmp_path/'out'),
        '--calibration',str(tmp_path/'calibration'),'--submit']
    return command,env,state,tmp_path/'out'


def test_duplicate_paste_returns_receipt_without_new_allocation(fake_scheduler):
    command,env,state,out=fake_scheduler
    first=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    assert json.loads(state.read_text())['counter']==3
    second=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    assert 'already_submitted' in second.stdout and json.loads(state.read_text())['counter']==3
    receipts=list((out/'submissions').glob('*.json'));assert len(receipts)==1
    r=json.loads(receipts[0].read_text())
    assert r['attempts'][0]['status']=='submitted' and sum(x['maximum_cpu_hours'] for x in r['attempts'][0]['jobs'])==19
    assert [x['mode'] for x in r['attempts'][0]['jobs']]==['gate','run','merge']


def test_active_or_unknown_prior_jobs_block_retry_and_alternate_selection(fake_scheduler):
    command,env,state,_=fake_scheduler
    subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    for extra in (['--retry'],['--only-task','0']):
        r=subprocess.run(command+extra,env=env,text=True,capture_output=True)
        assert r.returncode and 'still active' in r.stderr
    s=json.loads(state.read_text());s.update(active=False,missing_accounting=True);state.write_text(json.dumps(s))
    r=subprocess.run(command+['--retry'],env=env,text=True,capture_output=True)
    assert r.returncode and 'accounting is incomplete' in r.stderr
    assert json.loads(state.read_text())['counter']==3


def test_terminal_retry_is_explicit_and_capped_across_selection_keys(fake_scheduler):
    command,env,state,_=fake_scheduler
    subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    s=json.loads(state.read_text());s['active']=False;state.write_text(json.dumps(s))
    r=subprocess.run(command+['--only-task','0'],env=env,text=True,capture_output=True)
    assert r.returncode and 'explicit --retry' in r.stderr
    subprocess.run(command+['--only-task','0','--retry'],env=env,text=True,capture_output=True,check=True)
    assert json.loads(state.read_text())['counter']==6
    # A different selection cannot evade the two-allocation limit for task0.
    r=subprocess.run(command+['--only-task','0','--only-task','1','--retry'],env=env,text=True,capture_output=True)
    assert r.returncode and 'two-attempt' in r.stderr and json.loads(state.read_text())['counter']==6


def test_uncertain_scheduler_reply_is_never_retried_automatically(fake_scheduler):
    command,env,state,_=fake_scheduler
    state.write_text(json.dumps(dict(counter=0,active=False,calls=[],ambiguous=True)))
    r=subprocess.run(command,env=env,text=True,capture_output=True)
    assert r.returncode and 'numeric job ID' in r.stderr
    r=subprocess.run(command,env=env,text=True,capture_output=True)
    assert r.returncode and 'incomplete or uncertain' in r.stderr and json.loads(state.read_text())['counter']==1


def test_global_allocation_cap_counts_all_prior_gate_worker_merge_jobs(fake_scheduler):
    command,env,state,out=fake_scheduler
    subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    s=json.loads(state.read_text());s['active']=False;state.write_text(json.dumps(s))
    path=next((out/'submissions').glob('*.json'));receipt=json.loads(path.read_text())
    receipt['attempts'][0]['jobs'][0]['maximum_cpu_hours']=4096
    path.write_text(json.dumps(receipt))
    r=subprocess.run(command+['--retry'],env=env,text=True,capture_output=True)
    assert r.returncode and 'allocation-hour cap' in r.stderr and json.loads(state.read_text())['counter']==3


def test_complete_population_context_can_be_submitted_within_cap(fake_scheduler):
    command,env,state,out=fake_scheduler
    command[command.index('--stage')+1]='population'
    subprocess.run(command+['--observed','1024'],env=env,text=True,capture_output=True,check=True)
    receipt=json.loads(next((out/'submissions').glob('*.json')).read_text())
    jobs=receipt['attempts'][0]['jobs']
    assert sum(j['maximum_cpu_hours'] for j in jobs)==433
    worker=next(j for j in jobs if j['mode']=='run')
    assert worker['indices']==list(range(1728,2304))
    calls=json.loads(state.read_text())['calls'];worker_call=next(c for c in calls if any(x.startswith('--array=') for x in c))
    assert '--array=0-575%64' in worker_call and list(map(int,worker_call[-1].split(',')))==worker['indices']


def test_partial_context_merge_exit_uses_requested_contexts(monkeypatch,tmp_path):
    report=dict(passed=False,by_context={str(t):dict(passed=t==16) for t in p.TIMES})
    monkeypatch.setattr(p,'merge',lambda *args:report)
    base=['population','merge','--stage','population','--raw',str(tmp_path),
        '--dependencies',str(tmp_path),'--out',str(tmp_path),'--calibration',str(tmp_path)]
    monkeypatch.setattr(sys,'argv',base+['--observed','16']);assert p.main()==0
    monkeypatch.setattr(sys,'argv',base+['--observed','16','--observed','1024']);assert p.main()==3
