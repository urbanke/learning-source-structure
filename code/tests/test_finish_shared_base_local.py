"""Frozen-parent continuation, draw-target admission and mandatory overlays."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finish_shared_base_local as f
import shared_base_local_revalidation as audit


def write(path,value):
    f.campaign.atomic_json(path,value)
    return f.campaign.digest(path)


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    key=f.campaign.job_key(4,3,64,16)
    raw=tmp_path/'original'/'a1_L4'/'alpha16';raw.mkdir(parents=True)
    reference=raw/'seq_000003_t0064.json'
    checkpoint=raw/'seq_000003_t0064_chains.npz';checkpoint.write_bytes(b'original-rng-and-samples')
    moments=raw/'seq_000003_t0064_moments.npz';moments.write_bytes(b'original-moments')
    report=dict(identity={'source':'frozen','rng_seed':123},reference_id='ref',
        history=[{'draws':65536}],estimate={'diagnostics':{'draws':65536}},
        checkpoint=str(checkpoint),checkpoint_sha256=f.campaign.digest(checkpoint),
        moments_sha256=f.campaign.digest(moments))
    write(reference,report)
    task=dict(key=key,setting='a1_L4',depth=4,index=3,time=64,order=16,
        source_folder=str(tmp_path/'input'/'a1_L4'),reference_root=str(tmp_path/'original'),
        reference=str(reference),checkpoint=str(checkpoint),moments=str(moments),reference_id='ref')
    task['file_sha256']=f.raw_hashes(task)
    original=dict(source_root=str(tmp_path/'input'),references=str(tmp_path/'original'),
        global_paths=[],cases={'a1_L4/3':{'files':[]}},
        worker=dict(python=sys.executable,script='frozen_worker.py',burn=16384,
                    initial_draws=4096,max_draws=65536,seed=20261006,environment={}))
    original_path=tmp_path/'original_plan.json';write(original_path,original)
    out=tmp_path/'followup';out.mkdir()
    parent_row=dict(task=task,corrected={'audit':{'status':'needs_more_draws','ready':False}},
        launch_sha256='oldplan')
    parent_row['row_id']=f.campaign.fingerprint(parent_row)
    parent_receipt=tmp_path/'parent_receipt.json';write(parent_receipt,parent_row)
    plan=dict(schema=f.SCHEMA,output=str(out),plan_id='frozenplan',original_plan=str(original_path),
        audit_summary='oldsummary',audit_summary_sha256='oldsummarysha',
        selected={key:dict(parent_task=task,parent_hashes=f.raw_hashes(task),parent_reference_id='ref',
            parent_row_id=parent_row['row_id'],previous_draws=65536,target_draws=131072,
            estimated_peak_bytes=f.slot_bytes(131072,16),reason='Remaining corrected ESS failure')},
        effective_cases={key:dict(task=task,row_id=parent_row['row_id'],case_receipt=str(parent_receipt),
            case_receipt_sha256=f.campaign.digest(parent_receipt))},
        limits={'max_workers':12,'maximum_global_workers':12,'memory_budget_bytes':f.MEMORY_BUDGET,
                'disk_reserve_bytes':f.DISK_RESERVE})
    monkeypatch.setattr(f,'read_plan',lambda *args:deepcopy(plan))
    monkeypatch.setattr(f.campaign,'verify_assets',lambda *args:None)
    return plan,key,report


def test_forced_initial_and_max_draws_preserve_original_worker_settings(experiment):
    plan,key,_=experiment
    command=f.worker_command(plan,key)
    values={command[i]:command[i+1] for i in range(2,len(command),2)}
    assert values['--initial-draws']==values['--max-draws']=='131072'
    assert values['--burn']=='16384' and values['--seed']=='20261006'
    assert values['--prefix']=='64' and values['--alpha-order']=='16'
    assert values['--proposal']=='iid' and values['--steps']=='64'
    assert values['--out']==str(Path(plan['output'])/'references')


def test_parent_copy_preserves_all_bytes_and_cannot_silently_replace(experiment):
    plan,key,_=experiment
    before=deepcopy(plan['selected'][key]['parent_hashes'])
    paths=f.prepare_case(plan,key)
    for name,original in f.file_map(plan['selected'][key]['parent_task']).items():
        assert paths[name].read_bytes()==original.read_bytes()
    assert f.raw_hashes(plan['selected'][key]['parent_task'])==before
    assert f.prepare_case(plan,key)==paths
    Path(plan['selected'][key]['parent_task']['checkpoint']).write_bytes(b'changed-parent')
    with pytest.raises(ValueError,match='parent files changed'):
        f.prepare_case(plan,key)


def child_output(plan,key,report):
    paths=f.prepare_case(plan,key)
    paths['checkpoint'].write_bytes(b'continued-saved-rng-and-samples')
    paths['moments'].write_bytes(b'continued-moments')
    child=deepcopy(report)
    child.update(checkpoint=str(paths['checkpoint'].resolve()),
                 checkpoint_sha256=f.campaign.digest(paths['checkpoint']),
                 moments_sha256=f.campaign.digest(paths['moments']))
    child['history'].append({'draws':131072})
    child['estimate']['diagnostics']['draws']=131072
    write(paths['reference'],child)
    return paths,child


@pytest.mark.parametrize('change',['identity','history','draws','checksum'])
def test_child_must_preserve_identity_history_and_exact_target(experiment,change):
    plan,key,report=experiment
    paths,child=child_output(plan,key,report)
    assert f.verify_child(plan,key)==child
    if change=='identity':child['identity']['rng_seed']=124
    elif change=='history':child['history'][0]['draws']=32768
    elif change=='draws':child['estimate']['diagnostics']['draws']=65536
    else:child['checkpoint_sha256']='wrong'
    write(paths['reference'],child)
    with pytest.raises(ValueError):f.verify_child(plan,key)


def test_corrected_raw_audit_can_accept_historical_worker_exit_two(experiment,monkeypatch):
    plan,key,report=experiment
    before=f.raw_hashes(plan['selected'][key]['parent_task'])
    def worker(*args,**kwargs):
        child_output(plan,key,report)
        return SimpleNamespace(returncode=2)
    def raw_audit(task,digest):
        row=dict(task=task,launch_sha256=digest,
                 corrected={'audit':{'status':'ready','ready':True}})
        row['row_id']=f.campaign.fingerprint(row)
        return row
    monkeypatch.setattr(f.subprocess,'run',worker)
    monkeypatch.setattr(audit,'audit_case',raw_audit)
    monkeypatch.setattr(audit,'configure_runtime',lambda *args:None)
    monkeypatch.setattr(audit,'validate_row',lambda *args,**kwargs:None)
    receipt=f.run_case(plan['output'],key)
    assert receipt['worker_exit_code']==2 and receipt['state']=='ready'
    assert f.checked_result(plan,key)[0]==receipt
    assert f.raw_hashes(plan['selected'][key]['parent_task'])==before
    with pytest.raises(ValueError,match='already finished'):
        f.run_case(plan['output'],key)


def test_failed_worker_cannot_become_a_completed_replacement(experiment,monkeypatch):
    plan,key,_=experiment
    monkeypatch.setattr(f.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=1))
    with pytest.raises(ValueError,match='exit1'):
        f.run_case(plan['output'],key)
    assert not f.task_paths(plan,key)['receipt'].exists()


def test_missing_replacement_never_falls_back_to_parent(experiment,monkeypatch):
    plan,key,_=experiment
    captured={}
    def aggregate(spec,rows):
        captured.update(rows=rows,tasks=spec['cases'])
        return dict(complete=False,all_cohorts_passed=False,reported_mean_bayes_regret_bits=None)
    monkeypatch.setattr(audit,'summarize_rows',aggregate)
    result=f.export_overlay(plan)
    assert result['complete'] is False
    assert result['references'][key]['complete'] is False
    assert 'case_receipt' not in result['references'][key]
    assert captured['rows']==[] and len(captured['tasks'])==1
    summary=f.read(Path(plan['output'])/'summary.json')
    assert not summary['complete'] and summary['reported_mean_bayes_regret_bits'] is None


def test_memory_admission_grows_with_draws_and_never_claims_twelve_huge_jobs_fit():
    assert f.slot_bytes(262144,32)>f.slot_bytes(131072,32)==f.slot_bytes(131072,16)
    assert f.slot_bytes(131072,32)==3*1024**3
    assert f.slot_bytes(262144,32)==4.5*1024**3
    assert 6*f.slot_bytes(131072,32)==f.MEMORY_BUDGET
    assert 12*f.slot_bytes(262144,32)>f.MEMORY_BUDGET
    assert f.slot_bytes(2**21,32)>f.MEMORY_BUDGET


def test_only_original_fixed_time64_cases_are_accepted():
    task=dict(setting='a1_L4',index=255,time=64,order=16)
    assert f.task_key(task)=='a1_L4/seq_000255/t0064/alpha16'
    for field,value in [('time',256),('order',64),('index',256),('setting','a1_L2')]:
        wrong=dict(task,**{field:value})
        with pytest.raises(ValueError):f.task_key(wrong)


def test_unknown_external_workers_pause_new_dispatch(experiment,monkeypatch):
    plan,key,_=experiment
    out=Path(plan['output'])
    write(out/'status.json',dict(plan_id=plan['plan_id'],jobs={key:dict(state='pending',attempts=[])}))
    monkeypatch.setattr(f,'external_active',lambda *args:1)
    monkeypatch.setattr(f,'export_overlay',lambda *args:{})
    monkeypatch.setattr(f.shutil,'disk_usage',lambda *args:SimpleNamespace(free=100_000_000_000))
    times=iter([0.,0.,2.])
    monkeypatch.setattr(f.time,'monotonic',lambda:next(times))
    monkeypatch.setattr(f.time,'sleep',lambda *args:None)
    def forbidden(*args,**kwargs):raise AssertionError('Must not launch alongside unknown external memory use')
    monkeypatch.setattr(f.subprocess,'Popen',forbidden)
    result=f.run(out,max_hours=1/3600,poll_seconds=.001)
    assert result['jobs'][key]['state']=='pending'
    assert result['stop_reason']=='dispatch time budget reached'


def test_retained_raw_change_blocks_final_release(experiment,monkeypatch):
    plan,key,_=experiment
    plan['selected']={}
    Path(plan['effective_cases'][key]['task']['moments']).write_bytes(b'changed-unselected-moments')
    with pytest.raises(ValueError,match='Retained raw'):
        f.export_overlay(plan)
    assert not (Path(plan['output'])/'summary.json').exists()


def test_final_release_replays_new_depth_arithmetic(experiment,monkeypatch):
    plan,key,report=experiment
    def worker(*args,**kwargs):
        child_output(plan,key,report)
        return SimpleNamespace(returncode=0)
    def raw_audit(task,digest):
        row=dict(task=task,launch_sha256=digest,corrected={'audit':{'status':'ready','ready':True}})
        row['row_id']=f.campaign.fingerprint(row)
        return row
    replays=[]
    monkeypatch.setattr(f.subprocess,'run',worker)
    monkeypatch.setattr(audit,'audit_case',raw_audit)
    monkeypatch.setattr(audit,'configure_runtime',lambda *args:None)
    monkeypatch.setattr(audit,'validate_row',lambda *args,**kwargs:replays.append(kwargs['replay']))
    monkeypatch.setattr(audit,'summarize_rows',lambda *args:dict(complete=True,all_cohorts_passed=True))
    f.run_case(plan['output'],key)
    assert f.export_overlay(plan)['complete']
    assert replays==[True]
    summary=f.read(Path(plan['output'])/'summary.json')
    assert summary['all_cohorts_passed'] and summary['no_sampling'] is False


@pytest.mark.parametrize('target', [65536, 131073, 2**21])
def test_prepare_rejects_nonmonotone_nonblock_or_oversized_targets(experiment,tmp_path,monkeypatch,target):
    plan,key,_=experiment
    row=f.read(plan['effective_cases'][key]['case_receipt'])
    row['corrected']['audit']['reference_id']='ref'
    receipt=plan['effective_cases'][key]['case_receipt']
    summary={'references':{key:{'case_receipt':receipt,'case_receipt_sha256':f.campaign.digest(receipt)}}}
    monkeypatch.setattr(f,'checked_summary',lambda *args:(summary,{key:row}))
    targets=tmp_path/'targets.json'
    write(targets,dict(schema=f.TARGET_SCHEMA,cases=[dict(key=key,target_draws=target,reason='fixed diagnostic')]))
    with pytest.raises(ValueError,match='Target|memory allowance'):
        f.prepare(tmp_path/'audit_summary.json',targets,tmp_path/'campaign',tmp_path/'new_output')
    assert not (tmp_path/'new_output').exists()
