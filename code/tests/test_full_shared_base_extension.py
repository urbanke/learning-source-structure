"""No sampling: isolation, frozen ancestry, submission and cohort acceptance."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import full_shared_base_extension as ext


@pytest.fixture
def plan():return ext.read(ext.PLAN)


@pytest.mark.parametrize('change',[
    lambda p:p.update(target_draws=2097152),
    lambda p:p.update(tolerances_changed=True),
    lambda p:p['resources'].update(worker_seconds=28800),
    lambda p:p.update(tasks=p['tasks'][:2]),
    lambda p:p['tasks'][0]['task'].update(depth=4),
    lambda p:p['tasks'][0].update(parent_task_index=1792),
    lambda p:p.update(parent_launch_id='0'*64),
    lambda p:p['code_sha256'].pop(ext.FILES[0]),
])
def test_reject_changed_scope(plan,change):
    change(plan)
    with pytest.raises(ValueError):ext.validate_plan(plan)


def test_plan_code_matches(plan):
    ext.validate_plan(plan)
    assert {n:ext.sha(ext.ROOT/n) for n in ext.FILES}==plan['code_sha256']


def fake_parent(plan,spec):
    task=spec['task']
    reference=dict(setting=task['setting'],index=task['index'],time=task['time'],order=task['order'],
        reference_id=spec['parent_reference_id'],identity=dict(master_sampling_seed=2026091780,burn=16384,kernel='original'),
        estimate=dict(diagnostics=dict(draws=524288,chains=4,thin=1)),history=[dict(draws=524288)])
    value=dict(index=task['index'],time=1024,order=task['order'],reference_id=spec['parent_reference_id'],
        draws_per_chain=524288,report_sha256=spec['parent_report_sha256'],ready=False,status='needs_more_draws',
        gates=dict(split_rhat_below_1_05=True,probability_ess_at_least_200=False,
                   maximum_probability_mcse_at_most_0_005=True,raw_normalization=True))
    spec['parent_audit_fingerprint']=ext.fingerprint(value)
    receipt=dict(task=task,launch_id=plan['parent_launch_id'],status='unresolved')
    audit=dict(task=task,launch_id=plan['parent_launch_id'],corrected=dict(audit=value))
    return receipt,audit,reference


def test_parent_frozen_probability_ess_only(plan):
    spec=plan['tasks'][0];r,a,ref=fake_parent(plan,spec)
    assert ext.validate_parent_bindings(plan,spec,r,a,ref)==a['corrected']['audit']
    a['corrected']['audit']['gates']['split_rhat_below_1_05']=False
    spec['parent_audit_fingerprint']=ext.fingerprint(a['corrected']['audit'])
    with pytest.raises(ValueError,match='probability-ESS'):ext.validate_parent_bindings(plan,spec,r,a,ref)


@pytest.mark.parametrize('which', ['status','audit','seed','burn','draws','identity'])
def test_reject_parent_drift(plan,which):
    spec=plan['tasks'][0];r,a,ref=fake_parent(plan,spec)
    if which=='status':r['status']='running'
    elif which=='audit':a['corrected']['audit']['draws_per_chain']=1048576
    elif which=='seed':ref['identity']['master_sampling_seed']+=1
    elif which=='burn':ref['identity']['burn']+=1
    elif which=='draws':ref['estimate']['diagnostics']['draws']-=1
    else:ref['reference_id']='0'*64
    with pytest.raises(ValueError):ext.validate_parent_bindings(plan,spec,r,a,ref)


def test_continuation_and_fixed_worker_command(plan,tmp_path):
    spec=plan['tasks'][0];_,_,ref=fake_parent(plan,spec);child=deepcopy(ref)
    child['history'].append(dict(draws=1048576));child['estimate']['diagnostics']['draws']=1048576
    ext.validate_continuation(child,ref,spec['task'])
    command=ext.worker_command(dict(inputs='/frozen/input'),tmp_path,spec['task'])
    for key,value in [('--seed','2026091780'),('--burn','16384'),('--steps','64'),('--proposal','iid'),
                      ('--initial-draws','1048576'),('--max-draws','1048576')]:
        assert command[command.index(key)+1]==value
    assert command[command.index('--out')+1]==str(tmp_path/'references')
    child['identity']['kernel']='changed'
    with pytest.raises(ValueError):ext.validate_continuation(child,ref,spec['task'])


@pytest.mark.parametrize('change',[
    lambda c:c['history'][0].update(draws=262144),
    lambda c:c['estimate']['diagnostics'].update(draws=524288),
    lambda c:c['estimate']['diagnostics'].update(chains=2),
    lambda c:c.update(order=16),
])
def test_reject_noncontinuation(plan,change):
    spec=plan['tasks'][0];_,_,ref=fake_parent(plan,spec);child=deepcopy(ref)
    child['history'].append(dict(draws=1048576));child['estimate']['diagnostics']['draws']=1048576
    change(child)
    with pytest.raises(ValueError):ext.validate_continuation(child,ref,spec['task'])


def test_writer_claim_is_exclusive_and_released(plan,tmp_path):
    task=plan['tasks'][0]['task']
    with ext.writer_claim(tmp_path,task):
        with pytest.raises(RuntimeError,match='Writer claim'): 
            with ext.writer_claim(tmp_path,task):pass
    with ext.writer_claim(tmp_path,task):pass


@pytest.fixture
def overlays(plan):
    rows=[];replacements=[]
    for spec in plan['tasks']:
        _,a,_=fake_parent(plan,spec);v=a['corrected']['audit']
        rows.append(dict(task=spec['task'],status='unresolved',audit=v))
        changed=deepcopy(v);changed.update(draws_per_chain=1048576,ready=True,status='ready')
        replacements.append(dict(task=spec['task'],status='ready',audit=changed,
                                 parent_audit_fingerprint=spec['parent_audit_fingerprint']))
    # An unrelated record must not be changed or selected away.
    rows.append(dict(task=dict(id='other',depth=8,index=3,time=256,order=16),status='running'))
    snapshot=dict(launch_id=plan['parent_launch_id'],rows=rows,groups=[dict(old_approval=True)])
    overlay=dict(schema='full-shared-base-extension-overlay-v1',complete=True,replacements=replacements,
                 parent_launch_id=plan['parent_launch_id'],plan_sha256='1'*64,context_sha256='2'*64,
                 parent_snapshot_sha256=plan['selection_snapshot_sha256'])
    return snapshot,overlay


def test_overlay_changes_only_three_copied_rows_and_drops_stale_approval(overlays):
    snapshot,overlay=overlays;before=deepcopy(snapshot)
    result=ext.apply_overlay(snapshot,overlay)
    assert snapshot==before and result['rows'][-1]==snapshot['rows'][-1]
    assert result['counts']==dict(ready=3,running=1)
    assert result['groups']==[] and result['all_numerical_checks_passed'] is False
    assert result['status']=='requires_group_revalidation'


@pytest.mark.parametrize('change',[
    lambda o:o.update(complete=False),
    lambda o:o['replacements'].pop(),
    lambda o:o['replacements'][0].update(parent_audit_fingerprint='0'*64),
    lambda o:o['replacements'][0]['audit'].update(draws_per_chain=524288),
    lambda o:o['replacements'][0]['audit'].update(index=196),
    lambda o:o['replacements'][0].update(status='unresolved'),
])
def test_overlay_rejects_unbound_replacement(overlays,change):
    snapshot,overlay=overlays;change(overlay)
    with pytest.raises(ValueError):ext.apply_overlay(snapshot,overlay)


def test_failed_extension_is_retained(overlays):
    snapshot,overlay=overlays
    r=overlay['replacements'][1];r.update(status='unresolved');r['audit'].update(ready=False,status='needs_more_draws')
    result=ext.apply_overlay(snapshot,overlay)
    assert result['counts']==dict(ready=2,unresolved=1,running=1)


def test_parent_copy_is_verified_and_resume_does_not_recopy(plan,tmp_path,monkeypatch):
    spec=plan['tasks'][0];task=spec['task'];parentroot=tmp_path/'parent';out=tmp_path/'extension'
    original=ext.paths(parentroot,task);dest=ext.paths(out,task)
    for key in ('reference','checkpoint','moments','receipt','audit'):
        original[key].parent.mkdir(parents=True,exist_ok=True);original[key].write_text(key+' original bytes')
    lock=parentroot/'claims'/(task['id']+'.lock');lock.parent.mkdir();lock.touch()
    hashes={str(original[k]):ext.sha(original[k]) for k in ('reference','checkpoint','moments')}
    bundle=(original,dict(runtime=dict(version='fixed')),dict(raw_sha256=hashes),dict(reference_id='frozen'))
    monkeypatch.setitem(sys.modules,'shared_base_posterior_full',SimpleNamespace())
    monkeypatch.setattr(ext,'parent_bundle',lambda *args,**kwargs:bundle)
    context=dict(output=str(out));parent=dict(output=str(parentroot))
    ext.prepare(plan,parent,{},context,spec)
    for key in ('reference','checkpoint','moments'):assert dest[key].read_bytes()==original[key].read_bytes()
    # A resumed child is allowed to have progressed; never copy the parent over it.
    dest['checkpoint'].write_text('extended bytes');ext.prepare(plan,parent,{},context,spec)
    assert dest['checkpoint'].read_text()=='extended bytes'
    assert {name:ext.sha(name) for name in hashes}==hashes
    dest['moments'].unlink()
    with pytest.raises(ValueError,match='disappeared'):ext.prepare(plan,parent,{},context,spec)


@pytest.mark.parametrize('target_inside',[False,True])
def test_child_paths_reject_symlinks(plan,tmp_path,target_inside):
    out=tmp_path/'child';out.mkdir();target=out/'other' if target_inside else tmp_path/'parent'
    target.mkdir();(out/'references').symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink|escapes'):ext.paths(out,plan['tasks'][0]['task'])


def test_incomplete_group_never_publishes_an_available_case_mean():
    # Production aggregator is reused, with no replacement tolerance or approval.
    result=ext.revalidate_groups(dict(rows=[],launch_id='test'),{})
    assert len(result['groups'])==8
    assert not result['scientific_reference_validated']
    for g in result['groups']:
        assert not g['numerical_audit']['complete'] and not g['numerical_audit']['passed']
        assert g['numerical_audit']['reported_mean_bayes_regret_bits'] is None


def test_launcher_dry_run_reads_no_parent_or_creates_output(tmp_path):
    out=tmp_path/'new';missing=tmp_path/'no-parent.json'
    command=['bash',str(ext.ROOT/ext.FILES[1]),'--parent-launch',str(missing),'--out',str(out)]
    run=subprocess.run(command,check=True,capture_output=True,text=True)
    assert '--array=0,1,2%3' in run.stdout and not out.exists()
    merged=subprocess.run(command+['--merge','--after-job','1234'],check=True,capture_output=True,text=True)
    assert '--dependency=afterany:1234' in merged.stdout and '--cpus-per-task=1' in merged.stdout
    assert '--mem=6G' in merged.stdout and not out.exists()
    bad=subprocess.run(command+['--only-task','3'],capture_output=True,text=True)
    assert bad.returncode!=0


def test_compute_work_refuses_login_node(plan,tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ValueError,match='compute-node'):ext.load_context(ext.PLAN,tmp_path/'no-parent',tmp_path/'out')
