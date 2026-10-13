"""No sampling: second-generation ancestry, bounded dispatch and full coverage."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import full_shared_base_continuation as c


@pytest.fixture
def plan():return c.read(c.PLAN)


def test_frozen_files_and_exact_budget(plan):
    c.validate_plan(plan)
    assert plan['code_sha256']=={name:c.sha(c.ROOT/name) for name in c.FILES}
    assert c.TARGET_DRAWS-c.PARENT_DRAWS==524288
    assert c.RESOURCES['maximum_attempts']==c.RESOURCES['concurrency']==1
    assert c.RESOURCES['wall_seconds']==28800
    assert c.RESOURCES['memory_gib']==72 and c.RESOURCES['cpus']==12


@pytest.mark.parametrize('mutate',[
    lambda p:p.update(target_draws=2097152),lambda p:p.update(parent_draws=524288),
    lambda p:p.update(tolerances_changed=True),lambda p:p['task'].update(index=195),
    lambda p:p['resources'].update(maximum_attempts=2),lambda p:p.update(previous_task_index=0),
    lambda p:p.update(previous_plan_sha256='0'*64),lambda p:p['previous_files'].pop('context.json')])
def test_reject_changed_scope(plan,mutate):
    mutate(plan)
    with pytest.raises(ValueError):c.validate_plan(plan)


@pytest.fixture
def family(plan,tmp_path,monkeypatch):
    """Real receipt readers and file hashes, with only original deployment mocked."""
    old=c.read(c.base.PLAN);root=tmp_path/'previous';parentroot=tmp_path/'original';inputs=tmp_path/'inputs'
    launch=tmp_path/'launch.json';c.atomic_json(launch,dict(original=True))
    context=dict(schema='full-shared-base-extension-context-v1',output=str(root),
        plan_sha256=c.sha(c.base.PLAN),parent_launch_sha256=c.sha(launch),parent_launch_id=old['parent_launch_id'])
    c.atomic_json(root/'context.json',context)
    plan['previous_context_sha256']=c.fingerprint(context)
    plan['previous_files']={'context.json':c.sha(root/'context.json')}
    references=[];audits=[]
    stat=lambda name:dict(size_bytes=Path(name).stat().st_size,mtime_ns=Path(name).stat().st_mtime_ns)
    for i,spec in enumerate(old['tasks']):
        task=spec['task'];p=c.paths(root,task);ready=i!=1
        ref=dict(setting=task['setting'],index=task['index'],time=1024,order=task['order'],
            identity=dict(master_sampling_seed=2026091780,burn=16384,kernel='unchanged'),
            reference_id=spec['parent_reference_id'],history=[dict(draws=524288),dict(draws=1048576)],
            estimate=dict(diagnostics=dict(draws=1048576,chains=4,thin=1)))
        c.atomic_json(p['reference'],ref)
        for key in ('checkpoint','moments'):p[key].write_bytes((key+' saved state and RNG').encode())
        c.atomic_json(p['prepared'],dict(task=task,unchanged=True))
        a=dict(index=task['index'],time=1024,order=task['order'],draws_per_chain=1048576,ready=ready,
            status='ready' if ready else 'needs_more_draws',report_sha256=c.sha(p['reference']),
            gates=dict(split_rhat_below_1_05=True,probability_ess_at_least_200=ready,
                       maximum_probability_mcse_at_most_0_005=True,raw_normalization=True))
        audit=dict(task=task,context_sha256=c.fingerprint(context),parent_audit_fingerprint=spec['parent_audit_fingerprint'],
            parent_preparation_sha256=c.sha(p['prepared']),corrected=dict(audit=a),
            raw_sha256={str(p[k]):c.sha(p[k]) for k in ('reference','checkpoint','moments')},
            raw_stat={str(p[k]):stat(p[k]) for k in ('reference','checkpoint','moments')})
        c.atomic_json(p['audit'],audit)
        receipt=dict(task=task,context_sha256=c.fingerprint(context),status='ready' if ready else 'unresolved',
            worker_exit_code=0 if ready else 2,audit_exit_code=0,audit_sha256=c.sha(p['audit']),runtime=dict(frozen=True))
        c.atomic_json(p['receipt'],receipt)
        for key in ('receipt','audit','reference','prepared'):
            plan['previous_files'][str(p[key].relative_to(root))]=c.sha(p[key])
        if i==1:plan.update(parent_audit_fingerprint=c.fingerprint(a),parent_reference_id=ref['reference_id'])
        references.append(ref);audits.append(a)
    parent=dict(output=str(parentroot),inputs=str(inputs),launch_id=old['parent_launch_id'],
                plan_sha256=old['parent_plan_sha256'],dependency_root=str(tmp_path/'dependencies'))
    parent_plan=dict(tasks=[None]*2000)
    for spec in old['tasks']:parent_plan['tasks'][spec['parent_task_index']]=spec['task']
    full=SimpleNamespace(file_stat=stat,load_launch=lambda *a,**k:(parent,parent_plan))
    monkeypatch.setitem(sys.modules,'shared_base_posterior_full',full)
    monkeypatch.setattr(c.base,'parent_bundle',lambda *a,**k:None)
    monkeypatch.setenv('SLURM_JOB_ID','12345')
    return SimpleNamespace(plan=plan,old=old,root=root,context=context,parent=parent,parent_plan=parent_plan,
                           launch=launch,refs=references,audits=audits,full=full,out=tmp_path/'new')


def test_reads_and_copies_only_selected_parent_without_mutation(family):
    f=family;before={str(p):p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
    for i in range(3):c.previous_bundle(f.plan,f.context,f.old,i,verify_raw=True)
    context=dict(output=str(f.out))
    dest,runtime,reference=c.prepare(f.plan,context,f.context,f.old)
    original=c.paths(f.root,c.TASK)
    for k in ('reference','checkpoint','moments'):assert dest[k].read_bytes()==original[k].read_bytes()
    assert runtime==dict(frozen=True) and reference==f.refs[1]
    assert {str(p):p.read_bytes() for p in f.root.rglob('*') if p.is_file()}==before
    assert list((f.out/'references/a1_L1/alpha16').glob('*chains.npz'))==[dest['checkpoint']]
    with pytest.raises(ValueError,match='one allocation'):c.prepare(f.plan,context,f.context,f.old)


@pytest.mark.parametrize('drift',['receipt','report','audit','writer','raw'])
def test_parent_drift_is_refused(family,drift):
    f=family;p=c.paths(f.root,c.TASK)
    if drift=='writer':(f.root/'claims'/(c.TASK['id']+'.writer')).mkdir(parents=True)
    else:
        key={'receipt':'receipt','report':'reference','audit':'audit','raw':'checkpoint'}[drift]
        p[key].write_bytes(p[key].read_bytes()+b' ')
    with pytest.raises(ValueError):c.previous_bundle(f.plan,f.context,f.old,1,verify_raw=True)


def test_real_context_and_parent_reader_integration(family,tmp_path):
    f=family;plan_path=tmp_path/'plan.json';c.atomic_json(plan_path,f.plan)
    result=c.load_context(plan_path,f.launch,f.root,f.out)
    assert result[0]==f.plan and result[3]['previous_context_sha256']==c.fingerprint(f.context)
    assert c.read(f.out/'context.json')==result[3]
    for protected in (f.root,Path(f.parent['output']),Path(f.parent['inputs'])):
        with pytest.raises(ValueError,match='overlaps'):c.load_context(plan_path,f.launch,f.root,protected/'bad')


def test_login_node_cannot_copy_or_sample(plan,tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ValueError,match='compute-node'):c.load_context(c.PLAN,tmp_path/'missing',tmp_path,tmp_path/'new')


def test_continuation_exact_identity_history_and_budget(family):
    f=family;ref=deepcopy(f.refs[1]);ref['history'].append(dict(draws=1572864));ref['estimate']['diagnostics']['draws']=1572864
    c.validate_continuation(ref,f.refs[1])
    command=c.worker_command(f.parent,f.out)
    for flag,value in [('--initial-draws','1572864'),('--max-draws','1572864'),('--seed','2026091780'),
                       ('--burn','16384'),('--steps','64'),('--proposal','iid'),('--index','207'),('--alpha-order','16')]:
        assert command[command.index(flag)+1]==value
    for change in (lambda r:r['identity'].update(kernel='changed'),lambda r:r['history'][0].update(draws=1),
                   lambda r:r['estimate']['diagnostics'].update(draws=2097152)):
        bad=deepcopy(ref);change(bad)
        with pytest.raises(ValueError):c.validate_continuation(bad,f.refs[1])


def test_composed_overlay_preserves_two_ready_and_all_other_cases(family):
    f=family;rows=[];replacements=[]
    for i,spec in enumerate(f.old['tasks']):
        old=dict(index=spec['task']['index'],time=1024,order=spec['task']['order'],draws_per_chain=524288,ready=False)
        rows.append(dict(task=spec['task'],status='unresolved',audit=old))
        replacements.append(dict(task=spec['task'],status='ready' if i!=1 else 'unresolved',
            audit=f.audits[i],parent_audit_fingerprint=c.fingerprint(old)))
    other=dict(task=dict(id='other'),status='running');rows.append(other)
    snapshot=dict(launch_id=f.old['parent_launch_id'],rows=rows,groups=[dict(stale=True)])
    previous=dict(schema='full-shared-base-extension-overlay-v1',complete=True,replacements=replacements,
        parent_launch_id=f.old['parent_launch_id'],plan_sha256='a'*64,context_sha256='b'*64,parent_snapshot_sha256='c'*64)
    improved=deepcopy(f.audits[1]);improved.update(draws_per_chain=1572864,ready=True,status='ready')
    replacement=dict(task=c.TASK,status='ready',audit=improved,parent_audit_fingerprint=f.plan['parent_audit_fingerprint'])
    before=deepcopy(snapshot);result=c.apply_overlay(snapshot,previous,replacement,f.plan)
    assert snapshot==before and result['counts']==dict(ready=3,running=1)
    assert result['rows'][0]['audit']==f.audits[0] and result['rows'][2]['audit']==f.audits[2]
    assert result['rows'][3]==other and result['groups']==[] and not result['all_numerical_checks_passed']
    replacement['audit']['ready']=False;replacement['status']='unresolved'
    assert c.apply_overlay(snapshot,previous,replacement,f.plan)['counts']==dict(ready=2,unresolved=1,running=1)
    replacement['audit']['draws_per_chain']=1048576
    with pytest.raises(ValueError):c.apply_overlay(snapshot,previous,replacement,f.plan)


def test_dry_run_no_parent_reads_no_output(tmp_path):
    out=tmp_path/'new'
    run=subprocess.run(['bash',str(c.ROOT/c.FILES[1]),'--out',str(out),'--parent-launch',str(tmp_path/'missing'),
                        '--previous-output',str(tmp_path/'missing-previous')],check=True,capture_output=True,text=True)
    assert sum(line.startswith('sbatch ') for line in run.stdout.splitlines())==2
    assert '--dependency=afterany:WORKER_JOB' in run.stdout
    assert '--array=' not in run.stdout and not out.exists()


def test_mocked_one_command_submission_and_idempotency(plan,tmp_path,monkeypatch):
    submitted=[]
    def run(cmd,**kwargs):
        if cmd[0]=='squeue':return SimpleNamespace(stdout='',stderr='')
        submitted.append(cmd);return SimpleNamespace(stdout=str(4000+len(submitted))+'\n',stderr='')
    monkeypatch.setattr(c.subprocess,'run',run)
    args=SimpleNamespace(plan=c.PLAN,out=tmp_path/'new',parent_launch=tmp_path/'parent',
                         previous_output=tmp_path/'previous',venv=tmp_path/'venv',submit=True)
    c.launch(args);c.launch(args)
    assert len(submitted)==2
    assert '--dependency=afterany:4001' in submitted[1] and submitted[0][-1]=='run' and submitted[1][-1]=='merge'
    assert '--cpus-per-task=1' in submitted[1] and '--mem=6G' in submitted[1]


def test_uncertain_submission_refuses_retry(tmp_path,monkeypatch):
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout='' if cmd[0]=='squeue' else 'not a job id',stderr='')
    monkeypatch.setattr(c.subprocess,'run',run)
    for _ in range(2):
        with pytest.raises(ValueError):c.submission(['sbatch','test'],tmp_path/'out','same','job')
    assert sum(cmd[0]=='sbatch' for cmd in calls)==1


def test_no_second_allocation(family,tmp_path,monkeypatch):
    f=family;context=dict(output=str(f.out));c.atomic_json(f.out/'allocation.json',dict(previous=True))
    monkeypatch.setattr(c,'load_context',lambda *a:(f.plan,f.parent,f.parent_plan,context,f.context,f.old))
    monkeypatch.setattr(c,'prepare',lambda *a:pytest.fail('must not copy or sample twice'))
    with pytest.raises(ValueError,match='One allocation'):c.execute(c.PLAN,f.launch,f.root,f.out)


@pytest.mark.parametrize('ready',[True,False])
def test_compute_execution_and_real_audit_adapter(family,tmp_path,monkeypatch,ready):
    """Run preparation, dispatch, audit adapter and receipt without a Markov step."""
    f=family;plan_path=tmp_path/'plan.json';c.atomic_json(plan_path,f.plan)
    before={str(p):p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
    commands=[];prefix_checks=[]
    from contextlib import nullcontext
    import unseen_context_refinement as preservation
    monkeypatch.setattr(preservation,'verify_preserved_samples',lambda *a,**k:prefix_checks.append((a,k)))
    def raw_audit(folder,out,index,time,order):
        assert (index,time,order)==(207,1024,16)
        value=deepcopy(f.audits[1]);value.update(draws_per_chain=1572864,ready=ready,
             status='ready' if ready else 'needs_more_draws',report_sha256=c.sha(c.paths(f.out,c.TASK)['reference']))
        return value
    monkeypatch.setitem(sys.modules,'shared_base_curve_audit',SimpleNamespace(audit_reference=raw_audit))
    monkeypatch.setitem(sys.modules,'recover_shared_base_posterior_audit',
                        SimpleNamespace(canonical_model_loader=lambda *a:nullcontext()))
    monkeypatch.setitem(sys.modules,'revalidate_shared_base_depth',SimpleNamespace(
        diagnostic=SimpleNamespace(summarize_depths=lambda *a:dict(unchanged=True)),
        adjust_depth_gate=lambda reference,raw,depths:dict(audit=raw)))
    monkeypatch.setitem(sys.modules,'validate_shared_base_mcmc',SimpleNamespace(json_safe=lambda v:v))
    def run_child(command,log,seconds,progress=None):
        commands.append((command,seconds))
        if 'audit' in command:
            c.audit_task(plan_path,f.launch,f.root,f.out);return 0
        p=c.paths(f.out,c.TASK);ref=c.read(p['reference']);ref['history'].append(dict(draws=1572864))
        ref['estimate']['diagnostics']['draws']=1572864;c.atomic_json(p['reference'],ref)
        p['checkpoint'].write_bytes(b'continued saved checkpoint');return 0 if ready else 2
    f.full.original=SimpleNamespace(runtime_setup=lambda *a:dict(frozen=True),progress_record=lambda *a:{},run_child=run_child)
    assert c.execute(plan_path,f.launch,f.root,f.out)==(0 if ready else 2)
    p=c.paths(f.out,c.TASK);receipt=c.read(p['receipt'])
    assert receipt['status']==('ready' if ready else 'unresolved')
    context=c.read(f.out/'context.json')
    c.base.load_child(context,dict(task=c.TASK,parent_audit_fingerprint=f.plan['parent_audit_fingerprint']))
    assert len(commands)==2 and commands[0][1]<=23400 and commands[1][1]<=3600
    assert commands[1][0][1]==str(Path(c.__file__))
    assert prefix_checks==[((c.paths(f.root,c.TASK)['checkpoint'],p['checkpoint'],
                            dict(baseline_draws=1048576,target_draws=1572864)),dict(complete=True))]
    assert {str(p):p.read_bytes() for p in f.root.rglob('*') if p.is_file()}==before
    with pytest.raises(ValueError,match='One allocation'):c.execute(plan_path,f.launch,f.root,f.out)


def test_retained_array_prefixes_are_checked_byte_for_byte(tmp_path):
    import hashlib
    import numpy as np
    from unseen_context_refinement import verify_preserved_samples
    def checkpoint(path,draws,corrupt=False):
        arrays=dict(burn_done=np.array([16]*4))
        for chain in range(4):
            for kind,width in (('predictions',64),('depths',9),('scalars',2)):
                arrays[f'{kind}_{chain}']=np.arange(draws*width,dtype=np.float64).reshape(draws,width)
            arrays[f'alpha_indices_{chain}']=np.arange(draws,dtype=np.int64)
        if corrupt:arrays['predictions_3'][2,17]+=.5
        def digest(value):
            h=hashlib.sha256(value.dtype.str.encode());h.update(json.dumps(value.shape).encode());h.update(value.tobytes());return h.hexdigest()
        meta=dict(format='shared_base_chain_run_v1',fingerprint=dict(original=True),config=dict(chains=4),
            initializations=['saved']*4,proposal_descriptor=dict(original=True),
            array_hashes={k:digest(v) for k,v in arrays.items()})
        serialized=json.dumps(meta,sort_keys=True,separators=(',',':'))
        np.savez_compressed(path,**arrays,metadata=np.asarray(serialized),
                            metadata_sha256=np.asarray(hashlib.sha256(serialized.encode()).hexdigest()))
    parent=tmp_path/'parent.npz';child=tmp_path/'child.npz';checkpoint(parent,4);checkpoint(child,6)
    spec=dict(baseline_draws=4,target_draws=6)
    assert verify_preserved_samples(parent,child,spec,complete=True)==[6]*4
    checkpoint(child,6,corrupt=True)
    with pytest.raises(ValueError,match='original retained'):verify_preserved_samples(parent,child,spec,complete=True)
