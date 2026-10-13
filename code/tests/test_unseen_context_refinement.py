"""Independent adversarial checks for the isolated unseen-context continuation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/scripts'))
import unseen_context_refinement as driver


def specimen():
    task = dict(id='L4_i5_t64_a16', setting='a1_L4', depth=4, index=5, time=64, order=16)
    identity = dict(inputs=dict(source_id='source', index=5, prefix=64),
        target=dict(d=64, prefix_length=64, depths=list(range(9)), depth_prior=[1/9]*9,
                    alpha_nodes=list(np.linspace(.51,1.49,16))),
        proposal=dict(name='frozen kernel'), burn=8, sampling_seed=7, master_sampling_seed=11)
    spec = dict(task=task, source_id='source', parent_identity=identity,
                parent_reference_id=driver.fingerprint(identity), baseline_draws=16, target_draws=64)
    report = dict(**{k:task[k] for k in ('setting','index','time','order')}, identity=identity,
                  reference_id=spec['parent_reference_id'], history=[dict(draws=16, original=True)],
                  estimate=dict(diagnostics=dict(draws=16, chains=4, thin=1, burn=8, seed=7)))
    source = dict(source_id='source', tokens=[0]*63+[53,2])
    return spec, report, source


def test_source_pairing_requires_frozen_event_and_prior():
    spec, report, source = specimen()
    driver.check_pairing(spec,report,source,report['identity']['inputs'])
    for field in ('source','event','input','depth_prior','coordinates'):
        a,b,c = deepcopy(spec),deepcopy(report),deepcopy(source)
        if field=='source':c['source_id']='another'
        if field=='event':c['tokens'][3]=53
        if field=='input':b['identity']['inputs']['index']=6
        if field=='depth_prior':b['identity']['target']['depths']=[4]
        if field=='coordinates':b['index']=6
        with pytest.raises(ValueError):driver.check_pairing(a,b,c,a['parent_identity']['inputs'])


def test_complete_requires_actual_target_and_unmodified_history():
    spec,parent,_=specimen();child=deepcopy(parent)
    child['estimate']['diagnostics']['draws']=64;child['history'].append(dict(draws=64))
    driver.validate_continuation(child,parent,spec)
    for field in ('draws','chains','thin','burn','seed','history','identity'):
        bad=deepcopy(child)
        if field in ('draws','chains','thin','burn','seed'):
            bad['estimate']['diagnostics'][field]+=1
        elif field=='history':bad['history'][0]['original']=False
        else:bad['identity']['sampling_seed']=19
        with pytest.raises(ValueError):driver.validate_continuation(bad,parent,spec)


def array_digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode()+json.dumps(a.shape).encode()+a.tobytes()).hexdigest()


def checkpoint(path, counts, *, original=None, mutate=None):
    arrays=dict(burn_done=np.full(4,8,dtype=np.int64))
    for i,n in enumerate(counts):
        for name,width in [('predictions',3),('depths',2),('scalars',2),('alpha_indices',None)]:
            shape=(n,) if width is None else (n,width)
            a=np.arange(np.prod(shape),dtype=np.int64 if width is None else np.float64).reshape(shape)
            if original is not None:
                a[:len(original[f'{name}_{i}'])]=original[f'{name}_{i}']
            arrays[f'{name}_{i}']=a
    if mutate is not None:arrays[mutate][2]=999
    meta=dict(format='shared_base_chain_run_v1', fingerprint={'prior':'unchanged'},
        config=dict(chains=4,burn=8,thin=1,seed=7), initializations=['same']*4,
        proposal_descriptor={'kernel':'unchanged'},
        array_hashes={name:array_digest(a) for name,a in arrays.items()})
    serialized=json.dumps(meta,sort_keys=True,separators=(',',':'))
    np.savez_compressed(path,**arrays,metadata=np.asarray(serialized),
        metadata_sha256=np.asarray(hashlib.sha256(serialized.encode()).hexdigest()))
    return arrays


def test_streamed_prefix_rejects_changed_original_even_with_self_consistent_new_metadata(tmp_path):
    spec,_,_=specimen();parent=tmp_path/'parent.npz';child=tmp_path/'child.npz'
    original=checkpoint(parent,[16]*4);checkpoint(child,[64]*4,original=original)
    assert driver.verify_preserved_samples(parent,child,spec,complete=True)==[64]*4
    checkpoint(child,[64]*4,original=original,mutate='predictions_2')
    with pytest.raises(ValueError,match='original retained samples'):
        driver.verify_preserved_samples(parent,child,spec,complete=True)


def test_lossless_partial_checkpoint_can_resume_but_cannot_count_as_completion(tmp_path):
    spec,_,_=specimen();parent=tmp_path/'parent.npz';child=tmp_path/'child.npz'
    original=checkpoint(parent,[16]*4);checkpoint(child,[64,40,16,16],original=original)
    assert driver.verify_preserved_samples(parent,child,spec,complete=False)==[64,40,16,16]
    with pytest.raises(ValueError,match='All four chains'):
        driver.verify_preserved_samples(parent,child,spec,complete=True)
    checkpoint(child,[64,64,64,65],original=original)
    with pytest.raises(ValueError,match='outside'):
        driver.verify_preserved_samples(parent,child,spec,complete=False)


def cohort():
    groups=json.loads((ROOT/'code/transformer/configs/unseen_context_refinement_plan_2026-09-19.json').read_text())['groups']
    tasks=[];rows=[]
    for g in groups:
        for i in g['selected_indices']:
            for order in (16,32):
                task=dict(id=f"L{g['depth']}_i{i}_t64_a{order}", setting=f"a1_L{g['depth']}",
                    depth=g['depth'],index=i,time=64,order=order)
                spec=dict(task=task);tasks.append(spec)
                precision=dict(bayes_regret_bits=i/100.,mc_variance_bits2=1e-8,jensen_proxy_bits=1e-8,
                    scalar_mixing_passed=True,jensen_local_check_passed=True,deterministic=False)
                record=dict(case_id=i,sampling_seed=g['depth']*100000+i*100+order,precision=precision,
                    full_prior_valid=True,provenance_valid=True,mixing_passed=True,row_grid_passed=True,
                    per_sequence_precision_passed=False)
                audited=dict(index=i,time=64,order=order,status='ready',ready=True,record=record)
                rows.append(dict(task=dict(task,key=driver.key(spec)),corrected=dict(audit=audited)))
    return dict(tasks=tasks,groups=groups),rows


def test_complete_246_records_two_full_groups_use_unchanged_gate():
    plan,rows=cohort();result=driver.aggregate_groups(plan,rows)
    assert result['complete'] and result['all_groups_passed']
    assert [g['n_histories'] for g in result['groups']]==[66,57]
    for g in result['groups']:
        assert g['bayes_released'] and g['numerical_validation']['budget']==dict(
            resolution_bits=.01,numerical_fraction=1/3,relative_fraction=.25,mc_multiplier=2.,bias_multiplier=2.)
        assert g['numerical_validation']['rules'][0]['per_sequence_precision_status']
        assert not any(g['numerical_validation']['rules'][0]['per_sequence_precision_status'].values())


def test_missing_rule_withholds_entire_group_but_other_full_group_is_independent():
    plan,rows=cohort();result=driver.aggregate_groups(plan,rows[:-1])
    assert not result['complete'] and not result['all_groups_passed']
    assert result['groups'][0]['bayes_released']
    failed=result['groups'][1]
    assert not failed['bayes_released'] and not failed['paired_bayes_comparisons_released']
    assert failed['numerical_validation']['reported_mean_bayes_regret_bits'] is None
    assert 'rules' not in failed['numerical_validation']


def test_failed_case_is_retained_and_cannot_select_passing_subset():
    plan,rows=cohort();rows[0]['corrected']['audit']['record']['full_prior_valid']=False
    result=driver.aggregate_groups(plan,rows)
    assert result['complete'] and not result['all_groups_passed']
    failed=result['groups'][0]
    assert failed['n_histories']==66 and failed['numerical_validation']['n_sequences']==66
    assert not failed['bayes_released'] and failed['numerical_validation']['reported_mean_bayes_regret_bits'] is None


def test_complete_individual_checks_do_not_override_group_precision_budget():
    plan,rows=cohort()
    for row in rows:row['corrected']['audit']['record']['precision']['mc_variance_bits2']=1.
    result=driver.aggregate_groups(plan,rows)
    assert result['complete'] and not any(g['bayes_released'] for g in result['groups'])


@pytest.mark.parametrize('corruption',['duplicate','foreign','mispaired'])
def test_merge_rejects_invented_or_mispaired_reference(corruption):
    plan,rows=cohort()
    if corruption=='duplicate':rows.append(rows[0])
    elif corruption=='foreign':rows[0]['task']['key']='foreign'
    else:rows[0]['corrected']['audit']['index']+=1
    with pytest.raises(ValueError):driver.aggregate_groups(plan,rows)


def test_retry_is_explicit_bounded_and_no_numerical_failure_retry(tmp_path):
    spec,_,_=specimen();ctx=dict(output=str(tmp_path));plan=dict(tasks=[spec],resources=dict(
        max_attempts=2,task_wall_seconds=7200,total_worker_seconds=20000))
    attempts=driver.allocate(plan,ctx,spec,retry=False)
    attempts[-1].update(status='timeout',wall_seconds=7000)
    driver.atomic_json(driver.paths(tmp_path,spec)['attempts'],attempts)
    with pytest.raises(ValueError,match='explicit'):driver.allocate(plan,ctx,spec,retry=False)
    attempts=driver.allocate(plan,ctx,spec,retry=True)
    assert len(attempts)==2
    with pytest.raises(ValueError,match='exhausted'):driver.allocate(plan,ctx,spec,retry=True)
    attempts[:]=attempts[:1];attempts[-1]['status']='unresolved'
    driver.atomic_json(driver.paths(tmp_path,spec)['attempts'],attempts)
    with pytest.raises(ValueError,match='terminal'):driver.allocate(plan,ctx,spec,retry=True)


def test_lost_process_reservation_counts_toward_cumulative_limit(tmp_path):
    spec,_,_=specimen();ctx=dict(output=str(tmp_path));plan=dict(tasks=[spec],resources=dict(
        max_attempts=2,task_wall_seconds=7200,total_worker_seconds=10000))
    driver.allocate(plan,ctx,spec,retry=False)
    with pytest.raises(ValueError,match='Cumulative'):driver.allocate(plan,ctx,spec,retry=True)


def test_output_symlink_escape_is_rejected(tmp_path):
    spec,_,_=specimen();out=tmp_path/'out';outside=tmp_path/'elsewhere'
    out.mkdir();outside.mkdir();(out/'references').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):driver.paths(out,spec)


def test_memory_admission_largest_is_bounded_and_cannot_pair_with_itself():
    largest=driver.slot_bytes(1048576);budget=18*1024**3
    assert largest<budget<2*largest
    assert driver.slot_bytes(16384)*4<budget


def test_saved_cohort_summary_serializes_every_numpy_check(tmp_path):
    from validate_shared_base_mcmc import json_safe
    plan,rows=cohort()
    result=json_safe(driver.aggregate_groups(plan,rows))
    path=tmp_path/'summary.json'
    driver.atomic_json(path,result)
    assert json.loads(path.read_text())['all_groups_passed'] is True


def dispatch_fixture(tmp_path, monkeypatch, targets, *, child_exit=0):
    tasks=[]
    for i,target in enumerate(targets):
        spec,_,_=specimen()
        spec['task'].update(id=f'L4_i{i}_t64_a16',index=i)
        spec['target_draws']=target;tasks.append(spec)
    resources=dict(max_workers=4,memory_budget_bytes=18*1024**3,
        disk_reserve_bytes=1,max_attempts=2,task_wall_seconds=7200,
        total_worker_seconds=128*3600,max_dispatch_seconds=43200)
    plan=dict(tasks=tasks,resources=resources,worker=dict(python=sys.executable),environment={})
    ctx=dict(output=str(tmp_path),plan_sha256='fixed')
    monkeypatch.setattr(driver,'context',lambda *a:(plan,ctx))
    monkeypatch.setattr(driver,'external_active',lambda p:0)
    monkeypatch.setattr(driver.time,'sleep',lambda t:None)
    spawned=[];snapshots=[];save=driver.atomic_json
    def record_save(path,value,**kwargs):
        if Path(path).name=='status.json':snapshots.append(deepcopy(value))
        return save(path,value,**kwargs)
    monkeypatch.setattr(driver,'atomic_json',record_save)
    class Process:
        def __init__(self,command,**kwargs):
            self.index=int(command[command.index('--index')+1])
            self.pid=1000000+len(spawned);self.returncode=child_exit
            spawned.append(self.index)
            # Stub only the bounded child process, leaving controller bookkeeping,
            # file paths, locks and receipts exercised by the real dispatcher.
            if child_exit==0:
                save(driver.paths(tmp_path,tasks[self.index])['receipt'],dict(status='ready'))
        def poll(self):return self.returncode
    monkeypatch.setattr(driver.subprocess,'Popen',Process)
    return plan,ctx,spawned,snapshots


def test_dispatch_admits_at_most_four_workers_with_memory_sum_below_18gib(tmp_path,monkeypatch):
    plan,_,spawned,snapshots=dispatch_fixture(tmp_path,monkeypatch,[16384]*10)
    result=driver.dispatch(tmp_path/'plan.json',tmp_path)
    assert spawned==list(range(10)) and result['counts']=={'ready':10}
    assert max(s['active_workers'] for s in snapshots)==4
    assert all(s['reserved_memory_bytes']<=plan['resources']['memory_budget_bytes'] for s in snapshots)


def test_dispatch_big_checkpoints_reduce_concurrency_before_launch(tmp_path,monkeypatch):
    plan,_,spawned,snapshots=dispatch_fixture(tmp_path,monkeypatch,[1048576]*3)
    result=driver.dispatch(tmp_path/'plan.json',tmp_path)
    assert spawned==[0,1,2] and result['counts']=={'ready':3}
    assert max(s['active_workers'] for s in snapshots)==1
    assert all(s['reserved_memory_bytes']<=plan['resources']['memory_budget_bytes'] for s in snapshots)


def test_dispatch_smoke_selection_preserves_every_declared_job(tmp_path,monkeypatch):
    _,_,spawned,_=dispatch_fixture(tmp_path,monkeypatch,[16384]*8)
    result=driver.dispatch(tmp_path/'plan.json',tmp_path,only_tasks=[1,4])
    assert spawned==[1,4] and len(result['jobs'])==8
    assert result['counts']==dict(pending=6,ready=2)
    assert all(result['jobs'][i]['state']=='pending' for i in [0,2,3,5,6,7])


def test_dispatch_child_exit_without_audited_receipt_is_failure(tmp_path,monkeypatch):
    _,_,spawned,_=dispatch_fixture(tmp_path,monkeypatch,[16384],child_exit=1)
    result=driver.dispatch(tmp_path/'plan.json',tmp_path)
    assert spawned==[0] and result['counts']=={'failed':1}
    assert result['jobs'][0]['exit_code']==1


def test_second_dispatcher_cannot_claim_same_output(tmp_path,monkeypatch):
    _,_,spawned,_=dispatch_fixture(tmp_path,monkeypatch,[16384])
    with driver.lock(tmp_path/'dispatcher.lock'):
        with pytest.raises(BlockingIOError):driver.dispatch(tmp_path/'plan.json',tmp_path)
    assert not spawned


@pytest.mark.parametrize('caffeinate_available',[False,True])
@pytest.mark.parametrize('retry',[False,True])
def test_bash3_launcher_runs_preflight_dispatch_and_merge_without_sampling(
        tmp_path,caffeinate_available,retry):
    """Exercise the actual launcher under macOS /bin/bash, replacing only work.

    Bash 3 treats an empty array expansion as unbound under set -u. The ordinary
    no-retry launch must reach both dispatcher and merge without that failure.
    """
    import os
    import shutil
    import subprocess
    sandbox=tmp_path/'isolated repository';directory=sandbox/'code/transformer'
    directory.mkdir(parents=True)
    launcher=directory/'start_unseen_context_refinement.sh'
    shutil.copyfile(ROOT/'code/transformer/start_unseen_context_refinement.sh',launcher)
    runtime=sandbox/'output/unseen_context_refinement_runtime_2026-09-19/bin/python'
    runtime.parent.mkdir(parents=True)
    calls=tmp_path/'mock_calls.jsonl'
    runtime.write_text(f'#!{sys.executable}\n'+
        'import json, os, sys\n'+
        'with open(os.environ["MOCK_LAUNCHER_CALLS"],"a") as stream:\n'+
        '    stream.write(json.dumps(sys.argv[1:])+"\\n")\n')
    runtime.chmod(0o755)
    commands=tmp_path/'commands';commands.mkdir()
    (commands/'dirname').symlink_to('/usr/bin/dirname')
    if caffeinate_available:
        shim=commands/'caffeinate'
        shim.write_text('#!/bin/sh\n[ "$1" = "-i" ] || exit 91\nshift\nexec "$@"\n')
        shim.chmod(0o755)
    env=dict(os.environ,PATH=str(commands),MOCK_LAUNCHER_CALLS=str(calls))
    command=['/bin/bash',str(launcher),'--run']
    if retry:command.append('--retry-incomplete')
    process=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=10)
    assert process.returncode==0,process.stderr
    recorded=[json.loads(line) for line in calls.read_text().splitlines()]
    assert len(recorded)==3
    assert recorded[0][0]=='-'  # Mocked preflight; no numerical modules loaded.
    assert recorded[1][1]=='run' and recorded[2][1]=='merge'
    assert ('--retry-incomplete' in recorded[1]) is retry
    assert '--retry-incomplete' not in recorded[2]
    assert all(row[row.index('--out')+1].startswith(str(sandbox)) for row in recorded[1:])


def test_parallel_allocations_wait_for_shared_budget_lock_and_each_reserve_once(tmp_path):
    """Real processes contend for one ledger without dropping worker allocations."""
    import os
    import selectors
    import subprocess
    tasks=[]
    for i in (5,6):
        spec,_,_=specimen();spec['task'].update(id=f'L4_i{i}_t64_a16',index=i);tasks.append(spec)
    plan=dict(tasks=tasks,resources=dict(max_attempts=2,task_wall_seconds=7200,total_worker_seconds=14400))
    ctx=dict(output=str(tmp_path));payload=tmp_path/'payload.json'
    payload.write_text(json.dumps(dict(plan=plan,ctx=ctx)))
    program=('import json,sys;sys.path.insert(0,sys.argv[1]);import unseen_context_refinement as d;'
             'p=json.load(open(sys.argv[2]));print("waiting",flush=True);'
             'a=d.allocate(p["plan"],p["ctx"],p["plan"]["tasks"][int(sys.argv[3])],retry=False);'
             'print(json.dumps(a),flush=True)')
    processes=[]
    try:
        with driver.lock(tmp_path/'budget.lock'):
            for i in (0,1):
                processes.append(subprocess.Popen([sys.executable,'-c',program,str(ROOT/'code/scripts'),str(payload),str(i)],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=dict(os.environ)))
            for process in processes:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout,selectors.EVENT_READ)
                    assert selector.select(5),'Allocation process did not start'
                    assert process.stdout.readline().strip()=='waiting'
                # It has announced entry and cannot finish until the parent
                # releases the ledger. The old nonblocking call exits here.
                with pytest.raises(subprocess.TimeoutExpired):process.wait(timeout=.1)
        for process in processes:
            stdout,stderr=process.communicate(timeout=5)
            assert process.returncode==0,stderr
            assert len(json.loads(stdout.strip()))==1
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait(timeout=5)
    attempts=[driver.attempts_for(tmp_path,spec) for spec in tasks]
    assert all(len(a)==1 and a[0]['status']=='running' for a in attempts)
    assert sum(driver.cost_seconds(a[0]) for a in attempts)==14400
