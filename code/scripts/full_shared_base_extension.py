"""Three frozen, copied-checkpoint extensions of terminal shared-base cases.

The parent queue and its immutable numerical plan are never changed. All file
verification, copying, sampling and auditing must run on a compute node. This
driver performs one bounded allocation; an interrupted extension is explicitly
resubmitted, preserving its copied chains. Numerical failure is terminal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'code/transformer/configs/full_shared_base_extension_plan_2026-09-19.json'
TASKS=((195,32),(207,16),(232,16))
PARENT_DRAWS=524288
TARGET_DRAWS=1048576
RESOURCES=dict(cpus=8,memory_gib=48,wall_seconds=28800,worker_seconds=23400,audit_seconds=3600,concurrency=3)
FILES=('code/scripts/full_shared_base_extension.py',
       'code/transformer/scitas/submit_full_shared_base_extension.sh',
       'code/transformer/scitas/full_shared_base_extension.sbatch')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def immutable_json(path,value):
    path=Path(path)
    if path.exists():require(read(path)==value,'Immutable extension receipt differs: '+str(path))
    else:atomic_json(path,value)


def task_key(task):return task['depth'],task['index'],task['time'],task['order']


def validate_plan(plan):
    require(plan['schema']=='full-shared-base-extension-plan-v1' and plan['parent_draws']==PARENT_DRAWS and
            plan['target_draws']==TARGET_DRAWS and plan['resources']==RESOURCES and plan['tolerances_changed'] is False and
            plan['parent_plan_sha256']=='da2d156e5df8999d0a64c3e610393808e4f8da52ed13895529a53888fc2aff7e' and
            plan['parent_launch_id']=='dfc0a20712755bed14da9e81e2f761e8e57ddf8d11e639e8c0cb0fb064d35e8c',
            'Unexpected extension scope, budget or parent')
    require(len(plan['tasks'])==3 and [(s['task']['index'],s['task']['order']) for s in plan['tasks']]==list(TASKS),
            'Need exactly the three declared integration-order records')
    for spec in plan['tasks']:
        t=spec['task']
        require(t==dict(id=f"L1_i{t['index']}_t1024_a{t['order']}",depth=1,index=t['index'],time=1024,
                        order=t['order'],setting='a1_L1'),'Source/task identity differs')
        require(all(re.fullmatch('[0-9a-f]{64}',spec[k]) for k in
                    ('parent_audit_fingerprint','parent_report_sha256','parent_reference_id')),
                'Invalid parent fingerprint')
    require([s['parent_task_index'] for s in plan['tasks']]==[1791,1838,1938] and
            re.fullmatch('[0-9a-f]{64}',plan['selection_snapshot_sha256']), 'Parent snapshot/task positions differ')
    require(set(plan['code_sha256'])==set(FILES),'Extension implementation inventory differs')
    return plan


def load_context(plan_path,parent_path,out):
    require('SLURM_JOB_ID' in os.environ,'Run extension work through its compute-node launcher')
    plan=validate_plan(read(plan_path))
    for name,digest in plan['code_sha256'].items():require(sha(ROOT/name)==digest,'Extension code changed: '+name)
    import shared_base_posterior_full as full
    parent,parent_plan=full.load_launch(parent_path,check_dependencies=True)
    require(parent['launch_id']==plan['parent_launch_id'] and parent['plan_sha256']==plan['parent_plan_sha256'],
            'Frozen parent launch changed')
    out=Path(out).resolve();parent_root=Path(parent['output']).resolve()
    require(out!=parent_root and out not in parent_root.parents and parent_root not in out.parents,
            'Extension output must be separate from parent outputs')
    require(out!=Path(parent['inputs']).resolve() and out not in Path(parent['inputs']).resolve().parents and
            Path(parent['inputs']).resolve() not in out.parents,'Extension output overlaps source inputs')
    for spec in plan['tasks']:
        require(parent_plan['tasks'][spec['parent_task_index']]==spec['task'],'Frozen parent task position differs')
    context=dict(schema='full-shared-base-extension-context-v1',plan=str(Path(plan_path).resolve()),
        plan_sha256=sha(plan_path),parent_launch=str(Path(parent_path).resolve()),parent_launch_sha256=sha(parent_path),
        parent_launch_id=parent['launch_id'],output=str(out))
    immutable_json(out/'context.json',context)
    return plan,parent,parent_plan,context


def paths(out,task):
    out=Path(out);stem=f"seq_{task['index']:06d}_t{task['time']:04d}"
    report=out/'references'/task['setting']/f"alpha{task['order']}"/(stem+'.json')
    result=dict(reference=report,checkpoint=report.with_name(stem+'_chains.npz'),
        moments=report.with_name(stem+'_moments.npz'),receipt=out/'tasks'/(task['id']+'.json'),
        audit=out/'audits'/(task['id']+'.json'),prepared=out/'prepared'/(task['id']+'.json'))
    ensure_output_paths(out,result.values())
    return result


def ensure_output_paths(out,files):
    """Reject aliasing of child artifacts, including through ancestor symlinks."""
    base=Path(out).absolute();resolved=base.resolve()
    for path in files:
        path=Path(path).absolute()
        require(path.resolve()!=resolved and resolved in path.resolve().parents,'Child artifact escapes extension output')
        while path!=base:
            require(not path.is_symlink(),'Symlink in extension output: '+str(path))
            require(path!=path.parent,'Child artifact is outside extension output')
            path=path.parent


def validate_parent_bindings(plan,spec,receipt,audit,reference):
    task=spec['task'];a=audit['corrected']['audit']
    require(receipt['launch_id']==audit['launch_id']==plan['parent_launch_id'] and
            receipt['task']==audit['task']==task and receipt['status']=='unresolved',
            'Parent is not the declared terminal unresolved task')
    require(fingerprint(a)==spec['parent_audit_fingerprint'] and a['report_sha256']==spec['parent_report_sha256'] and
            a['reference_id']==spec['parent_reference_id'] and a['draws_per_chain']==PARENT_DRAWS,
            'Parent numerical audit changed since the frozen snapshot')
    require(a['ready'] is False and a['status']=='needs_more_draws' and
            a['gates']==dict(split_rhat_below_1_05=True,probability_ess_at_least_200=False,
               maximum_probability_mcse_at_most_0_005=True,raw_normalization=True),
            'Parent failure is no longer the selected probability-ESS failure')
    require((reference['setting'],reference['index'],reference['time'],reference['order'])==
            (task['setting'],task['index'],task['time'],task['order']) and
            reference['reference_id']==spec['parent_reference_id'] and
            reference['identity']['master_sampling_seed']==2026091780 and
            reference['identity']['burn']==16384 and reference['estimate']['diagnostics']['draws']==PARENT_DRAWS,
            'Parent reference identity or sampling history changed')
    return a


@contextmanager
def writer_claim(out,task):
    claim=Path(out)/'claims'/(task['id']+'.writer');ensure_output_paths(out,[claim]);claim.parent.mkdir(parents=True,exist_ok=True)
    try:claim.mkdir()
    except FileExistsError as e:raise RuntimeError('Writer claim exists; inspect its Slurm owner before retrying: '+str(claim)) from e
    atomic_json(claim/'owner.json',dict(job=os.environ.get('SLURM_JOB_ID'),pid=os.getpid(),task=task))
    try:yield
    finally:
        (claim/'owner.json').unlink();claim.rmdir()


def parent_bundle(plan,parent,parent_plan,spec,*,verify_raw):
    import shared_base_posterior_full as full
    task=spec['task'];p=full.task_paths(parent,task)
    full.verify_task_inputs(parent['inputs'],parent_plan,task)
    queue=read(Path(parent['output'])/'queue.json')
    require(queue['launch_id']==parent['launch_id'] and queue['tasks'][spec['parent_task_index']]['id']==task['id'] and
            queue['tasks'][spec['parent_task_index']]['status']=='unresolved','Parent task is active or changed')
    receipt=read(p['receipt']);audit=read(p['audit']);reference=read(p['reference'])
    require(receipt['audit_sha256']==sha(p['audit']) and sha(p['reference'])==spec['parent_report_sha256'],
            'Parent file/audit binding changed')
    validate_parent_bindings(plan,spec,receipt,audit,reference)
    expected={str(p[k]):audit['raw_sha256'][str(p[k])] for k in ('reference','checkpoint','moments')}
    require(set(expected)==set(audit['raw_sha256'])==set(audit['raw_stat']) and
            all(full.file_stat(name)==audit['raw_stat'][name] for name in expected),'Parent raw file metadata changed')
    if verify_raw:
        for name,digest in expected.items():require(sha(name)==digest,'Parent raw checksum differs: '+name)
    return p,receipt,audit,reference


def prepare(plan,parent,parent_plan,context,spec):
    import shared_base_posterior_full as full
    task=spec['task'];dest=paths(context['output'],task)
    lock=Path(parent['output'])/'claims'/(task['id']+'.lock')
    # Existing parent claims are opened without creating or modifying them.
    with lock.open('r+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original,receipt,audit,reference=parent_bundle(plan,parent,parent_plan,spec,verify_raw=True)
        expected=dict(task=task,context_sha256=fingerprint(context),parent_receipt_sha256=sha(original['receipt']),
            parent_audit_sha256=sha(original['audit']),parent_raw_sha256=audit['raw_sha256'],
            parent_reference_id=reference['reference_id'])
        if dest['prepared'].exists():
            require(read(dest['prepared'])==expected and all(dest[k].exists() for k in ('reference','checkpoint','moments')),
                    'Copied checkpoint preparation changed or disappeared')
        else:
            for key in ('reference','checkpoint','moments'):
                src,dst=original[key],dest[key];dst.parent.mkdir(parents=True,exist_ok=True)
                digest=audit['raw_sha256'][str(src)]
                print(json.dumps(dict(task=task['id'],stage='copying_parent',file=key,bytes=src.stat().st_size)),flush=True)
                if dst.exists():require(sha(dst)==digest,'Unbound existing copy differs from frozen parent')
                else:
                    temp=dst.with_name(dst.name+'.copying');shutil.copyfile(src,temp)
                    require(sha(temp)==digest,'Copied parent checksum failed');os.replace(temp,dst)
            immutable_json(dest['prepared'],expected)
        # The original remains frozen; the child is the only destination written.
        parent_bundle(plan,parent,parent_plan,spec,verify_raw=False)
    return dest,receipt['runtime'],reference


def worker_command(parent,out,task):
    cmd=[sys.executable,str(ROOT/'code/scripts/shared_base_posterior_full_worker.py'),
         '--input',str(Path(parent['inputs'])/task['setting']),'--out',str(Path(out)/'references'),
         '--index',str(task['index']),'--prefix',str(task['time']),'--alpha-order',str(task['order'])]
    for k,v in dict(proposal='iid',steps=64,burn=16384,initial_draws=TARGET_DRAWS,max_draws=TARGET_DRAWS,seed=2026091780).items():
        cmd.extend(['--'+k.replace('_','-'),str(v)])
    return cmd


def validate_continuation(reference,parent,task):
    require((reference['setting'],reference['index'],reference['time'],reference['order'])==
            (task['setting'],task['index'],task['time'],task['order']) and reference['identity']==parent['identity'] and
            reference['reference_id']==parent['reference_id'] and
            reference['history'][:len(parent['history'])]==parent['history'],
            'Continuation changed source, kernel, RNG identity or original history')
    d=reference['estimate']['diagnostics']
    require(d['draws']==TARGET_DRAWS and d['chains']==4 and d['thin']==1,
            'Continuation did not reach the fixed draw target')


def audit_task(plan_path,parent_path,out,index):
    plan,parent,parent_plan,context=load_context(plan_path,parent_path,out);spec=plan['tasks'][index];task=spec['task']
    import shared_base_posterior_full as full
    import shared_base_curve_audit as auditor
    import recover_shared_base_posterior_audit as recovery
    import revalidate_shared_base_depth as depth_check
    from validate_shared_base_mcmc import json_safe
    original,receipt,_,parent_reference=parent_bundle(plan,parent,parent_plan,spec,verify_raw=False)
    runtime=full.original.runtime_setup(parent['dependency_root'])
    require(runtime==receipt['runtime'],'Extension numerical runtime differs from parent')
    p=paths(out,task);reference=read(p['reference']);validate_continuation(reference,parent_reference,task)
    before={str(p[k]):sha(p[k]) for k in ('reference','checkpoint','moments')};observations=[]
    with recovery.canonical_model_loader(auditor,observations):
        raw=auditor.audit_reference(Path(parent['inputs'])/task['setting'],Path(out)/'references',
                                   task['index'],task['time'],task['order'])
    require(raw['status'] in ('ready','needs_more_draws'),'Independent raw audit failed: '+str(raw.get('error')))
    depths=depth_check.diagnostic.summarize_depths(p['checkpoint'],reference['estimate'])
    corrected=depth_check.adjust_depth_gate(reference,raw,depths)
    require(before=={str(p[k]):sha(p[k]) for k in ('reference','checkpoint','moments')},'Extension raw data changed during audit')
    result=json_safe(dict(schema='full-shared-base-extension-audit-v1',context_sha256=fingerprint(context),task=task,
        parent_audit_fingerprint=spec['parent_audit_fingerprint'],runtime=runtime,raw_sha256=before,
        raw_stat={name:full.file_stat(name) for name in before},raw_audit=raw,depth_diagnostics=depths,
        corrected=corrected,canonical_model_readback=observations,parent_preparation_sha256=sha(p['prepared'])))
    atomic_json(p['audit'],result)
    print(json.dumps(dict(task=task['id'],stage='audited',status=corrected['audit']['status'])),flush=True)


def load_child(context,spec):
    import shared_base_posterior_full as full
    task=spec['task'];p=paths(context['output'],task);r=read(p['receipt']);a=read(p['audit'])
    require(r['context_sha256']==a['context_sha256']==fingerprint(context) and r['task']==a['task']==task and
            r['status'] in ('ready','unresolved') and r['worker_exit_code'] in (0,2) and r['audit_exit_code']==0 and
            r['audit_sha256']==sha(p['audit']) and a['parent_audit_fingerprint']==spec['parent_audit_fingerprint'] and
            a['parent_preparation_sha256']==sha(p['prepared']), 'Extension receipt/audit binding differs')
    require(set(a['raw_sha256'])==set(a['raw_stat'])=={str(p[k]) for k in ('reference','checkpoint','moments')} and
            all(full.file_stat(name)==stat for name,stat in a['raw_stat'].items()),'Extension raw files changed after audit')
    require(sha(p['reference'])==a['corrected']['audit']['report_sha256']==a['raw_sha256'][str(p['reference'])] and
            r['status']==('ready' if a['corrected']['audit']['ready'] else 'unresolved'), 'Extension audit status differs')
    return r,a


def execute(plan_path,parent_path,out,index):
    allocation_started=time.monotonic()
    plan,parent,parent_plan,context=load_context(plan_path,parent_path,out);spec=plan['tasks'][index];task=spec['task']
    import shared_base_posterior_full as full
    p=paths(out,task)
    with writer_claim(out,task):
        if p['receipt'].exists():
            old=read(p['receipt']);require(old['context_sha256']==fingerprint(context),'Existing extension identity differs')
            if old['status'] in ('ready','unresolved'):
                load_child(context,spec);return 0 if old['status']=='ready' else 2
            require(old['status'] in ('retry','running','failed'),'Unexpected prior extension status')
        print(json.dumps(dict(task=task['id'],stage='verifying_parent',target_draws=TARGET_DRAWS)),flush=True)
        p,parent_runtime,_=prepare(plan,parent,parent_plan,context,spec)
        runtime=full.original.runtime_setup(parent['dependency_root']);require(runtime==parent_runtime,'Numerical runtime changed')
        run=f"{os.environ['SLURM_JOB_ID']}-{time.time_ns()}";log=Path(out)/'worker_logs'/task['id']/(run+'.log')
        receipt=dict(schema='full-shared-base-extension-task-v1',context_sha256=fingerprint(context),task=task,
            status='running',runtime=runtime,parent_preparation_sha256=sha(p['prepared']),worker_log=str(log))
        attempt=Path(out)/'attempts'/task['id']/(run+'.json');started=time.monotonic();last=[-1e6]
        ensure_output_paths(out,[log,attempt])
        def save(pid=None,elapsed=0):
            if pid is not None and elapsed-last[0]<30:return
            last[0]=elapsed;receipt.update(elapsed_seconds=time.monotonic()-started,progress=full.original.progress_record(log,p))
            atomic_json(p['receipt'],receipt);atomic_json(attempt,receipt)
            print(json.dumps(dict(task=task['id'],status=receipt['status'],progress=receipt['progress'])),flush=True)
        save()
        try:
            remaining=RESOURCES['wall_seconds']-(time.monotonic()-allocation_started)
            seconds=min(RESOURCES['worker_seconds'],remaining-RESOURCES['audit_seconds']-300)
            require(seconds>=60,'Insufficient allocation remains after input preparation; copied chains retained')
            rc=full.original.run_child(worker_command(parent,out,task),log,seconds,save)
            receipt['worker_exit_code']=rc
            if rc in (0,2):
                auditlog=log.with_name(log.stem+'-audit.log')
                command=[sys.executable,str(Path(__file__)),'audit','--plan',str(plan_path),'--parent-launch',str(parent_path),
                         '--out',str(out),'--index',str(index)]
                seconds=min(RESOURCES['audit_seconds'],RESOURCES['wall_seconds']-(time.monotonic()-allocation_started)-180)
                require(seconds>=60,'Insufficient allocation remains for audit; completed chains retained')
                arc=full.original.run_child(command,auditlog,seconds);receipt['audit_exit_code']=arc
                if arc==0:
                    a=read(p['audit']);receipt['audit_sha256']=sha(p['audit'])
                    receipt['status']='ready' if a['corrected']['audit']['ready'] else 'unresolved'
                else:receipt['status']='retry' if arc==75 else 'failed'
            else:receipt['status']='retry' if rc==75 else 'failed'
        except Exception as error:receipt.update(status='failed',error=str(error));save();raise
        save();return {'ready':0,'unresolved':2,'retry':75,'failed':1}[receipt['status']]


def apply_overlay(snapshot,overlay):
    """Replace only the three frozen original audit records in a copied snapshot.

    This function does not approve a scientific cohort or mutate its argument.
    The caller must replay aggregate_time over all 256 cases and both orders.
    """
    require(overlay['schema']=='full-shared-base-extension-overlay-v1' and overlay['complete'] is True and
            snapshot['launch_id']==overlay['parent_launch_id'],'Incomplete or unrelated overlay')
    replacements=overlay['replacements']
    require(len(replacements)==3 and [(r['task']['index'],r['task']['order']) for r in replacements]==list(TASKS),
            'Overlay does not contain the fixed three tasks')
    result=deepcopy(snapshot);rows={r['task']['id']:r for r in result['rows']}
    require(len(rows)==len(result['rows']),'Repeated snapshot task')
    for r in replacements:
        t=r['task'];old=rows[t['id']];a=r['audit']
        require(old['task']==t and old['status']=='unresolved' and
                fingerprint(old['audit'])==r['parent_audit_fingerprint'] and
                (a['index'],a['time'],a['order'])==(t['index'],t['time'],t['order']) and
                a['draws_per_chain']==TARGET_DRAWS and r['status']==('ready' if a['ready'] else 'unresolved'),
                'Overlay source, parent, draw budget or status differs')
        old.update(status=r['status'],audit=deepcopy(a))
    result['counts']=dict(Counter(r['status'] for r in result['rows']))
    result['extension_provenance']={k:overlay[k] for k in ('plan_sha256','parent_launch_id','context_sha256','parent_snapshot_sha256')}
    # Old group acceptance cannot certify a changed predictor. The separate merge
    # must recompute groups before any publication exporter consumes this object.
    result['groups']=[];result['all_numerical_checks_passed']=False;result['status']='requires_group_revalidation'
    return result


def snapshot_parent(parent,parent_plan):
    """Read terminal receipts without writing, resuming, or locking the live queue."""
    import shared_base_posterior_full as full
    rows=[];bindings=[]
    for task in parent_plan['tasks']:
        p=full.task_paths(parent,task)
        receipt=read(p['receipt']) if p['receipt'].exists() else None
        row=dict(task=task,status='missing' if receipt is None else receipt['status'])
        if receipt is not None:
            require(receipt['launch_id']==parent['launch_id'] and receipt['task']==task,'Live parent receipt identity changed')
            if receipt['status'] in ('ready','unresolved'):
                digest=sha(p['audit']);require(digest==receipt['audit_sha256'],'Live parent audit changed')
                a=read(p['audit'])
                require(a['launch_id']==parent['launch_id'] and a['task']==task and
                        set(a['raw_stat'])==set(a['raw_sha256']) and
                        all(full.file_stat(name)==stat for name,stat in a['raw_stat'].items()),
                        'Live parent audit/raw metadata changed')
                row['audit']=a['corrected']['audit']
                require(row['status']==('ready' if row['audit']['ready'] else 'unresolved'),'Live parent status/audit differs')
                bindings.append(dict(task_id=task['id'],receipt_sha256=fingerprint(receipt),audit_sha256=digest))
        rows.append(row)
    return dict(schema='shared-base-posterior-full-summary-v1',launch_id=parent['launch_id'],rows=rows,
                expected_new_tasks=4064,reused_pilot_sequences=8,parent_terminal_bindings=bindings)


def revalidate_groups(snapshot,pilot_audits):
    """All-source numerical acceptance; missing cases never yield a partial mean."""
    from shared_base_curve_audit import aggregate_time
    from validate_shared_base_mcmc import json_safe
    audits=dict(pilot_audits)
    for row in snapshot['rows']:
        if 'audit' in row:audits[task_key(row['task'])]=row['audit']
    groups=[]
    for depth in (0,1,4,8):
        for time_ in (256,1024):
            orders=[32,64] if (depth,time_)==(0,1024) else [16,32]
            selected=[a for (d,i,t,o),a in audits.items() if (d,t)==(depth,time_) and o in orders]
            groups.append(dict(depth=depth,time=time_,orders=orders,
                               numerical_audit=aggregate_time(selected,list(range(256)),orders)))
    result=deepcopy(snapshot);result['groups']=json_safe(groups)
    result['coverage_complete']=all(r['status'] in ('ready','unresolved','failed','timeout_exhausted') for r in result['rows'])
    result['all_numerical_checks_passed']=all(g['numerical_audit']['passed'] for g in groups)
    result['scientific_reference_validated']=result['coverage_complete'] and result['all_numerical_checks_passed']
    result['status']='validated' if result['scientific_reference_validated'] else ('failed_numerical_checks' if result['coverage_complete'] else 'incomplete')
    result['scope']='All 256 cases and both integration orders per group; only individually complete, passed groups may be published.'
    return result


def read_pilot(parent,parent_plan):
    from shared_base_long_context_results import select_references
    inputs=Path(parent['inputs'])
    for name in ('pilot/base.json','pilot/refinement.json'):
        require(sha(inputs/name)==parent_plan['inputs']['files'][name]['sha256'],'Frozen pilot inputs changed')
    return select_references(read(inputs/'pilot/base.json'),read(inputs/'pilot/refinement.json'),
        read(ROOT/'code/transformer/configs/shared_base_posterior_plan_2026-09-16.json'),
        read(ROOT/'code/transformer/configs/shared_base_posterior_refinement_plan_2026-09-17.json'))


def merge(plan_path,parent_path,out):
    plan,parent,parent_plan,context=load_context(plan_path,parent_path,out)
    replacements=[];missing=[]
    for spec in plan['tasks']:
        p=paths(out,spec['task'])
        if not p['receipt'].exists() or read(p['receipt'])['status'] not in ('ready','unresolved'):
            missing.append(spec['task']['id']);continue
        parent_bundle(plan,parent,parent_plan,spec,verify_raw=False)
        receipt,audit=load_child(context,spec)
        replacements.append(dict(task=spec['task'],status=receipt['status'],audit=audit['corrected']['audit'],
            parent_audit_fingerprint=spec['parent_audit_fingerprint'],audit_sha256=sha(p['audit']),raw_sha256=audit['raw_sha256']))
    overlay=dict(schema='full-shared-base-extension-overlay-v1',plan_sha256=sha(plan_path),context_sha256=fingerprint(context),
        parent_launch_id=parent['launch_id'],parent_snapshot_sha256=plan['selection_snapshot_sha256'],
        complete=not missing,all_extensions_ready=not missing and all(r['status']=='ready' for r in replacements),
        missing_or_unfinished_tasks=missing,replacements=replacements,
        scientific_reference_validated=False,scope='Only the three declared reference replacements; see summary.json for independent all-source cohort revalidation.')
    atomic_json(Path(out)/'overlay.json',overlay)
    target=None
    if overlay['complete']:
        snapshot=snapshot_parent(parent,parent_plan)
        # Preserve the exact readback used, rather than modifying the live summary.
        atomic_json(Path(out)/'parent_snapshot.json',snapshot)
        merged=revalidate_groups(apply_overlay(snapshot,overlay),read_pilot(parent,parent_plan))
        atomic_json(Path(out)/'summary.json',merged)
        target=next(g for g in merged['groups'] if (g['depth'],g['time'])==(1,1024))
        print(json.dumps(dict(stage='cohort_revalidation',depth=1,time=1024,
            complete=target['numerical_audit']['complete'],passed=target['numerical_audit']['passed'])),flush=True)
    print(json.dumps({k:overlay[k] for k in ('complete','all_extensions_ready','missing_or_unfinished_tasks','scientific_reference_validated')}),flush=True)
    if not overlay['complete']:return 3
    if not overlay['all_extensions_ready']:return 2
    if not target['numerical_audit']['complete']:return 3
    return 0 if target['numerical_audit']['passed'] else 2


def launch(args):
    """Small plan/scheduler operations only; no parent arrays or chains on login."""
    plan=validate_plan(read(args.plan));out=args.out.resolve()
    selected=sorted(set(args.only_task)) if args.only_task else list(range(3))
    require(not args.merge or not args.only_task,'Merge does not accept task selection')
    require(re.fullmatch('[A-Za-z0-9_-]{1,32}',args.attempt),'Invalid submission attempt name')
    require(args.after_job is None or re.fullmatch('[0-9]+',args.after_job),'Invalid dependency job')
    digest=fingerprint(dict(plan_sha256=sha(args.plan),out=str(out),parent=str(args.parent_launch.resolve())))[:10]
    name='sb-ext-'+digest+('-merge' if args.merge else '-run')
    cmd=['sbatch','--parsable','--chdir='+str(ROOT),'--job-name='+name,
         '--output='+str(out/'logs'/('merge-%j.out' if args.merge else 'extend-%A_%a.out'))]
    if args.merge:cmd+=['--time=00:30:00','--cpus-per-task=1','--mem=6G']
    else:cmd+=['--array='+','.join(map(str,selected))+'%3']
    if args.after_job:cmd+=['--dependency=afterany:'+args.after_job,'--kill-on-invalid-dep=yes']
    cmd+=[str(ROOT/FILES[2]),str(args.plan.resolve()),str(args.parent_launch.resolve()),str(out),str(args.venv.resolve()),
          'merge' if args.merge else 'run']
    print(shlex.join(cmd),flush=True)
    if not args.submit:return 0
    (out/'logs').mkdir(parents=True,exist_ok=True);(out/'submissions').mkdir(exist_ok=True)
    key=('merge' if args.merge else 'run')+'-'+args.attempt+'-'+fingerprint(cmd)[:12]
    intent=out/'submissions'/(key+'.json')
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_'))}
    with (out/'submissions'/'submission.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if intent.exists():
            prior=read(intent);require(prior['command']==cmd,'Submission intent changed')
            require(prior.get('job_id'),'Previous submission is uncertain; inspect scheduler before retrying')
            print('Already submitted: '+prior['job_id'],flush=True);return 0
        active=subprocess.run(['squeue','-h','-u',os.environ.get('USER','urbanke'),'-o','%i|%j'],
                              check=True,capture_output=True,text=True,env=env)
        require(not any(len(parts:=line.split('|'))==2 and parts[1].strip()==name for line in active.stdout.splitlines()),
                'An extension job in this mode is already queued/running; refusing a duplicate')
        value=dict(command=cmd,status='intent',name=name);atomic_json(intent,value)
        try:
            done=subprocess.run(cmd,check=True,capture_output=True,text=True,cwd=ROOT,env=env)
            require(re.fullmatch(r'\d+(?:;[A-Za-z0-9_-]+)?',done.stdout.strip()),'Ambiguous submission response')
            value.update(status='submitted',job_id=done.stdout.strip().split(';')[0],stderr=done.stderr)
            atomic_json(intent,value);print(done.stdout.strip(),flush=True)
            if done.stderr:print(done.stderr,file=sys.stderr,end='')
        except Exception as error:
            value.update(status='submission_uncertain',error=str(error));atomic_json(intent,value);raise
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('launch','run','audit','merge'))
    scratch=Path('/scratch')/os.environ.get('USER','urbanke')
    p.add_argument('--plan',type=Path,default=PLAN)
    p.add_argument('--parent-launch',type=Path,default=scratch/'shared_base_posterior_full_2026-09-17/launch.json')
    p.add_argument('--out',type=Path,default=scratch/'shared_base_full_extension_2026-09-19')
    p.add_argument('--index',type=int,choices=range(3));p.add_argument('--only-task',type=int,choices=range(3),action='append')
    p.add_argument('--merge',action='store_true');p.add_argument('--submit',action='store_true')
    p.add_argument('--after-job');p.add_argument('--attempt',default='initial');p.add_argument('--venv',type=Path,default=Path.home()/'venv-tf')
    a=p.parse_args()
    if a.mode=='launch':return launch(a)
    require(a.mode=='merge' or a.index is not None,'Task index required')
    if a.mode=='run':return execute(a.plan,a.parent_launch,a.out,a.index)
    if a.mode=='audit':audit_task(a.plan,a.parent_launch,a.out,a.index);return 0
    return merge(a.plan,a.parent_launch,a.out)


if __name__=='__main__':sys.exit(main())
