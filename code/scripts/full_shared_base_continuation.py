"""One bounded continuation of the sole unresolved September 19 extension.

The original three-case driver, numerical worker and all parent outputs remain
unchanged. Only one copied checkpoint advances; the two ready cases are reused.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import full_shared_base_extension as base

ROOT=base.ROOT
PLAN=ROOT/'code/transformer/configs/full_shared_base_continuation_plan_2026-09-20.json'
TASK=dict(id='L1_i207_t1024_a16',depth=1,index=207,time=1024,order=16,setting='a1_L1')
PARENT_DRAWS=1048576
TARGET_DRAWS=1572864
RESOURCES=dict(cpus=12,memory_gib=72,wall_seconds=28800,worker_seconds=23400,
               audit_seconds=3600,concurrency=1,maximum_attempts=1)
FILES=('code/scripts/full_shared_base_continuation.py',
       'code/transformer/scitas/submit_full_shared_base_continuation.sh',
       'code/transformer/scitas/full_shared_base_continuation.sbatch',
       'code/scripts/unseen_context_refinement.py')
require,read,sha,fingerprint=base.require,base.read,base.sha,base.fingerprint
atomic_json,immutable_json,paths=base.atomic_json,base.immutable_json,base.paths


def progress(stage,**details):
    print(json.dumps(dict(task=TASK['id'],stage=stage,target_draws=TARGET_DRAWS,**details)),flush=True)


def validate_plan(plan):
    require(plan['schema']=='full-shared-base-continuation-plan-v1' and plan['task']==TASK
            and plan['parent_draws']==PARENT_DRAWS and plan['target_draws']==TARGET_DRAWS
            and plan['resources']==RESOURCES and plan['tolerances_changed'] is False,
            'Continuation scope, budget or thresholds changed')
    require(plan['previous_plan_sha256']==sha(base.PLAN)
            and plan['previous_task_index']==1 and set(plan['code_sha256'])==set(FILES),
            'Previous extension or implementation inventory changed')
    require(all(re.fullmatch('[0-9a-f]{64}',plan[k]) for k in
                ('previous_context_sha256','parent_audit_fingerprint','parent_reference_id')),
            'Invalid continuation ancestry')
    previous=base.validate_plan(read(base.PLAN))
    expected={'context.json'}
    for spec in previous['tasks']:
        p=paths(Path('/parent'),spec['task'])
        expected.update(str(p[k].relative_to('/parent')) for k in ('receipt','audit','reference','prepared'))
    require(set(plan['previous_files'])==expected,'Need all three frozen previous results')
    return plan


def previous_bundle(plan,previous_context,previous_plan,index,*,verify_raw=False):
    """Read a terminal extension result without touching its receipts or claims."""
    spec=previous_plan['tasks'][index];p=paths(previous_context['output'],spec['task'])
    require(not (Path(previous_context['output'])/'claims'/(spec['task']['id']+'.writer')).exists(),
            'Previous extension still has a writer claim')
    receipt,audit=base.load_child(previous_context,spec)
    for name in ('receipt','audit','reference','prepared'):
        relative=str(p[name].relative_to(previous_context['output']))
        require(sha(p[name])==plan['previous_files'][relative],'Frozen previous result changed: '+relative)
    reference=read(p['reference']);a=audit['corrected']['audit']
    require(a['draws_per_chain']==PARENT_DRAWS and reference['estimate']['diagnostics']['draws']==PARENT_DRAWS,
            'Previous retained draw count changed')
    if index==1:
        require(receipt['status']=='unresolved' and fingerprint(a)==plan['parent_audit_fingerprint']
                and reference['reference_id']==plan['parent_reference_id'] and a['ready'] is False
                and a['status']=='needs_more_draws' and a['gates']==dict(split_rhat_below_1_05=True,
                    probability_ess_at_least_200=False,maximum_probability_mcse_at_most_0_005=True,
                    raw_normalization=True),'Selected parent is no longer the frozen ESS-only failure')
    else:require(receipt['status']=='ready' and a['ready'],'A reused ready result changed')
    if verify_raw:
        for name,digest in audit['raw_sha256'].items():require(sha(name)==digest,'Previous raw checksum changed: '+name)
    return p,receipt,audit,reference


def load_context(plan_path,parent_path,previous_out,out):
    require('SLURM_JOB_ID' in os.environ,'Run continuation through its compute-node launcher')
    plan=validate_plan(read(plan_path));previous_plan=base.validate_plan(read(base.PLAN))
    for name,digest in {**previous_plan['code_sha256'],**plan['code_sha256']}.items():
        require(sha(ROOT/name)==digest,'Frozen implementation changed: '+name)
    import shared_base_posterior_full as full
    parent,parent_plan=full.load_launch(parent_path,check_dependencies=True)
    require(parent['launch_id']==previous_plan['parent_launch_id'] and parent['plan_sha256']==previous_plan['parent_plan_sha256'],
            'Original full-cohort launch changed')
    previous_out=Path(previous_out).resolve();previous_context=read(previous_out/'context.json')
    require(sha(previous_out/'context.json')==plan['previous_files']['context.json']
            and fingerprint(previous_context)==plan['previous_context_sha256']
            and previous_context['output']==str(previous_out)
            and previous_context['plan_sha256']==plan['previous_plan_sha256']
            and previous_context['parent_launch_sha256']==sha(parent_path)
            and previous_context['parent_launch_id']==parent['launch_id'],'Previous context binding changed')
    out=Path(out).resolve()
    for protected in (previous_out,Path(parent['output']).resolve(),Path(parent['inputs']).resolve()):
        require(out!=protected and out not in protected.parents and protected not in out.parents,
                'Continuation output overlaps a protected parent')
    for i,spec in enumerate(previous_plan['tasks']):
        require(parent_plan['tasks'][spec['parent_task_index']]==spec['task'],'Original task position changed')
        base.parent_bundle(previous_plan,parent,parent_plan,spec,verify_raw=False)
        previous_bundle(plan,previous_context,previous_plan,i)
    context=dict(schema='full-shared-base-continuation-context-v1',plan=str(Path(plan_path).resolve()),
        plan_sha256=sha(plan_path),parent_launch=str(Path(parent_path).resolve()),parent_launch_sha256=sha(parent_path),
        parent_launch_id=parent['launch_id'],previous_output=str(previous_out),
        previous_context_sha256=fingerprint(previous_context),output=str(out))
    immutable_json(out/'context.json',context)
    return plan,parent,parent_plan,context,previous_context,previous_plan


def prepare(plan,context,previous_context,previous_plan):
    original,receipt,audit,reference=previous_bundle(plan,previous_context,previous_plan,1,verify_raw=True)
    dest=paths(context['output'],TASK)
    expected=dict(task=TASK,context_sha256=fingerprint(context),parent_receipt_sha256=sha(original['receipt']),
        parent_audit_sha256=sha(original['audit']),parent_raw_sha256=audit['raw_sha256'],
        parent_reference_id=reference['reference_id'],baseline_draws=PARENT_DRAWS,target_draws=TARGET_DRAWS)
    require(not dest['prepared'].exists(),'This continuation allows one allocation only')
    for key in ('reference','checkpoint','moments'):
        src,dst=original[key],dest[key];dst.parent.mkdir(parents=True,exist_ok=True)
        require(not dst.exists(),'Unbound destination already exists')
        temporary=dst.with_name(dst.name+'.copying');shutil.copyfile(src,temporary)
        require(sha(temporary)==audit['raw_sha256'][str(src)],'Isolated parent copy differs')
        os.replace(temporary,dst)
    immutable_json(dest['prepared'],expected)
    previous_bundle(plan,previous_context,previous_plan,1)
    return dest,receipt['runtime'],reference


def worker_command(parent,out):
    cmd=base.worker_command(parent,out,TASK)
    for flag in ('--initial-draws','--max-draws'):cmd[cmd.index(flag)+1]=str(TARGET_DRAWS)
    return cmd


def validate_continuation(reference,parent):
    require(reference['identity']==parent['identity'] and reference['reference_id']==parent['reference_id']
            and reference['history'][:len(parent['history'])]==parent['history']
            and (reference['setting'],reference['index'],reference['time'],reference['order'])==('a1_L1',207,1024,16),
            'Continuation changed source, prior, proposal, RNG identity or original history')
    d=reference['estimate']['diagnostics']
    require(d['draws']==TARGET_DRAWS and d['chains']==4 and d['thin']==1,'Incomplete fixed draw target')


def audit_task(plan_path,parent_path,previous_out,out):
    progress('audit_starting',resources=RESOURCES)
    plan,parent,_,context,previous_context,previous_plan=load_context(plan_path,parent_path,previous_out,out)
    import shared_base_posterior_full as full
    import shared_base_curve_audit as auditor
    import recover_shared_base_posterior_audit as recovery
    import revalidate_shared_base_depth as depth_check
    from unseen_context_refinement import verify_preserved_samples
    from validate_shared_base_mcmc import json_safe
    original,receipt,_,parent_reference=previous_bundle(plan,previous_context,previous_plan,1)
    runtime=full.original.runtime_setup(parent['dependency_root'])
    require(runtime==receipt['runtime'],'Numerical runtime differs from the saved parent')
    p=paths(out,TASK);reference=read(p['reference']);validate_continuation(reference,parent_reference)
    before={str(p[k]):sha(p[k]) for k in ('reference','checkpoint','moments')};observations=[]
    verify_preserved_samples(original['checkpoint'],p['checkpoint'],
                             dict(baseline_draws=PARENT_DRAWS,target_draws=TARGET_DRAWS),complete=True)
    with recovery.canonical_model_loader(auditor,observations):
        raw=auditor.audit_reference(Path(parent['inputs'])/TASK['setting'],Path(out)/'references',207,1024,16)
    require(raw['status'] in ('ready','needs_more_draws'),'Independent raw audit failed: '+str(raw.get('error')))
    depths=depth_check.diagnostic.summarize_depths(p['checkpoint'],reference['estimate'])
    corrected=depth_check.adjust_depth_gate(reference,raw,depths)
    require(before=={str(p[k]):sha(p[k]) for k in ('reference','checkpoint','moments')},'Raw data changed during audit')
    result=json_safe(dict(schema='full-shared-base-extension-audit-v1',context_sha256=fingerprint(context),task=TASK,
        parent_audit_fingerprint=plan['parent_audit_fingerprint'],runtime=runtime,raw_sha256=before,
        raw_stat={name:full.file_stat(name) for name in before},raw_audit=raw,depth_diagnostics=depths,
        corrected=corrected,canonical_model_readback=observations,parent_preparation_sha256=sha(p['prepared']),
        original_sample_prefix_verified=True))
    previous_bundle(plan,previous_context,previous_plan,1)
    atomic_json(p['audit'],result)
    progress('audit_finished',status=corrected['audit']['status'])


def execute(plan_path,parent_path,previous_out,out):
    allocation_started=time.monotonic()
    progress('run_starting',resources=RESOURCES,output=str(out))
    plan,parent,_,context,previous_context,previous_plan=load_context(plan_path,parent_path,previous_out,out)
    import shared_base_posterior_full as full
    p=paths(out,TASK)
    with base.writer_claim(out,TASK):
        allocation=Path(out)/'allocation.json'
        require(not allocation.exists(),'One allocation already used; no automatic retry or budget increase')
        immutable_json(allocation,dict(context_sha256=fingerprint(context),job=os.environ['SLURM_JOB_ID'],
                                       wall_seconds=RESOURCES['wall_seconds'],started_unix=time.time()))
        p,parent_runtime,_=prepare(plan,context,previous_context,previous_plan)
        runtime=full.original.runtime_setup(parent['dependency_root']);require(runtime==parent_runtime,'Numerical runtime changed')
        log=Path(out)/'worker_logs'/(TASK['id']+'.log');last=[-1e6]
        receipt=dict(schema='full-shared-base-extension-task-v1',context_sha256=fingerprint(context),task=TASK,
            status='running',runtime=runtime,parent_preparation_sha256=sha(p['prepared']),worker_log=str(log))
        def save(pid=None,elapsed=0):
            if pid is not None and elapsed-last[0]<30:return
            last[0]=elapsed;receipt.update(elapsed_seconds=time.monotonic()-allocation_started,
                                         progress=full.original.progress_record(log,p))
            atomic_json(p['receipt'],receipt)
        save()
        try:
            seconds=min(RESOURCES['worker_seconds'],RESOURCES['wall_seconds']-(time.monotonic()-allocation_started)-3900)
            require(seconds>=60,'Insufficient allocation after preparation; copied chains retained')
            progress('sampling_starting',parent_draws=PARENT_DRAWS,maximum_seconds=seconds,resources=RESOURCES)
            rc=full.original.run_child(worker_command(parent,out),log,seconds,save);receipt['worker_exit_code']=rc
            progress('sampling_finished',exit_code=rc,elapsed_seconds=time.monotonic()-allocation_started)
            if rc in (0,2):
                command=[sys.executable,str(Path(__file__)),'audit','--plan',str(plan_path),'--parent-launch',str(parent_path),
                         '--previous-output',str(previous_out),'--out',str(out)]
                seconds=min(3600,28800-(time.monotonic()-allocation_started)-180)
                require(seconds>=60,'Insufficient remaining audit time; completed chains retained')
                progress('audit_dispatching',maximum_seconds=seconds,resources=RESOURCES)
                arc=full.original.run_child(command,log.with_name(TASK['id']+'-audit.log'),seconds)
                progress('audit_returned',exit_code=arc,elapsed_seconds=time.monotonic()-allocation_started)
                receipt['audit_exit_code']=arc
                if arc==0:
                    a=read(p['audit']);receipt.update(audit_sha256=sha(p['audit']),
                        status='ready' if a['corrected']['audit']['ready'] else 'unresolved')
                else:receipt['status']='time_limit' if arc==75 else 'failed'
            else:receipt['status']='time_limit' if rc==75 else 'failed'
        except Exception as error:receipt.update(status='failed',error=str(error));save();raise
        save();return {'ready':0,'unresolved':2,'time_limit':75,'failed':1}[receipt['status']]


def apply_overlay(snapshot,previous_overlay,replacement,plan):
    """Compose both generations; retain both ready records and every other case."""
    result=base.apply_overlay(snapshot,previous_overlay)
    rows={r['task']['id']:r for r in result['rows']};old=rows[TASK['id']];a=replacement['audit']
    require(old['status']=='unresolved' and fingerprint(old['audit'])==plan['parent_audit_fingerprint']
            and replacement['task']==TASK and replacement['parent_audit_fingerprint']==plan['parent_audit_fingerprint']
            and (a['index'],a['time'],a['order'])==(207,1024,16) and a['draws_per_chain']==TARGET_DRAWS
            and replacement['status']==('ready' if a['ready'] else 'unresolved'), 'Continuation overlay ancestry differs')
    old.update(status=replacement['status'],audit=deepcopy(a));result['counts']=dict(Counter(r['status'] for r in result['rows']))
    result['continuation_provenance']=dict(previous_context_sha256=plan['previous_context_sha256'],
        parent_audit_fingerprint=plan['parent_audit_fingerprint'],replacement=deepcopy(replacement))
    return result


def merge(plan_path,parent_path,previous_out,out):
    progress('merge_starting',output=str(out),resources=dict(cpus=1,memory_gib=6,wall_seconds=1800))
    plan,parent,parent_plan,context,previous_context,previous_plan=load_context(plan_path,parent_path,previous_out,out)
    replacements=[]
    for i,spec in enumerate(previous_plan['tasks']):
        p,r,a,_=previous_bundle(plan,previous_context,previous_plan,i)
        replacements.append(dict(task=spec['task'],status=r['status'],audit=a['corrected']['audit'],
            parent_audit_fingerprint=spec['parent_audit_fingerprint'],audit_sha256=sha(p['audit']),raw_sha256=a['raw_sha256']))
    previous_overlay=dict(schema='full-shared-base-extension-overlay-v1',complete=True,replacements=replacements,
        parent_launch_id=parent['launch_id'],plan_sha256=sha(base.PLAN),context_sha256=fingerprint(previous_context),
        parent_snapshot_sha256=previous_plan['selection_snapshot_sha256'])
    p=paths(out,TASK);complete=p['receipt'].exists() and read(p['receipt'])['status'] in ('ready','unresolved')
    overlay=dict(schema='full-shared-base-continuation-overlay-v1',plan_sha256=sha(plan_path),
        context_sha256=fingerprint(context),previous_overlay=previous_overlay,complete=bool(complete),
        scientific_reference_validated=False)
    if complete:
        r,a=base.load_child(context,dict(task=TASK,parent_audit_fingerprint=plan['parent_audit_fingerprint']))
        overlay['replacement']=dict(task=TASK,status=r['status'],audit=a['corrected']['audit'],
            parent_audit_fingerprint=plan['parent_audit_fingerprint'],audit_sha256=sha(p['audit']),raw_sha256=a['raw_sha256'])
        snapshot=base.snapshot_parent(parent,parent_plan);atomic_json(Path(out)/'parent_snapshot.json',snapshot)
        result=base.revalidate_groups(apply_overlay(snapshot,previous_overlay,overlay['replacement'],plan),
                                      base.read_pilot(parent,parent_plan))
        atomic_json(Path(out)/'summary.json',result)
    atomic_json(Path(out)/'overlay.json',overlay)
    progress('merge_finished',complete=bool(complete))
    if not complete:return 3
    target=next(g for g in result['groups'] if (g['depth'],g['time'])==(1,1024))['numerical_audit']
    return 0 if target['passed'] else 2 if target['complete'] else 3


def submission(command,out,key,name):
    """Idempotent receipt; uncertain sbatch responses never cause blind retries."""
    (out/'submissions').mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    intent=out/'submissions'/(key+'.json');env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_'))}
    with (out/'submissions'/'submission.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if intent.exists():
            prior=read(intent);require(prior['command']==command and prior.get('job_id'),
                                      'Prior submission changed or is uncertain; inspect scheduler')
            return prior['job_id']
        active=subprocess.run(['squeue','-h','-u',os.environ.get('USER','urbanke'),'-o','%i|%j'],
                               check=True,capture_output=True,text=True,env=env)
        require(not any(line.split('|')[-1].strip()==name for line in active.stdout.splitlines()),'Continuation already active')
        receipt=dict(command=command,name=name,status='intent');immutable_json(intent,receipt)
        try:
            done=subprocess.run(command,check=True,capture_output=True,text=True,cwd=ROOT,env=env)
            require(re.fullmatch(r'\d+(?:;[A-Za-z0-9_-]+)?',done.stdout.strip()),'Ambiguous submission response')
            receipt.update(status='submitted',job_id=done.stdout.strip().split(';')[0],stderr=done.stderr)
            atomic_json(intent,receipt);return receipt['job_id']
        except Exception as error:
            receipt.update(status='submission_uncertain',error=str(error));atomic_json(intent,receipt);raise


def launch(args):
    validate_plan(read(args.plan));out=args.out.resolve();worker=None
    digest=fingerprint(dict(plan_sha256=sha(args.plan),out=str(out),parent=str(args.parent_launch.resolve()),
                            previous=str(args.previous_output.resolve()),venv=str(args.venv.resolve())))[:12]
    for mode in ('run','merge'):
        name='sb-cont-'+digest+'-'+mode
        command=['sbatch','--parsable','--chdir='+str(ROOT),'--job-name='+name,
                 '--output='+str(out/'logs'/(mode+'-%j.out'))]
        if mode=='merge':command+=['--time=00:30:00','--cpus-per-task=1','--mem=6G',
                                  '--dependency=afterany:'+(worker or 'WORKER_JOB'),'--kill-on-invalid-dep=yes']
        command+=[str(ROOT/FILES[2]),str(args.plan.resolve()),str(args.parent_launch.resolve()),
                  str(args.previous_output.resolve()),str(out),str(args.venv.resolve()),mode]
        print(shlex.join(command),flush=True)
        if args.submit:
            job=submission(command,out,mode+'-'+digest,name)
            print(mode+' job: '+job,flush=True)
            if mode=='run':worker=job
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('launch','run','audit','merge'))
    scratch=Path('/scratch')/os.environ.get('USER','urbanke')
    p.add_argument('--plan',type=Path,default=PLAN)
    p.add_argument('--parent-launch',type=Path,default=scratch/'shared_base_posterior_full_2026-09-17/launch.json')
    p.add_argument('--previous-output',type=Path,default=scratch/'shared_base_full_extension_2026-09-19')
    p.add_argument('--out',type=Path,default=scratch/'shared_base_full_continuation_2026-09-20')
    p.add_argument('--venv',type=Path,default=Path.home()/'venv-tf');p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    if args.mode=='launch':return launch(args)
    return {'run':execute,'audit':audit_task,'merge':merge}[args.mode](args.plan,args.parent_launch,args.previous_output,args.out)


if __name__=='__main__':sys.exit(main())
