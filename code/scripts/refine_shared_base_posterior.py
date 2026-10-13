"""Nine fixed posterior refinements, preserving every prior sample and decision.

The unchanged worker resumes copied chains or starts the three declared new
references. Independent raw auditing, canonical probability readback and the
uniform depth-arithmetic correction are reused. No available-case mean or
outcome-selected substitution is permitted.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import revalidate_shared_base_depth as depth_check
import recover_shared_base_posterior_audit as recovery
from prepare_shared_base_posterior_cluster import (
    ROOT, require, sha, fingerprint, atomic_json, immutable_json)

PLAN=ROOT/'code/transformer/configs/shared_base_posterior_refinement_plan_2026-09-17.json'
DEFAULT_OUT=ROOT/'output/shared_base_posterior_refinement_2026-09-17'
OWN_FILES=('code/scripts/refine_shared_base_posterior.py',
    'code/transformer/scitas/submit_shared_base_posterior_refinement.sh',
    'code/transformer/scitas/shared_base_posterior_refinement.sbatch')
CODE_FILES=tuple(sorted(set(OWN_FILES+depth_check.OWN_FILES+recovery.OWN_FILES)))
EXPECTED=(('continuation',20,16,131072,524288,2026091670),
    ('continuation',21,32,131072,524288,2026091670),
    ('continuation',25,32,131072,262144,2026091670),
    ('continuation',28,16,65536,262144,2026091670),
    ('continuation',29,32,65536,262144,2026091670),
    ('continuation',19,32,262144,524288,2026091670),
    ('quadrature',3,64,None,65536,2026091670),
    ('quadrature',7,64,None,65536,2026091670),
    ('repeat',7,32,None,65536,2026091771))
RESOURCES=dict(cpus=4,memory_gib=24,wall_seconds=28800,worker_seconds=26400,audit_seconds=1800,concurrency=9)
WORKER=dict(proposal='iid',steps=64,burn=16384,chains=4,thin=1)
REPEAT_GATE=dict(agreement_mc_multiplier=5.,bias_multiplier=2.,agreement_floor_bits=.0002,
    precision_mc_multiplier=2.,precision_budget_bits=.003333333333333333)


def read_plan(path):
    plan,digest=recovery.checked_json(path)
    require(plan.get('schema')=='shared-base-posterior-refinement-plan-v1','Wrong refinement plan schema')
    require(plan['resources']==RESOURCES and plan['worker']==WORKER,'Fixed worker/resource contract changed')
    require(plan['repeat_gate']==REPEAT_GATE,'Fixed independent-repeat gate changed')
    comparisons=plan['comparisons']
    require(comparisons['retained_source_indices']==[0,1]
            and comparisons['l0_time1024_primary_orders']==[32,64]
            and comparisons['l0_time1024_historical_orders']==[16,32], 'Fixed comparison roles changed')
    require(plan['no_auto_retry'] is True and plan['production_approved'] is False,'Unexpected production/retry authorization')
    require(len(plan['tasks'])==9 and len({t['id'] for t in plan['tasks']})==9,'Need all nine unique fixed tasks')
    for task,(kind,index,order,previous,maximum,seed) in zip(plan['tasks'],EXPECTED):
        require(re.fullmatch(r'[A-Za-z0-9_-]+',task['id']) and
            (task['kind'],task['base_task_index'],task['order'],task.get('previous_draws'),task['max_draws'],task['master_seed'])==
            (kind,index,order,previous,maximum,seed),'Refinement task identity or fixed budget changed')
        require(task['initial_draws']==(maximum if kind=='continuation' else 16384),'Initial draw budget changed')
    require(sorted(plan['code_files'])==list(CODE_FILES),'Incomplete refinement helper inventory')
    return plan,digest


def load_base(plan):
    root=(ROOT/plan['base']['revalidation_output']).resolve()
    spec,spec_sha=recovery.checked_json(root/'launch.json')
    summary,summary_sha=recovery.checked_json(root/'summary.json')
    require(spec['launch_id']==fingerprint({k:v for k,v in spec.items() if k!='launch_id'})
            and spec['code_sha256']=={name:sha(ROOT/name) for name in depth_check.OWN_FILES},'Base revalidation code/identity changed')
    require(spec['schema']=='shared-base-depth-revalidation-launch-v1' and spec['task_indices']==list(range(32))
            and spec['depths']==list(range(9)) and spec['no_sampling'] is True,'Wrong base revalidation scope')
    require(summary['schema']=='shared-base-depth-revalidation-v1' and summary['launch_sha256']==spec_sha
            and summary['complete'] is True and summary['no_sampling'] is True,'Need the complete fixed base revalidation')
    require(sha(spec['recovery_launch'])==spec['recovery_launch_sha256'],'Base recovery launch changed')
    launch,old,base_plan=recovery.read_recovery(spec['recovery_launch'])
    require([r['task'] for r in summary['tasks']]==base_plan['tasks'],'Base revalidation coverage changed')
    for task in plan['tasks']:
        base=base_plan['tasks'][task['base_task_index']]
        require(all(task[k]==base[k] for k in ('setting','index','time')),'Refinement source differs from fixed base case')
        require(task['kind']=='quadrature' or task['order']==base['order'],'Continuation/repeat exponent order changed')
    binding=dict(launch_path=str(root/'launch.json'),launch_sha256=spec_sha,
                 summary_path=str(root/'summary.json'),summary_sha256=summary_sha)
    return dict(root=root,spec=spec,summary=summary,launch=launch,old=old,plan=base_plan,binding=binding)


def verify_base_row(base,index):
    task=base['plan']['tasks'][index];row=base['summary']['tasks'][index]
    saved,_=recovery.checked_json(base['root']/'tasks'/f"{task['id']}.json")
    require(saved==row,'Base task differs from frozen summary')
    depth_check.check_row(row,task,base['binding']['launch_sha256'],base['summary']['runtime'],
        base['launch'],base['old'],base['plan'],replay_depths=False)
    paths=recovery.original.task_paths(base['old']['campaign'],task)
    reference,_=recovery.checked_json(paths['reference'])
    return row,paths,reference


def build_launch(args):
    plan,plan_hash=read_plan(args.plan);base=load_base(plan)
    out=args.out.expanduser().resolve()
    for protected in (base['root'],Path(base['launch']['context']['output']),
                      Path(base['old']['campaign']['output']),Path(base['old']['campaign']['inputs'])):
        require(out!=protected and out not in protected.parents and protected not in out.parents,
                'Refinement output must be separate from every base artifact')
    selected=list(range(9)) if args.only_task is None else sorted(set(args.only_task))
    require(selected and all(0<=i<9 for i in selected) and not(args.merge and args.only_task),'Invalid task selection')
    context=dict(schema='shared-base-refinement-context-v1',plan=str(args.plan.resolve()),plan_sha256=plan_hash,
        base=base['binding'],output=str(out),code_sha256={name:sha(ROOT/name) for name in CODE_FILES})
    launch=dict(schema='shared-base-refinement-launch-v1',context=context,mode='merge' if args.merge else 'run',
                selected_tasks=[] if args.merge else selected,venv=base['launch']['venv'])
    launch['launch_id']=fingerprint(launch);path=out/'launches'/f"{launch['launch_id'][:16]}.json"
    command=['sbatch','--parsable',f'--chdir={ROOT}','--job-name=sb-posterior-refine',
             f'--output={out}/logs/refine-%A_%a.out']
    if args.merge:command.append('--time=00:30:00')
    else:command.append('--array='+('0-8' if selected==list(range(9)) else ','.join(map(str,selected)))+'%9')
    command.extend([str(ROOT/OWN_FILES[2]),str(path),launch['venv']])
    return launch,path,command


def submit(args):
    launch,path,command=build_launch(args);print(shlex.join(command),flush=True)
    if not args.submit:print('Dry run only; no files written or jobs submitted.',flush=True);return
    require((Path(launch['venv'])/'bin/python').is_file(),'Prepared Python environment is missing')
    immutable_json(Path(launch['context']['output'])/'context.json',launch['context']);immutable_json(path,launch)
    (Path(launch['context']['output'])/'logs').mkdir(exist_ok=True)
    subprocess.run(command,cwd=ROOT,check=True)


def read_launch(path):
    launch,_=recovery.checked_json(path);context=launch['context']
    require(launch['schema']=='shared-base-refinement-launch-v1' and launch['launch_id']==
            fingerprint({k:v for k,v in launch.items() if k!='launch_id'}),'Refinement launch identity changed')
    plan,digest=read_plan(context['plan']);require(digest==context['plan_sha256'],'Refinement plan changed')
    require(context['code_sha256']=={name:sha(ROOT/name) for name in CODE_FILES},'Refinement code changed')
    base=load_base(plan);require(base['binding']==context['base'],'Frozen base revalidation changed')
    require(recovery.checked_json(Path(context['output'])/'context.json')[0]==context,'Refinement deployment changed')
    return launch,plan,base


def task_paths(context,task):
    out=Path(context['output']);work=out/'work'/task['id'];raw=work/'references'
    ref=raw/task['setting']/f"alpha{task['order']}"/f"seq_{task['index']:06d}_t{task['time']:04d}.json"
    return dict(work=work,raw_root=raw,reference=ref,checkpoint=ref.with_name(ref.stem+'_chains.npz'),
        moments=ref.with_name(ref.stem+'_moments.npz'),prepared=work/'prepared.json',
        receipt=out/'tasks'/f"{task['id']}.json",audit=out/'audits'/f"{task['id']}.json")


def prepare_work(context,task,base):
    row,old_paths,reference=verify_base_row(base,task['base_task_index']);paths=task_paths(context,task)
    source={str(old_paths[k]):sha(old_paths[k]) for k in ('reference','checkpoint','moments')}
    expected=dict(task=task,context_sha256=fingerprint(context),source_sha256=source,
                  base_row_id=row['row_id'],parent_reference_id=reference['reference_id'])
    if task['kind']=='continuation':
        require(reference['estimate']['diagnostics']['draws']==task['previous_draws'],'Parent draw count changed')
        require(reference['identity']['master_sampling_seed']==task['master_seed'],'Continuation parent seed changed')
    if paths['prepared'].exists():
        require(recovery.checked_json(paths['prepared'])[0]==expected,'Prepared refinement identity changed')
        require(task['kind']!='continuation' or all(paths[k].is_file() for k in ('reference','checkpoint','moments')),
                'Copied continuation artifacts disappeared')
        return paths
    paths['work'].mkdir(parents=True,exist_ok=True)
    if task['kind']=='continuation':
        for key in ('reference','checkpoint','moments'):
            src,dst=old_paths[key],paths[key];dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():require(sha(dst)==source[str(src)],'Unbound partial copy differs from parent')
            else:
                tmp=dst.with_name(dst.name+'.copying');shutil.copyfile(src,tmp)
                require(sha(tmp)==source[str(src)],'Continuation copy checksum failed');os.replace(tmp,dst)
    else:require(not any(paths[k].exists() for k in ('reference','checkpoint','moments')),'Unbound fresh-run artifacts exist')
    require(source=={str(old_paths[k]):sha(old_paths[k]) for k in ('reference','checkpoint','moments')},'Parent changed during copy')
    immutable_json(paths['prepared'],expected);return paths


def worker_command(context,task,base):
    paths=task_paths(context,task)
    command=[sys.executable,str(ROOT/'code/scripts/shared_base_curve_reference.py'),
        '--input',str(Path(base['old']['campaign']['inputs'])/task['setting']), '--out',str(paths['raw_root']),
        '--index',str(task['index']),'--prefix',str(task['time']),'--alpha-order',str(task['order'])]
    for key,value in dict(proposal=WORKER['proposal'],steps=WORKER['steps'],burn=WORKER['burn'],
            initial_draws=task['initial_draws'],max_draws=task['max_draws'],seed=task['master_seed']).items():
        command.extend(['--'+key.replace('_','-'),str(value)])
    return command


def validate_new_reference(task,base,reference):
    from source import seed_of
    _,_,parent=verify_base_row(base,task['base_task_index']);diag=reference['estimate']['diagnostics']
    require((reference['setting'],reference['index'],reference['time'],reference['order'])==
            (task['setting'],task['index'],task['time'],task['order']),'New reference case changed')
    identity=reference['identity']
    require(identity['inputs']==parent['identity']['inputs'] and identity['burn']==WORKER['burn']
            and identity['master_sampling_seed']==task['master_seed']
            and identity['sampling_seed']==seed_of('shared-base-curve-reference-v1',task['master_seed'],
                task['setting'],task['index'],task['time'],task['order'])
            and identity['proposal']==parent['identity']['proposal'],'New reference source/proposal/seed changed')
    require(diag['chains']==4 and diag['thin']==1 and task['initial_draws']<=diag['draws']<=task['max_draws'],
            'New reference sampling budget differs')
    if task['kind']=='continuation':
        require(identity==parent['identity'] and reference['history'][:len(parent['history'])]==parent['history']
                and diag['draws']==task['max_draws'],'Continuation did not preserve identity/history or forced budget')


def audit_task(launch_path,index):
    launch,plan,base=read_launch(launch_path);task=plan['tasks'][index];context=launch['context']
    require(index in launch['selected_tasks'],'Undeclared refinement audit')
    runtime=recovery.original.runtime_setup(base['old']['campaign']['dependency_root'])
    require(runtime==base['summary']['runtime'],'Refinement numerical runtime changed')
    from shared_base_curve_audit import audit_reference
    import shared_base_curve_audit as auditor
    from validate_shared_base_mcmc import json_safe
    paths=task_paths(context,task);reference,_=recovery.checked_json(paths['reference'])
    validate_new_reference(task,base,reference)
    before={str(paths[k]):sha(paths[k]) for k in ('reference','checkpoint','moments')};models=[]
    with recovery.canonical_model_loader(auditor,models):
        raw=audit_reference(Path(base['old']['campaign']['inputs'])/task['setting'],paths['raw_root'],
                            task['index'],task['time'],task['order'])
    require(raw['status'] in ('ready','needs_more_draws'),'New raw audit failed: '+str(raw.get('error')))
    depths=depth_check.diagnostic.summarize_depths(paths['checkpoint'],reference['estimate'])
    corrected=depth_check.adjust_depth_gate(reference,raw,depths)
    require(before=={str(paths[k]):sha(paths[k]) for k in ('reference','checkpoint','moments')},'New raw data changed during audit')
    read_launch(launch_path);verify_base_row(base,task['base_task_index'])
    result=json_safe(dict(schema='shared-base-refined-audit-v1',context_sha256=fingerprint(context),task=task,
        runtime=runtime,raw_sha256=before,raw_audit=raw,depth_diagnostics=depths,corrected=corrected,
        canonical_model_readback=models,base_row_id=base['summary']['tasks'][task['base_task_index']]['row_id']))
    atomic_json(paths['audit'],result)
    print(json.dumps(dict(task=task['id'],status=corrected['audit']['status'])),flush=True)
    return 0


def load_result(context,task,base,*,replay_depths=True):
    from shared_base_curve_audit import same
    paths=task_paths(context,task);receipt,_=recovery.checked_json(paths['receipt'])
    require(receipt['context_sha256']==fingerprint(context) and receipt['task']==task
            and receipt['status'] in ('ready','unresolved') and receipt['worker_exit_code'] in (0,2)
            and receipt['audit_exit_code']==0,'Incomplete or mismatched refinement receipt')
    require(paths['prepared'].is_file(),'Refinement preparation receipt disappeared')
    prepare_work(context,task,base)
    audit,digest=recovery.checked_json(paths['audit'])
    require(receipt['audit_sha256']==digest and audit['context_sha256']==fingerprint(context)
            and audit['task']==task and audit['runtime']==receipt['runtime']==base['summary']['runtime']
            and receipt['parent_preparation_sha256']==sha(paths['prepared']),'Refinement audit or preparation changed')
    require(audit['raw_sha256']=={str(paths[k]):sha(paths[k]) for k in ('reference','checkpoint','moments')},
            'Refined raw files changed after audit')
    reference,_=recovery.checked_json(paths['reference']);validate_new_reference(task,base,reference)
    raw=audit['raw_audit']
    require(raw['reference_id']==reference['reference_id'] and raw['report_sha256']==sha(paths['reference'])
            and same(raw['record'],reference['record'])
            and raw['recomputed_final_states'] is True and raw['recomputed_finer_grid'] is True,
            'Refined raw audit no longer matches its reference')
    depths=depth_check.diagnostic.summarize_depths(paths['checkpoint'],reference['estimate']) if replay_depths else audit['depth_diagnostics']
    require(same(depths,audit['depth_diagnostics'],atol=0,rtol=2e-12),'Saved refined depth diagnostics differ from raw arrays')
    corrected=depth_check.adjust_depth_gate(reference,raw,depths)
    require(same(corrected,audit['corrected']) and receipt['status']==
            ('ready' if corrected['audit']['ready'] else 'unresolved'),'Corrected refinement decisions changed')
    require(audit['raw_sha256']=={str(paths[k]):sha(paths[k]) for k in ('reference','checkpoint','moments')}
            and sha(paths['audit'])==digest,'Refinement artifacts changed during replay')
    return {**audit,'corrected':corrected}


def execute(launch_path,index):
    launch,plan,base=read_launch(launch_path);task=plan['tasks'][index];context=launch['context']
    require(launch['mode']=='run' and index in launch['selected_tasks'],'Undeclared refinement task')
    runtime=recovery.original.runtime_setup(base['old']['campaign']['dependency_root'])
    require(runtime==base['summary']['runtime'],'Refinement runtime differs from parent')
    out=Path(context['output']);paths=task_paths(context,task);paths['receipt'].parent.mkdir(parents=True,exist_ok=True)
    with paths['receipt'].with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        old=recovery.checked_json(paths['receipt'])[0] if paths['receipt'].exists() else None
        if old:
            require(old['context_sha256']==fingerprint(context) and old['task']==task and old['runtime']==runtime,
                    'Cannot resume changed refinement identity/runtime')
            if old['status'] in ('ready','unresolved'):load_result(context,task,base);return 0 if old['status']=='ready' else 2
            require(not old.get('progress',{}).get('checkpoint_exists') or paths['checkpoint'].is_file(),
                    'Saved refinement checkpoint disappeared')
        paths=prepare_work(context,task,base)
        run_id=f"{os.environ.get('SLURM_JOB_ID','local')}-{time.time_ns()}";log=out/'worker_logs'/task['id']/f'{run_id}.log'
        attempt=out/'attempts'/task['id']/f'{run_id}.json';started=time.monotonic();last=[-1e9]
        receipt=dict(context_sha256=fingerprint(context),task=task,runtime=runtime,status='running',
            worker_command=worker_command(context,task,base),worker_log=str(log),started_utc=recovery.original.utc(),
            parent_preparation_sha256=sha(paths['prepared']))
        def save(pid=None,elapsed=0):
            if pid is not None and elapsed-last[0]<30:return
            last[0]=elapsed;receipt.update(elapsed_seconds=time.monotonic()-started,
                progress=recovery.original.progress_record(log,paths))
            atomic_json(attempt,receipt);atomic_json(paths['receipt'],receipt)
            print(json.dumps(dict(task=task['id'],status=receipt['status'],progress=receipt['progress'])),flush=True)
        save()
        try:
            # A successful saved worker need only finish its audit after an audit interruption.
            worker_rc=old['worker_exit_code'] if old and old.get('worker_exit_code') in (0,2) else recovery.original.run_child(
                receipt['worker_command'],log,RESOURCES['worker_seconds'],save)
            receipt['worker_exit_code']=worker_rc
            if worker_rc in (0,2):
                audit_log=out/'worker_logs'/task['id']/f'{run_id}-audit.log'
                rc=recovery.original.run_child([sys.executable,str(Path(__file__).resolve()),'audit-task',
                    '--launch',str(launch_path),'--task-index',str(index)],audit_log,RESOURCES['audit_seconds'])
                receipt['audit_exit_code']=rc
                if rc==0:
                    audit,digest=recovery.checked_json(paths['audit']);receipt['audit_sha256']=digest
                    receipt['status']='ready' if audit['corrected']['audit']['ready'] else 'unresolved'
                else:receipt['status']='audit_timeout' if rc==75 else 'audit_failed'
            else:receipt['status']='timeout' if worker_rc==75 else 'worker_failed'
            read_launch(launch_path);verify_base_row(base,task['base_task_index'])
        except Exception as error:receipt.update(status='failed',error=str(error))
        receipt.update(finished_utc=recovery.original.utc(),peak_child_rss_bytes=recovery.original.process_peak_bytes());save()
        return 0 if receipt['status']=='ready' else 2 if receipt['status']=='unresolved' else 75 if 'timeout' in receipt['status'] else 1


def repeat_check(original_audit,repeat_audit,gate=REPEAT_GATE):
    """The independent run is a diagnostic, never a replacement predictor."""
    require(gate==REPEAT_GATE,'Independent-repeat gate changed')
    if repeat_audit is None:return dict(complete=False,passed=False,reason='Independent repeat missing')
    require((original_audit['index'],original_audit['time'],original_audit['order'])==
            (repeat_audit['index'],repeat_audit['time'],repeat_audit['order'])==(1,1024,32),
            'Independent repeat case/order differs')
    records=[r['record'] for r in (original_audit,repeat_audit)]
    seeds=[r['sampling_seed'] for r in records]
    require(None not in seeds and seeds[0]!=seeds[1],'Repeat must use an independent sampling seed')
    a,b=[r['precision'] for r in records]
    for p in (a,b):
        require(all(math.isfinite(p[k]) for k in ('bayes_regret_bits','mc_variance_bits2','jensen_proxy_bits'))
                and p['mc_variance_bits2']>=0 and p['jensen_proxy_bits']>=0,'Invalid repeat precision')
    sigma=math.sqrt(a['mc_variance_bits2']+b['mc_variance_bits2'])
    bias=gate['bias_multiplier']*(a['jensen_proxy_bits']+b['jensen_proxy_bits'])
    difference=abs(a['bayes_regret_bits']-b['bayes_regret_bits'])
    allowance=gate['agreement_mc_multiplier']*sigma+bias+gate['agreement_floor_bits']
    numerical=gate['precision_mc_multiplier']*sigma+bias
    ready=all(r['ready'] is True and r['status']=='ready' and all(r['record'][key] is True
        for key in ('full_prior_valid','provenance_valid','mixing_passed','row_grid_passed'))
        for r in (original_audit,repeat_audit))
    agreement=difference<=allowance; precision=numerical<=gate['precision_budget_bits']
    return dict(complete=True,passed=ready and agreement and precision,absolute_difference_bits=difference,
        agreement_allowance_bits=allowance,numerical_allowance_bits=numerical,
        precision_budget_bits=gate['precision_budget_bits'],agreement_passed=agreement,
        precision_passed=precision,both_references_ready=ready,sampling_seeds=seeds,
        note='Independent same-order diagnostic; never substituted for the original estimate. Local variance and bias diagnostics do not certify convergence.')


def merge_rows(base_rows,base_tasks,new_audits,plan):
    """Apply only the nine predeclared changes, with no incomplete-case mean."""
    from shared_base_curve_audit import aggregate_time
    require(len(base_rows)==len(base_tasks)==32 and [r['task'] for r in base_rows]==base_tasks,
            'Need all 32 original fixed identities')
    expected={t['id']:t for t in plan['tasks']}
    require(len(expected)==9 and set(new_audits)<=set(expected)
            and all(a['task']==expected[k] for k,a in new_audits.items()),'Undeclared or mismatched new audit')
    by_index={i:deepcopy(r['corrected']['audit']) for i,r in enumerate(base_rows)}
    finer={};repeat=None
    for task in plan['tasks']:
        new=new_audits.get(task['id']);audit=new['corrected']['audit'] if new else None
        if audit is not None:
            require((audit['index'],audit['time'],audit['order'])==
                    (task['index'],task['time'],task['order']),'Refined audit numerical case differs')
        if task['kind']=='continuation':
            # A declared but unfinished continuation removes the earlier estimate.
            by_index[task['base_task_index']]=audit
        elif task['kind']=='quadrature':finer[task['index']]=audit
        else:repeat=audit
    repeat_result=repeat_check(base_rows[7]['corrected']['audit'],repeat,plan['repeat_gate'])
    missing=[t['id'] for t in plan['tasks'] if t['id'] not in new_audits]
    complete=not missing;groups=[]
    for depth in (0,1,4,8):
        for t in (256,1024):
            orders=[32,64] if (depth,t)==(0,1024) else [16,32]
            selected=[by_index[i] for i,task in enumerate(base_tasks)
                      if task['depth']==depth and task['time']==t and task['order'] in orders and by_index[i] is not None]
            if orders==[32,64]:selected.extend(a for a in finer.values() if a is not None)
            numerical=aggregate_time(selected,[0,1],orders)
            if orders==[32,64]:
                numerical['order_comparison_passed_before_repeat']=numerical['passed']
                numerical['independent_repeat_passed']=repeat_result['passed']
                numerical.setdefault('checks',[]).append(dict(name='independent_same_order_repeat',passed=repeat_result['passed']))
                numerical['passed']=bool(numerical['passed'] and repeat_result['passed'])
            if not complete or not numerical['passed']:
                numerical['reported_mean_bayes_regret_bits']=None
            if not complete:numerical['release_blocked_by_incomplete_batch']=True
            groups.append(dict(depth=depth,time=t,orders=orders,numerical_audit=numerical))
    return dict(complete=complete,expected_new_tasks=9,missing_tasks=missing,groups=groups,
        repeat_check=repeat_result,all_numerical_checks_passed=complete and all(g['numerical_audit']['passed'] for g in groups))


def merge(launch_path):
    launch,plan,base=read_launch(launch_path);context=launch['context'];out=Path(context['output'])
    require(launch['mode']=='merge','Need an explicit merge launch')
    runtime=recovery.original.runtime_setup(base['old']['campaign']['dependency_root'])
    require(runtime==base['summary']['runtime'],'Merge runtime differs from the original audit')
    from validate_shared_base_mcmc import json_safe
    with (out/'merge.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out/'tasks').mkdir(exist_ok=True)
        base_rows=[verify_base_row(base,i)[0] for i in range(32)]
        new={};statuses=[];bindings={}
        for task in plan['tasks']:
            paths=task_paths(context,task)
            with paths['receipt'].with_suffix('.lock').open('a+') as guard:
                fcntl.flock(guard,fcntl.LOCK_SH | fcntl.LOCK_NB)
                row=dict(task=task,status='missing')
                if paths['receipt'].exists():
                    receipt,digest=recovery.checked_json(paths['receipt']);bindings[str(paths['receipt'])]=digest
                    require(receipt['task']==task and receipt['context_sha256']==fingerprint(context),
                            'Task receipt has a different refinement identity')
                    row.update(status=receipt['status'],receipt_sha256=digest,
                        worker_exit_code=receipt.get('worker_exit_code'),audit_exit_code=receipt.get('audit_exit_code'),
                        progress=receipt.get('progress'),error=receipt.get('error'))
                    if receipt['status'] in ('ready','unresolved'):
                        audit=load_result(context,task,base);new[task['id']]=audit
                        bindings.update(audit['raw_sha256']);bindings[str(paths['audit'])]=sha(paths['audit'])
                statuses.append(row)
                print(json.dumps(dict(task=task['id'],status=row['status'])),flush=True)
        result=json_safe(dict(schema='shared-base-posterior-refinement-summary-v1',
            context_sha256=fingerprint(context),launch_sha256=sha(launch_path),runtime=runtime,
            **merge_rows(base_rows,base['plan']['tasks'],new,plan),tasks=statuses,
            original_reference_count=32,base=base['binding'],input_sha256=bindings,
            historical_groups=base['summary']['groups'],new_audits=new,
            historical_reports_unchanged=True,production_approved=False,full_population_claim=False,
            interpretation='Fixed two-sequence-per-depth computational pilot. All original gates are retained, with uniform corrected depth arithmetic. Independent repeats are diagnostics, never selectable replacement estimates; passing checks do not certify convergence.'))
        for i in range(32):verify_base_row(base,i)
        read_launch(launch_path)
        require(bindings=={p:sha(p) for p in bindings},'Inputs changed during complete merge')
        # Preserve prior attempts (including incomplete summaries), then publish the latest view.
        run_id=f"{os.environ.get('SLURM_JOB_ID','local')}-{time.time_ns()}"
        atomic_json(out/'merges'/f'{run_id}.json',result);atomic_json(out/'summary.json',result)
        print(json.dumps(dict(complete=result['complete'],passed=result['all_numerical_checks_passed'],
                              missing_tasks=result['missing_tasks'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_subparsers(dest='mode',required=True)
    launch=modes.add_parser('launch');launch.add_argument('--plan',type=Path,default=PLAN)
    launch.add_argument('--out',type=Path,default=DEFAULT_OUT);launch.add_argument('--only-task',type=int,nargs='+')
    launch.add_argument('--merge',action='store_true');launch.add_argument('--submit',action='store_true')
    launch.add_argument('--dry-run',action='store_true')
    for mode in ('run','audit-task'):
        p=modes.add_parser(mode);p.add_argument('--launch',type=Path,required=True)
        p.add_argument('--task-index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','-1')))
    args=parser.parse_args()
    if args.mode=='launch':
        require(not(args.submit and args.dry_run),'Choose submit or dry-run');submit(args);return 0
    if args.mode=='audit-task':return audit_task(args.launch,args.task_index)
    spec,_,_=read_launch(args.launch)
    if spec['mode']=='merge':merge(args.launch);return 0
    require(0<=args.task_index<9,'Need one fixed array task index');return execute(args.launch,args.task_index)


if __name__=='__main__':sys.exit(main())
