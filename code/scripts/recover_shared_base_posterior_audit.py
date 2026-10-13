"""Audit-only recovery of a redundant cross-platform probability equality check.

The canonical predictor remains the frozen saved probability vector. The
original independent loader first validates checkpoints, logits, hashes and
causality. Its reconstructed exp(logp) must then agree within BOTH 1e-15 and
one ULP before that saved vector is returned to the unchanged raw auditor.
No posterior sample, prediction, diagnostic gate or failed receipt is edited.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shlex
import socket
import subprocess
import sys
import time

import shared_base_posterior_cluster as original
from prepare_shared_base_posterior_cluster import (
    ROOT, DEPTHS, TIMES, ORDERS, INDICES, require, sha, fingerprint, atomic_json,
    immutable_json)

DEFAULT_OUT = ROOT/'output/shared_base_posterior_audit_recovery_2026-09-17'
ABS_TOL = 1e-15
MAX_ULP = 1
KNOWN_ERROR = 'model predictions differ between loaders'
OWN_FILES = ('code/scripts/recover_shared_base_posterior_audit.py',
             'code/transformer/scitas/submit_shared_base_posterior_recovery.sh',
             'code/transformer/scitas/shared_base_posterior_recovery.sbatch')


def checked_json(path):
    raw = Path(path).read_bytes()
    import hashlib
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def discrepancy(reconstructed, saved):
    """A rounding allowance only; not a prediction-accuracy tolerance."""
    import numpy as np
    left, right = np.asarray(reconstructed), np.asarray(saved)
    require(left.dtype == right.dtype == np.dtype('float64') and left.shape == right.shape == (64,),
            'Reconstructed/canonical probability shape or dtype changed')
    require(np.isfinite(left).all() and np.isfinite(right).all() and np.all(left > 0) and np.all(right > 0),
            'Invalid reconstructed/canonical probabilities')
    require(abs(float(left.sum())-1) <= 1e-12 and abs(float(right.sum())-1) <= 1e-12,
            'Unnormalized reconstructed/canonical probabilities')
    difference = np.abs(left-right)
    # Positive IEEE float64 numbers are ordered by their unsigned bit pattern.
    a, b = left.view(np.uint64), right.view(np.uint64)
    ulp = np.maximum(a,b)-np.minimum(a,b)
    result = dict(max_abs=float(difference.max()), max_relative=float((difference/right).max()),
                  max_ulp=int(ulp.max()), unequal_coordinates=int(np.count_nonzero(difference)),
                  absolute_tolerance=ABS_TOL, ulp_limit=MAX_ULP)
    require(result['max_abs'] <= ABS_TOL and result['max_ulp'] <= MAX_ULP,
            'Reconstructed probabilities exceed the rounding-only allowance')
    return result


@contextmanager
def canonical_model_loader(auditor, observations):
    """Scope the adapter to the redundant loader comparison, never NumPy."""
    independent_loader = auditor.load_models
    def adapted(audit, folder, source, index, prefix):
        import numpy as np
        folder = Path(folder)
        paths = [folder/'models'/name/f'seq_{index:06d}.{suffix}'
                 for name in ('d64','d64_seed1') for suffix in ('json','npz')]
        before = {str(path):sha(path) for path in paths}
        reconstructed = independent_loader(audit,folder,source,index,prefix)
        require(set(reconstructed) == {'d64','d64_seed1'}, 'Independent model loader coverage changed')
        canonical = {}
        for name in ('d64','d64_seed1'):
            path = folder/'models'/name/f'seq_{index:06d}.json'
            record, _ = checked_json(path)
            array_path = path.with_name(record['arrays_file'])
            require(Path(record['arrays_file']).name == record['arrays_file']
                    and array_path == path.with_suffix('.npz'), 'Noncanonical model array path')
            require(sha(array_path) == record['arrays_sha256'], 'Saved prediction file changed')
            j = record['prefixes'].index(prefix)
            with np.load(array_path,allow_pickle=False) as arrays:
                saved = arrays['predictive_probs'][j].copy()
            require(np.array_equal(saved,np.asarray(record['predictive_probs'][j])),
                    'Canonical probability JSON/array readback changed')
            detail = discrepancy(reconstructed[name],saved)
            observations.append(dict(model=name,index=index,time=prefix,**detail,
                                     model_record_sha256=before[str(path)],
                                     array_sha256=before[str(array_path)],
                                     scoring_probabilities='unchanged saved predictive_probs'))
            canonical[name] = saved
        require(before == {str(path):sha(path) for path in paths}, 'Model artifacts changed during independent loading')
        return canonical
    auditor.load_models = adapted
    try:
        yield
    finally:
        auditor.load_models = independent_loader


def original_launch(path, original_out):
    if path is not None:
        return Path(path).expanduser().resolve()
    candidates = []
    for candidate in (Path(original_out)/'launches').glob('*.json'):
        value, _ = checked_json(candidate)
        if value.get('mode') == 'pilot' and value.get('selected_tasks') == list(range(32)):
            candidates.append(candidate.resolve())
    require(len(candidates) == 1, 'Need one original full-pilot launch; specify --original-launch explicitly')
    return candidates[0]


def build_launch(args):
    old_path = original_launch(args.original_launch,args.original_out)
    old, plan = original.read_launch(old_path)
    require(old['mode'] == 'pilot' and old['selected_tasks'] == list(range(32)),
            'Recovery requires the original fixed 32-task pilot')
    selected = list(range(32)) if args.only_task is None else sorted(set(args.only_task))
    require(selected and all(0 <= i < 32 for i in selected), 'Task selection outside fixed pilot')
    require(not args.merge or args.only_task is None, 'Merge does not select tasks')
    require(1 <= args.concurrency <= 32,'Concurrency must be 1..32')
    out, old_out = args.out.expanduser().resolve(), Path(old['campaign']['output']).resolve()
    require(out != old_out and out not in old_out.parents and old_out not in out.parents,
            'Recovery output must be separate from original artifacts')
    inputs = Path(old['campaign']['inputs']).resolve()
    require(out != inputs and out not in inputs.parents and inputs not in out.parents,
            'Recovery output must be separate from frozen inputs')
    context = dict(schema='shared-base-posterior-recovery-context-v1',
        original_launch=str(old_path), original_launch_sha256=sha(old_path),
        original_plan_sha256=old['campaign']['plan_sha256'], output=str(out), repo_root=str(ROOT),
        code_sha256={name:sha(ROOT/name) for name in OWN_FILES},
        adapter=dict(absolute_tolerance=ABS_TOL,max_ulp=MAX_ULP,canonical='unchanged saved predictive_probs'),
        no_sampling=True, original_files_immutable=True)
    launch = dict(schema='shared-base-posterior-recovery-launch-v1',context=context,
                  mode='merge' if args.merge else 'audit',selected_tasks=[] if args.merge else selected,
                  concurrency=args.concurrency,venv=str(args.venv.expanduser().resolve()))
    launch['launch_id'] = fingerprint(launch)
    path = out/'launches'/f"{launch['launch_id'][:16]}.json"
    command = ['sbatch','--parsable',f'--chdir={ROOT}', '--job-name=sb-posterior-recovery',
               f'--output={out}/logs/recovery-%A_%a.out']
    if not args.merge:
        selection = '0-31' if selected == list(range(32)) else ','.join(map(str,selected))
        command.append(f'--array={selection}%{args.concurrency}')
    command += [str(ROOT/OWN_FILES[2]),str(path),launch['venv']]
    return launch,path,command


def submit(args):
    launch,path,command = build_launch(args)
    print(shlex.join(command),flush=True)
    if not args.submit:
        print('Dry run only; no files written or jobs submitted.',flush=True)
        return
    require((Path(launch['venv'])/'bin/python').is_file(),'Prepared Python environment is missing')
    out = Path(launch['context']['output'])
    immutable_json(out/'context.json',launch['context'])
    immutable_json(path,launch)
    (out/'logs').mkdir(exist_ok=True)
    done = subprocess.run(command,cwd=ROOT,check=True,text=True,capture_output=True)
    print(done.stdout,end='',flush=True)
    if done.stderr:
        print(done.stderr,file=sys.stderr,end='',flush=True)
    atomic_json(out/'submissions'/f'{time.time_ns()}.json',dict(launch_id=launch['launch_id'],
        submitted_utc=original.utc(),command=command,scheduler_response=done.stdout.strip()))


def read_recovery(path):
    launch, _ = checked_json(path)
    require(launch.get('schema') == 'shared-base-posterior-recovery-launch-v1'
            and launch.get('launch_id') == fingerprint({k:v for k,v in launch.items() if k != 'launch_id'}),
            'Recovery launch checksum/schema mismatch')
    context = launch['context']
    require(context['repo_root'] == str(ROOT) and context['no_sampling'] is True
            and context['original_files_immutable'] is True,'Recovery context changed')
    require(context['code_sha256'] == {name:sha(ROOT/name) for name in OWN_FILES},'Recovery implementation changed')
    require(context['adapter'] == dict(absolute_tolerance=ABS_TOL,max_ulp=MAX_ULP,
            canonical='unchanged saved predictive_probs'),'Rounding adapter contract changed')
    require(sha(context['original_launch']) == context['original_launch_sha256'],'Original launch changed')
    old,plan = original.read_launch(context['original_launch'])
    require(old['campaign']['plan_sha256'] == context['original_plan_sha256'],'Original plan changed')
    require(checked_json(Path(context['output'])/'context.json')[0] == context,'Recovery context file changed')
    require(launch['mode'] in ('audit','merge') and (launch['mode'] != 'merge' or not launch['selected_tasks']),
            'Unknown recovery mode')
    return launch,old,plan


def original_snapshot(old,plan,task):
    campaign = old['campaign']
    paths = original.task_paths(campaign,task)
    receipt, receipt_hash = checked_json(paths['receipt'])
    audit, audit_hash = checked_json(paths['audit'])
    require(receipt['identity'] == dict(plan_sha256=campaign['plan_sha256'],task=task,
            deployment_sha256=fingerprint(campaign)),'Original task identity mismatch')
    require(receipt['sampling_complete'] is True and receipt['worker_exit_code'] in (0,2)
            and receipt['status'] == 'audit_failed','Original task is not a completed audit failure')
    require(receipt['audit_sha256'] == audit_hash,'Original failed audit changed')
    require(audit['task'] == task and audit['plan_sha256'] == campaign['plan_sha256']
            and audit['audit_code_sha256'] == plan['code_sha256']['code/scripts/shared_base_curve_audit.py'],
            'Original audit identity/code mismatch')
    result = audit['result']
    require((result['index'],result['time'],result['order']) == (task['index'],task['time'],task['order'])
            and result['status'] == 'invalid' and result['error'] == KNOWN_ERROR,
            'Original failure is outside this specific recovery')
    raw = {str(paths[key]):sha(paths[key]) for key in ('reference','checkpoint','moments')}
    require(audit['raw_sha256'] == raw,'Original raw reference changed after failed audit')
    return dict(files={str(paths['receipt']):receipt_hash,str(paths['audit']):audit_hash,**raw},
                runtime=receipt['runtime'],original_error=KNOWN_ERROR)


def validate_recovered(report,context,old,plan,task):
    from shared_base_curve_audit import same
    require(report.get('schema') == 'shared-base-posterior-recovered-audit-v1'
            and report['context_sha256'] == fingerprint(context) and report['task'] == task,
            'Recovered task identity mismatch')
    require(report['original'] == original_snapshot(old,plan,task),'Original artifacts changed after recovery')
    require(report['runtime'] == report['original']['runtime'],'Recovered runtime differs from original')
    result = report['result']
    paths = original.task_paths(old['campaign'],task)
    reference, reference_hash = checked_json(paths['reference'])
    require((result['index'],result['time'],result['order']) == (task['index'],task['time'],task['order']),
            'Recovered raw audit case mismatch')
    require(result['report'] == str(paths['reference'].resolve())
            and result['report_sha256'] == reference_hash
            and result['reference_id'] == reference['reference_id']
            and same(result['record'],reference['record'])
            and result['draws_per_chain'] == reference['estimate']['diagnostics']['draws']
            and result['ready'] == reference['ready_for_cohort_check']
            and result['recomputed_final_states'] is True and result['recomputed_finer_grid'] is True,
            'Recovered audit differs from the hash-bound original reference')
    require(result['status'] in ('ready','needs_more_draws') and report['status'] ==
            ('ready' if result['status']=='ready' else 'unresolved'),'Incomplete or failed recovered raw audit')
    require(result['ready'] == (result['status']=='ready'),'Recovered readiness/status mismatch')
    require(len(report['model_readback']) == 2 and {d['model'] for d in report['model_readback']} == {'d64','d64_seed1'},
            'Incomplete independent model readback')
    for d in report['model_readback']:
        require(d['index']==task['index'] and d['time']==task['time']
                and d['absolute_tolerance']==ABS_TOL and d['ulp_limit']==MAX_ULP
                and 0 <= d['max_abs'] <= ABS_TOL and 0 <= d['max_ulp'] <= MAX_ULP,
                'Invalid recorded rounding discrepancy')
    return result


def run_task(launch_path,task_index):
    launch,old,plan = read_recovery(launch_path)
    require(launch['mode']=='audit' and task_index in launch['selected_tasks'],'Undeclared recovery task')
    context,task = launch['context'],plan['tasks'][task_index]
    out = Path(context['output']); path = out/'tasks'/f"{task['id']}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            report,_ = checked_json(path)
            validate_recovered(report,context,old,plan,task)
            print(json.dumps(dict(task=task['id'],status=report['status'],reused=True)),flush=True)
            return 0 if report['status']=='ready' else 2
        before = original_snapshot(old,plan,task)
        runtime = original.runtime_setup(old['campaign']['dependency_root'])
        require(runtime == before['runtime'],'Recovery must use original numerical runtime')
        from shared_base_curve_audit import audit_reference
        import shared_base_curve_audit as auditor
        from validate_shared_base_mcmc import json_safe
        started = time.monotonic(); details = []
        attempt = out/'attempts'/task['id']/f"{os.environ.get('SLURM_JOB_ID','local')}-{time.time_ns()}.json"
        receipt = dict(schema='shared-base-posterior-recovered-audit-v1',context_sha256=fingerprint(context),
            task=task,original=before,runtime=runtime,status='running',started_utc=original.utc(),
            hostname=socket.gethostname(),no_sampling=True,model_readback=details)
        atomic_json(attempt,receipt)
        print(json.dumps(dict(task=task['id'],status='auditing_saved_chains',no_sampling=True)),flush=True)
        try:
            with canonical_model_loader(auditor,details):
                result = audit_reference(Path(old['campaign']['inputs'])/task['setting'],
                    Path(old['campaign']['output'])/'references',task['index'],task['time'],task['order'])
            require(before == original_snapshot(old,plan,task),'Original artifacts changed during recovery')
            receipt.update(result=result,status='ready' if result['status']=='ready' else
                           'unresolved' if result['status']=='needs_more_draws' else 'audit_failed')
            if receipt['status'] in ('ready','unresolved'):
                validate_recovered(receipt,context,old,plan,task)
        except Exception as error:
            receipt.update(status='audit_failed',error=str(error))
        receipt.update(finished_utc=original.utc(),elapsed_seconds=time.monotonic()-started,
                       peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                          (1 if sys.platform=='darwin' else 1024)))
        receipt=json_safe(receipt)
        atomic_json(attempt,receipt)
        # Completed recovery records are immutable; failed attempts remain separate.
        if receipt['status'] in ('ready','unresolved'):
            immutable_json(path,receipt)
        print(json.dumps(dict(task=task['id'],status=receipt['status'],elapsed_seconds=receipt['elapsed_seconds'],
                              error=receipt.get('error'),result_error=receipt.get('result',{}).get('error'))),flush=True)
        return 0 if receipt['status']=='ready' else 2 if receipt['status']=='unresolved' else 1


def merge(launch_path):
    launch,old,plan = read_recovery(launch_path)
    context=launch['context']; out=Path(context['output'])
    from shared_base_curve_audit import aggregate_time
    from validate_shared_base_mcmc import json_safe
    rows=[]; by_id={}; hashes={}; runtimes=[]
    for task in plan['tasks']:
        path=out/'tasks'/f"{task['id']}.json"
        row=dict(task=task,status='missing')
        if path.is_file():
            report,digest=checked_json(path)
            by_id[task['id']]=validate_recovered(report,context,old,plan,task)
            row.update(status=report['status'],audit=report['result'],model_readback=report['model_readback'])
            hashes[str(path)]=digest; runtimes.append(report['runtime'])
        else:
            attempts=sorted((out/'attempts'/task['id']).glob('*.json'),key=lambda p:p.stat().st_mtime_ns)
            if attempts:
                attempt,digest=checked_json(attempts[-1])
                require(attempt['context_sha256']==fingerprint(context) and attempt['task']==task,
                        'Recovery attempt identity mismatch')
                row.update(status=attempt['status'],error=attempt.get('error'),
                           audit=attempt.get('result'),attempt=str(attempts[-1]),attempt_sha256=digest)
        rows.append(row)
    require(not runtimes or all(r==runtimes[0] for r in runtimes),'Recovery tasks used different runtimes')
    complete=len(by_id)==32
    groups=[]
    for depth in DEPTHS:
        for t in TIMES:
            tasks=[task for task in plan['tasks'] if task['depth']==depth and task['time']==t]
            checked=[by_id[task['id']] for task in tasks if task['id'] in by_id]
            aggregate=aggregate_time(checked,list(INDICES),list(ORDERS))
            # Never publish a passing subset while another fixed pilot task is missing.
            if not complete:
                aggregate['reported_mean_bayes_regret_bits']=None
                aggregate['release_blocked_by_incomplete_pilot']=True
            groups.append(dict(depth=depth,time=t,numerical_audit=aggregate))
    result=dict(schema='shared-base-posterior-recovery-summary-v1',context_sha256=fingerprint(context),
        complete=complete,all_numerical_checks_passed=complete and all(g['numerical_audit']['passed'] for g in groups),
        expected_references=32,expected_sources=8,rows=rows,groups=groups,recovered_receipt_sha256=hashes,
        production_approved=False,full_population_claim=False,no_sampling=True,
        interpretation='All fixed cases and both exponent orders retained; two sequences per condition are a numerical pilot, not population evidence.',
        summarized_utc=original.utc())
    result=json_safe(result)
    atomic_json(out/'summary.json',result)
    print(json.dumps(dict(complete=complete,passed=result['all_numerical_checks_passed'],
        counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})})),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    launch=subs.add_parser('launch')
    launch.add_argument('--original-launch',type=Path)
    launch.add_argument('--original-out',type=Path,default=original.DEFAULT_OUT)
    launch.add_argument('--out',type=Path,default=DEFAULT_OUT)
    launch.add_argument('--venv',type=Path,default=Path(os.environ.get('SHARED_BASE_VENV',str(Path.home()/'venv-tf'))))
    launch.add_argument('--only-task',type=int,nargs='+')
    launch.add_argument('--concurrency',type=int,default=32)
    launch.add_argument('--merge',action='store_true')
    action=launch.add_mutually_exclusive_group()
    action.add_argument('--submit',action='store_true');action.add_argument('--dry-run',action='store_true')
    run=subs.add_parser('run');run.add_argument('--launch',type=Path,required=True)
    run.add_argument('--task-index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=parser.parse_args()
    if args.command=='launch':
        submit(args)
    else:
        launch,_,_=read_recovery(args.launch)
        if launch['mode']=='merge':
            merge(args.launch)
        else:
            raise SystemExit(run_task(args.launch,args.task_index))


if __name__=='__main__':
    main()
