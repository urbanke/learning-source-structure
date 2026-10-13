"""Bounded saved-chain follow-up for the complete shared-base time64 cohort.

Prepare freezes corrected-audit evidence and explicit per-case draw targets.
Run copies selected parent chains and invokes the unchanged sampler. It never
alters original results or selects a passing subset. A complete overlay records
all2048 cases and requires every planned replacement, including failed ones.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_shared_base_curve_campaign as campaign

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'shared-base-local-finish-v1'
TARGET_SCHEMA = 'shared-base-local-targets-v1'
MEMORY_BUDGET = 18 * 1024**3
DISK_RESERVE = 50_000_000_000
MAX_WORKERS = 12
MEMORY_CALIBRATION = dict(
    case='a1_L8/seq_000004/t0064/alpha32', draws_per_chain=131072,
    checkpoint_sha256='de24ce69810ee5acc5fb0cf34078f99badeb02483464d91f57e9dbd43d93bfe4',
    measured_peak_rss_bytes=2586673152, elapsed_seconds=3.623,
    phases=['load','report','predictive_moments','case_precision','model_metrics','row_grid_checks'],
    no_sampling=True, originals_unchanged=True,
    admission='1.5GiB fixed allowance plus six copies of four-chain64-coordinate float64 predictive samples',
    limitation='One saved-reference memory probe, not a proven memory bound. Admission includes margin and stays below18GiB total on the24GB host.')
DEFAULT_CAMPAIGN = ROOT/'output/shared_base_curve_campaign_2026-09-15'
DEFAULT_AUDIT = ROOT/'output/shared_base_local_revalidation_v2_2026-09-17/summary.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_map(task):
    return {name: Path(task[name]) for name in ('reference', 'checkpoint', 'moments')}


def raw_hashes(task):
    return {str(path): campaign.digest(path) for path in file_map(task).values()}


def task_key(task):
    depth = int(task['setting'].split('L')[-1])
    require(task['setting'] == f'a1_L{depth}' and depth in (0,1,4,8)
            and task['time'] == 64 and task['order'] in (16,32)
            and type(task['index']) is int and 0 <= task['index'] < 256,
            'Unexpected reference outside the fixed time64 cohort')
    return campaign.job_key(depth, task['index'], 64, task['order'])


def slot_bytes(draws, order):
    """Measured-summary allowance with margin; order32 is the probe case."""
    require(order in (16,32), 'Unsupported exponent order for local memory admission')
    return 3 * 1024**3 // 2 + 6 * 4 * draws * 64 * 8



def task_paths(plan, key):
    parent = plan['selected'][key]['parent_task']
    root = Path(plan['output'])
    folder = root/'references'/parent['setting']/f"alpha{parent['order']}"
    stem = f"seq_{parent['index']:06d}_t0064"
    directory = root/'tasks'/key.replace('/', '_')
    return dict(reference=folder/(stem+'.json'), checkpoint=folder/(stem+'_chains.npz'),
        moments=folder/(stem+'_moments.npz'), lock=folder/(stem+'.lock'),
        work_lock=directory/'task.lock', prepared=directory/'prepared.json',
        audit=directory/'audit.json', receipt=directory/'receipt.json', log=directory/'worker.log')


def code_hashes():
    import shared_base_local_revalidation as audit
    extra = [name if '/' in name else 'code/scripts/'+name
             for name in getattr(audit, 'CODE_FILES', getattr(audit, 'CODE', []))]
    names = sorted(set(campaign.CODE + ['code/scripts/finish_shared_base_local.py',
        'code/scripts/shared_base_local_revalidation.py', 'code/scripts/revalidate_shared_base_depth.py',
        'code/scripts/inspect_shared_base_depth_diagnostics.py', 'code/scripts/depth_rhat_statistics.py',
        'code/scripts/recover_shared_base_posterior_audit.py'] + list(extra)))
    return {name: campaign.digest(ROOT/name) for name in names}


def checked_summary(path):
    """Admission uses a complete, immutable corrected audit, never its old flag."""
    import shared_base_local_revalidation as audit
    summary = read(path)
    audit_plan, audit_hash = audit.read_plan(Path(path).parent)
    require(summary['schema'] == 'shared-base-local-revalidation-v1'
            and summary['complete'] is True and summary['plan_id'] == audit_plan['plan_id']
            and summary['launch_sha256'] == audit_hash and not summary.get('failures'),
            'A complete successful raw-audit execution is required')
    references = summary['references']
    expected = {campaign.job_key(d,i,64,o) for d in (0,1,4,8) for i in range(256) for o in (16,32)}
    require(isinstance(references, dict) and set(references) == expected,
            'Corrected audit must cover all2048 declared time64 references')
    rows = {}
    for key, item in references.items():
        require(task_key(item['task']) == key, 'Corrected-audit case identity differs')
        receipt = Path(item['case_receipt'])
        require(campaign.digest(receipt) == item['case_receipt_sha256'], 'Corrected-audit receipt changed')
        row = read(receipt)
        require(row['launch_sha256'] == audit_hash and row['task'] == item['task'] and row['row_id'] == campaign.fingerprint(
            {k:v for k,v in row.items() if k != 'row_id'}), 'Corrected-audit row identity failed')
        corrected = row['corrected']['audit']
        require(corrected['status'] in ('ready','needs_more_draws')
                and corrected['ready'] is (corrected['status'] == 'ready')
                and item['ready'] == corrected['ready'], 'Missing or invalid corrected raw audit')
        rows[key] = row
    return summary, rows


def prepare(audit_summary, targets_path, campaign_root, out, *, max_workers=12):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a new follow-up output directory')
    require(type(max_workers) is int and 1 <= max_workers <= MAX_WORKERS, 'Worker count must be1..12')
    summary, rows = checked_summary(audit_summary)
    targets = read(targets_path)
    require(targets['schema'] == TARGET_SCHEMA and targets['cases'], 'Explicit draw-target list required')
    selected = {}
    for request in targets['cases']:
        key, target = request['key'], request['target_draws']
        require(key in rows and key not in selected, 'Duplicate or undeclared continuation')
        task = rows[key]['task']; depth = int(task['setting'].split('L')[-1])
        require(depth in (4,8), 'This follow-up is restricted to incomplete L4/L8 time64 cohorts')
        reference = read(task['reference'])
        draws = reference['estimate']['diagnostics']['draws']
        require(type(target) is int and draws < target <= 2**21 and target % 2048 == 0,
                'Target must strictly extend saved draws, in2048-draw blocks, at most2^21')
        require(isinstance(request.get('reason'), str) and request['reason'].strip(), 'Target needs an explicit reason')
        memory = slot_bytes(target, task['order'])
        require(memory <= MEMORY_BUDGET, 'One continuation exceeds the local memory allowance; revise its fixed target')
        require(reference['reference_id'] == rows[key]['corrected']['audit']['reference_id'], 'Parent reference differs from audit')
        hashes = raw_hashes(task)
        require(hashes == task['file_sha256'], 'Parent raw files changed after corrected audit')
        for protected in (Path(task['reference_root']).resolve(), Path(task['source_folder']).resolve()):
            require(out != protected and out not in protected.parents and protected not in out.parents,
                    'Follow-up output must be separate from original source/reference trees')
        selected[key] = dict(parent_task=task, parent_hashes=hashes, parent_reference_id=reference['reference_id'],
            parent_row_id=rows[key]['row_id'], previous_draws=draws, target_draws=target,
            reason=request['reason'], estimated_peak_bytes=memory)
    unresolved = {key for key,row in rows.items() if row['task']['setting'] in ('a1_L4','a1_L8')
                  and not row['corrected']['audit']['ready']}
    require(unresolved.issubset(selected), 'Targets omit an unresolved L4/L8 reference')
    original_path = Path(campaign_root).resolve()/'plan.json'
    original = read(original_path)
    campaign.verify_assets(original, original['global_paths'])
    for key, case in selected.items():
        task = case['parent_task']
        paths = original['cases'][f"{task['setting']}/{task['index']}"]['files']
        campaign.verify_assets(original, paths)
    plan = dict(schema=SCHEMA, output=str(out), time=64, no_auto_retry=True, no_other_times=True,
        audit_summary=str(Path(audit_summary).resolve()), audit_summary_sha256=campaign.digest(audit_summary),
        targets_file=str(Path(targets_path).resolve()), targets_sha256=campaign.digest(targets_path),
        original_plan=str(original_path), original_plan_sha256=campaign.digest(original_path),
        code_sha256=code_hashes(), selected=selected,
        effective_cases={key:dict(task=row['task'], row_id=row['row_id'],
            case_receipt=summary['references'][key]['case_receipt'],
            case_receipt_sha256=summary['references'][key]['case_receipt_sha256']) for key,row in rows.items()},
        memory_calibration=MEMORY_CALIBRATION,
        limits=dict(max_workers=max_workers, maximum_global_workers=MAX_WORKERS,
            memory_budget_bytes=MEMORY_BUDGET, disk_reserve_bytes=DISK_RESERVE),
        interpretation='Same saved chains, seeds, source and target; fixed draw extensions only. Historical individual precision flags do not select cases. All2048 cases remain declared; no fallback for an unfinished planned replacement.')
    plan['plan_id'] = campaign.fingerprint(plan)
    campaign.atomic_json(out/'memory_calibration.json', MEMORY_CALIBRATION)
    campaign.atomic_json(out/'plan.json', plan)
    campaign.atomic_json(out/'status.json', dict(schema=SCHEMA, plan_id=plan['plan_id'],
        jobs={key:dict(state='pending', attempts=[]) for key in selected}))
    return plan


def read_plan(out):
    plan = read(Path(out)/'plan.json')
    require(plan['schema'] == SCHEMA and plan['plan_id'] == campaign.fingerprint(
        {k:v for k,v in plan.items() if k != 'plan_id'}), 'Follow-up plan identity changed')
    require(plan['code_sha256'] == code_hashes(), 'Frozen follow-up code changed')
    for name in ('audit_summary', 'targets', 'original_plan'):
        path = plan['targets_file'] if name == 'targets' else plan[name]
        require(campaign.digest(path) == plan[name+'_sha256'], f'Frozen {name} changed')
    require(plan['memory_calibration'] == MEMORY_CALIBRATION
            and read(Path(out)/'memory_calibration.json') == MEMORY_CALIBRATION,
            'Memory admission calibration changed')
    require(plan['limits']['maximum_global_workers'] == MAX_WORKERS
            and plan['limits']['memory_budget_bytes'] == MEMORY_BUDGET
            and plan['limits']['disk_reserve_bytes'] == DISK_RESERVE,
            'Frozen resource limits changed')
    return plan


def prepare_case(plan, key):
    case = plan['selected'][key]; task = case['parent_task']; paths = task_paths(plan,key)
    require(raw_hashes(task) == case['parent_hashes'], 'Original parent files changed')
    expected = dict(plan_id=plan['plan_id'], key=key, parent_sha256=case['parent_hashes'],
                    target_draws=case['target_draws'])
    require(all(not paths[name].is_symlink() and paths[name].resolve().is_relative_to(
        Path(plan['output']).resolve()) for name in ('reference','checkpoint','moments')),
        'Continuation files must be local copies inside the new output tree')
    if paths['prepared'].exists():
        require(read(paths['prepared']) == expected and all(paths[k].exists() for k in file_map(task)),
                'Prepared continuation changed or lost files')
        return paths
    for name, source in file_map(task).items():
        target = paths[name]
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            require(campaign.digest(target) == case['parent_hashes'][str(source)], 'Unbound continuation copy differs')
        else:
            temporary = target.with_suffix(target.suffix+'.copying')
            shutil.copyfile(source, temporary)
            require(campaign.digest(temporary) == case['parent_hashes'][str(source)], 'Continuation copy checksum failed')
            os.replace(temporary,target)
    require(raw_hashes(task) == case['parent_hashes'], 'Parent changed while copying')
    campaign.atomic_json(paths['prepared'],expected)
    return paths


def worker_command(plan, key):
    case = plan['selected'][key]; task = case['parent_task']
    original = read(plan['original_plan'])
    adapted = dict(original, references=str(Path(plan['output'])/'references'),
        worker=dict(original['worker'], initial_draws=case['target_draws'], max_draws=case['target_draws']))
    job = dict(depth=int(task['setting'].split('L')[-1]), index=task['index'], prefix=64, order=task['order'])
    return campaign.command(adapted,job)


def verify_child(plan, key):
    case = plan['selected'][key]; paths = task_paths(plan,key)
    parent, child = read(case['parent_task']['reference']), read(paths['reference'])
    require(child['identity'] == parent['identity'] and child['reference_id'] == case['parent_reference_id']
            and child['history'][:len(parent['history'])] == parent['history']
            and child['estimate']['diagnostics']['draws'] == case['target_draws'],
            'Continuation changed identity/history or missed its fixed draw target')
    require(child['checkpoint'] == str(paths['checkpoint'].resolve())
            and campaign.digest(paths['checkpoint']) == child['checkpoint_sha256']
            and campaign.digest(paths['moments']) == child['moments_sha256'], 'Continued raw files are inconsistent')
    return child


def run_case(out, key):
    import shared_base_local_revalidation as audit
    plan = read_plan(out)
    require(key in plan['selected'], 'Undeclared continuation key')
    paths = task_paths(plan,key)
    with campaign.campaign_lock(paths['work_lock']):
        require(not paths['receipt'].exists(), 'Case already finished; no automatic retry')
        require(not campaign.lock_busy(paths['lock']), 'Continuation worker is already active')
        prepare_case(plan,key)
        original = read(plan['original_plan']); case = plan['selected'][key]
        campaign.verify_assets(original, original['global_paths']+
            original['cases'][f"{case['parent_task']['setting']}/{case['parent_task']['index']}"]['files'])
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
            **{k:v for k,v in original['worker']['environment'].items() if v is not None})
        with paths['log'].open('a') as stream:
            result = subprocess.run(worker_command(plan,key), cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        require(result.returncode in (0,2), f'Frozen worker failed with exit{result.returncode}')
        verify_child(plan,key)
        task = deepcopy(case['parent_task'])
        task.update(reference_root=str(Path(plan['output'])/'references'),
                    **{name:str(paths[name]) for name in ('reference','checkpoint','moments')})
        task['file_sha256'] = raw_hashes(task)
        audit.configure_runtime(dict(environment={k:v for k,v in original['worker']['environment'].items() if v is not None}))
        row = audit.audit_case(task, plan['plan_id'])
        require(row['corrected']['audit']['status'] in ('ready','needs_more_draws'),
                'Continued raw audit failed')
        require(raw_hashes(task) == task['file_sha256'], 'Child raw files changed during audit')
        require(raw_hashes(case['parent_task']) == case['parent_hashes'], 'Parent files changed during continuation')
        read_plan(out)
        campaign.atomic_json(paths['audit'],row)
        receipt = dict(schema=SCHEMA, plan_id=plan['plan_id'], key=key,
            state='ready' if row['corrected']['audit']['ready'] else 'unresolved',
            worker_exit_code=result.returncode, target_draws=case['target_draws'],
            audit_sha256=campaign.digest(paths['audit']), child_task=task,
            parent_row_id=case['parent_row_id'])
        campaign.atomic_json(paths['receipt'],receipt)
        return receipt


def checked_result(plan, key, *, replay_depths=False):
    paths = task_paths(plan,key); receipt = read(paths['receipt']); row = read(paths['audit'])
    require(receipt['plan_id'] == plan['plan_id'] and receipt['key'] == key
            and receipt['worker_exit_code'] in (0,2)
            and campaign.digest(paths['audit']) == receipt['audit_sha256']
            and row['task'] == receipt['child_task']
            and row['row_id'] == campaign.fingerprint({k:v for k,v in row.items() if k!='row_id'})
            and raw_hashes(receipt['child_task']) == receipt['child_task']['file_sha256'],
            'Saved follow-up receipt/raw data changed')
    verify_child(plan,key)
    import shared_base_local_revalidation as audit
    original = read(plan['original_plan'])
    audit.configure_runtime(dict(environment={k:v for k,v in original['worker']['environment'].items() if v is not None}))
    audit.validate_row(row, receipt['child_task'], plan['plan_id'], replay=replay_depths)
    require(receipt['state'] == ('ready' if row['corrected']['audit']['ready'] else 'unresolved'),
            'Saved corrected follow-up status differs')
    return receipt, row


def export_overlay(plan):
    cases = {}; complete = True
    for key, item in plan['effective_cases'].items():
        if key in plan['selected']:
            paths = task_paths(plan,key)
            if not paths['receipt'].exists():
                cases[key] = dict(planned_replacement=True, complete=False,
                    expected_reference=str(paths['reference']))
                complete = False
                continue
            receipt,row = checked_result(plan,key,replay_depths=True)
            cases[key] = dict(planned_replacement=True, complete=True, task=row['task'],
                case_receipt=str(paths['audit']), case_receipt_sha256=receipt['audit_sha256'],
                ready=row['corrected']['audit']['ready'])
        else:
            require(campaign.digest(item['case_receipt']) == item['case_receipt_sha256'],
                    'Retained corrected-audit receipt changed')
            require(raw_hashes(item['task']) == item['task']['file_sha256'],
                    'Retained raw reference files changed before release')
            row = read(item['case_receipt'])
            cases[key] = dict(planned_replacement=False, complete=True, **item,
                             ready=row['corrected']['audit']['ready'])
    value = dict(schema='shared-base-local-finish-overlay-v1', plan_id=plan['plan_id'], time=64,
        complete=complete, references=cases, n_declared=2048,
        interpretation='Every fixed source/order remains declared. Every planned replacement is mandatory; unresolved cases remain unresolved. Complete is coverage only, not a passed numerical cohort gate.')
    # Reuse the exact complete-cohort numerical gates. Missing replacements
    # remain missing and suppress every reported mean; no available-case mean.
    import shared_base_local_revalidation as audit
    rows=[]; tasks=[]; combined=Path(plan['output'])/'combined'
    for key,item in cases.items():
        if item['complete']:
            row=read(item['case_receipt']);tasks.append(row['task']);rows.append(row)
            campaign.atomic_json(audit.case_path(combined,row['task']),row)
        else:
            tasks.append(deepcopy(plan['selected'][key]['parent_task']))
    summary=audit.summarize_rows(dict(cases=tasks,output=str(combined),plan_id=plan['plan_id']),rows)
    read_plan(plan['output'])
    campaign.verify_assets(read(plan['original_plan']))
    campaign.atomic_json(Path(plan['output'])/'overlay.json',value)
    summary.update(schema='shared-base-local-finish-summary-v1', no_sampling=False,
        sampling_performed=True, correction_not_sampling=False,
        base_audit_summary=plan['audit_summary'], base_audit_summary_sha256=plan['audit_summary_sha256'],
        overlay_sha256=campaign.digest(Path(plan['output'])/'overlay.json'),
        interpretation='All2048 declared time64 cases retained; selected chains continued to fixed targets. Original data stay unchanged. Complete refers to coverage; all_cohorts_passed is the unchanged numerical gate.')
    campaign.atomic_json(Path(plan['output'])/'summary.json',summary)
    return value


def external_active(plan):
    roots = {Path(item['task']['reference_root']) for item in plan['effective_cases'].values()}
    paths = {p for root in roots for p in root.glob('a1_L*/alpha*/seq*_t*.lock')}
    return sum(campaign.lock_busy(path) for path in paths)


def run(out, *, max_hours=12., poll_seconds=5.):
    require(0 < max_hours <= 24 and poll_seconds > 0, 'Invalid bounded dispatcher duration')
    plan = read_plan(out); out=Path(out); original=read(plan['original_plan'])
    campaign.verify_assets(original, original['global_paths'])
    state = read(out/'status.json')
    require(state['plan_id']==plan['plan_id'] and set(state['jobs'])==set(plan['selected']), 'Status differs from fixed plan')
    with campaign.campaign_lock(out/'dispatcher.lock'):
        for key,item in state['jobs'].items():
            paths=task_paths(plan,key)
            require(not campaign.lock_busy(paths['work_lock']) and not campaign.lock_busy(paths['lock']),
                    'A previous follow-up process is still active; do not duplicate it')
            if paths['receipt'].exists():
                item['state']=checked_result(plan,key)[0]['state']
            elif item['state']=='running':
                item['state']='pending';item['attempts'][-1]['interrupted']=True
        active={};stop=[];started=time.monotonic()
        previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        for sig in previous:
            signal.signal(sig,lambda signum,frame:stop.append('signal: active bounded cases allowed to finish'))
        def save():
            state.update(updated=campaign.utc(),counts=dict(Counter(v['state'] for v in state['jobs'].values())),
                active_workers=len(active),stop_reason=stop[0] if stop else None)
            campaign.atomic_json(out/'status.json',state)
        try:
            while True:
                for key,(process,stream,began) in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    stream.close();item=state['jobs'][key]
                    item['attempts'][-1].update(exit_code=code,finished=campaign.utc(),wall_seconds=time.monotonic()-began)
                    try:
                        require(code==0,'Case worker/audit process failed')
                        item['state']=checked_result(plan,key)[0]['state']
                    except (OSError,ValueError,KeyError) as error:
                        item.update(state='failed',error=str(error))
                    del active[key]
                if time.monotonic()-started>=max_hours*3600:stop.append('dispatch time budget reached')
                if shutil.disk_usage(out).free<DISK_RESERVE:stop.append('50GB disk reserve reached')
                elsewhere=external_active(plan)
                memory=sum(plan['selected'][key]['estimated_peak_bytes'] for key in active)
                state.update(external_active_workers=elsewhere,reserved_peak_memory_bytes=memory)
                if not stop:
                    for key,item in state['jobs'].items():
                        if len(active)>=plan['limits']['max_workers'] or len(active)+elsewhere>=MAX_WORKERS:break
                        if item['state']!='pending':continue
                        need=plan['selected'][key]['estimated_peak_bytes']
                        # Unknown external memory costs cannot safely be treated as zero.
                        if elsewhere or memory+need>MEMORY_BUDGET:continue
                        log=out/'logs'/(key.replace('/','_')+f"_attempt{len(item['attempts'])+1}.log")
                        log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('x')
                        command=[sys.executable,str(Path(__file__).resolve()),'--case',key,'--out',str(out)]
                        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        item.update(state='running',pid=process.pid)
                        item['attempts'].append(dict(pid=process.pid,started=campaign.utc(),log=str(log)))
                        active[key]=(process,stream,time.monotonic());memory+=need;save()
                save()
                if not active and (stop or all(v['state']!='pending' for v in state['jobs'].values())):break
                time.sleep(poll_seconds)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
        export_overlay(plan)
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--run',action='store_true')
    action.add_argument('--case');action.add_argument('--overlay',action='store_true')
    parser.add_argument('--audit-summary',type=Path,default=DEFAULT_AUDIT);parser.add_argument('--targets',type=Path)
    parser.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-workers',type=int,default=12);parser.add_argument('--max-hours',type=float,default=12.)
    args=parser.parse_args()
    if args.prepare:
        require(args.audit_summary is not None and args.targets is not None,'Prepare needs corrected audit and explicit targets')
        result=prepare(args.audit_summary,args.targets,args.campaign,args.out,max_workers=args.max_workers)
        print(json.dumps(dict(prepared=len(result['selected']),plan_id=result['plan_id'],launched=0)))
    elif args.run:print(json.dumps(run(args.out,max_hours=args.max_hours)['counts']))
    elif args.case:print(json.dumps(run_case(args.out,args.case)))
    else:print(json.dumps(dict(complete=export_overlay(read_plan(args.out))['complete'])))


if __name__=='__main__':main()
