"""Audit every fixed local time-64 reference with corrected depth R-hat.

The exact 36-case saved continuation overlays the original 2048 references.
All raw-data, prior, precision and integration checks reuse the original auditor;
only scale-dependent depth R-hat arithmetic changes. No sampling is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT/'output/shared_base_curve_campaign_2026-09-15'
DEFAULT_CONTINUATION = ROOT/'output/shared_base_curve_continuation_2026-09-16'
DEFAULT_OUT = ROOT/'output/shared_base_local_revalidation_2026-09-17'
DEPTHS = (0, 1, 4, 8)
ORDERS = (16, 32)
INDICES = tuple(range(256))
CODE_FILES = ('shared_base_local_revalidation.py', 'shared_base_curve_audit.py',
              'revalidate_shared_base_depth.py', 'inspect_shared_base_depth_diagnostics.py',
              'depth_rhat_statistics.py', 'recover_shared_base_posterior_audit.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(read_json(path)[0] == value, f'Existing artifact differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def expected_keys():
    return {f'a1_L{d}/seq_{i:06d}/t0064/alpha{o}'
            for d in DEPTHS for i in INDICES for o in ORDERS}


def code_hashes():
    return {str(Path(__file__).with_name(name)): sha(Path(__file__).with_name(name))
            for name in CODE_FILES}


def check_task_files(task):
    require(all(sha(path) == digest for path, digest in task['file_sha256'].items()),
            f"Frozen reference files changed: {task['key']}")


def prepare(campaign=DEFAULT_CAMPAIGN, continuation=DEFAULT_CONTINUATION, out=DEFAULT_OUT):
    import run_shared_base_curve_campaign as original
    campaign, continuation, out = map(lambda p: Path(p).resolve(), (campaign, continuation, out))
    old, old_hash = read_json(campaign/'plan.json')
    extension, extension_hash = read_json(continuation/'plan.json')
    require(old['plan_id'] == original.fingerprint({k:v for k,v in old.items() if k!='plan_id'}),
            'Original campaign identity differs')
    require(extension['plan_id'] == original.fingerprint({k:v for k,v in extension.items() if k!='plan_id'})
            and extension['original_plan_sha256'] == old_hash
            and Path(extension['original_plan']).resolve() == campaign/'plan.json',
            'Continuation does not bind the original campaign')
    original.verify_assets(old, old['global_paths'])
    overlay = {c['job']['key']:c for c in extension['cases']}
    require(len(overlay) == len(extension['cases']) == 36 and set(overlay) <= expected_keys(),
            'Expected exactly the fixed 36 time-64 continuations')
    selected = [j for j in old['jobs'] if j['prefix']==64]
    require(len(selected)==len(expected_keys()) and {j['key'] for j in selected}==expected_keys(),
            'Original time-64 scope is incomplete')
    for protected in (campaign, continuation, Path(old['references']), Path(old['source_root'])):
        protected = protected.resolve()
        require(out != protected and out not in protected.parents and protected not in out.parents,
                'Output must be separate from all original artifacts')
    tasks=[]
    for job in selected:
        base = Path(extension['references'] if job['key'] in overlay else old['references'])
        reference, _ = original.job_paths(dict(old, references=str(base)), job)
        report, digest = read_json(reference)
        setting=f"a1_L{job['depth']}"
        checkpoint=reference.with_name(reference.stem+'_chains.npz')
        moments=reference.with_name(reference.stem+'_moments.npz')
        require((report['setting'], report['index'], report['time'], report['order']) ==
                (setting, job['index'], 64, job['order']), 'Reference case identity differs')
        require(report['reference_id']==original.fingerprint(report['identity'])
                and report['checkpoint']==str(checkpoint.resolve()), 'Reference/checkpoint identity differs')
        if job['key'] in overlay:
            require(report['reference_id']==overlay[job['key']]['reference_id']
                    and report['estimate']['diagnostics']['draws']==extension['total_draws_per_chain'],
                    'Incomplete or changed continuation')
        require(checkpoint.is_file() and moments.is_file(), 'Raw checkpoint/moments missing')
        tasks.append(dict(key=job['key'], setting=setting, depth=job['depth'], index=job['index'],
            time=64, order=job['order'], source_folder=str(Path(old['source_root'])/setting),
            reference_root=str(base), reference=str(reference), checkpoint=str(checkpoint), moments=str(moments),
            reference_id=report['reference_id'],
            file_sha256={str(reference):digest, str(checkpoint):report['checkpoint_sha256'],
                         str(moments):report['moments_sha256']}))
    spec=dict(schema='shared-base-local-revalidation-plan-v1', output=str(out), cases=tasks,
        original_plan=str(campaign/'plan.json'), original_plan_sha256=old_hash,
        continuation_plan=str(continuation/'plan.json'), continuation_plan_sha256=extension_hash,
        overlay_keys=sorted(overlay), code_sha256=code_hashes(),
        environment=old['worker']['environment'], python=sys.executable,
        no_sampling=True, correction='scale-invariant depth split-Rhat only')
    spec['plan_id']=fingerprint(spec)
    write_json(out/'plan.json', spec, immutable=True)
    return spec


def read_plan(out):
    plan, digest=read_json(Path(out)/'plan.json')
    require(plan.get('schema')=='shared-base-local-revalidation-plan-v1'
            and plan['plan_id']==fingerprint({k:v for k,v in plan.items() if k!='plan_id'}),
            'Local revalidation plan identity differs')
    require(len(plan['cases'])==len(expected_keys()) and {t['key'] for t in plan['cases']}==expected_keys()
            and len(plan['overlay_keys'])==36 and set(plan['overlay_keys'])<=expected_keys(),
            'Fixed local cohort scope differs')
    require(plan['code_sha256']==code_hashes() and plan['no_sampling'] is True,
            'Revalidation code or scope changed')
    for name in ('original_plan','continuation_plan'):
        require(sha(plan[name])==plan[name+'_sha256'], 'Frozen parent plan changed')
    return plan, digest


def configure_runtime(plan):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='1'
    for name in ('PMM_PHI_BIAS','PMM_PHI_WAVE'):
        require(float(os.environ.get(name,'0'))==0, 'Numerical perturbation is not allowed')
    os.environ.update(plan['environment'])


def audit_one_reference(task, launch_sha256):
    """Full original raw audit, then only the uniform depth arithmetic correction."""
    import shared_base_curve_audit as auditor
    from inspect_shared_base_depth_diagnostics import summarize_depths
    from revalidate_shared_base_depth import adjust_depth_gate
    from recover_shared_base_posterior_audit import canonical_model_loader
    from validate_shared_base_mcmc import json_safe
    started=time.monotonic(); observations=[]
    path=Path(task['reference'])
    with path.with_suffix('.lock').open('rb') as guard:
        fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
        check_task_files(task)
        reference,_=read_json(path)
        with canonical_model_loader(auditor, observations):
            checked=auditor.audit_reference(task['source_folder'],task['reference_root'],
                                            task['index'],task['time'],task['order'])
        require(checked['status'] in ('ready','needs_more_draws'),
                f"Original raw audit failed: {checked.get('error',checked['status'])}")
        depths=summarize_depths(task['checkpoint'],reference['estimate'])
        corrected=adjust_depth_gate(reference,checked,depths)
        check_task_files(task)
    row=json_safe(dict(task=task, launch_sha256=launch_sha256, original_audit=checked,
        depth_diagnostics=depths, corrected=corrected, canonical_loader_checks=observations,
        elapsed_seconds=time.monotonic()-started, no_sampling=True))
    row['row_id']=fingerprint(row)
    return row


audit_case = audit_one_reference


def validate_row(row,task,launch_sha256, replay=True):
    from shared_base_curve_audit import same
    from inspect_shared_base_depth_diagnostics import summarize_depths
    from revalidate_shared_base_depth import adjust_depth_gate
    require(row['row_id']==fingerprint({k:v for k,v in row.items() if k!='row_id'})
            and row['task']==task and row['launch_sha256']==launch_sha256 and row['no_sampling'] is True,
            'Saved case receipt identity differs')
    check_task_files(task)
    reference,_=read_json(task['reference'])
    depths=summarize_depths(task['checkpoint'],reference['estimate']) if replay else row['depth_diagnostics']
    require(same(depths,row['depth_diagnostics'],atol=0,rtol=2e-12)
            and same(adjust_depth_gate(reference,row['original_audit'],depths),row['corrected']),
            'Saved correction differs from raw depth arithmetic')
    check_task_files(task)


def case_path(out,task):
    return Path(out)/'cases'/(task['key'].replace('/','_')+'.json')


def summarize_rows(plan,rows):
    from shared_base_curve_audit import aggregate_time
    from validate_shared_base_mcmc import json_safe
    expected={t['key']:t for t in plan['cases']};found={r['task']['key']:r for r in rows}
    require(set(expected)==expected_keys() and len(found)==len(rows) and set(found)<=set(expected),
            'Repeated or undeclared case in summary')
    require(all(r['task']==expected[key] for key,r in found.items()), 'Summary task identity differs')
    complete=len(found)==len(expected);groups=[];references={}
    for key, row in found.items():
        a=row['corrected']['audit'];p=case_path(plan['output'],row['task'])
        references[key]=dict(task=row['task'],status=a['status'],ready=a['ready'],
            draws_per_chain=a['draws_per_chain'],case_receipt=str(p),case_receipt_sha256=sha(p),
            minimum_probability_ess=a['minimum_probability_ess'],scalar_ess=a['scalar_ess'])
    for depth in DEPTHS:
        selected=[r['corrected']['audit'] for r in rows if r['task']['depth']==depth]
        value=aggregate_time(selected,list(INDICES),list(ORDERS))
        if not complete:
            value['reported_mean_bayes_regret_bits']=None
            value['release_blocked_by_incomplete_fixed_audit']=True
        groups.append(dict(depth=depth,time=64,numerical_audit=value,
                           counts=dict(Counter(r['status'] for r in selected))))
    return json_safe(dict(schema='shared-base-local-revalidation-v1',plan_id=plan['plan_id'],
        complete=complete,expected_references=len(expected),observed_references=len(found),groups=groups,
        counts=dict(Counter(r['corrected']['audit']['status'] for r in rows)),references=references,
        original_counts=dict(Counter(r['original_audit']['status'] for r in rows)),
        changed_cases=[r['task']['key'] for r in rows if r['corrected']['change']['old_status']!=r['corrected']['change']['new_status']],
        missing_keys=sorted(set(expected)-set(found)),
        all_cohorts_passed=complete and all(g['numerical_audit']['passed'] for g in groups),
        no_sampling=True, historical_reports_unchanged=True, correction_not_sampling=True,
        interpretation='Fixed whole-cohort raw audit; individual readiness is not cohort integration acceptance.'))


def audit_pool(workers,plan):
    # Python's recycling pool can deadlock once every worker reaches its quota
    # while queued work remains. Numerical caches are bounded; keep workers alive.
    return ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                               initializer=configure_runtime,initargs=(plan,))


def run(out=DEFAULT_OUT,workers=4):
    require(1<=workers<=4,'Use one through four single-thread audit processes')
    out=Path(out).resolve();plan,digest=read_plan(out);configure_runtime(plan)
    import run_shared_base_curve_campaign as campaign
    original,_=read_json(plan['original_plan']);campaign.verify_assets(original,original['global_paths'])
    rows=[];pending=[]
    with (out/'revalidation.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in plan['cases']:
            path=case_path(out,task)
            if path.exists():
                row,_=read_json(path);validate_row(row,task,digest);rows.append(row)
            else:pending.append(task)
        failures={}
        with audit_pool(workers,plan) as pool:
            futures={pool.submit(audit_one_reference,task,digest):task for task in pending}
            for future in as_completed(futures):
                task=futures[future]
                try:
                    row=future.result();write_json(case_path(out,task),row,immutable=True);rows.append(row)
                except Exception as error:
                    failures[task['key']]=dict(type=type(error).__name__,error=str(error))
                progress=dict(completed=len(rows),expected=len(plan['cases']),failures=failures,
                    counts=dict(Counter(r['corrected']['audit']['status'] for r in rows)),no_sampling=True)
                write_json(out/'status.json',progress)
                print(json.dumps(dict(key=task['key'],**progress)),flush=True)
        require(read_plan(out)[1]==digest,'Plan changed during execution')
        campaign.verify_assets(original,original['global_paths'])
        for row in rows:check_task_files(row['task'])
        summary=summarize_rows(plan,rows);summary['failures']=failures;summary['launch_sha256']=digest
        write_json(out/'summary.json',summary)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN)
    parser.add_argument('--continuation',type=Path,default=DEFAULT_CONTINUATION)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    result=prepare(args.campaign,args.continuation,args.out) if args.action=='prepare' else run(args.out,args.workers)
    print(json.dumps({k:v for k,v in result.items() if k in ('plan_id','complete','counts','all_cohorts_passed')}))


if __name__=='__main__':main()
