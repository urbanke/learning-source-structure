"""Revalidate depth R-hat arithmetic on all 32 saved posterior references.

Only the depth contribution to the split-Rhat gate changes. Samples, scores,
all other numerical gates and historical individual precision flags stay fixed.
This is an arithmetic correction, not new sampling or a convergence certificate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

import inspect_shared_base_depth_diagnostics as diagnostic
import recover_shared_base_posterior_audit as recovery
from prepare_shared_base_posterior_cluster import (
    ROOT, DEPTHS, TIMES, ORDERS, INDICES, require, sha, fingerprint, immutable_json)

DEFAULT_OUT = ROOT/'output/shared_base_depth_revalidation_2026-09-17'
OWN_FILES = ('code/scripts/revalidate_shared_base_depth.py',
             'code/scripts/inspect_shared_base_depth_diagnostics.py',
             'code/scripts/depth_rhat_statistics.py',
             'code/transformer/scitas/submit_shared_base_depth_revalidation.sh',
             'code/transformer/scitas/shared_base_depth_revalidation.sbatch')


def adjust_depth_gate(reference, audited_result, depth_diagnostics):
    """Copy an audit, correcting only depth R-hat and its derived decisions."""
    from shared_base_curve_audit import diagnostic_gates, same
    require([d['depth'] for d in depth_diagnostics] == list(range(9)), 'All nine ordered depths are required')
    require(same(audited_result['record'],reference['record']), 'Audited/reference records differ')
    old_gates = diagnostic_gates(reference['estimate'])
    require(same(old_gates,audited_result['gates']), 'Historical gates differ from saved estimate')
    corrected_rhats=[]
    for depth,row in enumerate(depth_diagnostics):
        require(row['chains']==4 and row['draws']==reference['estimate']['diagnostics']['draws'],
                'Depth diagnostic sampling dimensions differ')
        previous=reference['estimate']['diagnostics']['rhat_depth'][depth]
        require(same(row['original']['rhat'],previous), 'Legacy depth Rhat readback differs')
        require(math.isclose(row['mean'],reference['estimate']['posterior_depth'][depth],rel_tol=2e-12,abs_tol=0),
                'Depth diagnostic mean differs')
        value=row['rescaled']['rhat']; status=row['rescaled']['status']
        if status=='constant_uninformative' and row['min']==row['max']:
            require(previous==1.0, 'Exact common constant must reproduce the legacy unit fallback')
            value=1.0  # Preserve the original convention; this carries no mixing information.
        elif not (status=='finite' and value is not None and math.isfinite(value) and value>=0):
            value=None  # Different constants or unresolved arithmetic remain failures.
        corrected_rhats.append(value)
    estimate=deepcopy(reference['estimate'])
    estimate['diagnostics']['rhat_depth']=corrected_rhats
    new_gates=diagnostic_gates(estimate)
    require(all(new_gates[k]==old_gates[k] for k in old_gates if k!='split_rhat_below_1_05'),
            'An unrelated numerical gate changed')
    result=deepcopy(audited_result)
    result['gates']=new_gates
    precision=result['record']['precision']
    result['record']['mixing_passed']=bool(all(new_gates.values()) and precision['scalar_mixing_passed'])
    result['ready']=bool(result['record']['mixing_passed'] and result['record']['row_grid_passed']
                         and precision['jensen_local_check_passed'])
    result['status']='ready' if result['ready'] else 'needs_more_draws'
    historical_guidance={key:result.pop(key) for key in (
        'estimated_minimum_draws_for_ess','suggested_next_draws','draw_deficit','todo') if key in result}
    change=dict(old_status=audited_result['status'],new_status=result['status'],old_gates=old_gates,new_gates=new_gates,
        old_depth_rhat=reference['estimate']['diagnostics']['rhat_depth'],corrected_depth_rhat=corrected_rhats,
        historical_per_sequence_precision_passed=reference['record']['per_sequence_precision_passed'],
        historical_flag_note='Preserved unchanged; this historical flag is not a cohort acceptance gate.',
        historical_continuation_guidance=historical_guidance,
        correction_not_sampling=True)
    return dict(audit=result,change=change)


def aggregate_rows(rows,tasks):
    from shared_base_curve_audit import aggregate_time
    from validate_shared_base_mcmc import json_safe
    expected={t['id']:t for t in tasks}; found={r['task']['id']:r for r in rows}
    require(len(tasks)==32 and len(expected)==32 and len(found)==len(rows) and set(found)<=set(expected),
            'Duplicate or undeclared revalidation tasks')
    require(all(row['task']==expected[key] for key,row in found.items()), 'Revalidation task identity differs')
    complete=len(found)==32; groups=[]
    for depth in DEPTHS:
        for t in TIMES:
            selected=[task for task in tasks if task['depth']==depth and task['time']==t]
            audits=[found[task['id']]['corrected']['audit'] for task in selected if task['id'] in found]
            value=aggregate_time(audits,list(INDICES),list(ORDERS))
            if not complete:
                value['reported_mean_bayes_regret_bits']=None
                value['release_blocked_by_incomplete_pilot']=True
            groups.append(dict(depth=depth,time=t,numerical_audit=value))
    return json_safe(dict(complete=complete,expected_references=32,groups=groups,
        all_numerical_checks_passed=complete and all(g['numerical_audit']['passed'] for g in groups)))


def build_launch(args):
    source=diagnostic.locate_launch(args.recovery)
    recovered,old,_=recovery.read_recovery(source)
    out=args.out.expanduser().resolve()
    for protected in (Path(recovered['context']['output']),Path(old['campaign']['output']),
                      Path(old['campaign']['inputs']),diagnostic.DEFAULT_OUT):
        protected=protected.resolve()
        require(out!=protected and out not in protected.parents and protected not in out.parents,
                'Revalidation output must be separate from all earlier artifacts')
    spec=dict(schema='shared-base-depth-revalidation-launch-v1',recovery_launch=str(source),
        recovery_launch_sha256=sha(source),output=str(out),task_indices=list(range(32)),depths=list(range(9)),
        code_sha256={name:sha(ROOT/name) for name in OWN_FILES},no_sampling=True,
        correction='scale-invariant depth split-Rhat only',venv=recovered['venv'])
    spec['launch_id']=fingerprint(spec)
    path=out/'launch.json'
    command=['sbatch','--parsable',f'--chdir={ROOT}','--job-name=sb-depth-revalidation',
        f'--output={out}/logs/revalidation-%j.out',str(ROOT/OWN_FILES[-1]),str(path),spec['venv']]
    return spec,path,command


def submit(args):
    spec,path,command=build_launch(args)
    print(shlex.join(command),flush=True)
    if not args.submit:
        print('Dry run only; no files written or jobs submitted.',flush=True);return
    require((Path(spec['venv'])/'bin/python').is_file(),'Prepared Python environment is missing')
    immutable_json(path,spec);(Path(spec['output'])/'logs').mkdir(exist_ok=True)
    subprocess.run(command,cwd=ROOT,check=True)


def check_row(row,task,spec_hash,runtime,launch,old,plan,*,replay_depths=True):
    from shared_base_curve_audit import same
    require(row.get('row_id')==fingerprint({k:v for k,v in row.items() if k!='row_id'})
            and row['task']==task and row['launch_sha256']==spec_hash and row['runtime']==runtime,
            'Saved revalidation row identity/checksum mismatch')
    receipt_path=Path(launch['context']['output'])/'tasks'/f"{task['id']}.json"
    receipt,digest=recovery.checked_json(receipt_path)
    audit=recovery.validate_recovered(receipt,launch['context'],old,plan,task)
    require(digest==row['recovery_receipt_sha256'] and receipt['runtime']==runtime
            and receipt['original']['files']==row['original_file_sha256'],'Revalidation inputs changed')
    paths=recovery.original.task_paths(old['campaign'],task)
    reference,_=recovery.checked_json(paths['reference'])
    depth_rows=row['depth_diagnostics']
    if replay_depths:
        depth_rows=diagnostic.summarize_depths(paths['checkpoint'],reference['estimate'])
        require(same(depth_rows,row['depth_diagnostics'],atol=0,rtol=2e-12),
                'Saved depth diagnostics differ from raw depth arrays')
    require(same(row['corrected'],adjust_depth_gate(reference,audit,depth_rows)),
            'Saved revalidation arithmetic differs')
    require(sha(receipt_path)==digest and receipt['original']==recovery.original_snapshot(old,plan,task),
            'Original artifacts changed during depth replay')


def run(path):
    spec,spec_hash=recovery.checked_json(path)
    require(spec.get('schema')=='shared-base-depth-revalidation-launch-v1'
            and spec['launch_id']==fingerprint({k:v for k,v in spec.items() if k!='launch_id'}),
            'Revalidation launch identity/checksum mismatch')
    require(spec['task_indices']==list(range(32)) and spec['depths']==list(range(9)) and spec['no_sampling'] is True
            and spec['correction']=='scale-invariant depth split-Rhat only','Revalidation scope changed')
    require(spec['code_sha256']=={name:sha(ROOT/name) for name in OWN_FILES},'Revalidation code changed')
    require(sha(spec['recovery_launch'])==spec['recovery_launch_sha256'],'Recovery launch changed')
    launch,old,plan=recovery.read_recovery(spec['recovery_launch'])
    runtime=recovery.original.runtime_setup(old['campaign']['dependency_root'])
    from validate_shared_base_mcmc import json_safe
    out=Path(spec['output']);rows=[]
    with (out/'revalidation.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in plan['tasks']:
            row_path=out/'tasks'/f"{task['id']}.json"
            paths=recovery.original.task_paths(old['campaign'],task)
            with paths['reference'].with_suffix('.lock').open('rb') as lock:
                fcntl.flock(lock,fcntl.LOCK_SH | fcntl.LOCK_NB)
                if row_path.exists():
                    row,_=recovery.checked_json(row_path)
                    check_row(row,task,spec_hash,runtime,launch,old,plan)
                else:
                    receipt_path=Path(launch['context']['output'])/'tasks'/f"{task['id']}.json"
                    receipt,digest=recovery.checked_json(receipt_path)
                    audit=recovery.validate_recovered(receipt,launch['context'],old,plan,task)
                    require(receipt['runtime']==runtime,'Use the original numerical runtime')
                    reference,_=recovery.checked_json(paths['reference'])
                    started=time.monotonic()
                    depth_rows=diagnostic.summarize_depths(paths['checkpoint'],reference['estimate'])
                    row=json_safe(dict(task=task,launch_sha256=spec_hash,runtime=runtime,
                        recovery_receipt_sha256=digest,original_file_sha256=receipt['original']['files'],
                        depth_diagnostics=depth_rows,corrected=adjust_depth_gate(reference,audit,depth_rows),
                        elapsed_seconds=time.monotonic()-started))
                    row['row_id']=fingerprint(row)
                    check_row(row,task,spec_hash,runtime,launch,old,plan,replay_depths=False)
                    immutable_json(row_path,row)
            rows.append(row)
            print(json.dumps(dict(task=task['id'],old_status=row['corrected']['change']['old_status'],
                new_status=row['corrected']['change']['new_status'],correction_not_sampling=True)),flush=True)
        # Recheck all earlier inputs once more before the complete release.
        for row in rows:
            check_row(row,row['task'],spec_hash,runtime,launch,old,plan,replay_depths=False)
        require(sha(path)==spec_hash and spec['code_sha256']=={name:sha(ROOT/name) for name in OWN_FILES}
                and sha(spec['recovery_launch'])==spec['recovery_launch_sha256'],
                'Revalidation manifest, helper code or recovery launch changed during the run')
        recovery.read_recovery(spec['recovery_launch'])  # Recheck every original bound code/input/dependency file.
        result=dict(schema='shared-base-depth-revalidation-v1',launch_sha256=spec_hash,runtime=runtime,
            **aggregate_rows(rows,plan['tasks']),tasks=rows,no_sampling=True,correction_not_sampling=True,
            old_counts={s:sum(r['corrected']['change']['old_status']==s for r in rows) for s in ('ready','needs_more_draws')},
            new_counts={s:sum(r['corrected']['change']['new_status']==s for r in rows) for s in ('ready','needs_more_draws')},
            historical_reports_unchanged=True,production_approved=False,full_population_claim=False,
            interpretation='Uniform arithmetic correction across all32 references and all9 depths. Samples and numerical scores are unchanged. This does not certify convergence; two fixed sequences per condition are a computational pilot only.')
        immutable_json(out/'summary.json',result)
        print(json.dumps(dict(complete=result['complete'],passed=result['all_numerical_checks_passed'],
                              old_counts=result['old_counts'],new_counts=result['new_counts'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    launch=modes.add_parser('launch')
    launch.add_argument('--recovery',type=Path,default=recovery.DEFAULT_OUT)
    launch.add_argument('--out',type=Path,default=DEFAULT_OUT);launch.add_argument('--submit',action='store_true')
    worker=modes.add_parser('run');worker.add_argument('--launch',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='launch':submit(args)
    else:run(args.launch)


if __name__=='__main__':main()
