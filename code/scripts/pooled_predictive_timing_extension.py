#!/usr/bin/env python3
"""Nine isolated timing measurements; no numerical or scientific approval.

Login-node imports and submission use only the standard library. Numerical
validation and the unchanged population runner execute inside allocations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'code/transformer/configs/pooled_predictive_timing_extension_queue_recovery_plan_2026-09-19.json'
PARENT = ROOT/'code/transformer/configs/pooled_predictive_population_plan_2026-09-19.json'
CORE = 'code/scripts/pooled_predictive_population.py'
TASK_IDS = (2, 10, 13, 18, 21, 24, 27, 30, 33)
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','PREEMPTED','NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED'}
DEADLINE_ERROR = 'bounded SMC/prediction repetition exceeded its time limit'


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_plan(path=PLAN):
    extension = json.loads(Path(path).read_text())
    require(extension['schema'] == 'pooled-predictive-timing-extension-plan-v1', 'Wrong extension plan')
    require(extension['task_ids'] == list(TASK_IDS), 'Timing selection changed')
    require(extension['timing'] == dict(seconds_per_replica=7200, allocation_minutes=150, concurrency=3,
            maximum_attempts_per_task=1, worker_cpu_hour_cap=22.5, setup_and_merge_cpu_hour_cap=1.), 'Timing limits changed')
    require(extension['original_array_job'] == '66595885', 'Original timing dependency changed')
    require(extension['parent_plan']['path'] == str(PARENT.relative_to(ROOT))
            and sha(PARENT) == extension['parent_plan']['sha256'], 'Original frozen plan changed')
    parent = json.loads(PARENT.read_text())
    required = set(parent['code_sha256']) | {CORE, str(Path(__file__).resolve().relative_to(ROOT)),
        'code/transformer/scitas/pooled_predictive_timing_extension.sbatch',
        'code/transformer/scitas/submit_pooled_predictive_timing_extension.sh',
        'code/transformer/scitas/environment.sh'}
    require(set(extension['code_sha256']) == required, 'Incomplete implementation binding')
    for name, value in extension['code_sha256'].items():
        require(sha(ROOT/name) == value, 'Frozen implementation changed: ' + name)
    require(all(extension['code_sha256'][k] == v for k,v in parent['code_sha256'].items()), 'Original numerical implementation changed')
    require(digest(parent['selection']) == parent['selection_digest'], 'Original selection changed')
    require(sha(ROOT/'code/transformer/configs/pooled_predictive_calibration_plan_2026-09-18.json')
            == parent['calibration_plan_sha256'], 'Original calibration changed')
    return extension, sha(path)


def adapted_plan(parent, extension):
    value = copy.deepcopy(parent)
    value['stages']['timing'].update({k: extension['timing'][k]
        for k in ('seconds_per_replica','allocation_minutes','concurrency')})
    value['resource_limits']['maximum_attempts_per_task'] = extension['timing']['maximum_attempts_per_task']
    return value


@contextmanager
def timing_adapter(module, parent, extension, extension_hash):
    original = module.read_plan
    adjusted = adapted_plan(parent, extension)
    module.read_plan = lambda path=PLAN: (copy.deepcopy(adjusted), extension_hash)
    try:
        yield module
    finally:
        module.read_plan = original


def compute_context(args):
    require(os.environ.get('SLURM_JOB_ID'), 'Numerical work requires a compute allocation')
    extension, extension_hash = read_plan(args.plan)
    sys.path.insert(0, str(ROOT/'code/scripts'))
    import pooled_predictive_population as module
    parent, parent_hash = module.read_plan(PARENT)
    require(parent_hash == extension['parent_plan']['sha256'], 'Parent plan differs')
    cp, _ = module.calibration.read_plan(module.calibration.PLAN)
    dep = Path(args.dependencies)
    dependencies = module.calibration.dependencies(cp, dep/'tables/anchors_prod', dep/'src')
    environment = module.calibration.numerical_environment(dep/'tables/anchors_prod', dep/'src')
    return module, parent, extension, extension_hash, dependencies, environment


def validate_report(module, plan, plan_hash, spec, path, raw, dependencies, environment):
    value = json.loads(Path(path).read_text())
    require(value['schema'] == 'pooled-predictive-population-task-v1' and value['task'] == spec, 'Task settings changed')
    identity = value['identity']
    require(identity['plan_sha256'] == plan_hash and identity['driver_sha256'] == sha(ROOT/CORE)
        and identity['code_sha256'] == plan['code_sha256'], 'Task implementation identity changed')
    require(identity['dependency_sha256'] == dependencies and identity['numerical_environment'] == environment,
        'Task numerical dependencies changed')
    require(identity['runtime'] == dict(python=sys.version, numpy=module.np.__version__,
        scipy=module.calibration.scipy.__version__), 'Original numerical runtime changed')
    tokens, oracle, scores, hashes = module.load_case(raw, plan, spec)
    require(identity['source_sha256'] == hashes and value['tokens'] == tokens.tolist()
        and value['oracle'] == oracle.tolist() and value['transformer_regret_bits'] == scores,
        'Original source/prior/history pairing changed')
    records = value['records']
    require(value['records_sha256'] == module.digest(records) and len(records) <= 1
        and value['complete'] == (len(records) == 1), 'Repetition coverage or digest changed')
    require(1 <= value['attempts'] <= plan['resource_limits']['maximum_attempts_per_task'], 'Attempt cap exceeded')
    for replica, record in enumerate(records):
        module.validate_estimate(record, plan, spec, replica)
    require(set(value['cheap_components']) == {'product','uniform_base'}, 'Fixed components changed')
    for component in value['cheap_components'].values():
        module.checked_component(component)
    require(value['cheap_components_sha256'] == module.digest(value['cheap_components']), 'Fixed components changed')
    if records:
        require(value['summary'] == module.record_summary(records, value['cheap_components'], oracle, scores), 'Statistics do not replay')
        require(value['status'] == 'complete', 'Completion status differs')
    return value


def require_original_terminal(job):
    # Completed jobs can disappear from squeue while remaining in sacct.
    # Query this user's live queue, then require positive accounting evidence.
    queued = subprocess.run(['squeue','--noheader','--array','--user',getpass.getuser(),
        '--format=%i|%T'], check=True, text=True, stdout=subprocess.PIPE)
    for line in queued.stdout.splitlines():
        live_id = line.split('|', 1)[0].strip()
        require(live_id != job and not live_id.startswith(job+'_'), 'Original timing array is still active')
    result = subprocess.run(['sacct','-X','--noheader','--parsable2','--jobs',job,'--format=JobID,State'],
        check=True, text=True, stdout=subprocess.PIPE)
    seen = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split('|')
        if len(fields) < 2 or not fields[0].startswith(job+'_'):
            continue
        task = fields[0].split('_',1)[1]
        if not task.isdigit():
            continue
        require(fields[1].split()[0].rstrip('+') in TERMINAL, 'Original array accounting is not terminal')
        seen.add(int(task))
    require(set(TASK_IDS) <= seen, 'Original selected tasks lack terminal scheduler accounting')


def paths_identity(args, extension_hash):
    return dict(plan_sha256=extension_hash, **{k:str(Path(getattr(args,k)).resolve())
        for k in ('raw','dependencies','out','calibration','original_out')}, after_job=args.after_job)


def validate_parent_submission(args, parent_hash):
    """Link the dependency job to the declared original output and selected IDs."""
    found = False
    for path in (Path(args.original_out)/'submissions').glob('*.json'):
        receipt = json.loads(path.read_text())
        ident = receipt['identity']
        for attempt in receipt['attempts']:
            for job in attempt['jobs']:
                if job['job_id'] != args.after_job:
                    continue
                require(receipt['key'] == digest(ident) and ident['stage'] == 'timing'
                    and ident['plan_sha256'] == parent_hash and ident['out'] == str(Path(args.original_out).resolve())
                    and all(ident[k] == str(Path(getattr(args,k)).resolve()) for k in ('raw','dependencies','calibration'))
                    and job['mode'] == 'run' and set(TASK_IDS) <= set(job['indices']), 'Original submission provenance differs')
                found = True
    require(found, 'No original submission receipt binds the dependency array')


def prepare(args):
    print(json.dumps(dict(event='preparation_started', original_array=args.after_job)), flush=True)
    module, parent, extension, extension_hash, deps, env = compute_context(args)
    print(json.dumps(dict(event='inputs_and_runtime_verified')), flush=True)
    require(args.after_job == extension['original_array_job'], 'Original array dependency required')
    require_original_terminal(args.after_job)
    validate_parent_submission(args, extension['parent_plan']['sha256'])
    print(json.dumps(dict(event='original_array_and_receipt_verified')), flush=True)
    specs = module.task_specs(parent, 'timing')
    entries, errors = {}, {}
    for index in TASK_IDS:
        path = module.stage_path(args.original_out, 'timing', index)
        try:
            require(not path.with_suffix('.writer').exists(), 'Original task still has an unresolved writer claim')
            value = validate_report(module, parent, extension['parent_plan']['sha256'], specs[index], path, args.raw, deps, env)
            require(value['complete'] or (value['status'] == 'time_limit' and value.get('error') == DEADLINE_ERROR),
                'Selected original task did not finish or reach the sampling deadline')
            entries[str(index)] = dict(path=str(path.resolve()), sha256=sha(path),
                action='reuse_original' if value['complete'] else 'extend_timeout')
            print(json.dumps(dict(event='original_report_verified', task=index,
                action=entries[str(index)]['action'])), flush=True)
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors[str(index)] = str(error)
            print(json.dumps(dict(event='original_report_rejected', task=index, error=str(error))), flush=True)
    require(not errors, 'Original report checks failed: '+json.dumps(errors, sort_keys=True))
    report = dict(schema='pooled-predictive-timing-extension-prepared-v1', identity=paths_identity(args, extension_hash),
        original_plan_sha256=extension['parent_plan']['sha256'], original_array_job=args.after_job,
        tasks=entries, scientific_reference_validated=False)
    path = Path(args.out)/'prepared.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with module.calibration.writer_claim(path):
        if path.exists():
            require(json.loads(path.read_text()) == report, 'Prepared inputs changed')
        else:
            atomic_json(path, report)
    print(json.dumps(dict(event='preparation_complete', tasks=len(entries), sampling_started=False)), flush=True)
    return 0


def read_prepared(args, extension_hash):
    value = json.loads((Path(args.out)/'prepared.json').read_text())
    require(value['schema'] == 'pooled-predictive-timing-extension-prepared-v1'
        and value['identity'] == paths_identity(args, extension_hash)
        and set(value['tasks']) == {str(i) for i in TASK_IDS}, 'Prepared timing identity changed')
    for index, item in value['tasks'].items():
        expected = Path(args.original_out)/'timing/tasks'/f'task-{int(index):04d}.json'
        require(Path(item['path']).resolve() == expected.resolve(), 'Original task path changed')
        require(sha(item['path']) == item['sha256'], 'Original timing report changed after preparation')
        original = json.loads(Path(item['path']).read_text())
        require(item['action'] == ('reuse_original' if original['complete'] else 'extend_timeout'),
            'Original completion action changed')
    return value


def run(args):
    module, parent, extension, extension_hash, deps, env = compute_context(args)
    require(args.task_index in TASK_IDS, 'Task outside the nine selected measurements')
    prepared = read_prepared(args, extension_hash)
    item = prepared['tasks'][str(args.task_index)]
    if item['action'] == 'reuse_original':
        print(json.dumps(dict(task_index=args.task_index, action='reuse_original', original_report_sha256=item['sha256'],
            original_plan_sha256=extension['parent_plan']['sha256'], scientific_reference_validated=False)))
        return 0
    require(item['action'] == 'extend_timeout', 'Unknown prepared action')
    with timing_adapter(module, parent, extension, extension_hash):
        return module.run(args.plan, 'timing', args.task_index, args.raw, args.dependencies, args.out, args.calibration)


def summarize(extension, rows):
    require(set(rows) <= set(TASK_IDS), 'Unexpected timing task')
    cases = []
    for index in TASK_IDS:
        row = rows.get(index)
        if row is None:
            cases.append(dict(task_index=index, complete=False, status='missing'))
            continue
        records = row['records']
        complete = row['complete'] is True and len(records) == 1 and row['status'] == 'complete'
        seconds = float(records[0]['elapsed_seconds']) if complete else None
        require(seconds is None or (math.isfinite(seconds) and seconds >= 0), 'Invalid measured time')
        cases.append(dict(task_index=index, task=row['task'], complete=complete, status=row['status'],
            repetition_seconds=seconds, invocation_elapsed_seconds=row.get('invocation_elapsed_seconds'),
            source=row.get('timing_source','extension'), original_report_sha256=row.get('original_report_sha256'),
            error=row.get('error'), deadline_censored=(not complete and row.get('error') == DEADLINE_ERROR),
            sampler_seconds=records[0].get('sampler_seconds') if complete else None,
            prediction_seconds=records[0].get('prediction_seconds') if complete else None))
    completed = [r['task_index'] for r in cases if r['complete']]
    missing = [r['task_index'] for r in cases if not r['complete']]
    return dict(schema='pooled-predictive-timing-extension-summary-v1', expected_tasks=list(TASK_IDS),
        completed_tasks=completed, missing_tasks=missing, complete=not missing, cases=cases,
        timing=extension['timing'], scientific_reference_validated=False, accuracy_approved=False,
        production_approved=False, note='One repetition per selected development case measures cost only. '
        'Incomplete measurements are retained. No population or accuracy decision follows from these timings.')


def merge(args):
    module, parent, extension, extension_hash, deps, env = compute_context(args)
    rows, hashes, errors = {}, {}, {}
    prepared_path = Path(args.out)/'prepared.json'
    if not prepared_path.exists():
        errors['prepare'] = 'Preparation did not complete; no timing workers were authorized'
    else:
        prepared = read_prepared(args, extension_hash)
        adjusted = adapted_plan(parent, extension)
        specs = module.task_specs(parent, 'timing')
        for index in TASK_IDS:
            item = prepared['tasks'][str(index)]
            reused = item['action'] == 'reuse_original'
            path = Path(item['path']) if reused else module.stage_path(args.out, 'timing', index)
            if not path.exists():
                continue
            try:
                value = validate_report(module, parent if reused else adjusted,
                    extension['parent_plan']['sha256'] if reused else extension_hash, specs[index], path, args.raw, deps, env)
                value['timing_source'] = 'original_completed' if reused else 'extension'
                value['original_report_sha256'] = item['sha256']
                rows[index] = value
                hashes[str(index)] = dict(path=str(path.resolve()), sha256=sha(path))
            except (ValueError, KeyError, TypeError) as error:
                errors[str(index)] = str(error)
    summary = summarize(extension, rows)
    summary.update(identity=paths_identity(args, extension_hash), parent_plan_sha256=extension['parent_plan']['sha256'],
        report_sha256=hashes, validation_errors=errors,
        prepared_sha256=sha(prepared_path) if prepared_path.exists() else None)
    path = Path(args.out)/'summary.json'
    with module.calibration.writer_claim(path):
        atomic_json(path, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['complete'] and not errors else 3


def validate_paths(args):
    for key in ('plan','raw','dependencies','out','calibration','original_out'):
        value = Path(getattr(args,key)).expanduser().resolve()
        require(not any(c in str(value) for c in '\n\r\t'), 'Paths cannot contain control characters')
        setattr(args,key,value)
    protected = [args.raw,args.dependencies,args.calibration,args.original_out,
        *[ROOT/q for q in ('code','results','notes','figures','.git')]]
    require(args.out != ROOT and args.out not in ROOT.parents and not any(
        args.out == q or args.out in q.parents or q in args.out.parents for q in protected),
        'Extension output must be separate from all original data, outputs and source code')


@contextmanager
def submission_lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path/'.submission.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def submit(args):
    validate_paths(args)
    extension, extension_hash = read_plan(args.plan)
    require(args.after_job == extension['original_array_job'], 'Use --after-job 66595885 to identify the original timing array')
    identity = paths_identity(args, extension_hash)
    identity['plan'] = str(args.plan)
    key = digest(identity)
    script = ROOT/'code/transformer/scitas/pooled_predictive_timing_extension.sbatch'
    def command(mode, dependency):
        condition = 'afterok:' if mode == 'run' else 'afterany:'
        cmd = ['sbatch','--parsable','--chdir='+str(ROOT),'--job-name=ppop-time-ext-'+mode,
            '--time='+('02:30:00' if mode == 'run' else '00:30:00'),
            '--output='+str(args.out/'logs'/f'{mode}-%A_%a.out')]
        # Preparation checks terminal accounting itself; a scheduler dependency
        # on an expired original job ID can be rejected before it can run.
        if mode != 'prepare':
            cmd += ['--dependency='+condition+dependency,'--kill-on-invalid-dep=yes']
        if mode == 'run':
            cmd += ['--array=0-8%3']
        return cmd + [str(script),str(args.plan),str(args.raw),str(args.dependencies),str(args.out),
            str(args.calibration),str(args.original_out),args.after_job,mode]
    print(json.dumps(dict(tasks=list(TASK_IDS), concurrency=3, worker_cpu_hour_cap=22.5,
        setup_and_merge_cpu_hour_cap=1., scientific_reference_validated=False, automatic_production=False)))
    if not args.submit:
        print(shlex.join(command('prepare', args.after_job)))
        print(shlex.join(command('run', 'PREPARE_JOB')))
        print(shlex.join(command('merge', 'WORKER_JOB')))
        return 0
    directory = args.out/'submissions'
    with submission_lock(directory):
        saved = list(directory.glob('*.json'))
        if saved:
            require(len(saved) == 1, 'Multiple extension receipts; inspect before submitting')
            receipt = json.loads(saved[0].read_text())
            require(receipt['key'] == key and receipt['identity'] == identity,
                'Extension output already has a different submission; no overlapping or additional wave is permitted')
            require(receipt['status'] == 'submitted' and receipt.get('submitting') is None and len(receipt['jobs']) == 3,
                'Earlier dispatch is incomplete or uncertain; inspect scheduler jobs before any further action')
            print(json.dumps(dict(action='already_submitted', receipt=str(saved[0]), jobs=receipt['jobs'])))
            return 0
        require(not (args.out/'prepared.json').exists() and not (args.out/'timing').exists(), 'Unreceipted extension output already exists')
        require((args.raw/'heldout/manifest.json').is_file()
            and (args.dependencies/'tables/anchors_prod/manifest.json').is_file(), 'Frozen source or dependency paths missing')
        receipt_path = directory/(key+'.json')
        receipt = dict(schema='pooled-predictive-timing-extension-submission-v1', key=key, identity=identity,
            status='preparing', submitting=None, jobs=[])
        (args.out/'logs').mkdir(parents=True, exist_ok=True)
        atomic_json(receipt_path, receipt)
        dependency = args.after_job
        for mode in ('prepare','run','merge'):
            cmd = command(mode, dependency)
            receipt.update(status='submitting', submitting=dict(mode=mode, command=cmd))
            atomic_json(receipt_path, receipt)
            print(shlex.join(cmd), flush=True)
            result = subprocess.run(cmd, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE)
            job = result.stdout.strip().split(';')[0]
            require(job.isdigit(), 'Scheduler reply is uncertain; inspect the saved receipt before resubmitting')
            receipt['jobs'].append(dict(mode=mode, job_id=job, maximum_cpu_hours=22.5 if mode=='run' else .5))
            receipt.update(status='preparing', submitting=None)
            atomic_json(receipt_path, receipt)
            dependency = job
        receipt['status'] = 'submitted'
        atomic_json(receipt_path, receipt)
        print(json.dumps(dict(action='submitted', receipt=str(receipt_path), jobs=receipt['jobs'])))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('submit','prepare','run','merge'))
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--raw', type=Path, default=Path.home()/'Projects/layered_architectures_multifamily/output/multifamily_2026-09-16')
    parser.add_argument('--dependencies', type=Path, default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
    parser.add_argument('--out', type=Path, default=ROOT/'output/pooled_predictive_timing_extension_queue_recovery_2026-09-19')
    parser.add_argument('--original-out', type=Path, default=ROOT/'output/pooled_predictive_population_2026-09-19')
    parser.add_argument('--calibration', type=Path, default=ROOT/'output/pooled_predictive_calibration_2026-09-18/stronger')
    parser.add_argument('--after-job')
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    validate_paths(args)
    return globals()[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
