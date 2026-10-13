"""Resume a bounded campaign without changing its declared source population.

The frozen plan always contains all 256 cases at depths 0,1,4,8, times
16,64,256,1024 and exponent orders 16/32. CLI selections restrict dispatch
for this invocation; they never remove cases from the plan. Completed ready
references and exhausted unresolved cases are retained, not rerun or replaced.
This dispatcher never declares a validated cohort mean.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'shared-base-curve-campaign-v1'
DEPTHS, TIMES, ORDERS, N_CASES = (0, 1, 4, 8), (16, 64, 256, 1024), (16, 32), 256
TERMINAL = {'ready', 'unresolved', 'failed'}
CODE = [
    'code/scripts/run_shared_base_curve_campaign.py',
    'code/scripts/shared_base_curve_reference.py',
    'code/scripts/shared_base_conditional_cohort.py',
    'code/scripts/conditional_shared_base_benchmark.py',
    'code/scripts/shared_base_precision_cohort.py',
    *['code/transformer/'+name for name in (
        'shared_base_curve_runner.py', 'shared_base_chain_runner.py',
        'shared_base_surrogate.py', 'shared_base_efficiency.py',
        'surrogate_chain_runner.py', 'shared_base_mcmc.py', 'ranked_lsa.py',
        'weighted_lsa.py', 'shared_depth.py', 'lsa_seq.py', 'phi_source.py',
        'source.py', 'stable_chain_statistics.py', 'full_context_statistics.py',
        'eval_shared_base_reference.py', 'eval_shared_base_models.py',
        'validate_shared_base_mcmc.py')],
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def job_key(depth, index, prefix, order):
    return f'a1_L{depth}/seq_{index:06d}/t{prefix:04d}/alpha{order}'


def all_jobs():
    # Complete a time stage first; interleave depth and quadrature rules
    # within each declared case index, rather than finishing one depth first.
    return [dict(key=job_key(d, i, t, a), depth=d, index=i, prefix=t, order=a)
            for t in TIMES for i in range(N_CASES) for a in ORDERS for d in DEPTHS]


def selected_jobs(plan, times, depths, indices):
    for values, allowed, name in ((times, TIMES, 'times'), (depths, DEPTHS, 'depths'),
                                  (indices, range(N_CASES), 'indices')):
        if not values or len(values) != len(set(values)) or not set(values).issubset(allowed):
            raise ValueError(f'unsupported, duplicate or empty {name}')
    return [j for j in plan['jobs'] if j['prefix'] in times and j['depth'] in depths and j['index'] in indices]


def build_plan(args):
    source_root, references = args.input.resolve(), args.references.resolve()
    assets, cases, global_paths = {}, {}, []
    def freeze(path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'declared file changed: {path}')
        assets[str(path)] = actual
        return actual
    for relative in CODE:
        path = (ROOT/relative).resolve()
        freeze(path)
        global_paths.append(str(path))
    for depth in DEPTHS:
        folder = source_root/f'a1_L{depth}'
        cohort_path, protocol_path = folder/'cohort.json', folder/'protocol.json'
        cohort = json.loads(cohort_path.read_text())
        protocol = json.loads(protocol_path.read_text())
        if (cohort.get('schema') != 'shared-base-conditional-cohort-v1'
                or cohort.get('n_sequences') != N_CASES or len(cohort.get('records', [])) != N_CASES
                or cohort.get('alpha') != 1. or cohort.get('depth') != depth
                or cohort.get('n_tokens_per_sequence') != 1025
                or not set(TIMES).issubset(cohort.get('prefixes', []))
                or protocol.get('alpha') != 1. or protocol.get('depth') != depth
                or protocol.get('conditions_hidden_from_predictors') is not True
                or protocol.get('reference_prior') != dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation')):
            raise ValueError(f'unsupported or incomplete cohort: {folder}')
        protocol_sha = freeze(protocol_path, cohort['protocol_sha256'])
        freeze(cohort_path)
        global_paths.extend([str(cohort_path.resolve()), str(protocol_path.resolve())])
        for index, entry in enumerate(cohort['records']):
            if entry['index'] != index:
                raise ValueError('cohort indices omitted, duplicated or reordered')
            data_path = folder/'data'/f'seq_{index:06d}.json'
            data_sha = freeze(data_path, entry['file_sha256'])
            record = json.loads(data_path.read_text())
            if record.get('source_id') != entry['source_id'] or len(record.get('tokens', [])) != 1025:
                raise ValueError('source differs from frozen cohort')
            model_files, case_paths = {}, [str(data_path.resolve())]
            for model in ('d64', 'd64_seed1'):
                model_path = folder/'models'/model/data_path.name
                model_sha = freeze(model_path)
                model_record = json.loads(model_path.read_text())
                if model_record.get('source_id') != entry['source_id']:
                    raise ValueError('model/source identity differs')
                array_path = model_path.with_suffix('.npz')
                freeze(array_path, model_record['arrays_sha256'])
                model_files[model] = dict(path=str(model_path.resolve()), sha256=model_sha)
                case_paths.extend([str(model_path.resolve()), str(array_path.resolve())])
            cases[f'a1_L{depth}/{index}'] = dict(
                files=case_paths, protocol_sha256=protocol_sha, source_id=entry['source_id'],
                source_file_sha256=data_sha, tokens_sha256=record['tokens_sha256'],
                prefix_sha256={str(t): hashlib.sha256(struct.pack('<'+'h'*t, *record['tokens'][:t])).hexdigest() for t in TIMES},
                model_files=model_files)
    environment = {name: os.environ.get(name) for name in ('LSA_STORE', 'PMM_SRC')}
    for name, filenames in (('LSA_STORE', ['manifest.json', 'anchors.json']),
                            ('PMM_SRC', ['product_model_with_memory/'+f for f in ('universal_tables.py', 'production_coding.py', 'mellin.py', 'kernel.py')])):
        if environment[name]:
            environment[name] = str(Path(environment[name]).expanduser().resolve())
            for filename in filenames:
                path = Path(environment[name])/filename
                freeze(path)
                global_paths.append(str(path.resolve()))
    plan = dict(schema=SCHEMA, source_root=str(source_root), references=str(references),
                n_cases_per_setting=N_CASES, depths=list(DEPTHS), times=list(TIMES), orders=list(ORDERS),
                jobs=all_jobs(), cases=cases, assets=assets, global_paths=global_paths,
                worker=dict(python=os.path.abspath(sys.executable), script=str((ROOT/'code/scripts/shared_base_curve_reference.py').resolve()),
                            proposal='iid', steps=64, burn=args.burn, initial_draws=args.initial_draws,
                            max_draws=args.max_draws, seed=args.seed, environment=environment),
                interpretation='Fixed full plan. Dispatch selections are temporary; no unresolved case may be omitted from a cohort audit.')
    return dict(plan_id=fingerprint(plan), **plan)


def verify_assets(plan, paths=None):
    for path in plan['assets'] if paths is None else paths:
        if digest(path) != plan['assets'][path]:
            raise ValueError(f'frozen campaign input or code changed: {path}')


def freeze_or_resume(args):
    plan = build_plan(args)
    path = args.campaign_out/'plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('campaign plan changed; refusing drift in cohort, code, sources or numerical protocol')
    else:
        atomic_json(path, plan)
    return plan


def job_paths(plan, job):
    folder = Path(plan['references'])/f"a1_L{job['depth']}"/f"alpha{job['order']}"
    stem = f"seq_{job['index']:06d}_t{job['prefix']:04d}"
    return folder/(stem+'.json'), folder/(stem+'.lock')


def report_state(plan, job):
    path, _ = job_paths(plan, job)
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    identity = report['identity']
    case = plan['cases'][f"a1_L{job['depth']}/{job['index']}"]
    expected = {key: case[key] for key in ('protocol_sha256', 'source_id', 'source_file_sha256', 'tokens_sha256', 'model_files')}
    expected.update(index=job['index'], prefix=job['prefix'], prefix_tokens_sha256=case['prefix_sha256'][str(job['prefix'])])
    scoring = {Path(p).name: plan['assets'][p] for p in plan['global_paths']
               if Path(p).name in ('shared_base_curve_reference.py', 'shared_base_conditional_cohort.py', 'shared_base_precision_cohort.py')}
    proposal = identity.get('proposal', {})
    spec = proposal.get('proposal_spec', {})
    if (report.get('schema') != 'shared-base-curve-reference-v1'
            or report.get('reference_id') != fingerprint(identity)
            or identity.get('inputs') != expected or identity.get('scoring_code') != scoring
            or identity.get('master_sampling_seed') != plan['worker']['seed']
            or identity.get('burn') != plan['worker']['burn']
            or spec.get('surrogate_steps') != 64
            or proposal.get('sampler_class') != 'shared_base_surrogate.SharedBaseSurrogateMCMC'
            or report.get('time') != job['prefix'] or report.get('order') != job['order']
            or report.get('index') != job['index'] or report.get('setting') != f"a1_L{job['depth']}"):
        raise ValueError(f'existing reference identity differs from campaign: {path}')
    for name, expected_sha in proposal.get('code_sha256', {}).items():
        file_path = str((ROOT/'code/transformer'/name).resolve())
        if plan['assets'].get(file_path) != expected_sha:
            raise ValueError('existing reference proposal implementation differs')
    if report.get('ready_for_cohort_check'):
        if digest(report['checkpoint']) != report['checkpoint_sha256']:
            raise ValueError('ready reference checkpoint changed')
        moments = path.with_name(path.stem+'_moments.npz')
        if digest(moments) != report['moments_sha256']:
            raise ValueError('ready reference moments changed')
        return 'ready'
    if report.get('history') and report['history'][-1]['draws'] >= plan['worker']['max_draws']:
        return 'unresolved'
    return None


def lock_busy(path):
    if not path.exists():
        return False
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


@contextmanager
def campaign_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('campaign dispatcher is already running') from error
        yield


def command(plan, job):
    worker = plan['worker']
    args = [worker['python'], worker['script'], '--input', str(Path(plan['source_root'])/f"a1_L{job['depth']}"),
            '--out', plan['references'], '--index', str(job['index']), '--prefix', str(job['prefix']),
            '--alpha-order', str(job['order']), '--proposal', 'iid', '--steps', '64']
    for name in ('burn', 'initial_draws', 'max_draws', 'seed'):
        args.extend(['--'+name.replace('_', '-'), str(worker[name])])
    return args


def cost_report(state, eligible):
    counts = Counter(v.get('state', 'pending') for v in state['jobs'].values())
    attempts = [attempt for v in state['jobs'].values() for attempt in v.get('attempts', [])]
    return dict(full_plan_jobs=len(state['jobs']), eligible_jobs=len(eligible), states=dict(counts),
                finished_worker_seconds=sum(a.get('wall_seconds', 0.) for a in attempts),
                worker_attempts=len(attempts), stop_reason=state.get('stop_reason'),
                interpretation='Ready cases still require the independent complete-cohort audit; unresolved cases remain declared.')


def dispatch(args, plan):
    eligible = selected_jobs(plan, args.times, args.depths, args.indices)
    status_path = args.campaign_out/'status.json'
    state = json.loads(status_path.read_text()) if status_path.exists() else dict(
        schema=SCHEMA, plan_id=plan['plan_id'], jobs={j['key']: dict(state='pending', attempts=[]) for j in plan['jobs']})
    if state.get('plan_id') != plan['plan_id'] or set(state['jobs']) != {j['key'] for j in plan['jobs']}:
        raise ValueError('campaign status does not match its complete frozen plan')
    started, active, stop = time.monotonic(), {}, []
    state.update(last_started=utc(), stop_reason=None,
                 dispatch_selection=dict(times=args.times, depths=args.depths, indices=args.indices),
                 limits=dict(max_workers=args.max_workers, max_hours=args.max_hours, disk_reserve_gb=args.disk_reserve_gb))
    # A previous ready status is not a replacement for checking its artifacts.
    for job in eligible:
        item = state['jobs'][job['key']]
        if item['state'] == 'ready':
            if report_state(plan, job) != 'ready':
                raise ValueError('a previously ready reference disappeared or changed')
    old_handlers = {}
    def request_stop(signum, frame):
        stop.append('signal; active bounded workers allowed to finish')
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_stop)
    def save():
        state['updated'] = utc()
        state['cost'] = cost_report(state, eligible)
        atomic_json(status_path, state)
    try:
        while True:
            for key, (process, stream, job, began) in list(active.items()):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                stream.close()
                item = state['jobs'][key]
                item['attempts'][-1].update(exit_code=exit_code, finished=utc(), wall_seconds=time.monotonic()-began)
                _, lock_path = job_paths(plan, job)
                outcome = report_state(plan, job) if not lock_busy(lock_path) else None
                item['state'] = ('ready' if outcome == 'ready' else 'unresolved' if exit_code == 2 or outcome == 'unresolved'
                                 else 'external_busy' if lock_busy(lock_path) else 'failed')
                del active[key]
            if time.monotonic()-started >= args.max_hours*3600:
                stop.append('time budget reached')
            if shutil.disk_usage(plan['references']).free < args.disk_reserve_gb*1e9:
                stop.append('disk reserve reached')
            if stop:
                state['stop_reason'] = stop[0]
                save()
                if not active:
                    break
                time.sleep(args.poll_seconds)
                continue
            pending = []
            for job in eligible:
                key, item = job['key'], state['jobs'][job['key']]
                if key in active or item['state'] in TERMINAL:
                    continue
                _, lock_path = job_paths(plan, job)
                if lock_busy(lock_path):
                    item['state'] = 'external_busy'
                else:
                    existing = report_state(plan, job)
                    item['state'] = existing or 'pending'
                if item['state'] not in TERMINAL:
                    pending.append(job)
            if not pending and not active:
                state['stop_reason'] = 'all selected jobs ready or retained as unresolved/failed'
                save()
                break
            stage = min([j['prefix'] for j in pending]+[v[2]['prefix'] for v in active.values()])
            owned_locks = {job_paths(plan, v[2])[1] for v in active.values()}
            external_running = sum(lock_busy(p) for p in Path(plan['references']).glob('a1_L*/alpha*/seq*_t*.lock')
                                   if p not in owned_locks)
            state['external_active_workers'] = external_running
            for job in pending:
                if len(active)+external_running >= args.max_workers:
                    break
                if job['prefix'] != stage or state['jobs'][job['key']]['state'] == 'external_busy':
                    continue
                case = plan['cases'][f"a1_L{job['depth']}/{job['index']}"]
                verify_assets(plan, plan['global_paths']+case['files'])
                if time.monotonic()-started >= args.max_hours*3600 or shutil.disk_usage(plan['references']).free < args.disk_reserve_gb*1e9:
                    break
                item = state['jobs'][job['key']]
                attempt = len(item['attempts'])+1
                log_path = args.campaign_out/'logs'/(job['key'].replace('/', '_')+f'_attempt{attempt:03d}.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open('xb')
                env = os.environ.copy()
                env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                env.update({k: v for k, v in plan['worker']['environment'].items() if v is not None})
                process = subprocess.Popen(command(plan, job), stdout=stream, stderr=subprocess.STDOUT,
                                           cwd=ROOT, env=env, start_new_session=True)
                item.update(state='running', pid=process.pid)
                item['attempts'].append(dict(started=utc(), log=str(log_path.resolve()), pid=process.pid))
                active[job['key']] = process, stream, job, time.monotonic()
                save()
            save()
            time.sleep(args.poll_seconds)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return state['cost']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'output/shared_base_conditional_curves_2026-09-15')
    p.add_argument('--references', type=Path, default=ROOT/'output/shared_base_curve_references_2026-09-15')
    p.add_argument('--campaign-out', type=Path, default=ROOT/'output/shared_base_curve_campaign_2026-09-15')
    p.add_argument('--times', nargs='+', type=int, default=list(TIMES))
    p.add_argument('--depths', nargs='+', type=int, default=list(DEPTHS))
    p.add_argument('--indices', nargs='+', type=int, default=list(range(N_CASES)))
    p.add_argument('--max-workers', type=int, default=2)
    p.add_argument('--max-hours', type=float, default=12.)
    p.add_argument('--disk-reserve-gb', type=float, default=50.)
    p.add_argument('--poll-seconds', type=float, default=5.)
    p.add_argument('--burn', type=int, default=16384)
    p.add_argument('--initial-draws', type=int, default=4096)
    p.add_argument('--max-draws', type=int, default=65536)
    p.add_argument('--seed', type=int, default=20261006)
    p.add_argument('--plan-only', action='store_true')
    return p


def main():
    args = parser().parse_args()
    selected_jobs(dict(jobs=all_jobs()), args.times, args.depths, args.indices)
    if (not all(math.isfinite(v) for v in (args.max_hours, args.disk_reserve_gb, args.poll_seconds))
            or args.max_workers < 1 or args.max_hours <= 0 or args.disk_reserve_gb < 0 or args.poll_seconds <= 0
            or args.burn < 0 or args.initial_draws < 16 or args.max_draws < args.initial_draws or args.seed < 0):
        raise ValueError('invalid campaign limits or numerical settings')
    args.campaign_out.mkdir(parents=True, exist_ok=True)
    args.references.mkdir(parents=True, exist_ok=True)
    with campaign_lock(args.campaign_out/'dispatcher.lock'):
        plan = freeze_or_resume(args)
        eligible = selected_jobs(plan, args.times, args.depths, args.indices)
        if args.plan_only:
            print(json.dumps(dict(plan_id=plan['plan_id'], full_plan_jobs=len(plan['jobs']), eligible_jobs=len(eligible), launched=0)), flush=True)
        else:
            if not all(plan['worker']['environment'].values()):
                raise ValueError('LSA_STORE and PMM_SRC must identify the sealed production store before dispatch')
            print(json.dumps(dispatch(args, plan)), flush=True)


if __name__ == '__main__':
    main()
