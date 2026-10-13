"""Launch and audit a fixed, independent shared-base posterior pilot on Jed.

This is conditional prediction, not the three-family absolute-likelihood SMC.
No numerical worker, gate, source record or transformer output is modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import signal
import socket
import subprocess
import sys
import time

from prepare_shared_base_posterior_cluster import (
    ROOT, PLAN, DEPTHS, TIMES, ORDERS, INDICES, WORKER, require, sha, fingerprint,
    atomic_json, immutable_json, read_plan, verify_inputs)

DEFAULT_OUT = ROOT/'output/shared_base_posterior_cluster_2026-09-16'


def utc():
    return datetime.now(timezone.utc).isoformat()


def configure_dependency(root):
    root = Path(root).resolve()
    os.environ['LSA_STORE'] = str(root/'tables/anchors_prod')
    os.environ['PMM_SRC'] = str(root/'src')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'


def verify_dependencies(root, plan):
    # Keep the submit-side launcher standard-library only. The actual node
    # checks imports and the compiled kernel after activating the given venv.
    root = Path(root)
    manifest = json.loads((ROOT/plan['dependency_manifest']).read_bytes())
    for name, expected in manifest['files'].items():
        path = root/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['size_bytes']
                and sha(path) == expected['sha256'], f'Numerical dependency changed: {name}')


def runtime_setup(root):
    """Verify the actual node's sealed read-only store and compiled kernel."""
    configure_dependency(root)
    from calibrate_shared_base_evidence import numerical_environment
    numerical_environment()  # Reject perturbations before importing the backend.
    sys.path.insert(0, str(ROOT/'code/transformer'))
    import numpy as np
    import scipy
    from ranked_lsa import RankedLSARowEvidence
    kernel = RankedLSARowEvidence(64, [1.], term_cache_size=16)
    backend = kernel._get_backend()
    compiled = sys.modules['product_model_with_memory.kernel']
    require(backend.read_only and backend.sealed and compiled.available, 'Sealed read-only store and compiled kernel required')
    require(Path(compiled.__file__).resolve().parent == Path(root).resolve()/'src/product_model_with_memory',
            'Wrong numerical package imported')
    effective = numerical_environment()
    effective.pop('PMM_UNIVERSAL_TABLES')  # Deployment path is bound separately.
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        effective_numerical_environment=effective,
        longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant+1),
        compiled_kernel_available=True, store_sealed=True, store_read_only=True,
        thread_environment={name: os.environ[name] for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})


def build_launch(args):
    plan, plan_hash = read_plan(args.plan)
    selected = list(range(32)) if not args.only_task else sorted(set(args.only_task))
    require(selected and all(0 <= i < 32 for i in selected), 'Task selection outside the fixed pilot')
    require(1 <= args.concurrency <= 32, 'Concurrency must be 1..32')
    require(not args.preflight or not args.only_task, 'Preflight cannot select posterior tasks')
    paths = dict(repo_root=str(ROOT), plan=str(args.plan.expanduser().resolve()),
                 inputs=str(args.inputs.expanduser().resolve()), dependency_root=str(args.dependency_root.expanduser().resolve()),
                 output=str(args.out.expanduser().resolve()), venv=str(args.venv.expanduser().resolve()))
    require(Path(paths['output']) != Path(paths['inputs'])
            and Path(paths['inputs']) not in Path(paths['output']).parents
            and Path(paths['output']) not in Path(paths['inputs']).parents,
            'Output and frozen inputs must be separate')
    campaign = dict(schema='shared-base-posterior-deployment-v1', plan_sha256=plan_hash, **paths)
    launch = dict(schema='shared-base-posterior-launch-v1', campaign=campaign,
                  mode='preflight' if args.preflight else 'pilot', selected_tasks=selected,
                  concurrency=1 if args.preflight else args.concurrency)
    launch['launch_id'] = fingerprint(launch)
    out = Path(paths['output'])
    manifest = out/'launches'/f"{launch['launch_id'][:16]}.json"
    command = ['sbatch', '--parsable', f'--chdir={ROOT}', '--job-name=sb-posterior'+('-preflight' if args.preflight else ''),
               '--time=00:15:00' if args.preflight else '--time=08:00:00',
               f"--output={out}/logs/{'preflight' if args.preflight else 'posterior'}-%A_%a.out"]
    if not args.preflight:
        selection = '0-31' if selected == list(range(32)) else ','.join(map(str, selected))
        command.append(f'--array={selection}%{args.concurrency}')
    command.extend([str(ROOT/'code/transformer/scitas/shared_base_posterior.sbatch'), str(manifest), paths['venv']])
    return plan, launch, manifest, command


def submit(args):
    plan, launch, manifest, command = build_launch(args)
    print(shlex.join(command), flush=True)
    if not args.submit:
        print('Dry run only; no files written or jobs submitted.', flush=True)
        return
    paths = launch['campaign']
    verify_inputs(paths['inputs'], plan)
    verify_dependencies(paths['dependency_root'], plan)
    require((Path(paths['venv'])/'bin/python').is_file(), 'Prepared Python environment is missing')
    out = Path(paths['output'])
    immutable_json(out/'campaign.json', paths)
    immutable_json(manifest, launch)
    (out/'logs').mkdir(exist_ok=True)
    completed = subprocess.run(command, check=True, text=True, capture_output=True, cwd=ROOT)
    print(completed.stdout, end='', flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end='', flush=True)
    atomic_json(out/'submissions'/f"{time.time_ns()}.json",
                dict(launch_id=launch['launch_id'], submitted_utc=utc(), command=command,
                     scheduler_response=completed.stdout.strip()))


def read_launch(path):
    launch = json.loads(Path(path).read_bytes())
    require(launch.get('schema') == 'shared-base-posterior-launch-v1'
            and launch.get('launch_id') == fingerprint({k: v for k, v in launch.items() if k != 'launch_id'}),
            'Launch checksum/schema mismatch')
    campaign = launch['campaign']
    require(Path(campaign['repo_root']).resolve() == ROOT, 'Launch belongs to another checkout')
    plan, digest = read_plan(campaign['plan'])
    require(digest == campaign['plan_sha256'], 'Plan changed after submission')
    require(json.loads((Path(campaign['output'])/'campaign.json').read_bytes()) == campaign, 'Deployment changed')
    verify_inputs(campaign['inputs'], plan)
    verify_dependencies(campaign['dependency_root'], plan)
    configure_dependency(campaign['dependency_root'])
    return launch, plan


def task_paths(campaign, task):
    out = Path(campaign['output'])
    stem = f"seq_{task['index']:06d}_t{task['time']:04d}"
    reference = out/'references'/task['setting']/f"alpha{task['order']}"/f'{stem}.json'
    return dict(reference=reference, checkpoint=reference.with_name(stem+'_chains.npz'),
                moments=reference.with_name(stem+'_moments.npz'), audit=out/'audits'/f"{task['id']}.json",
                receipt=out/'tasks'/f"{task['id']}.json")


def worker_command(campaign, task):
    command = [sys.executable, str(ROOT/'code/scripts/shared_base_curve_reference.py'),
               '--input', str(Path(campaign['inputs'])/task['setting']),
               '--out', str(Path(campaign['output'])/'references'),
               '--index', str(task['index']), '--prefix', str(task['time']), '--alpha-order', str(task['order'])]
    for name in ('proposal', 'steps', 'burn', 'initial_draws', 'max_draws', 'seed'):
        command.extend(['--'+name.replace('_', '-'), str(WORKER[name])])
    return command


def process_peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak*1024)


def run_child(command, log, seconds, progress=None):
    """Bound wall time without deleting the worker's atomic checkpoints."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic()-started >= seconds:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    return 75
                if progress:
                    progress(child.pid, time.monotonic()-started)
                time.sleep(.5)
            return child.returncode
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            raise


def progress_record(log, paths):
    latest = None
    if Path(log).exists():
        with Path(log).open('rb') as stream:
            stream.seek(max(0, Path(log).stat().st_size-65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict):
                    latest = candidate
                    break
            except (ValueError, UnicodeDecodeError):
                continue
    checkpoint = paths['checkpoint']
    return dict(last_worker_event=latest,
                checkpoint_exists=checkpoint.exists(),
                checkpoint_bytes=checkpoint.stat().st_size if checkpoint.exists() else None)


def audit_one(launch_path, task_index):
    launch, plan = read_launch(launch_path)
    require(launch['mode'] == 'pilot' and task_index in launch['selected_tasks'], 'Undeclared audit task')
    campaign, task = launch['campaign'], plan['tasks'][task_index]
    paths = task_paths(campaign, task)
    reference = json.loads(paths['reference'].read_bytes())
    diagnostics = reference['estimate']['diagnostics']
    require(reference['identity']['master_sampling_seed'] == WORKER['seed']
            and reference['identity']['burn'] == WORKER['burn']
            and diagnostics['chains'] == WORKER['chains'] and diagnostics['thin'] == WORKER['thin']
            and WORKER['initial_draws'] <= diagnostics['draws'] <= WORKER['max_draws']
            and reference['identity']['proposal']['sampler_class'] == 'shared_base_surrogate.SharedBaseSurrogateMCMC'
            and reference['identity']['proposal']['proposal_spec']['surrogate_steps'] == WORKER['steps'],
            'Reference sampling budget/proposal differs from fixed pilot')
    from shared_base_curve_audit import audit_reference
    from validate_shared_base_mcmc import json_safe
    result = audit_reference(Path(campaign['inputs'])/task['setting'], Path(campaign['output'])/'references',
                             task['index'], task['time'], task['order'])
    # Hash the complete triplet audited on this host; these paths are retained
    # so the unchanged raw auditor can be replayed in this deployment.
    raw_hashes = {str(paths[key]): sha(paths[key]) for key in ('reference', 'checkpoint', 'moments') if paths[key].is_file()}
    atomic_json(paths['audit'], json_safe(dict(schema='shared-base-posterior-task-audit-v1',
        plan_sha256=campaign['plan_sha256'], task=task, audited_utc=utc(), result=result, raw_sha256=raw_hashes,
        audit_code_sha256=sha(ROOT/'code/scripts/shared_base_curve_audit.py'))))
    return 0 if result['status'] in ('ready', 'needs_more_draws') else 2


def execute(launch_path, task_index):
    launch, plan = read_launch(launch_path)
    campaign = launch['campaign']
    out = Path(campaign['output'])
    runtime = runtime_setup(campaign['dependency_root'])
    if launch['mode'] == 'preflight':
        report = dict(schema='shared-base-posterior-preflight-v1', plan_sha256=campaign['plan_sha256'],
                      passed=True, runtime=runtime, hostname=socket.gethostname(), checked_utc=utc(),
                      scientific_reference_validated=False)
        atomic_json(out/'preflight.json', report)
        print(json.dumps(report), flush=True)
        return 0
    require(task_index in launch['selected_tasks'], 'Task not selected in this launch')
    task = plan['tasks'][task_index]
    paths = task_paths(campaign, task)
    paths['receipt'].parent.mkdir(parents=True, exist_ok=True)
    with paths['receipt'].with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads(paths['receipt'].read_bytes()) if paths['receipt'].exists() else None
        identity = dict(plan_sha256=campaign['plan_sha256'], task=task, deployment_sha256=fingerprint(campaign))
        if old:
            require(old['identity'] == identity and old['runtime'] == runtime, 'Cannot resume changed target/runtime')
            require(old['status'] not in ('ready', 'unresolved'), 'Completed bounded task is retained; no automatic additional run')
            require(not old.get('progress', {}).get('checkpoint_exists') or paths['checkpoint'].is_file(),
                    'Saved checkpoint disappeared; refusing a silent fresh restart')
        require(not paths['reference'].exists() or paths['checkpoint'].is_file(),
                'Reference report exists without its saved chains; refusing a fresh restart')
        run_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{time.time_ns()}"
        attempt_path = out/'attempts'/task['id']/f'{run_id}.json'
        log = out/'worker_logs'/task['id']/f'{run_id}.log'
        report = dict(schema='shared-base-posterior-task-v1', identity=identity, runtime=runtime,
                      status='running', started_utc=utc(), hostname=socket.gethostname(),
                      platform=platform.platform(), task_index=task_index, launch_id=launch['launch_id'],
                      worker_command=worker_command(campaign, task), worker_log=str(log),
                      attempt_path=str(attempt_path), prior_attempts=[] if old is None else old['prior_attempts']+[old['attempt_path']],
                      cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                      sampling_complete=False, scientific_reference_validated=False)
        started = time.monotonic()
        last_progress = [-1e9]
        def save(pid=None, elapsed=0):
            if elapsed-last_progress[0] < 30 and pid is not None:
                return
            last_progress[0] = elapsed
            report.update(elapsed_seconds=time.monotonic()-started, progress=progress_record(log, paths))
            if pid is not None:
                report['worker_pid'] = pid
            atomic_json(attempt_path, report)
            atomic_json(paths['receipt'], report)
            print(json.dumps(dict(task=task['id'], status=report['status'], elapsed=report['elapsed_seconds'],
                                 progress=report['progress'])), flush=True)
        save()
        try:
            worker_rc = run_child(report['worker_command'], log, plan['resources']['worker_seconds'], save)
            report['worker_exit_code'] = worker_rc
            report['worker_peak_rss_bytes'] = process_peak_bytes()
            report['sampling_complete'] = worker_rc in (0, 2)
            if worker_rc in (0, 2):
                audit_log = out/'worker_logs'/task['id']/f'{run_id}-audit.log'
                audit_rc = run_child([sys.executable, str(Path(__file__).resolve()), 'audit-task',
                                      '--launch', str(launch_path), '--task-index', str(task_index)], audit_log, 600)
                report['audit_exit_code'] = audit_rc
                if paths['audit'].is_file():
                    audit = json.loads(paths['audit'].read_bytes())
                    report['audit_sha256'] = sha(paths['audit'])
                    state = audit['result']['status']
                    report['status'] = 'ready' if audit_rc == 0 and state == 'ready' else 'unresolved' if audit_rc == 0 and state == 'needs_more_draws' else 'audit_failed'
                else:
                    report['status'] = 'audit_timeout' if audit_rc == 75 else 'audit_failed'
            else:
                report['status'] = 'timeout' if worker_rc == 75 else 'worker_failed'
        except Exception as error:
            report.update(status='failed', error=str(error))
        report.update(finished_utc=utc(), peak_child_rss_bytes=process_peak_bytes())
        save()
        return 0 if report['status'] == 'ready' else 2 if report['status'] == 'unresolved' else 75 if 'timeout' in report['status'] else 1


def summarize(plan_path, output):
    plan, plan_hash = read_plan(plan_path)
    output = Path(output)
    campaign = json.loads((output/'campaign.json').read_bytes())
    require(campaign['plan_sha256'] == plan_hash and Path(campaign['repo_root']).resolve() == ROOT
            and Path(campaign['output']).resolve() == output.resolve(), 'Wrong deployment for summary')
    verify_inputs(campaign['inputs'], plan)
    verify_dependencies(campaign['dependency_root'], plan)
    configure_dependency(campaign['dependency_root'])
    from shared_base_curve_audit import aggregate_time
    rows, audited, runtimes = [], {}, []
    for task in plan['tasks']:
        paths = task_paths(campaign, task)
        row = dict(task=task, status='missing', audit=None)
        if paths['receipt'].is_file():
            receipt = json.loads(paths['receipt'].read_bytes())
            require(receipt['identity'] == dict(plan_sha256=plan_hash, task=task, deployment_sha256=fingerprint(campaign)), 'Task receipt identity mismatch')
            runtimes.append(receipt['runtime'])
            row.update(status=receipt['status'], elapsed_seconds=receipt['elapsed_seconds'],
                       peak_child_rss_bytes=receipt.get('peak_child_rss_bytes'), receipt_sha256=sha(paths['receipt']))
            if receipt['status'] in ('ready', 'unresolved'):
                require(paths['audit'].is_file() and sha(paths['audit']) == receipt['audit_sha256'], 'Missing or altered task audit')
                audit = json.loads(paths['audit'].read_bytes())
                require(audit['task'] == task and audit['plan_sha256'] == plan_hash
                        and audit['audit_code_sha256'] == plan['code_sha256']['code/scripts/shared_base_curve_audit.py'],
                        'Audit task/code mismatch')
                require(audit['raw_sha256'] == {str(paths[key]): sha(paths[key]) for key in ('reference', 'checkpoint', 'moments')}, 'Raw reference changed after audit')
                require((audit['result']['index'], audit['result']['time'], audit['result']['order']) ==
                        (task['index'], task['time'], task['order']), 'Audit case mismatch')
                require(audit['result']['status'] == ('ready' if receipt['status'] == 'ready' else 'needs_more_draws'), 'Audit status mismatch')
                audited[task['id']] = audit['result']
                row['audit'] = audit['result']
        rows.append(row)
    require(not runtimes or all(runtime == runtimes[0] for runtime in runtimes), 'Posterior tasks used different numerical runtimes')
    groups = []
    for depth in DEPTHS:
        for t in TIMES:
            tasks = [task for task in plan['tasks'] if task['depth'] == depth and task['time'] == t]
            cases = [audited[task['id']] for task in tasks if task['id'] in audited]
            aggregate = aggregate_time(cases, list(INDICES), list(ORDERS))
            groups.append(dict(depth=depth, time=t, numerical_audit=aggregate,
                               interpretation='Two fixed sequences: computational pilot only, not population accuracy evidence.'))
    result = dict(schema='shared-base-posterior-pilot-summary-v1', plan_sha256=plan_hash,
                  complete=all(row['status'] in ('ready', 'unresolved') for row in rows),
                  all_numerical_checks_passed=all(group['numerical_audit']['passed'] for group in groups),
                  expected_references=32, expected_sources=8, rows=rows, groups=groups,
                  production_approved=False, full_population_claim=False,
                  interpretation='Failures and missing tasks remain explicit; no partial-population mean or automatic expansion.',
                  summarized_utc=utc())
    atomic_json(output/'summary.json', result)
    print(json.dumps(dict(complete=result['complete'], passed=result['all_numerical_checks_passed'],
                          states={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})})))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    launch = subs.add_parser('launch')
    launch.add_argument('--plan', type=Path, default=PLAN)
    launch.add_argument('--inputs', type=Path, default=Path.home()/'Projects/shared_base_posterior_inputs_2026-09-16')
    launch.add_argument('--dependency-root', type=Path, default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
    launch.add_argument('--out', type=Path, default=DEFAULT_OUT)
    launch.add_argument('--venv', type=Path, default=Path(os.environ.get('SHARED_BASE_VENV', str(Path.home()/'venv-tf'))))
    launch.add_argument('--concurrency', type=int, default=32)
    launch.add_argument('--only-task', type=int, nargs='+')
    launch.add_argument('--preflight', action='store_true')
    action = launch.add_mutually_exclusive_group()
    action.add_argument('--submit', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    for name in ('run', 'audit-task'):
        cmd = subs.add_parser(name)
        cmd.add_argument('--launch', type=Path, required=True)
        cmd.add_argument('--task-index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    summary = subs.add_parser('summary')
    summary.add_argument('--plan', type=Path, default=PLAN)
    summary.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == 'launch':
        submit(args)
    elif args.command == 'run':
        raise SystemExit(execute(args.launch, args.task_index))
    elif args.command == 'audit-task':
        raise SystemExit(audit_one(args.launch, args.task_index))
    else:
        summarize(args.plan, args.out)


if __name__ == '__main__':
    main()
