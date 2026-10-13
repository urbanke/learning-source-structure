"""Isolated, bounded continuation of the 246 frozen unseen-context references.

Every task resumes a verified local copy with the unchanged numerical worker.
Completion means reaching the fixed target and passing an independent raw audit.
Only the two complete selected groups can enter the original cohort validator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'unseen-context-refinement-v1'
RAW_NAMES = ('reference', 'checkpoint', 'moments')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(read(path) == value, 'Immutable receipt differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def rooted(path):
    return (ROOT / path).resolve()


def key(spec):
    t = spec['task']
    return f"{t['setting']}/seq_{t['index']:06d}/t{t['time']:04d}/alpha{t['order']}"


def slot_bytes(draws):
    # The established local finish probe used this allowance, including margin.
    return 3 * 1024**3 // 2 + 6 * 4 * draws * 64 * 8


def safe_paths(out, paths):
    base = Path(out).absolute()
    require(not base.is_symlink(), 'Output directory is a symlink')
    for path in paths:
        path = Path(path).absolute()
        require(base.resolve() in path.resolve().parents, 'Output artifact escapes its isolated tree')
        while path != base:
            require(not path.is_symlink(), 'Symlink inside output tree: ' + str(path))
            require(path != path.parent, 'Artifact is outside output tree')
            path = path.parent


def paths(out, spec):
    t = spec['task']; out = Path(out)
    stem = f"seq_{t['index']:06d}_t{t['time']:04d}"
    report = out / 'references' / t['setting'] / f"alpha{t['order']}" / (stem + '.json')
    directory = out / 'tasks' / t['id']
    result = dict(reference=report, checkpoint=report.with_name(stem + '_chains.npz'),
        moments=report.with_name(stem + '_moments.npz'), sample_lock=report.with_suffix('.lock'),
        task_lock=directory/'task.lock', prepared=directory/'prepared.json',
        parent_report=directory/'parent_reference.json', parent_receipt=directory/'parent_audit.json',
        audit=directory/'audit.json', receipt=directory/'receipt.json',
        attempts=directory/'attempts.json', worker_log=directory/'worker.log', audit_log=directory/'audit.log')
    safe_paths(out, result.values())
    return result


@contextmanager
def lock(path, *, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def busy(path):
    path = Path(path)
    if not path.exists():
        return False
    with path.open('rb') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def context(plan_path, out):
    from prepare_unseen_context_refinement import read_plan
    plan, digest = read_plan(plan_path)
    out = Path(out).absolute()
    for spec in plan['tasks']:
        protected = [rooted(spec['source_folder'])]
        protected += [rooted(v['path']).parent for v in spec['parent_files'].values()]
        for original in protected:
            require(out.resolve() != original and out.resolve() not in original.parents
                    and original not in out.resolve().parents,
                    'Continuation output overlaps frozen source or reference files')
    receipt = dict(schema=SCHEMA, plan=str(Path(plan_path).resolve()), plan_sha256=digest,
                   output=str(out.resolve()))
    safe_paths(out, [out/'context.json'])
    atomic_json(out/'context.json', receipt, immutable=True)
    return plan, receipt


def environment(plan):
    env = dict(os.environ)
    env.update({k: v for k, v in plan['environment'].items() if v is not None})
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    for name in ('PMM_PHI_BIAS', 'PMM_PHI_WAVE'):
        require(float(env.get(name, '0')) == 0, 'Numerical perturbation is forbidden')
    return env


def check_pairing(spec, report, source, source_identity):
    t = spec['task']; tokens = source['tokens'][:t['time']]
    require(len(tokens) == 64 and tokens[-1] not in tokens[:-1],
            'History is outside the frozen unseen-context event')
    require(source['source_id'] == spec['source_id'] == source_identity['source_id']
            and report['identity']['inputs'] == source_identity
            and report['identity'] == spec['parent_identity'], 'Source or predictor pairing changed')
    require((report['setting'], report['index'], report['time'], report['order']) ==
            (t['setting'], t['index'], 64, t['order'])
            and report['reference_id'] == spec['parent_reference_id'] == fingerprint(report['identity']),
            'Reference identity changed')
    prior = report['identity']['target']
    require(prior['d'] == 64 and prior['prefix_length'] == 64
            and prior['depths'] == list(range(9))
            and all(abs(p - 1/9) < 1e-15 for p in prior['depth_prior'])
            and len(prior['alpha_nodes']) == t['order']
            and min(prior['alpha_nodes']) > .5 and max(prior['alpha_nodes']) < 1.5,
            'Reference no longer retains the full training prior')


def verify_inputs(plan, spec, *, raw=False):
    from prepare_unseen_context_refinement import verify_task_inputs
    verify_task_inputs(plan, spec, root=ROOT, verify_raw=raw)
    from shared_base_conditional_cohort import inputs, load_source
    report = read(rooted(spec['parent_files']['reference']['path']))
    folder = rooted(spec['source_folder']); t = spec['task']
    _, _, _, identity = inputs(folder, t['index'], t['time'])
    check_pairing(spec, report, load_source(folder, t['index']), identity)
    return report


def checkpoint_metadata(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as stored:
        serialized = str(stored['metadata'].item())
        require(hashlib.sha256(serialized.encode()).hexdigest() == str(stored['metadata_sha256'].item()),
                'Checkpoint metadata checksum failed')
        return json.loads(serialized)


def checkpoint_counts(path):
    """Read shapes without decompressing the predictive arrays."""
    import numpy as np
    result = []
    with zipfile.ZipFile(path) as archive:
        for chain in range(4):
            with archive.open(f'alpha_indices_{chain}.npy') as stream:
                version = np.lib.format.read_magic(stream)
                reader = { (1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
                require(reader is not None, 'Unsupported checkpoint array format')
                shape, fortran, dtype = reader(stream)
                require(len(shape) == 1 and not fortran and dtype.kind in 'iu',
                        'Malformed retained draw array')
                result.append(shape[0])
    return result


def prefix_hash(path, name, retained):
    """Hash the original prefix with the checkpoint's canonical array hash.

    Stream decompressed bytes in bounded blocks; never materialize a large chain.
    """
    import numpy as np
    with zipfile.ZipFile(path) as archive, archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        require(reader is not None, 'Unsupported checkpoint array format')
        shape, fortran, dtype = reader(stream)
        require(not fortran and not dtype.hasobject and shape and shape[0] >= retained,
                'Continuation lost or reordered saved draws')
        shape = (retained,) + shape[1:]
        h = hashlib.sha256(dtype.str.encode())
        h.update(json.dumps(shape).encode())
        remaining = math.prod(shape) * dtype.itemsize
        while remaining:
            data = stream.read(min(1024 * 1024, remaining))
            require(bool(data), 'Truncated checkpoint array')
            h.update(data); remaining -= len(data)
    return h.hexdigest()


def verify_preserved_samples(parent_path, child_path, spec, *, complete):
    parent, child = checkpoint_metadata(parent_path), checkpoint_metadata(child_path)
    for field in ('format', 'fingerprint', 'config', 'initializations', 'proposal_descriptor'):
        require(parent[field] == child[field], 'Continuation changed checkpoint ' + field)
    counts = checkpoint_counts(child_path)
    require(all(spec['baseline_draws'] <= n <= spec['target_draws'] for n in counts),
            'Retained draw count is outside the fixed continuation budget')
    if complete:
        require(counts == [spec['target_draws']] * 4, 'All four chains must reach the fixed target')
    require(checkpoint_counts(parent_path) == [spec['baseline_draws']] * 4,
            'Original checkpoint draw budget differs')
    for chain in range(4):
        for name in ('predictions', 'depths', 'alpha_indices', 'scalars'):
            name = f'{name}_{chain}'
            require(prefix_hash(child_path, name, spec['baseline_draws']) == parent['array_hashes'][name],
                    'Continuation changed original retained samples: ' + name)
    require(parent['array_hashes']['burn_done'] == child['array_hashes']['burn_done'],
            'Continuation changed completed burn-in')
    return counts


def prepare_case(plan, ctx, spec):
    p = paths(ctx['output'], spec)
    require(not busy(p['sample_lock']), 'A sampling process still owns this copied checkpoint')
    parent = verify_inputs(plan, spec, raw=True)
    expected = dict(schema=SCHEMA, context_sha256=fingerprint(ctx), task=spec['task'],
        parent_files=spec['parent_files'], baseline_draws=spec['baseline_draws'], target_draws=spec['target_draws'])
    if p['prepared'].exists():
        require(read(p['prepared']) == expected and all(p[n].is_file() for n in RAW_NAMES),
                'Prepared copied checkpoint is missing or changed')
        require(sha(p['parent_report']) == spec['parent_files']['reference']['sha256']
                and sha(p['parent_receipt']) == spec['parent_files']['case_receipt']['sha256'],
                'Preserved original report or audit changed')
    else:
        copies = {name: name for name in RAW_NAMES}
        copies.update(parent_report='reference', parent_receipt='case_receipt')
        for dest_name, source_name in copies.items():
            original = spec['parent_files'][source_name]
            src, dst = rooted(original['path']), p[dest_name]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                require(sha(dst) == original['sha256'], 'Unbound local copy differs from frozen input')
            else:
                temporary = dst.with_name(dst.name + '.copying')
                safe_paths(ctx['output'], [temporary])
                shutil.copyfile(src, temporary)
                require(sha(temporary) == original['sha256'], 'Copied original checksum failed')
                os.replace(temporary, dst)
        verify_inputs(plan, spec, raw=True)
        atomic_json(p['prepared'], expected, immutable=True)
    verify_preserved_samples(rooted(spec['parent_files']['checkpoint']['path']), p['checkpoint'], spec,
                             complete=False)
    return p, parent


def validate_continuation(child, parent, spec):
    t = spec['task']
    require(child['identity'] == parent['identity'] == spec['parent_identity']
            and child['reference_id'] == parent['reference_id'] == spec['parent_reference_id']
            and child['history'][:len(parent['history'])] == parent['history'],
            'Continuation changed source, prior, kernel, seed or original history')
    require((child['setting'], child['index'], child['time'], child['order']) ==
            (t['setting'], t['index'], t['time'], t['order']), 'Continuation changed the task coordinates')
    d = child['estimate']['diagnostics']
    require(d['draws'] == spec['target_draws'] and d['chains'] == 4 and d['thin'] == 1
            and d['burn'] == parent['estimate']['diagnostics']['burn']
            and d['seed'] == parent['estimate']['diagnostics']['seed'],
            'Continuation has not reached the unchanged four-chain draw target')


def completed_reference(ctx, spec):
    p = paths(ctx['output'], spec)
    child, parent = read(p['reference']), read(p['parent_report'])
    validate_continuation(child, parent, spec)
    require(child['checkpoint'] == str(p['checkpoint'].resolve())
            and sha(p['checkpoint']) == child['checkpoint_sha256']
            and sha(p['moments']) == child['moments_sha256'], 'Completed report does not bind its raw arrays')
    return child


def child_task(ctx, spec):
    p = paths(ctx['output'], spec); t = spec['task']
    return dict(key=key(spec), **{k: t[k] for k in ('setting','depth','index','time','order')},
        source_folder=str(rooted(spec['source_folder'])), reference_root=str(Path(ctx['output'])/'references'),
        **{name: str(p[name]) for name in RAW_NAMES}, reference_id=spec['parent_reference_id'],
        file_sha256={str(p[name]): sha(p[name]) for name in RAW_NAMES})


def sample(plan_path, out, index):
    plan, ctx = context(plan_path, out); spec = plan['tasks'][index]; t = spec['task']
    os.environ.update(environment(plan))
    parent = verify_inputs(plan, spec)
    p = paths(out, spec)
    require(p['prepared'].exists() and read(p['prepared'])['context_sha256'] == fingerprint(ctx)
            and all(p[name].is_file() for name in RAW_NAMES), 'Sampling requires verified copied originals')
    import shared_base_curve_reference as reference
    from shared_base_posterior_full_worker import periodic_checkpoints
    proposal = parent['identity']['proposal']
    require(proposal['sampler_class'] == 'shared_base_surrogate.SharedBaseSurrogateMCMC',
            'Unexpected frozen proposal class')
    args = argparse.Namespace(input=rooted(spec['source_folder']), out=Path(out)/'references',
        index=t['index'], prefix=t['time'], alpha_order=t['order'], proposal='iid',
        steps=proposal['proposal_spec']['surrogate_steps'], power=.5, concentration=1.,
        burn=parent['identity']['burn'], seed=parent['identity']['master_sampling_seed'],
        initial_draws=spec['target_draws'], max_draws=spec['target_draws'])
    with periodic_checkpoints(paths(out, spec)['checkpoint'], plan['resources']['checkpoint_seconds']):
        result = reference.run_job(args)
    return 0 if result['ready_for_cohort_check'] else 2


def audit_case(plan_path, out, index):
    plan, ctx = context(plan_path, out); spec = plan['tasks'][index]
    os.environ.update(environment(plan))
    verify_inputs(plan, spec, raw=True)
    completed_reference(ctx, spec)
    p = paths(out, spec)
    verify_preserved_samples(rooted(spec['parent_files']['checkpoint']['path']), p['checkpoint'], spec,
                             complete=True)
    import shared_base_local_revalidation as auditor
    task = child_task(ctx, spec)
    row = auditor.audit_one_reference(task, fingerprint(ctx))
    require(row['corrected']['audit']['status'] in ('ready', 'needs_more_draws'),
            'Independent canonical full-chain audit did not complete')
    verify_inputs(plan, spec, raw=True)
    atomic_json(p['audit'], row)
    return row


def checked_result(plan, ctx, spec, *, replay=True):
    import shared_base_local_revalidation as auditor
    p = paths(ctx['output'], spec); receipt, row = read(p['receipt']), read(p['audit'])
    require(receipt['schema'] == SCHEMA and receipt['context_sha256'] == fingerprint(ctx)
            and receipt['task'] == spec['task'] and receipt['sampling_complete'] is True
            and receipt['target_draws'] == spec['target_draws'] and receipt['worker_exit_code'] in (0, 2)
            and receipt['audit_sha256'] == sha(p['audit'])
            and receipt['prepared_sha256'] == sha(p['prepared']), 'Completed task receipt differs')
    verify_inputs(plan, spec, raw=True)
    completed_reference(ctx, spec)
    task = child_task(ctx, spec)
    auditor.validate_row(row, task, fingerprint(ctx), replay=replay)
    require(receipt['status'] == ('ready' if row['corrected']['audit']['ready'] else 'unresolved'),
            'Receipt does not reflect independent audit')
    if replay:
        verify_preserved_samples(rooted(spec['parent_files']['checkpoint']['path']), p['checkpoint'], spec,
                                 complete=True)
    return receipt, row


def attempts_for(out, spec):
    path = paths(out, spec)['attempts']
    return read(path) if path.exists() else []


def cost_seconds(attempt):
    # A process lost before recording its elapsed time consumes its reservation.
    return attempt.get('wall_seconds', attempt['reserved_seconds'])


def allocate(plan, ctx, spec, *, retry):
    out = ctx['output']; p = paths(out, spec); limits = plan['resources']
    with lock(Path(out)/'budget.lock', blocking=True):
        attempts = attempts_for(out, spec)
        require(len(attempts) < limits['max_attempts'], 'Fixed allocation count exhausted')
        if attempts:
            require(retry and attempts[-1]['status'] in ('interrupted','timeout','running'),
                    'Recovery requires explicit --retry-incomplete; numerical or deterministic failure is terminal')
        all_attempts = [a for s in plan['tasks'] for a in attempts_for(out, s)]
        require(sum(cost_seconds(a) for a in all_attempts) + limits['task_wall_seconds']
                <= limits['total_worker_seconds'], 'Cumulative worker budget exhausted')
        attempt = dict(number=len(attempts)+1, context_sha256=fingerprint(ctx), status='running',
            reserved_seconds=limits['task_wall_seconds'], started_unix=time.time(), pid=os.getpid())
        attempts.append(attempt); atomic_json(p['attempts'], attempts)
    return attempts


def run_case(plan_path, out, index, *, retry=False):
    plan, ctx = context(plan_path, out); spec = plan['tasks'][index]; p = paths(out, spec)
    os.environ.update(environment(plan))
    with lock(p['task_lock']):
        if p['receipt'].exists():
            return checked_result(plan, ctx, spec, replay=False)[0]
        require(not busy(p['sample_lock']), 'An earlier sampler remains active')
        attempts = allocate(plan, ctx, spec, retry=retry); attempt = attempts[-1]
        began = time.monotonic(); process = None
        def finish(status, **extra):
            attempt.update(status=status, wall_seconds=time.monotonic()-began, finished_unix=time.time(), **extra)
            atomic_json(p['attempts'], attempts)
            return dict(task=spec['task'], status=status, sampling_complete=False, **extra)
        try:
            prepare_case(plan, ctx, spec)
            remaining = plan['resources']['task_wall_seconds'] - (time.monotonic()-began)
            require(remaining > 0, 'Task allocation expired during preparation')
            try:
                completed_reference(ctx, spec)
                sample_exit = 0  # Actual target/report/arrays checked; audit readiness remains independent.
            except (ValueError, KeyError, OSError):
                command = [plan['worker']['python'], str(Path(__file__).resolve()), 'sample', '--plan', str(plan_path),
                           '--out', str(out), '--index', str(index)]
                with p['worker_log'].open('a') as stream:
                    process = subprocess.run(command, cwd=ROOT, env=environment(plan), stdout=stream,
                        stderr=subprocess.STDOUT, timeout=max(.001, min(remaining-plan['resources']['audit_seconds'], plan['resources']['sampling_seconds'])))
                sample_exit = process.returncode
                if sample_exit < 0:
                    return finish('interrupted', worker_exit_code=sample_exit)
                require(sample_exit in (0, 2), f'Unchanged sampling worker failed with exit {sample_exit}')
            completed_reference(ctx, spec)
            remaining = plan['resources']['task_wall_seconds'] - (time.monotonic()-began)
            if remaining <= 0:
                return finish('timeout', worker_exit_code=sample_exit)
            command = [plan['worker']['python'], str(Path(__file__).resolve()), 'audit', '--plan', str(plan_path),
                       '--out', str(out), '--index', str(index)]
            with p['audit_log'].open('a') as stream:
                process = subprocess.run(command, cwd=ROOT, env=environment(plan), stdout=stream,
                    stderr=subprocess.STDOUT, timeout=min(remaining, plan['resources']['audit_seconds']))
            if process.returncode < 0:
                return finish('interrupted', worker_exit_code=sample_exit, audit_exit_code=process.returncode)
            require(process.returncode == 0, f'Independent audit failed with exit {process.returncode}')
            row = read(p['audit'])
            receipt = dict(schema=SCHEMA, context_sha256=fingerprint(ctx), task=spec['task'],
                sampling_complete=True, target_draws=spec['target_draws'], worker_exit_code=sample_exit,
                status='ready' if row['corrected']['audit']['ready'] else 'unresolved',
                audit_sha256=sha(p['audit']), prepared_sha256=sha(p['prepared']))
            atomic_json(p['receipt'], receipt)
            checked_result(plan, ctx, spec, replay=False)
            finish(receipt['status'])
            return receipt
        except subprocess.TimeoutExpired:
            return finish('timeout', explanation='Bounded allocation expired; last atomic checkpoint retained')
        except Exception as error:
            finish('failed', error=str(error))
            raise


def aggregate_groups(plan, rows):
    from shared_base_curve_audit import aggregate_time
    expected = {s['task']['id']: s for s in plan['tasks']}
    found = {r['task']['key']: r for r in rows}
    require(len(found) == len(rows) and set(found) <= {key(s) for s in expected.values()},
            'Duplicate or undeclared refinement audit')
    for spec in expected.values():
        if key(spec) not in found:
            continue
        row = found[key(spec)]; t = spec['task']; audited = row['corrected']['audit']
        require(all(row['task'][name] == t[name] for name in ('depth','index','time','order','setting'))
                and (audited['index'], audited['time'], audited['order']) == (t['index'], t['time'], t['order']),
                'Audited task coordinates differ from frozen selected history')
    complete = len(found) == len(expected)
    groups = []
    for group in plan['groups']:
        selected = group['selected_indices']; depth = group['depth']
        audits = [r['corrected']['audit'] for r in rows if r['task']['depth'] == depth]
        value = aggregate_time(audits, selected, [16, 32])
        # Each full selected group is independent; never release an available-case mean.
        released = value['complete'] and value['passed']
        groups.append(dict(depth=depth, time=64, selected_indices=selected, n_histories=len(selected),
            numerical_validation=value, bayes_released=released,
            paired_bayes_comparisons_released=released))
    return dict(schema=SCHEMA, complete=complete, expected_references=len(expected), observed_references=len(rows),
        missing_keys=sorted({key(s) for s in expected.values()}-set(found)), groups=groups,
        all_groups_passed=complete and all(g['numerical_validation']['passed'] for g in groups),
        training=False, model_inference=False, histories_added=False, tolerances_changed=False,
        historical_artifacts_modified=False,
        interpretation='Both complete fixed selected groups and both integration rules are retained. Missing or failed references never produce an available-case mean. Numerical failure withholds every affected Bayesian comparison.')


def merge(plan_path, out):
    plan, ctx = context(plan_path, out); os.environ.update(environment(plan))
    rows = []; bindings = {}
    for spec in plan['tasks']:
        p = paths(out, spec)
        require(not busy(p['task_lock']) and not busy(p['sample_lock']), 'Refinement task is still active')
        if p['receipt'].exists():
            receipt, row = checked_result(plan, ctx, spec, replay=True)
            rows.append(row)
            bindings[spec['task']['id']] = dict(receipt_sha256=sha(p['receipt']), audit_sha256=sha(p['audit']),
                                               files=row['task']['file_sha256'])
    result = aggregate_groups(plan, rows)
    result.update(context_sha256=fingerprint(ctx), bindings=bindings,
                  source_plan_id=plan.get('plan_id'), sampling_performed=bool(rows))
    context(plan_path, out)
    from validate_shared_base_mcmc import json_safe
    result = json_safe(result)
    atomic_json(Path(out)/'summary.json', result)
    return result


def external_active(plan):
    roots = {rooted(s['parent_files']['reference']['path']).parent for s in plan['tasks']}
    return sum(busy(p) for root in roots for p in root.glob('*.lock'))


def dispatch(plan_path, out, *, retry=False, max_seconds=None, only_tasks=None, poll_seconds=2.):
    plan, ctx = context(plan_path, out); out = Path(out); limits = plan['resources']
    duration = limits['max_dispatch_seconds'] if max_seconds is None else max_seconds
    require(0 < duration <= limits['max_dispatch_seconds'], 'Dispatch duration exceeds the frozen bound')
    started = time.monotonic(); active = {}; stop = []; jobs = {}
    selected = set(range(len(plan['tasks']))) if only_tasks is None else set(only_tasks)
    require(selected and selected <= set(range(len(plan['tasks']))), 'Undeclared dispatch task index')
    with lock(out/'dispatcher.lock'):
        for index, spec in enumerate(plan['tasks']):
            p = paths(out, spec)
            require(not busy(p['task_lock']) and not busy(p['sample_lock']), 'A previous allocation remains active')
            attempts = attempts_for(out, spec)
            if p['receipt'].exists():
                state = checked_result(plan, ctx, spec, replay=False)[0]['status']
            elif not attempts:
                state = 'pending'
            elif retry and len(attempts) < limits['max_attempts'] and attempts[-1]['status'] in ('timeout','interrupted','running'):
                state = 'pending'
            else:
                state = attempts[-1]['status']
            jobs[index] = dict(task_id=spec['task']['id'], state=state)
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in old_handlers:
            signal.signal(sig, lambda signum, frame: stop.append('signal'))
        def save():
            state = dict(schema=SCHEMA, context_sha256=fingerprint(ctx), counts=dict(Counter(j['state'] for j in jobs.values())),
                jobs=jobs, active_workers=len(active), reserved_memory_bytes=sum(slot_bytes(plan['tasks'][i]['target_draws']) for i in active),
                stop_reason=stop[0] if stop else None, updated_unix=time.time())
            atomic_json(out/'status.json', state)
            return state
        try:
            while True:
                for index, (proc, stream, began) in list(active.items()):
                    if proc.poll() is None and (time.monotonic()-began >= limits['task_wall_seconds'] or stop):
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
                    if proc.poll() is None:
                        continue
                    stream.close(); spec = plan['tasks'][index]; p = paths(out, spec)
                    if p['receipt'].exists():
                        jobs[index]['state'] = read(p['receipt'])['status']
                    else:
                        attempts = attempts_for(out, spec)
                        jobs[index]['state'] = attempts[-1]['status'] if attempts else 'failed'
                        if jobs[index]['state'] == 'running':
                            attempts[-1].update(status='interrupted', wall_seconds=min(limits['task_wall_seconds'],time.monotonic()-began),
                                                finished_unix=time.time())
                            atomic_json(p['attempts'], attempts); jobs[index]['state'] = 'interrupted'
                    jobs[index]['exit_code'] = proc.returncode
                    del active[index]
                if time.monotonic()-started >= duration and not stop:
                    stop.append('dispatch wall budget exhausted')
                if shutil.disk_usage(out).free < limits['disk_reserve_bytes'] and not stop:
                    stop.append('disk reserve reached')
                if not stop and not external_active(plan):
                    memory = sum(slot_bytes(plan['tasks'][i]['target_draws']) for i in active)
                    for index, item in jobs.items():
                        if len(active) >= limits['max_workers']:
                            break
                        if item['state'] != 'pending' or index not in selected:
                            continue
                        spec = plan['tasks'][index]; need = slot_bytes(spec['target_draws'])
                        require(need <= limits['memory_budget_bytes'], 'A task exceeds frozen memory admission')
                        if memory+need > limits['memory_budget_bytes']:
                            continue
                        cost = sum(cost_seconds(a) for s in plan['tasks'] for a in attempts_for(out, s))
                        # Account for launched children that have not yet written reservations.
                        unwritten = sum(limits['task_wall_seconds'] for i in active
                                        if not attempts_for(out, plan['tasks'][i])
                                        or attempts_for(out, plan['tasks'][i])[-1].get('pid') != active[i][0].pid)
                        if cost+unwritten+limits['task_wall_seconds'] > limits['total_worker_seconds']:
                            stop.append('cumulative worker budget exhausted'); break
                        log = out/'logs'/f"{spec['task']['id']}-{time.time_ns()}.log"
                        log.parent.mkdir(parents=True, exist_ok=True); stream = log.open('x')
                        command = [plan['worker']['python'], str(Path(__file__).resolve()), 'case', '--plan', str(plan_path),
                                   '--out', str(out), '--index', str(index)]
                        if retry:
                            command.append('--retry-incomplete')
                        proc = subprocess.Popen(command, cwd=ROOT, env=environment(plan), stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        active[index] = (proc, stream, time.monotonic()); memory += need
                        item.update(state='running', pid=proc.pid); save()
                state = save()
                if not active and (stop or all(jobs[i]['state'] != 'pending' for i in selected)):
                    break
                time.sleep(poll_seconds)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            for proc, stream, _ in active.values():
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                stream.close()
    # A separate merge performs potentially expensive independent numerical checks.
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','case','sample','audit','merge'))
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int); parser.add_argument('--retry-incomplete', action='store_true')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--only-task', type=int, action='append', help='Dispatch a smoke subset; complete manifest remains mandatory for release')
    args = parser.parse_args()
    if args.action in ('case','sample','audit'):
        require(args.index is not None and 0 <= args.index < 246, 'Need a declared task index 0..245')
    if args.action == 'sample':
        raise SystemExit(sample(args.plan,args.out,args.index))
    if args.action == 'case':
        result = run_case(args.plan,args.out,args.index,retry=args.retry_incomplete)
    elif args.action == 'audit':
        result = audit_case(args.plan,args.out,args.index)
        result = dict(status=result['corrected']['audit']['status'])
    elif args.action == 'merge':
        result = merge(args.plan,args.out)
    else:
        result = dispatch(args.plan,args.out,retry=args.retry_incomplete,max_seconds=args.max_seconds,only_tasks=args.only_task)
    print(json.dumps({k:v for k,v in result.items() if k in ('complete','all_groups_passed','counts','status','sampling_complete')}),flush=True)
    if args.action == 'merge':
        raise SystemExit(0 if result['all_groups_passed'] else 2 if result['complete'] else 3)
    if args.action == 'case' and not result.get('sampling_complete'):
        raise SystemExit(3)


if __name__ == '__main__':
    main()
