"""One bounded local continuation; original unseen-context outputs stay immutable."""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import unseen_context_refinement as base
from prepare_unseen_context_refinement import PLAN, RUNTIME_PYTHON, read_plan

ROOT = base.ROOT
sys.path.insert(0, str(ROOT/'code/transformer'))
PARENT = ROOT/'output/unseen_context_refinement_2026-09-19'
DEFAULT_OUT = ROOT/'output/unseen_context_recovery_2026-09-20'
TASK_ID = 'L4_i58_t64_a32'
BASELINE, TARGET, WALL_SECONDS = 32768, 65536, 900
require, read, sha, fingerprint, write = base.require, base.read, base.sha, base.fingerprint, base.atomic_json


def progress(stage, **fields):
    import json
    print(json.dumps(dict(stage=stage, task=TASK_ID, **fields)), flush=True)


def validate_parent(spec, receipt, row):
    a = row['corrected']['audit']
    require(spec['task']['id'] == TASK_ID and spec['target_draws'] == BASELINE
        and receipt['status'] == 'unresolved' and receipt['target_draws'] == BASELINE
        and a['draws_per_chain'] == BASELINE and not a['ready']
        and a['status'] == 'needs_more_draws' and a['gates'] == dict(
            split_rhat_below_1_05=True, probability_ess_at_least_200=False,
            maximum_probability_mcse_at_most_0_005=True, raw_normalization=True),
        'Parent must remain the declared completed ESS-only failure')


def parent_bundle():
    plan, digest = read_plan(PLAN)
    runtime(plan)
    ctx = read(PARENT/'context.json')
    require(ctx == dict(schema=base.SCHEMA, plan=str(PLAN.resolve()),
        plan_sha256=digest, output=str(PARENT.resolve())), 'Parent context changed')
    spec = next(s for s in plan['tasks'] if s['task']['id'] == TASK_ID)
    p = base.paths(PARENT, spec)
    require(not base.busy(p['task_lock']) and not base.busy(p['sample_lock']), 'Parent case remains active')
    receipt, row = base.checked_result(plan, ctx, spec, replay=True)
    validate_parent(spec, receipt, row)
    return plan, ctx, spec, p, read(p['reference'])


def runtime(plan):
    os.environ.update(base.environment(plan))
    import numpy as np
    import scipy
    require(Path(sys.prefix).resolve() == RUNTIME_PYTHON.parent.parent.resolve(), 'Use the frozen local runtime')
    require(dict(python=list(sys.version_info[:3]), numpy=np.__version__, scipy=scipy.__version__)
        == plan['runtime'] and np.finfo(np.longdouble).nmant+1 == 53, 'Numerical runtime changed')


def bound_plan(out, parent, ctx, spec, paths):
    return dict(schema='unseen-context-recovery-plan-v1', task=spec['task'],
        baseline_draws=BASELINE, target_draws=TARGET, wall_seconds=WALL_SECONDS,
        max_attempts=1, max_workers=1, memory_allowance_bytes=base.slot_bytes(TARGET),
        parent_output=str(PARENT), parent_context_sha256=fingerprint(ctx),
        parent_plan_sha256=sha(PLAN), runtime=parent['runtime'], python=str(RUNTIME_PYTHON),
        parent_files={str(paths[k]): sha(paths[k]) for k in
            ('reference', 'checkpoint', 'moments', 'receipt', 'audit', 'prepared')},
        driver_sha256=sha(__file__), tolerances_changed=False)


def load(out, *, freeze=False):
    out = Path(out).absolute()
    require(out.parent.resolve() == (ROOT/'output').resolve() and out.resolve() != PARENT.resolve(),
            'Use a separate direct child of the project output directory')
    base.safe_paths(out, [out/'plan.json', out/'context.json'])
    parent, ctx, spec, original, report = parent_bundle()
    runtime(parent)
    plan = bound_plan(out, parent, ctx, spec, original)
    context = dict(schema='unseen-context-recovery-context-v1', output=str(out.resolve()),
                   plan_sha256=fingerprint(plan))
    if freeze:
        write(out/'plan.json', plan, immutable=True)
        write(out/'context.json', context, immutable=True)
    else:
        require(read(out/'plan.json') == plan and read(out/'context.json') == context,
                'Recovery plan, parent results or implementation changed')
    child = deepcopy(spec)
    child.update(baseline_draws=BASELINE, target_draws=TARGET)
    return parent, context, child, original, report


def prepare(out, ctx, spec, original):
    p = base.paths(out, spec)
    for name in base.RAW_NAMES:
        p[name].parent.mkdir(parents=True, exist_ok=True)
        require(not p[name].exists(), 'Recovery never overwrites or restarts a child')
        shutil.copyfile(original[name], p[name])
        require(sha(p[name]) == sha(original[name]), 'Copied parent differs')
    write(p['prepared'], dict(context_sha256=fingerprint(ctx), baseline_draws=BASELINE,
        target_draws=TARGET, parent_checkpoint_sha256=sha(original['checkpoint'])), immutable=True)
    base.verify_preserved_samples(original['checkpoint'], p['checkpoint'], spec, complete=False)
    return p


def validate_child(out, ctx, spec, original, parent_report):
    p = base.paths(out, spec)
    report = read(p['reference'])
    base.validate_continuation(report, parent_report, spec)
    require(report['checkpoint'] == str(p['checkpoint'].resolve())
        and report['checkpoint_sha256'] == sha(p['checkpoint'])
        and report['moments_sha256'] == sha(p['moments']), 'Child raw bindings differ')
    base.verify_preserved_samples(original['checkpoint'], p['checkpoint'], spec, complete=True)
    return base.child_task(ctx, spec)


def worker(out):
    allocation = read(Path(out)/'allocation.json')
    require(allocation['status'] == 'running' and allocation['task'] == TASK_ID
        and allocation['reserved_seconds'] == WALL_SECONDS and allocation['pid'] == os.getppid(),
        'Worker requires its live bounded supervisor')
    with (Path(out)/'worker.claim').open('x') as claim:
        claim.write(str(os.getpid())+'\n')  # Persistent exclusive claim: no second worker or retry.
    progress('preparing', target_draws=TARGET, wall_seconds=WALL_SECONDS)
    parent, ctx, spec, original, report = load(out, freeze=True)
    p = prepare(out, ctx, spec, original)
    import shared_base_curve_reference as reference
    from shared_base_posterior_full_worker import periodic_checkpoints
    identity = report['identity']
    require(identity['proposal']['sampler_class'] == 'shared_base_surrogate.SharedBaseSurrogateMCMC',
            'Unsupported original kernel')
    args = argparse.Namespace(input=base.rooted(spec['source_folder']), out=Path(out)/'references',
        index=58, prefix=64, alpha_order=32, proposal='iid',
        steps=identity['proposal']['proposal_spec']['surrogate_steps'], power=.5, concentration=1.,
        burn=identity['burn'], seed=identity['master_sampling_seed'], initial_draws=TARGET, max_draws=TARGET)
    progress('sampling', baseline_draws=BASELINE, target_draws=TARGET)
    with periodic_checkpoints(p['checkpoint'], 60):
        reference.run_job(args)
    progress('auditing')
    task = validate_child(out, ctx, spec, original, report)
    import shared_base_local_revalidation as auditor
    row = auditor.audit_one_reference(task, fingerprint(ctx))
    require(row['corrected']['audit']['status'] in ('ready', 'needs_more_draws'), 'Audit did not finish')
    auditor.validate_row(row, task, fingerprint(ctx), replay=True)
    load(out)  # Parent receipts, raw arrays, runtime and all bound code remain unchanged.
    write(p['audit'], row, immutable=True)
    receipt = dict(schema='unseen-context-recovery-receipt-v1', context_sha256=fingerprint(ctx),
        task=spec['task'], target_draws=TARGET, sampling_complete=True,
        status='ready' if row['corrected']['audit']['ready'] else 'unresolved',
        audit_sha256=sha(p['audit']), prepared_sha256=sha(p['prepared']))
    write(p['receipt'], receipt, immutable=True)
    progress('finished', status=receipt['status'], minimum_probability_ess=row['corrected']['audit']['minimum_probability_ess'])
    return 0 if receipt['status'] == 'ready' else 2


def run(out):
    began = time.monotonic()
    out = Path(out).absolute()
    require(out.parent.resolve() == (ROOT/'output').resolve() and out.resolve() != PARENT.resolve(),
            'Use a separate direct child of the project output directory')
    require(not out.exists(), 'One allocation only; preserve the previous recovery directory')
    out.mkdir()
    allocation = dict(schema='unseen-context-recovery-allocation-v1', task=TASK_ID,
        reserved_seconds=WALL_SECONDS, started_unix=time.time(), status='running', pid=os.getpid())
    write(out/'allocation.json', allocation)
    progress('allocation_started', output=str(out), seconds=WALL_SECONDS)
    process = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor interrupted')
    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        # The subprocess deadline includes all preparation, sampling and audit work.
        with (out/'worker.log').open('x') as log:
            process = subprocess.Popen([str(RUNTIME_PYTHON), str(Path(__file__).resolve()),
                'worker', '--out', str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            allocation['worker_pid'] = process.pid
            write(out/'allocation.json', allocation)
            code = process.wait(timeout=max(.001, WALL_SECONDS-(time.monotonic()-began)))
        allocation.update(status={0:'ready', 2:'unresolved'}.get(code, 'failed'), exit_code=code)
    except subprocess.TimeoutExpired:
        allocation.update(status='timeout', exit_code=3)
    except BaseException:
        allocation.update(status='interrupted', exit_code=3)
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        allocation.update(wall_seconds=time.monotonic()-began, finished_unix=time.time())
        write(out/'allocation.json', allocation)
        progress('allocation_finished', status=allocation['status'])
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return allocation['exit_code']


def validated_overlay(out, *, replay=True):
    """Return one audited replacement row; caller reruns the original aggregate gate."""
    require((PARENT/'summary.json').exists() and not base.busy(PARENT/'dispatcher.lock'),
            'Wait for the parent automatic merge to finish')
    parent, ctx, spec, original, report = load(out)
    require(read(Path(out)/'allocation.json')['status'] in ('ready', 'unresolved'), 'Allocation incomplete')
    p = base.paths(out, spec)
    require(not base.busy(p['sample_lock']), 'Child remains active')
    receipt, row = read(p['receipt']), read(p['audit'])
    task = validate_child(out, ctx, spec, original, report)
    require(receipt == dict(schema='unseen-context-recovery-receipt-v1',
        context_sha256=fingerprint(ctx), task=spec['task'], target_draws=TARGET, sampling_complete=True,
        status='ready' if row['corrected']['audit']['ready'] else 'unresolved',
        audit_sha256=sha(p['audit']), prepared_sha256=sha(p['prepared'])), 'Receipt differs')
    import shared_base_local_revalidation as auditor
    auditor.validate_row(row, task, fingerprint(ctx), replay=replay)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker', 'check-overlay'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.mode == 'check-overlay':
        row = validated_overlay(args.out)
        progress('overlay_validated', status=row['corrected']['audit']['status'])
    else:
        raise SystemExit(run(args.out) if args.mode == 'run' else worker(args.out))
