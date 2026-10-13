"""Gated, bounded continuation of the fixed shared-base likelihood experiment.

This controller submits existing immutable CPU shards. It never changes an old
gate, retries a failed experiment, or promotes the 12-case pilot to production.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / 'code/transformer/configs/shared_base_overnight_protocol_2026-09-16.json'
SCHEMA = 'shared-base-overnight-launch-v1'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_bytes()) == value, f'Frozen artifact changed: {path}')
    else:
        atomic_json(path, value)


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another overnight controller/submission holds this output lock') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def clean_environment(venv):
    # sbatch reads SBATCH_* options; inherited allocation variables can also
    # affect nested submission. Keep authentication, modules and PATH intact.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('SBATCH_', 'SLURM_'))}
    env['SHARED_BASE_VENV'] = str(venv)
    return env


def read_protocol(path):
    path = Path(path).resolve()
    protocol = json.loads(path.read_bytes())
    require(protocol['schema'] == 'shared-base-overnight-protocol-v1', 'Wrong overnight protocol')
    require(protocol['stage_groups'] == [['confirm'], ['alpha_refine', 'row_refine'], ['pilot16']],
            'Only confirmation, two refinements and pilot16 are authorized')
    require(protocol['future_not_queued'] == ['pilot64', 'pilot256', 'pilot1024']
            and protocol['production_reference_approved'] is False, 'Longer or production work is not authorized')
    for key in ('original_plan', 'followup_plan'):
        require(sha(ROOT / protocol[key]) == protocol[key + '_sha256'], f'Frozen {key} changed')
    return protocol


def frozen_code():
    paths = [Path(__file__), ROOT / 'code/transformer/scitas/launch_shared_base_overnight.py',
             ROOT / 'code/transformer/scitas/shared_base_overnight.sbatch',
             ROOT / 'code/transformer/scitas/submit_shared_base_overnight.sh',
             ROOT / 'code/scripts/review_shared_base_abaca.py',
             ROOT / 'code/transformer/scitas/launch_shared_base.py',
             ROOT / 'code/scripts/compare_shared_base_cluster.py',
             ROOT / 'code/transformer/shared_base_cluster.py',
             ROOT / 'code/scripts/calibrate_shared_base_evidence.py']
    paths += [ROOT / 'code/transformer/scitas' / name for name in
              ('shared_base_cpu.sbatch', 'dispatch_shared_base.py', 'environment.sh')]
    paths += [ROOT / 'code/transformer' / name for name in
              ('shared_base_evidence_smc.py', 'shared_base_mcmc.py', 'shared_base_surrogate.py',
               'shared_base_efficiency.py', 'ranked_lsa.py', 'weighted_lsa.py', 'phi_source.py',
               'lsa_seq.py', 'shared_base_short_evidence.py')]
    paths += [ROOT / 'code/scripts/shared_base_cluster.py',
              ROOT / 'code/transformer/configs/shared_base_dependency_manifest_2026-09-16.json']
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def validate_launch(launch):
    require(launch['schema'] == SCHEMA and launch['launch_id'] == fingerprint(
        {k: v for k, v in launch.items() if k != 'launch_id'}), 'Overnight launch identity changed')
    require(launch['repo_root'] == str(ROOT), 'Wrong checkout')
    require(sha(launch['protocol']) == launch['protocol_sha256'], 'Overnight protocol changed')
    require(launch['code_sha256'] == frozen_code(), 'Overnight implementation changed after submission')
    return read_protocol(launch['protocol'])


class Slurm:
    def __init__(self, venv):
        self.env = clean_environment(venv)

    def run(self, command):
        return subprocess.run(command, cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=60, check=True)

    def submit(self, command):
        text = self.run(command).stdout.strip()
        require(re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', text) is not None,
                f'Ambiguous sbatch result; inspect scheduler before any resubmission: {text}')
        return text.split(';')[0]

    def array_state(self, job, tasks):
        require(re.fullmatch(r'[0-9]+', str(job)) is not None, 'Invalid Slurm job ID')
        result = self.run(['sacct', '-n', '-P', '-X', '-j', str(job),
                           '--format=JobID%80,State%40,ExitCode'])
        records = {}
        for line in result.stdout.splitlines():
            columns = [v.strip() for v in line.split('|')]
            if len(columns) < 3:
                continue
            job_id, state, exit_code = columns[:3]
            match = re.fullmatch(re.escape(str(job)) + r'_(\d+)', job_id)
            if match:
                index = int(match[1])
            elif tasks == 1 and job_id == str(job):
                index = 0
            else:
                continue  # Ignore array summary and .batch/.extern steps.
            require(0 <= index < tasks and index not in records, 'Unexpected/duplicate Slurm array task')
            records[index] = (state.split()[0].rstrip('+'), exit_code)
        detail = {'expected_tasks': tasks, 'observed_tasks': len(records),
                  'states': {str(k): list(v) for k, v in records.items()}}
        if any(state not in ACTIVE | {'COMPLETED'} or (state == 'COMPLETED' and code != '0:0')
               for state, code in records.values()):
            return 'failed', detail
        if len(records) == tasks and all(pair == ('COMPLETED', '0:0') for pair in records.values()):
            return 'success', detail
        if any(state in ACTIVE for state, _ in records.values()):
            return 'active', detail
        # Missing accounting records can lag. squeue confirms an active array;
        # otherwise the controller waits a bounded accounting grace period.
        try:
            queue = self.run(['squeue', '-h', '-r', '-j', str(job), '-o', '%i|%T']).stdout.strip()
        except subprocess.CalledProcessError as error:
            if 'Invalid job id' not in (error.stderr or ''):
                raise
            queue = ''
        return ('active' if queue else 'accounting_pending'), detail


def strict_receipt(plan, stage, path):
    sys.path.insert(0, str(ROOT / 'code/transformer'))
    from shared_base_cluster import read_plan, validate_merge_receipt
    saved = json.loads(Path(path).read_bytes())
    checked = validate_merge_receipt(read_plan(plan, stage), saved, require_gate=True)
    require(all(saved['cases'][cid]['statistics'] == value['statistics']
                for cid, value in checked['cases'].items()), 'Nested case statistics changed')
    return checked


def bridge(launch, protocol):
    """Recompute the mixed-source short-control decision; no fabricated old pass."""
    sys.path.insert(0, str(ROOT / 'code/transformer'))
    from shared_base_cluster import merge_shards
    from review_shared_base_abaca import review
    out = Path(launch['out']) / 'bridge'
    old = Path(launch['original_receipt'])
    original = out / 'original_calibrate.json'
    immutable_json(original, json.loads(old.read_bytes()))
    followup = out / 'abaca_check.json'
    if not followup.exists():
        _, code = merge_shards(ROOT / protocol['followup_plan'], 'abaca_check', launch['followup_shards'], followup)
        require(code == 0, 'ABACA has incomplete shard coverage; workflow stopped')
    report = review(ROOT / protocol['original_plan'], ROOT / protocol['followup_plan'], original, followup)
    immutable_json(out / 'review.json', report)
    require(report['all_six_controls_meet_threshold_with_followup'] is True,
            'ABACA follow-up did not satisfy the unchanged six-control criteria')
    return {'review': str(out / 'review.json'), 'review_sha256': sha(out / 'review.json'),
            'original_receipt': str(original), 'original_receipt_sha256': sha(original),
            'source_original_receipt': str(old), 'source_original_receipt_sha256': sha(old),
            'followup_receipt': str(followup), 'followup_receipt_sha256': sha(followup),
            'original_calibration_gate_preserved': report['original_calibration_gate_preserved'],
            'all_six_controls_meet_threshold_with_followup': True}


def successor(launch, protocol, binding):
    plan = json.loads((ROOT / protocol['original_plan']).read_bytes())
    plan = copy.deepcopy(plan)
    plan['purpose'] = protocol['scope']
    # The failed old calibrate receipt cannot be a prerequisite in this new
    # plan. Its replacement is explicitly identified, never relabeled passed.
    del plan['stages']['preflight']
    del plan['stages']['calibrate']
    for stage, updates in protocol['stage_overrides'].items():
        plan['stages'][stage].update(updates)
    plan['overnight_successor'] = dict(
        protocol_sha256=launch['protocol_sha256'], original_plan_sha256=protocol['original_plan_sha256'],
        followup_plan_sha256=protocol['followup_plan_sha256'], external_prerequisite=binding,
        automatic_stage_groups=protocol['stage_groups'], future_not_queued=protocol['future_not_queued'],
        pilot_effort_rationale=protocol['pilot_effort_rationale'], production_reference_approved=False)
    path = Path(launch['out']) / 'successor_plan.json'
    immutable_json(path, plan)
    return path


def stage_launch(launch, protocol, plan, stage):
    sys.path.insert(0, str(ROOT / 'code/transformer/scitas'))
    from launch_shared_base import build_launch
    settings = protocol['resources'][stage]
    args = SimpleNamespace(plan=plan, out=Path(launch['out']), stage=stage, gpu_benchmark=False, merge=False,
        concurrency=settings['concurrency'], wall_minutes=settings['wall_minutes'], seconds=None,
        replicas_per_shard=settings['replicas_per_shard'], store=Path(launch['store']), src=Path(launch['src']),
        only_task=None, account=launch['account'], venv=Path(launch['venv']))
    result = build_launch(args)
    require(not result['errors'], 'Stage prerequisite blocked: ' + '; '.join(result['errors']))
    # Explicit globally unique names help a human reconcile an interrupted
    # sbatch response. Uncertain submission is never retried automatically.
    name = f"sbo-{launch['launch_id'][:12]}-{stage}"
    result['command'] = [f'--job-name={name}' if arg.startswith('--job-name=') else arg for arg in result['command']]
    return result, name


class Controller:
    def __init__(self, launch, slurm, *, clock=time.monotonic, sleep=time.sleep):
        self.launch, self.slurm, self.clock, self.sleep = launch, slurm, clock, sleep
        self.out = Path(launch['out'])
        self.status_path = self.out / 'status.json'
        self.deadline = clock() + launch['controller_seconds']
        self.state = json.loads(self.status_path.read_bytes()) if self.status_path.exists() else dict(
            schema='shared-base-overnight-status-v1', launch_id=launch['launch_id'], jobs={}, receipts={},
            production_reference_approved=False)
        require(self.state['launch_id'] == launch['launch_id'], 'Output belongs to another overnight launch')

    def save(self, status, reason=None):
        self.state.update(status=status, reason=reason, updated_unix=time.time())
        atomic_json(self.status_path, self.state)

    def wait(self, jobs):
        missing_since = {}
        while True:
            require(self.clock() < self.deadline, 'Controller deadline; saved child jobs may still be running. Resume attaches to them.')
            all_done = True
            for stage, job in jobs.items():
                status, detail = self.slurm.array_state(job['job_id'], job['tasks'])
                job['scheduler'] = detail
                require(status != 'failed', f'{stage} scheduler failure; no successor will be submitted')
                if status == 'accounting_pending':
                    missing_since.setdefault(stage, self.clock())
                    require(self.clock() - missing_since[stage] < 300,
                            f'{stage} has missing scheduler records after the accounting grace period')
                else:
                    missing_since.pop(stage, None)
                all_done = all_done and status == 'success'
            self.save('running')
            if all_done:
                return
            self.sleep(min(self.launch['poll_seconds'], max(0, self.deadline - self.clock())))

    def submit_stage(self, protocol, plan, stage):
        job = self.state['jobs'].get(stage)
        if job is not None:
            require('job_id' in job, f'{stage} has an uncertain previous submission; reconcile its saved unique job name before resuming')
            require(sha(job['manifest']) == job['manifest_sha256'], f'{stage} launch manifest changed')
            return job
        result, name = stage_launch(self.launch, protocol, plan, stage)
        immutable_json(result['manifest'], result['payload'])
        (self.out / 'logs').mkdir(exist_ok=True)
        job = dict(phase='submitting', job_name=name, tasks=len(result['selected_tasks']),
                   manifest=str(result['manifest']), manifest_sha256=sha(result['manifest']), command=result['command'])
        self.state['jobs'][stage] = job
        self.save('running')  # Write intent BEFORE sbatch; no duplicate on an ambiguous interruption.
        job['job_id'] = self.slurm.submit(result['command'])
        job['phase'] = 'submitted'
        self.save('running')
        return job

    def run(self):
        protocol = validate_launch(self.launch)
        with lock(self.out / 'controller.lock'):
            try:
                if self.state.get('status') == 'stopped':
                    raise ValueError('A failed workflow is not automatically retried; inspect status.json')
                self.save('running')
                self.wait({'abaca_check': dict(job_id=self.launch['after_job'], tasks=protocol['followup_tasks'])})
                binding = bridge(self.launch, protocol)
                plan = successor(self.launch, protocol, binding)
                self.state['successor_plan_sha256'] = sha(plan)
                for group in protocol['stage_groups']:
                    # Repeat bridge verification before every new batch; a
                    # persisted Boolean is never sufficient authorization.
                    validate_launch(self.launch)
                    require(bridge(self.launch, protocol) == binding, 'External prerequisite changed')
                    for stage in group:
                        for dependency in json.loads(plan.read_bytes())['stages'][stage]['requires']:
                            strict_receipt(plan, dependency, self.out / 'merged' / f'{dependency}.json')
                    jobs = {stage: self.submit_stage(protocol, plan, stage) for stage in group}
                    self.wait(jobs)
                    from shared_base_cluster import merge_shards
                    for stage in group:
                        target = self.out / 'merged' / f'{stage}.json'
                        if stage not in self.state['receipts']:
                            _, code = merge_shards(plan, stage, self.out / 'shards' / stage, target)
                            require(code == 0, f'{stage} shard coverage is incomplete')
                        else:
                            require(sha(target) == self.state['receipts'][stage], f'{stage} receipt changed')
                        strict_receipt(plan, stage, target)
                        receipt = json.loads(target.read_bytes())
                        expected_parents = {name: sha(self.out / 'merged' / f'{name}.json')
                                            for name in json.loads(plan.read_bytes())['stages'][stage]['requires']}
                        require(receipt['prerequisite_receipts'] == expected_parents,
                                f'{stage} used different prerequisite receipts')
                        previous = json.loads(Path(binding['followup_receipt']).read_bytes())['numerical']
                        for key in ('code_sha256', 'backend_sha256', 'dependency_manifest_sha256', 'dependency_sha256',
                                    'store_file_sha256', 'effective_numerical_environment', 'numpy_version',
                                    'scipy_version', 'python_version'):
                            require(previous.get(key) == receipt['numerical'].get(key),
                                    f'{stage} numerical implementation changed from the ABACA bridge: {key}')
                        self.state['receipts'][stage] = sha(target)
                        jobs[stage]['phase'] = 'numerical_gate_passed'
                        self.save('running')
                from compare_shared_base_cluster import compare, plot
                report = compare(plan, [self.out / 'merged/pilot16.json'])
                require(16 in [p['time'] for p in report['points']], 'Validated pilot16 comparison was not produced')
                directory = self.out / 'comparison'
                immutable_json(directory / 'comparison.json', report)
                plot(report, directory / 'pilot_comparison.png')
                self.state['comparison'] = str(directory / 'comparison.json')
                self.save('complete', 'Pilot16 complete. Review numerical precision and cost before longer prefixes; no full-cohort production.')
                return 0
            except Exception as error:
                paused = 'Controller deadline' in str(error)
                self.save('paused' if paused else 'stopped', f'{type(error).__name__}: {error}')
                print(self.state['reason'], file=sys.stderr, flush=True)
                return 75 if paused else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args(argv)
    launch = json.loads(args.launch.read_bytes())
    validate_launch(launch)
    return Controller(launch, Slurm(launch['venv'])).run()


if __name__ == '__main__':
    raise SystemExit(main())
