"""Print, submit or merge one explicit stage of the shared-base CPU campaign.

The default is a read-only dry run. No next stage, resubmission or full
cohort is launched automatically. Numerical dependency checks are repeated
inside each allocation by the authoritative shard runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCHEMA = 'shared-base-scitas-launch-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def stage_of(plan, name):
    require(name in plan['stages'], f'Unknown stage: {name}')
    stage = plan['stages'][name]
    for value in [name, *stage['case_ids']]:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is not None, 'Stage and case IDs must be simple filename components')
    require(stage['replicates'] > 0 and len(set(stage['case_ids'])) == len(stage['case_ids']), 'Invalid stage repetition/case plan')
    available = {case['id'] for case in plan['cases']}
    require(set(stage['case_ids']) <= available, 'Stage references missing fixed cases')
    return stage


def gate_errors(plan, plan_hash, stage_name, out):
    """Promotion reads completed merge receipts; it never submits dependencies."""
    errors = []
    for dependency in stage_of(plan, stage_name).get('requires', []):
        previous = stage_of(plan, dependency)
        path = out/'merged'/f'{dependency}.json'
        if not path.is_file():
            errors.append(f'missing merged prerequisite: {path}')
            continue
        receipt = json.loads(path.read_bytes())
        if (receipt.get('schema') != 'shared-base-cluster-merge-v1'
                or receipt.get('plan_sha256') != plan_hash or receipt.get('stage') != dependency
                or receipt.get('stage_config') != previous or receipt.get('complete') is not True
                or receipt.get('expected_records') != len(previous['case_ids'])*previous['replicates']
                or receipt.get('observed_records') != len(previous['case_ids'])*previous['replicates']
                or receipt.get('coverage') != {cid: list(range(previous['replicates'])) for cid in previous['case_ids']}
                or set(receipt.get('case_summaries', {})) != set(previous['case_ids'])):
            errors.append(f'incomplete or mismatched prerequisite: {path}')
            continue
        gate = receipt.get('gate', {})
        purpose = previous['purpose']
        if purpose == 'preflight':
            passed = gate.get('status') == 'preflight_only' and gate.get('runtime_ready') is True
        elif purpose in ('exact_short', 'confirmation', 'quadrature_refinement'):
            passed = (gate.get('status') == 'passed_exact_short' and gate.get('exact_short_gate_passed') is True
                      and gate.get('passed_by_case') == {cid: True for cid in previous['case_ids']})
        else:
            passed = (gate.get('status') == 'repeatability_only' and gate.get('precision_passed') is True
                      and gate.get('precision_met_by_case') == {cid: True for cid in previous['case_ids']})
        if not passed:
            errors.append(f'prerequisite gate has not passed: {path}')
    return errors


def time_string(minutes):
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    return (f'{days}-' if days else '') + f'{hours:02d}:{minutes:02d}:00'


def build_launch(args):
    plan_path, out = args.plan.expanduser().resolve(), args.out.expanduser().resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    stage_name = args.stage or 'preflight'
    stage = None if args.gpu_benchmark else stage_of(plan, stage_name)
    require(not (args.gpu_benchmark and (args.stage or args.merge)), '--gpu-benchmark cannot be combined with --stage or --merge')
    errors = [] if args.gpu_benchmark else gate_errors(plan, plan_hash, stage_name, out)
    if args.merge:
        command = [str(args.venv.expanduser()/'bin/python'), str(ROOT/'code/scripts/shared_base_cluster.py'),
                   'merge', '--plan', str(plan_path), '--stage', stage_name,
                   '--shards', str(out/'shards'/stage_name), '--out', str(out/'merged'/f'{stage_name}.json')]
        return dict(mode='merge', command=command, errors=[], plan=plan, out=out)
    preflight = stage is not None and stage['purpose'] == 'preflight'
    concurrency = args.concurrency if args.concurrency is not None else (1 if preflight else 64)
    require(1 <= concurrency <= 64, '--concurrency must be between 1 and 64')
    if preflight:
        require(concurrency == 1, 'Preflight runs one CPU task at a time')
    wall_minutes = args.wall_minutes if args.wall_minutes is not None else (30 if args.gpu_benchmark else 15 if preflight else 120)
    require(2 <= wall_minutes <= (60 if args.gpu_benchmark else 10080), 'Wall time exceeds the selected QoS limit')
    seconds = args.seconds if args.seconds is not None else max(30, wall_minutes*60-120)
    require(1 <= seconds <= wall_minutes*60-60, 'Leave at least 60 seconds between the numerical deadline and Slurm limit')
    require(args.replicas_per_shard >= 1, '--replicas-per-shard must be positive')
    store, src = args.store.expanduser().resolve(), args.src.expanduser().resolve()
    tasks = []
    mode = 'gpu_benchmark' if args.gpu_benchmark else 'cpu'
    if args.gpu_benchmark:
        cases = Path(plan['gpu_cases'])
        if not cases.is_absolute():
            cases = ROOT/cases
        require(cases.is_file(), f'Missing GPU benchmark controls: {cases}')
        tasks = [dict(output=str(out/'gpu_benchmark/benchmark.json'), cases=str(cases), cases_sha256=sha(cases))]
        label = 'gpu-benchmark'
    else:
        for case_id in stage['case_ids']:
            for start in range(0, stage['replicates'], args.replicas_per_shard):
                stop = min(start+args.replicas_per_shard, stage['replicates'])
                tasks.append(dict(case_id=case_id, replica_start=start, replica_stop=stop,
                                  output=str(out/'shards'/stage_name/f'{case_id}_r{start:06d}_{stop:06d}.json')))
        label = stage_name
    selected = sorted(set(args.only_task)) if args.only_task else list(range(len(tasks)))
    require(bool(selected) and all(0 <= index < len(tasks) for index in selected), '--only-task is outside this stage task list')
    require(not args.gpu_benchmark or selected == [0], 'GPU benchmark has one task')
    script = HERE/('shared_base_gpu_benchmark.sbatch' if args.gpu_benchmark else 'shared_base_cpu.sbatch')
    code_files = [HERE/name for name in ('dispatch_shared_base.py', 'environment.sh')]+[script]
    code_files += [ROOT/'code/transformer'/name for name in (
        'shared_base_evidence_smc.py', 'shared_base_mcmc.py', 'shared_base_surrogate.py',
        'shared_base_efficiency.py', 'ranked_lsa.py', 'weighted_lsa.py', 'phi_source.py',
        'lsa_seq.py', 'shared_base_short_evidence.py')]
    code_files += [ROOT/'code/scripts/calibrate_shared_base_evidence.py', ROOT/plan['dependency_manifest']]
    if args.gpu_benchmark:
        code_files += [ROOT/'code/scripts/benchmark_shared_base_evidence_gpu.py', ROOT/'code/transformer/shared_base_evidence_gpu.py',
                       ROOT/'code/scripts/prepare_shared_base_cluster.py', ROOT/'code/scripts/shortprefix_shared_base_exact.py']
    else:
        code_files += [ROOT/'code/transformer/shared_base_cluster.py', ROOT/'code/scripts/shared_base_cluster.py']
    body = dict(schema=SCHEMA, mode=mode, repo_root=str(ROOT), plan=str(plan_path), plan_sha256=plan_hash,
                stage=None if args.gpu_benchmark else stage_name, store=str(store), src=str(src),
                seconds=seconds, tasks=tasks, runtime_directory=str(out/'runtime'/label), receipts=str(out/'merged'),
                dispatch_sha256=sha(HERE/'dispatch_shared_base.py'),
                code_sha256={str(path.relative_to(ROOT)): sha(path) for path in code_files},
                gpu_settings=dict(particles=8192, temperatures=64, moves=4, replicates=16,
                                  alpha_order=16, seed=2026091603, cpu_replicates=1) if args.gpu_benchmark else None)
    body['launch_id'] = fingerprint(body)
    manifest = out/'launches'/label/f"{body['launch_id'][:16]}.json"
    command = ['sbatch', '--parsable', f'--chdir={ROOT}', f'--job-name=sb-{label}',
               f'--time={time_string(wall_minutes)}', f'--output={out}/logs/{label}-%A_%a.out']
    if not args.gpu_benchmark:
        selector = (f'0-{len(tasks)-1}' if selected == list(range(len(tasks))) and len(tasks) > 1 else ','.join(map(str, selected)))
        command.append(f'--array={selector}%{concurrency}')
    if args.account:
        command.append(f'--account={args.account}')
    command += [str(script), str(manifest)]
    return dict(mode=mode, command=command, errors=errors, plan=plan, out=out, manifest=manifest,
                payload=body, selected_tasks=selected, concurrency=1 if args.gpu_benchmark else concurrency,
                wall_minutes=wall_minutes, store=store, src=src)


def main():
    dependency_root = Path(os.environ.get('CLUSTER_DEP_ROOT', str(Path.home()/'Projects/shared_base_dependencies_2026-09-16')))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'code/transformer/configs/shared_base_cluster_plan_2026-09-16.json')
    parser.add_argument('--stage')
    parser.add_argument('--gpu-benchmark', action='store_true')
    parser.add_argument('--merge', action='store_true')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--submit', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'output/scitas_shared_base_2026-09-16')
    parser.add_argument('--store', type=Path, default=Path(os.environ.get('SHARED_BASE_STORE', str(dependency_root/'tables/anchors_prod'))))
    parser.add_argument('--src', type=Path, default=Path(os.environ.get('SHARED_BASE_SRC', str(dependency_root/'src'))))
    parser.add_argument('--venv', type=Path, default=Path(os.environ.get('SHARED_BASE_VENV', str(Path.home()/'venv-tf'))))
    parser.add_argument('--account', default=os.environ.get('SHARED_BASE_ACCOUNT'))
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--replicas-per-shard', type=int, default=4)
    parser.add_argument('--only-task', type=int, action='append')
    parser.add_argument('--seconds', type=int)
    parser.add_argument('--wall-minutes', type=int)
    args = parser.parse_args()
    try:
        launch = build_launch(args)
        print(json.dumps(dict(mode=launch['mode'], requested_stage=args.stage or ('gpu-benchmark' if args.gpu_benchmark else 'preflight'),
                              tasks=len(launch.get('selected_tasks', [])), concurrency=launch.get('concurrency'),
                              prerequisite_errors=launch['errors'], action='execute' if args.submit else 'dry run')), flush=True)
        print(shlex.join(launch['command']), flush=True)
        if not args.submit:
            return
        require(not launch['errors'], 'Stage blocked: ' + '; '.join(launch['errors']))
        if launch['mode'] == 'merge':
            require(Path(launch['command'][0]).is_file(), 'Prepared Python environment is missing; set --venv')
        else:
            require((launch['store']/'anchors.json').is_file() and (launch['store']/'manifest.json').is_file(), 'Extract the sealed dependency bundle or select an existing matching --store before submitting')
            require((launch['src']/'product_model_with_memory/universal_tables.py').is_file(), 'Matching numerical source is missing; set --src')
            require((args.venv.expanduser()/'bin/python').is_file(), 'Prepared Python environment is missing; set --venv')
            manifest = launch['manifest']
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                require(json.loads(manifest.read_bytes()) == launch['payload'], 'Existing launch manifest differs; do not overwrite it')
            else:
                manifest.write_text(json.dumps(launch['payload'], indent=2)+'\n')
            (launch['out']/'logs').mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, SHARED_BASE_VENV=str(args.venv.expanduser().resolve()))
        # SLURM_SUBMIT_DIR is the submission process's cwd. Keep it aligned
        # with --chdir even when this launcher is called by absolute path.
        raise SystemExit(subprocess.run(launch['command'], env=environment, cwd=ROOT, check=False).returncode)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
