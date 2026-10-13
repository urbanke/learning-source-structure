"""Execute exactly one frozen launch task inside an allocated compute node."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def read_launch(path):
    launch = json.loads(Path(path).read_bytes())
    require(launch.get('schema') == 'shared-base-scitas-launch-v1', 'Unsupported launch manifest')
    identity = {key: value for key, value in launch.items() if key != 'launch_id'}
    require(fingerprint(identity) == launch.get('launch_id'), 'Launch identity mismatch')
    require(hashlib.sha256(Path(launch['plan']).read_bytes()).hexdigest() == launch['plan_sha256'], 'Plan changed after submission')
    root = Path(launch['repo_root'])
    for name, expected in launch['code_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source filename')
        require(hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, f'Source changed after submission: {name}')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == launch['dispatch_sha256'], 'Dispatcher changed after submission')
    return launch


def task_command(launch, index):
    require(0 <= index < len(launch['tasks']), 'Array task index is outside launch manifest')
    task = launch['tasks'][index]
    root = Path(launch['repo_root'])
    command = [sys.executable]
    if launch['mode'] == 'cpu':
        command += [str(root/'code/scripts/shared_base_cluster.py'), 'run', '--plan', launch['plan'],
                    '--stage', launch['stage'], '--case-id', task['case_id'],
                    '--replica-start', str(task['replica_start']), '--replica-stop', str(task['replica_stop']),
                    '--receipts', launch['receipts']]
    elif launch['mode'] == 'gpu_benchmark':
        require(hashlib.sha256(Path(task['cases']).read_bytes()).hexdigest() == task['cases_sha256'], 'GPU controls changed after submission')
        command += [str(root/'code/scripts/benchmark_shared_base_evidence_gpu.py'), '--device', 'cuda', '--cases', task['cases']]
        for name, value in launch['gpu_settings'].items():
            command += ['--'+name.replace('_', '-'), str(value)]
    else:
        raise ValueError('Unsupported launch mode')
    command += ['--out', task['output'], '--store', launch['store'], '--src', launch['src'], '--seconds', str(launch['seconds'])]
    return command


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def execute(launch, index):
    command = task_command(launch, index)
    task = launch['tasks'][index]
    Path(task['output']).parent.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    # Retain every invocation, including manual resumptions of the same shard.
    runtime_path = Path(launch['runtime_directory'])/f'task-{index:04d}-job-{job_id}-{os.getpid()}.json'
    started = time.monotonic()
    report = dict(schema='shared-base-scitas-runtime-v1', launch_id=launch['launch_id'],
                  plan_sha256=launch['plan_sha256'], task_index=index, task=task, mode=launch['mode'],
                  status='running', started_utc=utc_now(), hostname=socket.gethostname(),
                  platform=platform.platform(), python_executable=sys.executable, python_version=platform.python_version(),
                  slurm={name: value for name, value in os.environ.items() if name.startswith('SLURM_')},
                  thread_environment={name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  command=command, scientific_reference_validated=False)
    atomic_json(runtime_path, report)
    print(json.dumps(dict(event='task_started', task_index=index, output=task['output'], runtime=str(runtime_path))), flush=True)
    # The child owns its scientific output lock and checkpointing. Preserve its
    # timeout exit code (75); never convert this into a successful completion.
    completed = subprocess.run(command, check=False, cwd=launch['repo_root'])
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report.update(status='completed' if completed.returncode == 0 else 'deadline' if completed.returncode == 75 else 'failed',
                  exit_code=completed.returncode, finished_utc=utc_now(), elapsed_seconds=time.monotonic()-started,
                  child_peak_rss_bytes=int(peak if sys.platform == 'darwin' else 1024*peak))
    try:
        output = json.loads(Path(task['output']).read_bytes())
        report['numerical_runtime_resources'] = output.get('runtime_resources')
        report['completed_records'] = len(output.get('records', []))
        report['numerical_complete'] = output.get('complete')
    except (OSError, ValueError):
        report['numerical_complete'] = False
    atomic_json(runtime_path, report)
    print(json.dumps(dict(event='task_finished', task_index=index, exit_code=completed.returncode,
                         elapsed_seconds=report['elapsed_seconds'], child_peak_rss_bytes=report['child_peak_rss_bytes'],
                         runtime=str(runtime_path))), flush=True)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--task-index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    try:
        raise SystemExit(execute(read_launch(args.launch), args.task_index))
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
