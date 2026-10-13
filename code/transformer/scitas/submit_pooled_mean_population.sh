#!/bin/bash
# Login side reads only the small plan and receipts. No scientific imports.
set -euo pipefail
echo 'Pooled mean t16: bounded submission planner; no numerical work on this node.'
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 - "$repo_root" "$@" <<'PY'
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

root = Path(sys.argv[1]).resolve()
parser = argparse.ArgumentParser(description='One bounded t16 dependency graph; no retries or later contexts.')
parser.add_argument('--plan', type=Path, default=root / 'code/transformer/configs/pooled_mean_population_plan_2026-09-20.json')
parser.add_argument('--raw', type=Path, default=Path.home() / 'Projects/layered_architectures_multifamily/output/multifamily_2026-09-16')
parser.add_argument('--dependencies', type=Path, default=Path.home() / 'Projects/shared_base_dependencies_2026-09-16')
parser.add_argument('--out', type=Path, default=root / 'output/pooled_mean_population_2026-09-20')
parser.add_argument('--max-cpu-hours', type=float, help='Explicit hard allocation-hour ceiling for this entire wave.')
parser.add_argument('--max-cost-chf', type=float, help='Explicit cost ceiling at the rate recorded in the plan.')
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--submit', action='store_true')
mode.add_argument('--dry-run', action='store_true')
args = parser.parse_args(sys.argv[2:])
for name in ('plan', 'raw', 'dependencies', 'out'):
    value = getattr(args, name).expanduser().resolve()
    if any(c in str(value) for c in '\n\r\t'):
        parser.error('paths cannot contain control characters')
    setattr(args, name, value)
protected = [args.raw, args.dependencies, *[root / name for name in ('code', 'results', 'notes', 'figures', '.git')]]
if args.out == root or args.out in root.parents or any(
        args.out == path or args.out in path.parents or path in args.out.parents for path in protected):
    parser.error('output must be separate from original data, dependencies and repository sources')
plan = json.loads(args.plan.read_text())
if plan.get('schema') != 'pooled-mean-population-plan-v1' or plan.get('observed') != 16:
    parser.error('requires the separate mean-targeted plan at observed context 16 only')
scheduler = plan['scheduler']

def positive_int(value, name, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        parser.error('invalid bounded scheduler field: ' + name)
    return value

concurrency = positive_int(scheduler['concurrency'], 'concurrency', 64)
array_size = positive_int(scheduler['maximum_array_size'], 'maximum_array_size', 768)
prepare_minutes = positive_int(scheduler['prepare_minutes'], 'prepare_minutes', 60)
merge_minutes = positive_int(scheduler['merge_minutes'], 'merge_minutes', 60)
rate = scheduler['cpu_hour_rate_chf']
if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0:
    parser.error('a finite positive cost assumption is required')
stages = {}
for stage in ('calibration', 'population'):
    settings = scheduler[stage]
    count = positive_int(settings['tasks'], stage + '.tasks')
    minutes = positive_int(settings['allocation_minutes'], stage + '.allocation_minutes', 480)
    chunks = [(offset, min(array_size, count - offset)) for offset in range(0, count, array_size)]
    if len(chunks) > concurrency:
        parser.error('array chunk count exceeds concurrency; revise the bounded plan')
    stages[stage] = dict(tasks=count, minutes=minutes, chunks=chunks,
                         chunk_concurrency=concurrency // len(chunks))
cpu_hours = (prepare_minutes + 2 * merge_minutes + sum(
    stage['tasks'] * stage['minutes'] for stage in stages.values())) / 60
cost = cpu_hours * rate
for option in ('max_cpu_hours', 'max_cost_chf'):
    value = getattr(args, option)
    if value is not None and (not math.isfinite(value) or value <= 0):
        parser.error('budget caps must be finite and positive')
if args.submit and (args.max_cpu_hours is None or args.max_cost_chf is None):
    parser.error('--submit requires --max-cpu-hours and --max-cost-chf from the reviewed budget')
if args.max_cpu_hours is not None and cpu_hours > args.max_cpu_hours + 1e-9:
    parser.error(f'allocation ceiling {cpu_hours:.6f} CPU hours exceeds --max-cpu-hours')
if args.max_cost_chf is not None and cost > args.max_cost_chf + 1e-9:
    parser.error(f'cost ceiling CHF {cost:.6f} exceeds --max-cost-chf at the declared rate')
script = root / 'code/transformer/scitas/pooled_mean_population.sbatch'
worker = root / 'code/scripts/pooled_mean_population.py'
launcher = root / 'code/transformer/scitas/submit_pooled_mean_population.sh'

# Jed standard permits at most 7000 MiB per CPU. The cost calculation above
# assumes one CPU, so check the actual batch directives before any submission.
directives = dict(re.findall(r'^#SBATCH --([a-z-]+)=(\S+)\s*$', script.read_text(), re.M))
if directives.get('cpus-per-task') != '1' or directives.get('ntasks') != '1' or directives.get('nodes') != '1':
    parser.error('this allocation budget requires exactly one CPU, task and node')
memory = re.fullmatch(r'(\d+)([MG])?', directives.get('mem', ''))
if memory is None or int(memory[1]) <= 0:
    parser.error('batch memory must be a positive fixed MiB/GiB allocation')
memory_mib = int(memory[1]) * (1024 if memory[2] == 'G' else 1)
if memory_mib > 7000:
    parser.error('Jed standard memory exceeds its 7000 MiB per-CPU limit')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

identity = dict(plan=str(args.plan), plan_sha256=sha(args.plan), observed=16,
    raw=str(args.raw), dependencies=str(args.dependencies), out=str(args.out),
    code_sha256={str(path.relative_to(root)): sha(path) for path in (worker, script, launcher)},
    maximum_cpu_hours=cpu_hours, maximum_cost_chf=cost, memory_mib=memory_mib,
    cpu_hour_rate_chf=rate, approved_cpu_hour_cap=args.max_cpu_hours,
    approved_cost_cap_chf=args.max_cost_chf)
print(json.dumps(dict(stage='planned', observed=16, tasks={key: value['tasks'] for key, value in stages.items()},
    maximum_simultaneous_workers=concurrency, maximum_cpu_hours=cpu_hours, maximum_cost_chf=cost,
    cpu_hour_rate_chf=rate, memory_mib=memory_mib, cost_is_rate_based_projection=True,
    population_submitted=False, population_requires_calibration_pass=True,
    automatic_retries=False, later_contexts_submitted=False)), flush=True)

def command(stage, minutes, dependency=None, merge_stage=None, chunk=None):
    name = stage if merge_stage is None else merge_stage + '-merge'
    result = ['sbatch', '--parsable', '--chdir=' + str(root), '--job-name=pmean16-' + name,
              f'--time={minutes // 60:02d}:{minutes % 60:02d}:00',
              '--output=' + str(args.out / 'logs' / (name + '-%A_%a.out'))]
    if dependency:
        result += ['--dependency=' + ('afterany:' if stage == 'merge' else 'afterok:') + dependency,
                   '--kill-on-invalid-dep=yes']
    offset = 0
    if chunk is not None:
        offset, count = chunk
        result += [f'--array=0-{count - 1}%{stages[stage]["chunk_concurrency"]}']
    return result + [str(script), str(args.plan), str(args.raw), str(args.dependencies),
                     str(args.out), stage, merge_stage or '-', str(offset)]

def now():
    return datetime.now(timezone.utc).isoformat()

receipt_path = args.out / 'submission.json'
receipt = dict(schema='pooled-mean-population-submission-v1', identity=identity, jobs=[],
    created_utc=now(), status='preparing', submitting=None, population_submitted=False,
    population_waits_for_calibration=True, automatic_retries=False, later_contexts_submitted=False)

def save():
    receipt['updated_utc'] = now()
    temporary = receipt_path.with_name('.submission.json.tmp.' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt_path)

def dispatch(stage, minutes, dependency=None, merge_stage=None, chunk=None):
    cmd = command(stage, minutes, dependency, merge_stage, chunk)
    print(shlex.join(cmd), flush=True)
    if not args.submit:
        return 'DRY_' + (merge_stage or stage)
    receipt['submitting'] = dict(stage=stage, merge_stage=merge_stage, chunk=chunk, command=cmd)
    receipt['status'] = 'submitting'
    save()
    result = subprocess.run(cmd, check=True, cwd=root, text=True, stdout=subprocess.PIPE)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Scheduler response is uncertain; inspect submission.json and jobs before further action')
    record = dict(stage=stage, merge_stage=merge_stage, chunk=chunk, job_id=job,
                  allocation_minutes=minutes, submitted_utc=now())
    receipt['jobs'].append(record)
    receipt['population_submitted'] |= stage == 'population'
    receipt['submitting'] = None
    receipt['status'] = 'preparing'
    save()
    print(json.dumps(record), flush=True)
    return job

def launch():
    prepare = dispatch('prepare', prepare_minutes)
    calibration = [dispatch('calibration', stages['calibration']['minutes'], prepare, chunk=chunk)
                   for chunk in stages['calibration']['chunks']]
    gate = dispatch('merge', merge_minutes, ':'.join(calibration), merge_stage='calibration')
    population = [dispatch('population', stages['population']['minutes'], gate, chunk=chunk)
                  for chunk in stages['population']['chunks']]
    dispatch('merge', merge_minutes, ':'.join(population), merge_stage='population')
    if args.submit:
        receipt['status'] = 'submitted'
        save()
        print(json.dumps(dict(receipt=str(receipt_path), stage='submitted',
            population_submitted=True, population_waits_for_calibration=True,
            job_ids=[job['job_id'] for job in receipt['jobs']])), flush=True)
        status = subprocess.run(['squeue', '--noheader', '--jobs',
            ','.join(job['job_id'] for job in receipt['jobs']), '--format=%i|%j|%T|%M'],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(json.dumps(dict(stage='scheduler_snapshot', returncode=status.returncode,
            output=status.stdout, error=status.stderr,
            sampling_start_verified=False)), flush=True)

if not args.submit:
    launch()
else:
    if not (args.raw / 'heldout/manifest.json').is_file() or not (args.dependencies / 'tables/anchors_prod/anchors.json').is_file():
        parser.error('frozen source manifest or sealed anchor-store marker is missing')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.submission.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another submission is in progress; do not submit twice')
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if old['identity'] != identity or old.get('status') != 'submitted' or old.get('submitting') is not None:
                parser.error('existing submission differs or is uncertain; inspect its receipt and scheduler state')
            print(json.dumps(dict(action='already_submitted', receipt=str(receipt_path), jobs=old['jobs'],
                population_submitted=old['population_submitted'])), flush=True)
        else:
            if any(path.name != '.submission.lock' for path in args.out.iterdir()):
                parser.error('output is nonempty without a launch receipt; inspect it before any launch')
            (args.out / 'logs').mkdir()
            save()
            launch()
PY
