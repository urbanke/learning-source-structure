#!/usr/bin/env python3
"""Slurm duplicate protection, outside the immutable scientific implementation.

Each output/mode/arm/seed has append-only mkdir claims. Advancing to another
claim requires a confirmed terminal allocation; old claims are never removed.
This avoids both NFS flock assumptions and stale-lock deletion races.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid

SCRIPTS = {'hidden_factor': 'hidden_factor_campaign.sbatch',
           'shared_base': 'shared_base_campaign_train.sbatch'}
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


class GuardError(RuntimeError):
    pass


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise GuardError(f'Unreadable claim {path}; inspect its owner before retrying: {exc}') from exc


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, sort_keys=True) + '\n')
    os.replace(temp, path)


def scheduler(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardError(f'Scheduler check failed; no submission permitted: {exc}') from exc
    if result.returncode:
        raise GuardError(f'Scheduler check failed ({shlex.join(command)}): {result.stderr.strip()}')
    return result.stdout


def task(mode, arm, seed):
    if mode not in ('train', 'benchmark', 'eval') or arm not in ('continue_large', 'xlarge') or str(seed) not in ('0', '1'):
        raise GuardError('Invalid campaign mode/arm/seed')
    return (mode, arm, str(seed))


def job_identity(family, command, job_id, cwd):
    """Decode original, pre-guard submissions as well as the new launchers."""
    words = shlex.split(command)
    positions = [i for i, word in enumerate(words) if Path(word).name == SCRIPTS[family]]
    if not positions:
        return None
    args = words[positions[0] + 1:]
    if not args:
        # Kuma's %o contains only the script path. The launchers' fixed stdout
        # paths identify the output root/mode/arm without reading any run data.
        details = scheduler(['scontrol', '--oneliner', 'show', 'job', job_id])
        match = re.search(r'\bStdOut=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)', details.strip())
        if not match:
            raise GuardError(f'Cannot identify output of existing campaign job {job_id}; inspect scontrol show job {job_id}.')
        logfile = Path(match.group(1))
        if logfile.parent.name != 'logs' or not logfile.is_absolute():
            raise GuardError(f'Unexpected output path for campaign job {job_id}: {logfile}')
        root = str(logfile.parent.parent)
        if family == 'hidden_factor':
            if re.fullmatch(r'benchmark-(\d+|%j)\.out', logfile.name):
                args = [root, '', '', '', 'benchmark']
            elif re.fullmatch(r'train-(\d+|%A)_(\d+|%a)\.out', logfile.name):
                args = [root, '', '', '', 'train']
        else:
            match = re.fullmatch(r'(train|benchmark|eval)-(continue_large|xlarge)-(\d+|%A)_(\d+|%a|4294967294)\.out', logfile.name)
            if match:
                args = [root, match[1], match[2]]
    try:
        if family == 'hidden_factor':
            root, _, _, _, mode = args[:5]
            if mode == 'benchmark':
                key = task(mode, 'xlarge', 0)
            else:
                index = int(job_id.rsplit('_', 1)[1])
                arm, seed = [('continue_large', 0), ('continue_large', 1), ('xlarge', 0), ('xlarge', 1)][index]
                key = task(mode, arm, seed)
        else:
            root, mode, arm = args[:3]
            key = task(mode, arm, 0 if mode == 'benchmark' else job_id.rsplit('_', 1)[1])
        root = Path(root)
        if not root.is_absolute():
            root = Path(cwd) / root
        return root.resolve(), key
    except (ValueError, IndexError, GuardError) as exc:
        raise GuardError(f'Cannot identify existing campaign job {job_id}: {command}. Inspect it before submitting.') from exc


def check_active(family, root, keys, exclude=None):
    output = scheduler(['squeue', '--user=' + getpass.getuser(), '--noheader', '--array',
                        '--format=%i|%T|%o|%Z'])
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|', 3)
        if len(fields) != 4:
            raise GuardError('Unrecognized squeue response; no submission permitted')
        job_id, state, command, cwd = (s.strip() for s in fields)
        if job_id == exclude:
            continue
        identity = job_identity(family, command, job_id, cwd)
        if identity and identity[0] == Path(root).resolve() and identity[1] in keys:
            raise GuardError(f'Duplicate refused: {family} {identity[1]} already belongs to job {job_id} ({state}), output {root}.')


def confirmed_terminal(job_id):
    # Query the user's queue rather than --jobs: Slurm may reject a job ID
    # already purged from its active queue instead of returning empty output.
    active = scheduler(['squeue', '--user=' + getpass.getuser(), '--noheader', '--array', '--format=%i|%T'])
    if any(line.split('|', 1)[0].strip() == job_id for line in active.splitlines()):
        return False
    rows = scheduler(['sacct', '-X', '--noheader', '--parsable2', '--jobs=' + job_id,
                      '--format=JobID,State'])
    states = [parts[1].strip().split()[0].rstrip('+') for line in rows.splitlines()
              if len(parts := line.split('|')) >= 2 and parts[0].strip() == job_id]
    return bool(states) and all(state in TERMINAL for state in states)


def slot(root, family, key):
    return Path(root).resolve() / '.campaign_claims' / family / '/'.join(key)


def owner_job(claim):
    for name in ('started.json', 'allocation.json'):
        if (claim / name).exists():
            return read(claim / name)['job_id']
    return None


def reserve(root, family, key, token):
    base = slot(root, family, key)
    base.mkdir(parents=True, exist_ok=True)
    generation = 0
    while True:
        claim = base / f'attempt-{generation:06d}'
        try:
            claim.mkdir()
        except FileExistsError:
            owner = read(claim / 'owner.json')
            if (claim / 'not_submitted.json').exists():
                generation += 1
                continue
            job_id = owner_job(claim)
            if job_id and confirmed_terminal(job_id):
                generation += 1
                continue
            raise GuardError(f'Claim occupied: {claim}; job={job_id or "submission not yet recorded"}, '
                             f'host={owner.get("host")}, pid={owner.get("pid")}. '
                             'Retry after that job is terminal; if submission was interrupted, inspect the claim and scheduler first. No run files were removed.')
        write(claim / 'owner.json', dict(token=token, host=socket.gethostname(), pid=os.getpid(),
                                       created_unix=time.time(), family=family, key=key))
        return claim


def submit(family, root, keys, command):
    if not command or Path(command[0]).name != 'sbatch':
        raise GuardError('Expected an sbatch command')
    check_active(family, root, keys)
    token = uuid.uuid4().hex
    claims = []
    try:
        for key in keys:
            claims.append((key, reserve(root, family, key, token)))
        # Another legacy launcher may have submitted while claims were reserved.
        check_active(family, root, keys)
    except BaseException:
        for _, claim in claims:
            write(claim / 'not_submitted.json', dict(reason='Stopped before calling sbatch'))
        raise
    # Do not clear reservations after uncertain sbatch failures: an allocation
    # might exist even when the client failed to return its ID.
    result = subprocess.run([command[0], '--export=ALL,CAMPAIGN_GUARD_TOKEN=' + token, *command[1:]],
                            text=True, stdout=subprocess.PIPE)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    match = re.fullmatch(r'\s*(\d+)(?:;[^\s]+)?\s*', result.stdout or '')
    if result.returncode or not match:
        raise GuardError('Submission outcome uncertain; claims retained. Inspect the scheduler and .campaign_claims before retrying.')
    job = match.group(1)
    for key, claim in claims:
        mode, arm, seed = key
        index = str(int(seed) + (2 if arm == 'xlarge' else 0)) if family == 'hidden_factor' else seed
        allocation = job if mode == 'benchmark' else f'{job}_{index}'
        write(claim / 'allocation.json', dict(job_id=allocation))


def acquire(family, root, key):
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        raise GuardError('Batch claim requires a Slurm allocation')
    if os.environ.get('SLURM_ARRAY_TASK_ID'):
        job_id = os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
    check_active(family, root, {key}, exclude=job_id)
    token = os.environ.get('CAMPAIGN_GUARD_TOKEN')
    if token:
        matches = [p for p in slot(root, family, key).glob('attempt-*')
                   if read(p / 'owner.json').get('token') == token]
        if len(matches) != 1:
            raise GuardError('Missing or ambiguous submission claim; no batch work permitted')
        claim = matches[0]
        if (claim / 'not_submitted.json').exists():
            raise GuardError('Submission claim was abandoned; no batch work permitted')
        previous = owner_job(claim)
        if previous and previous != job_id:
            raise GuardError(f'Claim belongs to job {previous}, not {job_id}')
    else:
        claim = reserve(root, family, key, uuid.uuid4().hex)
        write(claim / 'allocation.json', dict(job_id=job_id))
    # Even two starts carrying the same token cannot enter the same batch.
    try:
        (claim / 'active').mkdir()
    except FileExistsError as exc:
        raise GuardError(f'Batch already entered claim {claim}; no work permitted') from exc
    write(claim / 'started.json', dict(job_id=job_id, host=socket.gethostname(), pid=os.getpid()))
    print(claim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('submit', 'acquire'):
        p = sub.add_parser(name)
        p.add_argument('--family', choices=SCRIPTS, required=True)
        p.add_argument('--root', type=Path, required=True)
        if name == 'submit':
            p.add_argument('--task', action='append', required=True, help='mode:arm:seed')
            p.add_argument('command', nargs=argparse.REMAINDER)
        else:
            p.add_argument('--mode', required=True)
            p.add_argument('--arm', required=True)
            p.add_argument('--seed', required=True)
    p = sub.add_parser('finish')
    p.add_argument('--claim', type=Path, required=True)
    p.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'submit':
            keys = [task(*s.split(':')) for s in args.task]
            command = args.command[1:] if args.command[:1] == ['--'] else args.command
            submit(args.family, args.root, keys, command)
        elif args.action == 'acquire':
            acquire(args.family, args.root, task(args.mode, args.arm, args.seed))
        else:
            write(args.claim / 'finished.json', dict(exit_code=args.exit_code, finished_unix=time.time()))
    except (GuardError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
