"""Real concurrent claims and mock Slurm checks, without models or cluster jobs."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'code/scripts/campaign_job_guard.py'
spec = importlib.util.spec_from_file_location('campaign_job_guard', SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    binpath = tmp_path / 'bin'
    binpath.mkdir()
    state = tmp_path / 'slurm.json'
    state.write_text(json.dumps(dict(queue='', accounting='', submitted=0)))
    executable = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
p=Path(os.environ['MOCK_SLURM']);s=json.loads(p.read_text());name=Path(sys.argv[0]).name
if s.get(name+'_fail'):sys.exit(1)
if name=='squeue':
    lines=s['queue'].splitlines()
    print('\\n'.join('|'.join(x.split('|')[:2]) if sys.argv[-1]=='--format=%i|%T' else x for x in lines))
elif name=='sacct': print(s['accounting'])
elif name=='scontrol': print(s.get('details',{}).get(sys.argv[-1],''))
else:
    s['submitted']+=1;s['args']=sys.argv[1:];p.write_text(json.dumps(s));print(12345)
'''
    for name in ('squeue', 'sacct', 'scontrol', 'sbatch'):
        p = binpath / name
        p.write_text(executable)
        p.chmod(0o755)
    monkeypatch.setenv('PATH', str(binpath) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('MOCK_SLURM', str(state))
    return state


def update(path, **kw):
    data = json.loads(path.read_text())
    path.write_text(json.dumps(dict(data, **kw)))


def legacy(root, job='101_0', family='hidden_factor', mode='train', arm='continue_large'):
    if family == 'hidden_factor':
        command = f'/repo/{guard.SCRIPTS[family]} {root} /parent /baseline /plan {mode}'
    else:
        command = f'/repo/{guard.SCRIPTS[family]} {root} {mode} {arm} /plan /parent /cohorts fresh'
    return f'{job}|RUNNING|{command}|/repo'


@pytest.mark.parametrize('family', ['hidden_factor', 'shared_base'])
def test_legacy_overlap_rejected_but_seeds_arms_modes_and_outputs_distinct(tmp_path, slurm, family):
    root = tmp_path / 'run'
    update(slurm, queue=legacy(root, family=family))
    with pytest.raises(guard.GuardError, match='101_0'):
        guard.check_active(family, root, {('train', 'continue_large', '0')})
    for key in [('train', 'continue_large', '1'), ('train', 'xlarge', '0'),
                ('benchmark', 'xlarge', '0'), ('eval', 'continue_large', '0')]:
        guard.check_active(family, root, {key})
    guard.check_active(family, tmp_path / 'other', {('train', 'continue_large', '0')})


def test_expanded_legacy_xlarge_array_and_benchmark(tmp_path, slurm):
    root = tmp_path / 'run'
    update(slurm, queue=legacy(root, job='101_3'))
    with pytest.raises(guard.GuardError, match='101_3'):
        guard.check_active('hidden_factor', root, {('train', 'xlarge', '1')})
    update(slurm, queue=legacy(root, job='102', mode='benchmark'))
    with pytest.raises(guard.GuardError, match='102'):
        guard.check_active('hidden_factor', root, {('benchmark', 'xlarge', '0')})


@pytest.mark.parametrize('family,job,filename,key', [
    ('hidden_factor', '4288103_0', 'train-4288103_0.out', ('train', 'continue_large', '0')),
    ('hidden_factor', '4288307', 'benchmark-4288307.out', ('benchmark', 'xlarge', '0')),
    ('shared_base', '4288114_1', 'train-continue_large-4288114_1.out', ('train', 'continue_large', '1')),
    ('shared_base', '4288318', 'benchmark-xlarge-4288318_4294967294.out', ('benchmark', 'xlarge', '0')),
])
def test_real_kuma_script_only_command_uses_stdout_identity(tmp_path, slurm, family, job, filename, key):
    root = tmp_path / 'run with space'
    update(slurm, queue=f'{job}|RUNNING|/repo/{guard.SCRIPTS[family]}|/repo',
           details={job: f'JobId={job} JobState=RUNNING StdOut={root}/logs/{filename} StdErr=/tmp/error'})
    with pytest.raises(guard.GuardError, match=job):
        guard.check_active(family, root, {key})
    guard.check_active(family, tmp_path / 'other', {key})


def test_atomic_mkdir_allows_exactly_one_simultaneous_owner(tmp_path):
    barrier = threading.Barrier(8)
    def run(index):
        barrier.wait()
        try:
            return guard.reserve(tmp_path, 'hidden_factor', ('train', 'xlarge', '0'), str(index))
        except guard.GuardError:
            return None
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, range(8)))
    assert sum(p is not None for p in results) == 1
    assert len(list(tmp_path.rglob('attempt-*'))) == 1


def test_terminal_owner_advances_without_deleting_previous_claim(tmp_path, slurm):
    key = ('train', 'xlarge', '0')
    old = guard.reserve(tmp_path, 'hidden_factor', key, 'old')
    guard.write(old / 'allocation.json', dict(job_id='100_2'))
    update(slurm, accounting='100_2|CANCELLED by 1|')
    new = guard.reserve(tmp_path, 'hidden_factor', key, 'new')
    assert old.is_dir() and new.name == 'attempt-000001'
    assert guard.read(old / 'owner.json')['token'] == 'old'


@pytest.mark.parametrize('accounting,queue', [('', ''), ('100_2|RUNNING|', ''),
                                            ('100_2|COMPLETED|', '100_2|COMPLETING|ignored|/repo')])
def test_unknown_or_active_owner_never_reclaimed(tmp_path, slurm, accounting, queue):
    key = ('train', 'xlarge', '0')
    old = guard.reserve(tmp_path, 'hidden_factor', key, 'old')
    guard.write(old / 'allocation.json', dict(job_id='100_2'))
    update(slurm, accounting=accounting, queue=queue)
    with pytest.raises(guard.GuardError, match='100_2'):
        guard.reserve(tmp_path, 'hidden_factor', key, 'new')
    assert len(list(tmp_path.rglob('attempt-*'))) == 1


def test_scheduler_errors_and_incomplete_claim_fail_closed(tmp_path, slurm):
    key = ('train', 'xlarge', '0')
    update(slurm, squeue_fail=True)
    with pytest.raises(guard.GuardError, match='Scheduler check failed'):
        guard.submit('hidden_factor', tmp_path, [key], ['sbatch', 'anything'])
    assert json.loads(slurm.read_text())['submitted'] == 0
    update(slurm, squeue_fail=False)
    claim = guard.slot(tmp_path, 'hidden_factor', key) / 'attempt-000000'
    claim.mkdir(parents=True)
    with pytest.raises(guard.GuardError, match='Unreadable claim'):
        guard.reserve(tmp_path, 'hidden_factor', key, 'new')


def test_submit_reserves_queued_task_and_same_token_only_enters_once(tmp_path, slurm, monkeypatch):
    root = tmp_path / 'run'
    key = ('train', 'xlarge', '0')
    guard.submit('hidden_factor', root, [key], ['sbatch', '--parsable', '/repo/hidden_factor_campaign.sbatch'])
    claim = next(guard.slot(root, 'hidden_factor', key).glob('attempt-*'))
    assert guard.owner_job(claim) == '12345_2'
    with pytest.raises(guard.GuardError, match='Claim occupied'):
        guard.submit('hidden_factor', root, [key], ['sbatch', '--parsable', 'anything'])
    assert json.loads(slurm.read_text())['submitted'] == 1
    monkeypatch.setenv('CAMPAIGN_GUARD_TOKEN', guard.read(claim / 'owner.json')['token'])
    monkeypatch.setenv('SLURM_JOB_ID', '12347')
    monkeypatch.setenv('SLURM_ARRAY_JOB_ID', '12345')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    guard.acquire('hidden_factor', root, key)
    with pytest.raises(guard.GuardError, match='already entered'):
        guard.acquire('hidden_factor', root, key)


def test_concurrent_cli_submissions_make_only_one_allocation(tmp_path, slurm):
    command = [sys.executable, str(SCRIPT), 'submit', '--family', 'hidden_factor',
               '--root', str(tmp_path / 'run'), '--task', 'train:continue_large:0', '--',
               'sbatch', '--parsable', '/repo/hidden_factor_campaign.sbatch']
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(5)]
    for process in processes:
        process.communicate(timeout=20)
    assert sorted(p.returncode for p in processes) == [0, 2, 2, 2, 2]
    assert json.loads(slurm.read_text())['submitted'] == 1


def test_uncertain_submission_keeps_claim_and_partial_failure_only_abandons_own_claim(tmp_path, slurm):
    keys = [('train', 'continue_large', '0'), ('train', 'continue_large', '1')]
    occupied = guard.reserve(tmp_path, 'shared_base', keys[1], 'other')
    with pytest.raises(guard.GuardError, match='occupied'):
        guard.submit('shared_base', tmp_path, keys, ['sbatch', '--parsable', 'anything'])
    own = next(guard.slot(tmp_path, 'shared_base', keys[0]).glob('attempt-*'))
    assert (own / 'not_submitted.json').exists()
    assert not (occupied / 'not_submitted.json').exists()
    update(slurm, sbatch_fail=True)
    with pytest.raises(guard.GuardError, match='uncertain'):
        guard.submit('shared_base', tmp_path, keys[:1], ['sbatch', '--parsable', 'anything'])
    newest = guard.slot(tmp_path, 'shared_base', keys[0]) / 'attempt-000001'
    assert not (newest / 'not_submitted.json').exists()
    with pytest.raises(guard.GuardError, match='occupied'):
        guard.reserve(tmp_path, 'shared_base', keys[0], 'again')


def test_batch_may_start_before_submit_receipt_is_written(tmp_path, slurm, monkeypatch):
    key = ('train', 'continue_large', '1')
    claim = guard.reserve(tmp_path, 'shared_base', key, 'reserved')
    monkeypatch.setenv('CAMPAIGN_GUARD_TOKEN', 'reserved')
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    monkeypatch.setenv('SLURM_ARRAY_JOB_ID', '100')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '1')
    guard.acquire('shared_base', tmp_path, key)
    assert guard.owner_job(claim) == '100_1'
    guard.write(claim / 'allocation.json', dict(job_id='100_1'))
    assert (claim / 'active').is_dir()


def test_batch_exit_trap_preserves_status_and_does_not_intercept_usr1(tmp_path, slurm):
    # Execute the actual trap text, independently of unavailable GPU/module setup.
    for name in guard.SCRIPTS.values():
        batch = (ROOT / 'code/transformer/kuma' / name).read_text()
        trap = next(line for line in batch.splitlines() if line.startswith('trap '))
        assert trap.endswith(' EXIT') and 'USR1' not in trap
        claim = tmp_path / name
        claim.mkdir()
        shell = 'claim=' + str(claim) + '\n' + trap.replace('python code/scripts/', sys.executable + ' code/scripts/') + '\nexit 75\n'
        result = subprocess.run(['bash', '-c', shell], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 75
        assert json.loads((claim / 'finished.json').read_text())['exit_code'] == 75
