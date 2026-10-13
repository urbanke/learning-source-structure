"""Independent orchestration checks; no scheduler submission or sampling."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shared_base_posterior_full as campaign


def fixture_campaign(tmp_path, n=2):
    tasks = [dict(id=f'L0_i{i+2}_t256_a16', setting='a1_L0', depth=0,
                  index=i+2, time=256, order=16) for i in range(n)]
    plan = dict(tasks=tasks)
    launch = dict(launch_id='frozen-launch', output=str(tmp_path / 'out'),
                  inputs=str(tmp_path / 'inputs'), dependency_root=str(tmp_path / 'deps'),
                  created_date='2026-09-17', venv=str(tmp_path / 'venv'))
    out = Path(launch['output'])
    campaign.queue_init(out, plan, launch)
    return launch, plan, out


def test_claim_excludes_held_tasks_and_does_not_repeat_finished_cases(tmp_path):
    launch, plan, out = fixture_campaign(tmp_path)
    first = campaign.claim(out, plan, 0)
    second = campaign.claim(out, plan, 1)
    assert (first[0], second[0]) == (0, 1)
    assert campaign.claim(out, plan, 2) is None
    campaign.finish(out, first[0], 'ready')
    first[1].close()
    assert campaign.claim(out, plan, 2) is None
    campaign.finish(out, second[0], 'retry')
    second[1].close()
    resumed = campaign.claim(out, plan, 3)
    assert resumed[0] == 1 and resumed[2] == 2
    resumed[1].close()


def test_abandoned_running_claim_is_recoverable_without_new_case(tmp_path):
    _, plan, out = fixture_campaign(tmp_path, n=1)
    prior = campaign.claim(out, plan, 0)
    prior[1].close()  # A killed allocation releases its OS lock.
    resumed = campaign.claim(out, plan, 1)
    assert resumed[0] == 0 and resumed[2] == 2
    resumed[1].close()


def test_terminal_receipt_repairs_queue_after_interrupted_finish(tmp_path):
    launch, plan, out = fixture_campaign(tmp_path, n=1)
    prior = campaign.claim(out, plan, 0)
    campaign.atomic_json(campaign.task_paths(launch, plan['tasks'][0])['receipt'],
                         dict(status='ready'))
    prior[1].close()
    assert campaign.claim(out, plan, 1) is None
    assert campaign.read(out / 'queue.json')['tasks'][0]['status'] == 'ready'


def test_timeout_limit_remains_visible(tmp_path):
    _, plan, out = fixture_campaign(tmp_path, n=1)
    for attempt in range(1, campaign.MAX_ATTEMPTS + 1):
        claimed = campaign.claim(out, plan, 0)
        assert claimed[2] == attempt
        campaign.finish(out, 0, 'retry')
        claimed[1].close()
    assert campaign.claim(out, plan, 0) is None
    assert campaign.read(out / 'queue.json')['tasks'][0]['status'] == 'timeout_exhausted'


def test_scheduler_submit_once_records_receipt_and_reuses_id(tmp_path, monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout='12345\n', stderr='estimate')
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    path = tmp_path / 'submit.json'
    for _ in range(2):
        assert campaign.submit_once(path, ['sbatch', 'fixed'], 'job', '2026-09-17') == '12345'
    assert len(calls) == 1


def test_uncertain_submission_is_never_blindly_retried(tmp_path, monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    path = tmp_path / 'submit.json'
    with pytest.raises(subprocess.CalledProcessError):
        campaign.submit_once(path, ['sbatch', 'fixed'], 'job', '2026-09-17')
    monkeypatch.setattr(campaign, 'find_job', lambda *args: None)
    with pytest.raises(ValueError, match='no automatic duplicate'):
        campaign.submit_once(path, ['sbatch', 'fixed'], 'job', '2026-09-17')
    assert len(calls) == 1
    monkeypatch.setattr(campaign, 'find_job', lambda *args: '12345')
    assert campaign.submit_once(path, ['sbatch', 'fixed'], 'job', '2026-09-17') == '12345'
    assert len(calls) == 1


def test_worker_budget_is_single_fixed_stage_with_original_seed(tmp_path):
    launch, plan, _ = fixture_campaign(tmp_path)
    cmd = campaign.worker_command(launch, plan['tasks'][0], 8192)
    assert cmd[1].endswith('shared_base_posterior_full_worker.py')
    assert cmd[cmd.index('--initial-draws') + 1] == '8192'
    assert cmd[cmd.index('--max-draws') + 1] == '8192'
    assert cmd[cmd.index('--seed') + 1] == str(campaign.WORKER['seed'])


def mock_execution(monkeypatch, launch):
    monkeypatch.setattr(campaign, 'verify_task_inputs', lambda *args: None)
    monkeypatch.setattr(campaign.original, 'progress_record', lambda *args: {})


def test_timeout_reuses_checkpoint_and_recorded_draw_budget(tmp_path, monkeypatch):
    launch, plan, out = fixture_campaign(tmp_path, n=1)
    mock_execution(monkeypatch, launch)
    paths = campaign.task_paths(launch, plan['tasks'][0])
    runtime = {'runtime': 'unchanged'}
    calls = []
    def interrupted(cmd, log, timeout, callback=None):
        calls.append(cmd)
        paths['checkpoint'].parent.mkdir(parents=True, exist_ok=True)
        paths['checkpoint'].write_bytes(b'partial-chain-state')
        return 75
    monkeypatch.setattr(campaign.original, 'run_child', interrupted)
    campaign.execute_task('launch.json', launch, plan, 0, 1, 20000, runtime)
    receipt = campaign.read(paths['receipt'])
    assert receipt['status'] == 'retry' and receipt['checkpoint_existed']
    assert receipt['next_draw_budget'] == campaign.WORKER['initial_draws']
    def completed(cmd, log, timeout, callback=None):
        calls.append(cmd)
        assert paths['checkpoint'].read_bytes() == b'partial-chain-state'
        if 'audit' in cmd:
            campaign.atomic_json(paths['audit'], {'corrected': {'audit': {'ready': True}}})
        return 0
    monkeypatch.setattr(campaign.original, 'run_child', completed)
    campaign.execute_task('launch.json', launch, plan, 0, 2, 20000, runtime)
    assert campaign.read(paths['receipt'])['status'] == 'ready'
    budgets = [c[c.index('--initial-draws') + 1] for c in calls if '--initial-draws' in c]
    assert budgets == ['4096', '4096']


def test_corrected_audit_stops_legacy_failure_without_extra_draws(tmp_path, monkeypatch):
    launch, plan, _ = fixture_campaign(tmp_path, n=1)
    mock_execution(monkeypatch, launch)
    paths = campaign.task_paths(launch, plan['tasks'][0])
    calls = []
    def completed(cmd, log, timeout, callback=None):
        calls.append(cmd)
        if 'audit' in cmd:
            campaign.atomic_json(paths['audit'], {'corrected': {'audit': {'ready': True}}})
            return 0
        return 2  # Old depth cutoff failed; corrected independent audit passes.
    monkeypatch.setattr(campaign.original, 'run_child', completed)
    campaign.execute_task('launch.json', launch, plan, 0, 1, 20000, {'runtime': 'same'})
    assert campaign.read(paths['receipt'])['status'] == 'ready'
    assert sum('--initial-draws' in c for c in calls) == 1


def test_nonconverged_stage_doubles_then_audits_again(tmp_path, monkeypatch):
    launch, plan, _ = fixture_campaign(tmp_path, n=1)
    mock_execution(monkeypatch, launch)
    paths = campaign.task_paths(launch, plan['tasks'][0])
    calls = []
    def completed(cmd, log, timeout, callback=None):
        calls.append(cmd)
        if 'audit' in cmd:
            ready = sum('audit' in c for c in calls) == 2
            campaign.atomic_json(paths['audit'], {'corrected': {'audit': {'ready': ready}}})
        return 0
    monkeypatch.setattr(campaign.original, 'run_child', completed)
    campaign.execute_task('launch.json', launch, plan, 0, 1, 20000, {'runtime': 'same'})
    budgets = [c[c.index('--initial-draws') + 1] for c in calls if '--initial-draws' in c]
    assert budgets == ['4096', '8192']
    assert campaign.read(paths['receipt'])['status'] == 'ready'


def test_changed_runtime_or_missing_saved_chain_rejected(tmp_path, monkeypatch):
    launch, plan, _ = fixture_campaign(tmp_path, n=1)
    mock_execution(monkeypatch, launch)
    paths = campaign.task_paths(launch, plan['tasks'][0])
    previous = dict(launch_id=launch['launch_id'], task=plan['tasks'][0], runtime={'v': 1},
                    checkpoint_existed=True, status='retry')
    campaign.atomic_json(paths['receipt'], previous)
    with pytest.raises(ValueError, match='runtime'):
        campaign.execute_task('launch.json', launch, plan, 0, 2, 20000, {'v': 2})
    with pytest.raises(ValueError, match='chains disappeared'):
        campaign.execute_task('launch.json', launch, plan, 0, 2, 20000, {'v': 1})


def test_idle_worker_does_not_queue_empty_successor_generations(tmp_path, monkeypatch):
    launch, plan, out = fixture_campaign(tmp_path, n=1)
    held = campaign.claim(out, plan, 0)
    monkeypatch.setattr(campaign, 'load_launch', lambda *args: (launch, plan))
    submitted = []
    monkeypatch.setattr(campaign, 'submit_once', lambda *args: submitted.append(args))
    campaign.run_worker('launch.json', slot=1, generation=3)
    assert submitted == []
    held[1].close()


def test_failure_status_never_claims_scientific_validation(tmp_path):
    _, _, out = fixture_campaign(tmp_path, n=1)
    campaign.finish(out, 0, 'unresolved')
    status = campaign.read(out / 'status.json')
    assert status['coverage_complete'] is True
    assert status['scientific_reference_validated'] is False
    assert status['counts'] == {'unresolved': 1}


def test_audit_timeout_retries_same_completed_budget(tmp_path, monkeypatch):
    launch, plan, _ = fixture_campaign(tmp_path, n=1)
    mock_execution(monkeypatch, launch)
    paths = campaign.task_paths(launch, plan['tasks'][0])
    def completed_worker_timed_out_audit(cmd, log, timeout, callback=None):
        if 'audit' in cmd:
            assert timeout == campaign.RESOURCES['audit_seconds']
            return 75
        return 0
    monkeypatch.setattr(campaign.original, 'run_child', completed_worker_timed_out_audit)
    campaign.execute_task('launch.json', launch, plan, 0, 1, 20000, {'runtime': 'same'})
    result = campaign.read(paths['receipt'])
    assert result['status'] == 'retry'
    assert result['next_draw_budget'] == campaign.WORKER['initial_draws']
    assert result['worker_exit_code'] == 0 and result['audit_exit_code'] == 75


def test_draw_limit_records_unresolved_without_relaxing_audit(tmp_path, monkeypatch):
    launch, plan, _ = fixture_campaign(tmp_path, n=1)
    mock_execution(monkeypatch, launch)
    paths = campaign.task_paths(launch, plan['tasks'][0])
    runtime = {'runtime': 'same'}
    campaign.atomic_json(paths['receipt'], dict(launch_id=launch['launch_id'], task=plan['tasks'][0],
                         runtime=runtime, checkpoint_existed=False, status='retry',
                         next_draw_budget=campaign.WORKER['max_draws']))
    calls = []
    def completed(cmd, log, timeout, callback=None):
        calls.append(cmd)
        if 'audit' in cmd:
            campaign.atomic_json(paths['audit'], {'corrected': {'audit': {'ready': False}}})
        return 0
    monkeypatch.setattr(campaign.original, 'run_child', completed)
    campaign.execute_task('launch.json', launch, plan, 0, 2, 20000, runtime)
    assert campaign.read(paths['receipt'])['status'] == 'unresolved'
    assert len(calls) == 2


def test_scheduler_reconciliation_collapses_array_records(tmp_path, monkeypatch):
    responses = iter(['123_[2-63%64]|job\n123_0|job\n', '123_0|job|\n123_1|job|\n'])
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout=next(responses)))
    assert campaign.find_job('job', '2026-09-17') == '123'


def test_scheduler_reconciliation_rejects_duplicate_jobs(tmp_path, monkeypatch):
    responses = iter(['123|job\n', '456|job|\n'])
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout=next(responses)))
    with pytest.raises(ValueError, match='Ambiguous'):
        campaign.find_job('job', '2026-09-17')
