"""Synthetic receipts and mocked Slurm; no sampler or scheduler is executed."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/scripts'))
sys.path.insert(0, str(ROOT / 'code/transformer/scitas'))
import shared_base_overnight as overnight
import launch_shared_base_overnight as launcher
import review_shared_base_abaca as review
import shared_base_cluster as cluster
from test_review_shared_base_abaca import receipt as short_receipt

READY = dict(compiled_kernel_available=True, store_sealed=True, store_read_only=True)
NUMERICAL = dict(compiled_kernel_available=True, code_sha256={'kernel.py': 'unchanged'})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def shards(plan, stage, tasks, parent_directory, *, failure=False):
    context = cluster.read_plan(plan, stage)
    context['prerequisite_receipts'] = {name: overnight.sha(parent_directory / f'{name}.json')
                                       for name in context['config'].get('requires', [])}
    config = context['config']
    for task in tasks:
        cid, start, stop = task['case_id'], task['replica_start'], task['replica_stop']
        shard = cluster.new_shard(context, cid, NUMERICAL, start, stop, READY)
        for index in range(start, stop):
            logz = context['cases'][cid].get('known_log_evidence')
            if logz is None:
                logz = -config['n_tokens'] * math.log(64)
            if failure and cid == config['case_ids'][0]:
                logz -= .02 * math.log(2)
            estimate = dict(log_evidence=logz, seed=cluster.expected_seed(context, cid, index),
                sampler_class='SharedBaseMCMC', particles=config['particles'], temperatures=config['temperatures'],
                moves=config['moves'], n_tokens=config['n_tokens'], elapsed_seconds=1.)
            cluster.append_record(shard, context, index, estimate)
        shard.update(complete=True, status='completed')
        write(Path(task['output']), shard)


@pytest.fixture
def trial(tmp_path):
    protocol = json.loads(overnight.DEFAULT_PROTOCOL.read_bytes())
    original = tmp_path / 'original_plan.json'
    write(original, json.loads(review.DEFAULT_ORIGINAL.read_bytes()))
    followup = tmp_path / 'followup_plan.json'
    new = json.loads(review.DEFAULT_FOLLOWUP.read_bytes())
    new['followup_of'].update(original_plan=str(original), original_plan_sha256=overnight.sha(original))
    write(followup, new)
    protocol.update(original_plan=str(original), original_plan_sha256=overnight.sha(original),
                    followup_plan=str(followup), followup_plan_sha256=overnight.sha(followup))
    protocol_path = tmp_path / 'protocol.json'
    write(protocol_path, protocol)
    previous = tmp_path / 'original/merged/calibrate.json'
    write(previous, short_receipt(cluster.read_plan(original, 'calibrate'), {'cal_abaca': .02}))
    followup_shards = tmp_path / 'abaca/shards/abaca_check'
    tasks = [dict(case_id='cal_abaca', replica_start=start, replica_stop=start+4,
                  output=str(followup_shards / f'{start}.json')) for start in range(0, 128, 4)]
    shards(followup, 'abaca_check', tasks, tmp_path)
    launch = dict(schema=overnight.SCHEMA, repo_root=str(ROOT), protocol=str(protocol_path),
        protocol_sha256=overnight.sha(protocol_path), code_sha256=overnight.frozen_code(), after_job='66543429',
        out=str(tmp_path / 'overnight'), original_receipt=str(previous), followup_shards=str(followup_shards),
        store=str(tmp_path / 'store'), src=str(tmp_path / 'src'), venv=str(tmp_path / 'venv'), account=None,
        poll_seconds=1, controller_seconds=100)
    launch['launch_id'] = overnight.fingerprint(launch)
    return launch, protocol


class FakeSlurm:
    def __init__(self, *, failed_stage=None, missing_stage=None):
        self.events, self.jobs = [], {}
        self.failed_stage, self.missing_stage = failed_stage, missing_stage

    def submit(self, command):
        payload = json.loads(Path(command[-1]).read_bytes())
        stage = payload['stage']
        self.events.append(('submit', stage))
        tasks = payload['tasks'][:-1] if self.missing_stage == stage else payload['tasks']
        shards(payload['plan'], stage, tasks, Path(payload['receipts']), failure=self.failed_stage == stage)
        job = str(9000 + len(self.jobs))
        self.jobs[job] = stage
        return job

    def array_state(self, job, tasks):
        self.events.append(('wait', self.jobs.get(job, 'abaca_check')))
        return 'success', {'expected_tasks': tasks, 'observed_tasks': tasks}


def test_fixed_protocol_retains_budgets_seeds_and_has_bounded_scope(trial):
    launch, protocol = trial
    original = json.loads(Path(protocol['original_plan']).read_bytes())
    old_bytes = Path(launch['original_receipt']).read_bytes()
    binding = overnight.bridge(launch, protocol)
    path = overnight.successor(launch, protocol, binding)
    new = json.loads(path.read_bytes())
    assert new['acceptance'] == original['acceptance']
    assert new['prior'] == original['prior']
    assert new['comparison'] == original['comparison']
    assert new['cases'] == original['cases']
    for stage in ('confirm', 'alpha_refine', 'row_refine', 'pilot16', 'pilot64', 'pilot256', 'pilot1024'):
        expected = copy.deepcopy(original['stages'][stage])
        expected.update(protocol['stage_overrides'].get(stage, {}))
        assert new['stages'][stage] == expected
    assert new['stages']['pilot16']['particles'] == 16384
    assert new['stages']['pilot16']['replicates'] == 32
    assert binding['original_calibration_gate_preserved'] == 'failed_exact_short'
    assert Path(launch['original_receipt']).read_bytes() == old_bytes
    assert protocol['stage_groups'][-1] == ['pilot16']
    assert protocol['resources']['confirm']['concurrency'] == 64
    assert sum(protocol['resources'][s]['concurrency'] for s in ('alpha_refine', 'row_refine')) == 64
    assert cluster.read_plan(path, 'confirm')['config']['requires'] == []


def test_full_mocked_pipeline_recomputes_gates_then_stops_at_pilot16(trial, monkeypatch):
    launch, _ = trial
    import compare_shared_base_cluster as comparison
    monkeypatch.setattr(comparison, 'plot', lambda report, path: path.write_bytes(b'test figure'))
    slurm = FakeSlurm()
    assert overnight.Controller(launch, slurm).run() == 0
    state = json.loads((Path(launch['out']) / 'status.json').read_bytes())
    assert state['status'] == 'complete'
    assert set(state['jobs']) == {'confirm', 'alpha_refine', 'row_refine', 'pilot16'}
    assert state['production_reference_approved'] is False
    assert slurm.events.index(('submit', 'row_refine')) < slurm.events.index(('wait', 'alpha_refine'))
    report = json.loads(Path(state['comparison']).read_bytes())
    assert [point['time'] for point in report['points']] == [1, 4, 16]
    assert report['production_reference_validated'] is False
    assert len([r for r in report['sequence_records'] if r['time'] == 16]) == 12
    previous_submissions = list(slurm.events)
    # Reconciliation revalidates all raw receipts and does not submit a job twice.
    assert overnight.Controller(launch, slurm).run() == 0
    assert [e for e in slurm.events if e[0] == 'submit'] == [e for e in previous_submissions if e[0] == 'submit']


@pytest.mark.parametrize('kind', ['bad_gate', 'missing_shard'])
def test_complete_batch_failure_stops_before_next_stage(trial, kind):
    launch, _ = trial
    slurm = FakeSlurm(failed_stage='confirm' if kind == 'bad_gate' else None,
                      missing_stage='confirm' if kind == 'missing_shard' else None)
    assert overnight.Controller(launch, slurm).run() == 1
    assert [e for e in slurm.events if e[0] == 'submit'] == [('submit', 'confirm')]
    state = json.loads((Path(launch['out']) / 'status.json').read_bytes())
    assert state['status'] == 'stopped'
    assert not (Path(launch['out']) / 'comparison').exists()


def test_failed_abaca_or_incomplete_scheduler_never_submits(trial):
    launch, _ = trial
    slurm = FakeSlurm()
    slurm.array_state = lambda *_: ('failed', {'failed_tasks': [3]})
    assert overnight.Controller(launch, slurm).run() == 1
    assert slurm.jobs == {}
    assert not (Path(launch['out']) / 'successor_plan.json').exists()


def test_bad_followup_raw_record_cannot_be_hidden_by_pass_flags(trial):
    launch, protocol = trial
    overnight.bridge(launch, protocol)
    path = Path(launch['out']) / 'bridge/abaca_check.json'
    value = json.loads(path.read_bytes())
    value['cases']['cal_abaca']['records'][0]['estimate']['log_evidence'] += .01
    write(path, value)
    slurm = FakeSlurm()
    assert overnight.Controller(launch, slurm).run() == 1
    assert slurm.jobs == {}


def test_nested_stats_and_pretended_gate_are_rejected(trial):
    launch, protocol = trial
    plan = overnight.successor(launch, protocol, overnight.bridge(launch, protocol))
    path = Path(launch['out']) / 'confirm.json'
    value = short_receipt(cluster.read_plan(plan, 'confirm'))
    value = json.loads(json.dumps(value))
    value['cases']['cal_abaca']['statistics']['repetitions'] = 900
    write(path, value)
    with pytest.raises(ValueError, match='Nested'):
        overnight.strict_receipt(plan, 'confirm', path)
    value = short_receipt(cluster.read_plan(plan, 'confirm'), {'cal_abaca': .02})
    value['gate']['status'] = 'passed_exact_short'
    write(path, value)
    with pytest.raises(ValueError, match='gate'):
        overnight.strict_receipt(plan, 'confirm', path)


def test_ambiguous_submission_is_not_retried(trial):
    launch, protocol = trial
    plan = overnight.successor(launch, protocol, overnight.bridge(launch, protocol))
    slurm = FakeSlurm()
    controller = overnight.Controller(launch, slurm)
    def interrupted(command):
        raise TimeoutError('lost sbatch response')
    slurm.submit = interrupted
    with pytest.raises(TimeoutError):
        controller.submit_stage(protocol, plan, 'confirm')
    resumed = overnight.Controller(launch, FakeSlurm())
    with pytest.raises(ValueError, match='uncertain previous submission'):
        resumed.submit_stage(protocol, plan, 'confirm')
    assert resumed.slurm.jobs == {}


def test_saved_child_is_reused_after_coordinator_deadline(trial):
    launch, protocol = trial
    plan = overnight.successor(launch, protocol, overnight.bridge(launch, protocol))
    slurm = FakeSlurm()
    controller = overnight.Controller(launch, slurm)
    job = controller.submit_stage(protocol, plan, 'confirm')
    resumed = overnight.Controller(launch, slurm)
    assert resumed.submit_stage(protocol, plan, 'confirm') == job
    assert len(slurm.jobs) == 1
    ticks = iter([0, 101])
    expired = overnight.Controller(launch, slurm, clock=lambda: next(ticks))
    with pytest.raises(ValueError, match='deadline'):
        expired.wait({'confirm': job})


def test_environment_removes_inherited_scheduler_options(monkeypatch):
    monkeypatch.setenv('SBATCH_DEPENDENCY', 'afterok:old')
    monkeypatch.setenv('SBATCH_MEM_PER_CPU', '100G')
    monkeypatch.setenv('SLURM_JOB_ID', 'old')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '17')
    monkeypatch.setenv('PATH', '/safe/bin')
    env = overnight.clean_environment('/prepared/python')
    assert not any(key.startswith(('SBATCH_', 'SLURM_')) for key in env)
    assert env['PATH'] == '/safe/bin' and env['SHARED_BASE_VENV'] == '/prepared/python'


@pytest.mark.parametrize('rows,expected', [
    ('12_0|COMPLETED|0:0\n12_1|COMPLETED|0:0\n', 'success'),
    ('12_0|COMPLETED|0:0\n12_1|RUNNING|0:0\n', 'active'),
    ('12_0|COMPLETED|0:0\n12_1|FAILED|1:0\n', 'failed'),
    ('12_0|COMPLETED|0:0\n', 'accounting_pending'),
])
def test_scheduler_requires_every_array_task_and_zero_exit(rows, expected):
    slurm = overnight.Slurm('/unused')
    slurm.run = lambda cmd: SimpleNamespace(stdout=rows if cmd[0] == 'sacct' else '')
    assert slurm.array_state('12', 2)[0] == expected


def test_dry_run_writes_nothing_and_uses_afterany(tmp_path, capsys):
    out = tmp_path / 'preview'
    assert launcher.main(['--after-job', '66543429', '--out', str(out)]) == 0
    text = capsys.readouterr().out
    assert '--dependency=afterany:66543429' in text and '--kill-on-invalid-dep=yes' in text
    assert 'pilot16' in text and not out.exists()
    with pytest.raises(SystemExit):
        launcher.main(['--after-job', '66543429;bad', '--out', str(out)])
    assert not out.exists()


def test_frozen_implementation_drift_rejected(trial):
    launch, _ = trial
    altered = copy.deepcopy(launch)
    altered['code_sha256']['code/transformer/shared_base_evidence_smc.py'] = 'changed'
    altered['launch_id'] = overnight.fingerprint({k: v for k, v in altered.items() if k != 'launch_id'})
    with pytest.raises(ValueError, match='implementation changed'):
        overnight.validate_launch(altered)


def test_output_lock_is_exclusive(tmp_path):
    with overnight.lock(tmp_path / 'lock'):
        with pytest.raises(RuntimeError):
            with overnight.lock(tmp_path / 'lock'):
                pass


@pytest.mark.parametrize('failed_stage', ['alpha_refine', 'row_refine'])
def test_either_refinement_failure_blocks_pilot(trial, failed_stage):
    launch, _ = trial
    slurm = FakeSlurm(failed_stage=failed_stage)
    assert overnight.Controller(launch, slurm).run() == 1
    assert [e for e in slurm.events if e[0] == 'submit'] == [
        ('submit', 'confirm'), ('submit', 'alpha_refine'), ('submit', 'row_refine')]
    assert not (Path(launch['out']) / 'shards/pilot16').exists()


def test_resume_after_one_parallel_receipt_was_saved(trial, monkeypatch):
    launch, _ = trial
    import compare_shared_base_cluster as comparison
    monkeypatch.setattr(comparison, 'plot', lambda report, path: path.write_bytes(b'test figure'))
    class PauseOnce(overnight.Controller):
        paused_once = False
        def save(self, status, reason=None):
            super().save(status, reason)
            if (not self.paused_once and status == 'running' and 'alpha_refine' in self.state['receipts']
                    and 'row_refine' not in self.state['receipts']):
                self.paused_once = True
                raise ValueError('Controller deadline; test interruption after first parallel receipt')
    slurm = FakeSlurm()
    assert PauseOnce(launch, slurm).run() == 75
    state = json.loads((Path(launch['out']) / 'status.json').read_bytes())
    assert state['status'] == 'paused'
    assert set(state['receipts']) == {'confirm', 'alpha_refine'}
    assert len(slurm.jobs) == 3
    assert overnight.Controller(launch, slurm).run() == 0
    assert [e for e in slurm.events if e[0] == 'submit'] == [
        ('submit', 'confirm'), ('submit', 'alpha_refine'), ('submit', 'row_refine'), ('submit', 'pilot16')]


def test_failed_followup_gate_is_recorded_and_does_not_create_successor(trial):
    launch, protocol = trial
    tasks = [dict(case_id='cal_abaca', replica_start=start, replica_stop=start+4,
                  output=str(Path(launch['followup_shards']) / f'{start}.json')) for start in range(0, 128, 4)]
    shards(protocol['followup_plan'], 'abaca_check', tasks, Path(launch['out']), failure=True)
    slurm = FakeSlurm()
    assert overnight.Controller(launch, slurm).run() == 1
    assert slurm.jobs == {}
    out = Path(launch['out'])
    assert not (out / 'successor_plan.json').exists()
    saved = json.loads((out / 'bridge/review.json').read_bytes())
    assert saved['all_six_controls_meet_threshold_with_followup'] is False
    assert saved['original_calibration_gate_preserved'] == 'failed_exact_short'


@pytest.mark.parametrize('completed', [True, False])
def test_submission_handles_completed_and_running_abaca(trial, tmp_path, monkeypatch, completed):
    launch, _ = trial
    for path in (Path(launch['venv']) / 'bin/python', Path(launch['store']) / 'anchors.json',
                 Path(launch['store']) / 'manifest.json', Path(launch['src']) / 'product_model_with_memory/universal_tables.py'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    commands = []
    class SubmissionSlurm:
        def __init__(self, venv):
            pass
        def array_state(self, job, tasks):
            assert job == '66543429' and tasks == 32
            return ('success' if completed else 'active'), {}
        def submit(self, command):
            commands.append(command)
            return '7777'
    monkeypatch.setattr(overnight, 'Slurm', SubmissionSlurm)
    args = ['--submit', '--after-job', '66543429']
    for key in ('out', 'original_receipt', 'followup_shards', 'store', 'src', 'venv'):
        args += ['--'+key.replace('_', '-'), launch[key]]
    assert launcher.main(args) == 0
    assert len(commands) == 1
    assert any(part.startswith('--dependency=') for part in commands[0]) is not completed
    saved = json.loads((Path(launch['out']) / 'submission.json').read_bytes())
    assert saved['job_id'] == '7777'
    # A second ordinary invocation cannot duplicate the coordinator.
    with pytest.raises(SystemExit):
        launcher.main(args)
    assert len(commands) == 1
