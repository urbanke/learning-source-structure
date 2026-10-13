"""Launch orchestration checks: no scheduler or numerical experiment is invoked."""
from argparse import Namespace
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launcher = module('launch_shared_base')
dispatcher = module('dispatch_shared_base')


def args(tmp_path, **changes):
    values = dict(plan=launcher.ROOT/'code/transformer/configs/shared_base_cluster_plan_2026-09-16.json',
                  out=tmp_path/'out', stage=None, gpu_benchmark=False, merge=False,
                  store=tmp_path/'store', src=tmp_path/'src', venv=tmp_path/'venv',
                  concurrency=None, wall_minutes=None, seconds=None, replicas_per_shard=4,
                  only_task=None, account=None)
    values.update(changes)
    return Namespace(**values)


def receipt(plan, plan_hash, stage):
    config = plan['stages'][stage]
    count = len(config['case_ids'])*config['replicates']
    if config['purpose'] == 'preflight':
        gate = dict(status='preflight_only', runtime_ready=True)
    elif config['purpose'] == 'repeatability':
        gate = dict(status='repeatability_only', precision_passed=True,
                    precision_met_by_case={cid: True for cid in config['case_ids']})
    else:
        gate = dict(status='passed_exact_short', exact_short_gate_passed=True,
                    passed_by_case={cid: True for cid in config['case_ids']})
    return dict(schema='shared-base-cluster-merge-v1', plan_sha256=plan_hash, stage=stage,
                stage_config=config, complete=True, expected_records=count, observed_records=count,
                coverage={cid: list(range(config['replicates'])) for cid in config['case_ids']},
                case_summaries={cid: {} for cid in config['case_ids']}, gate=gate)


def save_receipt(out, value):
    path = out/'merged'/f"{value['stage']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_dry_run_never_writes_or_calls_slurm(tmp_path):
    out = tmp_path/'absent'
    result = subprocess.run([sys.executable, str(HERE/'launch_shared_base.py'), '--out', str(out)],
                            check=True, capture_output=True, text=True)
    assert '"action": "dry run"' in result.stdout
    assert '--array=0-5%1' in result.stdout
    assert not out.exists()


def test_submission_cwd_is_checkout_even_from_another_directory(tmp_path, monkeypatch):
    settings = args(tmp_path)
    for file in (settings.store/'anchors.json', settings.store/'manifest.json',
                 settings.src/'product_model_with_memory/universal_tables.py', settings.venv/'bin/python'):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('test fixture')
    called = []
    def fake_submit(command, **kwargs):
        called.append((command, kwargs))
        return Namespace(returncode=0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher.subprocess, 'run', fake_submit)
    monkeypatch.setattr(sys, 'argv', ['launch', '--submit', '--out', str(settings.out),
                                    '--store', str(settings.store), '--src', str(settings.src),
                                    '--venv', str(settings.venv)])
    with pytest.raises(SystemExit) as result:
        launcher.main()
    assert result.value.code == 0 and len(called) == 1
    assert called[0][0][0] == 'sbatch'
    assert called[0][1]['cwd'] == launcher.ROOT


def test_all_cases_and_replicas_covered_exactly_once(tmp_path):
    for stage in ('preflight', 'calibrate', 'confirm', 'alpha_refine', 'row_refine', 'pilot16', 'pilot64', 'pilot256', 'pilot1024'):
        launch = launcher.build_launch(args(tmp_path, stage=stage))
        config = launch['plan']['stages'][stage]
        covered = [(task['case_id'], replica) for task in launch['payload']['tasks']
                   for replica in range(task['replica_start'], task['replica_stop'])]
        assert len(covered) == len(set(covered))
        assert set(covered) == {(cid, replica) for cid in config['case_ids'] for replica in range(config['replicates'])}
        assert launch['concurrency'] == (1 if stage == 'preflight' else 64)
        assert len({task['output'] for task in launch['payload']['tasks']}) == len(launch['payload']['tasks'])


def test_preflight_missing_failed_or_incomplete_blocks_calibration(tmp_path):
    settings = args(tmp_path, stage='calibrate')
    first = launcher.build_launch(settings)
    assert first['errors']
    valid = receipt(first['plan'], first['payload']['plan_sha256'], 'preflight')
    save_receipt(settings.out, valid)
    assert not launcher.build_launch(settings)['errors']
    for key, value in (('plan_sha256', 'other'), ('complete', False), ('observed_records', 1), ('coverage', {})):
        bad = copy.deepcopy(valid)
        bad[key] = value
        save_receipt(settings.out, bad)
        assert launcher.build_launch(settings)['errors']
    bad = copy.deepcopy(valid)
    bad['gate']['runtime_ready'] = False
    save_receipt(settings.out, bad)
    assert launcher.build_launch(settings)['errors']


def test_every_short_confirmation_and_refinement_is_required(tmp_path):
    settings = args(tmp_path, stage='pilot16')
    first = launcher.build_launch(settings)
    assert len(first['errors']) == 4
    for name in ('calibrate', 'confirm', 'alpha_refine', 'row_refine'):
        save_receipt(settings.out, receipt(first['plan'], first['payload']['plan_sha256'], name))
    assert not launcher.build_launch(settings)['errors']
    bad = receipt(first['plan'], first['payload']['plan_sha256'], 'row_refine')
    bad['gate']['passed_by_case'][next(iter(bad['gate']['passed_by_case']))] = False
    save_receipt(settings.out, bad)
    assert len(launcher.build_launch(settings)['errors']) == 1


def test_longer_pilot_requires_previous_precision(tmp_path):
    settings = args(tmp_path, stage='pilot64')
    first = launcher.build_launch(settings)
    valid = receipt(first['plan'], first['payload']['plan_sha256'], 'pilot16')
    save_receipt(settings.out, valid)
    assert not launcher.build_launch(settings)['errors']
    valid['gate']['precision_passed'] = False
    save_receipt(settings.out, valid)
    assert launcher.build_launch(settings)['errors']


def test_partial_resubmission_keeps_global_replica_identity(tmp_path):
    full = launcher.build_launch(args(tmp_path, stage='calibrate'))
    partial = launcher.build_launch(args(tmp_path, stage='calibrate', only_task=[1, 5]))
    assert partial['payload'] == full['payload']
    assert '--array=1,5%64' in partial['command']
    command = dispatcher.task_command(partial['payload'], 1)
    assert command[command.index('--replica-start')+1] == '4'
    assert command[command.index('--replica-stop')+1] == '8'
    assert '--receipts' in command


@pytest.mark.parametrize('changes', [dict(concurrency=65), dict(concurrency=2), dict(seconds=0),
                                     dict(wall_minutes=0), dict(seconds=899), dict(only_task=[99])])
def test_invalid_resource_or_shard_request_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        launcher.build_launch(args(tmp_path, **changes))


def test_gpu_is_cuda_only_separate_job(tmp_path):
    launch = launcher.build_launch(args(tmp_path, gpu_benchmark=True))
    command = dispatcher.task_command(launch['payload'], 0)
    assert command[command.index('--device')+1] == 'cuda'
    assert not any(item.startswith('--array') for item in launch['command'])
    assert launch['command'][-2].endswith('shared_base_gpu_benchmark.sbatch')
    assert not launch['errors']


def test_frozen_plan_and_dispatch_identity_checked(tmp_path):
    plan = tmp_path/'plan.json'
    plan.write_bytes(args(tmp_path).plan.read_bytes())
    launch = launcher.build_launch(args(tmp_path, plan=plan))
    manifest = tmp_path/'launch.json'
    manifest.write_text(json.dumps(launch['payload']))
    assert dispatcher.read_launch(manifest)['launch_id'] == launch['payload']['launch_id']
    plan.write_bytes(plan.read_bytes()+b' ')
    with pytest.raises(ValueError, match='Plan changed'):
        dispatcher.read_launch(manifest)
    bad = copy.deepcopy(launch['payload'])
    bad['seconds'] += 1
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Launch identity'):
        dispatcher.read_launch(manifest)


def test_timeout_exit_and_partial_output_preserved(tmp_path, monkeypatch):
    launch = launcher.build_launch(args(tmp_path))['payload']
    output = Path(launch['tasks'][0]['output'])
    def fake_run(command, **kwargs):
        output.write_text(json.dumps(dict(complete=False, records=[{}], runtime_resources={'compiled_kernel_available': True})))
        return Namespace(returncode=75)
    monkeypatch.setattr(dispatcher.platform, 'platform', lambda: 'test-node')
    monkeypatch.setattr(dispatcher.subprocess, 'run', fake_run)
    assert dispatcher.execute(launch, 0) == 75
    report = json.loads(next(Path(launch['runtime_directory']).glob('*.json')).read_text())
    assert report['status'] == 'deadline'
    assert report['completed_records'] == 1
    assert report['numerical_complete'] is False
    assert output.exists()


def test_shell_syntax_and_memory_requests():
    for path in HERE.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    for path in HERE.glob('*.sbatch'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    cpu = (HERE/'shared_base_cpu.sbatch').read_text()
    gpu = (HERE/'shared_base_gpu_benchmark.sbatch').read_text()
    assert '--partition=standard' in cpu and '--qos=serial' in cpu
    assert '--gres=gpu' not in cpu and '--mem=6G' in cpu
    assert '--partition=h100' in gpu and '--qos=debug' in gpu
    assert '--cpus-per-task=8' in gpu and '--mem=32G' in gpu
    assert '--no-requeue' in cpu and '--no-requeue' in gpu
