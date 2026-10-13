"""ABACA-only follow-up: frozen science, fresh seeds, bounded manual routing."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
import shared_base_cluster as cluster

_spec = importlib.util.spec_from_file_location('abaca_parent_launcher', ROOT / 'code/transformer/scitas/launch_shared_base.py')
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)
FOLLOWUP = ROOT / 'code/transformer/configs/shared_base_abaca_plan_2026-09-16.json'
ORIGINAL = ROOT / 'code/transformer/configs/shared_base_cluster_plan_2026-09-16.json'
ORIGINAL_HASH = '4cb87e20efb79acc22df99c88631fbab39e089cc82e19591bef647730baf8331'
WRAPPER = Path('code/transformer/scitas/submit_shared_base_abaca.sh')


def args(tmp_path, **updates):
    values = dict(plan=FOLLOWUP, out=tmp_path / 'output', stage='abaca_check',
                  gpu_benchmark=False, merge=False, concurrency=32, wall_minutes=None,
                  seconds=None, replicas_per_shard=4, store=tmp_path / 'store', src=tmp_path / 'src',
                  only_task=None, account=None, venv=Path(sys.prefix))
    values.update(updates)
    return SimpleNamespace(**values)


def synthetic_shard(context, case_id, *, start=0, stop=None, log_offset=0.):
    """Receipt-validator fixture only; does not sample a numerical reference."""
    cfg = context['config']
    stop = cfg['replicates'] if stop is None else stop
    numerical = dict(compiled_kernel_available=True, code_sha256={'fixture': 'test-only'})
    ready = dict(compiled_kernel_available=True, store_sealed=True, store_read_only=True)
    shard = cluster.new_shard(context, case_id, numerical, start, stop, ready)
    for replica in range(start, stop):
        estimate = dict(log_evidence=cluster.known_evidence(context, case_id) + log_offset,
                        seed=cluster.expected_seed(context, case_id, replica), sampler_class='SharedBaseMCMC',
                        particles=cfg['particles'], temperatures=cfg['temperatures'], moves=cfg['moves'],
                        n_tokens=cfg['n_tokens'], elapsed_seconds=0.)
        cluster.append_record(shard, context, replica, estimate)
    shard.update(complete=True, status='completed')
    return shard


def test_frozen_followup_changes_only_case_budget_and_seed():
    old = cluster.read_plan(ORIGINAL, 'calibrate')
    new = cluster.read_plan(FOLLOWUP, 'abaca_check')
    plan = new['plan']
    assert cluster.file_hash(ORIGINAL) == ORIGINAL_HASH == plan['followup_of']['original_plan_sha256']
    assert ROOT / plan['followup_of']['original_plan'] == ORIGINAL
    assert plan['followup_of']['original_stage'] == 'calibrate'
    assert plan['prior'] == old['plan']['prior'] == cluster.SUPPORTED_PRIOR
    assert plan['acceptance'] == old['plan']['acceptance']
    assert plan['dependency_manifest_sha256'] == old['plan']['dependency_manifest_sha256']
    assert plan['cases'] == [old['cases']['cal_abaca']]
    assert plan['cases'][0]['tokens'] == [0, 1, 0, 2, 0]
    assert plan['cases'][0]['known_log_evidence'] == -17.70436765400922
    assert plan['production_reference_approved'] is False
    assert set(plan['stages']) == {'abaca_check'}
    expected = dict(old['config'], purpose='confirmation', requires=[], case_ids=['cal_abaca'],
                    particles=16384, replicates=128, seed=2026091650)
    assert new['config'] == expected


def test_all_followup_seeds_unique_and_disjoint_from_all_old_cpu_stages(tmp_path):
    new = cluster.read_plan(FOLLOWUP, 'abaca_check')
    seeds = {cluster.expected_seed(new, 'cal_abaca', r) for r in range(128)}
    assert len(seeds) == 128
    old_plan = json.loads(ORIGINAL.read_bytes())
    assert new['config']['seed'] not in {cfg['seed'] for cfg in old_plan['stages'].values()}
    for stage_name in old_plan['stages']:
        context = cluster.read_plan(ORIGINAL, stage_name)
        old_seeds = {cluster.expected_seed(context, case_id, r)
                     for case_id in context['config']['case_ids']
                     for r in range(context['config']['replicates'])}
        assert seeds.isdisjoint(old_seeds)
    gpu = launch.build_launch(args(tmp_path, plan=ORIGINAL, stage=None, gpu_benchmark=True))
    assert gpu['payload']['gpu_settings']['seed'] != new['config']['seed']


def test_shards_cover_each_fresh_repetition_once_and_never_promote(tmp_path):
    context = cluster.read_plan(FOLLOWUP, 'abaca_check')
    result = launch.build_launch(args(tmp_path))
    assert result['errors'] == [] and result['concurrency'] == 32
    assert '--array=0-31%32' in result['command']
    assert result['payload']['stage'] == 'abaca_check'
    tasks = result['payload']['tasks']
    assert len(tasks) == 32 and {task['case_id'] for task in tasks} == {'cal_abaca'}
    indices = [r for task in tasks for r in range(task['replica_start'], task['replica_stop'])]
    assert indices == list(range(128))
    assert all(task['replica_stop'] - task['replica_start'] == 4 for task in tasks)
    shards = [synthetic_shard(context, 'cal_abaca', start=t['replica_start'], stop=t['replica_stop']) for t in tasks]
    receipt, code = cluster.merge_reports(context, shards)
    assert code == 0 and receipt['complete'] and receipt['no_auto_advance']
    assert receipt['coverage'] == {'cal_abaca': list(range(128))}
    assert receipt['gate']['exact_short_gate_passed']
    assert receipt['gate']['accuracy_approved'] is False and receipt['gate']['production_approved'] is False
    with pytest.raises(ValueError, match='duplicate'):
        cluster.merge_reports(context, shards + shards[:1])
    partial, code = cluster.merge_reports(context, shards[:-1])
    assert code == 75 and len(partial['missing']) == 4
    assert not (tmp_path / 'output').exists()


@pytest.fixture
def isolated_wrapper(tmp_path):
    """Copy launcher inputs so even dry-run write regressions stay in tmp."""
    target = tmp_path / 'repo'
    built = launch.build_launch(args(tmp_path))
    files = set(built['payload']['code_sha256'])
    files.update(str(p.relative_to(ROOT)) for p in (FOLLOWUP, ORIGINAL))
    files.update([str(WRAPPER), 'code/transformer/scitas/submit_shared_base.sh',
                  'code/transformer/scitas/launch_shared_base.py'])
    for name in files:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    env = dict(os.environ, SHARED_BASE_LAUNCH_PYTHON=sys.executable,
               SHARED_BASE_VENV=str(tmp_path / 'venv'), SHARED_BASE_ACCOUNT='unit-test')
    return target, env


def invoke(isolated, *options):
    root, env = isolated
    return subprocess.run(['bash', str(root / WRAPPER), *options], env=env, cwd=root,
                          capture_output=True, text=True, timeout=10)


def test_wrapper_default_dry_run_is_read_only_and_routes_original_cpu_launcher(isolated_wrapper):
    root, _ = isolated_wrapper
    before = {str(p.relative_to(root)): cluster.file_hash(p) for p in root.rglob('*') if p.is_file()}
    result = invoke(isolated_wrapper)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    info = json.loads(lines[0])
    assert info == dict(mode='cpu', requested_stage='abaca_check', tasks=32, concurrency=32,
                        prerequisite_errors=[], action='dry run')
    command = shlex.split(lines[1])
    assert command[0] == 'sbatch' and '--array=0-31%32' in command
    assert str(root / 'code/transformer/scitas/shared_base_cpu.sbatch') in command
    assert 'scitas_shared_base_abaca_2026-09-16/launches/abaca_check/' in command[-1]
    assert not (root / 'output').exists()
    assert before == {str(p.relative_to(root)): cluster.file_hash(p) for p in root.rglob('*') if p.is_file()}


def test_merge_and_review_paths_are_explicit_and_read_only(isolated_wrapper):
    root, _ = isolated_wrapper
    merged = invoke(isolated_wrapper, '--merge')
    assert merged.returncode == 0, merged.stderr
    assert json.loads(merged.stdout.splitlines()[0])['mode'] == 'merge'
    command = shlex.split(merged.stdout.splitlines()[1])
    assert command[1] == str(root / 'code/scripts/shared_base_cluster.py') and command[2] == 'merge'
    assert command[command.index('--stage') + 1] == 'abaca_check'
    assert command[command.index('--shards') + 1].endswith('/shards/abaca_check')
    assert command[command.index('--out') + 1].endswith('/merged/abaca_check.json')
    reviewed = invoke(isolated_wrapper, '--review')
    assert reviewed.returncode == 0, reviewed.stderr
    command = shlex.split(reviewed.stdout.strip())
    assert command[1] == str(root / 'code/scripts/review_shared_base_abaca.py')
    assert command[command.index('--original-receipt') + 1].endswith('/scitas_shared_base_2026-09-16/merged/calibrate.json')
    assert command[command.index('--followup-receipt') + 1].endswith('/scitas_shared_base_abaca_2026-09-16/merged/abaca_check.json')
    assert command[command.index('--out') + 1].endswith('/scitas_shared_base_abaca_2026-09-16/comparison.json')
    assert not (root / 'output').exists()


@pytest.mark.parametrize('options', [
    ['--stage', 'confirm'], ['--out', '/unused'], ['--merge', '--review'],
    ['--merge', '--only-task', '0'], ['--review', '--wall-minutes', '10'],
    ['--only-task'], ['--only-task', '32'], ['--wall-minutes', '0'],
    ['--submit', '--dry-run'],
])
def test_invalid_wrapper_arguments_fail_without_writes(isolated_wrapper, options):
    result = invoke(isolated_wrapper, *options)
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert not (isolated_wrapper[0] / 'output').exists()


def test_selected_task_keeps_global_replica_range(isolated_wrapper):
    result = invoke(isolated_wrapper, '--only-task', '31', '--wall-minutes', '30')
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout.splitlines()[0])
    assert info['tasks'] == 1 and info['concurrency'] == 32
    command = shlex.split(result.stdout.splitlines()[1])
    assert '--array=31%32' in command and '--time=00:30:00' in command


def test_old_failed_calibration_still_blocks_original_confirmation(tmp_path):
    old = cluster.read_plan(ORIGINAL, 'calibrate')
    old['prerequisite_receipts'] = {'preflight': 'unit-test-prerequisite'}
    reports = [synthetic_shard(old, case_id, log_offset=-.02 if case_id == 'cal_abaca' else 0.)
               for case_id in old['config']['case_ids']]
    failed, code = cluster.merge_reports(old, reports)
    assert code == 0 and failed['complete'] and not failed['gate']['exact_short_gate_passed']
    assert sum(failed['gate']['passed_by_case'].values()) == 5
    confirm = cluster.read_plan(ORIGINAL, 'confirm')
    with pytest.raises(ValueError, match='calibration gate failed'):
        cluster.validate_prerequisite_receipt(confirm, 'calibrate', failed)
    directory = tmp_path / 'original'
    (directory / 'merged').mkdir(parents=True)
    (directory / 'merged/calibrate.json').write_text(json.dumps(failed))
    blocked = launch.build_launch(args(tmp_path, plan=ORIGINAL, stage='confirm', out=directory, concurrency=None))
    assert len(blocked['errors']) == 1 and 'gate has not passed' in blocked['errors'][0]
    assert '--array=0-191%64' in blocked['command']
    preflight = launch.build_launch(args(tmp_path, plan=ORIGINAL, stage='preflight', concurrency=None))
    assert preflight['concurrency'] == 1 and '--array=0-5%1' in preflight['command']
    fresh = launch.build_launch(args(tmp_path))
    assert fresh['errors'] == []  # standalone diagnostic does not alter the old gate
    assert json.loads((directory / 'merged/calibrate.json').read_text()) == failed
    assert cluster.file_hash(ORIGINAL) == ORIGINAL_HASH
