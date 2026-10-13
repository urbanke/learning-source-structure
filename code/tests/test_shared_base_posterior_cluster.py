"""Portable posterior pilot: immutable inputs, bounded execution and full coverage."""
import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_shared_base_posterior_cluster as prep
import shared_base_posterior_cluster as runner


@pytest.fixture
def package(tmp_path):
    source = tmp_path/'source'
    files = {}
    for name in prep.input_names():
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = dict(sha256=prep.sha(path), size_bytes=path.stat().st_size)
    archive = tmp_path/'inputs.tar.gz'
    prep.pack_files(source, archive, files)
    plan = dict(schema='shared-base-posterior-cluster-plan-v1', tasks=prep.task_list(), worker=prep.WORKER,
        resources=dict(cpus=2,memory_gib=12,wall_seconds=28800,worker_seconds=27900,concurrency=32),
        code_sha256={name: prep.sha(prep.ROOT/name) for name in prep.bound_code()},
        dependency_manifest=prep.DEPENDENCY_MANIFEST,
        dependency_manifest_sha256=prep.sha(prep.ROOT/prep.DEPENDENCY_MANIFEST),
        inputs=dict(root_directory=prep.BUNDLE_ROOT, files=files, archive_sha256=prep.sha(archive)),
        automatic_successors=False, production_approved=False)
    path = tmp_path/'plan.json'
    prep.atomic_json(path, plan)
    return dict(source=source, archive=archive, plan=plan, path=path, base=tmp_path)


def arguments(package, **updates):
    values = dict(plan=package['path'], inputs=package['source'], dependency_root=package['base']/'dependency',
                  out=package['base']/'output', venv=package['base']/'venv', concurrency=32,
                  only_task=None, preflight=False, submit=False)
    values.update(updates)
    return argparse.Namespace(**values)


def test_deterministic_archive_and_hash_preserving_roundtrip(package):
    p = package
    other = p['base']/'second.tar.gz'
    prep.pack_files(p['source'], other, p['plan']['inputs']['files'])
    assert prep.sha(other) == prep.sha(p['archive'])
    extracted = prep.unpack(p['archive'], p['base']/'unpacked', p['plan'])
    assert prep.verify_inputs(extracted, p['plan']) == 48
    with pytest.raises(ValueError, match='exists'):
        prep.unpack(p['archive'], p['base']/'unpacked', p['plan'])
    next(extracted.rglob('*.json')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Frozen input mismatch'):
        prep.verify_inputs(extracted, p['plan'])


def test_unsafe_archive_rejected_even_with_matching_archive_hash(package):
    p = package
    bad = p['base']/'bad.tar.gz'
    first = next(iter(p['plan']['inputs']['files']))
    with tarfile.open(bad, 'w:gz') as tar:
        for name, meta in p['plan']['inputs']['files'].items():
            entry = tarfile.TarInfo(f'{prep.BUNDLE_ROOT}/{name}')
            entry.size = meta['size_bytes']
            if name == first:
                entry.type = tarfile.SYMTYPE
                entry.linkname = '/tmp/escape'
                tar.addfile(entry)
            else:
                tar.addfile(entry, io.BytesIO(name.encode()))
    plan = copy.deepcopy(p['plan'])
    plan['inputs']['archive_sha256'] = prep.sha(bad)
    with pytest.raises(ValueError, match='Unsafe'):
        prep.unpack(bad, p['base']/'bad-unpack', plan)
    assert not (p['base']/'bad-unpack'/prep.BUNDLE_ROOT).exists()


@pytest.mark.parametrize('change', ['task','budget','code','dependency','production'])
def test_fixed_contract_changes_rejected(package, change):
    plan = copy.deepcopy(package['plan'])
    if change == 'task':
        plan['tasks'] = plan['tasks'][:-1]
    elif change == 'budget':
        plan['worker'] = dict(plan['worker'], max_draws=524288)
    elif change == 'code':
        plan['code_sha256']['code/scripts/shared_base_curve_reference.py'] = '0'*64
    elif change == 'dependency':
        plan['dependency_manifest_sha256'] = '0'*64
    else:
        plan['production_approved'] = True
    prep.atomic_json(package['path'], plan)
    with pytest.raises(ValueError):
        prep.read_plan(package['path'])


def test_dry_run_from_other_directory_writes_nothing(package, monkeypatch):
    args = arguments(package)
    monkeypatch.chdir(package['base'])
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('Unexpected submission'))
    runner.submit(args)
    assert not args.out.exists()
    _, _, _, command = runner.build_launch(args)
    assert '--array=0-31%32' in command and '--time=08:00:00' in command


def test_submission_uses_repository_cwd_and_frozen_manifest(package, monkeypatch):
    args = arguments(package, submit=True)
    (args.venv/'bin').mkdir(parents=True)
    (args.venv/'bin/python').touch()
    monkeypatch.chdir(package['base'])
    monkeypatch.setattr(runner, 'verify_dependencies', lambda *a: None)
    calls = []
    def submit(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command,0,'12345\n','')
    monkeypatch.setattr(runner.subprocess, 'run', submit)
    runner.submit(args)
    assert len(calls) == 1 and calls[0][1]['cwd'] == prep.ROOT
    manifest = next((args.out/'launches').glob('*.json'))
    launch = json.loads(manifest.read_text())
    assert launch['launch_id'] == prep.fingerprint({k:v for k,v in launch.items() if k != 'launch_id'})
    assert launch['campaign']['inputs'] == str(args.inputs.resolve())


def test_worker_arguments_preserve_posterior_and_full_fixed_coverage(package):
    _, launch, _, _ = runner.build_launch(arguments(package))
    tasks = package['plan']['tasks']
    assert len(tasks) == len({t['id'] for t in tasks}) == 32
    assert {(t['depth'],t['index'],t['time'],t['order']) for t in tasks} == {
        (d,i,t,a) for d in (0,1,4,8) for i in (0,1) for t in (256,1024) for a in (16,32)}
    command = runner.worker_command(launch['campaign'], tasks[-1])
    assert command[1].endswith('shared_base_curve_reference.py')
    assert command[command.index('--max-draws')+1] == '262144'
    assert command[command.index('--seed')+1] == '2026091670'
    assert '--particles' not in command


def test_timeout_preserves_existing_checkpoint(tmp_path):
    marker = tmp_path/'checkpoint'
    marker.write_text('last atomic state')
    command = [sys.executable, '-c', 'import time; time.sleep(10)']
    began = time.monotonic()
    result = runner.run_child(command, tmp_path/'worker.log', seconds=.05)
    assert result == 75 and time.monotonic()-began < 3
    assert marker.read_text() == 'last atomic state'


def test_empty_summary_is_incomplete_not_success(package, monkeypatch):
    args = arguments(package)
    _, launch, _, _ = runner.build_launch(args)
    prep.atomic_json(args.out/'campaign.json', launch['campaign'])
    monkeypatch.setattr(runner, 'verify_dependencies', lambda *a: None)
    result = runner.summarize(args.plan, args.out)
    assert not result['complete'] and not result['all_numerical_checks_passed']
    assert len(result['rows']) == 32 and len(result['groups']) == 8
    assert all(group['numerical_audit']['reported_mean_bayes_regret_bits'] is None for group in result['groups'])
    assert not result['production_approved']


def test_numerical_perturbation_rejected_before_backend(tmp_path, monkeypatch):
    monkeypatch.setenv('PMM_PHI_BIAS', '.01')
    with pytest.raises(ValueError, match='perturbation'):
        runner.runtime_setup(tmp_path/'nonexistent-dependency')


def test_task_audit_raw_corruption_rejected(package, monkeypatch):
    args = arguments(package)
    _, launch, _, _ = runner.build_launch(args)
    campaign = launch['campaign']
    prep.atomic_json(args.out/'campaign.json', campaign)
    task = package['plan']['tasks'][0]
    paths = runner.task_paths(campaign,task)
    for key in ('reference','checkpoint','moments'):
        paths[key].parent.mkdir(parents=True,exist_ok=True)
        paths[key].write_text('raw')
    audit = dict(task=task,plan_sha256=campaign['plan_sha256'],
        audit_code_sha256=package['plan']['code_sha256']['code/scripts/shared_base_curve_audit.py'],
        raw_sha256={str(paths[key]):prep.sha(paths[key]) for key in ('reference','checkpoint','moments')},
        result=dict(index=task['index'],time=task['time'],order=task['order'],status='needs_more_draws'))
    prep.atomic_json(paths['audit'],audit)
    prep.atomic_json(paths['receipt'],dict(identity=dict(plan_sha256=campaign['plan_sha256'],task=task,
        deployment_sha256=prep.fingerprint(campaign)),runtime={},status='unresolved',
        elapsed_seconds=1,audit_sha256=prep.sha(paths['audit'])))
    paths['checkpoint'].write_text('modified after audit')
    monkeypatch.setattr(runner,'verify_dependencies',lambda *a:None)
    with pytest.raises(ValueError,match='Raw reference changed'):
        runner.summarize(args.plan,args.out)


@pytest.mark.parametrize('state,had_checkpoint,message', [
    ('ready',False,'Completed bounded task'),
    ('unresolved',False,'Completed bounded task'),
    ('timeout',True,'Saved checkpoint disappeared')])
def test_resume_retains_completed_or_lost_checkpoint_task(package, monkeypatch, state, had_checkpoint, message):
    args = arguments(package)
    plan, launch, manifest, _ = runner.build_launch(args)
    campaign, task = launch['campaign'], plan['tasks'][0]
    paths = runner.task_paths(campaign,task)
    receipt = dict(identity=dict(plan_sha256=campaign['plan_sha256'],task=task,
        deployment_sha256=prep.fingerprint(campaign)),runtime={},status=state,
        progress=dict(checkpoint_exists=had_checkpoint))
    prep.atomic_json(paths['receipt'],receipt)
    monkeypatch.setattr(runner,'read_launch',lambda *a:(launch,plan))
    monkeypatch.setattr(runner,'runtime_setup',lambda *a:{})
    monkeypatch.setattr(runner,'run_child',lambda *a,**k:pytest.fail('Worker must not run'))
    with pytest.raises(ValueError,match=message):
        runner.execute(manifest,0)
    assert json.loads(paths['receipt'].read_text()) == receipt
