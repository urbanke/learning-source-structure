"""Independent guards for the isolated, bounded pooled timing extension.

Scientific kernels and Slurm are mocked. Frozen population code is never edited.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTED = [2, 10, 13, 18, 21, 24, 27, 30, 33]


@pytest.fixture
def scheduler(tmp_path):
    """A scheduler that records submits and requires array-safe accounting."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    executable = bindir / 'fake-scheduler'
    executable.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
path = Path(os.environ['TIMING_EXTENSION_FAKE_STATE'])
s = json.loads(path.read_text()) if path.exists() else {'counter': 0, 'active': False, 'calls': []}
name = Path(sys.argv[0]).name
if name == 'sbatch':
    s['counter'] += 1
    s['calls'].append(sys.argv[1:])
    path.write_text(json.dumps(s))
    print('uncertain accepted reply' if s.get('ambiguous') else str(81000 + s['counter']))
elif name == 'squeue':
    if s['active']:
        print('81002_0|RUNNING')
elif name == 'sacct':
    formats = [x for x in sys.argv if x.startswith('--format=')]
    assert formats and 'JobIDRaw' not in formats[0] and 'JobID' in formats[0]
    values = []
    for index, item in enumerate(sys.argv):
        if item in ('--jobs', '-j'):
            values = sys.argv[index + 1].split(',')
        elif item.startswith('--jobs='):
            values = item.split('=', 1)[1].split(',')
    if not s.get('missing_accounting'):
        for job in values:
            print(job + '_0|COMPLETED|')
            print(job + '_1|COMPLETED|')
''')
    executable.chmod(0o755)
    for name in ('sbatch', 'squeue', 'sacct'):
        (bindir / name).symlink_to(executable)
    state = tmp_path / 'scheduler-state.json'
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               TIMING_EXTENSION_FAKE_STATE=str(state))
    return state, env


@pytest.fixture
def extension():
    path = ROOT / 'code/scripts/pooled_predictive_timing_extension.py'
    spec = importlib.util.spec_from_file_location('independent_timing_extension', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def population():
    path = ROOT / 'code/scripts/pooled_predictive_population.py'
    spec = importlib.util.spec_from_file_location('frozen_population_for_extension_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_changes_only_bounded_timing_controls(extension, population):
    plan, _ = extension.read_plan()
    original, _ = population.read_plan()
    before = copy.deepcopy(original)
    adjusted = extension.adapted_plan(original, plan)
    expected = copy.deepcopy(original)
    expected['stages']['timing'].update(seconds_per_replica=7200, allocation_minutes=150, concurrency=3)
    expected['resource_limits']['maximum_attempts_per_task'] = 1
    assert adjusted == expected
    assert original == before
    adjusted['stages']['timing']['seconds_per_replica'] = 1
    assert original == before, 'The adapter must not share mutable scientific-plan structures'


def test_exact_nine_cases_preserve_every_original_replica_seed(extension, population):
    plan, _ = extension.read_plan()
    original, _ = population.read_plan()
    adjusted = extension.adapted_plan(original, plan)
    assert list(extension.TASK_IDS) == SELECTED
    before = population.task_specs(original, 'timing')
    after = population.task_specs(adjusted, 'timing')
    assert before == after and len(after) == 36
    assert [(before[i]['observed'], before[i]['family'], before[i]['settings']['particles']) for i in SELECTED] == [
        (16, 'product', 8192), (64, 'product', 1024), (64, 'shared_base', 1024),
        (256, 'product', 256), (256, 'shared_base', 256), (256, 'uniform_base', 256),
        (1024, 'product', 256), (1024, 'shared_base', 256), (1024, 'uniform_base', 256),
    ]
    for i in SELECTED:
        assert population.seed_for(original, before[i], 0) == population.seed_for(adjusted, after[i], 0)
        assert after[i]['repetitions'] == 1


@pytest.mark.parametrize('raise_inside', [False, True])
def test_read_plan_adapter_restores_original_module_even_after_error(extension, population, raise_inside):
    plan, digest = extension.read_plan()
    original, _ = population.read_plan()
    original_loader = population.read_plan
    original_runner = population.run
    original_specs = population.task_specs
    try:
        with extension.timing_adapter(population, original, plan, digest):
            adjusted, bound_hash = population.read_plan(extension.PLAN)
            assert adjusted == extension.adapted_plan(original, plan)
            assert bound_hash == digest
            assert population.run is original_runner and population.task_specs is original_specs
            if raise_inside:
                raise RuntimeError('synthetic failure during allocated work')
    except RuntimeError:
        assert raise_inside
    assert population.read_plan is original_loader
    assert population.run is original_runner and population.task_specs is original_specs
    assert population.read_plan()[0] == original


def timing_rows(extension, population):
    plan, _ = extension.read_plan()
    parent, _ = population.read_plan()
    specs = population.task_specs(parent, 'timing')
    rows = {i: dict(task=specs[i], complete=True, status='complete', records=[dict(elapsed_seconds=1000. + i)])
            for i in SELECTED}
    return plan, rows


def test_full_timing_completion_never_approves_science(extension, population):
    plan, rows = timing_rows(extension, population)
    report = extension.summarize(plan, rows)
    assert report['complete'] is True
    assert report['scientific_reference_validated'] is False
    for key in ('population_reference_validated', 'accuracy_approved', 'production_approved'):
        assert not report.get(key, False)


def test_missing_or_timed_out_case_keeps_complete_false(extension, population):
    plan, rows = timing_rows(extension, population)
    missing = copy.deepcopy(rows)
    missing.pop(SELECTED[-1])
    assert extension.summarize(plan, missing)['complete'] is False
    stopped = copy.deepcopy(rows)
    stopped[SELECTED[-1]].update(complete=False, status='time_limit', records=[])
    report = extension.summarize(plan, stopped)
    assert report['complete'] is False and report['scientific_reference_validated'] is False


def test_login_import_and_plan_check_need_only_standard_library(extension):
    code = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['read_plan'](); assert 'numpy' not in sys.modules and 'scipy' not in sys.modules"
    subprocess.run([sys.executable, '-c', code, str(Path(extension.__file__))],
                   check=True, text=True, capture_output=True)


def test_launch_dry_run_has_no_side_effects(extension, tmp_path):
    launcher = ROOT / 'code/transformer/scitas/submit_pooled_predictive_timing_extension.sh'
    out = tmp_path / 'not-created'
    result = subprocess.run(['bash', str(launcher), '--out', str(out), '--after-job', '66595885'],
                            check=True, text=True, capture_output=True)
    assert not out.exists()
    assert 'sbatch' in result.stdout and '--time=02:30:00' in result.stdout
    assert '--array=' in result.stdout and '%3' in result.stdout
    assert '--dependency=afterany:' in result.stdout
    assert 'scientific_reference_validated' in result.stdout


@pytest.fixture
def submit_command(extension, scheduler, tmp_path):
    state, env = scheduler
    raw = tmp_path / 'raw'
    (raw / 'heldout').mkdir(parents=True)
    (raw / 'heldout/manifest.json').write_text('{}')
    dependencies = tmp_path / 'dependencies'
    (dependencies / 'tables/anchors_prod').mkdir(parents=True)
    (dependencies / 'tables/anchors_prod/manifest.json').write_text('{}')
    original = tmp_path / 'original'
    (original / 'timing/tasks').mkdir(parents=True)
    out = tmp_path / 'extension'
    launcher = ROOT / 'code/transformer/scitas/submit_pooled_predictive_timing_extension.sh'
    command = ['bash', str(launcher), '--plan', str(extension.PLAN), '--raw', str(raw),
               '--dependencies', str(dependencies), '--original-out', str(original),
               '--calibration', str(tmp_path / 'calibration'), '--out', str(out),
               '--after-job', '66595885', '--submit']
    return command, state, env, out


def test_duplicate_submission_reuses_the_same_jobs(submit_command):
    command, state, env, _ = submit_command
    subprocess.run(command, check=True, text=True, capture_output=True, env=env)
    first = json.loads(state.read_text())
    assert first['counter'] == 3, 'Only preparation, one bounded worker array and merge are allowed'
    result = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
    assert json.loads(state.read_text())['counter'] == 3
    assert 'already' in result.stdout.lower() or 'existing' in result.stdout.lower()
    calls = first['calls']
    prepare = next(c for c in calls if c[-1] == 'prepare')
    assert not any(x.startswith('--dependency=') for x in prepare)
    assert prepare[-2] == '66595885', 'The original job remains a provenance input'
    worker = next(c for c in calls if any(x.startswith('--array=') for x in c))
    assert '--time=02:30:00' in worker
    assert any(x.startswith('--array=') and x.endswith('%3') for x in worker)
    assert '--dependency=afterok:81001' in worker
    merge = next(c for c in calls if c[-1] == 'merge')
    assert '--dependency=afterany:81002' in merge


def test_changed_identity_cannot_submit_an_overlapping_wave(submit_command, tmp_path):
    command, state, env, _ = submit_command
    subprocess.run(command, check=True, text=True, capture_output=True, env=env)
    changed = list(command)
    alternate = tmp_path / 'other-raw'
    (alternate / 'heldout').mkdir(parents=True)
    (alternate / 'heldout/manifest.json').write_text('{}')
    changed[changed.index('--raw') + 1] = str(alternate)
    result = subprocess.run(changed, text=True, capture_output=True, env=env)
    assert result.returncode != 0
    assert json.loads(state.read_text())['counter'] == 3


def test_uncertain_submission_reply_does_not_create_a_second_allocation(submit_command):
    command, state, env, _ = submit_command
    state.write_text(json.dumps(dict(counter=0, active=False, calls=[], ambiguous=True)))
    first = subprocess.run(command, text=True, capture_output=True, env=env)
    assert first.returncode != 0 and json.loads(state.read_text())['counter'] == 1
    second = subprocess.run(command, text=True, capture_output=True, env=env)
    assert second.returncode != 0 and json.loads(state.read_text())['counter'] == 1


def test_original_and_extension_output_must_be_disjoint(submit_command):
    command, state, env, _ = submit_command
    command[command.index('--out') + 1] = command[command.index('--original-out') + 1]
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    assert result.returncode != 0
    assert not state.exists(), 'Output overlap must be refused before any scheduler call'


@pytest.fixture
def prepared_case(extension, population, monkeypatch, tmp_path):
    from types import SimpleNamespace
    plan, plan_hash = extension.read_plan()
    parent, _ = population.read_plan()
    args = SimpleNamespace(plan=extension.PLAN, raw=tmp_path/'raw', dependencies=tmp_path/'dep',
                           original_out=tmp_path/'original', out=tmp_path/'extension',
                           calibration=tmp_path/'calibration', after_job='66595885', task_index=SELECTED[0])
    args.out.mkdir()
    rows = {}
    specs = population.task_specs(parent, 'timing')
    for index in SELECTED:
        rows[index] = dict(task=specs[index], complete=index == SELECTED[0],
                           status='complete' if index == SELECTED[0] else 'time_limit',
                           records=[dict(elapsed_seconds=10.)] if index == SELECTED[0] else [],
                           error=None if index == SELECTED[0] else extension.DEADLINE_ERROR)
        path = population.stage_path(args.original_out, 'timing', index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows[index]))
    monkeypatch.setattr(extension, 'compute_context', lambda _: (population, parent, plan, plan_hash, {}, {}))
    monkeypatch.setattr(extension, 'require_original_terminal', lambda _: None)
    monkeypatch.setattr(extension, 'validate_parent_submission', lambda *args: None)
    monkeypatch.setattr(extension, 'validate_report', lambda module, plan, h, spec, *args: copy.deepcopy(rows[spec['task_index']]))
    return args, rows, parent, plan, plan_hash


def test_prepare_reuses_completed_originals_and_extends_only_deadlines(extension, prepared_case):
    args, _, _, _, _ = prepared_case
    assert extension.prepare(args) == 0
    report = json.loads((args.out/'prepared.json').read_text())
    assert set(report['tasks']) == set(map(str, SELECTED))
    assert report['tasks'][str(SELECTED[0])]['action'] == 'reuse_original'
    assert all(report['tasks'][str(i)]['action'] == 'extend_timeout' for i in SELECTED[1:])
    assert report['scientific_reference_validated'] is False
    assert extension.prepare(args) == 0, 'Identical compute-node preparation must be idempotent'


@pytest.mark.parametrize('error', ['Signal SIGUSR1', 'source hash mismatch', None])
def test_prepare_refuses_non_deadline_failures(extension, prepared_case, error):
    args, rows, _, _, _ = prepared_case
    rows[SELECTED[1]]['error'] = error
    with pytest.raises(ValueError, match='deadline'):
        extension.prepare(args)
    assert not (args.out/'prepared.json').exists()


def test_prepare_refuses_unresolved_original_writer_claim(extension, population, prepared_case):
    args, _, _, _, _ = prepared_case
    path = population.stage_path(args.original_out, 'timing', SELECTED[0])
    path.with_suffix('.writer').mkdir()
    with pytest.raises(ValueError, match='writer claim'):
        extension.prepare(args)


def test_reused_completed_original_never_invokes_sampling(extension, population, prepared_case, monkeypatch):
    args, _, _, _, _ = prepared_case
    extension.prepare(args)
    monkeypatch.setattr(population, 'run', lambda *args: pytest.fail('Completed original was sampled again'))
    assert extension.run(args) == 0
    assert not (args.out/'timing').exists()


def test_timeout_extension_delegates_frozen_runner_with_new_limit_only(extension, population, prepared_case, monkeypatch):
    args, _, parent, plan, plan_hash = prepared_case
    extension.prepare(args)
    args.task_index = SELECTED[1]
    old_loader = population.read_plan
    seen = []
    def frozen_runner(plan_path, stage, index, raw, dep, out, calibration):
        adjusted, got_hash = population.read_plan(plan_path)
        assert adjusted == extension.adapted_plan(parent, plan) and got_hash == plan_hash
        assert stage == 'timing' and index == args.task_index and out == args.out and out != args.original_out
        seen.append(index)
        return 75
    monkeypatch.setattr(population, 'run', frozen_runner)
    assert extension.run(args) == 75
    assert seen == [SELECTED[1]] and population.read_plan is old_loader


def test_changed_parent_report_blocks_sampling_after_preparation(extension, population, prepared_case, monkeypatch):
    args, _, _, _, _ = prepared_case
    extension.prepare(args)
    args.task_index = SELECTED[1]
    path = population.stage_path(args.original_out, 'timing', SELECTED[0])
    path.write_text(path.read_text()+'\n')
    monkeypatch.setattr(population, 'run', lambda *args: pytest.fail('A changed original receipt was accepted'))
    with pytest.raises(ValueError, match='changed after preparation'):
        extension.run(args)


@pytest.mark.parametrize('failure', ['active', 'missing', 'running_accounting', None])
def test_original_array_must_have_all_selected_terminal_records(extension, monkeypatch, failure):
    from types import SimpleNamespace
    def query(command, **kwargs):
        if command[0] == 'squeue':
            return SimpleNamespace(stdout='66595885_2|RUNNING\n' if failure == 'active' else '')
        assert command[0] == 'sacct' and '--format=JobID,State' in command
        selected = SELECTED[:-1] if failure == 'missing' else SELECTED
        return SimpleNamespace(stdout='\n'.join('66595885_'+str(i)+'|'+(
            'RUNNING' if failure == 'running_accounting' and i == SELECTED[0] else 'COMPLETED')+'|' for i in selected))
    monkeypatch.setattr(extension.subprocess, 'run', query)
    if failure:
        with pytest.raises(ValueError):
            extension.require_original_terminal('66595885')
    else:
        extension.require_original_terminal('66595885')


@pytest.mark.parametrize('queued', [
    '',
    '91234567_2|RUNNING\n81234567_[0-9%3]|PENDING\n',
    '665958850_2|RUNNING\n166595885_2|PENDING\n',
])
def test_purged_original_live_job_uses_user_queue_and_terminal_accounting(extension, monkeypatch, queued):
    """Slurm returns exit 1 for targeted lookup of a purged completed job."""
    from types import SimpleNamespace
    import getpass
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        assert kwargs['check'] is True
        if command[0] == 'squeue':
            if '--jobs' in command or '-j' in command or any(x.startswith('--jobs=') for x in command):
                raise subprocess.CalledProcessError(1, command,
                    stderr='slurm_load_jobs error: Invalid job id specified')
            assert '--array' in command
            assert '--user' in command and command[command.index('--user')+1] == getpass.getuser()
            assert '--format=%i|%T' in command
            return SimpleNamespace(stdout=queued)
        assert command[0] == 'sacct' and '--format=JobID,State' in command
        assert command[command.index('--jobs')+1] == '66595885'
        states = ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED by 1234')
        return SimpleNamespace(stdout='\n'.join(
            f'66595885_{index}|{states[j % len(states)]}|' for j, index in enumerate(SELECTED)))

    monkeypatch.setattr(extension.subprocess, 'run', query)
    extension.require_original_terminal('66595885')
    assert [call[0] for call in calls] == ['squeue', 'sacct']


@pytest.mark.parametrize('queued', [
    '66595885|RUNNING\n',
    '66595885_2|RUNNING\n',
    '66595885_[2-35%3]|PENDING\n',
    '99123456|RUNNING\n66595885_[18-21]|CONFIGURING\n',
])
def test_active_original_job_and_compressed_arrays_block_prepare(extension, monkeypatch, queued):
    from types import SimpleNamespace

    def query(command, **kwargs):
        assert command[0] == 'squeue', 'Active original must block before accounting can approve it'
        return SimpleNamespace(stdout=queued)

    monkeypatch.setattr(extension.subprocess, 'run', query)
    with pytest.raises(ValueError, match='active'):
        extension.require_original_terminal('66595885')


@pytest.mark.parametrize('accounting', [
    '',
    '66595885|COMPLETED|\n',
    '\n'.join(f'66595885_{i}|COMPLETED|' for i in SELECTED[:-1]),
    '\n'.join(f'66595885_{i}|'+('RUNNING' if i == SELECTED[-1] else 'COMPLETED')+'|' for i in SELECTED),
])
def test_empty_live_queue_does_not_replace_terminal_array_accounting(extension, monkeypatch, accounting):
    from types import SimpleNamespace

    def query(command, **kwargs):
        return SimpleNamespace(stdout='' if command[0] == 'squeue' else accounting)

    monkeypatch.setattr(extension.subprocess, 'run', query)
    with pytest.raises(ValueError, match='accounting'):
        extension.require_original_terminal('66595885')


def test_dependency_receipt_must_bind_original_output_and_all_selected_tasks(extension, tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(original_out=tmp_path/'parent', raw=tmp_path/'raw', dependencies=tmp_path/'dep',
                           calibration=tmp_path/'cal', after_job='66595885')
    identity = dict(stage='timing', plan_sha256='parent-hash', out=str(args.original_out.resolve()),
                    **{k: str(getattr(args,k).resolve()) for k in ('raw','dependencies','calibration')})
    directory = args.original_out/'submissions'
    directory.mkdir(parents=True)
    path = directory/'parent.json'
    receipt = dict(key=extension.digest(identity), identity=identity,
                   attempts=[dict(jobs=[dict(job_id='66595885', mode='run', indices=SELECTED)])])
    path.write_text(json.dumps(receipt))
    extension.validate_parent_submission(args, 'parent-hash')
    receipt['attempts'][0]['jobs'][0]['indices'] = SELECTED[:-1]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='provenance'):
        extension.validate_parent_submission(args, 'parent-hash')


@pytest.fixture
def real_runner(extension, population, monkeypatch, tmp_path):
    """Exercise the frozen writer, validation and resume path, not a replica of it."""
    plan, plan_hash = extension.read_plan()
    parent, _ = population.read_plan()
    adjusted = extension.adapted_plan(parent, plan)
    spec = population.task_specs(parent, 'timing')[SELECTED[0]]
    tokens = np.arange(spec['observed'], dtype=np.int64) % 64
    oracle = np.full(64, 1/64)
    scores = dict(pooled_seed0=.01, pooled_seed1=.02)
    hashes = {'fixture': 'unchanged source identity'}
    deps, environment = {'fixture': 'unchanged dependency identity'}, {'fixture': 'production environment'}
    monkeypatch.setattr(population.calibration, 'dependencies', lambda *args: deps)
    monkeypatch.setattr(population.calibration, 'numerical_environment', lambda *args: environment)
    monkeypatch.setattr(population, 'load_case', lambda *args: (tokens, oracle, scores, hashes))
    monkeypatch.setattr(population, 'FamilyEvidence', lambda *args, **kwargs: object())
    def component(*args):
        return dict(log_evidence=-10., predictive_probs=oracle.tolist(),
                    log_joint_next=(-10. + np.log(oracle)).tolist())
    monkeypatch.setattr(population, 'component_prediction', component)
    monkeypatch.setattr(population, 'RankedLSARowEvidence', lambda *args, **kwargs: object())
    monkeypatch.setattr(population, 'SharedBaseMCMC', lambda *args, **kwargs: object())
    state = dict(calls=[], timeout=False)
    def estimate(sampler, observed, **kwargs):
        state['calls'].append(dict(kwargs))
        np.testing.assert_array_equal(observed, tokens)
        assert kwargs['seconds_limit'] == 7200
        assert kwargs['seed'] == population.seed_for(parent, spec, 0)
        assert {k: kwargs[k] for k in ('particles', 'temperatures', 'moves')} == {
            k: spec['settings'][k] for k in ('particles', 'temperatures', 'moves')}
        if state['timeout']:
            raise TimeoutError(extension.DEADLINE_ERROR)
        return dict(component(), schema='shared-base-predictive-smc-repetition-v1',
                    n_observed=len(tokens), seed=kwargs['seed'], resample_fraction=.75,
                    **{k: kwargs[k] for k in ('particles', 'temperatures', 'moves')},
                    maximum_conditional_normalization_error=0.,
                    effective_sample_sizes=[float(kwargs['particles'])]*kwargs['temperatures'],
                    elapsed_seconds=123., sampler_seconds=120., prediction_seconds=3.)
    monkeypatch.setattr(population, 'estimate', estimate)
    out = tmp_path/'new-extension'
    raw, dependency_root, calibration = tmp_path/'raw', tmp_path/'deps', tmp_path/'calibration'
    def execute():
        with extension.timing_adapter(population, parent, plan, plan_hash):
            return population.run(extension.PLAN, 'timing', spec['task_index'], raw,
                                  dependency_root, out, calibration)
    path = population.stage_path(out, 'timing', spec['task_index'])
    def validate():
        return extension.validate_report(population, adjusted, plan_hash, spec, path,
                                         raw, deps, environment)
    return execute, validate, path, state, plan_hash


def test_real_frozen_runner_saved_report_replays_under_extension(real_runner, extension, population):
    execute, validate, path, state, plan_hash = real_runner
    assert execute() == 0
    saved = validate()
    assert saved['complete'] and saved['attempts'] == 1
    assert saved['identity']['plan_sha256'] == plan_hash
    assert saved['identity']['driver_sha256'] == extension.sha(extension.ROOT/extension.CORE)
    assert saved['records'][0]['elapsed_seconds'] == 123.
    assert not saved['scientific_reference_validated'] and not saved['summary']['basic_passed']
    before = path.read_bytes()
    assert execute() == 0 and len(state['calls']) == 1
    assert path.read_bytes() == before, 'A completed timing measurement must be reused byte-for-byte'
    assert population.read_plan()[1] != plan_hash, 'The frozen plan loader must be restored'


def test_real_timeout_is_saved_and_one_attempt_cap_prevents_resampling(real_runner):
    execute, validate, path, state, _ = real_runner
    state['timeout'] = True
    assert execute() == 75
    saved = validate()
    assert saved['status'] == 'time_limit' and not saved['complete'] and saved['records'] == []
    with pytest.raises(ValueError, match='retry cap'):
        execute()
    assert len(state['calls']) == 1


def test_real_report_replay_rejects_invalid_probability_even_with_updated_payload_digest(real_runner, population):
    execute, validate, path, _, _ = real_runner
    assert execute() == 0
    saved = json.loads(path.read_text())
    saved['records'][0]['predictive_probs'][0] += .01
    saved['records_sha256'] = population.digest(saved['records'])
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='probabilities|normaliz'):
        validate()


def test_compute_context_uses_real_frozen_plan_loader_interfaces(extension, monkeypatch, tmp_path):
    """Cover the startup shared by prepare/run/merge, before the runner adapter.

    Only expensive dependency verification and environment setup are replaced.
    In particular, the real calibration loader requires an explicit path; the
    original no-argument call fails here before reaching either replacement.
    """
    from types import SimpleNamespace
    import importlib

    monkeypatch.syspath_prepend(str(ROOT/'code/scripts'))
    module = importlib.import_module('pooled_predictive_population')
    calibration_loader = module.calibration.read_plan
    expected_calibration, _ = calibration_loader(module.calibration.PLAN)
    expected_parent, _ = module.read_plan(extension.PARENT)
    expected_extension, expected_hash = extension.read_plan()
    args = SimpleNamespace(plan=extension.PLAN, dependencies=tmp_path/'dependencies')
    dependency_value = {'verified-existing-store': 'fixture digest'}
    environment_value = {'verified-existing-environment': 'fixture value'}
    calls = []

    def dependencies(plan, store, src):
        assert plan == expected_calibration
        assert store == args.dependencies/'tables/anchors_prod'
        assert src == args.dependencies/'src'
        calls.append('dependencies')
        return dependency_value

    def environment(store, src):
        assert store == args.dependencies/'tables/anchors_prod'
        assert src == args.dependencies/'src'
        calls.append('environment')
        return environment_value

    monkeypatch.setenv('SLURM_JOB_ID', 'regression-test-allocation')
    monkeypatch.setattr(module.calibration, 'dependencies', dependencies)
    monkeypatch.setattr(module.calibration, 'numerical_environment', environment)
    actual = extension.compute_context(args)
    assert actual == (module, expected_parent, expected_extension, expected_hash,
                      dependency_value, environment_value)
    assert calls == ['dependencies', 'environment']
    assert module.calibration.read_plan is calibration_loader
    assert not args.dependencies.exists(), 'Startup interface test must not create or sample dependencies'


def test_recovery_preserves_original_plan_and_all_numerical_settings(extension, population):
    original_path = ROOT/'code/transformer/configs/pooled_predictive_timing_extension_plan_2026-09-19.json'
    original = json.loads(original_path.read_text())
    recovery, recovery_hash = extension.read_plan()
    assert extension.PLAN.name == 'pooled_predictive_timing_extension_queue_recovery_plan_2026-09-19.json'
    assert extension.sha(original_path) == '925d990830418c2dea8fe027ed3f89e38cbc5bd9ec2127b5ec100120c5579776'
    assert recovery_hash != extension.sha(original_path)
    for name in ('task_ids', 'timing', 'parent_plan', 'original_array_job'):
        assert recovery[name] == original[name]
    assert recovery['code_sha256'][extension.CORE] == original['code_sha256'][extension.CORE]
    parent, _ = population.read_plan()
    assert extension.adapted_plan(parent, recovery) == extension.adapted_plan(parent, original)


def test_recovery_default_output_is_separate_from_failed_extension(extension):
    launcher = ROOT/'code/transformer/scitas/submit_pooled_predictive_timing_extension.sh'
    result = subprocess.run(['bash', str(launcher), '--after-job', '66595885'],
                            check=True, text=True, capture_output=True)
    assert '/output/pooled_predictive_timing_extension_queue_recovery_2026-09-19/logs/' in result.stdout
    assert '/output/pooled_predictive_timing_extension_recovery_2026-09-19/logs/' not in result.stdout
    assert '/output/pooled_predictive_timing_extension_2026-09-19/logs/' not in result.stdout


def test_real_prepare_worker_and_merge_with_actual_frozen_report_contracts(extension, population, monkeypatch, tmp_path):
    """Exercise orchestration and all real plan/report validators together.

    Original task reports are made by the frozen runner, not hand-written.
    Mock only scientific calculation, source-file access and the Slurm server.
    """
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'pooled_predictive_population', population)
    monkeypatch.setenv('SLURM_JOB_ID', 'preparation-regression-allocation')
    parent, parent_hash = population.read_plan()
    plan, plan_hash = extension.read_plan()
    args = SimpleNamespace(plan=extension.PLAN, raw=tmp_path/'raw', dependencies=tmp_path/'dependencies',
        original_out=tmp_path/'original', out=tmp_path/'extension', calibration=tmp_path/'calibration',
        after_job='66595885', task_index=SELECTED[0])
    deps = {'sealed-store-file': 'unchanged-fixture-hash'}
    env = {'PMM_KERNEL': 'unchanged-fixture-environment'}
    monkeypatch.setattr(population.calibration, 'dependencies', lambda p, store, src: deps)
    monkeypatch.setattr(population.calibration, 'numerical_environment', lambda store, src: env)
    oracle = np.full(64, 1/64)

    def source(raw, plan, spec):
        assert raw == args.raw
        tokens = np.full(spec['observed'], spec['task_index'], dtype=np.int64)
        return tokens, oracle.copy(), dict(pooled_seed0=.01, pooled_seed1=.02), {'source': str(spec['task_index'])}

    def component(*unused):
        return dict(log_evidence=-10., predictive_probs=oracle.tolist(),
                    log_joint_next=(-10. + np.log(oracle)).tolist())

    monkeypatch.setattr(population, 'load_case', source)
    monkeypatch.setattr(population, 'FamilyEvidence', lambda *a, **k: object())
    monkeypatch.setattr(population, 'component_prediction', component)
    monkeypatch.setattr(population, 'RankedLSARowEvidence', lambda *a, **k: object())
    monkeypatch.setattr(population, 'SharedBaseMCMC', lambda *a, **k: object())
    sampled = []

    def estimate(sampler, tokens, **settings):
        index = int(tokens[0])
        spec = population.task_specs(parent, 'timing')[index]
        assert settings['seed'] == population.seed_for(parent, spec, 0)
        sampled.append((index, settings['seconds_limit']))
        if settings['seconds_limit'] == 1200 and index != SELECTED[0]:
            raise TimeoutError(extension.DEADLINE_ERROR)
        return dict(component(), schema='shared-base-predictive-smc-repetition-v1', n_observed=len(tokens),
            seed=settings['seed'], resample_fraction=.75,
            **{k:settings[k] for k in ('particles','temperatures','moves')},
            maximum_conditional_normalization_error=0., effective_sample_sizes=[float(settings['particles'])]*settings['temperatures'],
            elapsed_seconds=123., sampler_seconds=120., prediction_seconds=3.)

    monkeypatch.setattr(population, 'estimate', estimate)
    for index in SELECTED:
        code = population.run(extension.PARENT, 'timing', index, args.raw, args.dependencies,
                              args.original_out, args.calibration)
        assert code == (0 if index == SELECTED[0] else 75)
    original_bytes = {i:population.stage_path(args.original_out, 'timing', i).read_bytes() for i in SELECTED}
    identity = dict(stage='timing', plan_sha256=parent_hash, out=str(args.original_out.resolve()),
        **{k:str(getattr(args,k).resolve()) for k in ('raw','dependencies','calibration')})
    extension.atomic_json(args.original_out/'submissions/original.json', dict(key=extension.digest(identity), identity=identity,
        attempts=[dict(jobs=[dict(job_id=args.after_job, mode='run', indices=SELECTED)])]))
    queries = []

    def scheduler(command, **kwargs):
        queries.append(command[0])
        if command[0] == 'squeue':
            assert '--jobs' not in command
            return SimpleNamespace(stdout='99900000_2|RUNNING\n')
        assert command[0] == 'sacct'
        return SimpleNamespace(stdout='\n'.join(f'66595885_{i}|'+('COMPLETED' if i == SELECTED[0] else 'FAILED')+'|'
                                                for i in SELECTED))

    monkeypatch.setattr(extension.subprocess, 'run', scheduler)
    assert extension.prepare(args) == 0
    prepared = extension.read_prepared(args, plan_hash)
    assert prepared['tasks'][str(SELECTED[0])]['action'] == 'reuse_original'
    assert all(prepared['tasks'][str(i)]['action'] == 'extend_timeout' for i in SELECTED[1:])
    assert queries == ['squeue', 'sacct']
    assert extension.run(args) == 0 and len(sampled) == len(SELECTED)
    args.task_index = SELECTED[1]
    assert extension.run(args) == 0
    assert sampled[-1] == (SELECTED[1], 7200)
    assert extension.merge(args) == 3
    summary = json.loads((args.out/'summary.json').read_text())
    assert summary['completed_tasks'] == SELECTED[:2]
    assert summary['missing_tasks'] == SELECTED[2:]
    assert summary['scientific_reference_validated'] is False
    assert [row['source'] for row in summary['cases'][:2]] == ['original_completed', 'extension']
    for index in SELECTED:
        assert population.stage_path(args.original_out, 'timing', index).read_bytes() == original_bytes[index]
