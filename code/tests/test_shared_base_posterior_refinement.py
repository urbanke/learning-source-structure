"""Fixed follow-ups preserve parent samples and cannot select easier comparisons."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transformer'))
import refine_shared_base_posterior as refinement
from prepare_shared_base_posterior_cluster import task_list
from shared_base_precision_cohort import deterministic_precision
from source import seed_of


@pytest.fixture
def plan():
    return refinement.read_plan(refinement.PLAN)[0]


def test_plan_keeps_the_fixed_sources_and_independent_repeat(plan):
    original = task_list()
    assert len(plan['tasks']) == 9
    assert [t['kind'] for t in plan['tasks']].count('continuation') == 6
    for task in plan['tasks']:
        parent = original[task['base_task_index']]
        assert all(task[k] == parent[k] for k in ('setting', 'index', 'time'))
        if task['kind'] == 'continuation':
            assert task['previous_draws'] < task['initial_draws'] == task['max_draws']
            assert task['master_seed'] == 2026091670
    assert plan['tasks'][-1]['master_seed'] != plan['tasks'][-2]['master_seed']
    assert plan['comparisons']['l0_time1024_primary_orders'] == [32, 64]
    assert plan['production_approved'] is False


@pytest.mark.parametrize('change', ['budget', 'seed', 'repeat_gate', 'worker', 'production'])
def test_changed_fixed_plan_contract_is_rejected(plan, tmp_path, change):
    if change == 'budget':
        plan['tasks'][0]['initial_draws'] += 1
    elif change == 'seed':
        plan['tasks'][8]['master_seed'] = plan['tasks'][7]['master_seed']
    elif change == 'repeat_gate':
        plan['repeat_gate']['agreement_mc_multiplier'] = 500
    elif change == 'worker':
        plan['worker']['burn'] = 0
    else:
        plan['production_approved'] = True
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        refinement.read_plan(path)


def test_worker_forces_continuation_and_uses_separate_repeat_directory(plan, tmp_path):
    context = {'output': str(tmp_path / 'new')}
    base = {'old': {'campaign': {'inputs': str(tmp_path / 'inputs')}}}
    for task in plan['tasks']:
        command = refinement.worker_command(context, task, base)
        def argument(name):
            return command[command.index(name) + 1]
        assert int(argument('--initial-draws')) == task['initial_draws']
        assert int(argument('--max-draws')) == task['max_draws']
        assert int(argument('--seed')) == task['master_seed']
        assert argument('--input') == str(tmp_path / 'inputs' / task['setting'])
        assert Path(argument('--out')).is_relative_to(tmp_path / 'new' / 'work' / task['id'])
    assert len({str(refinement.task_paths(context, t)['raw_root']) for t in plan['tasks']}) == 9


@pytest.fixture
def copy_case(plan, tmp_path, monkeypatch):
    task = plan['tasks'][0]
    parent_dir = tmp_path / 'parent'
    parent_dir.mkdir()
    paths = {}
    for key in ('reference', 'checkpoint', 'moments'):
        paths[key] = parent_dir / key
        paths[key].write_bytes(('original ' + key).encode())
    reference = {'reference_id': 'parent-id', 'estimate': {'diagnostics': {'draws': task['previous_draws']}},
                 'identity': {'master_sampling_seed': task['master_seed']}}
    row = {'row_id': 'parent-row-id'}
    monkeypatch.setattr(refinement, 'verify_base_row', lambda *a: (row, paths, reference))
    return task, {'output': str(tmp_path / 'new')}, {}, paths, reference


def test_copy_is_lossless_and_resuming_keeps_new_checkpoint(copy_case):
    task, context, base, original, _ = copy_case
    before = {k: p.read_bytes() for k, p in original.items()}
    copied = refinement.prepare_work(context, task, base)
    for key in original:
        assert copied[key].read_bytes() == before[key]
        assert copied[key].resolve() != original[key].resolve()
    copied['checkpoint'].write_bytes(b'progress from the resumed sampler')
    refinement.prepare_work(context, task, base)
    assert copied['checkpoint'].read_bytes() == b'progress from the resumed sampler'
    assert {k: p.read_bytes() for k, p in original.items()} == before


def test_changed_parent_cannot_be_silently_recopied(copy_case):
    task, context, base, original, _ = copy_case
    refinement.prepare_work(context, task, base)
    original['checkpoint'].write_bytes(b'changed parent')
    with pytest.raises(ValueError, match='identity changed'):
        refinement.prepare_work(context, task, base)


def test_corrupt_partial_copy_is_rejected(copy_case):
    task, context, base, _, _ = copy_case
    target = refinement.task_paths(context, task)['checkpoint']
    target.parent.mkdir(parents=True)
    target.write_bytes(b'not a copy of parent')
    with pytest.raises(ValueError, match='partial copy differs'):
        refinement.prepare_work(context, task, base)


def test_continuation_requires_the_recorded_draw_count(copy_case):
    task, context, base, _, reference = copy_case
    reference['estimate']['diagnostics']['draws'] += 1
    with pytest.raises(ValueError, match='draw count changed'):
        refinement.prepare_work(context, task, base)


def test_new_control_starts_without_reusing_parent_samples(copy_case, plan):
    _, context, base, _, _ = copy_case
    task = plan['tasks'][-1]
    paths = refinement.prepare_work(context, task, base)
    assert paths['prepared'].exists()
    assert not any(paths[k].exists() for k in ('reference', 'checkpoint', 'moments'))


def make_audit(task, seed, *, ready=True):
    precision = deterministic_precision(np.ones(64)/64, np.ones(64)/64)
    # A synthetic validated precision record, with nonzero between-source spread.
    precision['bayes_regret_bits'] = .05 + .2 * task['index']
    precision['bayes_cross_entropy_bits'] = 6 + precision['bayes_regret_bits']
    record = dict(case_id=task['index'], sampling_seed=seed, precision=precision,
                  full_prior_valid=True, provenance_valid=True, mixing_passed=ready,
                  row_grid_passed=True, per_sequence_precision_passed=True)
    return dict(index=task['index'], time=task['time'], order=task['order'],
                ready=ready, status='ready' if ready else 'needs_more_draws', record=record)


@pytest.fixture
def cohort(plan):
    tasks = task_list()
    rows = [dict(task=t, corrected=dict(audit=make_audit(t, 100+i))) for i, t in enumerate(tasks)]
    new = {}
    for i, task in enumerate(plan['tasks']):
        seed = 100+task['base_task_index'] if task['kind']=='continuation' else 1000+i
        new[task['id']] = dict(task=task, corrected=dict(audit=make_audit(task, seed)))
    return rows, tasks, new


def test_complete_merge_keeps_all_groups_and_fixed_orders(cohort, plan):
    rows, tasks, new = cohort
    old = deepcopy((rows, tasks, new))
    result = refinement.merge_rows(rows, tasks, new, plan)
    assert result['complete'] and result['all_numerical_checks_passed']
    assert len(result['groups']) == 8
    for group in result['groups']:
        expected = [32,64] if (group['depth'],group['time'])==(0,1024) else [16,32]
        assert group['orders'] == expected
        assert group['numerical_audit']['n_sequences'] == 2
    assert (rows, tasks, new) == old


def test_missing_extension_cannot_fall_back_to_passing_parent(cohort, plan):
    rows, tasks, new = cohort
    del new[plan['tasks'][0]['id']]
    result = refinement.merge_rows(rows, tasks, new, plan)
    assert not result['complete'] and not result['all_numerical_checks_passed']
    assert plan['tasks'][0]['id'] in result['missing_tasks']
    assert all(g['numerical_audit']['reported_mean_bayes_regret_bits'] is None for g in result['groups'])


def test_unresolved_extension_remains_in_the_group(cohort, plan):
    rows, tasks, new = cohort
    item = new[plan['tasks'][0]['id']]['corrected']['audit']
    item.update(ready=False, status='needs_more_draws')
    item['record']['mixing_passed'] = False
    result = refinement.merge_rows(rows, tasks, new, plan)
    group = next(g for g in result['groups'] if (g['depth'],g['time'])==(4,256))
    assert result['complete'] and not result['all_numerical_checks_passed']
    assert not group['numerical_audit']['passed']
    assert group['numerical_audit']['reported_mean_bayes_regret_bits'] is None


def test_repeat_disagreement_blocks_l0_but_never_replaces_original(cohort, plan):
    rows, tasks, new = cohort
    repeated = new[plan['tasks'][-1]['id']]['corrected']['audit']
    repeated['record']['precision']['bayes_regret_bits'] += .02
    result = refinement.merge_rows(rows, tasks, new, plan)
    group = next(g for g in result['groups'] if (g['depth'],g['time'])==(0,1024))
    assert result['complete'] and not result['repeat_check']['passed']
    assert not group['numerical_audit']['passed']
    assert group['numerical_audit']['reported_mean_bayes_regret_bits'] is None
    assert {'name': 'independent_same_order_repeat', 'passed': False} in group['numerical_audit']['checks']
    assert [r['candidate_mean_bayes_regret_bits'] for r in group['numerical_audit']['rules']] == pytest.approx([.15,.15])


def test_repeat_must_be_informative_as_well_as_agree(plan):
    task = plan['tasks'][-1]
    original, repeated = make_audit(task, 7), make_audit(task, 8)
    for audit in (original, repeated):
        audit['record']['precision']['mc_variance_bits2'] = .0001
    result = refinement.repeat_check(original, repeated, plan['repeat_gate'])
    assert result['agreement_passed'] and not result['precision_passed']
    assert not result['passed']
    repeated['record']['precision']['mc_variance_bits2'] = 0
    original['record']['precision']['mc_variance_bits2'] = 0
    repeated.update(ready=False, status='needs_more_draws')
    assert not refinement.repeat_check(original, repeated, plan['repeat_gate'])['passed']
    assert not refinement.repeat_check(original, None, plan['repeat_gate'])['complete']


def test_repeated_sampling_seed_is_not_an_independent_check(plan):
    task = plan['tasks'][-1]
    with pytest.raises(ValueError, match='independent sampling seed'):
        refinement.repeat_check(make_audit(task, 7), make_audit(task, 7), plan['repeat_gate'])


@pytest.mark.parametrize('change', [None, 'prior', 'source', 'history', 'draws'])
def test_continuation_identity_history_and_draw_budget(plan, monkeypatch, change):
    task = plan['tasks'][0]
    parent = dict(identity=dict(inputs={'source_sha256': 'same-source'},
        target={'family': 'full_shared_base'}, proposal={'surrogate_steps': 64},
        burn=16384, master_sampling_seed=task['master_seed'],
        sampling_seed=seed_of('shared-base-curve-reference-v1',task['master_seed'],
                             task['setting'],task['index'],task['time'],task['order'])),
        history=[{'draws': task['previous_draws']}])
    reference = dict(identity=deepcopy(parent['identity']), history=deepcopy(parent['history'])+[
        {'draws': task['max_draws']}], setting=task['setting'], index=task['index'],
        time=task['time'], order=task['order'], estimate={'diagnostics': {
        'chains': 4, 'thin': 1, 'draws': task['max_draws']}})
    monkeypatch.setattr(refinement, 'verify_base_row', lambda *a: ({}, {}, parent))
    if change == 'prior': reference['identity']['target']['family'] = 'different'
    if change == 'source': reference['identity']['inputs']['source_sha256'] = 'different'
    if change == 'history': reference['history'][0]['draws'] = 1
    if change == 'draws': reference['estimate']['diagnostics']['draws'] -= 1
    if change is None:
        refinement.validate_new_reference(task, {}, reference)
    else:
        with pytest.raises(ValueError):
            refinement.validate_new_reference(task, {}, reference)


def test_default_preview_does_not_write_or_submit(tmp_path, monkeypatch):
    base = dict(root=tmp_path/'base', launch={'context': {'output':str(tmp_path/'recovery')},
               'venv':str(tmp_path/'venv')}, old={'campaign': {
               'output': str(tmp_path/'original'), 'inputs': str(tmp_path/'inputs')}},
               binding={'summary_sha256':'summary', 'launch_sha256':'launch'})
    monkeypatch.setattr(refinement, 'load_base', lambda p: base)
    monkeypatch.setattr(refinement.subprocess, 'run', lambda *a, **k: pytest.fail('Unexpected submission'))
    args = argparse.Namespace(plan=refinement.PLAN, out=tmp_path/'new', only_task=None,
                              merge=False, submit=False)
    refinement.submit(args)
    assert not args.out.exists()
    launch, _, command = refinement.build_launch(args)
    assert launch['selected_tasks'] == list(range(9))
    assert '--array=0-8%9' in command
