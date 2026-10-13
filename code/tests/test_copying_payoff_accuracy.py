"""Independent no-sampling checks for the fixed copying accuracy allocation."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import copying_payoff_accuracy as accuracy
import copying_payoff_bayes_pilot as pilot
from prepare_copying_payoff_references import reference_key


FOUR_TOKEN_PATTERNS = {
    (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
    (0, 0, 1, 2), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2),
    (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 2, 0),
    (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3),
}
FAMILY_COMPARISONS = 64 * (30 + 21)


class SyntheticCatalog:
    def next_probs(self, prefix):
        values = np.ones(64)
        for token in prefix:
            values[token] += 1
        return values / values.sum()


@pytest.fixture
def parent_plan():
    patterns = []
    for length in (5, 8, 16):
        for tokens, selection in (([0] * (length - 1) + [1], 'fewest_seen'),
                                  (list(range(length)), 'most_seen')):
            patterns.append(dict(tokens=tokens, time=length, n_seen=len(set(tokens)),
                                 selection=selection, reference_key=reference_key(tokens)))
    inventory = dict(schema='copying-payoff-canonical-reference-readiness-v1',
                     histories_total=768, depths=[0, 2, 3],
                     pilot={'representative_patterns': patterns})
    return dict(tasks=pilot.tasks_from_inventory(inventory))


@pytest.fixture
def tasks(parent_plan):
    return accuracy.build_tasks(parent_plan, SyntheticCatalog())


def test_all_fifteen_exact_prefixes_have_both_orders(tasks):
    assert len(tasks) == 42
    assert [task['task_index'] for task in tasks] == list(range(42))
    assert len({task['id'] for task in tasks}) == 42
    fresh = [task for task in tasks if task['kind'] == 'exact_t4']
    assert len(fresh) == 30
    assert {(tuple(task['tokens']), task['order']) for task in fresh} == {
        (prefix, order) for prefix in FOUR_TOKEN_PATTERNS for order in (16, 32)
    }
    assert len({task['seed'] for task in fresh}) == 30
    for task in fresh:
        assert task['time'] == 4 and task['baseline_draws'] == 0
        assert task['reference_key'] == reference_key(task['tokens'])
        np.testing.assert_array_equal(task['expected_probs'],
                                      SyntheticCatalog().next_probs(task['tokens']))


def test_every_pilot_task_continues_unchanged_including_failed_cases(tasks, parent_plan):
    continued = [task for task in tasks if task['kind'] == 'pilot_continuation']
    assert len(continued) == 12
    assert {task['parent_index'] for task in continued} == set(range(12))
    for task in continued:
        parent = parent_plan['tasks'][task['parent_index']]
        for field in ('tokens', 'time', 'order', 'seed', 'reference_key'):
            assert task[field] == parent[field]
        assert task['baseline_draws'] == 4096
    assert {4, 9}.issubset(task['parent_index'] for task in continued)
    assert all(task['target_draws'] == 16384 for task in tasks)


def test_fresh_selection_deterministic_and_pilot_outcomes_irrelevant(parent_plan):
    unchanged = deepcopy(parent_plan)
    first = accuracy.build_tasks(parent_plan, SyntheticCatalog())
    assert parent_plan == unchanged
    for task in parent_plan['tasks']:
        task['historical_mixing_passed'] = task['task_index'] not in (4, 9)
        task['reported_loss_gap'] = 1000 - task['task_index']
    assert accuracy.build_tasks(parent_plan, SyntheticCatalog()) == first


def test_frozen_allocation_is_single_worker_one_attempt():
    assert accuracy.TARGET_DRAWS == 16384
    assert accuracy.RESOURCES == dict(max_workers=1, task_wall_seconds=900,
        total_worker_seconds=42 * 900, maximum_attempts_per_task=1, checkpoint_seconds=120)
    assert accuracy.RESOURCES['total_worker_seconds'] / 3600 == 10.5


def test_exact_vector_check_uses_global_two_sided_family_threshold():
    expected = np.full(64, 1 / 64)
    mcse = np.full(64, 0.0001)
    check = accuracy.exact_vector_check(expected, mcse, expected)
    assert check['passed'] and check['failing_coordinates'] == []
    assert check['family_comparisons'] == FAMILY_COMPARISONS
    assert check['z_multiplier'] == pytest.approx(norm.ppf(1 - .05 / (2 * FAMILY_COMPARISONS)))
    z = norm.ppf(1 - .05 / (2 * FAMILY_COMPARISONS))
    for factor, accepted in ((.99, True), (1.01, False)):
        prediction = expected.copy()
        prediction[0] += factor * (z * mcse[0] + 1e-7)
        prediction[1] -= factor * (z * mcse[1] + 1e-7)
        assert accuracy.exact_vector_check(prediction, mcse, expected)['passed'] is accepted


def test_exact_check_inspects_every_coordinate_including_last_unseen_label():
    expected = np.full(64, 1 / 64)
    prediction = expected.copy()
    prediction[63] += .01
    prediction[62] -= .01
    check = accuracy.exact_vector_check(prediction, np.zeros(64), expected)
    assert not check['passed']
    assert check['failing_coordinates'] == [62, 63]


def test_exact_deterministic_allowance_remains_one_estimate():
    expected = np.full(64, 1 / 64)
    for delta, accepted in ((.75e-7, True), (1.25e-7, False)):
        prediction = expected.copy()
        prediction[2] += delta
        prediction[3] -= delta
        assert accuracy.exact_vector_check(prediction, np.zeros(64), expected)['passed'] is accepted


def test_paired_orders_combine_independent_variances_and_two_allowances():
    right = np.full(64, 1 / 64)
    left_error = np.full(64, .0001)
    right_error = np.full(64, .0002)
    z = norm.ppf(1 - .05 / (2 * FAMILY_COMPARISONS))
    threshold = z * np.hypot(.0001, .0002) + 2e-7
    for factor, accepted in ((.99, True), (1.01, False)):
        left = right.copy()
        left[0] += factor * threshold
        left[1] -= factor * threshold
        result = accuracy.paired_order_check(left, left_error, right, right_error)
        swapped = accuracy.paired_order_check(right, right_error, left, left_error)
        assert result['passed'] is accepted
        assert swapped['passed'] is accepted
        assert result['family_comparisons'] == FAMILY_COMPARISONS
        assert result['failing_coordinates'] == swapped['failing_coordinates']
    left = right.copy()
    left[0] += 1.5e-7
    left[1] -= 1.5e-7
    assert accuracy.paired_order_check(left, np.zeros(64), right, np.zeros(64))['passed']


@pytest.mark.parametrize('bad', ['nan_prediction', 'infinite_error', 'negative_error',
                                 'short_prediction', 'short_reference'])
def test_invalid_exact_comparison_input_rejected(bad):
    prediction = np.full(64, 1 / 64)
    expected = prediction.copy()
    error = np.full(64, .0001)
    if bad == 'nan_prediction': prediction[5] = np.nan
    elif bad == 'infinite_error': error[5] = np.inf
    elif bad == 'negative_error': error[5] = -.0001
    elif bad == 'short_prediction': prediction = prediction[:-1]
    else: expected = expected[:-1]
    with pytest.raises(ValueError):
        accuracy.exact_vector_check(prediction, error, expected)


def reports_for(tasks):
    return [dict(task=deepcopy(task), audit_passed=True,
                 estimate=dict(predictive_probs=np.full(64, 1 / 64).tolist(),
                               mcse_probs=np.full(64, .0001).tolist())) for task in tasks]


def test_complete_accuracy_coverage_does_not_approve_payoff_forecast(tasks):
    reports = reports_for(tasks)
    before = deepcopy((tasks, reports))
    result = accuracy.coverage_summary(tasks, reports)
    assert result['complete'] and result['all_tasks_passed']
    assert result['accuracy_diagnostic_passed']
    assert len(result['paired_order_checks']) == 21
    assert result['payoff_forecast_approved'] is False
    assert result['population_reference_approved'] is False
    assert (tasks, reports) == before


@pytest.mark.parametrize('failed_parent', [4, 9])
def test_historical_failed_task_cannot_be_omitted_or_ignored(tasks, failed_parent):
    reports = reports_for(tasks)
    failed = next(report for report in reports
                  if report['task'].get('parent_index') == failed_parent)
    failed['audit_passed'] = False
    result = accuracy.coverage_summary(tasks, reports)
    assert result['complete']
    assert not result['all_tasks_passed']
    assert not result['accuracy_diagnostic_passed']
    reports.remove(failed)
    missing = accuracy.coverage_summary(tasks, reports)
    assert not missing['complete'] and not missing['accuracy_diagnostic_passed']


def test_paired_disagreement_blocks_even_when_all_individual_audits_pass(tasks):
    reports = reports_for(tasks)
    reports[0]['estimate']['predictive_probs'][0] += .01
    reports[0]['estimate']['predictive_probs'][1] -= .01
    result = accuracy.coverage_summary(tasks, reports)
    assert result['complete'] and result['all_tasks_passed']
    assert not result['accuracy_diagnostic_passed']
    assert sum(not check['passed'] for check in result['paired_order_checks']) == 1


def test_empty_coverage_cannot_pass_vacuously(tasks):
    result = accuracy.coverage_summary(tasks, [])
    assert not result['complete'] and not result['accuracy_diagnostic_passed']
    assert result['payoff_forecast_approved'] is False
    assert result['population_reference_approved'] is False


@pytest.mark.parametrize('changed', ['duplicate', 'task_identity'])
def test_ambiguous_or_changed_report_identity_cannot_receive_approval(tasks, changed):
    reports = reports_for(tasks)
    if changed == 'duplicate': reports.append(deepcopy(reports[0]))
    else: reports[0]['task']['seed'] += 1
    try:
        result = accuracy.coverage_summary(tasks, reports)
    except ValueError:
        return
    assert not result['accuracy_diagnostic_passed']


def loaded_parent_pair():
    rngs = [np.random.default_rng(seed) for seed in range(4)]
    states = [SimpleNamespace(alpha_index=chain, ranks=np.arange(64),
                              log_likelihood=-10. - chain, log_target=-20. - chain)
              for chain in range(4)]
    run = SimpleNamespace(chains=4, burn=16384, thin=1, seed=782,
        fingerprint={'original_target': True}, proposal_descriptor={'original_kernel': True},
        initializations=[{'initial_chain': chain} for chain in range(4)],
        _rngs=rngs, _states=states, draw_counts=np.full(4, 4096),
        _burn_done=np.full(4, 16384))
    metadata = dict(config={key: getattr(run, key) for key in ('chains', 'burn', 'thin', 'seed')},
        fingerprint=deepcopy(run.fingerprint), proposal_descriptor=deepcopy(run.proposal_descriptor),
        initializations=deepcopy(run.initializations),
        rng_states=[deepcopy(rng.bit_generator.state) for rng in rngs],
        states=[dict(alpha_index=state.alpha_index, ranks=state.ranks.tolist(),
                     log_likelihood=state.log_likelihood, log_target=state.log_target)
                for state in states])
    return run, metadata


def test_loaded_continuation_preserves_parent_rng_state_without_drawing():
    run, metadata = loaded_parent_pair()
    before = deepcopy([rng.bit_generator.state for rng in run._rngs])
    accuracy.verify_loaded_parent(run, metadata)
    assert [rng.bit_generator.state for rng in run._rngs] == before


@pytest.mark.parametrize('changed', ['seed', 'burn', 'thin', 'target', 'proposal',
                                    'rng', 'state', 'initialization'])
def test_changed_loaded_parent_is_rejected_before_any_extension(changed):
    run, metadata = loaded_parent_pair()
    if changed in ('seed', 'burn', 'thin'):
        setattr(run, changed, getattr(run, changed) + 1)
    elif changed == 'target': run.fingerprint['original_target'] = False
    elif changed == 'proposal': run.proposal_descriptor['original_kernel'] = False
    elif changed == 'rng': run._rngs[2] = np.random.default_rng(99999)
    elif changed == 'state': run._states[1].ranks = np.arange(63, -1, -1)
    else: run.initializations[0]['initial_chain'] = -1
    with pytest.raises(ValueError):
        accuracy.verify_loaded_parent(run, metadata)


@pytest.fixture
def simulated_dispatch(tasks, tmp_path, monkeypatch):
    """Exercise allocation records while replacing every process with a fake."""
    plan = dict(tasks=tasks, plan_id='synthetic-allocation', python='/never-executed/python',
                resources=deepcopy(accuracy.RESOURCES), environment={})
    monkeypatch.setattr(accuracy, 'load_plan', lambda out: plan)
    monkeypatch.setattr(accuracy.pilot, 'environment', lambda value: {})
    monkeypatch.setattr(accuracy, 'summary', lambda out: {'sampling_was_simulated': True})
    launches, waits = [], []

    class FailedProcess:
        pid = 987654

        def __init__(self, command, **kwargs):
            launches.append((command, kwargs))

        def wait(self, timeout):
            waits.append(timeout)
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(accuracy.subprocess, 'Popen', FailedProcess)
    return tmp_path, plan, launches, waits


def test_dispatch_reserves_all_forty_two_once_and_never_retries_failures(simulated_dispatch):
    out, plan, launches, waits = simulated_dispatch
    accuracy.dispatch(out)
    attempts = [accuracy.read(accuracy.paths(out, task)['attempt']) for task in plan['tasks']]
    assert len(launches) == 42
    assert waits == [900] * 42
    assert sum(attempt['reserved_seconds'] for attempt in attempts) == 37800
    assert all(attempt['status'] == 'failed' for attempt in attempts)
    accuracy.dispatch(out)
    assert len(launches) == 42
    assert attempts == [accuracy.read(accuracy.paths(out, task)['attempt']) for task in plan['tasks']]


@pytest.mark.parametrize('indices', [[], [-1], [42], [True]])
def test_dispatch_rejects_outside_scope_without_starting_process(simulated_dispatch, indices):
    out, _, launches, _ = simulated_dispatch
    with pytest.raises(ValueError):
        accuracy.dispatch(out, indices)
    assert launches == []


def test_timeout_gets_one_allocation_and_cannot_restart(simulated_dispatch, monkeypatch):
    out, plan, launches, waits = simulated_dispatch
    killed = []

    class TimedOutProcess:
        pid = 987654

        def __init__(self, command, **kwargs):
            launches.append((command, kwargs))
            self.wait_count = 0

        def wait(self, timeout):
            waits.append(timeout)
            self.wait_count += 1
            if self.wait_count == 1:
                raise accuracy.subprocess.TimeoutExpired('synthetic-worker', timeout)
            return -15

        def poll(self):
            return -15

    monkeypatch.setattr(accuracy.subprocess, 'Popen', TimedOutProcess)
    monkeypatch.setattr(accuracy.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    accuracy.dispatch(out, [0])
    attempt = accuracy.read(accuracy.paths(out, plan['tasks'][0])['attempt'])
    assert attempt['status'] == 'time_limit' and attempt['exit_code'] == 75
    assert attempt['reserved_seconds'] == 900
    assert waits == [900, 5]
    assert killed == [(987654, accuracy.signal.SIGTERM)]
    accuracy.dispatch(out, [0])
    assert len(launches) == 1


def test_dispatch_cannot_exceed_recorded_reservation_budget(simulated_dispatch):
    out, plan, launches, _ = simulated_dispatch
    accuracy.write(accuracy.paths(out, plan['tasks'][0])['attempt'],
                   {'reserved_seconds': 37800, 'status': 'time_limit'})
    with pytest.raises(ValueError, match='budget'):
        accuracy.dispatch(out, [1])
    assert launches == []
