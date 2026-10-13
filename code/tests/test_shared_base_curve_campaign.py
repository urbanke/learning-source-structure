"""Frozen complete plans and restart policy, with no Bayesian jobs launched."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_shared_base_curve_campaign as campaign


def test_full_plan_contains_every_case_and_both_rules_without_duplicates():
    jobs = campaign.all_jobs()
    assert len(jobs) == 4*256*4*2
    assert len({j['key'] for j in jobs}) == len(jobs)
    assert {j['prefix'] for j in jobs[:2048]} == {16}
    assert [j['depth'] for j in jobs[:8]] == [0, 1, 4, 8]*2
    assert [j['order'] for j in jobs[:8]] == [16]*4+[32]*4
    for t in campaign.TIMES:
        for d in campaign.DEPTHS:
            assert {(j['index'], j['order']) for j in jobs if j['prefix'] == t and j['depth'] == d} == {
                (i, a) for i in range(256) for a in (16, 32)}


@pytest.mark.parametrize('times,depths,indices', [([4], [0], [0]), ([16], [2], [0]),
                                               ([16], [0], [256]), ([16, 16], [0], [0]),
                                               ([], [0], [0])])
def test_unsupported_selections_fail(times, depths, indices):
    with pytest.raises(ValueError):
        campaign.selected_jobs({'jobs': campaign.all_jobs()}, times, depths, indices)


def test_runtime_selection_does_not_remove_cases_from_plan():
    plan = {'jobs': campaign.all_jobs()}
    subset = campaign.selected_jobs(plan, [16], [0, 4], [0, 255])
    assert len(subset) == 8 and len(plan['jobs']) == 8192
    assert len(campaign.selected_jobs(plan, [64], [0, 4], [0, 255])) == 8


def tiny_plan(tmp_path):
    references = tmp_path/'refs'
    references.mkdir()
    jobs = campaign.all_jobs()[:3]
    cases = {f"a1_L{j['depth']}/{j['index']}": {'files': []} for j in jobs}
    return dict(plan_id='fixed', jobs=jobs, cases=cases, assets={}, global_paths=[],
                source_root=str(tmp_path/'sources'), references=str(references),
                worker=dict(python=sys.executable, script='never_run.py', proposal='iid', steps=64,
                            burn=16384, initial_draws=4096, max_draws=65536, seed=20261006,
                            environment={'LSA_STORE': 'fake', 'PMM_SRC': 'fake'}))


def runtime_args(tmp_path):
    return SimpleNamespace(campaign_out=tmp_path/'campaign', times=[16], depths=[0, 1, 4, 8],
                           indices=list(range(256)), max_workers=2, max_hours=12.,
                           disk_reserve_gb=50., poll_seconds=.001)


def test_ready_and_exhausted_unresolved_are_not_restarted(tmp_path, monkeypatch):
    plan, args = tiny_plan(tmp_path), runtime_args(tmp_path)
    outcomes = {plan['jobs'][0]['key']: 'ready', plan['jobs'][1]['key']: 'unresolved',
                plan['jobs'][2]['key']: 'ready'}
    monkeypatch.setattr(campaign, 'report_state', lambda p, j: outcomes[j['key']])
    monkeypatch.setattr(campaign.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1e12))
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **k: pytest.fail('terminal reference restarted'))
    first = campaign.dispatch(args, plan)
    assert first['states'] == {'ready': 2, 'unresolved': 1}
    again = campaign.dispatch(args, plan)
    assert again['states'] == first['states']
    saved = json.loads((args.campaign_out/'status.json').read_text())
    assert set(saved['jobs']) == {j['key'] for j in plan['jobs']}


def test_exit_two_is_retained_and_resume_does_not_retry(tmp_path, monkeypatch):
    plan, args = tiny_plan(tmp_path), runtime_args(tmp_path)
    monkeypatch.setattr(campaign, 'report_state', lambda *a: None)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1e12))
    launched = []
    class FakeProcess:
        def __init__(self, command, **kwargs):
            launched.append(command)
            self.pid = len(launched)
        def poll(self):
            return 2
    monkeypatch.setattr(campaign.subprocess, 'Popen', FakeProcess)
    first = campaign.dispatch(args, plan)
    assert first['states'] == {'unresolved': 3} and len(launched) == 3
    second = campaign.dispatch(args, plan)
    assert second['states'] == first['states'] and len(launched) == 3


def test_disk_reserve_stops_before_launch_and_keeps_all_cases(tmp_path, monkeypatch):
    plan, args = tiny_plan(tmp_path), runtime_args(tmp_path)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', lambda p: SimpleNamespace(free=49e9))
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **k: pytest.fail('launched below reserve'))
    cost = campaign.dispatch(args, plan)
    assert cost['states'] == {'pending': 3} and cost['stop_reason'] == 'disk reserve reached'


def test_frozen_plan_refuses_drift_but_allows_runtime_time_selection(tmp_path, monkeypatch):
    args = runtime_args(tmp_path)
    current = {'plan_id': 'original', 'jobs': campaign.all_jobs(), 'assets': {'worker': 'abc'}}
    monkeypatch.setattr(campaign, 'build_plan', lambda args: current)
    saved = campaign.freeze_or_resume(args)
    args.times = [64]
    assert campaign.freeze_or_resume(args) == saved
    current = dict(current, assets={'worker': 'changed'})
    with pytest.raises(ValueError, match='plan changed'):
        campaign.freeze_or_resume(args)


def test_asset_drift_is_detected(tmp_path):
    worker = tmp_path/'worker.py'
    worker.write_text('original')
    plan = {'assets': {str(worker): campaign.digest(worker)}}
    campaign.verify_assets(plan)
    worker.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        campaign.verify_assets(plan)


def test_default_protocol_and_cli_selection():
    args = campaign.parser().parse_args(['--times', '16', '--max-workers', '2', '--max-hours', '12'])
    assert args.times == [16] and args.max_workers == 2 and args.max_hours == 12.
    assert args.disk_reserve_gb == 50. and args.max_draws == 65536
    assert args.input.name == 'shared_base_conditional_curves_2026-09-15'
    assert args.references.name == 'shared_base_curve_references_2026-09-15'


def test_worker_python_preserves_virtualenv_symlink(tmp_path, monkeypatch):
    link = tmp_path/'venv'/'bin'/'python'
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    monkeypatch.setattr(campaign.sys, 'executable', str(link))
    monkeypatch.setattr(campaign, 'DEPTHS', ())
    monkeypatch.setattr(campaign, 'CODE', [])
    monkeypatch.delenv('LSA_STORE', raising=False)
    monkeypatch.delenv('PMM_SRC', raising=False)
    args = campaign.parser().parse_args([])
    plan = campaign.build_plan(args)
    assert plan['worker']['python'] == os.path.abspath(link)
    assert plan['worker']['python'] != str(link.resolve())


def test_external_workers_consume_slots_without_touching_their_locks(tmp_path, monkeypatch):
    plan, args = tiny_plan(tmp_path), runtime_args(tmp_path)
    existing = []
    for index in [254, 255]:
        path = Path(plan['references'])/'a1_L4'/'alpha16'/f'seq_{index:06d}_t1024.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('external worker lock')
        existing.append(path)
    monkeypatch.setattr(campaign, 'lock_busy', lambda path: path in existing)
    monkeypatch.setattr(campaign, 'report_state', lambda *a: None)
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **k: pytest.fail('exceeded total worker limit'))
    calls = []
    def disk(path):
        calls.append(path)
        return SimpleNamespace(free=1e12 if len(calls) == 1 else 49e9)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', disk)
    cost = campaign.dispatch(args, plan)
    assert cost['states'] == {'pending': 3}
    assert all(path.read_text() == 'external worker lock' for path in existing)


def test_budget_stop_allows_active_bounded_workers_to_finish(tmp_path, monkeypatch):
    plan, args = tiny_plan(tmp_path), runtime_args(tmp_path)
    monkeypatch.setattr(campaign, 'report_state', lambda *a: None)
    calls = []
    def disk(path):
        calls.append(path)
        return SimpleNamespace(free=1e12 if len(calls) <= 3 else 49e9)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', disk)
    launched = []
    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.polls = 0
            launched.append(self)
            self.pid = len(launched)
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 2
    monkeypatch.setattr(campaign.subprocess, 'Popen', FakeProcess)
    cost = campaign.dispatch(args, plan)
    assert len(launched) == 2 and all(p.polls == 2 for p in launched)
    assert cost['states'] == {'unresolved': 2, 'pending': 1}
    assert cost['stop_reason'] == 'disk reserve reached'
