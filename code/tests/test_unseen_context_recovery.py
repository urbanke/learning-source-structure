"""Recovery scope, allocation and failure checks; never launch a numerical worker."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/scripts'))
import unseen_context_recovery as recovery


def parent():
    spec = dict(task=dict(id=recovery.TASK_ID), target_draws=32768)
    receipt = dict(status='unresolved', target_draws=32768)
    row = dict(corrected=dict(audit=dict(draws_per_chain=32768, ready=False,
        status='needs_more_draws', gates=dict(split_rhat_below_1_05=True,
        probability_ess_at_least_200=False, maximum_probability_mcse_at_most_0_005=True,
        raw_normalization=True))))
    return spec, receipt, row


def test_only_declared_completed_ess_failure_can_continue():
    recovery.validate_parent(*parent())
    for field in ('task', 'draws', 'status', 'new_failure', 'already_ready'):
        spec, receipt, row = deepcopy(parent())
        if field == 'task': spec['task']['id'] = 'L8_i215_t64_a32'
        elif field == 'draws': receipt['target_draws'] = 65536
        elif field == 'status': receipt['status'] = 'running'
        elif field == 'new_failure': row['corrected']['audit']['gates']['raw_normalization'] = False
        else: row['corrected']['audit']['ready'] = True
        with pytest.raises(ValueError, match='declared completed'):
            recovery.validate_parent(spec, receipt, row)


def test_parent_completion_required_before_overlay(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'PARENT', tmp_path)
    with pytest.raises(ValueError, match='automatic merge'):
        recovery.validated_overlay(tmp_path/'child')


def test_worker_cannot_run_without_own_supervisor(tmp_path):
    recovery.write(tmp_path/'allocation.json', dict(status='running', task=recovery.TASK_ID,
        reserved_seconds=900, pid=-1))
    with pytest.raises(ValueError, match='bounded supervisor'):
        recovery.worker(tmp_path)
    assert not (tmp_path/'worker.claim').exists()


def test_timeout_kills_child_and_preserves_one_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'PARENT', tmp_path/'output'/'parent')
    monkeypatch.setattr(recovery, 'WALL_SECONDS', .1)
    (tmp_path/'output').mkdir()
    actual_popen = subprocess.Popen
    children = []
    def sleeping_child(command, **kwargs):
        assert command[2] == 'worker'
        child = actual_popen([sys.executable, '-c', 'import time; time.sleep(10)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(recovery.subprocess, 'Popen', sleeping_child)
    out = tmp_path/'output'/'recovery'
    assert recovery.run(out) == 3
    allocation = recovery.read(out/'allocation.json')
    assert allocation['status'] == 'timeout' and allocation['reserved_seconds'] == .1
    assert children[0].poll() is not None
    assert not (out/'plan.json').exists()
    with pytest.raises(ValueError, match='One allocation'):
        recovery.run(out)
    assert len(children) == 1


def test_recovery_refuses_parent_or_nonisolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'PARENT', tmp_path/'output'/'parent')
    for out in (recovery.PARENT, tmp_path/'elsewhere', tmp_path/'output'/'nested'/'child'):
        with pytest.raises(ValueError, match='separate direct child'):
            recovery.run(out)
