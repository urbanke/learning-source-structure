"""Resource and isolation tests for the two-worker memory recovery."""
from pathlib import Path
import sys
from types import SimpleNamespace
import inspect


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import copying_payoff_alpha_refinement as failed
import copying_payoff_alpha_memory_recovery as recovery
import copying_payoff_alpha_memory_tail as tail
import copying_payoff_alpha_sampling_memory_recovery as sampling_recovery
import copying_payoff_alpha_t7_t8_recovery as t7_t8_recovery
import copying_payoff_alpha_t7_t8_single_worker_recovery as t7_t8_serial
import copying_payoff_alpha_audit_only_salvage as audit_only
import copying_payoff_alpha_t6_audit_only_salvage as t6_audit_only
import copying_payoff_alpha_t8_audit_only_salvage as t8_audit_only
import combine_copying_payoff_alpha_audit_recovery as combined
import merge_copying_payoff_alpha_refinement as merger


def test_memory_recovery_keeps_science_and_changes_only_allocation():
    assert recovery.TARGETS == failed.TARGETS
    assert recovery.BASELINE_DRAWS == failed.BASELINE_DRAWS == 16384
    assert recovery.TARGET_DRAWS == failed.TARGET_DRAWS == 262144
    assert recovery.TASK_WALL_SECONDS == failed.TASK_WALL_SECONDS == 1800
    assert recovery.MAX_WORKERS == 2
    assert recovery.RESERVED_BYTES_PER_WORKER == 5 * 2**30
    assert recovery.TOTAL_RESERVED_BYTES == 10 * 2**30
    assert failed.MAX_WORKERS == 4
    assert failed.RESERVED_BYTES_PER_WORKER == 5 * 2**29
    assert recovery.DEFAULT_OUT.name == "copying_payoff_alpha_memory_recovery_2026-09-21"


def test_recovery_plan_binds_wrapper_and_frozen_engine():
    hashes = recovery.code_hashes()
    assert set(hashes) >= {
        "code/scripts/copying_payoff_alpha_memory_recovery.py",
        "code/scripts/copying_payoff_alpha_refinement.py",
    }
    assert recovery._engine.__file__ == recovery.__file__


def test_merger_selects_recovery_producer(tmp_path):
    folder = tmp_path / "recovery"
    folder.mkdir()
    (folder / "plan.json").write_text('{"schema":"' + recovery.SCHEMA + '"}')
    assert merger.producer_for_overlay(folder / "overlay.json") is recovery


def test_merger_binds_wrapper_and_frozen_engine():
    names = {path.name for path in merger.producer_code_paths(
        {recovery, audit_only, sampling_recovery, t6_audit_only,
         t7_t8_serial, t8_audit_only, combined})}
    assert "copying_payoff_alpha_memory_recovery.py" in names
    assert "copying_payoff_alpha_audit_only_salvage.py" in names
    assert "copying_payoff_alpha_sampling_memory_recovery.py" in names
    assert "copying_payoff_alpha_t6_audit_only_salvage.py" in names
    assert "copying_payoff_alpha_t7_t8_single_worker_recovery.py" in names
    assert "copying_payoff_alpha_t8_audit_only_salvage.py" in names
    assert "combine_copying_payoff_alpha_audit_recovery.py" in names
    assert "copying_payoff_alpha_refinement.py" in names


def test_tail_is_exact_time_eight_pair_with_its_own_wall_allocation():
    assert tail.TARGETS == failed.TARGETS[6:]
    assert [(item[0], item[3]) for item in tail.TARGETS] == [(8, 16), (8, 32)]
    assert tail.MAX_WORKERS == 2
    assert tail.RESERVED_BYTES_PER_WORKER == 5 * 2**30
    assert tail.TOTAL_RESERVED_BYTES == 10 * 2**30
    assert tail.TASK_WALL_SECONDS == 1800
    assert tail.TOTAL_WALL_SECONDS == 1900
    assert tail.TARGET_DRAWS == failed.TARGET_DRAWS


def test_combined_overlay_partitions_all_eight_tasks_and_is_auto_detected(tmp_path):
    parts = combined.COMPONENT_IDS
    assert sum(map(len, parts)) == len(set().union(*parts)) == 8
    assert set().union(*parts) == {item[2] for item in failed.TARGETS}
    folder = tmp_path / "combined"
    folder.mkdir()
    (folder / "overlay.json").write_text('{"schema":"' + combined.SCHEMA + '"}')
    assert merger.producer_for_overlay(folder / "overlay.json") is combined


def test_sampling_memory_recovery_covers_exactly_the_six_longer_tasks():
    assert sampling_recovery.TARGETS == failed.TARGETS[2:]
    assert sampling_recovery.MAX_WORKERS == 2
    assert sampling_recovery.RESERVED_BYTES_PER_WORKER == 5 * 2**30
    assert sampling_recovery.TOTAL_RESERVED_BYTES == 10 * 2**30
    assert sampling_recovery.TASK_WALL_SECONDS == 1800
    assert sampling_recovery.TOTAL_WALL_SECONDS == 4400


def test_post_sampling_release_clears_only_recomputable_kernel_caches():
    class Kernel:
        def __init__(self):
            self._terms = {1: 1}; self._evidence = {2: 2}
            self._predictive = {3: 3}; self._baselines = {4: 4}

        def cache_info(self):
            return {name: len(getattr(self, name)) for name in
                    ("_terms", "_evidence", "_predictive", "_baselines")}

    kernel = Kernel()
    run = SimpleNamespace(sampler=SimpleNamespace(kernel=kernel))
    sampling_recovery._release_sampler_caches(run)
    assert all(not getattr(kernel, name) for name in
               ("_terms", "_evidence", "_predictive", "_baselines"))


def test_final_progress_releases_caches_before_checkpoint_callback(monkeypatch):
    events = []
    monkeypatch.setattr(sampling_recovery, "_release_sampler_caches",
                        lambda run: events.append("release"))
    callback = sampling_recovery._memory_safe_progress(
        object(), lambda event: events.append("save"))
    callback(dict(phase="draws", requested_draws=262144,
                  draw_counts=[262144] * 4))
    assert events == ["release", "save"]


def test_audit_only_salvage_is_one_bound_record_and_forbids_sampling():
    assert audit_only.TARGETS == (failed.TARGETS[1],)
    assert audit_only.RESERVED_BYTES == 5 * 2**30
    assert audit_only.TASK_SECONDS == 900
    source = inspect.getsource(audit_only.worker)
    assert "run.report()" in source
    assert ".extend(" not in source
    assert audit_only.EXPECTED_SOURCE_SHA256["checkpoint"] == (
        "3d2646fc19fc2dd4d05d3b4b8b0c1df90ab5d94475026db83f868bac994fce50")


def test_t6_audit_only_salvage_binds_complete_checkpoint_and_forbids_sampling():
    assert t6_audit_only.TARGETS == (failed.TARGETS[3],)
    assert t6_audit_only.RESERVED_BYTES == 5 * 2**30
    assert t6_audit_only.TASK_SECONDS == 900
    source = inspect.getsource(t6_audit_only.worker)
    assert "run.report()" in source
    assert ".extend(" not in source
    assert t6_audit_only.EXPECTED_SOURCE_SHA256["checkpoint"] == (
        "26cd553deaad2770476aa7b365331eae657a4c7dd7d64f9e467e8eca2cd70244")


def test_t8_audit_only_salvage_binds_complete_checkpoint_and_forbids_sampling():
    assert t8_audit_only.TARGETS == (failed.TARGETS[7],)
    assert t8_audit_only.RESERVED_BYTES == 5 * 2**30
    assert t8_audit_only.TASK_SECONDS == 900
    source = inspect.getsource(t8_audit_only.worker)
    assert "run.report()" in source
    assert ".extend(" not in source
    assert t8_audit_only.EXPECTED_SOURCE_SHA256["checkpoint"] == (
        "18dcb5e87e1774548e1245a552d1927e75b5e04dccbccff2b419674c13b18e69")


def test_t7_t8_recovery_uses_measured_individual_margin_under_same_total_bound():
    assert t7_t8_recovery.TARGETS == failed.TARGETS[4:]
    assert t7_t8_recovery.MAX_WORKERS == 2
    assert t7_t8_recovery.RESERVED_BYTES_PER_WORKER == 11 * 2**29
    assert t7_t8_recovery.TOTAL_RESERVED_BYTES == 10 * 2**30
    assert t7_t8_recovery.TASK_WALL_SECONDS == 1800
    assert t7_t8_recovery.TOTAL_WALL_SECONDS == 3600


def test_t7_t8_serial_recovery_has_consistent_memory_and_wall_reservations():
    assert t7_t8_serial.TARGETS == failed.TARGETS[4:]
    assert t7_t8_serial.MAX_WORKERS == 1
    assert t7_t8_serial.RESERVED_BYTES_PER_WORKER == 6 * 2**30
    assert t7_t8_serial.TOTAL_RESERVED_BYTES == 10 * 2**30
    assert t7_t8_serial.RESERVED_BYTES_PER_WORKER <= t7_t8_serial.TOTAL_RESERVED_BYTES
    assert t7_t8_serial.TASK_WALL_SECONDS == 1800
    assert t7_t8_serial.TOTAL_WALL_SECONDS >= 5200
