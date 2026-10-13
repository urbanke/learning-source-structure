"""Finish the four time-7--8 alpha refinements in a bounded serial allocation.

The preceding two-worker allocation measured a complete order-32 checkpoint at
just over 5 GiB per process while its simultaneous pair stayed below 10 GiB.
This immutable recovery therefore runs one original-parent continuation at a
time, with a 6-GiB process ceiling and the same 10-GiB aggregate ceiling.
It retains the pre-final-save release of recomputable sampler caches and leaves
the samples, states, RNG continuity and unchanged numerical audits untouched.
"""
from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
import sys


_BASE = Path(__file__).with_name("copying_payoff_alpha_refinement.py")
_SPEC = importlib.util.spec_from_file_location(
    "_copying_payoff_alpha_t7_t8_single_worker_recovery_engine", _BASE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Cannot load the frozen alpha-refinement engine")
_engine = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _engine
_SPEC.loader.exec_module(_engine)


SCHEMA = "copying-payoff-alpha-t7-t8-single-worker-recovery-v1"
OVERLAY_SCHEMA = _engine.OVERLAY_SCHEMA
DEFAULT_OUT = _engine.ROOT / "output/copying_payoff_alpha_t7_t8_single_worker_recovery_2026-09-21"
MAX_WORKERS = 1
RESERVED_BYTES_PER_WORKER = 6 * 2**30
TOTAL_RESERVED_BYTES = 10 * 2**30
TASK_WALL_SECONDS = 1800
TOTAL_WALL_SECONDS = 5200
TARGETS = _engine.TARGETS[4:]

_engine.SCHEMA = SCHEMA
_engine.DEFAULT_OUT = DEFAULT_OUT
_engine.MAX_WORKERS = MAX_WORKERS
_engine.RESERVED_BYTES_PER_WORKER = RESERVED_BYTES_PER_WORKER
_engine.TOTAL_RESERVED_BYTES = TOTAL_RESERVED_BYTES
_engine.TASK_WALL_SECONDS = TASK_WALL_SECONDS
_engine.TOTAL_WALL_SECONDS = TOTAL_WALL_SECONDS
_engine.TARGETS = TARGETS
_engine.__file__ = __file__


def parent_specs():
    campaign = _engine.parallel.load_campaign(_engine.CAMPAIGN_PARENT)
    specs = []
    common = None
    for observed, task_index, task_id, order in TARGETS:
        folder = _engine.parent_folder(observed)
        plan = _engine.dense.load_plan(folder)
        _engine.require(plan["observed"] == observed, "Parent wave time changed")
        task = next((item for item in plan["tasks"] if item["id"] == task_id), None)
        _engine.require(task is not None and task["task_index"] == task_index
                        and task["order"] == order and task["time"] == observed
                        and task["tokens"] == list(range(observed))
                        and task["target_draws"] == _engine.BASELINE_DRAWS,
                        "Fixed unfinished alpha task changed")
        parent = _engine.validate_parent(plan, task)
        identity = {name: plan[name] for name in
                    ("runtime", "python", "environment", "dependency_sha256", "prior")}
        if common is None:
            common = identity
        _engine.require(identity == common, "Recovery parents use different numerical environments")
        specs.append(dict(time=observed, task=task, parent_plan=_engine.bound(folder / "plan.json"),
                          parent_files=parent["files"], parent_identity=parent["identity"]))
    _engine.require(len(specs) == 4 and len({spec["task"]["id"] for spec in specs}) == 4,
                    "Time-7--8 recovery task family is incomplete")
    return campaign, specs, common


def code_hashes():
    names = (
        "copying_payoff_alpha_t7_t8_single_worker_recovery.py",
        "copying_payoff_alpha_refinement.py",
        "copying_payoff_accuracy.py",
        "copying_payoff_dense.py",
        "copying_payoff_parallel.py",
    )
    return {f"code/scripts/{name}": _engine.sha(_engine.ROOT / "code/scripts" / name)
            for name in names}


_original_build_plan = _engine.build_plan
_original_worker = _engine.worker


def build_plan(out):
    plan = _original_build_plan(out)
    plan.pop("plan_id")
    plan["post_sampling_memory_policy"] = dict(
        moment_caches_recomputable=True,
        release_before_unchanged_raw_and_fine_grid_audits=True,
        released=["rank_terms", "row_evidence", "predictives", "baselines"],
        samples_states_rng_untouched=True)
    plan["allocation_rationale"] = dict(
        serial=True,
        measured_order32_peak_bytes=5_372_395_520,
        individual_ceiling_bytes=RESERVED_BYTES_PER_WORKER,
        aggregate_ceiling_bytes=TOTAL_RESERVED_BYTES)
    plan["plan_id"] = _engine.fingerprint(plan)
    return plan


def _release_sampler_caches(run):
    kernel = run.sampler.kernel
    before = kernel.cache_info()
    for name in ("_terms", "_evidence", "_predictive", "_baselines"):
        getattr(kernel, name).clear()
    gc.collect()
    after = kernel.cache_info()
    print(json.dumps(dict(event="copying_alpha_post_sampling_cache_release",
        before=before, after=after)), flush=True)


def _memory_safe_progress(run, progress):
    def wrapped(event):
        if (event.get("phase") == "draws"
                and event.get("draw_counts")
                and all(value == event["requested_draws"] for value in event["draw_counts"])):
            _release_sampler_caches(run)
        if progress is not None:
            progress(event)
    return wrapped


def worker(out, index):
    """Release only recomputable caches before the final checkpoint save."""
    from shared_base_curve_runner import CurveChainRun

    original_extend = CurveChainRun.extend

    def extend_then_release(self, *args, **kwargs):
        progress = kwargs.get("progress")
        kwargs["progress"] = _memory_safe_progress(self, progress)
        estimate = original_extend(self, *args, **kwargs)
        _release_sampler_caches(self)
        return estimate

    CurveChainRun.extend = extend_then_release
    try:
        return _original_worker(out, index)
    finally:
        CurveChainRun.extend = original_extend


_engine.code_hashes = code_hashes
_engine.parent_specs = parent_specs
_engine.build_plan = build_plan
_engine.worker = worker

ROOT = _engine.ROOT
T5_PARENT = _engine.T5_PARENT
CAMPAIGN_PARENT = _engine.CAMPAIGN_PARENT
BASELINE_DRAWS = _engine.BASELINE_DRAWS
TARGET_DRAWS = _engine.TARGET_DRAWS
MINIMUM_FREE_DISK_BYTES = _engine.MINIMUM_FREE_DISK_BYTES
PARENT_FILES = _engine.PARENT_FILES

output_path = _engine.output_path
paths = _engine.paths
prepare = _engine.prepare
load_plan = _engine.load_plan
current_parent = _engine.current_parent
sample_prefix_proof = _engine.sample_prefix_proof
rng_proof = _engine.rng_proof
validate_replacement = _engine.validate_replacement
overlay = _engine.overlay
run = _engine.run
check_plan = _engine.check_plan
main = _engine.main


if __name__ == "__main__":
    main()
