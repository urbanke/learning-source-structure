"""Two-task tail allocation for the targeted copying-payoff continuation.

The full memory-recovery plan has only enough total wall allowance for its
first three two-task waves.  This separate immutable allocation covers the
remaining time-eight 16/32 pair directly from the original 16384-draw parent
checkpoints.  It keeps the same target, kernel and acceptance tests.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_BASE = Path(__file__).with_name("copying_payoff_alpha_refinement.py")
_SPEC = importlib.util.spec_from_file_location(
    "_copying_payoff_alpha_memory_tail_engine", _BASE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Cannot load the frozen alpha-refinement engine")
_engine = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _engine
_SPEC.loader.exec_module(_engine)


SCHEMA = "copying-payoff-alpha-memory-tail-v1"
OVERLAY_SCHEMA = _engine.OVERLAY_SCHEMA
DEFAULT_OUT = _engine.ROOT / "output/copying_payoff_alpha_memory_tail_2026-09-21"
MAX_WORKERS = 2
RESERVED_BYTES_PER_WORKER = 5 * 2**30
TOTAL_RESERVED_BYTES = 10 * 2**30
TASK_WALL_SECONDS = 1800
TOTAL_WALL_SECONDS = 1900
TARGETS = _engine.TARGETS[6:]

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
    """Freeze only the two declared tail parents."""
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
                        "Fixed all-distinct tail task changed")
        parent = _engine.validate_parent(plan, task)
        identity = {name: plan[name] for name in
                    ("runtime", "python", "environment", "dependency_sha256", "prior")}
        if common is None:
            common = identity
        _engine.require(identity == common, "Tail parents use different numerical environments")
        specs.append(dict(time=observed, task=task, parent_plan=_engine.bound(folder / "plan.json"),
                          parent_files=parent["files"], parent_identity=parent["identity"]))
    _engine.require(len(specs) == 2 and len({spec["task"]["id"] for spec in specs}) == 2,
                    "Tail task family is incomplete")
    return campaign, specs, common


def code_hashes():
    names = (
        "copying_payoff_alpha_memory_tail.py",
        "copying_payoff_alpha_refinement.py",
        "copying_payoff_accuracy.py",
        "copying_payoff_dense.py",
        "copying_payoff_parallel.py",
    )
    return {f"code/scripts/{name}": _engine.sha(_engine.ROOT / "code/scripts" / name)
            for name in names}


_engine.code_hashes = code_hashes
_engine.parent_specs = parent_specs

ROOT = _engine.ROOT
T5_PARENT = _engine.T5_PARENT
CAMPAIGN_PARENT = _engine.CAMPAIGN_PARENT
BASELINE_DRAWS = _engine.BASELINE_DRAWS
TARGET_DRAWS = _engine.TARGET_DRAWS
MINIMUM_FREE_DISK_BYTES = _engine.MINIMUM_FREE_DISK_BYTES
PARENT_FILES = _engine.PARENT_FILES

output_path = _engine.output_path
paths = _engine.paths
build_plan = _engine.build_plan
prepare = _engine.prepare
load_plan = _engine.load_plan
current_parent = _engine.current_parent
sample_prefix_proof = _engine.sample_prefix_proof
rng_proof = _engine.rng_proof
worker = _engine.worker
validate_replacement = _engine.validate_replacement
overlay = _engine.overlay
run = _engine.run
check_plan = _engine.check_plan
main = _engine.main


if __name__ == "__main__":
    main()
