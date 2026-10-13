"""Memory recovery for the eight targeted copying-payoff continuations.

The first immutable allocation established that the 262144-draw continuation
can exceed 2.5 GiB in one worker.  This fresh allocation keeps the scientific
plan unchanged and starts again from the eight frozen 16384-draw parents, but
admits only two workers with 5 GiB reserved for each.  The failed allocation
and every parent remain untouched.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_BASE = Path(__file__).with_name("copying_payoff_alpha_refinement.py")
_SPEC = importlib.util.spec_from_file_location(
    "_copying_payoff_alpha_memory_recovery_engine", _BASE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("Cannot load the frozen alpha-refinement engine")
_engine = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _engine
_SPEC.loader.exec_module(_engine)


SCHEMA = "copying-payoff-alpha-memory-recovery-v1"
OVERLAY_SCHEMA = _engine.OVERLAY_SCHEMA
DEFAULT_OUT = _engine.ROOT / "output/copying_payoff_alpha_memory_recovery_2026-09-21"
MAX_WORKERS = 2
RESERVED_BYTES_PER_WORKER = 5 * 2**30
TOTAL_RESERVED_BYTES = 10 * 2**30

# Every engine function resolves configuration through its private module
# namespace.  Pointing __file__ at this wrapper also makes spawned workers
# re-enter the recovery driver rather than the failed four-worker driver.
_engine.SCHEMA = SCHEMA
_engine.DEFAULT_OUT = DEFAULT_OUT
_engine.MAX_WORKERS = MAX_WORKERS
_engine.RESERVED_BYTES_PER_WORKER = RESERVED_BYTES_PER_WORKER
_engine.TOTAL_RESERVED_BYTES = TOTAL_RESERVED_BYTES
_engine.__file__ = __file__


def code_hashes():
    """Bind both this allocation policy and the unchanged sampling engine."""
    names = (
        "copying_payoff_alpha_memory_recovery.py",
        "copying_payoff_alpha_refinement.py",
        "copying_payoff_accuracy.py",
        "copying_payoff_dense.py",
        "copying_payoff_parallel.py",
    )
    return {f"code/scripts/{name}": _engine.sha(_engine.ROOT / "code/scripts" / name)
            for name in names}


_engine.code_hashes = code_hashes

# Expose the engine's validated interfaces without changing the failed plan's
# source file or module namespace.
ROOT = _engine.ROOT
T5_PARENT = _engine.T5_PARENT
CAMPAIGN_PARENT = _engine.CAMPAIGN_PARENT
BASELINE_DRAWS = _engine.BASELINE_DRAWS
TARGET_DRAWS = _engine.TARGET_DRAWS
TASK_WALL_SECONDS = _engine.TASK_WALL_SECONDS
TOTAL_WALL_SECONDS = _engine.TOTAL_WALL_SECONDS
MINIMUM_FREE_DISK_BYTES = _engine.MINIMUM_FREE_DISK_BYTES
TARGETS = _engine.TARGETS
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
