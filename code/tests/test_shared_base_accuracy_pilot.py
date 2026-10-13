"""Dispatcher scheduling and provenance tests; no Bayesian inference runs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_shared_base_accuracy_pilot as pilot


def reference(out, protocol, order, probabilities, *, valid=True):
    path = pilot.reference_path(out, 0, 16, order)
    path.parent.mkdir(parents=True, exist_ok=True)
    moments = path.with_suffix(".npz")
    np.savez(moments, cov_batch=np.zeros((2, 2)), cov_chain=np.zeros((2, 2)))
    identity = dict(inputs=protocol["inputs"][pilot.case_key(0, 16)],
                    code_sha256={name: protocol["code_sha256"][name] for name in pilot.CODE_FILES})
    value = dict(identity=identity, reference_id=pilot.fingerprint(identity), sequence_index=0, prefix_length=16,
                 settings=dict(alpha_order=order, seed=pilot.seed_of(pilot.SEED_TAG, pilot.SETTINGS["seed"], 0, 16, order),
                               **{k: pilot.SETTINGS[k] for k in ("chains", "burn", "thin", "surrogate_steps")}),
                 moments_file=moments.name, moments_sha256=pilot.sha_file(moments),
                 local_validation_passed=valid,
                 estimate=dict(predictive_probs=probabilities, mcse_probs=[0.0, 0.0], elapsed_seconds=1.0,
                               diagnostics=dict(draws=256 if valid else 4096)))
    path.write_text(json.dumps(value))
    return path


def test_fixed_plan_commands_and_endpoint_reuse():
    assert pilot.INDICES == tuple(range(8))
    assert pilot.TIMES == (16, 64, 256, 1024)
    assert [i for i in pilot.INDICES if pilot.requires_reuse_audit(i, 1024)] == [0, 1, 2, 3]
    assert not any(pilot.requires_reuse_audit(i, 16) for i in pilot.INDICES)
    command = pilot.job_command(Path("input"), Path("output"), 7, 16, 32)
    for name, value in (("burn", "16384"), ("max-draws", "4096"), ("chains", "4"),
                        ("seed", "20260926"), ("thin", "1"), ("surrogate-steps", "64")):
        assert command[command.index("--" + name) + 1] == value


def test_refinement_failure_is_distinct_from_sampling_failure_and_provenance_change(tmp_path, monkeypatch):
    protocol = dict(inputs={pilot.case_key(0, 16): {"fixed": "case"}},
                    code_sha256={name: "hash" for name in pilot.CODE_FILES})
    monkeypatch.setattr(pilot, "load_inputs", lambda *a: (None, None, [.4, .6],
                        {"d64": {"prediction": [.6, .4]}, "d64_seed1": {"prediction": [.55, .45]}}, None))
    reference(tmp_path, protocol, 16, [.5, .5])
    refined = reference(tmp_path, protocol, 32, [.5, .5])
    assert pilot.assess_pair(tmp_path, tmp_path, protocol, 0, 16, (16, 32))["state"] == "pair_validated"
    reference(tmp_path, protocol, 32, [.501, .499])
    assert pilot.assess_pair(tmp_path, tmp_path, protocol, 0, 16, (16, 32))["state"] == "exponent_refinement_failed"
    reference(tmp_path, protocol, 32, [.501, .499], valid=False)
    assert pilot.assess_pair(tmp_path, tmp_path, protocol, 0, 16, (16, 32))["state"] == "sampling_or_job_unresolved"
    value = json.loads(refined.read_text())
    value["settings"]["burn"] = 512
    refined.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance differs"):
        pilot.inspect_reference(tmp_path, protocol, 0, 16, 32)


def test_subprocesses_are_bounded_and_terminal_jobs_are_reused(tmp_path, monkeypatch):
    finished = set()
    active = 0
    maximum = 0
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            nonlocal active, maximum
            index = int(command[command.index("--index") + 1])
            prefix = int(command[command.index("--prefix") + 1])
            order = int(command[command.index("--alpha-order") + 1])
            self.key = (index, prefix, order)
            self.pid = 100 + len(launches)
            self.returncode = None
            self.polls = 0
            launches.append(self.key)
            active += 1
            maximum = max(maximum, active)

        def poll(self):
            nonlocal active
            self.polls += 1
            if self.polls == 1:
                return None
            if self.returncode is None:
                self.returncode = 0
                finished.add(self.key)
                active -= 1
            return self.returncode

    monkeypatch.setattr(pilot.subprocess, "Popen", Process)
    monkeypatch.setattr(pilot.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pilot, "inspect_reference", lambda out, protocol, i, t, a:
                        dict(state="locally_valid" if (i, t, a) in finished else "missing",
                             terminal=(i, t, a) in finished))
    jobs = [(i, 16, order) for i in range(3) for order in (16, 32)]
    status = {"jobs": {}}
    pilot.run_jobs(jobs, tmp_path, tmp_path, {}, status, 2, lambda: None)
    assert maximum == 2 and active == 0 and launches == jobs
    pilot.run_jobs(jobs, tmp_path, tmp_path, {}, status, 2, lambda: None)
    assert launches == jobs
    assert all(entry["reused"] and len(entry["attempts"]) == 1 for entry in status["jobs"].values())
