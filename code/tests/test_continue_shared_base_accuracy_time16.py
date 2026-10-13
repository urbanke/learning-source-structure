"""Continuation protocol tests; no new posterior samples are drawn."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import continue_shared_base_accuracy_time16 as continuation


def test_prepare_copies_closed_snapshots_and_never_resets_continued_files(tmp_path, monkeypatch):
    cost = continuation.cost
    original, out = tmp_path / "original", tmp_path / "continued"
    original.mkdir()
    out.mkdir()
    initial = dict(inputs={"fixed": "cohort"}, settings=cost.SETTINGS)
    (original / "protocol.json").write_text(json.dumps(initial))
    jobs = {}
    for index in continuation.INDICES:
        for order in cost.INITIAL_ORDERS:
            path = cost.reference_path(original, index, 16, order)
            path.parent.mkdir(parents=True, exist_ok=True)
            moments = path.with_suffix(".npz")
            checkpoint = original / "chains" / f"alpha{order}" / moments.name
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"closed checkpoint {index} {order}".encode())
            moments.write_bytes(f"closed moments {index} {order}".encode())
            path.write_text(json.dumps(dict(moments_file=moments.name, chain_checkpoint=str(checkpoint.resolve()))))
            jobs[cost.job_key(index, 16, order)] = dict(state="unresolved_at_cost_cap", terminal=True,
                                                      attempts=[dict(returncode=1)])
    (original / "status.json").write_text(json.dumps(dict(jobs=jobs)))
    monkeypatch.setattr(cost, "make_protocol", lambda folder: initial)
    monkeypatch.setattr(cost, "inspect_reference", lambda folder, protocol, index, prefix, order:
                        dict(terminal=True, draws_per_chain=4096,
                             report_path=str(cost.reference_path(folder, index, prefix, order))))
    protocol = continuation.prepare(tmp_path, original, out)
    assert len(protocol["copied_inputs"]) == 48
    for item in protocol["copied_inputs"]:
        assert (out / item["relative_path"]).read_bytes() == Path(item["source"]).read_bytes()
    retained = out / "chains/alpha16/seq_000000_t0016.npz"
    retained.write_bytes(b"later continued state")
    continuation.prepare(tmp_path, original, out)
    assert retained.read_bytes() == b"later continued state"
    retained.unlink()
    with pytest.raises(ValueError, match="refusing to reset"):
        continuation.prepare(tmp_path, original, out)


def test_budget_changes_only_draw_target_and_preserves_seed_warmup():
    args = continuation.command(Path("input"), Path("new"), 0, 16)
    for key, value in (("burn", "16384"), ("seed", "20260926"), ("chains", "4"),
                       ("thin", "1"), ("surrogate-steps", "64"),
                       ("initial-draws", "8192"), ("max-draws", "65536")):
        assert args[args.index("--" + key) + 1] == value


def test_even_locally_valid_copied_reports_must_extend_and_relocate(tmp_path):
    cost = continuation.cost
    inputs = {cost.case_key(0, 16): {"fixed": "inputs"}}
    code = {name: "hash" for name in cost.CODE_FILES}
    protocol = dict(source_protocol=dict(inputs=inputs, code_sha256=code))
    path = cost.reference_path(tmp_path, 0, 16, 16)
    path.parent.mkdir(parents=True)
    moments = path.with_suffix(".npz")
    moments.write_bytes(b"moments")
    identity = dict(inputs=inputs[cost.case_key(0, 16)], code_sha256=code)
    report = dict(identity=identity, reference_id=cost.fingerprint(identity),
                  settings=dict(alpha_order=16, seed=cost.seed_of(cost.SEED_TAG, 20260926, 0, 16, 16),
                                chains=4, burn=16384, thin=1, surrogate_steps=64),
                  moments_file=moments.name, moments_sha256=cost.sha_file(moments), local_validation_passed=True,
                  chain_checkpoint="/old/checkpoint.npz",
                  estimate=dict(diagnostics=dict(draws=4096), elapsed_seconds=1.0))
    path.write_text(json.dumps(report))
    assert not continuation.inspect(tmp_path, protocol, 0, 16)["terminal"]
    report["estimate"]["diagnostics"]["draws"] = 8192
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="external checkpoint"):
        continuation.inspect(tmp_path, protocol, 0, 16)
    report["chain_checkpoint"] = str(tmp_path / "chains/alpha16/seq_000000_t0016.npz")
    path.write_text(json.dumps(report))
    assert continuation.inspect(tmp_path, protocol, 0, 16)["terminal"]
