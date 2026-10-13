"""Controlled ancestry, indexed continuation and unchanged large-arm training."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import signal
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
import train_hidden_factor as original
import train_hidden_factor_scaling as scaling
from model import GPT


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


@pytest.fixture
def prepared(tmp_path):
    template = dict(schema="hidden-factor-training-v1", n_ctx=5, batch=3, steps=2,
        warmup=1, lr=3e-4, wd=.1, clip=1., n_layer=1, n_head=1, d_model=8,
        val_n=3, log_every=1, val_every=2, ckpt_every=1, val_chunk=1,
        validation_seed=20260917, weight_steps=[2])
    cfg = original.validate_config(dict(template, seed=0, source_dimensions=[2, 3]))
    parent_dir = tmp_path / "baseline"
    assert original.train(cfg, parent_dir) == 0
    long_cfg = dict(template, steps=6, weight_steps=[4, 6])
    large_cfg = dict(template, n_layer=2, n_head=2, d_model=16)
    plan = dict(schema="hidden-factor-scaling-plan-v1", source_dimensions=[2, 3], seeds=[0, 1],
        implementation_sha256=original.implementation_hashes(),
        baseline=dict(config=template, final_step=2, completed_sequences=6,
                      checkpoint_sha256={"0": scaling.file_hash(parent_dir / "model_2.pt"), "1": "0" * 64}),
        arms=dict(longsmall=dict(mode="continuation", config=long_cfg, parent_config=template,
                parent_step=2, parent_completed_sequences=6, additional_sequences=12,
                continuation_lr=3e-5, n_params=GPT(6, 5, 1, 1, 8).n_params()),
            large=dict(mode="fresh", config=large_cfg, n_params=GPT(6, 5, 2, 2, 16).n_params())),
        primary=dict(steps=dict(longsmall=6, large=2)))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, parent_dir / "latest.pt", cfg


def test_actual_plan_budgets_and_model_sizes():
    path = Path(scaling.__file__).parent / "configs/hidden_factor_scaling_plan_2026-09-16.json"
    plan, _ = scaling.read_plan(path)
    for arm, count, params in [("longsmall", 25_600_000, 531968), ("large", 6_400_000, 3429888)]:
        cfg = scaling.planned_config(plan, arm, 0)
        assert cfg["steps"] * cfg["batch"] == count
        assert GPT(16, cfg["n_ctx"], cfg["n_layer"], cfg["n_head"], cfg["d_model"]).n_params() == params


def test_continuation_matches_uninterrupted_reference_and_resumed_workers(prepared, tmp_path, monkeypatch):
    path, parent, base = prepared
    parent_bytes = parent.read_bytes()
    weights_bytes = parent.with_name("model_2.pt").read_bytes()
    whole, split, reference = [tmp_path / name for name in ("whole", "split", "reference")]
    assert scaling.train(path, "longsmall", 0, whole, parent_checkpoint=parent) == 0
    assert scaling.train(path, "longsmall", 0, split, parent_checkpoint=parent, max_steps=3) == 75
    assert scaling.train(path, "longsmall", 0, split, parent_checkpoint=parent, resume=True, workers=1) == 0
    plan, _ = scaling.read_plan(path)
    reference_cfg = scaling.planned_config(plan, "longsmall", 0)
    original_lr = original.lr_at
    with monkeypatch.context() as scope:
        scope.setattr(original, "lr_at", lambda cfg, step: original_lr(base, step) if step < base["steps"] else 3e-5)
        assert original.train(reference_cfg, reference) == 0
    left = load(whole / "latest.pt")
    for location in (split, reference):
        right = load(location / "latest.pt")
        for key in ("model", "opt", "rng", "validation_hashes", "completed_sequences", "completed_batches"):
            same(left[key], right[key])
    assert parent.read_bytes() == parent_bytes
    assert parent.with_name("model_2.pt").read_bytes() == weights_bytes
    assert {p.name for p in whole.glob("*.pt")} == {"latest.pt", "model_4.pt", "model_6.pt"}
    assert load(whole / "model_6.pt")["schema"] == scaling.WEIGHTS_SCHEMA
    provenance = left["cfg"]["config"]["scaling_provenance"]
    assert provenance["parent"]["checkpoint_sha256"] == scaling.file_hash(parent)
    assert provenance["parent"]["weights_checkpoint_sha256"] == scaling.file_hash(parent.with_name("model_2.pt"))
    assert left["cfg"]["learning_rate_schedule"]["value"] == 3e-5
    status = json.loads((whole / "status.json").read_text())
    assert status["completed_sequences"] == 18 and status["additional_sequences"] == 12


def test_large_delegates_same_source_and_schedule_and_records_runtime(prepared, tmp_path):
    path, parent, _ = prepared
    plan, _ = scaling.read_plan(path)
    cfg = scaling.planned_config(plan, "large", 0)
    raw, scaled = tmp_path / "raw", tmp_path / "scaled"
    assert original.train(cfg, raw) == 0
    assert scaling.train(path, "large", 0, scaled, max_steps=1) == 75
    runtime = json.loads((scaled / "scaling_runtime.json").read_text())
    assert runtime["return_code"] == 75 and runtime["step"] == 1 and runtime["reason"] == "max_steps"
    assert runtime["device"] == "cpu" and runtime["max_memory_allocated_bytes"] is None
    assert scaling.train(path, "large", 0, scaled, resume=True) == 0
    for key in ("model", "opt", "rng", "completed_sequences"):
        same(load(raw / "latest.pt")[key], load(scaled / "latest.pt")[key])
    assert load(scaled / "model_2.pt")["schema"] == "hidden-factor-weights-v1"
    assert load(scaled / "model_2.pt")["cfg"]["config"]["scaling_provenance"]["parent"] is None
    with pytest.raises(ValueError, match="start fresh"):
        scaling.train(path, "large", 0, tmp_path / "bad-large", parent_checkpoint=parent)


@pytest.mark.parametrize("change,match", [
    (lambda s: s.update(step=1), "completed"),
    (lambda s: s.pop("opt"), "Adam state missing"),
    (lambda s: s.update(rng={}), "RNG state missing"),
    (lambda s: s.update(implementation_hashes={}), "implementation hashes"),
    (lambda s: s["cfg"]["runtime"].update(torch_version="different"), "runtime differs"),
    (lambda s: s.update(validation_hashes={}), "monitoring cohort differs"),
    (lambda s: s["model"]["tok.weight"].add_(1), "model tensors differ"),
    (lambda s: s["opt"]["state"][0]["step"].zero_(), "Adam state.*inconsistent"),
    (lambda s: s["opt"]["param_groups"][0].update(weight_decay=.5), "optimizer settings differ"),
    (lambda s: s["opt"]["state"][0]["exp_avg"].fill_(float('nan')), "Adam state.*inconsistent"),
])
def test_reject_bad_parent(prepared, tmp_path, change, match):
    path, parent, _ = prepared
    value = load(parent); change(value); torch.save(value, parent)
    with pytest.raises(ValueError, match=match):
        scaling.train(path, "longsmall", 0, tmp_path / "bad", parent_checkpoint=parent)


def test_reject_stale_plan_and_weight_identity(prepared, tmp_path):
    path, parent, _ = prepared
    plan = json.loads(path.read_text())
    plan["implementation_sha256"]["model.py"] = "changed"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="implementation hashes"):
        scaling.train(path, "large", 0, tmp_path / "bad-code")
    plan["implementation_sha256"] = original.implementation_hashes()
    path.write_text(json.dumps(plan))
    with parent.with_name("model_2.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="evaluated baseline"):
        scaling.train(path, "longsmall", 0, tmp_path / "bad-weight", parent_checkpoint=parent)


def test_safe_output_resume_hashes_and_writer_lock(prepared, tmp_path, monkeypatch):
    path, parent, _ = prepared
    out = tmp_path / "continued"
    assert scaling.train(path, "longsmall", 0, out, parent_checkpoint=parent, max_steps=3) == 75
    with pytest.raises(ValueError, match="separate output"):
        scaling.train(path, "longsmall", 0, parent.parent, parent_checkpoint=parent)
    with pytest.raises(ValueError, match="nonempty"):
        scaling.train(path, "longsmall", 0, out, parent_checkpoint=parent)
    with pytest.raises(ValueError, match="existing latest"):
        scaling.train(path, "longsmall", 0, tmp_path / "missing", parent_checkpoint=parent, resume=True)
    old_plan = path.read_bytes(); path.write_bytes(old_plan + b"\n")
    with pytest.raises(ValueError, match="config_hash"):
        scaling.train(path, "longsmall", 0, out, parent_checkpoint=parent, resume=True)
    path.write_bytes(old_plan)
    real_hash = scaling.file_hash
    with monkeypatch.context() as scope:
        scope.setattr(scaling, "file_hash", lambda p: "changed" if str(p) == scaling.__file__ else real_hash(p))
        with pytest.raises(ValueError, match="config_hash"):
            scaling.train(path, "longsmall", 0, out, parent_checkpoint=parent, resume=True)
    current = load(out / "latest.pt")
    current["cfg"]["runtime"]["torch_version"] = "changed"
    torch.save(current, out / "latest.pt")
    with pytest.raises(ValueError, match="runtime differs"):
        scaling.train(path, "longsmall", 0, out, parent_checkpoint=parent, resume=True)
    with (out / ".training.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another trainer"):
            scaling.train(path, "longsmall", 0, out, parent_checkpoint=parent, resume=True)


def test_signal_checkpoint_resumes(prepared, tmp_path, monkeypatch):
    path, parent, _ = prepared
    step = torch.optim.AdamW.step
    def stop(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        signal.raise_signal(signal.SIGUSR1)
        return result
    with monkeypatch.context() as scope:
        scope.setattr(torch.optim.AdamW, "step", stop)
        assert scaling.train(path, "longsmall", 0, tmp_path / "signal", parent_checkpoint=parent) == 75
    value = load(tmp_path / "signal/latest.pt")
    assert value["step"] == 3 and value["stop_reason"] == "SIGUSR1"
    assert scaling.train(path, "longsmall", 0, tmp_path / "signal", parent_checkpoint=parent, resume=True) == 0
