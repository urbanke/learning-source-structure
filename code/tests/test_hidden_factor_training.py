"""Reproducible training, disjoint monitoring, and bounded safe continuation."""
from pathlib import Path
import fcntl
import json
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
import train_hidden_factor as training
from hidden_factor import draw_hidden_factor_batch


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config():
    return dict(schema="hidden-factor-training-v1", seed=17, source_dimensions=[2, 3],
                n_ctx=5, batch=3, steps=6, warmup=2, lr=3e-4, wd=.1, clip=1.,
                n_layer=1, n_head=1, d_model=8, val_n=3, log_every=2,
                val_every=3, ckpt_every=2, val_chunk=1, weight_steps=[3, 6])


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            same(left, right)
    else:
        assert a == b


def test_exact_budget_and_all_candidate_vocabularies():
    directory = Path(training.__file__).parent / "configs"
    paths = sorted(directory.glob("hidden_factor_seed*.json"))
    assert len(paths) == 2
    for path in paths:
        template = json.loads(path.read_text())
        assert "source_dimensions" not in template  # explicit choice after capacity check
        cfg = training.validate_config(dict(template, source_dimensions=[2, 9]))
        assert cfg["steps"] * cfg["batch"] == 6_400_000
        assert cfg["n_ctx"] + 1 == 1025
        assert (cfg["n_layer"], cfg["n_head"], cfg["d_model"]) == (2, 4, 128)
        assert cfg["seed"] in (0, 1)
        assert cfg["source_dimensions"] == [2, 9]


def test_indexed_stream_tokens_only_and_disjoint_monitoring():
    seed, index, dimensions, batch, n_ctx = 25, 7, [3, 4], 32, 12
    data = training.IndexedHiddenFactor(seed, dimensions, batch, n_ctx, index, index + 1)[0]
    expected = draw_hidden_factor_batch(training.stream_rng(seed, index, 1), batch, n_ctx + 1, *dimensions)
    assert isinstance(data, torch.Tensor) and data.shape == (batch, n_ctx + 1)
    assert data.dtype == torch.int64 and np.array_equal(data.numpy(), expected)
    assert 0 <= data.min() <= data.max() < 12
    val1 = training.draw_validation(dimensions, n_ctx, batch, seed)["hidden_factor"]
    val2 = training.draw_validation(dimensions, n_ctx, batch, seed)["hidden_factor"]
    assert torch.equal(val1, val2)
    assert not np.array_equal(val1.numpy(), training.draw_training_batch(seed, 0, dimensions, batch, n_ctx))
    assert not np.array_equal(training.stream_rng(seed, index, 1).integers(2**32, size=20),
                              training.stream_rng(seed, index + 2**32, 1).integers(2**32, size=20))


def test_cpu_continuation_matches_uninterrupted_across_workers(tmp_path):
    cfg = config()
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert training.train(cfg, full) == 0
    assert training.train(cfg, resumed, max_steps=2) == 75
    assert load(resumed / "latest.pt")["completed_sequences"] == 6
    assert training.train(cfg, resumed, resume=True, workers=1) == 0
    left, right = load(full / "latest.pt"), load(resumed / "latest.pt")
    for key in ("model", "opt", "rng", "validation_hashes", "config_hash", "completed_sequences"):
        same(left[key], right[key])
    assert left["step"] == right["step"] == 6
    assert {p.name for p in resumed.glob("*.pt")} == {"latest.pt", "model_3.pt", "model_6.pt"}
    weights = load(resumed / "model_6.pt")
    assert weights["schema"] == "hidden-factor-weights-v1"
    assert weights["cfg"]["model"]["vocab"] == 6
    assert weights["completed_sequences"] == 18
    assert json.loads((resumed / "status.json").read_text())["complete"]


def test_refuse_restart_config_code_runtime_drift_and_concurrent_writer(tmp_path, monkeypatch):
    cfg, out = config(), tmp_path / "run"
    assert training.train(cfg, out, max_steps=1) == 75
    with pytest.raises(ValueError, match="nonempty"):
        training.train(cfg, out)
    with pytest.raises(ValueError, match="requires an existing"):
        training.train(cfg, tmp_path / "absent", resume=True)
    with pytest.raises(ValueError, match="config_hash"):
        training.train(dict(cfg, source_dimensions=[3, 3]), out, resume=True)
    hashes = training.implementation_hashes()
    with monkeypatch.context() as scope:
        scope.setattr(training, "implementation_hashes", lambda: dict(hashes, changed="x"))
        with pytest.raises(ValueError, match="implementation_hashes"):
            training.train(cfg, out, resume=True)
    runtime = training.runtime_contract(torch.device("cpu"))
    with monkeypatch.context() as scope:
        scope.setattr(training, "runtime_contract", lambda _: dict(runtime, torch_version="changed"))
        with pytest.raises(ValueError, match="runtime torch_version"):
            training.train(cfg, out, resume=True)
    with (out / ".training.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another trainer"):
            training.train(cfg, out, resume=True)


def test_signal_stop_keeps_completed_step_and_continues(tmp_path, monkeypatch):
    real_step = torch.optim.AdamW.step

    def stop(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        signal.raise_signal(signal.SIGUSR1)
        return result

    with monkeypatch.context() as scope:
        scope.setattr(torch.optim.AdamW, "step", stop)
        assert training.train(config(), tmp_path) == 75
    saved = load(tmp_path / "latest.pt")
    assert saved["step"] == 1 and saved["stop_reason"] == "SIGUSR1"
    assert saved["completed_sequences"] == 3
    assert training.train(config(), tmp_path, resume=True) == 0


def test_launcher_is_dry_by_default_and_requires_selected_shape(tmp_path):
    script = Path(training.__file__).parent / "kuma" / "submit_hidden_factor.sh"
    target = tmp_path / "absent"
    result = subprocess.run(["bash", str(script), "--benchmark", "--out", str(target)],
                            capture_output=True, text=True, check=True)
    assert "sbatch" in result.stdout and "hidden_factor_benchmark.sbatch" in result.stdout
    assert not target.exists()
    refused = subprocess.run(["bash", str(script), "--out", str(target)],
                             capture_output=True, text=True)
    assert refused.returncode == 2 and "Select --shape" in refused.stderr
    for mode in ([], ["--training-benchmark"]):
        selected = subprocess.run(["bash", str(script), "--shape", "2x9", "--out", str(target)] + mode,
                                  capture_output=True, text=True, check=True)
        assert "hidden_factor_train.sbatch" in selected.stdout
    evaluation = subprocess.run(["bash", str(script), "--evaluate", "--shape", "2x9", "--out", str(target)],
                                capture_output=True, text=True, check=True)
    assert "hidden_factor_eval.sbatch" in evaluation.stdout
    assert "--array" not in evaluation.stdout
    assert not target.exists()
