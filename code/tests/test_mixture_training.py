"""The pooled stream must preserve priors and resume the same computation."""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
import train_mixture
from families import ProductFamily
from mixture_source import (FAMILIES, IndexedMixture, completed_counts,
                            draw_training_batch, draw_validation, family_counts,
                            family_sources, stream_rng)
from source import Source


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_config():
    return dict(schema="mixture-training-v1", seed=17, d=64, n_ctx=4,
                batch=3, steps=6, warmup=2, lr=3e-4, wd=.1, clip=1.,
                n_layer=1, n_head=1, d_model=8, val_n=2, log_every=2,
                val_every=3, ckpt_every=2, val_chunk=1, weight_steps=[3, 6])


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            assert_nested_equal(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_existing_priors_are_reused_exactly_without_label_input():
    # Independently reconstruct each existing source and the declared order.
    seed, index, count, n_ctx = 19, 7, 7, 5
    originals = (ProductFamily(8, 8, n_ctx + 1, ls=(1, 8), alpha=0.0),
                 Source(64, n_ctx + 1, alpha=(.5, 1.5), ls=(0, 8), permute=True),
                 Source(64, n_ctx + 1, alpha=0., ls=(0, 8), permute=True))
    counts = family_counts(count, index)
    unshuffled = np.concatenate([s.batch(int(n), stream_rng(seed, index, 1, k))[0]
                                 for k, (s, n) in enumerate(zip(originals, counts))])
    order = stream_rng(seed, index, 2).permutation(count)
    tokens, labels = draw_training_batch(seed, index, count, n_ctx, return_family_ids=True)
    assert np.array_equal(tokens, unshuffled[order])
    assert np.array_equal(labels, np.repeat(np.arange(3), counts)[order])
    supplied = IndexedMixture(seed, count, n_ctx, index, index + 1)[0]
    assert isinstance(supplied, torch.Tensor) and supplied.shape == (count, n_ctx + 1)
    assert supplied.dtype == torch.int64 and int(supplied.min()) >= 0 and int(supplied.max()) < 64
    assert np.array_equal(supplied.numpy(), tokens)


def test_production_counts_and_no_32bit_batch_seed_alias():
    assert [family_counts(64, n).tolist() for n in range(3)] == [[22, 21, 21], [21, 22, 21], [21, 21, 22]]
    assert completed_counts(64, 300000) == dict.fromkeys(FAMILIES, 6400000)
    for n in range(10):
        expected = np.sum([family_counts(64, i) for i in range(n)], axis=0) if n else np.zeros(3)
        assert list(completed_counts(64, n).values()) == list(expected)
    assert not np.array_equal(stream_rng(0, 5, 1).integers(2**32, size=10),
                              stream_rng(0, 5 + 2**32, 1).integers(2**32, size=10))


def test_monitoring_has_separate_rng_namespace_and_is_fixed():
    first = draw_validation(4, 2, 25)
    second = draw_validation(4, 2, 25)
    for k, (name, src) in enumerate(zip(FAMILIES, family_sources(4))):
        expected = src.batch(2, stream_rng(25, 0, 3, k))[0]
        training = src.batch(2, stream_rng(25, 0, 1, k))[0]
        assert torch.equal(first[name], second[name])
        assert np.array_equal(first[name].numpy(), expected)
        assert not np.array_equal(expected, training)


def test_cpu_resume_equals_uninterrupted_with_different_worker_counts(tmp_path):
    cfg = tiny_config()
    uninterrupted, continued = tmp_path / "full", tmp_path / "continued"
    assert train_mixture.train(cfg, uninterrupted, workers=0) == 0
    assert train_mixture.train(cfg, continued, workers=0, max_steps=2) == 75
    middle = load(continued / "latest.pt")
    assert middle["step"] == 2 and middle["completed_counts"] == dict.fromkeys(FAMILIES, 2)
    assert train_mixture.train(cfg, continued, resume=True, workers=1) == 0
    a, b = load(uninterrupted / "latest.pt"), load(continued / "latest.pt")
    for key in ("model", "opt", "rng", "completed_counts", "validation_hashes", "config_hash"):
        assert_nested_equal(a[key], b[key])
    assert a["step"] == b["step"] == cfg["steps"]
    assert {p.name for p in continued.glob("*.pt")} == {"latest.pt", "model_3.pt", "model_6.pt"}
    assert load(continued / "model_6.pt")["cfg"]["model"]["vocab"] == 64


def test_refuse_restart_config_drift_and_implementation_drift(tmp_path, monkeypatch):
    cfg = tiny_config()
    out = tmp_path / "run"
    assert train_mixture.train(cfg, out, max_steps=1) == 75
    with pytest.raises(ValueError, match="nonempty"):
        train_mixture.train(cfg, out)
    with pytest.raises(ValueError, match="requires an existing"):
        train_mixture.train(cfg, tmp_path / "missing", resume=True)
    with pytest.raises(ValueError, match="config_hash"):
        train_mixture.train(dict(cfg, seed=18), out, resume=True)
    hashes = train_mixture.implementation_hashes()
    monkeypatch.setattr(train_mixture, "implementation_hashes", lambda: dict(hashes, **{"source.py": "changed"}))
    with pytest.raises(ValueError, match="implementation_hashes"):
        train_mixture.train(cfg, out, resume=True)


def test_stop_signal_saves_completed_step_and_can_resume(tmp_path, monkeypatch):
    cfg = tiny_config()
    original_step = torch.optim.AdamW.step

    def stopping_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        signal.raise_signal(signal.SIGUSR1)
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.optim.AdamW, "step", stopping_step)
        assert train_mixture.train(cfg, tmp_path / "signal") == 75
    ck = load(tmp_path / "signal" / "latest.pt")
    assert ck["step"] == 1 and ck["stop_reason"] == "SIGUSR1"
    assert train_mixture.train(cfg, tmp_path / "signal", resume=True) == 0


def test_concurrent_writer_is_refused(tmp_path):
    out = tmp_path / "locked"
    out.mkdir()
    with (out / ".training.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another trainer"):
            train_mixture.train(tiny_config(), out, max_steps=1)
    assert not (out / "latest.pt").exists()
    assert train_mixture.train(tiny_config(), out, max_steps=1) == 75


def test_resume_refuses_runtime_drift(tmp_path, monkeypatch):
    cfg = tiny_config()
    out = tmp_path / "runtime"
    assert train_mixture.train(cfg, out, max_steps=1) == 75
    runtime = train_mixture.runtime_contract(torch.device("cpu"))
    for key, changed in (("torch_version", "different"), ("numpy_version", "different"),
                         ("device", "cuda")):
        with monkeypatch.context() as scoped:
            scoped.setattr(train_mixture, "runtime_contract", lambda _d: dict(runtime, **{key: changed}))
            with pytest.raises(ValueError, match=f"runtime {key}"):
                train_mixture.train(cfg, out, resume=True)


def test_slurm_group_warning_with_spawned_worker_checkpoints_and_resumes(tmp_path):
    """Actually warn an isolated process group, not just the trainer PID."""
    cfg = dict(tiny_config(), steps=10000, log_every=1, val_every=10000,
               ckpt_every=5000, weight_steps=[10000])
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "group"
    transcript = tmp_path / "subprocess.log"
    command = [sys.executable, str(Path(train_mixture.__file__).resolve()),
               "--config", str(cfg_path), "--out", str(out), "--device", "cpu",
               "--workers", "1"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with transcript.open("w") as capture:
        child = subprocess.Popen(command, stdout=capture, stderr=capture,
                                 start_new_session=True, env=env)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                log = out / "log.jsonl"
                if log.exists() and log.stat().st_size:
                    break  # a real worker supplied at least one training batch
                if child.poll() is not None:
                    pytest.fail(transcript.read_text())
                time.sleep(.01)
            else:
                pytest.fail("trainer did not start within 30 seconds")
            os.killpg(child.pid, signal.SIGUSR1)
            assert child.wait(timeout=30) == 75, transcript.read_text()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    saved = load(out / "latest.pt")
    assert 1 <= saved["step"] < cfg["steps"]
    assert saved["stop_reason"] == "SIGUSR1"
    assert saved["completed_counts"] == completed_counts(cfg["batch"], saved["step"])
    assert "exited unexpectedly" not in transcript.read_text()
    assert "resource_tracker: process died" not in transcript.read_text()
    assert train_mixture.train(cfg, out, resume=True, workers=0,
                               max_steps=saved["step"] + 1) == 75
    assert load(out / "latest.pt")["step"] == saved["step"] + 1
