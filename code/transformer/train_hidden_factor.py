"""Train the existing GPT on a fresh hidden joint encoding per sequence.

Every update uses a batch indexed by (seed, completed step), so worker count
and restart do not change its examples. Monitoring sequences use a separate
fixed stream. Training sees tokens only, never the encoding or latent states.

Exit 0 means all configured updates completed. Exit 75 means a safe capped
or signal stop; --resume continues the same config, code and numerical runtime.
One atomic rolling checkpoint and explicitly scheduled weights bound storage.
This module reuses the existing trainer's tested checkpoint/signal utilities
without changing any code used by the ongoing mixture experiment.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from hidden_factor import draw_hidden_factor_batch
from model import GPT
from train_mixture import (atomic_save, digest_json, evaluate, initialize_worker,
                           loader_iterator, lr_at, restore_rng, rng_state,
                           runtime_contract, write_json)

CHECKPOINT_SCHEMA = "hidden-factor-checkpoint-v1"
REQUIRED = {"schema", "seed", "source_dimensions", "n_ctx", "batch", "steps",
            "warmup", "lr", "wd", "clip", "n_layer", "n_head", "d_model",
            "val_n", "log_every", "val_every", "ckpt_every"}


def validate_config(config):
    cfg = dict(config)
    missing = REQUIRED - cfg.keys()
    if missing:
        raise ValueError(f"missing config fields: {sorted(missing)}")
    if cfg["schema"] != "hidden-factor-training-v1":
        raise ValueError("requires hidden-factor-training-v1")
    dims = cfg["source_dimensions"]
    if not isinstance(dims, (list, tuple)) or len(dims) != 2 or any(
            type(x) is not int or x < 2 for x in dims):
        raise ValueError("source_dimensions must contain two integers >= 2")
    if math.prod(dims) > 255:
        raise ValueError("source alphabet must contain at most 255 tokens")
    cfg["source_dimensions"] = list(dims)
    for key in ("seed", "n_ctx", "batch", "steps", "warmup", "n_layer", "n_head",
                "d_model", "val_n", "log_every", "val_every", "ckpt_every"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid positive integer {key}")
    if cfg["d_model"] % cfg["n_head"]:
        raise ValueError("model width must be divisible by heads")
    if cfg["seed"] >= 2**63 or cfg["warmup"] > cfg["steps"]:
        raise ValueError("invalid seed or warmup > steps")
    for key in ("lr", "wd", "clip"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"invalid nonnegative scalar {key}")
    if cfg["lr"] == 0:
        raise ValueError("learning rate must be positive")
    cfg.setdefault("validation_seed", 20260917)
    cfg.setdefault("val_chunk", 32)
    cfg.setdefault("weight_steps", [cfg["steps"]])
    if type(cfg["validation_seed"]) is not int or not 0 <= cfg["validation_seed"] < 2**64:
        raise ValueError("invalid validation_seed")
    if type(cfg["val_chunk"]) is not int or cfg["val_chunk"] < 1:
        raise ValueError("invalid val_chunk")
    if not isinstance(cfg["weight_steps"], list) or any(
            type(n) is not int or not 0 < n <= cfg["steps"] for n in cfg["weight_steps"]):
        raise ValueError("weight_steps must be a bounded list within the run")
    cfg["weight_steps"] = sorted(set(cfg["weight_steps"] + [cfg["steps"]]))
    return cfg


def source_contract(dimensions):
    return {"version": "hidden-factor-dirichlet1-v1", "dimensions": list(dimensions),
            "rows": "independent Dirichlet(1,...,1) for each coordinate",
            "initial_states": "independent uniform coordinates",
            "encoding": "fresh uniform joint-pair-to-token permutation per sequence",
            "parameters": "fresh independent transitions and encoding per sequence",
            "model_input": "encoded tokens only", "sampling": "indexed-batch-v1"}


def implementation_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("hidden_factor.py", "model.py", "train_hidden_factor.py", "train_mixture.py")}


def stream_rng(seed, index, purpose):
    if any(type(x) is not int or not 0 <= x < 2**64 for x in (seed, index)):
        raise ValueError("seed/index must be unsigned 64-bit integers")
    if type(purpose) is not int or not 0 <= purpose < 2**32:
        raise ValueError("purpose must be an unsigned 32-bit integer")
    return np.random.default_rng(np.random.SeedSequence([
        0x48463131, seed & 0xFFFFFFFF, seed >> 32,
        index & 0xFFFFFFFF, index >> 32, purpose]))


def draw_training_batch(seed, index, dimensions, batch_size, n_ctx):
    r, s = dimensions
    return draw_hidden_factor_batch(stream_rng(seed, index, 1), batch_size, n_ctx + 1, r, s)


def draw_validation(dimensions, n_ctx, n_sequences, seed):
    r, s = dimensions
    tokens = draw_hidden_factor_batch(stream_rng(seed, 0, 2), n_sequences, n_ctx + 1, r, s)
    return {"hidden_factor": torch.from_numpy(tokens)}


class IndexedHiddenFactor(torch.utils.data.Dataset):
    def __init__(self, seed, dimensions, batch_size, n_ctx, start, stop):
        self.seed, self.dimensions = seed, dimensions
        self.batch_size, self.n_ctx = batch_size, n_ctx
        self.start, self.stop = start, stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return torch.from_numpy(draw_training_batch(self.seed, self.start + index,
                               self.dimensions, self.batch_size, self.n_ctx))


def train(config, out, *, resume=False, device="cpu", workers=0, max_steps=None):
    """Hold one exclusive writer lock for the entire run, including resume."""
    validate_config(config)
    out = Path(out)
    if resume and not (out / "latest.pt").is_file():
        raise ValueError("resume requires an existing latest.pt; will not restart training")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another trainer holds this output directory") from exc
        try:
            return _train_locked(config, out, resume=resume, device=device,
                                 workers=workers, max_steps=max_steps)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def _train_locked(config, out, *, resume=False, device="cpu", workers=0, max_steps=None):
    cfg = validate_config(config)
    if workers < 0 or (max_steps is not None and max_steps < 0):
        raise ValueError("workers and max_steps must be nonnegative")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    target_device = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = Path(out)
    latest = out / "latest.pt"
    hashes = implementation_hashes()
    runtime = runtime_contract(target_device)
    config_hash = digest_json(cfg)
    contract = source_contract(cfg["source_dimensions"])
    contract_hash = digest_json(contract)
    checkpoint = None
    if resume:
        if not latest.is_file():
            raise ValueError("resume requires an existing latest.pt; will not restart training")
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        for key, expected in (("schema", CHECKPOINT_SCHEMA), ("config_hash", config_hash),
                              ("implementation_hashes", hashes), ("source_contract_hash", contract_hash)):
            if checkpoint.get(key) != expected:
                raise ValueError(f"resume refused: {key} differs")
        for key, expected in runtime.items():
            if checkpoint.get("cfg", {}).get("runtime", {}).get(key) != expected:
                raise ValueError(f"resume refused: runtime {key} differs")
        if checkpoint["completed_sequences"] != cfg["batch"] * checkpoint["step"]:
            raise ValueError("resume refused: inconsistent completed sequence count")
        if checkpoint["step"] < 0 or checkpoint["step"] > cfg["steps"] or checkpoint.get("completed_batches") != checkpoint["step"]:
            raise ValueError("resume refused: invalid completed step")
    else:
        if out.exists() and any(p.name != ".training.lock" for p in out.iterdir()):
            raise ValueError("output directory is nonempty; use a new directory or --resume")
        out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"] % 2**32)
    random.seed(cfg["seed"])
    model = GPT(math.prod(cfg["source_dimensions"]), cfg["n_ctx"], cfg["n_layer"], cfg["n_head"], cfg["d_model"], mlp=True).to(target_device)
    model.float()
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": cfg["wd"]},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95))
    step = 0
    total_elapsed = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        step = checkpoint["step"]
        total_elapsed = checkpoint.get("training_elapsed_seconds", 0.0)
        restore_rng(checkpoint["rng"])
    validation = draw_validation(cfg["source_dimensions"], cfg["n_ctx"], cfg["val_n"], cfg["validation_seed"])
    val_hashes = {name: hashlib.sha256(x.numpy().tobytes()).hexdigest() for name, x in validation.items()}
    if checkpoint is not None and checkpoint["validation_hashes"] != val_hashes:
        raise ValueError("resume refused: monitoring cohort drift")
    receipt = {"config": cfg, "config_hash": config_hash, "model": model.cfg,
               "n_params": model.n_params(), "source_contract": contract,
               "source_contract_hash": contract_hash, "implementation_hashes": hashes,
               "validation_hashes": val_hashes, "torch_version": str(torch.__version__),
               "numpy_version": np.__version__, "precision": "float32", "runtime": runtime,
               "device": str(target_device)}
    if checkpoint is None:
        write_json(out / "config.json", receipt)
    stopped = {"signal": None}
    old_handlers = {}

    def request_stop(signum, _frame):
        stopped["signal"] = signal.Signals(signum).name

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        old_handlers[sig] = signal.signal(sig, request_stop)
    start_time = time.monotonic()

    def save_checkpoint(reason):
        saved = {"schema": CHECKPOINT_SCHEMA, "model": model.state_dict(),
                 "opt": optimizer.state_dict(), "step": step, "cfg": receipt,
                 "config_hash": config_hash, "implementation_hashes": hashes,
                 "source_contract_hash": contract_hash, "rng": rng_state(),
                 "completed_batches": step, "completed_sequences": cfg["batch"] * step,
                 "validation_hashes": val_hashes, "stop_reason": reason,
                 "training_elapsed_seconds": total_elapsed + time.monotonic() - start_time}
        atomic_save(latest, lambda f: torch.save(saved, f))
        if step in cfg["weight_steps"]:
            weights = {"schema": "hidden-factor-weights-v1",
                       "model": saved["model"], "step": step, "cfg": receipt,
                       "config_hash": config_hash, "implementation_hashes": hashes,
                       "completed_sequences": saved["completed_sequences"]}
            atomic_save(out / f"model_{step}.pt", lambda f: torch.save(weights, f))
        write_json(out / "status.json", {"step": step, "production_steps": cfg["steps"],
                  "complete": step == cfg["steps"], "reason": reason,
                  "completed_sequences": saved["completed_sequences"], "config_hash": config_hash,
                  "training_elapsed_seconds": saved["training_elapsed_seconds"]})

    try:
        if checkpoint is None:
            save_checkpoint("initialized")
        stop_step = min(cfg["steps"], max_steps) if max_steps is not None else cfg["steps"]
        if stop_step < step:
            raise ValueError("max_steps is below the already completed step")
        dataset = IndexedHiddenFactor(cfg["seed"], cfg["source_dimensions"], cfg["batch"], cfg["n_ctx"], step, stop_step)
        # DataLoader bookkeeping gets a private generator; worker/prefetch changes
        # must never consume the model's global RNG stream.
        loader_generator = torch.Generator().manual_seed(171717)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers,
                    generator=loader_generator, persistent_workers=False,
                    worker_init_fn=initialize_worker,
                    multiprocessing_context="spawn" if workers else None,
                    timeout=120 if workers else 0,
                    pin_memory=(device == "cuda"))
        model.train()
        running_loss = 0.0
        running_steps = 0
        with (out / "log.jsonl").open("a", buffering=1) as log:
            for xb in loader_iterator(loader):
                if stopped["signal"]:
                    break
                learning_rate = lr_at(cfg, step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                xb = xb.to(target_device, non_blocking=(device == "cuda"))
                logits = model(xb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, math.prod(cfg["source_dimensions"])), xb[:, 1:].reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at batch {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg["clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                running_loss += float(loss.detach())
                running_steps += 1
                if step % cfg["log_every"] == 0:
                    rec = {"event": "train", "step": step, "loss_bits": running_loss / running_steps / math.log(2),
                           "lr": learning_rate, "completed_sequences": cfg["batch"] * step,
                           "run_elapsed_seconds": time.monotonic() - start_time}
                    log.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                    running_loss = 0.0
                    running_steps = 0
                # Stop signals take precedence over potentially slow monitoring.
                if not stopped["signal"] and (step % cfg["val_every"] == 0 or step == cfg["steps"]):
                    rec = {"event": "validation", "step": step,
                           "val": evaluate(model, validation, target_device, cfg["val_chunk"])}
                    log.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                if step % cfg["ckpt_every"] == 0 or step in cfg["weight_steps"]:
                    save_checkpoint("periodic")
                if stopped["signal"]:
                    break
        reason = "completed" if step == cfg["steps"] else (stopped["signal"] or "max_steps")
        save_checkpoint(reason)
        print(json.dumps({"event": "stop", "step": step, "reason": reason}), flush=True)
        return 0 if step == cfg["steps"] else 75
    finally:
        for sig, old_handler in old_handlers.items():
            signal.signal(sig, old_handler)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dimensions", type=int, nargs=2, required=True,
                        metavar=("R", "S"), help="selected source dimensions, recorded in the run config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        config = json.load(f)
    if "source_dimensions" in config and config["source_dimensions"] != args.dimensions:
        parser.error("--dimensions conflicts with source_dimensions in the config")
    config["source_dimensions"] = args.dimensions
    return train(config, args.out, resume=args.resume, device=args.device,
                 workers=args.workers, max_steps=args.max_steps)

if __name__ == "__main__":
    sys.exit(main())
