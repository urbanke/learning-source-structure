"""Train one existing GPT on the three hidden-family priors, in float32.

CLI::
    python code/transformer/train_mixture.py --config CONFIG.json --out RUN --device cuda --workers 6
    python code/transformer/train_mixture.py --config CONFIG.json --out RUN --resume --device cuda --workers 6

``--max-steps N`` is an absolute completed-step cap for smoke/continuation
checks; it does not alter the production learning-rate schedule. Exit 0 means
the configured production run completed; exit 75 means a safe checkpoint was
written after a signal or cap. SIGUSR1/SIGTERM request a stop at the next safe
boundary. A new run refuses a nonempty output directory. Resume requires an
atomic latest.pt checkpoint with matching config and implementation hashes.
For a Slurm warning sent to the entire process group, use SIGUSR1: workers
ignore it and the trainer checkpoints. Workers retain normal SIGTERM handling
so shutdown stays bounded. A forced group SIGTERM can interrupt a worker and
requires the previous atomic checkpoint; it is not the graceful-warning path.

Storage is bounded: one rolling optimizer checkpoint plus weights at the
explicit weight_steps (and final weights). No test set is used in training.
Resume requires the same Torch/NumPy versions, device type and arithmetic
settings. Bitwise CPU continuation is tested; GPU kernel nondeterminism can
still prevent bitwise equality even with identical source batches and RNGs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from mixture_source import (FAMILIES, SOURCE_CONTRACT, IndexedMixture,
                            completed_counts, draw_validation)
from model import GPT

CHECKPOINT_SCHEMA = "mixture-checkpoint-v1"
WINDOWS = ((1, 32), (33, 128), (129, 512), (513, 1024))
REQUIRED = {"schema", "seed", "d", "n_ctx", "batch", "steps", "warmup", "lr",
            "wd", "clip", "n_layer", "n_head", "d_model", "val_n", "log_every",
            "val_every", "ckpt_every"}


def validate_config(config):
    cfg = dict(config)
    missing = REQUIRED - cfg.keys()
    if missing:
        raise ValueError(f"missing config fields: {sorted(missing)}")
    if cfg["schema"] != "mixture-training-v1" or cfg["d"] != 64:
        raise ValueError("requires mixture-training-v1 with shared alphabet d=64")
    ints = ("seed", "d", "n_ctx", "batch", "steps", "warmup", "n_layer", "n_head",
            "d_model", "val_n", "log_every", "val_every", "ckpt_every")
    for key in ints:
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid positive integer {key}")
    if cfg["batch"] < 3 or cfg["d_model"] % cfg["n_head"]:
        raise ValueError("batch must be >=3 and model width divisible by heads")
    if cfg["seed"] >= 2**63 or cfg["warmup"] > cfg["steps"]:
        raise ValueError("invalid seed or warmup > steps")
    for key in ("lr", "wd", "clip"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"invalid nonnegative scalar {key}")
    if cfg["lr"] == 0:
        raise ValueError("learning rate must be positive")
    cfg.setdefault("validation_seed", 20260916)
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


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def implementation_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("source.py", "families.py", "model.py", "mixture_source.py", "train_mixture.py")}


def runtime_contract(device):
    return {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "device": str(device), "cuda_version": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "num_threads": torch.get_num_threads()}


def initialize_worker(_worker_id):
    """Slurm's warning reaches children too; only the trainer handles it."""
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    # The parent blocks USR1 during spawn to protect child import/startup and
    # multiprocessing's resource tracker. Now this worker can safely unblock.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})
    # Keep SIGTERM at its normal disposition: DataLoader uses it to terminate
    # workers that do not finish during its bounded graceful cleanup.


def loader_iterator(loader):
    """Spawn children with USR1 blocked, without losing a parent warning."""
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
    try:
        return iter(loader)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def atomic_save(path, writer):
    """Write, fsync and replace within the destination filesystem."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    atomic_save(path, lambda f: f.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()))


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("cannot resume a CUDA RNG checkpoint on a CPU-only host")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def lr_at(cfg, step):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    fraction = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, fraction))))


@torch.no_grad()
def evaluate(model, validation, device, chunk):
    model.eval()
    result = {}
    try:
        for name, xs in validation.items():
            losses = []
            for start in range(0, len(xs), chunk):
                xb = xs[start:start + chunk].to(device)
                logits = model(xb[:, :-1]).float()
                lp = F.log_softmax(logits, dim=-1)
                losses.append((-lp.gather(2, xb[:, 1:, None])[:, :, 0] / math.log(2)).cpu())
            bits = torch.cat(losses)
            result[name] = {"loss_bits": float(bits.mean()), "n_sequences": len(xs),
                            "windows": {f"{lo}-{min(hi, bits.shape[1])}":
                                float(bits[:, lo - 1:min(hi, bits.shape[1])].mean())
                                for lo, hi in WINDOWS if lo <= bits.shape[1]}}
    finally:
        model.train()
    return result


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
    contract_hash = digest_json(SOURCE_CONTRACT)
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
        if checkpoint["completed_counts"] != completed_counts(cfg["batch"], checkpoint["step"]):
            raise ValueError("resume refused: inconsistent completed family counts")
        if checkpoint["step"] < 0 or checkpoint["step"] > cfg["steps"] or checkpoint.get("completed_batches") != checkpoint["step"]:
            raise ValueError("resume refused: invalid completed step")
    else:
        if out.exists() and any(p.name != ".training.lock" for p in out.iterdir()):
            raise ValueError("output directory is nonempty; use a new directory or --resume")
        out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"] % 2**32)
    random.seed(cfg["seed"])
    model = GPT(64, cfg["n_ctx"], cfg["n_layer"], cfg["n_head"], cfg["d_model"], mlp=True).to(target_device)
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
    validation = draw_validation(cfg["n_ctx"], cfg["val_n"], cfg["validation_seed"])
    val_hashes = {name: hashlib.sha256(x.numpy().tobytes()).hexdigest() for name, x in validation.items()}
    if checkpoint is not None and checkpoint["validation_hashes"] != val_hashes:
        raise ValueError("resume refused: monitoring cohort drift")
    receipt = {"config": cfg, "config_hash": config_hash, "model": model.cfg,
               "n_params": model.n_params(), "source_contract": SOURCE_CONTRACT,
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
                 "completed_batches": step, "completed_counts": completed_counts(cfg["batch"], step),
                 "validation_hashes": val_hashes, "stop_reason": reason,
                 "training_elapsed_seconds": total_elapsed + time.monotonic() - start_time}
        atomic_save(latest, lambda f: torch.save(saved, f))
        if step in cfg["weight_steps"]:
            weights = {"model": saved["model"], "step": step, "cfg": receipt,
                       "config_hash": config_hash, "implementation_hashes": hashes,
                       "completed_counts": saved["completed_counts"]}
            atomic_save(out / f"model_{step}.pt", lambda f: torch.save(weights, f))
        write_json(out / "status.json", {"step": step, "production_steps": cfg["steps"],
                  "complete": step == cfg["steps"], "reason": reason,
                  "completed_counts": saved["completed_counts"], "config_hash": config_hash,
                  "training_elapsed_seconds": saved["training_elapsed_seconds"]})

    try:
        if checkpoint is None:
            save_checkpoint("initialized")
        stop_step = min(cfg["steps"], max_steps) if max_steps is not None else cfg["steps"]
        if stop_step < step:
            raise ValueError("max_steps is below the already completed step")
        dataset = IndexedMixture(cfg["seed"], cfg["batch"], cfg["n_ctx"], step, stop_step)
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
                loss = F.cross_entropy(logits.reshape(-1, 64), xb[:, 1:].reshape(-1))
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
                           "lr": learning_rate, "completed_counts": completed_counts(cfg["batch"], step),
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        config = json.load(f)
    return train(config, args.out, resume=args.resume, device=args.device,
                 workers=args.workers, max_steps=args.max_steps)


if __name__ == "__main__":
    sys.exit(main())
