"""Controlled hidden-factor follow-ups without changing the original trainer.

The large arm delegates to that trainer. The longsmall arm continues a verified
completed optimizer checkpoint at a fixed learning rate, using the next indexed
source batches. Its separate checkpoint schema records the original ancestry.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

import train_hidden_factor as original
from model import GPT
from train_mixture import (atomic_save, digest_json, evaluate, initialize_worker,
                           loader_iterator, restore_rng, rng_state,
                           runtime_contract, write_json)

CHECKPOINT_SCHEMA = "hidden-factor-scaling-checkpoint-v1"
WEIGHTS_SCHEMA = "hidden-factor-scaling-weights-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def planned_config(plan, arm, seed):
    require(arm in ("longsmall", "large") and seed in plan["seeds"], "unplanned arm or seed")
    template = plan["arms"][arm]["config"]
    require("seed" not in template and "source_dimensions" not in template
            and "scaling_provenance" not in template, "plan config must exclude injected identities")
    return original.validate_config(dict(template, seed=seed,
                                         source_dimensions=plan["source_dimensions"]))


def baseline_config(plan, seed):
    return original.validate_config(dict(plan["baseline"]["config"], seed=seed,
                                         source_dimensions=plan["source_dimensions"]))


def read_plan(path):
    payload = Path(path).read_bytes()
    plan = json.loads(payload)
    require(plan["schema"] == "hidden-factor-scaling-plan-v1", "unexpected scaling plan schema")
    require(plan["seeds"] == [0, 1], "scaling plan requires both declared seeds")
    require(plan["implementation_sha256"] == original.implementation_hashes(),
            "plan implementation hashes differ from the immutable original modules")
    require(plan["arms"]["longsmall"]["mode"] == "continuation"
            and plan["arms"]["large"]["mode"] == "fresh", "incorrect scaling arm modes")
    for seed in plan["seeds"]:
        base = baseline_config(plan, seed)
        small = planned_config(plan, "longsmall", seed)
        large = planned_config(plan, "large", seed)
        arm = plan["arms"]["longsmall"]
        require(arm["parent_config"] == plan["baseline"]["config"], "parent config differs from baseline")
        require(arm["parent_step"] == plan["baseline"]["final_step"] == base["steps"]
                and arm["parent_completed_sequences"] == plan["baseline"]["completed_sequences"]
                == base["steps"] * base["batch"], "parent training budget mismatch")
        require(small["steps"] > base["steps"] and arm["additional_sequences"]
                == (small["steps"] - base["steps"]) * base["batch"], "continuation budget mismatch")
        require({k: v for k, v in small.items() if k not in ("steps", "weight_steps")}
                == {k: v for k, v in base.items() if k not in ("steps", "weight_steps")},
                "continuation changes original training settings")
        require(math.isclose(arm["continuation_lr"], .1 * base["lr"], rel_tol=1e-14)
                and arm["continuation_lr"] > 0, "continuation must retain the original learning-rate floor")
        varying = ("n_layer", "n_head", "d_model", "weight_steps")
        require({k: v for k, v in large.items() if k not in varying}
                == {k: v for k, v in base.items() if k not in varying},
                "large arm changes settings beyond the declared architecture")
        require(plan["primary"]["steps"] == {"longsmall": small["steps"], "large": large["steps"]},
                "primary checkpoint mismatch")
    return plan, hashlib.sha256(payload).hexdigest()


def _parent_checkpoint(plan, seed, path, device, parent_weights=None):
    path = Path(path)
    require(path.is_file(), "continuation requires the original full checkpoint")
    cfg = baseline_config(plan, seed)
    weights_path = Path(parent_weights) if parent_weights else path.with_name(f"model_{cfg['steps']}.pt")
    require(weights_path.is_file(), "continuation requires the original evaluated weight checkpoint")
    weights_hash = file_hash(weights_path)
    require(weights_hash == plan["baseline"]["checkpoint_sha256"][str(seed)],
            "parent weight checkpoint differs from the evaluated baseline")
    checkpoint_hash = file_hash(path)
    parent = torch.load(path, map_location="cpu", weights_only=False)
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    require(parent.get("schema") == original.CHECKPOINT_SCHEMA
            and weights.get("schema") == "hidden-factor-weights-v1",
            "parent requires original full optimizer and weight checkpoint schemas")
    contract = original.source_contract(cfg["source_dimensions"])
    expected_model = dict(vocab=math.prod(cfg["source_dimensions"]), n_ctx=cfg["n_ctx"],
                          n_layer=cfg["n_layer"], n_head=cfg["n_head"], d_model=cfg["d_model"],
                          mlp=True, dropout=0.0)
    for value in (parent, weights):
        receipt = value["cfg"]
        require(receipt["config"] == cfg and receipt["model"] == expected_model,
                "parent configuration or model differs from plan")
        require(value["config_hash"] == receipt["config_hash"] == digest_json(cfg),
                "parent configuration hash mismatch")
        require(value["implementation_hashes"] == receipt["implementation_hashes"]
                == plan["implementation_sha256"], "parent implementation hashes differ")
        require(receipt["source_contract"] == contract
                and receipt["source_contract_hash"] == digest_json(contract), "parent source contract mismatch")
        require(value["step"] == cfg["steps"]
                and value["completed_sequences"] == cfg["batch"] * cfg["steps"],
                "parent must have completed the original training budget")
        require(receipt["runtime"] == runtime_contract(device), "parent runtime differs")
    require(parent["source_contract_hash"] == digest_json(contract)
            and parent["completed_batches"] == cfg["steps"]
            and parent["stop_reason"] == "completed", "parent is not a completed baseline")
    require(parent["model"].keys() == weights["model"].keys()
            and all(torch.equal(value, weights["model"][name]) for name, value in parent["model"].items()),
            "parent model tensors differ from the evaluated baseline weights")
    require(set(parent.get("rng", {})) == {"torch", "numpy", "python", "cuda"}, "parent RNG state missing")
    require(isinstance(parent.get("opt"), dict) and parent["opt"].get("state")
            and len(parent["opt"].get("param_groups", [])) == 2, "parent Adam state missing")
    validation = original.draw_validation(cfg["source_dimensions"], cfg["n_ctx"],
                                         cfg["val_n"], cfg["validation_seed"])
    validation_hashes = {name: hashlib.sha256(x.numpy().tobytes()).hexdigest()
                         for name, x in validation.items()}
    require(parent["validation_hashes"] == parent["cfg"]["validation_hashes"]
            == weights["cfg"]["validation_hashes"] == validation_hashes,
            "parent monitoring cohort differs")
    ancestry = dict(checkpoint_sha256=checkpoint_hash, weights_checkpoint_sha256=weights_hash,
                    schema=parent["schema"], config_hash=parent["config_hash"], step=parent["step"],
                    completed_sequences=parent["completed_sequences"],
                    implementation_hashes=parent["implementation_hashes"],
                    runtime=parent["cfg"]["runtime"], validation_hashes=validation_hashes)
    return parent, ancestry, validation


def train(plan_path, arm, seed, out, *, parent_checkpoint=None, parent_weights=None,
          resume=False, device="cpu", workers=0, max_steps=None):
    plan, plan_hash = read_plan(plan_path)
    cfg = planned_config(plan, arm, seed)
    require(device in ("cpu", "cuda"), "device must be cpu or cuda")
    require(workers >= 0 and (max_steps is None or max_steps >= 0), "negative worker count or step cap")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    provenance = dict(plan_sha256=plan_hash, arm=arm, seed=seed,
                      trainer_sha256=file_hash(__file__), parent=None)
    if arm == "large":
        require(parent_checkpoint is None and parent_weights is None, "large arm must start fresh")
        cfg["scaling_provenance"] = provenance
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = original.train(cfg, out, resume=resume, device=device, workers=workers, max_steps=max_steps)
        state = json.loads((Path(out) / "status.json").read_text())
        if device == "cuda":
            torch.cuda.synchronize()
        write_json(Path(out) / "scaling_runtime.json", dict(
            schema="hidden-factor-scaling-runtime-v1", plan_sha256=plan_hash, arm=arm, seed=seed,
            device=device, cuda_device=torch.cuda.get_device_name() if device == "cuda" else None,
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
            max_memory_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
            max_memory_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
            step=state["step"], reason=state["reason"], complete=state["complete"],
            max_steps=max_steps, return_code=result, training_elapsed_seconds=state["training_elapsed_seconds"]))
        return result
    require(parent_checkpoint is not None, "longsmall requires --parent-checkpoint")
    parent, ancestry, validation = _parent_checkpoint(plan, seed, parent_checkpoint,
                                                       torch.device(device), parent_weights)
    provenance["parent"] = ancestry
    cfg["scaling_provenance"] = provenance
    out = Path(out)
    require(out.resolve() != Path(parent_checkpoint).parent.resolve(), "continuation requires a separate output directory")
    if resume:
        require((out / "latest.pt").is_file(), "resume requires an existing latest.pt")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another trainer holds this output directory") from exc
        try:
            return _continue_locked(plan, cfg, out, parent, validation, resume=resume,
                                    device=torch.device(device), workers=workers, max_steps=max_steps)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _continue_locked(plan, cfg, out, parent, validation, *, resume, device, workers, max_steps):
    hashes = original.implementation_hashes()
    config_hash = digest_json(cfg)
    runtime = runtime_contract(device)
    contract = original.source_contract(cfg["source_dimensions"])
    validation_hashes = parent["validation_hashes"]
    saved = parent
    if resume:
        saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        for key, expected in (("schema", CHECKPOINT_SCHEMA), ("config_hash", config_hash),
                              ("implementation_hashes", hashes), ("source_contract_hash", digest_json(contract)),
                              ("validation_hashes", validation_hashes)):
            require(saved.get(key) == expected, f"resume refused: {key} differs")
        require(saved["cfg"]["config"] == cfg and saved["cfg"]["runtime"] == runtime,
                "resume refused: configuration or runtime differs")
        require(saved["cfg"]["config_hash"] == config_hash
                and saved["cfg"]["implementation_hashes"] == hashes
                and saved["cfg"]["validation_hashes"] == validation_hashes,
                "resume refused: checkpoint receipt differs")
        require(parent["step"] <= saved["step"] <= cfg["steps"]
                and saved["completed_batches"] == saved["step"]
                and saved["completed_sequences"] == saved["step"] * cfg["batch"],
                "resume refused: invalid completed training count")
    else:
        require(not any(p.name != ".training.lock" for p in out.iterdir()),
                "output directory is nonempty; use a new directory or --resume")
    stop_step = min(cfg["steps"], max_steps) if max_steps is not None else cfg["steps"]
    require(stop_step >= saved["step"], "max_steps is below the completed step")
    model = GPT(math.prod(cfg["source_dimensions"]), cfg["n_ctx"], cfg["n_layer"],
                cfg["n_head"], cfg["d_model"], mlp=True).to(device).float()
    require(model.n_params() == plan["arms"]["longsmall"]["n_params"], "planned parameter count differs")
    model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": cfg["wd"]},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(.9, .95))
    expected_groups = optimizer.state_dict()["param_groups"]
    require(len(saved["opt"]["param_groups"]) == len(expected_groups), "saved optimizer group count differs")
    for expected, actual in zip(expected_groups, saved["opt"]["param_groups"]):
        require({k: v for k, v in actual.items() if k != "lr"}
                == {k: v for k, v in expected.items() if k != "lr"},
                "saved optimizer settings differ from the original Adam")
    optimizer.load_state_dict(saved["opt"])
    for group in optimizer.param_groups:
        require(group["betas"] == (.9, .95), "saved optimizer is not the original Adam")
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                    and state["step"].item() == saved["step"]
                    and state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape
                    and torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all(),
                    "saved Adam state is incomplete or inconsistent")
    restore_rng(saved["rng"])
    step = saved["step"]
    total_elapsed = saved.get("training_elapsed_seconds", 0.)
    receipt = dict(config=cfg, config_hash=config_hash, model=model.cfg, n_params=model.n_params(),
                   source_contract=contract, source_contract_hash=digest_json(contract),
                   implementation_hashes=hashes, validation_hashes=validation_hashes,
                   torch_version=str(torch.__version__), numpy_version=np.__version__,
                   precision="float32", runtime=runtime, device=str(device),
                   learning_rate_schedule=dict(kind="constant continuation", value=plan["arms"]["longsmall"]["continuation_lr"],
                                               starts_after_step=parent["step"]))
    if not resume:
        write_json(out / "config.json", receipt)
    stopped = {"signal": None}
    handlers = {}

    def request_stop(signum, _frame):
        stopped["signal"] = signal.Signals(signum).name

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        handlers[sig] = signal.signal(sig, request_stop)
    start_time = time.monotonic()

    def save_checkpoint(reason):
        value = dict(schema=CHECKPOINT_SCHEMA, model=model.state_dict(), opt=optimizer.state_dict(),
                     step=step, cfg=receipt, config_hash=config_hash, implementation_hashes=hashes,
                     source_contract_hash=digest_json(contract), rng=rng_state(), completed_batches=step,
                     completed_sequences=cfg["batch"] * step, validation_hashes=validation_hashes,
                     stop_reason=reason, training_elapsed_seconds=total_elapsed + time.monotonic() - start_time)
        atomic_save(out / "latest.pt", lambda f: torch.save(value, f))
        if step in cfg["weight_steps"]:
            weights = {key: value[key] for key in ("model", "step", "cfg", "config_hash", "implementation_hashes", "completed_sequences")}
            weights["schema"] = WEIGHTS_SCHEMA
            atomic_save(out / f"model_{step}.pt", lambda f: torch.save(weights, f))
        write_json(out / "status.json", dict(step=step, production_steps=cfg["steps"], complete=step == cfg["steps"],
            reason=reason, completed_sequences=value["completed_sequences"],
            additional_sequences=(step - parent["step"]) * cfg["batch"], config_hash=config_hash,
            plan_sha256=cfg["scaling_provenance"]["plan_sha256"], arm="longsmall",
            training_elapsed_seconds=value["training_elapsed_seconds"]))

    try:
        if not resume:
            save_checkpoint("initialized from verified baseline")
        dataset = original.IndexedHiddenFactor(cfg["seed"], cfg["source_dimensions"], cfg["batch"],
                                               cfg["n_ctx"], step, stop_step)
        generator = torch.Generator().manual_seed(171717)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers,
            generator=generator, persistent_workers=False, worker_init_fn=initialize_worker,
            multiprocessing_context="spawn" if workers else None, timeout=120 if workers else 0,
            pin_memory=device.type == "cuda")
        model.train()
        running_loss = 0.; running_steps = 0
        with (out / "log.jsonl").open("a", buffering=1) as log:
            for xb in loader_iterator(loader):
                if stopped["signal"]:
                    break
                learning_rate = plan["arms"]["longsmall"]["continuation_lr"]
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                xb = xb.to(device, non_blocking=device.type == "cuda")
                logits = model(xb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, math.prod(cfg["source_dimensions"])), xb[:, 1:].reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at batch {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg["clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"], error_if_nonfinite=True)
                optimizer.step()
                step += 1; running_loss += float(loss.detach()); running_steps += 1
                if step % cfg["log_every"] == 0:
                    row = dict(event="train", step=step, loss_bits=running_loss / running_steps / math.log(2),
                               lr=learning_rate, completed_sequences=cfg["batch"] * step,
                               additional_sequences=(step - parent["step"]) * cfg["batch"],
                               run_elapsed_seconds=time.monotonic() - start_time)
                    log.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
                    running_loss = 0.; running_steps = 0
                if not stopped["signal"] and (step % cfg["val_every"] == 0 or step == cfg["steps"]):
                    row = dict(event="validation", step=step, val=evaluate(model, validation, device, cfg["val_chunk"]))
                    log.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
                if step % cfg["ckpt_every"] == 0 or step in cfg["weight_steps"]:
                    save_checkpoint("periodic")
                if stopped["signal"]:
                    break
        reason = "completed" if step == cfg["steps"] else (stopped["signal"] or "max_steps")
        save_checkpoint(reason)
        print(json.dumps(dict(event="stop", step=step, reason=reason)), flush=True)
        return 0 if step == cfg["steps"] else 75
    finally:
        for sig, old in handlers.items():
            signal.signal(sig, old)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=("longsmall", "large"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-weights", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return train(args.plan, args.arm, args.seed, args.out, parent_checkpoint=args.parent_checkpoint,
                 parent_weights=args.parent_weights, resume=args.resume, device=args.device,
                 workers=args.workers, max_steps=args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
