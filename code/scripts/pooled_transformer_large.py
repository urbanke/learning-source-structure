"""Bounded larger pooled-model timing, training and frozen-cohort evaluation.

The existing numerical trainer and source stream are reused without edits.
Benchmark weights are never used to initialize production.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configuration(plan, seed):
    if seed not in (0, 1) or plan["schema"] != "pooled-transformer-large-plan-v1":
        raise ValueError("Unexpected larger-model plan or seed")
    base_path = ROOT / f"code/transformer/configs/multifamily_seed{seed}.json"
    if sha(base_path) != plan["baseline_config_sha256"][str(seed)]:
        raise ValueError("Original pooled configuration changed")
    baseline = read(base_path)
    config = dict(baseline, **plan["architecture"], experiment=plan["experiment"])
    for name, expected in plan["implementation_sha256"].items():
        if sha(ROOT / "code/transformer" / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "train"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args()
    print(json.dumps(dict(event="pooled_large_start", mode=args.mode, seed=args.seed)), flush=True)
    plan = read(args.plan)
    config = configuration(plan, args.seed)
    submission = read(args.out / f"{args.mode}_submission.json")
    for name, expected in submission["implementation_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Launch implementation changed: {name}")
    if args.sources is None or sha(args.sources/"manifest.json") != submission["source_manifest_sha256"]:
        raise ValueError("Original frozen cohort manifest changed after submission")
    if args.mode == "benchmark" and args.seed != 0:
        raise ValueError("The independent benchmark uses seed 0")
    import torch
    import train_mixture as trainer
    from model import GPT
    from mixture_source import draw_validation, completed_counts
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError("A CUDA allocation is required")
    root = args.out.resolve()
    run = root / ("benchmark/seed0" if args.mode == "benchmark" else f"train/seed{args.seed}")
    if run.exists():
        raise ValueError("Fresh output required; inspect any old allocation before resuming")
    if args.mode == "train":
        benchmark = read(root / "benchmark/benchmark.json")
        if not benchmark["passed"] or benchmark["plan_sha256"] != sha(args.plan):
            raise ValueError("A passed matching benchmark is required")
        if args.sources is None or not (args.sources / "manifest.json").is_file():
            raise ValueError("The original pooled held-out cohort is required")
    measured_writes = []
    original_save = trainer.atomic_save

    def timed_save(path, writer):
        start = time.monotonic()
        original_save(path, writer)
        if str(path).endswith(".pt"):
            measured_writes.append(dict(file=Path(path).name, seconds=time.monotonic()-start))

    trainer.atomic_save = timed_save
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    code = trainer.train(config, run, device="cuda", workers=6,
                         max_steps=plan["benchmark_steps"] if args.mode == "benchmark" else None)
    training_wall = time.monotonic()-start
    state, receipt = read(run / "status.json"), read(run / "config.json")
    if state["completed_counts"] != completed_counts(config["batch"], state["step"]):
        raise ValueError("Completed family counts differ")
    if receipt["n_params"] != plan["n_params"]:
        raise ValueError("Larger parameter count differs")
    runtime = dict(plan_sha256=sha(args.plan), mode=args.mode, seed=args.seed,
                   source_manifest_sha256=submission["source_manifest_sha256"],
                   implementation_sha256=submission["implementation_sha256"],
                   cuda_device=torch.cuda.get_device_name(0), training_wall_seconds=training_wall,
                   checkpoint_writes=measured_writes, return_code=code,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(root / f"{args.mode}/runtime_seed{args.seed}.json", runtime)
    if args.mode == "benchmark":
        if code != 75 or state["step"] != plan["benchmark_steps"] or state["reason"] != "max_steps":
            raise SystemExit(code or 75)
        records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        events = [r for r in records if r["event"] == "train"]
        if len(events) < 2 or not all(math.isfinite(e["loss_bits"]) for e in events):
            raise ValueError("Incomplete or nonfinite benchmark logs")
        saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        model = GPT(**receipt["model"]).cuda()
        model.load_state_dict(saved["model"])
        del saved
        validation = draw_validation(config["n_ctx"], config["val_n"], config["validation_seed"])
        torch.cuda.synchronize()
        validation_start = time.monotonic()
        scores = trainer.evaluate(model, validation, torch.device("cuda"), config["val_chunk"])
        torch.cuda.synchronize()
        validation_seconds = time.monotonic()-validation_start
        if not all(math.isfinite(v["loss_bits"]) for v in scores.values()):
            raise ValueError("Nonfinite monitoring losses")
        first, last = events[0], events[-1]
        seconds_per_step = (last["run_elapsed_seconds"]-first["run_elapsed_seconds"])/(last["step"]-first["step"])
        checkpoint_seconds = max(r["seconds"] for r in measured_writes)
        # Conservative accounting of periodic/final monitoring and weight saves.
        projected = seconds_per_step*config["steps"] + validation_seconds*(config["steps"]//config["val_every"]+1)
        projected += checkpoint_seconds*(config["steps"]//config["ckpt_every"]+len(config["weight_steps"])+2)
        result = dict(runtime, passed=True, completed_steps=state["step"],
                      seconds_per_step=seconds_per_step, validation_seconds=validation_seconds,
                      validation=scores, checkpoint_seconds=checkpoint_seconds,
                      projected_training_hours_including_monitoring=projected/3600,
                      projection_scope="Measured training, full monitoring pass and checkpoint I/O; scheduling/evaluation add time.")
        write(root / "benchmark/benchmark.json", result)
        print(json.dumps(dict(event="pooled_large_benchmark_complete", **result)), flush=True)
        return 0
    if code or not state["complete"]:
        return code or 75
    # Score the predeclared final checkpoint on the unchanged pooled cohort.
    from eval_mixture import evaluate_checkpoint
    result = evaluate_checkpoint(args.sources, run / "model_300000.pt",
                                 f"pooled_large_seed{args.seed}", root / "eval",
                                 device="cuda", batch_size=16)
    write(root / "eval" / f"complete_seed{args.seed}.json",
          dict(complete=True, plan_sha256=sha(args.plan), run_id=result["run_id"],
               cohorts=len(result["cohorts"])))
    print(json.dumps(dict(event="pooled_large_training_and_evaluation_complete", seed=args.seed)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
