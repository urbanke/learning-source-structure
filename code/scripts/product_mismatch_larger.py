"""Evaluate existing larger product models on the original mismatch histories.

No source generation, training, or Bayesian reference calculation is performed.
Preparation binds every original record; execution is resumable by checked batch.
Numerical and provenance checks here do not replace independent result replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/product_shift_2026-09-18"
RECEIPTS = ROOT / "output/kuma_completion_2026-09-23/raw/specialist_transformer_large_2026-09-22"
MODEL_CODE = ROOT / "code/transformer/model.py"
SCHEMA = "product-mismatch-larger-v1"
SOURCE_PLAN = "f32337eebf7923d039c6ad705c6453a049b838cc1e39aa4abb70259d7ce1138e"
MODEL_SHA = "742fe0e5c8ffaecee8072a3b270b23beaa2e0bd2a94d1a27a9af67c31eff9c28"
TIMES = [1, 2, 4, 8, 16, 32, 64, 128, 192, 224, 240, 248, 255,
         256, 257, 260, 264, 272, 288, 320, 384, 448, 512, 640, 768, 896, 1024]
N_PER_DEPTH = 1024
LN2 = np.log(2.)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def check_probabilities(value, label, tolerance=2e-6, positive=True):
    require(np.isfinite(value).all() and np.all(value > 0 if positive else value >= 0),
            "Invalid probabilities: " + label)
    error = float(np.max(np.abs(value.sum(-1) - 1)))
    require(error <= tolerance, "Unnormalized probabilities: " + label)
    return error


def check_pairing(data):
    """Require the full original cohort, in a declared unambiguous order."""
    n = N_PER_DEPTH
    require(np.array_equal(data["times"], TIMES), "Prediction times changed")
    require(np.array_equal(data["depths"], np.repeat([1, 4], n)), "Depth coverage/order changed")
    require(np.array_equal(data["indices"], np.tile(np.arange(n), 2)), "Source indices incomplete or reordered")
    require(data["tokens"].shape == (2*n, 1025) and data["tokens"].dtype.kind in "iu"
            and np.all((data["tokens"] >= 0) & (data["tokens"] < 64)), "Token shape/support changed")
    for key in ("true_probs", "bayes_probs", "small_probs"):
        require(data[key].shape == (2*n, len(TIMES), 64), "Probability shape changed: " + key)
        check_probabilities(data[key], key, 2e-12 if key == "true_probs" else 2e-6,
                            positive=key != "true_probs")


def check_configuration(config, seed):
    expected = dict(family="product:8x8", d=64, n_ctx=1024, ls=[1, 8],
                    alpha=[0.5, 1.5], no_permute=False, steps=100000, batch=64,
                    seed=seed, n_layer=4, n_head=8, d_model=256)
    require(all(config.get(k) == v for k, v in expected.items()), "Larger checkpoint training family/configuration changed")
    require(config["model"] == dict(vocab=64, n_ctx=1024, n_layer=4, n_head=8,
                                    d_model=256, mlp=True, dropout=0.), "Larger architecture changed")


def stats(values):
    require(len(values) >= 2 and np.isfinite(values).all(), "Invalid sample for statistics")
    mean = values.mean(0)
    se = values.std(0, ddof=1) / np.sqrt(len(values))
    return dict(mean=mean.tolist(), se=se.tolist(),
                lower_pointwise_95=(mean-1.96*se).tolist(),
                upper_pointwise_95=(mean+1.96*se).tolist())


def risk(true, logp):
    # All zero-probability contributions are exactly zero; no probability clipping.
    log_true = np.zeros_like(true, dtype=float)
    np.log(true, out=log_true, where=true > 0)
    value = np.sum(true * (log_true - logp), axis=-1) / LN2
    require(np.isfinite(value).all() and value.min() >= -4e-6, "Invalid KL regret")
    return value


def quantities(data, logp):
    require(logp.shape == data["true_probs"].shape and np.isfinite(logp).all(), "Log probability shape/value changed")
    check_probabilities(np.exp(logp.astype(float)), "larger model")
    large = risk(data["true_probs"], logp.astype(float))
    bayes = risk(data["true_probs"], np.log(data["bayes_probs"]))
    small = risk(data["true_probs"], np.log(data["small_probs"]))
    gap = large - bayes
    difference = large - small
    require(np.max(np.abs(difference - (gap - (small-bayes)))) < 2e-12,
            "Paired gap difference identity failed")
    return dict(larger_regret_bits=large, bayes_regret_bits=bayes, small_regret_bits=small,
                larger_minus_bayes_bits=gap, small_minus_bayes_bits=small-bayes,
                gap_difference_larger_minus_small_bits=difference,
                kl_bayes_to_larger_bits=risk(data["bayes_probs"], logp.astype(float)),
                kl_bayes_to_small_bits=risk(data["bayes_probs"], np.log(data["small_probs"])),
                agreement_difference_larger_minus_small_bits=np.sum(
                    data["bayes_probs"]*(np.log(data["small_probs"])-logp.astype(float)),axis=-1)/LN2)


def aggregate(data, values):
    require(np.array_equal(data["depths"], np.repeat([1, 4], N_PER_DEPTH))
            and np.array_equal(data["indices"], np.tile(np.arange(N_PER_DEPTH), 2)),
            "Aggregation requires the full paired cohort")
    groups = []
    for depth in (1, 4):
        mask = data["depths"] == depth
        groups.append(dict(depth_pair=[depth, depth], condition="relabeled",
                           n_sequences=int(mask.sum()), metrics={k: stats(v[mask]) for k, v in values.items()}))
    return groups


def prepare(source, receipts, out):
    from product_shift_summary import read_record, risks
    require(not out.exists() or not any(out.iterdir()), "Preparation requires a fresh output directory")
    plan, summary, benchmark = (read(source/name) for name in ("plan.json", "summary.json", "benchmark.json"))
    require(plan["plan_id"] == SOURCE_PLAN == fingerprint({k:v for k,v in plan.items() if k != "plan_id"}), "Original source plan changed")
    require(plan["n_seq"] == N_PER_DEPTH and plan["times"] == TIMES
            and plan["pairs"] == [[1, 1], [4, 4]] and plan["conditions"][1] == "relabeled", "Original source specification changed")
    require(summary["complete"] and summary["plan_id"] == SOURCE_PLAN
            and summary["included_indices"] == list(range(N_PER_DEPTH))
            and benchmark["passed"] and benchmark["validation"]["passed"]
            and benchmark["plan_id"] == SOURCE_PLAN, "Original cohort/reference validation incomplete")
    require(plan["model"]["config"]["model"]["n_layer"] == 2 and sha(MODEL_CODE) == MODEL_SHA, "Baseline architecture/model implementation changed")
    paths = [source/"records"/f"L{d}_{d}"/f"seq_{i:06d}.npz" for d in (1,4) for i in range(N_PER_DEPTH)]
    require(all(p.is_file() for p in paths), "Missing original source records")
    rows = {k: [] for k in ("tokens", "true_probs", "bayes_probs", "small_probs")}
    raw_hash, raw_files = hashlib.sha256(), {}
    for depth in (1, 4):
        for i in range(N_PER_DEPTH):
            path = source/"records"/f"L{depth}_{depth}"/f"seq_{i:06d}.npz"
            record = read_record(path, plan, [depth, depth], i)
            digest = sha(path)
            relative = str(path.relative_to(source))
            raw_files[relative] = digest
            raw_hash.update(relative.encode()+b":"+bytes.fromhex(digest))
            risks(record)  # Replay all original probability/paired identities.
            for target, original in (("tokens","tokens"), ("true_probs","true_probs"),
                                     ("bayes_probs","bayes_probs"), ("small_probs","transformer_probs")):
                rows[target].append(record[original][1].copy())
            if (i+1) % 256 == 0:
                print(json.dumps(dict(event="original_records_verified", depth=depth, count=i+1)), flush=True)
    require(raw_hash.hexdigest() == summary["raw_records_sha256"], "Original record aggregate checksum changed")
    data = {k: np.asarray(v) for k,v in rows.items()}
    data.update(times=np.asarray(TIMES), depths=np.repeat([1,4],N_PER_DEPTH),
                indices=np.tile(np.arange(N_PER_DEPTH),2))
    check_pairing(data)
    old = quantities(data, np.log(data["small_probs"]))
    for group in aggregate(data, old):
        existing = next(g for g in summary["groups"] if g["depth_pair"] == group["depth_pair"] and g["condition"] == "relabeled")
        for new_name, old_name in (("small_regret_bits","transformer_regret_bits"),
                                    ("bayes_regret_bits","bayes_regret_bits"),
                                    ("small_minus_bayes_bits","transformer_minus_bayes_bits")):
            for statistic in ("mean", "se"):
                require(np.max(np.abs(np.asarray(group["metrics"][new_name][statistic])-existing["metrics"][old_name][statistic])) < 2e-12,
                        "Original paired summary does not replay")
    bindings = {}
    for seed in (0, 1):
        summary_path = receipts/f"eval/product/seed{seed}/summary.json"
        config_path = receipts/f"train/product/seed{seed}/config.json"
        status_path = config_path.parent/"status.json"
        prior, cfg, status = read(summary_path), read(config_path), read(status_path)
        check_configuration(cfg, seed)
        require(prior["complete"] and prior["family"] == "product" and prior["seed"] == seed
                and status["complete"] and status["passed"] and status["completed_steps"] == 100000
                and prior["plan_sha256"] == status["plan_sha256"], "Existing larger model completion/provenance changed")
        bindings[str(seed)] = dict(checkpoint_sha256=prior["checkpoint_sha256"], config=cfg,
            original_evaluation_sha256=sha(summary_path), config_sha256=sha(config_path),
            original_status_sha256=sha(status_path), training_plan_sha256=prior["plan_sha256"])
    manifest = dict(schema=SCHEMA, source_plan_id=SOURCE_PLAN, condition="relabeled", n_sequences=2048,
        source_root=str(source.resolve()), original_files=raw_files,
        source_metadata_sha256={name:sha(source/name) for name in ("plan.json","summary.json","benchmark.json")},
        original_record_aggregate_sha256=raw_hash.hexdigest(), model_code_sha256=MODEL_SHA,
        adapter_sha256=sha(__file__), original_reader_sha256=sha(Path(__file__).with_name("product_shift_summary.py")),
        times=TIMES, depths=[1,4], sequences_per_depth=N_PER_DEPTH, seeds=[0,1], models=bindings,
        baseline_model=plan["model"], no_retraining=True, no_new_sources=True, no_bayes_recomputation=True,
        probability_convention="float32 model and log_softmax, then float64 scoring; no renormalization or clipping",
        uncertainty="Paired differences and standard errors across original independent source histories, separately for each training seed.")
    out.mkdir(parents=True, exist_ok=True)
    save(out/"cohort.npz", **data)
    manifest["cohort_sha256"] = sha(out/"cohort.npz")
    manifest["manifest_id"] = fingerprint(manifest)
    write(out/"manifest.json", manifest)
    print(json.dumps(dict(event="prepared", out=str(out), n_sequences=2048,
                          manifest_id=manifest["manifest_id"])), flush=True)


def load_cohort(out):
    manifest = read(out/"manifest.json")
    require(manifest["schema"] == SCHEMA and manifest["source_plan_id"] == SOURCE_PLAN
            and manifest["manifest_id"] == fingerprint({k:v for k,v in manifest.items() if k != "manifest_id"}), "Manifest identity changed")
    require(manifest["adapter_sha256"] == sha(__file__) and manifest["model_code_sha256"] == sha(MODEL_CODE) == MODEL_SHA,
            "Frozen inference implementation changed")
    require(manifest["cohort_sha256"] == sha(out/"cohort.npz"), "Frozen cohort checksum changed")
    data = load_arrays(out/"cohort.npz")
    check_pairing(data)
    return manifest, data


def check_checkpoint(saved, checkpoint, binding, seed):
    require(sha(checkpoint) == binding["checkpoint_sha256"], "Checkpoint checksum changed")
    require(saved["step"] == 100000 and saved["cfg"] == binding["config"], "Checkpoint step/configuration changed")
    check_configuration(saved["cfg"], seed)


def run(out, models, batch):
    import torch
    sys.path.insert(0, str(ROOT/"code/transformer"))
    from model import GPT
    require(torch.cuda.is_available(), "This bounded evaluator requires the allocated CUDA GPU")
    require(1 <= batch <= 32, "Batch must be between 1 and 32")
    manifest, data = load_cohort(out)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    all_summaries = []
    for seed in (0, 1):
        started = time.monotonic()
        checkpoint = models/f"train/product/seed{seed}/model_100000.pt"
        binding = manifest["models"][str(seed)]
        require(sha(checkpoint) == binding["checkpoint_sha256"], "Checkpoint checksum changed before loading")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        check_checkpoint(saved, checkpoint, binding, seed)
        model = GPT(**saved["cfg"]["model"]).float().cuda().eval()
        model.load_state_dict(saved["model"])
        del saved
        dest = out/f"seed{seed}"
        dest.mkdir(exist_ok=True)
        identity = dict(manifest_id=manifest["manifest_id"], seed=seed,
            checkpoint_sha256=binding["checkpoint_sha256"], adapter_sha256=sha(__file__),
            model_code_sha256=MODEL_SHA, batch=batch, torch=str(torch.__version__),
            numpy=np.__version__, device=torch.cuda.get_device_name(), tf32=False, autocast=False,
            model_dtype="float32", log_probability_dtype="float32")
        if (dest/"receipt.json").exists():
            require(read(dest/"receipt.json")["identity"] == identity, "Existing evaluation identity changed; use a fresh output")
        else:
            require(not any(dest.iterdir()), "Unbound existing evaluation files")
            write(dest/"receipt.json", dict(identity=identity, checkpoint_path=str(checkpoint),
                                            independent_replay_passed=False))
        chunks, causality = [], []
        # Separate-prefix inference checks at both depths, before any scores release.
        with torch.inference_mode():
            for index in (0, N_PER_DEPTH):
                x = torch.as_tensor(data["tokens"][index:index+1].astype(np.int64), device="cuda")
                full = model(x[:,:-1]).float()
                errors = {str(t):float((model(x[:,:t]).float()[:,-1]-full[:,t-1]).abs().max().item())
                          for t in (1,32,256)}
                require(max(errors.values()) <= 2e-5, "Separate-prefix causal alignment failed")
                causality.append(dict(index=index, depth=int(data["depths"][index]),
                                       max_logit_errors=errors, tolerance=2e-5))
        for left in range(0, len(data["tokens"]), batch):
            right = min(left+batch,len(data["tokens"]))
            path = dest/f"batch_{left:06d}.npz"
            if path.exists():
                require(path.with_suffix(".json").is_file()
                        and read(path.with_suffix(".json"))["sha256"] == sha(path), "Saved inference batch checksum changed")
                chunk = load_arrays(path)
                require(str(chunk["identity_id"]) == fingerprint(identity)
                        and np.array_equal(chunk["row_indices"],np.arange(left,right)), "Saved inference batch identity/pairing changed")
                selected = chunk["logp"]
            else:
                torch.cuda.synchronize()
                tick = time.monotonic()
                with torch.inference_mode():
                    x = torch.as_tensor(data["tokens"][left:right].astype(np.int64), device="cuda")
                    lp = torch.log_softmax(model(x[:,:-1]).float(), dim=-1)
                    selected = lp[:,np.asarray(TIMES)-1,:].cpu().numpy().astype(np.float32)
                    target = lp.gather(2,x[:,1:,None])[:,:,0].cpu().numpy()
                require(np.array_equal(np.take_along_axis(selected,data["tokens"][left:right,np.asarray(TIMES),None],2)[:,:,0],
                                       target[:,np.asarray(TIMES)-1]), "Selected target/time alignment failed")
                torch.cuda.synchronize()
                seconds = time.monotonic()-tick
                if left == 0:
                    print(json.dumps(dict(event="inference_pilot",seed=seed,rows=right-left,
                        seconds=seconds,projected_seconds_this_seed=seconds*len(data["tokens"])/(right-left))), flush=True)
                save(path, identity_id=fingerprint(identity), row_indices=np.arange(left,right), logp=selected)
                write(path.with_suffix(".json"),dict(sha256=sha(path),identity_id=fingerprint(identity),left=left,right=right))
            require(selected.dtype == np.float32 and selected.shape == (right-left,len(TIMES),64), "Saved inference batch shape/dtype changed")
            check_probabilities(np.exp(selected.astype(float)), "saved model batch")
            chunks.append(selected)
            progress = dict(event="inference_progress",seed=seed,completed_rows=right,total_rows=len(data["tokens"]),
                            elapsed_seconds=time.monotonic()-started,manifest_id=manifest["manifest_id"])
            write(dest/"status.json",progress)
            if left == 0 or right % 128 == 0 or right == len(data["tokens"]):
                print(json.dumps(progress),flush=True)
        logp = np.concatenate(chunks)
        values = quantities(data,logp)
        scores = dict(logp=logp,times=data["times"],depths=data["depths"],indices=data["indices"],**values)
        score_path = dest/"scores.npz"
        if score_path.exists():
            previous = load_arrays(score_path)
            require(set(previous) == set(scores) and all(np.array_equal(previous[k],v) for k,v in scores.items()), "Completed score output changed")
        else:
            save(score_path,**scores)
        result = dict(schema=SCHEMA,complete=True,seed=seed,identity=identity,
            manifest_sha256=sha(out/"manifest.json"),scores_sha256=sha(score_path),
            causality=causality,probability_sum_max_error=check_probabilities(np.exp(logp.astype(float)),"complete scores"),
            groups=aggregate(data,values),times=TIMES,independent_replay_passed=False,
            publication_status="awaiting_independent_replay",no_retraining=True,no_bayes_recomputation=True,
            uncertainty=manifest["uncertainty"])
        write(dest/"summary.json",result)
        write(dest/"status.json",dict(complete=True,seed=seed,completed_rows=2048,
                                      summary_sha256=sha(dest/"summary.json"),elapsed_seconds=time.monotonic()-started))
        all_summaries.append(dict(seed=seed,summary=f"seed{seed}/summary.json",sha256=sha(dest/"summary.json")))
        print(json.dumps(dict(event="seed_complete",seed=seed,elapsed_seconds=time.monotonic()-started)),flush=True)
        del model
        torch.cuda.empty_cache()
    write(out/"status.json",dict(complete=True,seeds=all_summaries,manifest_sha256=sha(out/"manifest.json"),
                                 independent_replay_passed=False,publication_status="awaiting_independent_replay"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","run"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--receipts",type=Path,default=RECEIPTS)
    parser.add_argument("--models",type=Path,default=ROOT/"output/specialist_transformer_large_2026-09-22")
    parser.add_argument("--batch",type=int,default=8)
    args = parser.parse_args()
    print(json.dumps(dict(event="mismatch_larger_start",mode=args.mode,out=str(args.out))),flush=True)
    if args.mode == "prepare":
        prepare(args.source,args.receipts,args.out)
    else:
        run(args.out,args.models,args.batch)


if __name__ == "__main__":
    main()
