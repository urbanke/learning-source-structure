"""Shared-depth Bayesian references on an explicit, immutable token set.

The uniform-base chain mixes one depth over all context rows; product
chains mix one independent depth per coordinate. This deliberately writes
a new schema, never the historical *_famprior.npz files.

Example:
    python refs_shared_depth.py --tokens evaluation.npz --kind uniform --d 64 \
        --positions 1,2,4,8,16,32,64,128,256,512,1024 \
        --out output/shared_depth/reference.npz

Positions are observed prefix lengths: position 1 predicts the second
token. The first token is observed but omitted from the stored loss.
Set LSA_STORE and PMM_SRC to use the existing read-only anchor store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from lsa_seq import SeqLSA
from shared_depth import SharedDepthLSA, ProductSharedDepthLSA

SCHEMA = "shared_depth_v1"


def token_hash(tokens):
    x = np.ascontiguousarray(tokens, dtype="<i8")
    return hashlib.sha256(x.tobytes()).hexdigest()


def parse_depths(value, default):
    if value is None:
        return list(default)
    if "-" in value:
        a, b = (int(v) for v in value.split("-"))
        depths = list(range(a, b + 1))
    else:
        depths = [int(v) for v in value.split(",")]
    if not depths or min(depths) < 0 or len(set(depths)) != len(depths):
        raise ValueError("depths must be distinct nonnegative integers")
    return depths


def atomic_save(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=Path, required=True, help="NPZ containing a 'tokens' array")
    ap.add_argument("--kind", choices=("uniform", "product"), required=True)
    ap.add_argument("--d", type=int, help="alphabet size of a uniform-base chain")
    ap.add_argument("--d1", type=int, default=8)
    ap.add_argument("--d2", type=int, default=8)
    ap.add_argument("--depths", help="range such as 0-8, or comma-separated depths")
    ap.add_argument("--n_seq", type=int, help="use this many initial rows of the supplied token set")
    ap.add_argument("--positions", help="comma-separated prefix lengths at which to save full predictions")
    ap.add_argument("--depth_posteriors", action="store_true")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--table_dir", default="code/scripts/building_block")
    ap.add_argument("--u_lo", type=float, default=-30.0)
    ap.add_argument("--u_hi", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.chunk < 1 or args.u_lo >= args.u_hi:
        ap.error("chunk must be positive and u_lo < u_hi")
    with np.load(args.tokens, allow_pickle=False) as data:
        x = data["tokens"]
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.integer) or x.shape[1] < 2:
        ap.error("tokens must be an integer matrix with at least two tokens per sequence")
    if args.n_seq is not None:
        if not 1 <= args.n_seq <= len(x):
            ap.error("n_seq must be between 1 and the number of supplied sequences")
        x = x[:args.n_seq]
    if len(x) == 0:
        ap.error("the token set is empty")
    d = args.d if args.kind == "uniform" else args.d1 * args.d2
    if d is None or d < 2 or (args.kind == "product" and min(args.d1, args.d2) < 2):
        ap.error("supply alphabet sizes of at least two")
    if x.min() < 0 or x.max() >= d:
        ap.error("tokens outside the declared alphabet")
    depths = parse_depths(args.depths, range(9) if args.kind == "uniform" else range(1, 9))
    positions = np.array([] if args.positions is None else
                         [int(v) for v in args.positions.split(",")], dtype=int)
    if (len(set(positions)) != len(positions) or
            (len(positions) and (positions.min() < 1 or positions.max() >= x.shape[1]))):
        ap.error("positions must be distinct prefix lengths in 1..n_tokens-1")
    positions.sort()
    config = dict(
        schema=SCHEMA, kind=args.kind, d=d,
        d1=args.d1 if args.kind == "product" else None,
        d2=args.d2 if args.kind == "product" else None,
        depths=depths, prior="uniform", n_seq=len(x), n_tokens=x.shape[1],
        tokens_sha256=token_hash(x), positions=positions.tolist(),
        token_hash_encoding="little-endian int64",
        depth_posteriors=args.depth_posteriors,
        u_lo=args.u_lo, u_hi=args.u_hi, grid_step=1 / 64,
        table_dir=str(Path(args.table_dir).resolve()),
        anchor_store=os.environ.get("LSA_STORE"), anchor_src=os.environ.get("PMM_SRC"),
        code_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("shared_depth.py", "lsa_seq.py", "phi_source.py", "refs_shared_depth.py")
        },
    )
    config_json = json.dumps(config, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    part = args.out.with_name(args.out.name + ".part.npz")
    if args.out.exists():
        with np.load(args.out, allow_pickle=False) as old:
            if str(old["config_json"]) != config_json:
                raise ValueError("existing output has different provenance; choose a new output path")
            if old["loss_bits"].shape != (len(x), x.shape[1] - 1):
                raise ValueError("existing output has an incomplete loss array")
        print(f"Verified existing {args.out}", flush=True)
        return
    start = time.monotonic()
    options = dict(depths=depths, rmax=x.shape[1], table_dir=args.table_dir,
                   u_lo=args.u_lo, u_hi=args.u_hi)
    if args.kind == "uniform":
        reference = SharedDepthLSA(SeqLSA(d, **options))
    else:
        k1 = SeqLSA(args.d1, **options)
        k2 = k1 if args.d1 == args.d2 else SeqLSA(args.d2, **options)
        reference = ProductSharedDepthLSA(SharedDepthLSA(k1), SharedDepthLSA(k2))
    arrays = {}
    done = 0
    if part.exists():
        with np.load(part, allow_pickle=False) as data:
            if str(data["config_json"]) != config_json:
                raise ValueError("partial output has different provenance; choose a new output path")
            done = int(data["done"])
            arrays = {key: data[key] for key in data.files if key not in ("config_json", "done")}
        if not 0 <= done <= len(x) or arrays["loss_bits"].shape != (done, x.shape[1] - 1):
            raise ValueError("partial output has inconsistent completed rows")
    while done < len(x):
        stop = min(done + args.chunk, len(x))
        block = []
        for xi in x[done:stop]:
            result = reference.run(xi, full_probs=bool(len(positions)),
                                   positions=positions if len(positions) else None,
                                   depth_posteriors=args.depth_posteriors)
            row = {"loss_bits": result["loss"][1:]}
            if len(positions):
                row["probs"] = result["probs"]
            if args.depth_posteriors:
                for key in ("depth_posterior", "depth_posterior1", "depth_posterior2"):
                    if key in result:
                        row[key] = result[key][1:]
            block.append(row)
        for key in block[0]:
            values = np.stack([row[key] for row in block])
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite values in {key}")
            arrays[key] = values if key not in arrays else np.concatenate((arrays[key], values))
        if "probs" in arrays and np.max(np.abs(arrays["probs"][-len(block):].sum(-1) - 1)) > 1e-7:
            raise FloatingPointError("reference probabilities are not normalized")
        done = stop
        atomic_save(part, config_json=config_json, done=done, **arrays)
        print(f"{done}/{len(x)} sequences, {time.monotonic()-start:.1f}s", flush=True)
    atomic_save(args.out, config_json=config_json, tokens=x, positions=positions, **arrays)
    part.unlink()
    print(f"Wrote {args.out}; mean loss {arrays['loss_bits'].mean():.8f} bits/token", flush=True)


if __name__ == "__main__":
    main()
