"""Recompute only intact attention on the frozen 1024-sequence cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import time

import numpy as np

from attention_copy_candidate import read_copy_candidates
from eval_copier_time import checked_load
from eval_conditional_crossing import array_hash, fingerprint
from natural_circuit_probe import load_model

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"output/copier_time/run_2026-09-15")
    parser.add_argument("--out", type=Path, default=ROOT/"output/component_feasibility/attention_2026-09-15")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=1024)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 1024:
        parser.error("Require indices in the fixed 1024-sequence cohort")
    spec = json.loads((args.input/"protocol.json").read_text())
    if any(spec[key] != val for key,val in dict(n_seq=1024, n_ctx=1024, d=64, master_seed=20261004).items()):
        raise ValueError("Wrong frozen source cohort")
    model, model_meta = load_model()
    if model_meta["checkpoint"]["sha256"] != spec["model_metadata"]["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from frozen cohort")
    protocol = dict(schema="attention-copy-candidate-v1", input=str(args.input.resolve()),
                    source_spec_id=fingerprint(spec), source_protocol_sha256=sha(args.input/"protocol.json"),
                    checkpoint_sha256=model_meta["checkpoint"]["sha256"], n_sequences=1024, sequence_length=1025,
                    pooling="equal heads before conditioning on matching predecessor sites",
                    identification="candidate from second-layer attention, not an isolated neural expert",
                    producer_hashes={str(p.relative_to(ROOT)):sha(p) for p in (
                        Path(__file__), ROOT/"code/transformer/attention_copy_candidate.py",
                        ROOT/"code/transformer/circuit_probe.py", ROOT/"code/transformer/natural_circuit_probe.py",
                        ROOT/"notes/reviews/component_feasibility_protocol_2026-09-15.md")})
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out/"protocol.json"
    try:
        with protocol_path.open("x") as stream:
            json.dump(protocol, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing output uses a different protocol")
    started = time.monotonic()
    for index in range(args.start,args.stop):
        source_path = args.input/f"seq_{index:06d}.npz"
        path = args.out/source_path.name
        original, meta = checked_load(source_path, fingerprint(spec))
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                receipt = json.loads(str(saved["metadata_json"]))
                if receipt["protocol_id"] != fingerprint(protocol) or receipt["source_sha256"] != sha(source_path):
                    raise ValueError("Stale candidate output")
                for key, expected in receipt["array_hashes"].items():
                    if array_hash(saved[key]) != expected:
                        raise ValueError("Changed candidate array")
            continue
        result = read_copy_candidates(model, original["tokens"])
        error = float(np.max(np.abs(result["intact_probs"]-original["intact_probs"])))
        if error > 1e-12:
            raise ValueError("Intact replay differs from retained predictions")
        if not np.array_equal(result["counts"], original["metric_row_counts"]):
            raise ValueError("Successor counts differ from independent frozen evaluator")
        if not np.allclose(result["matching_attention"],original["intact_matching_successor_attention"][:,-1].mean(-1),rtol=0,atol=1e-12):
            raise ValueError("Attention mass differs from frozen circuit measurement")
        replay_error = 0.
        if index == 0:
            for t in (1,7,64,512):
                short = read_copy_candidates(model, original["tokens"][:t+1])
                replay_error = max(replay_error,float(np.max(np.abs(short["candidates"]["attention"][-1]-result["candidates"]["attention"][t-1]))))
            if replay_error > 1e-12:
                raise ValueError("Short-history replay failed")
        arrays = dict(attention_candidate=result["candidates"]["attention"],
                      defined=result["defined"], matching_attention=result["matching_attention"],
                      source_depth=original["source_depth"])
        receipt = dict(index=index, protocol_id=fingerprint(protocol), source_sha256=sha(source_path),
                       intact_replay_max_abs=error, truncated_replay_max_abs=replay_error,
                       array_hashes={key:array_hash(value) for key,value in arrays.items()})
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata_json=json.dumps(receipt,sort_keys=True), **arrays)
        os.link(temporary,path)
        temporary.unlink()
        if (index+1)%64 == 0 or index+1 == args.stop:
            print(f"Completed {index+1-args.start}/{args.stop-args.start}, elapsed {time.monotonic()-started:.1f}s",flush=True)


if __name__ == "__main__":
    main()
