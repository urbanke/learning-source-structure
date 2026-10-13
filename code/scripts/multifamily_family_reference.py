"""Evaluate one tractable family prior on all three saved full-prior cohorts.

The frozen cohort identity and an already-passed numerical validation receipt
are required. No source parameters or oracle probabilities enter the reference.
Per-sequence natural-log marginal likelihoods are retained for mixture Bayes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from mixture_family_evidence import FamilyEvidence, sha256


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--family", choices=("product", "uniform_base"), required=True)
    parser.add_argument("--chunk", type=int, default=32)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("chunk must be positive")
    validation = json.loads(args.validation.read_text())
    if not validation.get("passed"):
        raise ValueError("numerical validation has not passed")
    for name, expected in validation["code_sha256"].items():
        if sha256(ROOT / "code/transformer" / name) != expected:
            raise ValueError(f"validated code changed: {name}")
    manifest = json.loads((args.heldout / "manifest.json").read_text())
    ref = FamilyEvidence(args.family)
    store = Path(os.environ["LSA_STORE"])
    store_files = ["manifest.json", "anchors.json"] + [f"level_{depth:02d}.{suffix}" for depth in range(2, 9) for suffix in ("index.json", "bin")]
    store_hashes = {name: sha256(store / name) for name in store_files}
    common = dict(schema="multifamily-family-reference-v1", reference_family=args.family,
                  evidence_kind="normalized_sequence_marginal", first_token_log_marginal=float(-np.log(64)),
                  loss_convention="time t scores tokens[1:t+1]; first token omitted",
                  source_plan_id=manifest["plan_id"], source_manifest_sha256=sha256(args.heldout / "manifest.json"),
                  bounds=[-30., 30.], grid_step=1 / 64,
                  validation_sha256=sha256(args.validation), validated_code_sha256=validation["code_sha256"],
                  runner_sha256=sha256(__file__), store=str(store.resolve()), store_file_sha256=store_hashes,
                  backend_sha256=sha256(Path(os.environ["PMM_SRC"]) / "product_model_with_memory/universal_tables.py"),
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  backend_environment={name: os.environ.get(name) for name in
                                       ("PMM_UNIVERSAL_TABLES", "PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE", "PMM_PHI_SADDLE_MIN_L")})
    args.out.mkdir(parents=True, exist_ok=True)
    for family in ("product", "shared_base", "uniform_base"):
        cohort = f"{family}_prior"
        source_path = args.heldout / f"{cohort}.npz"
        source = json.loads((args.heldout / f"{cohort}.json").read_text())
        source_hash = sha256(source_path)
        if (source_hash != source["data_sha256"] or source_hash != validation["source_sha256"][family]
                or source["plan_id"] != manifest["plan_id"]):
            raise ValueError("frozen cohort or source identity changed")
        with np.load(source_path, allow_pickle=False) as saved:
            tokens, prefixes, seeds = saved["tokens"], saved["prefixes"], saved["sequence_seed"]
        if tokens.shape != (1024, 1025) or not np.array_equal(prefixes, [1, 4, 16, 64, 256, 1024]):
            raise ValueError("unexpected production cohort or prediction times")
        config = dict(common, cohort=cohort, n_seq=len(tokens), n_tokens=tokens.shape[1],
                      prefixes=prefixes.tolist(), source_data_sha256=source_hash,
                      source_record_sha256=sha256(args.heldout / f"{cohort}.json"))
        config_json = json.dumps(config, sort_keys=True, allow_nan=False)
        target = args.out / f"{args.family}_on_{cohort}.npz"
        record = target.with_suffix(".json")
        partial = target.with_name(target.name + ".part.npz")
        marginals = np.empty((len(tokens), len(prefixes)))
        losses = np.empty_like(marginals)
        done = 0
        existing = target if target.exists() else partial
        if existing.exists():
            with np.load(existing, allow_pickle=False) as saved:
                if str(saved["config_json"]) != config_json:
                    raise ValueError("saved reference provenance changed")
                done = int(saved["done"])
                if (not 0 <= done <= len(tokens)
                        or saved["log_marginals"].shape != (done, len(prefixes))
                        or not np.array_equal(saved["sequence_seed"], seeds[:done])):
                    raise ValueError("inconsistent saved reference")
                marginals[:done], losses[:done] = saved["log_marginals"], saved["cumulative_loss_bits"]
            if existing == target:
                if done != len(tokens) or sha256(target) != json.loads(record.read_text())["data_sha256"]:
                    raise ValueError("final reference record is inconsistent")
                print(f"Verified existing {target}", flush=True)
                continue
        started = time.monotonic()
        while done < len(tokens):
            stop = min(done + args.chunk, len(tokens))
            for i in range(done, stop):
                marginals[i], losses[i] = ref.cumulative_losses(tokens[i], prefixes)
            done = stop
            write_npz(partial, config_json=config_json, done=done, prefixes=prefixes,
                      sequence_seed=seeds[:done], log_marginals=marginals[:done],
                      cumulative_loss_bits=losses[:done])
            print(json.dumps(dict(cohort=cohort, reference=args.family, done=done, total=len(tokens),
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
        partial.replace(target)
        write_json(record, dict(config, complete=True, data_sha256=sha256(target),
                                runtime_seconds=time.monotonic() - started,
                                cache=ref.kernel.cache_info()))


if __name__ == "__main__":
    main()
