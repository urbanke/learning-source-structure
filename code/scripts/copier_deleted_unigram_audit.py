"""Independent full-vector audit of the frozen copier-deleted predictions.

Read only the complete frozen natural cohort. No probability repair, model
evaluation, smoothing, or fitting. All inferential standard errors use
independent source sequences; pooled history quantiles are descriptive.
"""
from pathlib import Path
import argparse
import csv
import io
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output/copier_time/run_2026-09-15"
DEFAULT_OUTPUT = ROOT / "results/transformer/component_feasibility/unigram_audit"
N, T, D = 1024, 1024, 64
TIMES = (64, 256, 1024)
WINDOWS = ((1, 32), (33, 128), (129, 512), (513, 1024), (1, 1024))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def seed_of(*parts):
    h = 1469598103934665603
    for p in parts:
        for c in str(p).encode():
            h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h % (2**32 - 1))


def statistics(values, *, independent_sequences):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    result = dict(n=int(len(values)), mean=float(values.mean()), minimum=float(values.min()),
                  maximum=float(values.max()),
                  quantiles={str(q): float(np.quantile(values, q)) for q in (.5, .9, .95, .99)})
    if independent_sequences:
        result["sequence_se"] = float(values.std(ddof=1) / np.sqrt(len(values)))
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Complete frozen 1,024-sequence natural cohort")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true",
                        help="Recompute all statistics and verify saved JSON, TSV, and hash receipt without writing")
    args = parser.parse_args(argv)
    RAW, OUT = args.input.resolve(), args.out.resolve()
    require({p.name for p in RAW.glob("seq_*.npz")} == {f"seq_{i:06d}.npz" for i in range(N)},
            "Input must contain exactly the complete declared cohort")
    protocol = json.loads((RAW / "protocol.json").read_text())
    for name, value in dict(schema="natural-copier-sequence-v1", n_seq=N, n_ctx=T, d=D,
                            master_seed=20261004, candidate_head=[0, 1], control_head=[0, 0],
                            alpha=[0., 0.], source_depths=list(range(9)), source_relabelled=True,
                            bayes_prior="uniform shared depth 0..8, uniform base; source parameters withheld").items():
        require(protocol.get(name) == value, "Unexpected source protocol: " + name)
    spec_id = sha_bytes(canonical(protocol).encode())
    metrics = {name: np.empty((N, T), dtype=np.float64) for name in
               ("kl_deleted_to_uniform_bits", "total_variation", "maximum_token_probability_error")}
    depths = np.empty(N, dtype=np.int64)
    normalization_error = 0.0
    records = []
    for index in range(N):
        path = RAW / f"seq_{index:06d}.npz"
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata_json"]))
            require(meta["index"] == index and meta["spec_id"] == spec_id, f"Record identity mismatch: {index}")
            require(sha_bytes(canonical(meta["spec"]).encode()) == spec_id, f"Record protocol mismatch: {index}")
            require(meta["sequence_seed"] == seed_of(protocol["schema"], protocol["master_seed"], index), f"Record seed mismatch: {index}")
            q, depth = a["copier_deleted_probs"], a["source_depth"]
            for name, value in (("copier_deleted_probs", q), ("source_depth", depth)):
                require(array_hash(value) == meta["array_hashes"][name], f"Record array hash mismatch: {index}/{name}")
            require(q.shape == (T, D) and q.dtype == np.float64 and np.isfinite(q).all() and (q > 0).all(),
                    f"Invalid raw probabilities in record {index}")
            err = float(np.max(np.abs(q.sum(-1) - 1)))
            require(err <= 1e-12, f"Raw probability normalization failed: {index}")
            normalization_error = max(normalization_error, err)
            require(depth.shape == () and depth.dtype.kind in "iu", f"Invalid source depth in record {index}")
            depths[index] = int(depth)
            require(0 <= depths[index] <= 8, f"Source depth outside prior in record {index}")
            delta = np.abs(q - 1/D)
            metrics["kl_deleted_to_uniform_bits"][index] = np.sum(q * np.log2(D*q), axis=-1)
            metrics["total_variation"][index] = .5 * delta.sum(-1)
            metrics["maximum_token_probability_error"][index] = delta.max(-1)
            records.append(dict(index=index, source_depth=int(depth), sequence_seed=meta["sequence_seed"],
                                input=relative(path),
                                prediction_array_sha256=meta["array_hashes"]["copier_deleted_probs"],
                                depth_array_sha256=meta["array_hashes"]["source_depth"]))
    groups, rows = {}, []
    for group, mask in [("all", np.ones(N, dtype=bool)), ("depth_positive", depths > 0),
                        *((f"depth_{depth}", depths == depth) for depth in range(9))]:
        values = {key: value[mask] for key, value in metrics.items()}
        group_stats = dict(n_sequences=int(mask.sum()), at_prefix={}, sequence_window_means={},
                           pooled_history_descriptions={})
        for prefix in TIMES:
            group_stats["at_prefix"][str(prefix)] = {}
            for key, value in values.items():
                stats = statistics(value[:, prefix-1], independent_sequences=True)
                group_stats["at_prefix"][str(prefix)][key] = stats
                rows.append(dict(group=group, scope="prefix", first=prefix, last=prefix,
                                 quantity=key, **{k: v for k, v in stats.items() if k != "quantiles"},
                                 **{"q"+k: v for k, v in stats["quantiles"].items()}))
        for first, last in WINDOWS:
            label = f"{first}:{last}"
            group_stats["sequence_window_means"][label] = {}
            group_stats["pooled_history_descriptions"][label] = {}
            for key, value in values.items():
                window = value[:, first-1:last]
                stats = statistics(window.mean(axis=1), independent_sequences=True)
                group_stats["sequence_window_means"][label][key] = stats
                group_stats["pooled_history_descriptions"][label][key] = statistics(window, independent_sequences=False)
                rows.append(dict(group=group, scope="sequence_window_mean", first=first, last=last,
                                 quantity=key, **{k: v for k, v in stats.items() if k != "quantiles"},
                                 **{"q"+k: v for k, v in stats["quantiles"].items()}))
        groups[group] = group_stats
    worst = {}
    for key, values in metrics.items():
        index, offset = np.unravel_index(np.argmax(values), values.shape)
        worst[key] = dict(value=float(values[index, offset]), sequence_index=int(index),
                          source_depth=int(depths[index]), prefix=int(offset+1))
    result = dict(schema="copier-deleted-uniform-full-vector-audit-v1", passed=True,
                  n_sequences=N, n_prefixes=T, d=D, source_depth_counts=np.bincount(depths, minlength=9).tolist(),
                  spec_id=spec_id, protocol_path=relative(RAW/"protocol.json"),
                  protocol_sha256=sha_bytes((RAW/"protocol.json").read_bytes()),
                  script_path=relative(__file__), script_sha256=sha_bytes(Path(__file__).read_bytes()),
                  definitions=dict(kl="sum_b Q_deleted(b) log2[64 Q_deleted(b)]",
                                   total_variation="one half sum_b abs[Q_deleted(b)-1/64]",
                                   maximum_token_probability_error="max_b abs[Q_deleted(b)-1/64]"),
                  uncertainty="At each prefix, sources are independent units. Window statistics average within each sequence first. Pooled-history quantiles are descriptive only, without a standard error.",
                  validation=dict(maximum_raw_probability_sum_error=normalization_error,
                                  probability_repair="none", inspected_arrays="copier_deleted_probs and source_depth; their stored hashes and full protocol fingerprint verified"),
                  groups=groups, worst_histories=worst, inputs=records)
    summary_bytes = (json.dumps(result, indent=2, allow_nan=False)+"\n").encode()
    table_stream = io.StringIO(newline="")
    writer = csv.DictWriter(table_stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    table_bytes = table_stream.getvalue().encode()
    artifacts = {"summary.json": summary_bytes, "summary.tsv": table_bytes}
    receipt = dict(schema="copier-deleted-uniform-audit-receipt-v1", passed=True,
                   n_sequences=N, n_prefixes=T, spec_id=spec_id,
                   script=dict(path=result["script_path"], sha256=result["script_sha256"]),
                   protocol=dict(path=result["protocol_path"], sha256=result["protocol_sha256"]),
                   artifacts={name: dict(path=relative(OUT/name), sha256=sha_bytes(data))
                              for name, data in artifacts.items()})
    artifacts["receipt.json"] = (json.dumps(receipt, indent=2, allow_nan=False)+"\n").encode()
    if args.check_only:
        for name, expected in artifacts.items():
            require((OUT/name).is_file() and (OUT/name).read_bytes() == expected,
                    f"Saved audit differs from independent recomputation: {OUT/name}")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, data in artifacts.items():
            (OUT/name).write_bytes(data)
    print(json.dumps(dict(passed=True, check_only=args.check_only, n_sequences=N,
                          maximum_probability_sum_error=normalization_error,
                          receipt=relative(OUT/"receipt.json"))))



if __name__ == "__main__":
    main()
