"""Replot saved predictions as expected next-token regret against the true source.

No training, transformer forward pass, LSA inference, or new evaluation cohort.
The product source matrices and coin biases are recovered by replaying the
original source seeds, with exact full-sequence checks. All means and standard
errors use independent sequences, not the correlated positions within them.

    OPENBLAS_NUM_THREADS=1 python code/scripts/context_accuracy_audit.py

Use --check-only to recompute and compare with the saved numerical summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from families import ProductFamily
from source import seed_of
from coin_bayes_audit import entropy, posterior_prediction, reconstruct_binary

DEFAULT_OUT = ROOT / "results/transformer/context_accuracy_audit"
TIMES = 2 ** np.arange(11)
LN2 = math.log(2)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def require(passed, message):
    if not passed:
        raise ValueError(message)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


class Audit:
    def __init__(self):
        self.files = {}
        self.maxima = {}
        self.cohorts = {}

    def file(self, path):
        path = Path(path)
        value = sha_file(path)
        self.files[relative(path)] = value
        return value

    def error(self, name, value, tolerance):
        value = float(value)
        require(np.isfinite(value) and value <= tolerance, f"{name}: {value} > {tolerance}")
        self.maxima[name] = max(self.maxima.get(name, 0.0), value)

    def json(self, path):
        self.file(path)
        return json.loads(Path(path).read_text())


def kl(p, q, audit):
    require(p.shape == q.shape, "Source/predictor shape mismatch")
    require(np.isfinite(p).all() and np.isfinite(q).all(), "Nonfinite probabilities")
    require(np.min(p) >= 0 and np.min(q) > 0, "Invalid probability support")
    audit.error("source_normalization", np.max(np.abs(p.sum(-1) - 1)), 2e-12)
    audit.error("predictor_normalization", np.max(np.abs(q.sum(-1) - 1)), 2e-6)
    value = np.sum(xlogy(p, p) - xlogy(p, q), axis=-1) / LN2
    require(np.min(value) >= -1e-7, "Negative KL beyond numerical tolerance")
    return value


def evaluate(true, transformer, bayes, audit):
    tf, bb = kl(true, transformer, audit), kl(true, bayes, audit)
    gap = np.sum(true * (np.log2(bayes) - np.log2(transformer)), axis=-1)
    audit.error("paired_regret_difference_identity_bits", np.max(np.abs(tf - bb - gap)), 2e-13)
    return tf, bb, gap


def summarize(array):
    array = np.asarray(array)
    require(array.ndim == 2 and len(array) >= 2 and np.isfinite(array).all(), "Invalid cohort array")
    return {"mean": array.mean(axis=0).tolist(),
            "se": (array.std(axis=0, ddof=1) / np.sqrt(len(array))).tolist()}


def panel(name, title, n_tokens, times, tf, bb, gap, details):
    return dict(case=name, title=title, n_sequences=len(tf), n_tokens=n_tokens,
                times=np.asarray(times).tolist(), transformer=summarize(tf),
                informed_bayes=summarize(bb), transformer_minus_bayes=summarize(gap),
                details=details)


def checked_protocol(directory, audit, source_files):
    protocol = audit.json(directory / "protocol.json")
    require(protocol["n_seq"] == 1024 and protocol["n_ctx"] == 1024, "Unexpected cohort size")
    require(sorted(p.name for p in directory.glob("seq_*.npz")) ==
            [f"seq_{i:06d}.npz" for i in range(1024)], "Incomplete or enlarged cohort")
    for name in source_files:
        binding = protocol["provenance"][name]
        require(audit.file(ROOT / "code/transformer" / name) == binding["sha256"],
                f"Source code changed since saved run: {name}")
    for name in ("checkpoint", "config"):
        binding = protocol["provenance"][name]
        require(audit.file(Path(binding["path"])) == binding["sha256"], f"Changed {name}")
    return protocol


def record(path, protocol, keys, digest):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"]))
        require(meta["spec"] == protocol and meta["spec_id"] == fingerprint(protocol), "Changed record protocol")
        index = int(path.stem.rsplit("_", 1)[1])
        require(meta["index"] == index and meta["sequence_seed"] ==
                seed_of(protocol["schema"], protocol["master_seed"], index), "Changed source seed/index")
        arrays = {key: archive[key] for key in keys}
        hashes = {}
        for key, value in arrays.items():
            hashes[key] = array_hash(value)
            require(hashes[key] == meta["array_hashes"][key], f"Corrupt array {path}/{key}")
        digest.update(canonical(dict(index=index, spec_id=meta["spec_id"], arrays=hashes)).encode())
    require(np.array_equal(arrays["prefix_lengths"], np.arange(1, 1025)), "Wrong time alignment")
    require(arrays["tokens"].shape == (1025,), "Wrong sequence length")
    return arrays, meta


def coins(audit):
    prior_audit = audit.json(ROOT / "results/transformer/coins_rechecked/summary.json")
    require(prior_audit["passed"], "Existing coin audit did not pass")
    outputs = []
    for prior in ("twopoint", "uniform"):
        tag = "binary_" + prior
        ref_path = ROOT / f"results/transformer/refs_fam/{tag}.npz"
        tf_path = ROOT / f"results/transformer/eval/bin_{prior}/{tag}_step20000.npz"
        bindings = prior_audit["provenance"]["inputs"]
        for path, name in ((ref_path, "reference:" + tag),
                           (tf_path, f"evaluation:bin_{prior}:{tag}")):
            require(audit.file(path) == bindings[name]["sha256"], "Coin evaluation differs from validated run")
        with np.load(ref_path, allow_pickle=False) as ref, np.load(tf_path, allow_pickle=False) as saved:
            tokens = ref["tokens"]
            require(tokens.shape == (4096, 257) and np.array_equal(tokens, saved["tokens"]), "Coin cohort mismatch")
            require(str(saved["spec"]) == "binary:" + prior and int(saved["step"]) == 20000,
                    "Coin checkpoint/source mismatch")
            rng = np.random.default_rng(seed_of("eval", "binary:" + prior, 256, 4096))
            theta = rng.choice(np.array([0.2, 0.8]), size=4096) if prior == "twopoint" else rng.uniform(0, 1, size=4096)
            replay = (rng.random((4096, 257)) < theta[:, None]).astype(np.int16)
            require(np.array_equal(tokens, replay), "Coin full-sequence source replay failed")
            require(np.array_equal(ref["h_true"], np.repeat(entropy(theta)[:, None], 256, axis=1).astype(np.float32)),
                    "Coin recovered bias disagrees with saved true entropy")
            p_tf = reconstruct_binary(saved["logp_true"], tokens[:, 1:])
        times = TIMES[TIMES <= 256]
        counts = np.cumsum(tokens, axis=1)[:, times - 1]
        p_bb = posterior_prediction(counts, times[None, :], prior)
        true = np.broadcast_to(np.stack([1-theta, theta], axis=-1)[:, None], (4096, len(times), 2))
        tf = np.stack([1-p_tf[:, times - 1], p_tf[:, times - 1]], axis=-1)
        bb = np.stack([1-p_bb, p_bb], axis=-1)
        values = evaluate(true, tf, bb, audit)
        outputs.append(panel("coin_" + prior, "Coins: two-point prior" if prior == "twopoint" else "Coins: uniform prior",
                             257, times, *values, details={"source_bias_replay": "All 4096 complete sequences match exactly.",
                             "reference": "Exact posterior formula for the matched training prior.",
                             "checkpoint": "bin_" + prior + "/model_20000.pt"}))
        print(f"Audited {prior} coins: 4096 saved sequences", flush=True)
    return outputs


def product(audit):
    directory = ROOT / "output/shared_depth/full_prior_final_2026-09-14/product_8x8"
    old_summary = audit.json(directory / "summary.json")
    curated = audit.json(ROOT / "results/transformer/shared_depth/summary.json")
    previous = next(p for p in curated["cases"] if p["case"] == "product_8x8")
    require(previous["independent_validation"]["passed"] and previous["n_sequences"] == 128,
            "Complete product cohort lacks independent validation")
    spec = old_summary["source"]
    require(spec["family"] == "product:8x8" and spec["depth_range"] == [1, 8]
            and old_summary["n_sequences"] == 128, "Unexpected product cohort")
    for name, key in (("source.py", "source_sha256"), ("families.py", "families_sha256")):
        require(audit.file(ROOT / "code/transformer" / name) == spec[key], "Source replay code changed")
    require(audit.file(old_summary["model"]["checkpoint"]) == old_summary["model"]["checkpoint_sha256"],
            "Changed product checkpoint")
    for part in ("data", "references", "model"):
        require(sorted(p.name for p in (directory / part).glob("seq_*.npz")) ==
                [f"seq_{i:06d}.npz" for i in range(128)], "Incomplete product cohort")
    digest, rows, depths, by_predictive_kl = hashlib.sha256(), [], [], []
    family = ProductFamily(8, 8, 1025, ls=(1, 8))
    for i in range(128):
        records = {}
        for part in ("data", "references", "model"):
            path = directory / part / f"seq_{i:06d}.npz"
            digest.update((f"{part}/{path.name}:" + sha_file(path)).encode())
            with np.load(path, allow_pickle=False) as z:
                records[part] = {key: z[key] for key in z.files}
        data, ref, net = (records[part] for part in ("data", "references", "model"))
        meta, rm, nm = (json.loads(str(records[part]["metadata_json"])) for part in ("data", "references", "model"))
        require(meta["source"] == spec and meta["index"] == i and meta["sequence_seed"] ==
                seed_of(spec["schema"], "product_8x8", spec["seed"], i), "Wrong product source identity")
        token_hash = hashlib.sha256(np.asarray(data["tokens"], dtype="<i2").tobytes()).hexdigest()
        require(token_hash == str(data["tokens_sha256"]) == rm["tokens_sha256"] == nm["tokens_sha256"],
                "Product source/reference/model token mismatch")
        require(rm["reference"] == old_summary["reference"] and nm["model"] == old_summary["model"],
                "Changed product reference or model settings")
        require(np.array_equal(ref["prefixes"], TIMES) and np.array_equal(net["prefixes"], TIMES),
                "Product saved time alignment failed")
        rng = np.random.default_rng(meta["sequence_seed"])
        u, u0, t1, p1, l1 = family._draw(family.s1, 1, rng)
        v, v0, t2, p2, l2 = family._draw(family.s2, 1, rng)
        require(np.array_equal(u[0] * 8 + v[0], data["tokens"]), f"Product full-sequence replay failed: {i}")
        params = json.loads(str(data["parameters_json"]))
        require(params == dict(ls1=int(l1[0]), ls2=int(l2[0]), perm1=p1[0].tolist(), perm2=p2[0].tolist()),
                f"Product parameters/permutations replay failed: {i}")
        idx = TIMES - 1
        a = t1[0][u0[0, idx]][:, np.argsort(p1[0])]
        b = t2[0][v0[0, idx]][:, np.argsort(p2[0])]
        true = (a[:, :, None] * b[:, None, :]).reshape(len(TIMES), 64)
        # Independently index a fully relabelled 64-state transition matrix.
        full = np.kron(t1[0], t2[0])
        perm = (p1[0, :, None] * 8 + p2[0, None, :]).ravel()
        inv = np.argsort(perm)
        other = full[np.ix_(inv, inv)][data["tokens"][idx]]
        audit.error("product_relabelled_source_law_identity", np.max(np.abs(true - other)), 2e-15)
        transformer, bayes = np.exp(net["logp"].astype(float)), ref["bayes_probs"]
        rows.append(evaluate(true, transformer, bayes, audit))
        by_predictive_kl.append(np.sum(xlogy(bayes, bayes) - xlogy(bayes, transformer), axis=-1) / LN2)
        depths.append([int(l1[0]), int(l2[0])])
    audit.error("product_prior_validated_predictive_kl_mean_bits", np.max(np.abs(
        np.mean(by_predictive_kl, axis=0) - previous["by_prefix"]["kl_bayes_to_transformer"]["mean"])), 2e-12)
    incomplete = ROOT / "output/product_component_evidence/run_2026-09-15"
    present = len(list(incomplete.glob("seq_*.npz")))
    audit.cohorts["product_8x8"] = {"directory": relative(directory), "n_sequences": 128,
        "raw_record_files_sha256": digest.hexdigest(),
        "cohort_digest_rule": "concatenated part/filename:raw-file-SHA256, data/references/model per sequence in order",
        "full_tokens_depths_and_permutations_replayed": 128,
        "separate_component_cohort_not_used": {"directory": relative(incomplete), "requested_sequences": 1024,
                                               "saved_files_present": present}}
    print("Audited product: all 128 sequences in the complete validated cohort", flush=True)
    values = np.asarray(rows).transpose(1, 0, 2)
    return panel("product_8x8", "Product chains: 8 × 8", 1025, TIMES, *values,
                 details={"source": "Two independent coordinate depths, each uniform on 1..8; fresh transition matrices and coordinate permutations per sequence.",
                          "source_law": "Product of the two true current transition rows, in observed token labels.",
                          "depth_counts": {f"{a},{b}": depths.count([a, b]) for a in range(1,9) for b in range(1,9)},
                          "checkpoint_sha256": old_summary["model"]["checkpoint_sha256"]})


def uniform(audit):
    directory = ROOT / "output/copier_time/run_2026-09-15"
    protocol = checked_protocol(directory, audit, ("source.py",))
    require(protocol["source_depths"] == list(range(9)) and protocol["alpha"] == [0., 0.], "Unexpected uniform-base source")
    require(audit.json(ROOT / "results/transformer/copier_time/numerical_validation.json")["passed"],
            "Existing uniform-base reference validation failed")
    keys = ("tokens", "prefix_lengths", "source_depth", "transition_probs", "intact_probs", "bayes_probs",
            "metric_oracle_regret_intact", "metric_oracle_regret_bayes")
    digest, rows, depths = hashlib.sha256(), [], []
    for i in range(1024):
        data, meta = record(directory / f"seq_{i:06d}.npz", protocol, keys, digest)
        require(meta["network_checks"]["passed"], "Saved network validation failed")
        idx = TIMES - 1
        true = data["transition_probs"][data["tokens"][idx]]
        values = evaluate(true, data["intact_probs"][idx], data["bayes_probs"][idx], audit)
        for metric, recomputed in zip(("intact", "bayes"), values[:2]):
            audit.error("saved_uniform_oracle_regret_identity_bits",
                        np.max(np.abs(recomputed - data["metric_oracle_regret_" + metric][idx])), 3e-13)
        rows.append(values)
        depth = int(data["source_depth"])
        require(0 <= depth <= 8, "Uniform-base source depth outside its prior")
        depths.append(depth)
        if (i + 1) % 256 == 0:
            print(f"Audited uniform-base: {i+1}/1024 saved sequences", flush=True)
    audit.cohorts["uniform_base_64"] = {"directory": relative(directory), "n_sequences": 1024,
        "consumed_arrays_sha256": digest.hexdigest(), "array_hash_rule": "dtype.str, JSON shape, contiguous bytes",
        "cohort_digest_rule": "concatenated canonical JSON(index,spec_id,consumed array hashes) in sequence order"}
    values = np.asarray(rows).transpose(1, 0, 2)
    return panel("uniform_base_64", "Uniform-base control: d = 64", 1025, TIMES, *values,
                 details={"source": "One depth uniform on 0..8, shared by all rows; fresh transition matrix and whole-alphabet permutation per sequence.",
                          "source_law": "Saved true transition row of the current observed token.",
                          "depth_counts": [depths.count(depth) for depth in range(9)],
                          "checkpoint_sha256": protocol["provenance"]["checkpoint"]["sha256"]})


def shared_base(audit):
    path = ROOT / "results/transformer/shared_base_full_context/pilot_2026-09-15.json"
    existing = audit.json(path)
    require(existing["prefix_length"] == 1024 and existing["n_sources"] == 4, "Shared-base pilot changed")
    status = audit.json(ROOT / "results/transformer/shared_base_full_context/pilot_status_2026-09-15.json")
    return {"n_sources": 4, "n_tokens": 1025, "available_times": [1024], "n_training_checkpoints": 2,
            "resolved_source_indices": [p["source"] for p in status["pairs"] if p["passed"]],
            "unresolved_source_indices": [p["source"] for p in status["pairs"] if not p["passed"]],
            "aggregate_reported": False, "curve_available": False,
            "reason": "Only the final prediction was evaluated for four independent sources; one Bayesian reference remains unresolved. No source-family learning curve or resolved-subset mean is reported."}


def draw(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))
    for ax, item in zip(axes.ravel(), summary["panels"]):
        times = item["times"]
        for name, color, marker, linestyle, label in (
                ("transformer", "#276B9A", "o", "-", "Transformer"),
                ("informed_bayes", "#D17A29", "s", "--", "Informed Bayes")):
            value = item[name]
            ax.errorbar(times, value["mean"], yerr=value["se"], color=color, marker=marker,
                        linestyle=linestyle, linewidth=1.25, markersize=4.3,
                        markerfacecolor="white", capsize=2, elinewidth=.85, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256] + ([1024] if item["n_tokens"] == 1025 else []))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Number of observed tokens")
        ax.set_ylabel("Expected next-token regret [bits]")
        ax.set_title(item["title"] + f"\n{item['n_sequences']:,} sequences, length {item['n_tokens']:,}", fontsize=9.5)
        ax.grid(axis="y", color="0.90", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=.08, right=.985, bottom=.115, top=.935, wspace=.26, hspace=.52)
    fig.text(.5, .025, "Means ± one standard error across independent sequences. Lines join selected times.",
             ha="center", fontsize=8)
    fig.savefig(out / "next_token_regret.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "next_token_regret.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    audit = Audit()
    panels = coins(audit) + [product(audit), uniform(audit)]
    summary = {"schema": "context-accuracy-audit-v1", "units": "bits per next-token prediction",
               "quantity": "Mean KL(true current next-token distribution || predictor distribution), averaged across independent sequences and their freshly drawn source parameters.",
               "time": "At time t, observe tokens 1 through t and predict token t+1. The first token is supplied free.",
               "uncertainty": "One sample standard error across independent sequence/source draws; positions within a sequence are not treated as independent; training-seed uncertainty is not included.",
               "selected_times": "Powers of two fixed before computing the curves; no smoothing, interpolation-based threshold, or source-dependent selection.",
               "probability_repair": "None; raw saved probabilities are checked, not clipped or renormalized.",
               "panels": panels, "shared_base_status": shared_base(audit)}
    if args.check_only:
        require(json.loads((args.out / "summary.json").read_text()) == summary, "Recomputed summary differs")
        receipt = json.loads((args.out / "receipt.json").read_text())
        require(receipt["passed"] and receipt["script_sha256"] == sha_file(Path(__file__)),
                "Receipt was produced by a different script")
        require(receipt["inputs"] == audit.files and receipt["source_cohorts"] == audit.cohorts,
                "Fresh input/cohort fingerprints differ from the receipt")
        require(receipt["maximum_check_errors"] == audit.maxima, "Recomputed checks differ from receipt")
        for filename, expected in receipt["outputs"].items():
            require(sha_file(args.out / filename) == expected, f"Changed saved output: {filename}")
        print("All saved-data checks and exact summary comparison passed.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    header = ["source", "n_sequences", "n_tokens", "t", "transformer_regret_bits", "transformer_se_bits",
              "bayes_regret_bits", "bayes_se_bits", "paired_gap_bits", "paired_gap_se_bits"]
    lines = ["\t".join(header)]
    for item in panels:
        for j, t in enumerate(item["times"]):
            row = [item["case"], item["n_sequences"], item["n_tokens"], t]
            for name in ("transformer", "informed_bayes", "transformer_minus_bayes"):
                row += [item[name]["mean"][j], item[name]["se"][j]]
            lines.append("\t".join(str(v) for v in row))
    (args.out / "summary.tsv").write_text("\n".join(lines) + "\n")
    draw(summary, args.out)
    receipt = dict(schema="context-accuracy-audit-receipt-v1", passed=True,
                   script_sha256=sha_file(Path(__file__)), inputs=audit.files,
                   source_cohorts=audit.cohorts, maximum_check_errors=audit.maxima,
                   checkpoint_inference=False, training=False, new_source_draws=False,
                   source_seed_replay="Original saved coin and product draws are reconstructed and all tokens checked exactly.",
                   reference_computation="Closed-form coin posterior only; saved LSA Bayesian predictions reused.",
                   reference_validation_scope="Existing numerical grid checks are inherited; this consumer checks hashes, normalization and metric identities, not a new posterior convergence study.",
                   outputs={p.name: sha_file(p) for p in args.out.iterdir() if p.name in
                            ("summary.json", "summary.tsv", "next_token_regret.pdf", "next_token_regret.png")})
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"Wrote four source panels and separate shared-base status to {args.out}")


if __name__ == "__main__":
    main()
