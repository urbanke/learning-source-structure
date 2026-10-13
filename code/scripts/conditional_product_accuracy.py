"""Fixed-depth product Markov comparisons using existing source/reference/model APIs.

Each of the four depth pairs has 256 independently generated transition-matrix
pairs and length-1025 sequences. The frozen transformer and Bayesian reference
retain their training prior: independent depths uniform on 1..8, shared across
the rows within each coordinate. Neither predictor receives the test depths.
The risk at time t is KL(true next-token row || predictor), in bits.

Set LSA_STORE and PMM_SRC to the existing sealed table store, then run --stage
all. Raw caches live in output/, curated numerical summaries and figures in
results/. --stage validate independently checks selected references with a
wider integration range, exact Dirichlet formulas, relabeling and causality.
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
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
import eval_shared_depth as existing
from families import ProductFamily
from source import seed_of

SCHEMA = "conditional-product-accuracy-v1"
DEPTH_PAIRS = ((1, 1), (1, 8), (4, 4), (8, 8))
TIMES = 2 ** np.arange(11)
LN2 = np.log(2.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def setup(args):
    case = existing._case("product", args)
    spec = dict(schema=SCHEMA, depth_pairs=[list(pair) for pair in DEPTH_PAIRS],
                n_sequences_per_pair=args.n_seq, n_tokens=args.n_ctx + 1,
                times=args.prefixes.tolist(), master_seed=args.seed,
                selection="low, asymmetric, middle and high depths selected before evaluation",
                source="fresh transition matrices and independent coordinate permutations for every sequence",
                source_depths_fixed=True, reference_depths_revealed=False,
                reference_depth_prior="independent uniform 1..8, shared across each coordinate's rows",
                source_code_sha256={name: existing._sha_file(ROOT / "code/transformer" / name)
                                    for name in ("source.py", "families.py")},
                checkpoint_sha256=case["checkpoint_sha256"],
                config_sha256=existing._sha_file(case["config_path"]),
                source_numpy_version=np.__version__)
    path = Path(args.out) / "protocol.json"
    if path.exists():
        require(json.loads(path.read_text()) == spec, "Existing product protocol differs; use a new output folder")
    else:
        json_write(path, spec)
    return case, spec


def source_record(pair, index, spec):
    sequence_seed = seed_of(SCHEMA, spec["master_seed"], pair[0], pair[1], index)
    metadata = dict(protocol=spec, index=index, sequence_seed=sequence_seed, depth_pair=list(pair))
    family = ProductFamily(8, 8, spec["n_tokens"], fixed_ls=pair)
    rng = np.random.default_rng(sequence_seed)
    u, u0, t1, p1, l1 = family._draw(family.s1, 1, rng)
    v, v0, t2, p2, l2 = family._draw(family.s2, 1, rng)
    x = (u[0] * 8 + v[0]).astype(np.int16)
    times = np.asarray(spec["times"])
    first = t1[0][u0[0, times - 1]][:, np.argsort(p1[0])]
    second = t2[0][v0[0, times - 1]][:, np.argsort(p2[0])]
    true = (first[:, :, None] * second[:, None, :]).reshape(len(times), 64)
    joint = np.kron(t1[0], t2[0])
    permutation = (p1[0, :, None] * 8 + p2[0, None, :]).ravel()
    inverse = np.argsort(permutation)
    full_relabelled = joint[np.ix_(inverse, inverse)][x[times - 1]]
    require(np.max(np.abs(true - full_relabelled)) < 2e-15, "True product law/relabeling identity failed")
    replay, _ = family.batch(1, np.random.default_rng(sequence_seed))
    require(np.array_equal(replay[0], x), "ProductFamily public API replay failed")
    require(int(l1[0]) == pair[0] and int(l2[0]) == pair[1], "Fixed depths not respected")
    return dict(cache_id=existing._fingerprint(metadata), metadata_json=existing._json(metadata),
                tokens=x, tokens_sha256=existing._sha_tokens(x), true_probs=true,
                transition1=t1[0], transition2=t2[0], permutation1=p1[0], permutation2=p2[0],
                prefixes=times)


def dataset(pair, spec, args):
    base = Path(args.out) / f"L{pair[0]}_{pair[1]}" / "data"
    records = []
    for i in range(args.n_seq):
        metadata = dict(protocol=spec, index=i,
                        sequence_seed=seed_of(SCHEMA, spec["master_seed"], pair[0], pair[1], i),
                        depth_pair=list(pair))
        path = base / f"seq_{i:06d}.npz"
        expected = existing._fingerprint(metadata)
        record = existing._read_cache(path, expected)
        if record is None:
            record = source_record(pair, i, spec)
            existing._atomic_npz(path, **record)
        require(existing._sha_tokens(record["tokens"]) == str(record["tokens_sha256"]), "Token checksum mismatch")
        existing._normalization_error(record["true_probs"], "true source", 2e-12)
        records.append(record)
    return records


def reference_spec(case, args):
    base = existing._reference_spec(case, args, args.prefixes)
    base.pop("local_prior")
    return dict(base, schema=SCHEMA, implementation="ProductSharedDepthLSA; no local-depth comparator",
                wrapper_sha256=existing._sha_file(Path(__file__)))


def make_reference(args, wide=False, depths=None):
    from lsa_seq import SeqLSA
    from shared_depth import SharedDepthLSA, ProductSharedDepthLSA
    kernel = SeqLSA(8, depths=list(range(1, 9)) if depths is None else depths,
                    rmax=args.n_ctx + 1, table_dir=args.table_dir,
                    u_lo=-45 if wide else args.u_lo, u_hi=40 if wide else args.u_hi)
    return ProductSharedDepthLSA(SharedDepthLSA(kernel), SharedDepthLSA(kernel))


def references(case, data, args, reference=None):
    spec = reference_spec(case, args)
    reference = make_reference(args) if reference is None else reference
    base = Path(args.out) / case["name"] / "references"
    start = time.monotonic()
    for i, record in enumerate(data):
        ident = existing._reference_id(record, spec)
        path = base / f"seq_{i:06d}.npz"
        if existing._read_cache(path, ident) is None:
            x = record["tokens"].astype(int)
            result = reference.run(x, full_probs=True, positions=args.prefixes)
            prob = result["probs"]
            error = existing._normalization_error(prob, "shared-depth Bayes", args.prob_tol)
            selected_loss = -np.log2(prob[np.arange(len(args.prefixes)), x[args.prefixes]])
            require(np.max(np.abs(selected_loss - result["loss"][args.prefixes])) < 2e-6,
                    "Bayesian selected-time/target alignment failed")
            existing._atomic_npz(path, cache_id=ident,
                metadata_json=existing._json(dict(reference=spec, tokens_sha256=str(record["tokens_sha256"]),
                                                  runtime=existing._runtime_versions())),
                prefixes=args.prefixes, bayes_probs=prob, bayes_loss=result["loss"][1:],
                log_evidence=result["log_evidence"], max_normalization_error=error)
        if (i + 1) % 64 == 0 or i + 1 == len(data):
            print(f"{case['name']}: {i+1}/{len(data)} references; {time.monotonic()-start:.1f}s", flush=True)


def stats(array):
    array = np.asarray(array, dtype=float)
    return dict(mean=array.mean(0).tolist(), se=(array.std(0, ddof=1) / np.sqrt(len(array))).tolist())


def collect(case, pair, data, args):
    values = {key: [] for key in ("transformer", "informed_bayes", "transformer_minus_bayes", "kl_bayes_to_transformer")}
    digest = hashlib.sha256()
    maximum_normalization = np.zeros(3)
    ref_spec, net_spec = reference_spec(case, args), existing._model_spec(case, args.prefixes)
    for i, record in enumerate(data):
        paths = {part: Path(args.out) / case["name"] / part / f"seq_{i:06d}.npz"
                 for part in ("data", "references", "model")}
        ref = existing._read_cache(paths["references"], existing._reference_id(record, ref_spec))
        net = existing._read_cache(paths["model"], existing._model_id(record, net_spec))
        require(ref is not None and net is not None, f"Missing model/reference: {case['name']} {i}")
        for part, path in paths.items():
            digest.update(f"{part}/{path.name}:{existing._sha_file(path)}".encode())
        for cache in (ref, net):
            require(json.loads(str(cache["metadata_json"]))["tokens_sha256"] == str(record["tokens_sha256"]),
                    "Mismatched model/reference tokens")
            require(np.array_equal(cache["prefixes"], args.prefixes), "Mismatched times")
        true, bayes, transformer = record["true_probs"], ref["bayes_probs"], np.exp(net["logp"].astype(float))
        errs = [existing._normalization_error(p, label, 2e-6) for p, label in
                ((true, "true"), (bayes, "Bayes"), (transformer, "transformer"))]
        maximum_normalization = np.maximum(maximum_normalization, errs)
        require(np.min(bayes) > 0 and np.min(transformer) > 0, "Invalid predictor support")
        rb = np.sum(xlogy(true, true) - xlogy(true, bayes), axis=-1) / LN2
        rt = np.sum(xlogy(true, true) - xlogy(true, transformer), axis=-1) / LN2
        gap = np.sum(true * (np.log2(bayes) - np.log2(transformer)), axis=-1)
        require(np.max(np.abs(gap - (rt-rb))) < 1e-12, "Paired regret difference identity failed")
        require(min(rb.min(), rt.min()) > -1e-7, "Negative KL beyond tolerance")
        kbt = np.sum(xlogy(bayes, bayes) - xlogy(bayes, transformer), axis=-1) / LN2
        for key, value in zip(values, (rt, rb, gap, kbt)):
            require(np.isfinite(value).all(), "Nonfinite risk")
            values[key].append(value)
    existing._atomic_npz(Path(args.out) / case["name"] / "per_sequence_risks.npz",
                         prefixes=args.prefixes, **{key: np.asarray(value) for key, value in values.items()})
    return dict(depth_pair=list(pair), n_sequences=len(data), n_tokens=args.n_ctx + 1,
                times=args.prefixes.tolist(), **{key: stats(value) for key, value in values.items()},
                raw_directory=str(Path(args.out) / case["name"]), raw_records_sha256=digest.hexdigest(),
                max_normalization_error=dict(zip(("true", "bayes", "transformer"), maximum_normalization.tolist())))


def figures(summary, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.7), sharex=True, sharey=True)
    for axis, case in zip(axes.flat, summary["cases"]):
        tt = np.asarray(case["times"])
        for key, color, style in (("transformer", "#1565a8", "-"), ("informed_bayes", "#d77719", "--")):
            mean, se = (np.asarray(case[key][name]) for name in ("mean", "se"))
            axis.plot(tt, mean, color=color, linestyle=style, linewidth=1.7)
            axis.fill_between(tt, np.maximum(0, mean-se), mean+se, color=color, alpha=.14, linewidth=0)
        a, b = case["depth_pair"]
        axis.set_title(rf"$L_1={a},\ L_2={b}$")
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=.16)
    fig.supylabel("Expected next-token regret (bits)", x=.018)
    fig.supxlabel("Observed tokens, $t$", y=.035)
    fig.legend([Line2D([], [], color="#1565a8", lw=1.7), Line2D([], [], color="#d77719", lw=1.7, ls="--")],
               ["Transformer", "Bayes with the training prior"], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(.035, .05, 1, .94))
    for extension in ("pdf", "png"):
        fig.savefig(Path(args.results) / f"curves.{extension}", dpi=180, metadata={"Creator": SCHEMA})
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), sharex=True, sharey=True)
    for axis, case in zip(axes.flat, summary["cases"]):
        tt = np.asarray(case["times"])
        mean, se = (np.asarray(case["transformer_minus_bayes"][name]) for name in ("mean", "se"))
        axis.axhline(0, color=".5", linewidth=.8)
        axis.errorbar(tt, mean, yerr=se, color="#1565a8", marker="o", markersize=3, capsize=2, lw=1.3)
        a, b = case["depth_pair"]
        axis.set_title(rf"$L_1={a},\ L_2={b}$")
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
        axis.grid(True, alpha=.16)
    fig.supylabel("Transformer regret minus Bayesian regret (bits)", x=.018)
    fig.supxlabel("Observed tokens, $t$", y=.035)
    fig.tight_layout(rect=(.035, .05, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(Path(args.results) / f"gaps.{extension}", dpi=180, metadata={"Creator": SCHEMA})
    plt.close(fig)


def validate(cases, args):
    wide = make_reference(args, wide=True)
    exact = make_reference(args, depths=[1])
    maximum = dict(wider_grid_probability_error=0.0, wider_grid_risk_error_bits=0.0,
                   dirichlet_probability_error=0.0, causal_future_change_error=0.0,
                   dirichlet_final_log_evidence_error=0.0)
    from scipy.special import gammaln
    for case, pair, data in cases:
        for index in (0, len(data)-1):
            original = source_record(pair, index, json.loads(str(data[index]["metadata_json"]))["protocol"])
            require(np.array_equal(original["tokens"], data[index]["tokens"]), "Source replay failed")
            require(np.array_equal(original["true_probs"], data[index]["true_probs"]), "True-law replay failed")
            path = Path(args.out) / case["name"] / "references" / f"seq_{index:06d}.npz"
            ref = existing._read_cache(path, existing._reference_id(data[index], reference_spec(case, args)))
            require(ref is not None, "Reference required for validation")
            x = data[index]["tokens"].astype(int)
            alternate = wide.run(x, full_probs=True, positions=args.prefixes)
            difference = float(np.max(np.abs(alternate["probs"] - ref["bayes_probs"])))
            maximum["wider_grid_probability_error"] = max(maximum["wider_grid_probability_error"], difference)
            risk_error = float(np.max(np.abs(np.sum(data[index]["true_probs"] *
                         np.log2(ref["bayes_probs"] / alternate["probs"]), axis=-1))))
            maximum["wider_grid_risk_error_bits"] = max(maximum["wider_grid_risk_error_bits"], risk_error)
            # Exact add-one probabilities and closed-form evidence at depth one.
            e = exact.run(x, full_probs=True, positions=args.prefixes)
            q = np.empty_like(e["probs"])
            counts1, counts2 = np.zeros((8, 8), dtype=int), np.zeros((8, 8), dtype=int)
            want = {int(t): j for j, t in enumerate(args.prefixes)}
            for t in range(1, len(x)):
                a, b = x[t-1] // 8, x[t-1] % 8
                if t in want:
                    q[want[t]] = np.outer((counts1[a]+1)/(counts1[a].sum()+8),
                                         (counts2[b]+1)/(counts2[b].sum()+8)).ravel()
                counts1[a, x[t]//8] += 1
                counts2[b, x[t]%8] += 1
            maximum["dirichlet_probability_error"] = max(maximum["dirichlet_probability_error"],
                                                        float(np.max(np.abs(e["probs"]-q))))
            loge = -np.log(64.) + sum(float(np.sum(gammaln(8.)-gammaln(8.+c.sum(1))+
                                                              gammaln(c+1).sum(1))) for c in (counts1, counts2))
            maximum["dirichlet_final_log_evidence_error"] = max(maximum["dirichlet_final_log_evidence_error"],
                                                               abs(float(e["log_evidence"][-1])-loge))
            altered = x.copy()
            altered[16:] = (altered[16:] + 13) % 64
            prior = wide.run(altered, full_probs=True, positions=[16])["probs"][0]
            j = int(np.flatnonzero(args.prefixes == 16)[0]) if 16 in args.prefixes else None
            if j is not None:
                maximum["causal_future_change_error"] = max(maximum["causal_future_change_error"],
                                                            float(np.max(np.abs(prior-alternate["probs"][j]))))
    tolerances = dict(wider_grid_probability_error=2e-6, wider_grid_risk_error_bits=2e-6,
                      dirichlet_probability_error=2e-12, causal_future_change_error=2e-12,
                      dirichlet_final_log_evidence_error=2e-8)
    passed = all(maximum[k] <= tolerances[k] for k in maximum)
    result = dict(passed=passed, selected_sequences_per_pair=[0, args.n_seq-1], maximum_errors=maximum,
                  tolerances=tolerances, wider_integration_range=[-45, 40], main_integration_range=[args.u_lo, args.u_hi])
    json_write(Path(args.results) / "validation.json", result)
    require(passed, "Product numerical validation failed")
    print("Validation passed: " + json.dumps(maximum), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("sources", "refs", "model", "summary", "validate", "all"), default="all")
    ap.add_argument("--out", default=str(ROOT / "output/conditional_accuracy/product"))
    ap.add_argument("--results", default=str(ROOT / "results/transformer/conditional_accuracy/product"))
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    require(args.n_seq >= 2 and args.n_ctx >= 16 and args.n_ctx <= 1024, "Invalid sample/context count")
    args.prefixes = np.unique(np.append(TIMES[TIMES <= args.n_ctx], args.n_ctx))
    args.product_run = str(ROOT / "results/transformer/runs/prod_8x8")
    args.step, args.u_lo, args.u_hi = 100000, -30., 30.
    args.table_dir, args.prob_tol = str(ROOT / "code/transformer"), 2e-6
    base, protocol = setup(args)
    cases = []
    for pair in DEPTH_PAIRS:
        case = dict(base, name=f"L{pair[0]}_{pair[1]}")
        cases.append((case, pair, dataset(pair, protocol, args)))
    reference = make_reference(args) if args.stage in ("refs", "all") else None
    for case, pair, data in cases:
        if args.stage in ("refs", "all"):
            references(case, data, args, reference)
        if args.stage in ("model", "all"):
            existing._models(case, data, args, args.prefixes)
    if args.stage in ("validate", "all"):
        validate(cases, args)
    if args.stage in ("summary", "all"):
        summary = dict(schema=SCHEMA, protocol=protocol, units="bits per next-token prediction",
                       uncertainty="one standard error across independent matrix-pair/sequence draws; paired SE for differences",
                       averaging="depths fixed separately in each panel; average over transition matrices, coordinate permutations and generated sequences",
                       predictors="same frozen transformer and independent uniform depth1..8 Bayesian prior for all four panels",
                       first_token_scored=False, checkpoint=str(base["checkpoint"]),
                       cases=[collect(case, pair, data, args) for case, pair, data in cases])
        json_write(Path(args.results) / "summary.json", summary)
        lines = ["L1\tL2\tn_sequences\tt\ttransformer_mean\ttransformer_se\tbayes_mean\tbayes_se\tpaired_gap_mean\tpaired_gap_se"]
        for case in summary["cases"]:
            for j, tt in enumerate(case["times"]):
                values = [*case["depth_pair"], case["n_sequences"], tt]
                values += [case[key][stat][j] for key in ("transformer", "informed_bayes", "transformer_minus_bayes")
                           for stat in ("mean", "se")]
                lines.append("\t".join(str(value) for value in values))
        (Path(args.results) / "summary.tsv").write_text("\n".join(lines) + "\n")
        figures(summary, args)
        print("Wrote fixed-depth product curves and numerical summary", flush=True)


if __name__ == "__main__":
    main()
