"""Independently audit complete natural-sequence records and summarize them.

No model or Bayesian inference is rerun. Source draws are replayed from
their seeds; completed transitions, probability masses, and expected losses
are reconstructed independently of the frozen metric producer. Every source
and prefix remains in the analysis. --smoke permits a separately declared
small complete cohort, never a partial production cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("intact", "copier_deleted", "control_deleted", "bayes")
CURVE_KEYS = tuple("excess_" + name for name in CONDITIONS) + (
    "copier_causal_mass", "control_causal_mass", "benefit_copier_bits", "benefit_control_bits", "benefit_bayes_bits",
    "paired_excess_intact_minus_bayes")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def draw_seed(master, index):
    value = 1469598103934665603
    for part in ("natural-copier-sequence-v1", master, index):
        for byte in str(part).encode():
            value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(value % (2 ** 32 - 1))


def require_close(actual, expected, name, tolerance=2e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Wrong shape or nonfinite values: {name}")
    if expected.dtype.kind in "biu":
        if actual.dtype.kind != expected.dtype.kind or not np.array_equal(actual, expected):
            raise ValueError(f"Discrete value/alignment mismatch: {name}")
        return 0.0
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if error > tolerance:
        raise ValueError(f"Independent recomputation mismatch: {name}, {error}")
    return error


def probabilities(value, shape, name, tolerance):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError(f"Expected finite positive raw probabilities: {name}")
    error = float(np.max(np.abs(value.sum(-1) - 1)))
    if error > tolerance:
        raise ValueError(f"Raw normalization error: {name}, {error}")
    return error


def independent_metrics(tokens, predictions, transition):
    """Use explicit completed-edge updates and log2 expectations, no repairs."""
    n, d = len(tokens) - 1, transition.shape[0]
    history = np.zeros((d, d), dtype=np.int64)
    rows = np.empty((n, d), dtype=np.int64)
    for query in range(n):
        if query:
            history[tokens[query - 1], tokens[query]] += 1
        rows[query] = history[tokens[query]]
    mask = rows.astype(bool)
    support = np.count_nonzero(mask, axis=-1)
    oracle = transition[tokens[:-1]]
    entropy = -(oracle * np.log2(oracle)).sum(-1)
    out = dict(prefix_lengths=np.arange(1, n+1, dtype=np.int64), row_counts=rows,
               row_totals=rows.sum(-1), successor_mask=mask, support_sizes=support,
               uniform_mass=support / d, source_entropy_bits=entropy,
               source_observed_loss_bits=-np.log2(transition[tokens[:-1], tokens[1:]]))
    uniform = np.full((n, d), 1 / d)
    for name, value in (*predictions.items(), ("uniform", uniform)):
        expected_loss = -(oracle * np.log2(value)).sum(-1)
        out["oracle_ce_" + name] = expected_loss
        out["oracle_regret_" + name] = expected_loss - entropy
        out["observed_loss_" + name] = -np.log2(value[np.arange(n), tokens[1:]])
        if name != "uniform":
            out["mass_" + name] = np.where(mask, value, 0.0).sum(-1)
            out["excess_" + name] = out["mass_" + name] - out["uniform_mass"]
    def divergence(first, second):
        return (first * np.log2(first / second)).sum(-1)
    for label in ("copier", "control"):
        name = label + "_deleted"
        out[label + "_causal_mass"] = out["mass_intact"] - out["mass_" + name]
        out["benefit_" + label + "_bits"] = out["oracle_ce_" + name] - out["oracle_ce_intact"]
        out["cumulative_benefit_" + label + "_bits"] = np.cumsum(out["benefit_" + label + "_bits"])
        out[name + "_to_uniform_kl_bits"] = divergence(predictions[name], uniform)
        out["uniform_to_" + name + "_kl_bits"] = divergence(uniform, predictions[name])
    out["benefit_bayes_bits"] = out["oracle_ce_uniform"] - out["oracle_ce_bayes"]
    out["cumulative_benefit_bayes_bits"] = np.cumsum(out["benefit_bayes_bits"])
    out["intact_bayes_kl_bits"] = divergence(predictions["intact"], predictions["bayes"])
    return out


def verify_spec(spec, smoke):
    expected = dict(schema="natural-copier-sequence-v1", master_seed=20261004, d=64,
                    source_family="uniform-base LSA Markov chains", alpha=[0., 0.], source_depths=list(range(9)),
                    source_relabelled=True, conditions=list(CONDITIONS), candidate_head=[0, 1], control_head=[0, 0],
                    first_token="supplied free", time="t=1..n_ctx: see tokens[:t], predict tokens[t]",
                    bayes_prior="uniform shared depth 0..8, uniform base; source parameters withheld")
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Wrong natural-sequence protocol: {key}")
    if (not isinstance(spec["n_seq"], int) or isinstance(spec["n_seq"], bool) or spec["n_seq"] < 1
            or not isinstance(spec["n_ctx"], int) or not 1 <= spec["n_ctx"] <= 1024):
        raise ValueError("Invalid declared cohort size or horizon")
    if not smoke and (spec["n_seq"], spec["n_ctx"]) != (1024, 1024):
        raise ValueError("Publication analysis requires the complete1,024-source,1,024-prefix cohort")
    for name, value in spec["provenance"].items():
        if sha(value["path"]) != value["sha256"]:
            raise ValueError(f"Current source/checkpoint/producer provenance changed: {name}")
    metadata = spec["model_metadata"]
    if metadata["step"] != 100000 or not metadata["parameter_values_preserved"] or metadata["torch_threads"] != 1:
        raise ValueError("Wrong fixed transformer checkpoint or numerical mode")
    if spec["quadrature"] != dict(u_lo=-30, u_hi=30, step=1/64, depths=list(range(9)), rmax=spec["n_ctx"]+1):
        raise ValueError("Bayesian depth prior or quadrature changed")
    if spec["tolerances"] != dict(bayes_probability=1e-7, bayes_loss_bits=2e-6,
                                  transformer_probability=1e-12, causal_replay=1e-10):
        raise ValueError("Numerical tolerances differ from the declared protocol")


def source_replay(arrays, n_ctx, seed):
    """Reproduce source draws and relabel by direct assignment, not inversion."""
    sys.path.insert(0, str(ROOT / "code/transformer"))
    from source import Source
    source = Source(64, n_ctx + 1, alpha=(0., 0.), ls=(0, 8), permute=True)
    rng = np.random.default_rng(seed)
    alpha, depth = source.draw_params(1, rng)
    initial, transition = source.matrices(alpha, depth, rng)
    original = source.sequences(initial, transition, rng)[0]
    permutation = rng.permutation(64)
    relabeled_initial, relabeled_transition = np.empty(64), np.empty((64, 64))
    relabeled_initial[permutation] = initial[0]
    relabeled_transition[np.ix_(permutation, permutation)] = transition[0]
    expected = dict(tokens=permutation[original], permutation=permutation, source_depth=np.asarray(depth[0]),
                    initial_probs=relabeled_initial, transition_probs=relabeled_transition)
    return max(require_close(arrays[name], value, "source/"+name, tolerance=0) for name, value in expected.items())


def network_receipt(receipt, n):
    if (receipt.get("passed") is not True or receipt["replay_tolerance"] != 1e-10
            or not isinstance(receipt["original_checked_this_call"], bool)
            or set(receipt["conditions"]) != set(CONDITIONS[:-1])):
        raise ValueError("Missing network validation")
    wanted = {str(min(n, value)) for value in (1, 7, 64, 512)}
    for values in receipt["conditions"].values():
        errors = [values["probability_sum_error"], values["attention_sum_error"]]
        if not np.isfinite(errors).all() or min(errors) < 0 or max(errors) > 1e-12:
            raise ValueError("Network probability or attention check failed")
        if set(values["causal_prefix_replay"]) != wanted:
            raise ValueError("Incomplete per-sequence causal replay")
        for check in values["causal_prefix_replay"].values():
            if not np.isfinite(list(check.values())).all() or min(check.values()) < 0 or max(check.values()) > 1e-10:
                raise ValueError("Failed causal replay")
    original = receipt["original_validation"]
    if original.get("passed") is not True or original["scored_prefixes"] != n or original["tolerance"] != 1e-10:
        raise ValueError("Missing original-model fidelity check")
    if set(original["comparisons"]) != {"manual", "fused"}:
        raise ValueError("Incomplete original-model comparisons")
    for values in original["comparisons"].values():
        if not np.isfinite(list(values.values())).all() or min(values.values()) < 0 or max(values.values()) > 1e-10:
            raise ValueError("Failed original-model fidelity check")
    return original["input_sha256"]


def inspect_record(path, spec, spec_id, index):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    meta = json.loads(str(arrays.pop("metadata_json")))
    seed = draw_seed(spec["master_seed"], index)
    if meta["spec"] != spec or meta["spec_id"] != spec_id or meta["index"] != index or meta["sequence_seed"] != seed:
        raise ValueError(f"Wrong protocol/source index/seed: {path}")
    if set(arrays) != set(meta["array_hashes"]) or any(array_hash(value) != meta["array_hashes"][key] for key, value in arrays.items()):
        raise ValueError(f"Raw-array checksum mismatch: {path}")
    n = spec["n_ctx"]
    tokens = arrays["tokens"]
    if tokens.shape != (n+1,) or tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens >= 64):
        raise ValueError("Wrong complete path")
    source_replay(arrays, n, seed)
    errors = dict(source_sum_error=probabilities(arrays["transition_probs"], (64,64), "source", 1e-12))
    predictions = {name: arrays[name+"_probs"] for name in CONDITIONS}
    for name, value in predictions.items():
        errors[name+"_sum_error"] = probabilities(value, (n,64), name, 1e-7 if name == "bayes" else 1e-12)
    computed = independent_metrics(tokens, predictions, arrays["transition_probs"])
    if {key[7:] for key in arrays if key.startswith("metric_")} != set(computed):
        raise ValueError("Missing or unexpected metric array")
    errors["metric_recomputation_error"] = max(require_close(arrays["metric_"+key], value, key,
        tolerance=2e-8 if key.startswith("cumulative_") else 2e-10) for key, value in computed.items())
    require_close(arrays["prefix_lengths"], computed["prefix_lengths"], "prefix positions")
    for key, expected in (("matching_successor_count", computed["row_totals"]),
                          ("matching_successor_defined", computed["row_totals"] > 0),
                          ("previous_token_defined", np.arange(n) > 0),
                          ("matching_successor_uniform_attention", computed["row_totals"] / np.arange(1,n+1))):
        require_close(arrays[key], expected, key)
    attention = {}
    for name in CONDITIONS[:-1]:
        for kind in ("previous_token_attention", "matching_successor_attention"):
            key = name+"_"+kind
            value = arrays[key]
            if value.shape != (n,2,4) or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1+1e-12):
                raise ValueError("Invalid compact attention observations")
            if kind == "previous_token_attention" and np.any(value[0] != 0):
                raise ValueError("Previous-token attention is undefined at the first token")
            if kind == "matching_successor_attention" and np.any(value[computed["row_totals"] == 0] != 0):
                raise ValueError("Attention assigned to nonexistent successor sites")
            attention[key] = value
    posterior = arrays["bayes_depth_posterior"]
    if posterior.shape != (n,9) or not np.isfinite(posterior).all() or np.any(posterior < 0):
        raise ValueError("Invalid global-depth posterior")
    require_close(posterior.sum(-1), np.ones(n), "posterior sum", 1e-7)
    require_close(posterior[0], np.full(9,1/9), "initial Bayes prior", 1e-7)
    require_close(predictions["bayes"][0], np.full(64,1/64), "initial Bayes prediction", 1e-7)
    evidence = arrays["bayes_log_evidence"]
    if evidence.shape != (n+1,):
        raise ValueError("Wrong Bayes evidence horizon")
    errors["bayes_evidence_increment_error"] = require_close(-np.diff(evidence)/np.log(2),
        computed["observed_loss_bayes"], "Bayes observed loss", 2e-6)
    errors["bayes_evidence_telescope_error"] = require_close(-(evidence[1:]-evidence[0])/np.log(2),
        np.cumsum(computed["observed_loss_bayes"]), "Bayes cumulative observed loss", 2e-6)
    require_close(np.array([evidence[0]]), np.array([-np.log(64)]), "first token evidence", 1e-12)
    original_input = network_receipt(meta["network_checks"], n)
    prefix_hash = hashlib.sha256(tokens[:-1].astype(np.int64).tobytes()).hexdigest()
    source_fields = {"tokens","initial_probs","transition_probs","permutation","source_depth"}
    expected_arrays = source_fields | {name+"_probs" for name in CONDITIONS} | {"metric_"+key for key in computed}
    expected_arrays |= set(attention) | {"prefix_lengths","bayes_depth_posterior","bayes_log_evidence",
        "previous_token_defined","matching_successor_count","matching_successor_defined","matching_successor_uniform_attention"}
    if set(arrays) != expected_arrays:
        raise ValueError("Unexpected raw array schema")
    return dict(metrics={key:value for key,value in computed.items() if value.ndim == 1}, attention=attention,
                depth=int(arrays["source_depth"]), original_input=original_input, prefix_hash=prefix_hash,
                original_checked=meta["network_checks"]["original_checked_this_call"], errors=errors,
                input=dict(index=index, path=relative(path), sha256=sha(path), metadata_sha256=fingerprint(meta),
                           tokens_sha256=array_hash(tokens), sequence_seed=seed, source_depth=int(arrays["source_depth"])))


def statistics(values):
    """First axis is independent sequences, including for window averages."""
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite sequence-level values")
    n, shape = values.shape[0], values.shape[1:]
    if not n:
        means = np.full(shape, np.nan)
        errors = np.full(shape, np.nan)
    else:
        means = values.mean(axis=0)
        errors = values.std(axis=0, ddof=1)/np.sqrt(n) if n > 1 else np.full(shape, np.nan)
    def serial(array):
        value = np.asarray(array)
        if value.ndim == 0:
            return float(value) if np.isfinite(value) else None
        return [serial(item) for item in value]
    return dict(mean=serial(means), se=serial(errors), n=n,
                ci95_multiplier=float(student_t.ppf(.975,n-1)) if n > 1 else None)


def numerical_validation(directory, spec):
    """Bind the separate grid check to the exact production paths and code."""
    path = ROOT / "results/transformer/copier_time/numerical_validation.json"
    receipt = json.loads(path.read_text())
    expected = dict(schema="natural-copier-bayes-grid-validation-v1", passed=True,
                    master_seed=spec["master_seed"], declared_cohort_size=spec["n_seq"],
                    d=64, n_ctx=spec["n_ctx"], alpha=[0.,0.], source_depth_prior=list(range(9)),
                    bayes_depth_prior=list(range(9)), shared_global_depth=True, source_parameters_withheld=True)
    if any(receipt.get(key) != value for key,value in expected.items()):
        raise ValueError("Numerical validation belongs to a different protocol")
    for value in receipt["provenance"].values():
        if sha(value["path"]) != value["sha256"]:
            raise ValueError("Numerical-validation provenance changed")
    checked = []
    for source in receipt["sources"]:
        if source["checks"]["passed"] is not True or sha(source["raw_path"]) != source["raw_sha256"]:
            raise ValueError("Numerical validation failed or its raw record changed")
        index = source["index"]
        if source["sequence_seed"] != draw_seed(spec["master_seed"], index):
            raise ValueError("Numerical-validation sequence identity changed")
        with np.load(directory/f"seq_{index:06d}.npz",allow_pickle=False) as archive:
            for key in ("tokens","initial_probs","transition_probs","permutation","source_depth"):
                if array_hash(archive[key]) != source["array_hashes"][key]:
                    raise ValueError("Numerical validation uses a different source path")
            if array_hash(archive["bayes_probs"]) != source["array_hashes"]["production_probs"]:
                raise ValueError("Numerical validation uses different Bayesian predictions")
        checked.append(index)
    if checked != [0,7]:
        raise ValueError("Incomplete declared numerical-validation paths")
    return dict(path=relative(path),sha256=sha(path),passed=True,matched_production_indices=checked,
                limitations=receipt["limitations"])


def aggregate(directory, *, smoke=False):
    directory = Path(directory)
    spec = json.loads((directory/"protocol.json").read_text())
    verify_spec(spec, smoke)
    expected = {f"seq_{index:06d}.npz" for index in range(spec["n_seq"])}
    observed = {path.name for path in directory.glob("seq_*.npz")}
    if observed != expected:
        raise ValueError(f"Require every declared source without extras: missing{len(expected-observed)}, unexpected{len(observed-expected)}; partial aggregation is forbidden")
    spec_id = fingerprint(spec)
    records = [inspect_record(directory/f"seq_{index:06d}.npz",spec,spec_id,index) for index in range(spec["n_seq"])]
    input_hashes = {record["prefix_hash"] for record in records}
    if not any(record["original_checked"] for record in records) or any(record["original_input"] not in input_hashes for record in records):
        raise ValueError("Original-model validation does not refer to the completed cohort")
    metrics = {key:np.stack([record["metrics"][key] for record in records]) for key in records[0]["metrics"]}
    metrics["paired_excess_intact_minus_bayes"] = metrics["excess_intact"]-metrics["excess_bayes"]
    attention = {key:np.stack([record["attention"][key] for record in records]) for key in records[0]["attention"]}
    depths = np.array([record["depth"] for record in records])
    masks = {"all":np.ones(len(depths),dtype=bool), "depth_0":depths==0, "depth_positive":depths>0}
    masks.update({f"depth_{depth}":depths==depth for depth in range(1,9)})
    groups = {}
    for group, mask in masks.items():
        selected = {key:value[mask] for key,value in metrics.items()}
        windows = []
        for first,last in ((1,32),(33,128),(129,1024)):
            last = min(last,spec["n_ctx"])
            if first <= last:
                windows.append(dict(first_prefix=first,last_prefix=last,
                    statistics={key:statistics(value[:,first-1:last].mean(axis=1)) for key,value in selected.items()}))
        groups[group] = dict(n_sequences=int(mask.sum()),
            curves={key:statistics(selected[key]) for key in CURVE_KEYS},
            time_averages={key:statistics(value.mean(axis=1)) for key,value in selected.items()},
            final_prefix={key:statistics(value[:,-1]) for key,value in selected.items()}, windows=windows,
            support=dict(nonempty_fraction=((selected["support_sizes"]>0).mean(axis=0).tolist() if mask.any() else [None]*spec["n_ctx"]),
                         full_fraction=((selected["support_sizes"]==64).mean(axis=0).tolist() if mask.any() else [None]*spec["n_ctx"]),
                         mean_size=(selected["support_sizes"].mean(axis=0).tolist() if mask.any() else [None]*spec["n_ctx"])),
            attention_time_averages={key:statistics(value[mask,1:].mean(axis=1) if "previous_token" in key and spec["n_ctx"]>1
                                                   else value[mask].mean(axis=1)) for key,value in attention.items()})
    return dict(schema="natural-copier-analysis-v1", passed=True, smoke=smoke, publication_cohort=not smoke,
                spec=spec,spec_id=spec_id,n_sequences=spec["n_seq"],prefix_lengths=list(range(1,spec["n_ctx"]+1)),
                groups=groups,source_depth_counts={str(depth):int((depths==depth).sum()) for depth in range(9)},
                maximum_check_errors={key:max(record["errors"][key] for record in records) for key in records[0]["errors"]},
                original_model_checked_sequences=sum(record["original_checked"] for record in records),
                numerical_validation=(None if smoke else numerical_validation(directory,spec)),
                inputs=[record["input"] for record in records],
                protocol=dict(path=relative(directory/"protocol.json"),sha256=sha(directory/"protocol.json")),
                uncertainty="Pointwise 95% Student-t intervals across independently drawn whole source sequences. Paired head effects and intact-minus-Bayes excess use within-sequence differences; successive prefixes are not independent observations.",
                averaging="Every sequence has equal weight at each prefix, including empty and full successor sets. Positive-depth means average the actual positive-depth sequences, not the eight subgroup means.",
                response="Probability assigned to tokens previously seen after the current token, minus the known uniform probability of that same set. No clipping, normalization, chosen threshold or smoothing.",
                comparisons="Head-deletion loss increases compare each deleted network with the intact network. Bayesian gain compares Bayes with the known uniform distribution; these are different comparisons.",
                limitations="Positive successor excess alone is not evidence of benefit. No successor available implies excess zero but does not establish head inactivity. Control-head effects, depth0 controls, and predictive benefits are retained.",
                retention="All raw arrays and metric checks are audited. Main response/causal-effect/benefit curves retain every prefix in all depth groups; every other scalar metric is retained as a whole-sequence mean, final-prefix value and predeclared-window mean. Full observations remain in immutable input records.",
                script=dict(path=relative(__file__),sha256=sha(__file__)))


def line(ax, positions, value, label, color, style="-"):
    means=np.asarray(value["mean"],dtype=float)
    errors=np.asarray(value["se"],dtype=float)
    ax.plot(positions,means,color=color,ls=style,lw=1.65,label=label)
    if value["ci95_multiplier"] is not None:
        radius=errors*value["ci95_multiplier"]
        ax.fill_between(positions,means-radius,means+radius,color=color,alpha=.14,linewidth=0)


def configure_axis(ax, n, ylabel):
    from matplotlib.ticker import PercentFormatter, ScalarFormatter
    ax.set_xscale("log",base=2)
    ax.set_xlim(1,max(n,2))
    ticks=[value for value in (1,2,4,8,16,32,64,128,256,512,1024) if value<=n]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Tokens observed in the sequence")
    ax.set_ylabel(ylabel)
    if "probability" in ylabel.lower():
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0,color="#8D9CA5",lw=.8)
    ax.grid(axis="y",color="#E3E9EC",lw=.6)


def plot(report,out):
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"output/copier_time/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    positions=np.asarray(report["prefix_lengths"]); n=len(positions)
    colors=dict(intact="#176C94",bayes="#D4662B",copier_deleted="#479278",control_deleted="#7D7C88")
    labels=dict(intact="Intact transformer",bayes="Informed Bayes",copier_deleted="Copying head removed",control_deleted="Comparison head removed")
    fig,ax=plt.subplots(figsize=(8.6,5.8))
    for name in ("intact","bayes","copier_deleted"):
        line(ax,positions,report["groups"]["all"]["curves"]["excess_"+name],labels[name],colors[name])
    configure_axis(ax,n,"Extra probability assigned to earlier successors")
    ax.legend(frameon=False,loc="best")
    title="Copying along a new sequence" if not report["smoke"] else "Smoke check: copying along a sequence"
    ax.set_title(title,pad=15,fontsize=14)
    fig.text(.52,.083,f"{report['n_sequences']:,} independently drawn sequences, each of length {n+1:,}; predict the next token at each time.",ha="center",fontsize=8.5)
    fig.text(.52,.042,"Earlier successors previously followed the current token; their uniform mass is subtracted. Shading: 95% confidence intervals.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.105,right=.96,bottom=.22,top=.90)
    outputs={}
    for extension in ("pdf","png"):
        path=out/("copying_response."+extension)
        fig.savefig(path,dpi=180,metadata={"CreationDate":None,"ModDate":None} if extension=="pdf" else None)
        outputs[path.name]=dict(path=relative(path),sha256=sha(path))
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11.3,8.3),gridspec_kw={"hspace":.45,"wspace":.3})
    for column,(group,title) in enumerate((("depth_0","Independent uniform sequences (depth 0)"),("depth_positive","Sequences with transition structure (depth 1–8)"))):
        values=report["groups"][group]
        for name in CONDITIONS:
            line(axes[0,column],positions,values["curves"]["excess_"+name],labels[name],colors[name],"--" if name=="control_deleted" else "-")
        axes[0,column].set_title(f"{title}\n{values['n_sequences']} independent sequences",fontsize=11)
        configure_axis(axes[0,column],n,"Extra probability on earlier successors")
        for key,label,color in (("benefit_copier_bits","Loss increase after copying-head removal",colors["copier_deleted"]),
                                ("benefit_control_bits","Loss increase after comparison-head removal",colors["control_deleted"]),
                                ("benefit_bayes_bits","Uniform loss minus Bayes loss",colors["bayes"])):
            line(axes[1,column],positions,values["curves"][key],label,color)
        configure_axis(axes[1,column],n,"Expected next-token loss difference (bits)")
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[1,0].legend(frameon=False,fontsize=8)
    fig.suptitle("Source controls and predictive benefit"+(" — smoke check" if report["smoke"] else ""),fontsize=14,y=.99)
    fig.text(.52,.045,"Head removals are compared with the intact transformer; Bayes is compared with the known uniform distribution.",ha="center",fontsize=9)
    fig.text(.52,.017,"All signed values are retained. Shading: pointwise 95% confidence intervals across independent sequences; curves are unsmoothed.",ha="center",fontsize=8.5)
    fig.subplots_adjust(left=.085,right=.965,bottom=.13,top=.90)
    for extension in ("pdf","png"):
        path=out/("usefulness."+extension)
        fig.savefig(path,dpi=180,metadata={"CreationDate":None,"ModDate":None} if extension=="pdf" else None)
        outputs[path.name]=dict(path=relative(path),sha256=sha(path))
    plt.close(fig)
    return outputs


def save_table(report,path):
    with Path(path).open("w",newline="") as stream:
        writer=csv.writer(stream,delimiter="\t",lineterminator="\n")
        writer.writerow(["group","quantity","scope","first_prefix","last_prefix","mean","sequence_standard_error","n_sequences","ci95_lower","ci95_upper"])
        for group,summary in report["groups"].items():
            entries=[]
            selected_prefixes={1,2,4,8,16,32,64,128,256,512,1024,len(report["prefix_lengths"])}
            for key,value in summary["curves"].items():
                entries.extend((key,"prefix",t,t,mean,se,value) for t,mean,se in zip(report["prefix_lengths"],value["mean"],value["se"]) if t in selected_prefixes)
            for scope,values,first,last in (("whole_sequence_mean",summary["time_averages"],1,len(report["prefix_lengths"])),
                                            ("final_prefix",summary["final_prefix"],len(report["prefix_lengths"]),len(report["prefix_lengths"]))):
                entries.extend((key,scope,first,last,value["mean"],value["se"],value) for key,value in values.items())
            for window in summary["windows"]:
                entries.extend((key,"sequence_window_mean",window["first_prefix"],window["last_prefix"],value["mean"],value["se"],value)
                               for key,value in window["statistics"].items())
            for key,scope,first,last,mean,se,value in entries:
                radius=se*value["ci95_multiplier"] if se is not None and value["ci95_multiplier"] is not None else None
                row=[group,key,scope,first,last,mean,se,value["n"],mean-radius if radius is not None else None,mean+radius if radius is not None else None]
                writer.writerow(["NA" if item is None else item for item in row])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"output/copier_time/run_2026-09-15")
    parser.add_argument("--out",type=Path,default=ROOT/"results/transformer/copier_time")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(argv)
    report=aggregate(args.input,smoke=args.smoke)
    if not args.check_only:
        args.out.mkdir(parents=True,exist_ok=True)
        report["artifacts"]=plot(report,args.out)
        table=args.out/"summary.tsv"
        save_table(report,table)
        report["artifacts"][table.name]=dict(path=relative(table),sha256=sha(table))
        (args.out/"summary.json").write_text(json.dumps(report,separators=(",",":"),allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True,smoke=args.smoke,n_sequences=report["n_sequences"],
                          source_depth_counts=report["source_depth_counts"],maximum_check_errors=report["maximum_check_errors"],output=relative(args.out))))


if __name__=="__main__":
    main()
