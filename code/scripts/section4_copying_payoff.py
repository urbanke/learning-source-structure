"""Extract payoff crossings from already saved copying interventions.

No model or Bayesian inference is rerun. Whole sequences are the resampling
units. The last crossing is only required to remain positive through the saved
horizon, not at all subsequent times. Source-depth grouping changes reporting,
not the information supplied to either predictor.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def copier_helpers():
    """Load numerical audit helpers only for validation/reanalysis paths."""
    from copier_time_figure import (CONDITIONS, array_hash, fingerprint,
                                    independent_metrics, relative, sha)
    return CONDITIONS, array_hash, fingerprint, independent_metrics, relative, sha

NAMES = ("transformer", "bayes", "control", "transformer_vs_uniform", "disabled_vs_uniform")
KEYS = ("benefit_copier_bits", "benefit_bayes_bits", "benefit_control_bits", "benefit_transformer_uniform_bits", "disabled_minus_uniform_bits")
ZERO_TOLERANCE = 1e-12  # Merely excludes floating-point noise at exact equality.


def crossings(values):
    """First positive point and start of final positive run; infinity is censored."""
    x = np.asarray(values)
    if x.ndim == 1:
        x = x[None, :]
    positive = x > ZERO_TOLERANCE
    n = x.shape[1]
    first = np.where(positive.any(axis=1), positive.argmax(axis=1) + 1, np.inf)
    last_nonpositive = np.where(~positive, np.arange(n), -1).max(axis=1)
    final_run = np.where(positive[:, -1], last_nonpositive + 2, np.inf)
    return first.astype(float), final_run.astype(float)


def serial_time(value):
    return int(value) if np.isfinite(value) else None


def percentile(values):
    # An order statistic keeps right-censored draws as +infinity; no filtering.
    result = np.quantile(values, [.025, .5, .975], method="inverted_cdf")
    return dict(zip(("lower", "median", "upper"), map(serial_time, result)))


def describe_crossing(mean, boot, which):
    first, last = crossings(mean)
    point = first[0] if which == "first_positive" else last[0]
    if not np.isfinite(point):
        status = "no_positive_point" if which == "first_positive" else "not_positive_at_horizon"
    elif point == 1:
        status = "positive_from_start"
    else:
        status = "upward_crossing_after_nonpositive_values"
    return dict(time=serial_time(point), status=status,
                bootstrap_percentiles=percentile(boot),
                bootstrap_censored_count=int(np.isinf(boot).sum()),
                bootstrap_positive_from_start_count=int((boot == 1).sum()))


def load_records(directory, audit_path):
    CONDITIONS, array_hash, fingerprint, independent_metrics, relative, sha = copier_helpers()
    previous = json.loads(audit_path.read_text())
    if not (previous["passed"] and previous["publication_cohort"] and previous["n_sequences"] == 1024):
        raise ValueError("Expected the validated complete natural-sequence cohort")
    spec = json.loads((directory / "protocol.json").read_text())
    if fingerprint(spec) != previous["spec_id"]:
        raise ValueError("Saved protocol differs from independent audit")
    if sha(directory / "protocol.json") != previous["protocol"]["sha256"]:
        raise ValueError("Protocol checksum changed")
    if len(list(directory.glob("seq_*.npz"))) != spec["n_seq"]:
        raise ValueError("Incomplete or unexpected cohort")
    values = np.empty((spec["n_seq"], len(NAMES), spec["n_ctx"]), dtype=np.float64)
    depths = np.empty(spec["n_seq"], dtype=np.int64)
    maximum_metric_error = 0.
    maximum_cumulative_error = 0.
    inputs = []
    for source in previous["inputs"]:
        index = source["index"]
        path = directory / f"seq_{index:06d}.npz"
        raw_hash = sha(path)
        if raw_hash != source["sha256"]:
            raise ValueError(f"Raw data changed since independent validation: {path}")
        with np.load(path, allow_pickle=False) as raw:
            tokens = raw["tokens"]
            predictions = {name: raw[name + "_probs"] for name in CONDITIONS}
            computed = independent_metrics(tokens, predictions, raw["transition_probs"])
            if array_hash(tokens) != source["tokens_sha256"]:
                raise ValueError("Token identity mismatch")
            depths[index] = int(raw["source_depth"])
            computed[KEYS[3]] = computed["oracle_ce_uniform"] - computed["oracle_ce_intact"]
            computed[KEYS[4]] = computed["oracle_ce_copier_deleted"] - computed["oracle_ce_uniform"]
            for j, key in enumerate(KEYS):
                values[index, j] = computed[key]
                if j > 2:
                    continue
                maximum_metric_error = max(maximum_metric_error, float(np.max(np.abs(computed[key] - raw["metric_" + key]))))
                maximum_cumulative_error = max(maximum_cumulative_error, float(np.max(np.abs(np.cumsum(computed[key]) - raw["metric_cumulative_" + key]))))
            inputs.append(dict(index=index, source_depth=int(depths[index]), sha256=raw_hash))
        if (index + 1) % 128 == 0:
            print(json.dumps(dict(event="verified_saved_sequences", completed=index+1, total=spec["n_seq"])), flush=True)
    if maximum_metric_error > 2e-10 or maximum_cumulative_error > 2e-8:
        raise ValueError("Recomputed losses disagree with saved records")
    audit_errors = []
    for depth in range(9):
        group = previous["groups"][f"depth_{depth}"]
        for j, key in enumerate(KEYS[:3]):
            selected = values[depths == depth, j]
            audit_errors.extend([float(np.max(np.abs(selected.mean(0) - group["curves"][key]["mean"]))),
                                 float(np.max(np.abs(selected.std(0, ddof=1)/np.sqrt(len(selected)) - group["curves"][key]["se"]))),
                                 abs(float(np.cumsum(selected, axis=1)[:, -1].mean()) - group["final_prefix"]["cumulative_" + key]["mean"])])
    if max(audit_errors) > 2e-8:
        raise ValueError("New aggregation differs from existing audited means/standard errors")
    validation = dict(passed=True, maximum_saved_metric_error=maximum_metric_error,
                      maximum_saved_cumulative_error=maximum_cumulative_error,
                      maximum_previous_summary_error=max(audit_errors),
                      independent_audit=dict(path=relative(audit_path), sha256=sha(audit_path)),
                      bayes_numerical_validation=previous["numerical_validation"],
                      input_manifest_sha256=fingerprint(inputs))
    return values, depths, spec, validation


def analyze(values, depths, spec, n_bootstrap, seed):
    from scipy.stats import t as student_t

    groups = []
    arrays = dict(time=np.arange(1, spec["n_ctx"]+1), depths=np.arange(9))
    bootstrap_arrays = {}
    for depth in range(9):
        selected = values[depths == depth]
        n = len(selected)
        both = np.stack((selected, np.cumsum(selected, axis=2)), axis=2)
        means = both.mean(axis=0)
        errors = both.std(axis=0, ddof=1) / np.sqrt(n)
        arrays[f"L{depth}_mean"] = means
        arrays[f"L{depth}_se"] = errors
        arrays[f"L{depth}_n"] = np.asarray(n)
        # One multinomial draw per full sequence; paired across every predictor
        # and time. Matrix multiplication avoids resampling the full arrays.
        rng = np.random.default_rng(np.random.SeedSequence([seed, depth]))
        weights = rng.multinomial(n, np.full(n, 1/n), size=n_bootstrap).astype(float) / n
        boot_inst = (weights @ selected.reshape(n, -1)).reshape(n_bootstrap, len(NAMES), spec["n_ctx"])
        curves = {}
        for j, name in enumerate(NAMES):
            curves[name] = {}
            for k, kind in enumerate(("instantaneous", "cumulative")):
                boot_values = boot_inst[:, j] if k == 0 else np.cumsum(boot_inst[:, j], axis=1)
                first, last = crossings(boot_values)
                bootstrap_arrays[f"L{depth}_{name}_{kind}_first"] = first
                bootstrap_arrays[f"L{depth}_{name}_{kind}_last"] = last
                curves[name][kind] = dict(
                    first_positive=describe_crossing(means[j, k], first, "first_positive"),
                    final_positive_run=describe_crossing(means[j, k], last, "final_positive_run"),
                    at_times={str(t): dict(mean=float(means[j, k, t-1]), se=float(errors[j, k, t-1]))
                              for t in (1, 4, 16, 64, 128, 256, 512, 1024)},
                )
        groups.append(dict(depth=depth, n_sequences=n, ci95_multiplier=float(student_t.ppf(.975,n-1)), curves=curves))
        print(json.dumps(dict(event="bootstrap_completed", depth=depth, n_sequences=n, repetitions=n_bootstrap)), flush=True)
    return groups, arrays, bootstrap_arrays


def plot(groups, arrays, out):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/section4_copying_payoff/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
                         "axes.titlesize":8, "axes.labelsize":8,
                         "xtick.labelsize":7.5, "ytick.labelsize":7.5,
                         "legend.fontsize":7, "axes.linewidth":.6,
                         "pdf.fonttype":42, "ps.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False})
    colors = dict(transformer="#2468A0", bayes="#BF611C", control="#777777")
    labels = dict(transformer="Transformer: path disabled − intact",
                  bayes="Informed Bayes: uniform − Bayes",
                  control="Control path disabled − intact")
    t = arrays["time"]
    bayes_points = np.unique(np.rint(2 ** np.linspace(0, 10, 25)).astype(int))
    bayes_points = bayes_points[(bayes_points >= 1) & (bayes_points <= 1024)]
    bayes_indices = bayes_points - 1

    def draw(ax, depth, kind, control=False, show_crossing=False):
        g = groups[depth]
        means, errors = arrays[f"L{depth}_mean"], arrays[f"L{depth}_se"]
        for j, name in enumerate(NAMES[:3] if control else NAMES[:2]):
            mean, se = means[j, kind], errors[j, kind]
            ax.fill_between(t, mean-g["ci95_multiplier"]*se, mean+g["ci95_multiplier"]*se,
                            color=colors[name], alpha=.14, lw=0)
            if name == "bayes":
                ax.plot(t[bayes_indices], mean[bayes_indices], color=colors[name],
                        linestyle="none", marker="o", markersize=3.0,
                        markerfacecolor="white", markeredgewidth=.7,
                        label=labels[name], zorder=5)
            else:
                ax.plot(t, mean, color=colors[name], lw=1.2,
                        ls={"control":"-.", "transformer":"-"}[name],
                        label=labels[name])
            crossing = g["curves"][name][("instantaneous", "cumulative")[kind]]["final_positive_run"]
            if show_crossing and crossing["time"] is not None:
                x = crossing["time"]
                ax.axvline(x, color=colors[name], lw=.8, ls=":", alpha=.8)
                ax.annotate(f"{x}", (x, .96 if name == "transformer" else .86),
                            xycoords=("data", "axes fraction"), xytext=(3, 0),
                            textcoords="offset points", color=colors[name],
                            va="top", fontsize=7.5)
        ax.axhline(0, color="#4B5563", lw=.75)
        ax.set_xscale("log", base=2)
        ax.set_xlim(.9,1180)
        ax.set_xticks([1, 4, 16, 64, 256, 1024],
                      ["1", "4", "16", "64", "256", "1024"])
        ax.minorticks_off()
        ax.grid(axis="y", color="#D8DEE5", lw=.45)
        ax.tick_params(length=2.5, pad=2)
        ax.set_xlabel("Observed tokens, $t$")
        ax.set_ylabel("Next-token loss difference (bits)" if kind == 0 else "Cumulative loss difference (bits)")

    fig, axes = plt.subplots(1,2,figsize=(6.5,3.0))
    for k, ax in enumerate(axes):
        draw(ax, 1, k, show_crossing=True)
        ax.set_title(("(a) Next-token benefit", "(b) Cumulative benefit")[k],
                     loc="left", pad=5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",bbox_to_anchor=(.5,.005),
               ncol=2,frameon=False)
    fig.subplots_adjust(left=.115,right=.97,bottom=.25,top=.92,wspace=.38)
    for ext in ("pdf","png"):
        fig.savefig(out/f"payoff.{ext}",dpi=200,
                    metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else None)
    plt.close(fig)

    for kind, name in enumerate(("instantaneous","cumulative")):
        fig, axes = plt.subplots(3,3,figsize=(12,9))
        for depth, ax in enumerate(axes.flat):
            draw(ax,depth,kind,control=True)
            ax.set_title(f"Depth {depth}; {groups[depth]['n_sequences']} sequences",fontsize=8.5)
            if depth % 3:
                ax.set_ylabel("")
            if depth < 6:
                ax.set_xlabel("")
        handles, names = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,names,loc="lower center",bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=9)
        fig.subplots_adjust(left=.075,right=.99,bottom=.09,top=.96,hspace=.40,wspace=.28)
        for ext in ("pdf","png"):
            fig.savefig(out/f"payoff_all_depths_{name}.{ext}",dpi=170,
                        metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else None)
        plt.close(fig)

    fig, axes = plt.subplots(4,2,figsize=(6.4,9.6))
    for row,depth in enumerate((0,1,4,8)):
        for kind in range(2):
            draw(axes[row,kind],depth,kind,control=True)
            axes[row,kind].set_title(f"Depth {depth}; {groups[depth]['n_sequences']} sequences",fontsize=8.5)
            axes[row,kind].set_ylabel("Next-token benefit (bits)" if kind == 0 else "Cumulative benefit (bits)")
            if row < 3:
                axes[row,kind].set_xlabel("")
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=7.6)
    fig.subplots_adjust(left=.13,right=.95,bottom=.105,top=.97,hspace=.43,wspace=.50)
    for ext in ("pdf","png"):
        fig.savefig(out/f"all_depths.{ext}",dpi=170,
                    metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else None)
    plt.close(fig)


def check_only(out):
    """Fast integrity/recalculation check; full mode alone rereads every raw NPZ."""
    _, _, fingerprint, _, relative, sha = copier_helpers()
    report = json.loads((out / "summary.json").read_text())
    artifacts = json.loads((out / "artifacts.json").read_text())
    for name, item in artifacts.items():
        path = out / name
        if relative(path) != item["path"] or sha(path) != item["sha256"]:
            raise ValueError(f"Retained artifact changed: {path}")
    source_receipts = [report["protocol"], report["source_checkpoint"], report["script"],
                       report["validation"]["independent_audit"],
                       report["validation"]["bayes_numerical_validation"]]
    for item in source_receipts:
        path = Path(item["path"])
        path = path if path.is_absolute() else ROOT / path
        if sha(path) != item["sha256"]:
            raise ValueError(f"Source/provenance file changed: {path}")
    old = json.loads((ROOT / report["validation"]["independent_audit"]["path"]).read_text())
    spec = json.loads((ROOT / report["protocol"]["path"]).read_text())
    if fingerprint(spec) != old["spec_id"] or not report["validation"]["passed"]:
        raise ValueError("Protocol or validation mismatch")
    manifest = [dict(index=item["index"], source_depth=item["source_depth"], sha256=item["sha256"])
                for item in old["inputs"]]
    if fingerprint(manifest) != report["validation"]["input_manifest_sha256"]:
        raise ValueError("Input-manifest identity mismatch")
    with np.load(out / "curves.npz", allow_pickle=False) as saved:
        arrays = {key:saved[key] for key in saved.files}
    with np.load(out / "bootstrap_crossings.npz", allow_pickle=False) as saved:
        bootstrap = {key:saved[key] for key in saved.files}
    count = 0
    for group in report["groups"]:
        depth = group["depth"]
        mean, se = arrays[f"L{depth}_mean"], arrays[f"L{depth}_se"]
        if mean.shape != (len(NAMES),2,1024) or se.shape != mean.shape:
            raise ValueError("Unexpected curve dimensions")
        if not np.isfinite(mean).all() or not np.isfinite(se).all() or np.any(se < 0):
            raise ValueError("Invalid curve values")
        if group["n_sequences"] != int(arrays[f"L{depth}_n"]):
            raise ValueError("Group size mismatch")
        if not np.allclose(np.cumsum(mean[:,0],axis=-1),mean[:,1],rtol=0,atol=1e-8):
            raise ValueError("Cumulative mean is not the sum of instantaneous means")
        if not np.allclose(mean[0],mean[3]+mean[4],rtol=0,atol=1e-9):
            raise ValueError("Uniform-baseline decomposition failed")
        for j,name in enumerate(NAMES):
            for k,kind in enumerate(("instantaneous","cumulative")):
                curve = group["curves"][name][kind]
                for suffix, definition in (("first","first_positive"),("last","final_positive_run")):
                    sample = bootstrap[f"L{depth}_{name}_{kind}_{suffix}"]
                    if sample.shape != (report["uncertainty"]["repetitions"],):
                        raise ValueError("Wrong bootstrap sample count")
                    if np.isnan(sample).any() or np.any(sample < 1) or np.any(np.isfinite(sample) & (sample > 1024)):
                        raise ValueError("Invalid bootstrap crossing/censoring value")
                    rebuilt = describe_crossing(mean[j,k], sample, definition)
                    if rebuilt != curve[definition]:
                        raise ValueError(f"Crossing summary mismatch: {depth}/{name}/{kind}/{definition}")
                for time, value in curve["at_times"].items():
                    t = int(time)-1
                    if value != dict(mean=float(mean[j,k,t]), se=float(se[j,k,t])):
                        raise ValueError("Selected numerical result changed")
                count += 1
            if j < 3:
                reference = old["groups"][f"depth_{depth}"]["curves"][KEYS[j]]
                if not np.array_equal(mean[j,0], reference["mean"]) or not np.array_equal(se[j,0], reference["se"]):
                    raise ValueError("Curves no longer match the old independent audit")
    print(json.dumps(dict(status="passed", check="retained_artifacts_and_crossing_recalculation",
                          curves=count, crossing_definitions=2*count,
                          bootstrap_repetitions=report["uncertainty"]["repetitions"],
                          raw_arrays_reread=False,
                          scope="Hashes of retained outputs and source/provenance files; all saved crossing values and bootstrap percentiles reconstructed. Full analysis validates every raw sequence.")), flush=True)


def main():
    _, _, _, _, relative, sha = copier_helpers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"output/copier_time/run_2026-09-15")
    parser.add_argument("--out", type=Path, default=ROOT/"results/transformer/section4_copying_payoff_2026-09-17")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--check-only", action="store_true", help="Check retained artifacts and rederive every saved crossing and percentile")
    args = parser.parse_args()
    if args.check_only:
        check_only(args.out)
        return
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    args.out.mkdir(parents=True,exist_ok=True)
    previous = ROOT/"results/transformer/copier_time/summary.json"
    values, depths, spec, validation = load_records(args.input, previous)
    groups, arrays, bootstrap_arrays = analyze(values, depths, spec, args.bootstrap, args.seed)
    np.savez_compressed(args.out/"curves.npz", **arrays)
    np.savez_compressed(args.out/"bootstrap_crossings.npz", **bootstrap_arrays)
    report = dict(schema="copying-payoff-crossings-v1", horizon=spec["n_ctx"], n_sequences=spec["n_seq"],
                  sequence_length=spec["n_ctx"]+1, source_alpha=0, alphabet_size=64,
                  source_depth_prior=list(range(9)), candidate_head=[0,1], control_head=[0,0],
                  protocol=dict(path=relative(args.input/"protocol.json"),sha256=sha(args.input/"protocol.json")),
                  source_checkpoint=spec["model_metadata"]["checkpoint"],
                  script=dict(path=relative(__file__),sha256=sha(__file__)),
                  definitions={
                      "transformer":"True-row expected loss with identified copying head removed minus intact-network loss, on identical observed sequences.",
                      "bayes":"Known-uniform expected loss minus saved informed Bayesian predictive loss. This is a different baseline from the disabled transformer.",
                      "control":"True-row expected loss with independently selected comparison head removed minus intact-network loss.",
                      "transformer_vs_uniform":"Known-uniform expected loss minus intact-transformer expected loss; matches the Bayesian comparison baseline.",
                      "disabled_vs_uniform":"Disabled-transformer expected loss minus known-uniform expected loss; directly measures the baseline discrepancy.",
                      "first_positive":"Earliest saved time whose mean benefit exceeds numerical zero; early isolated positive points may precede the eventual crossing.",
                      "final_positive_run":"First time of the final strictly positive run through time 1024; right-censored if the last value is nonpositive. It is not a claim of positivity beyond 1024.",
                      "zero_tolerance_bits":ZERO_TOLERANCE,
                      "expectation":"At each observed sequence, average next-token log loss over its true transition row; then average over independent sequences conditional on source depth.",
                      "cumulative":"Sum of those paired expected next-token loss differences from t=1 through n; first token supplied free.",
                  },
                  uncertainty=dict(repetitions=args.bootstrap,seed=args.seed,
                      resampling="Whole source sequences, independently within each depth, paired across all times and predictors.",
                      curve_intervals="Pointwise 95% Student-t intervals; not simultaneous bands.",
                      crossing_intervals="2.5%, 50%, 97.5% order statistics across all bootstrap draws. Right-censored crossings retained as infinity (JSON null for a censored quantile); finite-only intervals are not reported.",
                      caution="Last crossings of unsmoothed empirical means are sensitive to late fluctuations. Bootstrap intervals are exploratory, particularly if multiple crossings or censoring occur."),
                  selection="Depth one is highlighted after reviewing existing results because it has delayed payoff; all nine depths and the comparison-head intervention remain in the appendix data.",
                  limitations=["The ablated transformer has not been identified with the original adaptive LSA unigram.",
                               "The intact transformer and informed Bayes are not pure copier components.",
                               "Agreement between these two payoff times is therefore an empirical comparison, not a direct test of the archived pure-unigram/pure-copier crossing formula.",
                               "A causal effect of deleting this path need not equal an additive component weight."],
                  validation=validation, groups=groups)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (args.out/"crossings.tsv").open("w") as stream:
        writer=csv.writer(stream,delimiter="\t",lineterminator="\n")
        writer.writerow(["depth","n_sequences","predictor","criterion","first_positive","last_crossing","status","bootstrap_lower","bootstrap_median","bootstrap_upper","censored_bootstraps"])
        for g in groups:
            for name in NAMES:
                for kind in ("instantaneous","cumulative"):
                    c=g["curves"][name][kind]; last=c["final_positive_run"]; q=last["bootstrap_percentiles"]
                    writer.writerow([g["depth"],g["n_sequences"],name,kind,c["first_positive"]["time"],last["time"],last["status"],q["lower"],q["median"],q["upper"],last["bootstrap_censored_count"]])
    plot(groups,arrays,args.out)
    artifacts={p.name:dict(path=relative(p),sha256=sha(p)) for p in args.out.iterdir() if p.is_file() and p.name!="artifacts.json"}
    (args.out/"artifacts.json").write_text(json.dumps(artifacts,indent=2)+"\n")
    print(json.dumps(dict(status="complete",out=relative(args.out),depth_one=groups[1]["curves"])),flush=True)


if __name__ == "__main__":
    main()
