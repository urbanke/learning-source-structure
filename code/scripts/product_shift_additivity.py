"""Check the one-versus-two factor-change prediction from saved paired risks.

The independent, identically distributed factors give the population Bayes
identity E[R_both - 2 R_one + R_stationary] = 0 at each context length.
This does not hold draw by draw. Transformer agreement is an empirical test.
No source trajectories, Bayesian predictions, or model outputs are generated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/product_shift_2026-09-18"
RESULTS = ROOT / "results/transformer/product_shift_2026-09-18"
METRICS = ("bayes_regret_bits", "transformer_regret_bits")
CONDITIONS = ["stationary", "relabeled", "switch_one", "switch_both"]
LABELS = {"bayes_regret_bits": "Bayes", "transformer_regret_bits": "Transformer"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(array):
    mean = array.mean(axis=0)
    se = array.std(axis=0, ddof=1) / np.sqrt(len(array))
    return {"mean": mean.tolist(), "se": se.tolist(),
            "lower_pointwise_95": (mean - 1.96 * se).tolist(),
            "upper_pointwise_95": (mean + 1.96 * se).tolist()}


def load(source):
    from scipy.special import xlogy

    receipt = json.loads((source / "summary.json").read_text())
    require(receipt["complete"] is True, "Requires the complete frozen cohort")
    require(receipt["probabilities_renormalized"] is False, "Unexpected post-processing")
    with np.load(source / "per_sequence_risks.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    require(str(data["plan_id"]) == receipt["plan_id"], "Plan identity mismatch")
    require(data["conditions"].tolist() == CONDITIONS, "Condition order changed")
    require(data["depth_pairs"].tolist() == [[1, 1], [4, 4]], "Unexpected depths")
    n = receipt["expected_n_sequences_per_depth"]
    require(n == 1024 and data["indices"].tolist() == list(range(n)), "Incomplete cohort")
    require(data["times"].tolist() == receipt["times"], "Context grid mismatch")
    times = data["times"]
    pre = times < receipt["switch_observed"]
    checks = {"independent_raw_risk_max_error": 0.0, "summary_mean_max_error": 0.0}
    for di, pair in enumerate(data["depth_pairs"]):
        for metric in METRICS:
            array = data[metric][di]
            require(array.shape == (n, len(CONDITIONS), len(times)), "Risk shape mismatch")
            require(np.isfinite(array).all(), "Nonfinite risk")
            contrast = array[:, 3] - 2 * array[:, 2] + array[:, 0]
            require(np.array_equal(contrast[:, pre], np.zeros_like(contrast[:, pre])),
                    "Prechange contrast must be exactly zero")
            for ci, condition in enumerate(CONDITIONS):
                group = next(g for g in receipt["groups"]
                             if g["depth_pair"] == pair.tolist() and g["condition"] == condition)
                error = float(np.max(np.abs(array[:, ci].mean(0) - group["metrics"][metric]["mean"])))
                checks["summary_mean_max_error"] = max(checks["summary_mean_max_error"], error)
                require(error < 2e-12, "Saved risks disagree with existing summary")
        # Read a few original probability records independently of the summary.
        for index in (0, 511, 1023):
            path = source / "records" / f"L{pair[0]}_{pair[1]}" / f"seq_{index:06d}.npz"
            with np.load(path, allow_pickle=False) as raw:
                require(str(raw["plan_id"]) == receipt["plan_id"], "Raw identity mismatch")
                p = raw["true_probs"]
                for metric, key in zip(METRICS, ("bayes_probs", "transformer_probs")):
                    q = raw[key]
                    require(np.all(q > 0), "Predictor has zero support")
                    risk = np.sum(xlogy(p, p) - xlogy(p, q), axis=-1) / np.log(2)
                    error = float(np.max(np.abs(risk - data[metric][di, index])))
                    checks["independent_raw_risk_max_error"] = max(checks["independent_raw_risk_max_error"], error)
                    require(error < 2e-12, "Saved risks disagree with independent raw reconstruction")
    checks["prechange_contrasts_exactly_zero"] = True
    return receipt, data, checks


def simultaneous_bootstrap(contrasts, repetitions, seed):
    """Paired source bootstrap; a single max covers depths, predictors, times.

    contrasts has shape (depth, source, predictor, postchange time). Resampling
    is independent between depth cohorts and paired over conditions/predictors/
    contexts within each cohort. Centered bootstrap means are standardized by
    the reported paired standard errors (a fixed-SE max-statistic band).
    """
    nd, n, _, _ = contrasts.shape
    means = contrasts.mean(1)
    ses = contrasts.std(1, ddof=1) / np.sqrt(n)
    require(np.all(ses > 0), "Postchange contrast has zero SE")
    centered = contrasts - means[:, None]
    maxima = np.zeros(repetitions)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(nd)]
    for di in range(nd):
        matrix = centered[di].reshape(n, -1)
        scale = ses[di].reshape(-1)
        for start in range(0, repetitions, 200):
            count = min(200, repetitions - start)
            weights = rngs[di].multinomial(n, np.full(n, 1 / n), size=count)
            z = (weights @ matrix) / (n * scale)
            maxima[start:start + count] = np.maximum(maxima[start:start + count], np.abs(z).max(1))
    critical = float(np.quantile(maxima, .95, method="higher"))
    observed = float(np.max(np.abs(means / ses)))
    return dict(seed=seed, repetitions=repetitions, confidence=.95,
                method="Centered paired nonparametric source bootstrap; maximum absolute mean deviation divided by reported paired SE",
                family="Both depth cohorts, both predictors, all 14 postchange grid points (56 contrasts)",
                critical_value=critical, maximum_observed_standardized_contrast=observed,
                approximate_global_p_value=float((1 + np.count_nonzero(maxima >= observed)) / (repetitions + 1)),
                zero_in_all_simultaneous_intervals=bool(observed <= critical))


def save_figure(fig, directory, stem):
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"{stem}.{ext}", dpi=220,
                    metadata={"Creator": "product_shift_additivity.py"})


def plots(report, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "axes.linewidth": .6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    times = np.asarray(report["postchange_times"])
    depth_colors = {1: "#BF611C", 4: "#288167"}
    dark = "#333333"

    def xaxis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xlim(245, 1080)
        ax.set_xticks([256, 512, 1024], ["256", "512", "1024"])
        ax.minorticks_off()
        ax.set_xlabel("Observed tokens, $t$")
        ax.grid(axis="y", color="#D8DEE5", linewidth=.45)
        ax.tick_params(length=2.5, pad=2)

    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
        for panel, (ax, depth) in enumerate(zip(axes, (1, 4))):
            row = next(r for r in report["groups"] if r["depth"] == depth and r["metric"] == metric)
            color = depth_colors[depth]
            for name, linestyle, alpha in (("two_factor_excess", "-", .14),
                                            ("twice_one_factor_excess", ":", .08)):
                s = row[name]
                ax.fill_between(times, s["lower_pointwise_95"], s["upper_pointwise_95"],
                                color=color, alpha=alpha, linewidth=0)
                ax.plot(times, s["mean"], color=color, linestyle=linestyle,
                        linewidth=1.25)
            ax.set_title(rf"({chr(97 + panel)}) Source depth $L_1=L_2={depth}$",
                         loc="left", pad=5)
            ax.set_ylim(bottom=0)
            xaxis(ax)
        axes[0].set_ylabel("Additional regret over stationary control (bits)")
        handles = [Line2D([], [], color=dark, linewidth=1.25),
                   Line2D([], [], color=dark, linestyle=":", linewidth=1.25)]
        fig.legend(handles, ["Two factors changed: measured", "Twice one-factor change: prediction"],
                   ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.01), frameon=False)
        fig.text(.5, .012, f"{LABELS[metric]}; 1,024 paired sources per depth; shading: pointwise 95% intervals.",
                 fontsize=7, ha="center")
        fig.tight_layout(rect=(0, .045, 1, .885), w_pad=1.4)
        save_figure(fig, results, f"additivity_{LABELS[metric].lower()}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for panel, (ax, depth) in enumerate(zip(axes, (1, 4)), start=2):
        color = depth_colors[depth]
        ax.axhline(0, color=".3", linewidth=.75)
        tf_row = next(r for r in report["groups"]
                      if r["depth"] == depth and r["metric"] == "transformer_regret_bits")
        tf = tf_row["contrast"]
        ax.fill_between(times, tf["lower_simultaneous_95"], tf["upper_simultaneous_95"],
                        color=color, alpha=.12, linewidth=0)
        ax.plot(times, tf["mean"], color=color, linewidth=1.25)
        bayes_row = next(r for r in report["groups"]
                         if r["depth"] == depth and r["metric"] == "bayes_regret_bits")
        bayes = bayes_row["contrast"]
        mean = np.asarray(bayes["mean"])
        lower = np.asarray(bayes["lower_simultaneous_95"])
        upper = np.asarray(bayes["upper_simultaneous_95"])
        ax.errorbar(times, mean, yerr=np.vstack((mean-lower, upper-mean)),
                    color=color, linestyle="none", marker="o", markersize=3.2,
                    markerfacecolor="white", markeredgewidth=.75,
                    elinewidth=.55, capsize=1.1, zorder=5)
        ax.set_title(rf"({chr(97 + panel)}) Source depth $L_1=L_2={depth}$",
                     loc="left", pad=5)
        xaxis(ax)
    axes[0].set_ylabel(r"$R_{\rm both}-2R_{\rm one}+R_{\rm stationary}$ (bits)")
    fig.legend([Line2D([], [], color=dark, linewidth=1.25),
                Line2D([], [], color=dark, linestyle="none", marker="o",
                       markerfacecolor="white", markersize=4)],
               ["Transformer", "Informed Bayes"], ncol=2,
               loc="upper center", bbox_to_anchor=(.5, 1.01), frameon=False)
    fig.text(.5, .012, "Intervals: bootstrap 95% simultaneous bands across both depths, predictors, and postchange times.",
             fontsize=7, ha="center")
    fig.tight_layout(rect=(0, .045, 1, .885), w_pad=1.4)
    save_figure(fig, results, "additivity_contrast")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=SOURCE / "additivity")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    require(args.bootstrap >= 1000, "Use at least 1,000 bootstrap replications")
    receipt, data, checks = load(args.source)
    post = data["times"] >= receipt["switch_observed"]
    require(int(post.sum()) == 14, "Unexpected postchange grid")
    contrasts = np.stack([np.stack([data[key][di, :, 3, :] - 2 * data[key][di, :, 2, :]
                                   + data[key][di, :, 0, :] for key in METRICS], axis=1)[:, :, post]
                          for di in range(2)])
    bootstrap = simultaneous_bootstrap(contrasts, args.bootstrap, args.seed)
    report = dict(schema="product-shift-additivity-v1", plan_id=receipt["plan_id"],
                  input_sha256={name: sha(args.source / name) for name in ("summary.json", "per_sequence_risks.npz")},
                  n_sources_per_depth=1024, postchange_times=data["times"][post].tolist(),
                  switch_observed=receipt["switch_observed"], bootstrap=bootstrap, checks=checks,
                  uncertainty_scope="Source/trajectory sampling; one frozen transformer training seed",
                  interpretation="Exact population Bayes identity from independent identically distributed factors; empirical transformer prediction. No equivalence claim follows from absence of a significant contrast.",
                  no_new_predictions=True, probabilities_clipped_or_renormalized=False, groups=[])
    endpoint_rows = []
    for di, pair in enumerate(data["depth_pairs"]):
        for ki, key in enumerate(METRICS):
            risk = data[key][di]
            one = (risk[:, 2] - risk[:, 0])[:, post]
            both = (risk[:, 3] - risk[:, 0])[:, post]
            contrast = contrasts[di, :, ki]
            s = statistics(contrast)
            mean, se = np.asarray(s["mean"]), np.asarray(s["se"])
            critical = bootstrap["critical_value"]
            s.update(lower_simultaneous_95=(mean - critical * se).tolist(),
                     upper_simultaneous_95=(mean + critical * se).tolist())
            row = dict(depth=int(pair[0]), metric=key, one_factor_excess=statistics(one),
                       two_factor_excess=statistics(both), twice_one_factor_excess=statistics(2 * one),
                       contrast=s, pointwise_95_excludes_zero_at=data["times"][post][np.abs(mean) > 1.96 * se].tolist(),
                       max_absolute_standardized_contrast=float(np.max(np.abs(mean / se))))
            report["groups"].append(row)
            endpoint_rows.append(dict(depth=int(pair[0]), predictor=LABELS[key], observed_tokens=1024,
                one_factor_excess_mean_bits=float(one[:, -1].mean()),
                two_factor_excess_mean_bits=float(both[:, -1].mean()),
                twice_one_factor_prediction_bits=float(2 * one[:, -1].mean()),
                paired_contrast_bits=float(mean[-1]), paired_contrast_se_bits=float(se[-1]),
                paired_contrast_simultaneous_95_lower_bits=float(mean[-1] - critical * se[-1]),
                paired_contrast_simultaneous_95_upper_bits=float(mean[-1] + critical * se[-1])))
    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.results / "additivity_endpoints.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(endpoint_rows)
    plots(report, args.results)
    print(json.dumps(dict(bootstrap=bootstrap, checks=checks, endpoint_rows=endpoint_rows), indent=2))


if __name__ == "__main__":
    main()
