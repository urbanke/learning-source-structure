"""Compare saved d128 transformers with corrected shared-depth Bayesian losses.

This measures payoff against the known uniform distribution, not the effect of
disabling a copying circuit. All losses use observed next tokens. The script
does no model inference, Bayesian integration, or training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
HORIZON = 1024
ZERO_TOLERANCE = 1e-12
PREDICTORS = ("transformer", "bayes")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def token_hash(tokens):
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype="<i8").tobytes()).hexdigest()


def last_crossing(curves):
    """Start of final strictly positive run; infinity if nonpositive at end."""
    curves = np.atleast_2d(curves)
    positive = curves > ZERO_TOLERANCE
    last_nonpositive = np.where(~positive, np.arange(curves.shape[1]), -1).max(1)
    return np.where(positive[:, -1], last_nonpositive + 2, np.inf)


def time_json(value):
    return int(value) if np.isfinite(value) else None


def quantiles(values):
    # Retain infinite (right-censored) outcomes rather than deleting them.
    values = np.quantile(values, [.025, .5, .975], method="inverted_cdf")
    return {key: time_json(value) for key, value in zip(("lower", "median", "upper"), values)}


def crossing_summary(curve, bootstrap):
    point = last_crossing(curve)[0]
    return dict(time=time_json(point),
                status=("not_positive_at_horizon" if not np.isfinite(point) else
                        "positive_from_start" if point == 1 else "final_upward_crossing"),
                bootstrap_percentiles=quantiles(bootstrap),
                bootstrap_censored_count=int(np.isinf(bootstrap).sum()),
                bootstrap_positive_from_start_count=int((bootstrap == 1).sum()))


def difference_bounds(first, second):
    """Bounds for first minus second crossing, preserving every censored draw."""
    first, second = np.broadcast_arrays(np.asarray(first, float), np.asarray(second, float))
    a, b = np.isfinite(first), np.isfinite(second)
    lo = np.full(first.shape, -np.inf)
    hi = np.full(first.shape, np.inf)
    both = a & b
    lo[both] = hi[both] = first[both] - second[both]
    lo[~a & b] = HORIZON + 1 - second[~a & b]
    hi[a & ~b] = first[a & ~b] - (HORIZON + 1)
    return lo, hi


def difference_summary(point, bootstrap):
    lo, hi = difference_bounds(bootstrap[:, 0], bootstrap[:, 1])
    pl, ph = difference_bounds([point[0]], [point[1]])
    both = np.isfinite(bootstrap).all(1)
    result = dict(
        definition="transformer crossing minus Bayesian crossing, in observed tokens",
        point_difference=time_json(pl[0]) if pl[0] == ph[0] else None,
        point_bounds=dict(lower=time_json(pl[0]), upper=time_json(ph[0])),
        paired_bootstrap_count=len(bootstrap),
        both_observed_count=int(both.sum()),
        transformer_only_censored_count=int((~np.isfinite(bootstrap[:, 0]) & np.isfinite(bootstrap[:, 1])).sum()),
        bayes_only_censored_count=int((np.isfinite(bootstrap[:, 0]) & ~np.isfinite(bootstrap[:, 1])).sum()),
        both_censored_count=int((~np.isfinite(bootstrap).any(1)).sum()),
        conservative_percentile_interval=dict(
            lower=time_json(np.quantile(lo, .025, method="inverted_cdf")),
            upper=time_json(np.quantile(hi, .975, method="inverted_cdf"))),
        null_bound_meaning="unbounded; all censored replicates retained",
    )
    if both.all():
        result["paired_bootstrap_percentiles"] = quantiles(bootstrap[:, 0] - bootstrap[:, 1])
    return result, lo, hi


def validate_definitions():
    curves = np.array([[-1, 0, 1, 2], [1, -1, 1, 2], [1, 2, -1, 0], [1, 2, 3, 4]])
    np.testing.assert_array_equal(last_crossing(curves), [3, 3, np.inf, 1])
    lo, hi = difference_bounds([5, np.inf, 5, np.inf], [3, 3, np.inf, np.inf])
    np.testing.assert_array_equal(lo, [2, HORIZON - 2, -np.inf, -np.inf])
    np.testing.assert_array_equal(hi, [2, np.inf, -HORIZON + 4, np.inf])
    assert quantiles(np.array([3., np.inf]))["upper"] is None


def load_case(depth, reference_dir):
    model_path = ROOT / f"results/transformer/eval/lsa_a0_d128/d128_a0_L{depth}_step100000.npz"
    bayes_path = reference_dir / f"bayes_d128_L{depth}.npz"
    config_path = ROOT / "results/transformer/runs/lsa_a0_d128/config.json"
    cfg = json.loads(config_path.read_text())
    for key, expected in dict(d=128, alpha=[0., 0.], ls=[0, 8], n_ctx=1024, seed=0).items():
        if cfg[key] != expected:
            raise ValueError(f"Unexpected transformer prior/configuration: {key}")
    with np.load(model_path, allow_pickle=False) as model, np.load(bayes_path, allow_pickle=False) as bayes:
        x = model["tokens"]
        bc = json.loads(str(bayes["config_json"]))
        if x.shape != (2048, 1025) or not np.array_equal(x, bayes["tokens"]):
            raise ValueError("The two predictors must score exactly the same complete token arrays")
        if any(model[key].item() != value for key, value in dict(d=128, alpha=0., ls=depth, step=100000).items()):
            raise ValueError("Unexpected source setting or transformer checkpoint")
        expected = dict(schema="shared_depth_v1", kind="uniform", d=128, depths=list(range(9)), prior="uniform",
                        n_seq=2048, n_tokens=1025, tokens_sha256=token_hash(x))
        if any(bc.get(key) != value for key, value in expected.items()):
            raise ValueError("Reference is not the corrected shared-depth prior on the matched cohort")
        for name in ("shared_depth.py", "lsa_seq.py", "phi_source.py", "refs_shared_depth.py"):
            if bc["code_sha256"][name] != sha(ROOT / "code/transformer" / name):
                raise ValueError("Reference producer code differs from recorded version")
        losses = np.stack([-model["logp_true"].astype(float) / np.log(2), bayes["loss_bits"].astype(float)], axis=1)
        if losses.shape != (2048, 2, HORIZON) or not np.isfinite(losses).all() or np.any(losses < -1e-10):
            raise ValueError("Incomplete/nonfinite/negative observed-token losses")
    wide_path = reference_dir / f"check_d128_L{depth}_wide.npz"
    with np.load(wide_path, allow_pickle=False) as wide:
        wc = json.loads(str(wide["config_json"]))
        if not np.array_equal(wide["tokens"], x[:4]):
            raise ValueError("Wide-bound numerical check uses different sequences")
        for key in ("schema", "kind", "d", "depths", "prior", "grid_step", "code_sha256"):
            if wc[key] != bc[key]:
                raise ValueError("Wide-bound check differs in another numerical/source setting")
        if (bc["u_lo"], bc["u_hi"], wc["u_lo"], wc["u_hi"]) != (-30., 30., -36., 36.):
            raise ValueError("Expected production bounds[-30,30] and wider bounds[-36,36]")
        error = wide["loss_bits"] - losses[:4, 1]
        probs, positions = wide["probs"], wide["positions"]
        if probs.shape != (4, len(positions), 128) or not np.isfinite(probs).all() or np.any(probs <= 0):
            raise ValueError("Invalid wider-bound probabilities")
        actual = np.take_along_axis(probs, x[:4, positions, None], axis=2)[:, :, 0]
        audit = dict(sequences_checked=4, wider_integration_bounds=[-36, 36],
                     maximum_loss_difference_bits=float(np.max(np.abs(error))),
                     maximum_cumulative_difference_bits=float(np.max(np.abs(error.cumsum(1)))),
                     maximum_probability_normalization_error=float(np.max(np.abs(probs.sum(2) - 1))),
                     maximum_saved_probability_loss_error_bits=float(np.max(np.abs(-np.log2(actual) - wide["loss_bits"][:, positions - 1]))),
                     scope="First4 sequences at each source depth; bound expansion at unchanged grid spacing, not all2048 sequences")
        if max(audit[k] for k in ("maximum_loss_difference_bits", "maximum_cumulative_difference_bits", "maximum_saved_probability_loss_error_bits")) > 1e-7 or audit["maximum_probability_normalization_error"] > 1e-7:
            raise ValueError(f"Wider integration check failed: {audit}")
        audit["passed"] = True
    provenance = dict(
        transformer=dict(path=relative(model_path), sha256=sha(model_path)),
        bayes=dict(path=relative(bayes_path), sha256=sha(bayes_path), configuration=bc),
        transformer_config=dict(path=relative(config_path), sha256=sha(config_path)),
        wider_integration_check=dict(path=relative(wide_path), sha256=sha(wide_path), result=audit),
        tokens_sha256=token_hash(x),
    )
    return losses, provenance


def plot_figure(curves, groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.65))
    colors = {"transformer": "#2875B5", "bayes": "#D57629"}
    for ax, group in zip(axes, groups):
        depth = group["source_depth"]
        xmax = {2: 500, 3: 128, 4: 64}[depth]
        t = curves[f"L{depth}_time"]
        for name in PREDICTORS:
            m, se = (curves[f"L{depth}_{name}_{suffix}"] for suffix in ("cumulative_mean", "cumulative_se"))
            half = student_t.ppf(.975, group["n_sequences"] - 1) * se
            ax.plot(t, m, color=colors[name], lw=1.8, label="Transformer" if name == "transformer" else "Informed Bayes")
            ax.fill_between(t, m - half, m + half, color=colors[name], alpha=.15, lw=0)
            cross = group["crossings"][name]["time"]
            if cross is not None:
                ax.plot(cross, m[cross - 1], "o", color=colors[name], ms=4)
                ax.annotate(str(cross), (cross, m[cross - 1]), xytext=(4, -15 if name == "transformer" else 8),
                            textcoords="offset points", color=colors[name], fontsize=8)
        ax.axhline(0, color=".35", ls=":", lw=.9)
        ax.set_xlim(1, xmax)
        retained = t <= xmax
        low = min(np.min(curves[f"L{depth}_{name}_cumulative_mean"][retained] - 2 * curves[f"L{depth}_{name}_cumulative_se"][retained]) for name in PREDICTORS)
        high = max(np.max(curves[f"L{depth}_{name}_cumulative_mean"][retained] + 2 * curves[f"L{depth}_{name}_cumulative_se"][retained]) for name in PREDICTORS)
        padding = max(.01, .08 * (high - low))
        ax.set_ylim(low - padding, high + padding)
        ax.set_title(f"128 symbols; source depth L = {depth}")
        ax.set_xlabel("Observed tokens n")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Cumulative saving over uniform (bits)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "output/copying_parameter_screen_2026-09-17")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/copying_parameter_screen_2026-09-17")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260917128)
    args = parser.parse_args()
    if args.bootstrap < 100 or args.seed < 0:
        parser.error("At least100 bootstrap draws and a nonnegative seed are required")
    validate_definitions()
    args.out.mkdir(parents=True, exist_ok=True)
    curves, draws, groups, inputs, table = {}, {}, [], [], []
    for depth in (2, 3, 4):
        losses, provenance = load_case(depth, args.reference_dir)
        gain = 7. - losses
        cumulative = gain.cumsum(axis=2)
        mean = cumulative.mean(0)
        se = cumulative.std(0, ddof=1) / np.sqrt(len(losses))
        points = last_crossing(mean)
        boot = np.empty((args.bootstrap, 2))
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, depth]))
        for start in range(0, args.bootstrap, 100):
            count = min(100, args.bootstrap - start)
            weights = rng.multinomial(len(losses), np.full(len(losses), 1 / len(losses)), size=count) / len(losses)
            for j in range(2):
                boot[start:start + count, j] = last_crossing(weights @ cumulative[:, j])
        pair_summary, pair_lower, pair_upper = difference_summary(points, boot)
        group = dict(source_depth=depth, n_sequences=len(losses),
                     crossings={name: crossing_summary(mean[j], boot[:, j]) for j, name in enumerate(PREDICTORS)},
                     paired_crossing_difference=pair_summary, selected_times={}, cumulative_minima={})
        for j, name in enumerate(PREDICTORS):
            minimum = int(mean[j].argmin())
            point = points[j]
            lo = max(0, int(point) - 17) if np.isfinite(point) else HORIZON - 32
            hi = min(HORIZON, lo + 32)
            local = gain[:, j, lo:hi].mean(1)
            group["cumulative_minima"][name] = dict(time=minimum + 1, mean_bits=float(mean[j, minimum]), se_bits=float(se[j, minimum]))
            group["crossings"][name]["local_slope"] = dict(first_time=lo + 1, last_time=hi,
                                                          mean_bits_per_token=float(local.mean()),
                                                          se_bits_per_token=float(local.std(ddof=1) / np.sqrt(len(local))))
        for time in (16, 32, 64, 128, 256, 500, 512, 1024):
            group["selected_times"][str(time)] = {
                name: dict(cumulative_saving_bits=float(mean[j, time - 1]), se_bits=float(se[j, time - 1]))
                for j, name in enumerate(PREDICTORS)}
        curves[f"L{depth}_time"] = np.arange(1, HORIZON + 1)
        for j, name in enumerate(PREDICTORS):
            curves[f"L{depth}_{name}_cumulative_mean"] = mean[j]
            curves[f"L{depth}_{name}_cumulative_se"] = se[j]
            curves[f"L{depth}_{name}_instantaneous_mean"] = gain[:, j].mean(0)
            curves[f"L{depth}_{name}_instantaneous_se"] = gain[:, j].std(0, ddof=1) / np.sqrt(len(losses))
            for time in range(1, HORIZON + 1):
                table.append(dict(source_depth=depth, n_sequences=len(losses), time=time, predictor=name,
                                  cumulative_saving_bits=mean[j, time - 1], cumulative_se_bits=se[j, time - 1]))
        paired = cumulative[:, 0] - cumulative[:, 1]
        curves[f"L{depth}_transformer_minus_bayes_gain_mean"] = paired.mean(0)
        curves[f"L{depth}_transformer_minus_bayes_gain_se"] = paired.std(0, ddof=1) / np.sqrt(len(losses))
        draws[f"L{depth}_crossings_transformer_bayes"] = boot
        draws[f"L{depth}_paired_difference_lower"] = pair_lower
        draws[f"L{depth}_paired_difference_upper"] = pair_upper
        provenance["source_depth"] = depth
        inputs.append(provenance)
        groups.append(group)
    np.savez_compressed(args.out / "curves.npz", **curves)
    np.savez_compressed(args.out / "bootstrap_crossings.npz", **draws)
    with (args.out / "curves.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    plot_figure(curves, groups, args.out / "cumulative_payoff.png")
    summary = dict(
        schema="copying-parameter-screen-v1", d=128, alpha=0., n_tokens=1025, horizon=HORIZON,
        source_depths=[2, 3, 4], supplied_first_token=True,
        estimator="Observed next-token log loss in bits; uniform loss7 minus predictor loss; average cumulative losses before taking a crossing",
        bayes_prior="One shared source depth drawn uniformly0..8; independent symmetric LSA rows conditional on that depth",
        crossing="First time after the last nonpositive cumulative mean, required positive through1024 only; null if right-censored",
        numerical_zero_tolerance=ZERO_TOLERANCE,
        uncertainty=dict(pointwise_bands="95% Student intervals across independent whole sequences",
                         bootstrap=args.bootstrap, seed=args.seed, seed_rule="SeedSequence([seed,source_depth])",
                         method="Multinomial whole-sequence resampling, identical weights for both predictors; inverted-CDF quantiles",
                         censoring="Infinite crossings retained; paired differences bounded when one or both crossings are censored"),
        cautions=["No copying circuit intervention is available for this checkpoint; this is intact prediction versus known uniform.",
                  "These are realized next-token losses; the earlier d64 intervention used lower-variance conditional expected next-token losses.",
                  "Parameters were explored after inspecting d64 results; crossing intervals are exploratory and not corrected for parameter selection.",
                  "A final positive run within1024 does not establish positivity for all larger context lengths.",
                  "The historic row-local-depth references are not used; these references integrate one shared depth for the entire sequence."],
        validation=dict(token_identity=True, prior_identity=True, complete_loss_arrays=True,
                        synthetic_crossing_and_censoring_checks=True,
                        wider_integration_checks_passed=True,
                        scope="Input identities and analysis checked; expanded integration bounds checked on first4 sequences per source depth only"),
        groups=groups, inputs=inputs,
        script=dict(path=relative(Path(__file__)), sha256=sha(__file__)),
        outputs={name: sha(args.out / name) for name in ("curves.npz", "bootstrap_crossings.npz", "curves.tsv", "cumulative_payoff.png")},
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for group in groups:
        print(json.dumps(dict(source_depth=group["source_depth"], crossings=group["crossings"], paired=group["paired_crossing_difference"])))


if __name__ == "__main__":
    main()
