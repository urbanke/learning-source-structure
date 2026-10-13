"""Replot saved adaptive-unigram versus independent-copier losses.

The data use the general shared-base LSA source, with d=64, alpha=1 and
source depth 2 or 3. These archived LSA coding rules are neither the informed
shared-base Bayesian predictor nor transformer interventions. No prediction,
integration, source generation or model training is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/transformer/general_lsa_copying_2026-09-17/archived"
TOLERANCE_BITS = 1e-12


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def last_crossing(curves):
    """First time of final positive run, or infinity if no such observed run."""
    curves = np.atleast_2d(curves)
    positive = curves > TOLERANCE_BITS
    last_nonpositive = np.where(~positive, np.arange(curves.shape[1]), -1).max(axis=1)
    return np.where(positive[:, -1], last_nonpositive + 2, np.inf)


def json_time(value):
    return int(value) if np.isfinite(value) else None


def load_depth(depth):
    path = ROOT / f"results/transformer/refs2048/d64/d64_a1_L{depth}.npz"
    with np.load(path, allow_pickle=False) as saved:
        tokens = saved["tokens"]
        if tokens.shape != (2048, 1025) or tokens.dtype.kind not in "iu":
            raise ValueError("Expected 2048 complete length-1025 sequences")
        if any(saved[key].item() != value for key, value in
               dict(d=64, alpha=1.0, ls=depth).items()):
            raise ValueError("Unexpected source parameters")
        token_bytes = np.ascontiguousarray(tokens, dtype=np.int16).tobytes()
        if hashlib.md5(token_bytes).hexdigest() != str(saved["md5"]):
            raise ValueError("Saved token identity does not match its MD5")
        unigram, copier = (saved[key].astype(np.float64) for key in ("lF", "lC"))
        if (unigram.shape != (2048, 1024) or copier.shape != unigram.shape
                or not np.isfinite(unigram).all() or not np.isfinite(copier).all()
                or np.min(unigram) < -1e-8 or np.min(copier) < -1e-8):
            raise ValueError("Incomplete or invalid saved losses")
        provenance = dict(path=str(path.relative_to(ROOT)), sha256=sha256(path),
                          tokens_md5=str(saved["md5"]),
                          tokens_sha256=hashlib.sha256(token_bytes).hexdigest())
    return np.cumsum(unigram - copier, axis=1), provenance


def make_summary(depth, cumulative, provenance, n_bootstrap, seed):
    n, horizon = cumulative.shape
    mean = cumulative.mean(axis=0)
    se = cumulative.std(axis=0, ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(np.random.SeedSequence([seed, depth]))
    crossing_draws = np.empty(n_bootstrap)
    for left in range(0, n_bootstrap, 64):
        right = min(left + 64, n_bootstrap)
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=right - left)
        crossing_draws[left:right] = last_crossing(weights @ cumulative / n)
    crossing = last_crossing(mean)[0]
    quantiles = np.quantile(crossing_draws, [.025, .5, .975], method="inverted_cdf")
    minimum = int(np.argmin(mean))
    result = dict(
        source_depth=depth, n_sequences=n, horizon=horizon, input=provenance,
        cumulative_crossing=dict(
            time=json_time(crossing),
            bootstrap_percentiles={k: json_time(v) for k, v in
                                   zip(("lower", "median", "upper"), quantiles)},
            censored_bootstrap_count=int(np.isinf(crossing_draws).sum()),
            positive_from_start_bootstrap_count=int((crossing_draws == 1).sum())),
        minimum_mean_gain=dict(time=minimum + 1, bits=float(mean[minimum]),
                               standard_error_bits=float(se[minimum])),
        selected_times=[dict(time=t, mean_bits=float(mean[t - 1]),
                             standard_error_bits=float(se[t - 1]))
                        for t in (64, 128, 256, 512, 1024)],
    )
    return result, dict(mean=mean, se=se, bootstrap_crossings=crossing_draws)


def plot(groups, arrays, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    times = arrays["time"]
    for ax, group in zip(axes, groups):
        depth = group["source_depth"]
        mean, se = (arrays[f"L{depth}_{key}"] for key in ("mean", "se"))
        halfwidth = student_t.ppf(.975, group["n_sequences"] - 1) * se
        xmax = 650 if depth == 2 else 200
        selected = times <= xmax
        ax.axhline(0, color=".35", ls=":", lw=1)
        ax.fill_between(times, mean - halfwidth, mean + halfwidth,
                        color="#2875B5", alpha=.17, lw=0)
        ax.plot(times, mean, color="#2875B5", lw=1.8)
        crossing = group["cumulative_crossing"]["time"]
        if crossing is not None:
            ax.scatter(crossing, mean[crossing - 1], color="#2875B5", s=20, zorder=5)
            ax.annotate(f"Crossing: {crossing}",
                        (crossing, mean[crossing - 1]), xytext=(5, 9),
                        textcoords="offset points", color="#175785", fontsize=9)
        low = float(np.min((mean - halfwidth)[selected]))
        high = float(np.max((mean + halfwidth)[selected]))
        margin = .13 * (high - low)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlim(1, xmax)
        ax.set_title(f"Source depth L = {depth}")
        ax.set_xlabel("Observed tokens n")
        ax.grid(alpha=.14)
        ax.text(.025, .94, "Above zero: copier better", transform=ax.transAxes,
                fontsize=8.5, va="top")
        ax.text(.025, .06, "Below zero: unigram better", transform=ax.transAxes,
                fontsize=8.5, va="bottom")
    axes[0].set_ylabel("Cumulative unigram loss − copier loss (bits)")
    fig.suptitle("General shared-base LSA source: 64 symbols, α = 1", y=.995, fontsize=12)
    fig.text(.5, .013, "Archived LSA predictors; 2,048 sequences per panel; bands: pointwise 95% intervals.\n"
             "These are not informed-Bayes or transformer curves. Different horizontal ranges show the crossings.",
             ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(out / "cumulative_payoff.png", dpi=180)
    plt.close(fig)


def check(out):
    summary = json.loads((out / "summary.json").read_text())
    if summary["script_sha256"] != sha256(__file__):
        raise ValueError("Summary belongs to a different script version")
    with np.load(out / "curves.npz", allow_pickle=False) as arrays:
        for group in summary["groups"]:
            depth = group["source_depth"]
            values, provenance = load_depth(depth)
            if provenance != group["input"]:
                raise ValueError("Archived input identity changed")
            np.testing.assert_allclose(arrays[f"L{depth}_mean"], values.mean(0), atol=1e-12, rtol=0)
            np.testing.assert_allclose(arrays[f"L{depth}_se"], values.std(0, ddof=1) / np.sqrt(len(values)),
                                       atol=1e-12, rtol=0)
            if json_time(last_crossing(values.mean(0))[0]) != group["cumulative_crossing"]["time"]:
                raise ValueError("Stored crossing differs from its definition")
            draws = arrays[f"L{depth}_bootstrap_crossings"]
            quantiles = np.quantile(draws, [.025, .5, .975], method="inverted_cdf")
            actual = {k: json_time(v) for k, v in zip(("lower", "median", "upper"), quantiles)}
            if actual != group["cumulative_crossing"]["bootstrap_percentiles"]:
                raise ValueError("Stored bootstrap intervals differ from retained draws")
    for name, expected in summary["output_sha256"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"Changed output: {name}")
    print(json.dumps(dict(status="verified", groups=len(summary["groups"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026091764)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check(args.out)
        return
    if args.bootstrap < 100 or args.seed < 0:
        parser.error("At least 100 bootstrap replicates and a nonnegative seed are required")
    groups, arrays = [], dict(time=np.arange(1, 1025))
    for depth in (2, 3):
        cumulative, provenance = load_depth(depth)
        group, curves = make_summary(depth, cumulative, provenance, args.bootstrap, args.seed)
        groups.append(group)
        arrays.update({f"L{depth}_{key}": value for key, value in curves.items()})
        print(json.dumps(dict(depth=depth, crossing=group["cumulative_crossing"])), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "curves.npz", **arrays)
    plot(groups, arrays, args.out)
    summary = dict(
        schema="general-lsa-archived-payoff-v1", script_sha256=sha256(__file__),
        reproduction="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/general_lsa_archived_payoff.py",
        source=dict(d=64, alpha=1.0, source_depths=[2, 3],
                    shared_base="q_b proportional to b^(-alpha), with unknown random token relabeling",
                    n_tokens_per_sequence=1025, n_sequences_per_condition=2048),
        predictors=dict(
            unigram="One pooled LSA count table, with uniform coding-depth mixture on 1..20",
            copier="Independent LSA count tables per preceding token, each mixing coding depths 1..20",
            interpretation="Archived adaptive LSA statistical comparators. Neither is the informed shared-base Bayes predictor; no transformer or ablation is measured."),
        metric="Sample mean of sum_{t=1}^n [loss_unigram(X_{t+1}|X^t) - loss_copier(X_{t+1}|X^t)], in bits",
        scoring="X1 is observed and updates the unigram; scored tokens are X2 through X1025",
        sign="Positive means the independent copier has smaller cumulative loss",
        crossing="First integer n of the final strictly positive run through the observed horizon 1024; not a claim about all later n",
        zero_tolerance_bits=TOLERANCE_BITS,
        uncertainty=dict(curves="Pointwise Student-t 95% intervals across independent sequences; not simultaneous bands",
                         crossings="Exploratory whole-sequence paired bootstrap percentile intervals; no smoothing",
                         bootstrap_replicates=args.bootstrap, seed=args.seed,
                         sampling="Multinomial resampling of 2048 paired whole-sequence cumulative loss curves",
                         censoring="Infinite crossing retained when a bootstrap mean is not positive at the horizon; JSON null denotes beyond the observed horizon"),
        numerical_scope="Reanalysis of retained historical loss arrays; no new numerical-integration certification",
        precision="Original per-token losses stored as float32; converted to float64 before subtraction, accumulation and summary statistics",
        groups=groups,
        output_sha256={name: sha256(args.out / name) for name in ("curves.npz", "cumulative_payoff.png")},
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    check(args.out)


if __name__ == "__main__":
    main()
