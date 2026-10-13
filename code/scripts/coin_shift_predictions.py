"""Frozen coin transformers under a new bias and a change of bias.

The Bayesian comparator retains the training prior and the entire prefix.
All population Bayesian curves are finite binomial convolutions, not fitted
to network predictions. Raw tokens and full predictive vectors stay in output/.
Run --smoke for 32 sequences per condition in a separate output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))

SCHEMA = "coin-shift-predictions-v1"
PRIORS = ("twopoint", "uniform")
CONDITIONS = (
    dict(name="mismatch_p04", p_old=.4, p_new=.4, switch=257),
    dict(name="switch64", p_old=.2, p_new=.8, switch=64),
    dict(name="switch128", p_old=.2, p_new=.8, switch=128),
)
TIMES = np.arange(1, 257)


def require(passed, message):
    if not passed:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def binding(path):
    from coin_bayes_audit import sha_file

    return dict(path=str(Path(path).resolve()), sha256=sha_file(path))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(values):
    return dict(mean=values.mean(axis=0).tolist(),
                se=(values.std(axis=0, ddof=1) / math.sqrt(len(values))).tolist())


def kl(p, q):
    from scipy.special import xlogy

    require(np.isfinite(q).all() and np.all((q > 0) & (q < 1)), "Noninterior predictive probability")
    result = (xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q))) / math.log(2)
    require(np.isfinite(result).all() and result.min() >= -2e-14, "Invalid binary KL")
    return result


def true_bias(condition, positions):
    return np.where(np.asarray(positions) < condition["switch"], condition["p_old"], condition["p_new"])


def exact_curve(condition, prior, times=TIMES):
    from scipy.special import logsumexp
    from scipy.stats import binom
    from coin_bayes_audit import posterior_prediction

    means, variances, predictions, ranges = [], [], [], []
    max_norm, posterior_error = 0., 0.
    for observed in times:
        old = min(int(observed), condition["switch"])
        new = int(observed) - old
        old_law = binom.pmf(np.arange(old + 1), old, condition["p_old"])
        new_law = binom.pmf(np.arange(new + 1), new, condition["p_new"])
        weights = np.convolve(old_law, new_law)
        max_norm = max(max_norm, abs(float(weights.sum()) - 1))
        require(max_norm < 2e-13, "Binomial convolution does not normalize")
        ones = np.arange(int(observed) + 1)
        q = posterior_prediction(ones, observed, prior)
        if prior == "twopoint":
            theta = np.array([.2, .8])
            log_weights = ones[:, None] * np.log(theta) + (observed - ones[:, None]) * np.log1p(-theta)
            direct = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True)) @ theta
        else:
            direct = (ones + 1.) / (observed + 2.)
        posterior_error = max(posterior_error, float(np.max(np.abs(q - direct))))
        loss = kl(float(true_bias(condition, observed)), q)
        mean = float(weights @ loss)
        means.append(mean)
        variances.append(float(weights @ (loss - mean) ** 2))
        predictions.append(float(weights @ q))
        ranges.append(float(loss.max() - loss.min()))
    require(posterior_error < 1e-13, "Independent posterior calculation differs")
    return dict(mean=means, variance=variances, mean_prediction=predictions, risk_range=ranges,
                normalization_max_error=max_norm, posterior_formula_max_error=posterior_error)


def independent_formula_checks():
    """Enumerate histories, independent of the convolution sufficient statistic."""
    error = 0.
    for prior in PRIORS:
        for condition in (CONDITIONS[0], dict(name="tiny_switch", p_old=.2, p_new=.8, switch=3)):
            exact = exact_curve(condition, prior, np.arange(1, 9))
            for observed in range(1, 9):
                total = 0.
                for history in itertools.product((0, 1), repeat=observed):
                    history = np.asarray(history)
                    biases = true_bias(condition, np.arange(observed))
                    probability = float(np.prod(np.where(history, biases, 1 - biases)))
                    if prior == "twopoint":
                        hypotheses = np.array([.2, .8])
                        likelihood = np.prod(np.where(history[:, None], hypotheses, 1 - hypotheses), axis=0)
                        q = float(likelihood @ hypotheses / likelihood.sum())
                    else:
                        q = float((history.sum() + 1) / (observed + 2))
                    total += probability * kl(float(true_bias(condition, observed)), q)
                error = max(error, abs(float(total) - exact["mean"][observed - 1]))
    require(error < 2e-13, "Enumeration does not match exact Bayesian curve")
    for tau in (64, 128):
        condition = dict(p_old=.2, p_new=.8, switch=tau)
        at_balance = exact_curve(condition, "twopoint", [2 * tau])["mean_prediction"][0]
        require(abs(at_balance - .5) < 2e-13, "Symmetric change does not balance mean prediction")
        require(float(true_bias(condition, tau - 1)) == .2 and float(true_bias(condition, tau)) == .8,
                "Wrong change boundary")
    return dict(passed=True, enumerated_prefixes=list(range(1, 9)),
                enumeration_max_error_bits=error, balanced_mean_prediction_checked=True)


def load_checkpoint(prior, threads):
    import torch
    from model import GPT

    torch.set_num_threads(threads)
    run = ROOT / "results/transformer/runs" / ("bin_" + prior)
    config_path, checkpoint_path = run / "config.json", run / "model_20000.pt"
    config = json.loads(config_path.read_text())
    require(config["family"] == "binary:" + prior and config["seed"] == 0
            and config["steps"] == 20000, "Wrong training prior, seed or step")
    require(config["model"]["n_ctx"] == 256 and config["model"]["vocab"] == 2, "Wrong checkpoint context/alphabet")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["step"] == 20000 and canonical(checkpoint["cfg"]) == canonical(config), "Checkpoint/config mismatch")
    model = GPT(**config["model"])
    model.load_state_dict(checkpoint["model"])
    require(model.n_params() == 116736, "Unexpected model size")
    return model.eval(), dict(checkpoint=binding(checkpoint_path), config=binding(config_path),
                             model_code=binding(ROOT / "code/transformer/model.py"), parameters=model.n_params())


def evaluate(model, tokens, batch):
    import torch
    from eval_shared_base_models import check_model_alignment

    chunks, maximum_normalization, alignment = [], 0., None
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(tokens), batch):
            inputs = torch.from_numpy(tokens[start:start + batch, :-1].astype(np.int64))
            logits = model(inputs)
            if alignment is None:
                alignment = check_model_alignment(model, inputs, logits)
            # The saved model has float32 weights. Evaluate without autocast;
            # convert logits to float64 before log-softmax and all risk work.
            logp = torch.log_softmax(logits.double(), dim=-1).numpy()
            error = float(np.max(np.abs(np.exp(logp).sum(axis=-1) - 1)))
            maximum_normalization = max(maximum_normalization, error)
            require(error < 2e-14 and np.isfinite(logp).all(), "Invalid raw model probabilities")
            chunks.append(logp)
            if (start + batch) % 1024 == 0:
                print(json.dumps(dict(event="progress", sequences=min(start + batch, len(tokens)),
                                      total=len(tokens), elapsed_seconds=time.monotonic() - started)), flush=True)
    return np.concatenate(chunks), dict(alignment=alignment, raw_normalization_max_error=maximum_normalization,
                                       seconds=time.monotonic() - started,
                                       torch_version=torch.__version__, numpy_version=np.__version__)


def summarize(condition, prior, tokens, logp, meta):
    from scipy.special import xlogy
    from coin_bayes_audit import posterior_prediction

    require(logp.shape == (len(tokens), 256, 2), "Invalid predictive-vector shape")
    counts = np.cumsum(tokens[:, :-1], axis=1)
    q_b = posterior_prediction(counts, TIMES[None, :], prior)
    probs = np.exp(logp)
    require(np.max(np.abs(probs.sum(axis=-1) - 1)) < 2e-14, "Saved raw normalization failed")
    p = true_bias(condition, TIMES)[None, :]
    bayes = kl(p, q_b)
    transformer = (xlogy(p, p) + xlogy(1-p, 1-p) - p * logp[:, :, 1] - (1-p) * logp[:, :, 0]) / math.log(2)
    require(transformer.min() >= -2e-14 and np.isfinite(transformer).all(), "Invalid transformer KL")
    exact = exact_curve(condition, prior)
    error = np.abs(bayes.mean(axis=0) - np.asarray(exact["mean"]))
    population_se = np.sqrt(np.asarray(exact["variance"]) / len(tokens))
    # A fixed simultaneous Bernstein bound also covers very small smoke
    # cohorts and rare ambiguous counts, where a normal-tail rule is invalid.
    union_log = math.log(2 * 256 * 6 / 1e-6)
    tolerance = np.sqrt(2 * union_log) * population_se + 2 * union_log * np.asarray(exact["risk_range"]) / (3 * len(tokens))
    require(np.all(error < tolerance + 2e-13), "Sampled Bayes fails finite-sum check")
    direct_gap = (p * (np.log(q_b) - logp[:, :, 1]) + (1-p) * (np.log1p(-q_b) - logp[:, :, 0])) / math.log(2)
    gap_error = float(np.max(np.abs(direct_gap - (transformer - bayes))))
    require(gap_error < 2e-14, "Paired log-loss gap identity failed")
    crossing = None
    if condition["switch"] <= 256:
        mean = probs[:, :, 1].mean(axis=0)
        hits = np.flatnonzero((TIMES >= condition["switch"]) & (mean >= .5))
        crossing = dict(first_observed_tokens=int(TIMES[hits[0]]) if len(hits) else None,
                        new_observations=int(TIMES[hits[0]] - condition["switch"]) if len(hits) else None,
                        rule="First tested post-change sample mean head forecast >= 0.5; descriptive, not a recovery time or confidence interval.")
    return dict(condition=condition, prior=prior, n_sequences=len(tokens), times=TIMES.tolist(),
                transformer=stats(transformer), sampled_bayes=stats(bayes),
                transformer_mean_prediction=stats(probs[:, :, 1]), sampled_bayes_mean_prediction=stats(q_b),
                transformer_mean_forecast_crossing=crossing,
                transformer_minus_bayes=stats(transformer - bayes), exact_bayes=exact,
                validation=dict(passed=True, sampled_bayes_max_z=float(np.max(error / np.maximum(population_se, 2e-13))),
                                sample_check="Simultaneous Bernstein bound, 1536 comparisons, total tail bound 1e-6",
                                maximum_fraction_of_bernstein_bound=float(np.max(error / np.maximum(tolerance, 2e-13))),
                                paired_gap_max_error_bits=gap_error), provenance=meta), (transformer, bayes)


def plot(rows, out):
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
    colors = dict(twopoint="#2468A0", uniform="#BF611C")
    dark = "#333333"

    def context_axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xlim(.9, 272)
        ax.set_xticks([1, 4, 16, 64, 128, 256],
                      ["1", "4", "16", "64", "128", "256"])
        ax.minorticks_off()
        ax.set_xlabel("Observed tokens, $t$")
        ax.grid(axis="y", color="#D8DEE5", linewidth=.45)
        ax.tick_params(length=2.5, pad=2)

    def point_indices(switch=None):
        selected = {1, 2, 4, 8, 16, 32, 64, 128, 256}
        if switch is not None:
            selected.update({switch, min(256, switch + 1), min(256, switch + 4),
                             min(256, switch + 16), min(256, 2 * switch)})
        values = np.asarray(sorted(selected), dtype=int)
        return values - 1

    style_handles = [Line2D([0], [0], color=dark, lw=1.25),
                     Line2D([0], [0], color=dark, ls="none", marker="o",
                            mfc="white", ms=4)]
    prior_handles = [Line2D([0], [0], color=colors["twopoint"], lw=1.5),
                     Line2D([0], [0], color=colors["uniform"], lw=1.5)]
    for switch in (128, 64):
        fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.75))
        for panel, (ax, name, title) in enumerate(zip(
                axes, ("mismatch_p04", f"switch{switch}"),
                (r"Mismatch: test bias $p=0.4$",
                 rf"Change after {switch} tokens"))):
            for prior in PRIORS:
                row = next(r for r in rows if r["prior"] == prior and r["condition"]["name"] == name)
                mu, se = np.array(row["transformer"]["mean"]), np.array(row["transformer"]["se"])
                color = colors[prior]
                ax.plot(TIMES, mu, color=color, lw=1.25)
                ax.fill_between(TIMES, mu - 1.96 * se, mu + 1.96 * se,
                                color=color, alpha=.12, lw=0)
                indices = point_indices(None if name == "mismatch_p04" else switch)
                bayes = np.asarray(row["exact_bayes"]["mean"])
                ax.plot(TIMES[indices], bayes[indices], color=color, ls="none",
                        marker="o", ms=3.2, mfc="white", mew=.75, zorder=5)
            ax.set_title(f"({chr(97 + panel)}) {title}", loc="left", pad=5)
            ax.set_ylim(bottom=0)
            context_axis(ax)
            if name != "mismatch_p04":
                ax.axvline(switch, color=".45", lw=.75, ls=":")
        axes[0].set_ylabel("Expected next-token regret (bits)")
        handles = prior_handles + style_handles
        fig.legend(handles, [r"Prior $\{0.2,0.8\}$", r"Prior uniform on $[0,1]$",
                             "Transformer", "Informed Bayes (exact)"],
                   loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5, -.005))
        fig.subplots_adjust(left=.095, right=.972, bottom=.29, top=.91, wspace=.27)
        stem = "coin_shift_main" if switch == 128 else "coin_shift_tau64"
        fig.savefig(out / (stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out / (stem + ".png"), dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.75), sharey=True)
    for panel, (ax, switch) in enumerate(zip(axes, (64, 128)), start=2):
        for prior in PRIORS:
            row = next(r for r in rows if r["prior"] == prior and r["condition"]["name"] == f"switch{switch}")
            mu, se = (np.asarray(row["transformer_mean_prediction"][key]) for key in ("mean", "se"))
            color = colors[prior]
            ax.plot(TIMES, mu, color=color, lw=1.25)
            ax.fill_between(TIMES, mu - 1.96 * se, mu + 1.96 * se,
                            color=color, alpha=.12, lw=0)
            indices = point_indices(switch)
            bayes = np.asarray(row["exact_bayes"]["mean_prediction"])
            ax.plot(TIMES[indices], bayes[indices], color=color, ls="none",
                    marker="o", ms=3.2, mfc="white", mew=.75, zorder=5)
        ax.axvline(switch, color=".45", lw=.75, ls=":")
        ax.axhline(.5, color=".45", lw=.75, ls=":")
        ax.set_title(rf"({chr(97 + panel)}) Change after {switch} tokens",
                     loc="left", pad=5)
        context_axis(ax)
        ax.set_ylim(.15, .85)
    axes[0].set_ylabel("Mean probability of heads")
    fig.legend(prior_handles + style_handles,
               [r"Prior $\{0.2,0.8\}$", r"Prior uniform on $[0,1]$",
                "Transformer", "Informed Bayes (exact)"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.005))
    fig.subplots_adjust(left=.095, right=.972, bottom=.29, top=.91, wspace=.2)
    fig.savefig(out / "coin_shift_forecasts.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "coin_shift_forecasts.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--raw", type=Path, default=ROOT / "output/coin_shift_2026-09-18")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/coin_shift_2026-09-18")
    args = parser.parse_args()
    n = 32 if args.smoke else 4096
    if args.smoke:
        args.raw = args.raw.parent / (args.raw.name + "_smoke")
        args.out = args.raw / "figures"
    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    checks = independent_formula_checks()
    rows, saved_risks = [], {}
    for prior in PRIORS:
        model, checkpoint = load_checkpoint(prior, args.threads)
        for case_index, condition in enumerate(CONDITIONS):
            rng = np.random.default_rng(np.random.SeedSequence([2026091827, case_index]))
            tokens = (rng.random((n, 257)) < true_bias(condition, np.arange(257))[None, :]).astype(np.int16)
            identity = dict(schema=SCHEMA, condition=condition, prior=prior, n_sequences=n,
                            source_seed=[2026091827, case_index], checkpoint=checkpoint,
                            token_sha256=hashlib.sha256(tokens.tobytes()).hexdigest())
            raw = args.raw / f"{condition['name']}_{prior}.npz"
            if raw.exists():
                with np.load(raw, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata_json"]))
                    require(meta["identity"] == identity, "Raw experiment identity changed")
                    require(np.array_equal(saved["tokens"], tokens), "Deterministic full token replay failed")
                    logp = saved["logp"].copy()
                require(hashlib.sha256(logp.tobytes()).hexdigest() == meta["logp_sha256"], "Saved predictions changed")
            else:
                require(not args.check_only, f"Missing raw record {raw}")
                print(json.dumps(dict(event="start", prior=prior, condition=condition["name"], n_sequences=n)), flush=True)
                logp, validation = evaluate(model, tokens, args.batch)
                meta = dict(identity=identity, evaluation=validation, evaluation_code=binding(Path(__file__)),
                            device="cpu", torch_threads=args.threads, batch=args.batch,
                            logp_sha256=hashlib.sha256(logp.tobytes()).hexdigest())
                with raw.open("xb") as stream:
                    np.savez_compressed(stream, tokens=tokens, logp=logp, metadata_json=canonical(meta))
            row, risks = summarize(condition, prior, tokens, logp, dict(raw=binding(raw), metadata=meta))
            rows.append(row)
            saved_risks[condition["name"] + "_" + prior + "_transformer"] = risks[0]
            saved_risks[condition["name"] + "_" + prior + "_bayes"] = risks[1]
            print(json.dumps(dict(event="complete", prior=prior, condition=condition["name"],
                                  seconds=meta["evaluation"]["seconds"], exact_bayes_max_z=row["validation"]["sampled_bayes_max_z"])), flush=True)
    summary = dict(schema=SCHEMA, complete=True, scientific_results=not args.smoke, n_sequences_per_condition=n,
                   code=binding(Path(__file__)), formula_checks=checks, conditions=list(CONDITIONS), priors=list(PRIORS),
                   timing="At t=1,...,256 observe tokens[:t] and predict tokens[t]. Change at tau first affects token[tau].",
                   uncertainty="Transformer pointwise 95% normal intervals across independent sequences; paired differences use the same sequences.",
                   exact_bayes="Independent binomial count convolution under the test source, with the training prior.", rows=rows)
    summary_path = args.raw / "summary.json"
    if args.check_only:
        old = json.loads(summary_path.read_text())
        require(old == summary, "Saved numerical summary differs from recomputation")
        print("All raw checksums, token replays, analytic curves and numerical summaries passed", flush=True)
        return
    save_json(summary_path, summary)
    np.savez_compressed(args.raw / "per_sequence_risks.npz", **saved_risks)
    plot(rows, args.out)
    for row in rows:
        print(json.dumps(dict(prior=row["prior"], condition=row["condition"]["name"],
                              endpoint_tf=row["transformer"]["mean"][-1], endpoint_exact_bayes=row["exact_bayes"]["mean"][-1],
                              endpoint_gap=row["transformer_minus_bayes"]["mean"][-1], endpoint_gap_se=row["transformer_minus_bayes"]["se"][-1])), flush=True)
    print("Saved " + str(summary_path), flush=True)


if __name__ == "__main__":
    main()
