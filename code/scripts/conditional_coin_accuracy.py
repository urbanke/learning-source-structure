"""Coin learning curves at fixed biases, with the training prior held unchanged.

Reuse the validated two-point evaluation, grouping its sequences by true bias.
For the continuous-prior model, evaluate the existing checkpoint on fresh
independent sequences at five declared biases. No training is performed.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/conditional_coin_accuracy.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/conditional_coin_accuracy.py --check-only

Raw tokens and all 256 selected-token log probabilities remain in output/.
Means and standard errors use independent sequences, never prediction times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import betaln, logsumexp
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from source import seed_of
from coin_bayes_audit import binary_kl, posterior_prediction, reconstruct_binary, sha_file

SCHEMA = "conditional-coin-accuracy-v1"
TIMES = 2 ** np.arange(9)
BIASES = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_RAW = ROOT / "output/conditional_coin_accuracy_2026-09-15"
DEFAULT_OUT = ROOT / "results/transformer/conditional_accuracy/coins"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha_file(path))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(values):
    return dict(mean=values.mean(axis=0).tolist(),
                se=(values.std(axis=0, ddof=1) / math.sqrt(len(values))).tolist())


def exact_fixed_risk(theta, prior):
    """Enumerate the binomial sufficient statistic; no posterior Monte Carlo."""
    means, variances, formula_errors = [], [], []
    for t in TIMES:
        k = np.arange(t + 1)
        predicted = posterior_prediction(k, t, prior)
        if prior == "uniform":
            independent = np.exp(betaln(k + 2, t - k + 1) - betaln(k + 1, t - k + 1))
        else:
            hypotheses = np.array([0.2, 0.8])
            logw = k[:, None] * np.log(hypotheses) + (t - k[:, None]) * np.log1p(-hypotheses)
            independent = np.exp(logw - logsumexp(logw, axis=1, keepdims=True)) @ hypotheses
        formula_errors.append(float(np.max(np.abs(predicted - independent))))
        weights = binom.pmf(k, t, theta)
        require(abs(weights.sum() - 1) < 2e-13, "Binomial weights do not normalize")
        risk = binary_kl(theta, predicted)
        require(risk.min() >= -2e-15, "Negative exact KL")
        risk = np.maximum(risk, 0)
        mean = float(weights @ risk)
        means.append(mean)
        variances.append(float(weights @ ((risk - mean) ** 2)))
    require(max(formula_errors) < 5e-13, "Independent Bayes posterior formula disagrees")
    return dict(mean=means, variance=variances,
                posterior_formula_max_error=max(formula_errors))


def summarize_case(prior, theta, tokens, logp, provenance):
    require(tokens.shape[1] == 257 and logp.shape == (len(tokens), 256), "Wrong sequence length/alignment")
    require(np.all((tokens == 0) | (tokens == 1)), "Non-binary tokens")
    p_tf = reconstruct_binary(logp[:, TIMES - 1], tokens[:, TIMES])
    counts = np.cumsum(tokens, axis=1)[:, TIMES - 1]
    p_bb = posterior_prediction(counts, TIMES[None, :], prior)
    tf, bb = binary_kl(theta, p_tf), binary_kl(theta, p_bb)
    require(min(tf.min(), bb.min()) >= -2e-15, "Negative sampled KL")
    tf, bb = np.maximum(tf, 0), np.maximum(bb, 0)
    gap = tf - bb
    direct_gap = theta * np.log2(p_bb / p_tf) + (1-theta) * np.log2((1-p_bb)/(1-p_tf))
    gap_error = float(np.max(np.abs(gap - direct_gap)))
    require(gap_error < 2e-14, "Paired loss-gap identity failed")
    exact = exact_fixed_risk(theta, prior)
    population_se = np.sqrt(np.array(exact["variance"]) / len(tokens))
    error = np.abs(bb.mean(0) - exact["mean"])
    # Population variances remain meaningful even if no rare ambiguous count
    # occurred in the finite cohort. Tiny numerical values use an absolute floor.
    tolerance = 6 * population_se + 2e-14
    require(np.all(error <= tolerance), "Sampled Bayesian risk fails exact finite-sum check")
    return dict(prior=prior, theta=theta, n_sequences=len(tokens), n_tokens=257,
                times=TIMES.tolist(), transformer=stats(tf), informed_bayes=stats(bb),
                transformer_minus_bayes=stats(gap), exact_bayes=exact,
                validation=dict(passed=True, paired_gap_identity_max_error_bits=gap_error,
                    exact_bayes_sample_max_z=float(np.max(error / np.maximum(population_se, 2e-14))),
                    exact_bayes_sample_rule="absolute error <= 6 population SE + 2e-14 bits"),
                provenance=provenance)


def reused_two_point():
    audit_path = ROOT / "results/transformer/coins_rechecked/summary.json"
    audit = json.loads(audit_path.read_text())
    require(audit["passed"], "Earlier coin audit did not pass")
    ref_path = ROOT / "results/transformer/refs_fam/binary_twopoint.npz"
    eval_path = ROOT / "results/transformer/eval/bin_twopoint/binary_twopoint_step20000.npz"
    for path, key in ((ref_path, "reference:binary_twopoint"),
                      (eval_path, "evaluation:bin_twopoint:binary_twopoint")):
        require(sha_file(path) == audit["provenance"]["inputs"][key]["sha256"], "Validated input changed")
    for key in ("checkpoint:bin_twopoint", "config:bin_twopoint"):
        item = audit["provenance"]["inputs"][key]
        require(sha_file(item["path"]) == item["sha256"], "Previously validated two-point checkpoint/config changed")
    with np.load(ref_path, allow_pickle=False) as ref, np.load(eval_path, allow_pickle=False) as saved:
        tokens, logp = saved["tokens"], saved["logp_true"]
        require(tokens.shape == (4096, 257) and np.array_equal(tokens, ref["tokens"]), "Reused cohort differs")
        require(str(saved["spec"]) == "binary:twopoint" and int(saved["step"]) == 20000, "Wrong reused checkpoint")
    rng = np.random.default_rng(seed_of("eval", "binary:twopoint", 256, 4096))
    theta = rng.choice(np.array([0.2, 0.8]), size=4096)
    replay = (rng.random((4096, 257)) < theta[:, None]).astype(np.int16)
    require(np.array_equal(tokens, replay), "Complete two-point sequence replay failed")
    provenance = dict(mode="group existing validated evaluation by recovered true bias",
                      reference=binding(ref_path), evaluation=binding(eval_path),
                      previous_validation=binding(audit_path), full_sequence_replay_passed=True)
    return [summarize_case("twopoint", bias, tokens[theta == bias], logp[theta == bias], provenance)
            for bias in (0.2, 0.8)]


def uniform_inputs(args):
    run = ROOT / "results/transformer/runs/bin_uniform"
    config_path, checkpoint_path = run / "config.json", run / "model_20000.pt"
    config = json.loads(config_path.read_text())
    require(config["family"] == "binary:uniform" and config["steps"] == 20000 and config["seed"] == 0,
            "Uniform checkpoint training prior/step/seed mismatch")
    require(config["model"]["vocab"] == 2 and config["model"]["n_ctx"] == 256, "Wrong model support")
    spec = dict(schema=SCHEMA, training_prior="uniform on [0,1]", test_biases=list(BIASES),
                n_sequences_per_bias=args.n_seq, n_tokens=257, master_seed=args.seed,
                checkpoint=binding(checkpoint_path), config=binding(config_path),
                evaluator=binding(ROOT / "code/transformer/eval_model.py"),
                model_code=binding(ROOT / "code/transformer/model.py"),
                seed_code=binding(ROOT / "code/transformer/source.py"))
    return config, checkpoint_path, spec


def load_model(config, checkpoint_path):
    import torch
    from model import GPT
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["step"] == 20000 and canonical(checkpoint["cfg"]) == canonical(config),
            "Checkpoint configuration differs from audited training run")
    model = GPT(**config["model"])
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def fixed_uniform(args):
    config, checkpoint_path, spec = uniform_inputs(args)
    args.raw.mkdir(parents=True, exist_ok=True)
    spec_path = args.raw / "protocol.json"
    if spec_path.exists():
        require(json.loads(spec_path.read_text()) == spec, "Existing raw protocol differs; use a new raw directory")
    elif not args.check_only:
        save_json(spec_path, spec)
    else:
        raise FileNotFoundError(spec_path)
    model, rows = None, []
    for theta in BIASES:
        seed = seed_of(SCHEMA, args.seed, theta)
        rng = np.random.default_rng(seed)
        expected = (rng.random((args.n_seq, 257)) < theta).astype(np.int16)
        path = args.raw / f"theta_{theta:.1f}.npz"
        identity = dict(protocol_id=fingerprint(spec), theta=theta, sequence_seed=seed)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                meta = json.loads(str(saved["metadata_json"]))
                require(meta["identity"] == identity, "Raw output identity differs")
                tokens, logp = saved["tokens"], saved["logp_true"]
                require(np.array_equal(tokens, expected), "Full fixed-bias sequence replay failed")
                require(meta["tokens_sha256"] == hashlib.sha256(tokens.tobytes()).hexdigest(), "Token checksum differs")
                require(meta["logp_sha256"] == hashlib.sha256(logp.tobytes()).hexdigest(), "Prediction checksum differs")
            print(f"Reused theta={theta:g}: {len(tokens)} sequences", flush=True)
        else:
            require(not args.check_only, f"Missing raw output: {path}")
            import torch
            from eval_model import run_model
            from eval_shared_base_models import check_model_alignment
            if model is None:
                model = load_model(config, checkpoint_path)
            started = time.monotonic()
            tokens = expected
            inputs = torch.from_numpy(tokens[:4, :-1].astype(np.int64))
            with torch.no_grad():
                logits = model(inputs)
            alignment = check_model_alignment(model, inputs, logits)
            logp, _ = run_model(model, tokens, "cpu", 0, np.array([], dtype=int), chunk=32)
            direct_logp = torch.log_softmax(logits.float(), dim=-1).numpy()
            selected = np.take_along_axis(direct_logp, tokens[:4, 1:, None], axis=2)[:, :, 0]
            selected_error = float(np.max(np.abs(selected - logp[:4])))
            require(selected_error <= 2e-5, "Selected-token evaluator replay failed")
            reconstructed = reconstruct_binary(logp[:4], tokens[:4, 1:])
            distribution_error = float(np.max(np.abs(np.exp(direct_logp[:, :, 1].astype(float)) - reconstructed)))
            require(distribution_error < 2e-6, "Binary predictive-distribution reconstruction failed")
            meta = dict(identity=identity, runtime=dict(device="cpu", torch_threads=1,
                        torch_version=torch.__version__, numpy_version=np.__version__),
                        alignment=alignment, selected_logp_max_error=selected_error,
                        reconstructed_probability_max_error=distribution_error,
                        tokens_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
                        logp_sha256=hashlib.sha256(logp.tobytes()).hexdigest())
            # Exclusive creation preserves an existing run's immutable raw output.
            with path.open("xb") as stream:
                np.savez_compressed(stream, tokens=tokens, logp_true=logp, metadata_json=canonical(meta))
            print(f"Evaluated theta={theta:g}: {len(tokens)} sequences, {time.monotonic()-started:.1f}s", flush=True)
        provenance = dict(mode="new fixed-bias evaluation of frozen uniform-prior checkpoint",
                          raw=binding(path), protocol=binding(spec_path), validation=meta)
        rows.append(summarize_case("uniform", theta, tokens, logp, provenance))
    return rows, spec


def plot(summary, out, *, all_settings=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.65), sharey=True)
    displayed = [x for x in summary["cases"] if x["prior"] == "twopoint"
                 or all_settings or x["theta"] <= .5]
    upper = max(float(np.max(np.array(row[name]["mean"]) + np.array(row[name]["se"])))
                for row in displayed for name in ("transformer", "informed_bayes"))
    for ax, prior, title in zip(axes, ("twopoint", "uniform"),
                               ("Two-point training prior", "Uniform training prior")):
        cases = [x for x in summary["cases"] if x["prior"] == prior
                 and (prior == "twopoint" or all_settings or x["theta"] <= 0.5)]
        colors = ["#0072B2", "#D55E00"] if prior == "twopoint" else ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
        colors = colors[:len(cases)]
        for row, color in zip(cases, colors):
            for name, style in (("transformer", "-"), ("informed_bayes", "--")):
                mu, se = (np.array(row[name][key]) for key in ("mean", "se"))
                ax.plot(TIMES, mu, color=color, linestyle=style, linewidth=1.25,
                        marker="D" if name == "transformer" else "o", markerfacecolor="none",
                        markersize=3 if name == "transformer" else 4.5)
                ax.fill_between(TIMES, np.maximum(0, mu-se), mu+se,
                                color=color, alpha=.12, linewidth=0)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Observed tokens, $t$")
        ax.set_xlim(1, 256)
        ax.set_ylim(0, 1.12 * upper)
        ax.grid(alpha=.17)
        ax.legend([Line2D([0], [0], color=c, lw=2) for c in colors],
                  [rf"$\theta={r['theta']:g}$" for r in cases],
                  loc="upper right", ncol=min(3, len(cases)), frameon=False,
                  columnspacing=.6, handlelength=.8, handletextpad=.3, fontsize=7.5)
    axes[0].set_ylabel("Expected next-token regret (bits)")
    fig.legend([Line2D([0], [0], color=".15", linestyle="-", lw=1.8, marker="D", markerfacecolor="none"),
                Line2D([0], [0], color=".15", linestyle="--", lw=1.8, marker="o", markerfacecolor="none")],
               ["Transformer", "Bayes with training prior"], loc="lower center", ncol=2,
               bbox_to_anchor=(.5, -.005), frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=.095, right=.985, bottom=.23, top=.9, wspace=.18)
    stem = "conditional_coins_all_settings" if all_settings else "conditional_coins"
    fig.savefig(out / (stem + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / (stem + ".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seq", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(args.n_seq >= 32, "At least 32 independent sequences required")
    rows = reused_two_point()
    print("Reused both fixed biases from the validated two-point cohort", flush=True)
    uniform, spec = fixed_uniform(args)
    rows.extend(uniform)
    summary = dict(schema=SCHEMA, passed=True, metric="E KL(Bernoulli(theta) || next-token predictor), bits",
                   conditioning="Fixed displayed theta; Bayes retains the checkpoint's full training prior.",
                   timing="Sequences have length 257. At t=1,...,256 observe first t tokens and predict token t+1.",
                   uncertainty="One standard error over independent sequences; differences use paired predictions.",
                   two_point_cohort="All 4096 existing sequences, separated into theta=0.2 and theta=0.8 groups.",
                   uniform_protocol=spec, code=binding(Path(__file__)), cases=rows)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        require(json.loads((args.out / "summary.json").read_text()) == summary,
                "Recomputed numerical summary differs")
        receipt = json.loads((args.out / "receipt.json").read_text())
        require(receipt["passed"] and all(sha_file(args.out / name) == checksum
                for name, checksum in receipt["outputs"].items()), "Saved figure or table changed")
        print("PASS: raw identities, all sequence replays, exact Bayes checks, and summary reproduction", flush=True)
        return
    save_json(args.out / "summary.json", summary)
    table = ["prior\ttheta\tn_sequences\tt\ttransformer_mean_bits\ttransformer_se_bits\tbayes_mean_bits\tbayes_se_bits\tpaired_gap_mean_bits\tpaired_gap_se_bits\texact_bayes_mean_bits"]
    for row in rows:
        for i, t in enumerate(TIMES):
            values = [row["prior"], row["theta"], row["n_sequences"], int(t)]
            for key in ("transformer", "informed_bayes", "transformer_minus_bayes"):
                values.extend((row[key]["mean"][i], row[key]["se"][i]))
            values.append(row["exact_bayes"]["mean"][i])
            table.append("\t".join(str(x) for x in values))
    (args.out / "summary.tsv").write_text("\n".join(table) + "\n")
    plot(summary, args.out)
    plot(summary, args.out, all_settings=True)
    outputs = ["summary.json", "summary.tsv", "conditional_coins.pdf", "conditional_coins.png",
               "conditional_coins_all_settings.pdf", "conditional_coins_all_settings.png"]
    save_json(args.out / "receipt.json", dict(passed=True, script_sha256=sha_file(__file__),
              outputs={name: sha_file(args.out / name) for name in outputs}))
    print(f"Wrote seven parameter-specific curves to {args.out}", flush=True)


if __name__ == "__main__":
    main()
