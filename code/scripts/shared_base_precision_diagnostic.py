"""Read-only precision diagnosis for declared shared-base conditional pilots.

This does not change reference gates, estimate a population curve, or select
passing cases. It checks covariance propagation independently from scalar
series, inspects longer batches, and estimates the non-cancelling Jensen term
in log loss. Rank trajectories are not saved, so residual variation cannot be
attributed conclusively to particular rank moves.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/transformer")]
from conditional_shared_base_benchmark import inputs
from shared_base_full_context_summary import covariance_of_mean, comparison_metrics


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def read_raw(path):
    with np.load(path, allow_pickle=False) as saved:
        serialized = str(saved["metadata"].item())
        assert hashlib.sha256(serialized.encode()).hexdigest() == str(saved["metadata_sha256"].item())
        meta = json.loads(serialized)
        for key, expected in meta["array_hashes"].items():
            assert array_hash(saved[key]) == expected, key
        n = meta["config"]["chains"]
        return (np.stack([saved[f"predictions_{j}"] for j in range(n)]),
                np.stack([saved[f"scalars_{j}"] for j in range(n)]),
                np.stack([saved[f"depths_{j}"] for j in range(n)]), meta)


def loss(probs, oracle):
    return -np.log2(probs) @ oracle


def linear_r2(y, features):
    design = np.column_stack([np.ones(len(y)), features])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(1 - np.sum((y - fitted)**2) / np.sum((y-y.mean())**2))


def diagnose(samples, scalars, depths, oracle, transformer):
    n_chains, draws, _ = samples.shape
    moments = covariance_of_mean(samples)
    mean = moments["mean"]
    gradient = oracle / (mean * np.log(2))
    centered_gradient = gradient - gradient.mean()
    projected = samples @ centered_gradient
    covariances = {key: moments["covariance_" + key] for key in ("batch", "between")}
    metrics = comparison_metrics(mean, transformer, oracle, *covariances.values())
    # Independent route: project each draw first, then estimate scalar variance.
    projected_checks = {}
    length = moments["batch_length"]
    for name, values in (
        ("between", projected.mean(axis=1)),
        ("batch", projected[:, :draws//length*length].reshape(n_chains, -1, length).mean(axis=2).ravel()),
    ):
        scalar_variance = float(values.var(ddof=1) / len(values))
        matrix_variance = float(centered_gradient @ covariances[name] @ centered_gradient)
        assert np.isclose(scalar_variance, matrix_variance, rtol=1e-9, atol=1e-14)
        projected_checks[name] = dict(scalar_variance=scalar_variance,
                                      matrix_variance=matrix_variance)
    block_scales = []
    for length in (64, 128, 256, 512, 1024, 2048):
        if draws // length < 4:
            continue
        covariance = covariance_of_mean(samples, batch_length=length)["covariance_batch"]
        block_scales.append(dict(batch_length=length, mcse_bits=float(np.sqrt(
            max(0., centered_gradient @ covariance @ centered_gradient)))))
    chain_means = samples.mean(axis=1)
    full_loss = float(loss(mean, oracle))
    jackknife_bias = (n_chains-1) * (np.mean([
        loss(np.delete(chain_means, j, axis=0).mean(axis=0), oracle)
        for j in range(n_chains)]) - full_loss)
    all_chain_count_losses = []
    for n in (1, 2, n_chains):
        all_chain_count_losses.append(dict(chains_averaged=n,
            mean_cross_entropy_bits=float(np.mean([
                loss(chain_means[list(indices)].mean(axis=0), oracle)
                for indices in itertools.combinations(range(n_chains), n)]))))
    envelope = np.maximum.reduce([np.maximum(np.diag(c), 0) for c in covariances.values()])
    weighted_variance = gradient**2 * envelope
    flat = projected.ravel()
    alpha = scalars[:, :, 0].ravel()
    depth_mean = (depths @ np.arange(depths.shape[2])).ravel()
    features = np.column_stack([alpha, alpha**2, depth_mean, depth_mean**2])
    token_rows = []
    for token in np.argsort(-weighted_variance)[:8]:
        token_values = samples[:, :, token].ravel()
        token_rows.append(dict(token=int(token), true_probability=float(oracle[token]),
            bayes_probability=float(mean[token]), probability_mcse=float(np.sqrt(envelope[token])),
            loss_gradient_weight=float(gradient[token]),
            correlation_with_loss_projection=float(np.corrcoef(token_values, flat)[0, 1]),
            diagonal_loss_variance_scale=float(weighted_variance[token])))
    return dict(chains=n_chains, draws_per_chain=draws,
        bayes_cross_entropy_bits=full_loss, oracle_entropy_bits=float(loss(oracle, oracle)),
        bayes_regret_bits=full_loss-float(loss(oracle, oracle)),
        covariance_projection_checks=projected_checks,
        loss_mcse=metrics["gap_uncertainty"], longer_block_mcse=block_scales,
        max_relative_probability_mcse=metrics["max_relative_probability_mcse"],
        jensen_bias_local_scale=metrics["gap_second_order"],
        independent_chain_jackknife_bias_bits=float(jackknife_bias),
        chain_count_loss_diagnostic=all_chain_count_losses,
        chain_cross_entropy_bits=loss(chain_means, oracle).tolist(),
        chain_mean_alpha=scalars[:, :, 0].mean(axis=1).tolist(),
        chain_mean_expected_depth=(depths @ np.arange(depths.shape[2])).mean(axis=1).tolist(),
        correlation_with_alpha=float(np.corrcoef(flat, alpha)[0, 1]),
        correlation_with_expected_depth=float(np.corrcoef(flat, depth_mean)[0, 1]),
        joint_quadratic_alpha_depth_r2=linear_r2(flat, features),
        most_sensitive_tokens=token_rows,
        note="Token diagonal scales do not partition total variance; covariance matters. Alpha/depth associations are descriptive, not causal attributions. Jensen scales and jackknife do not bound unknown-mode bias.")


def self_test():
    # Exact finite sum for an unbiased two-token mean with known covariance.
    # This checks the sign and size of Jensen bias without any MCMC assumptions.
    p = np.array([.3, .7]); r = np.array([.6, .4]); delta = np.array([.1, -.1])
    draws = 64
    weights = np.array([math.comb(draws, k) / 2**draws for k in range(draws+1)])
    estimates = p + (2*np.arange(draws+1)/draws - 1)[:, None]*delta
    exact_bias = float(weights @ loss(estimates, r) - loss(p, r))
    covariance = np.outer(delta, delta) / draws
    hessian = r/(p*p*np.log(2))
    second_order = float(.5 * np.dot(hessian, np.diag(covariance)))
    assert exact_bias > 0 and abs(exact_bias/second_order-1) < .005
    # Central finite differences check the log-loss gradient and curvature.
    step = 1e-4
    derivative = float((loss(p+step*delta, r)-loss(p-step*delta, r))/(2*step))
    expected = float(-r/(p*np.log(2)) @ delta)
    assert np.isclose(derivative, expected, rtol=1e-8)
    curvature = float((loss(p+step*delta, r)-2*loss(p, r)+loss(p-step*delta, r))/step**2)
    assert np.isclose(curvature, np.dot(hessian, delta**2), rtol=1e-5)
    return dict(passed=True, exact_binomial_jensen_bias_bits=exact_bias,
                second_order_jensen_bias_bits=second_order,
                relative_second_order_error=abs(exact_bias/second_order-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"output/conditional_shared_base_benchmark_2026-09-15/a1_L4")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--times", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--orders", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--out", type=Path, default=ROOT/"results/transformer/shared_base_precision/summary.json")
    args = parser.parse_args()
    rows = []
    for time in args.times:
        _, oracle, model, _ = inputs(args.input, args.index, time)
        for order in args.orders:
            path = args.input/f"chains/alpha{order}/seq_{args.index:06d}_t{time:04d}.npz"
            samples, scalars, depths, meta = read_raw(path)
            report = diagnose(samples, scalars, depths, oracle, model["d64"]["prediction"])
            rows.append(dict(time=time, order=order, index=args.index,
                chain_file=str(path.resolve()), chain_sha256=sha_file(path),
                numerical_problem=meta["fingerprint"], **report))
    result = dict(schema="shared-base-precision-diagnostic-v1", checks=self_test(),
                  diagnostic_sha256=sha_file(__file__), rows=rows,
                  interpretation="Single declared sequence, no population average. Existing reference validation status is unchanged.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(checks=result["checks"], summaries=[dict(time=r["time"], order=r["order"],
          mcse=r["loss_mcse"]["mcse_bits"], longer_block_max=max(v["mcse_bits"] for v in r["longer_block_mcse"]),
          jensen_bias=r["jensen_bias_local_scale"]["noise_scale_bits"], jackknife=r["independent_chain_jackknife_bias_bits"],
          alpha_depth_r2=r["joint_quadratic_alpha_depth_r2"]) for r in rows]), indent=2))


if __name__ == "__main__":
    main()
