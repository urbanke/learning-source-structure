"""Reproduce why the original source3 endpoint reference remained unresolved.

The depth-zero calculation conditions on saved rank assignments and is a
quadrature diagnostic only. It is not a replacement Bayesian reference.
All old files remain read-only; this script writes a separate small receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "transformer"))
from stable_chain_statistics import SUMMARY_SPEC, stable_chain_summary


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditional_depth0_quadrature(tokens, ranks, order):
    """Integrate alpha given L=0 and a fixed rank assignment on the old prior."""
    counts = np.bincount(tokens, minlength=len(ranks))
    nodes, weights = leggauss(order)
    nodes, weights = 1 + nodes/2, weights/2
    log_h = logsumexp(-nodes[:, None]*np.log(np.arange(1, len(ranks)+1)), axis=1)
    log_terms = np.log(weights) - nodes*np.dot(counts, np.log(ranks+1)) - len(tokens)*log_h
    log_integral = logsumexp(log_terms)
    posterior = np.exp(log_terms-log_integral)
    return dict(order=order, conditional_log_integral=float(log_integral),
                mean_alpha=float(posterior @ nodes),
                rank_one_probability=float(posterior @ np.exp(-log_h)))


def diagnose(folder):
    source_path = folder / "data" / "seq_000003.json"
    tokens = np.asarray(json.loads(source_path.read_text())["tokens"][:1024], dtype=int)
    if len(tokens) != 1024:
        raise ValueError("expected the original 1024-token prefix")
    result = dict(schema="shared-base-source3-diagnosis-v1", source_index=3,
                  prefix_length=1024, summary_spec=SUMMARY_SPEC,
                  scope="Read-only arithmetic, mixing and conditional-depth0 quadrature diagnosis; not a family Bayesian reference.",
                  inputs={str(source_path.relative_to(ROOT)): file_hash(source_path)}, rules={})
    for order in (16, 32):
        chain_path = folder / "chains" / f"alpha{order}" / "seq_000003.npz"
        report_path = folder / "refs" / f"alpha{order}" / "seq_000003.json"
        moments_path = report_path.with_suffix(".npz")
        for path in (chain_path, report_path, moments_path):
            result["inputs"][str(path.relative_to(ROOT))] = file_hash(path)
        report = json.loads(report_path.read_text())
        estimate = report["estimate"]
        with np.load(moments_path, allow_pickle=False) as stored:
            moments = {key: stored[key].copy() for key in stored.files}
        with np.load(chain_path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            predictions = np.stack([stored[f"predictions_{i}"] for i in range(4)])
            alpha = np.stack([stored[f"scalars_{i}"][:, 0] for i in range(4)])
            raw_means = np.stack([stored[f"predictions_{i}"].mean(axis=0) for i in range(4)])
        stable = stable_chain_summary(predictions)
        rank_one = np.exp(-logsumexp(-alpha[:, :, None]*np.log(np.arange(1, 65)), axis=2))
        diagnostic = estimate["diagnostics"]
        result["rules"][str(order)] = dict(
            draws_per_chain=int(diagnostic["draws"]),
            old_min_probability_ess=float(min(diagnostic["ess_probs"])),
            old_min_ess_token=int(np.argmin(diagnostic["ess_probs"])),
            stable_min_probability_ess=float(min(stable["ess"])),
            old_chain_mean_readback_error=float(np.max(np.abs(moments["chain_means"].mean(0)-moments["mean"]))),
            raw_vs_saved_chain_mean_error=float(np.max(np.abs(raw_means-moments["chain_means"]))),
            stable_chain_mean_readback_error=float(np.max(np.abs(stable["chain_means"].mean(0)-stable["mean"]))),
            stable_vs_old_prediction_change=float(np.max(np.abs(stable["mean"]-estimate["predictive_probs"]))),
            posterior_depth=estimate["posterior_depth"],
            mean_alpha=float(alpha.mean()),
            token41_probability=float(estimate["predictive_probs"][41]),
            token41_ess=float(diagnostic["ess_probs"][41]),
            token41_mcse=float(estimate["mcse_probs"][41]),
            token41_chain_means=np.asarray(diagnostic["chain_mean_probs"])[:, 41].tolist(),
            token41_vs_rank_one_formula_error=float(np.max(np.abs(predictions[:, :, 41]-rank_one))),
            final_state_conditional_depth0=[dict(chain=i, quadratures=[
                conditional_depth0_quadrature(tokens, np.asarray(state["ranks"]), q)
                for q in (16, 32, 64, 128, 256)])
                for i, state in enumerate(metadata["states"])])
    all_states = [state for rule in result["rules"].values() for state in rule["final_state_conditional_depth0"]]
    result["largest_conditional_rank_one_change"] = {
        f"{a}_to_{b}": max(abs(state["quadratures"][i]["rank_one_probability"]-
                                  state["quadratures"][i+1]["rank_one_probability"])
                            for state in all_states)
        for i, (a, b) in enumerate(zip((16, 32, 64, 128), (32, 64, 128, 256)))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output" / "shared_base_full_context")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "transformer" /
                        "shared_base_full_context" / "source3_diagnosis_2026-09-15.json")
    args = parser.parse_args()
    result = diagnose(args.input.resolve())
    result["producer_sha256"] = file_hash(Path(__file__))
    result["summary_code_sha256"] = file_hash(ROOT / "code" / "transformer" / "stable_chain_statistics.py")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.out),
                          conditional_changes=result["largest_conditional_rank_one_change"])))


if __name__ == "__main__":
    main()
