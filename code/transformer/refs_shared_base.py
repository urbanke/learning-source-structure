"""Compute the original source's small-alphabet Bayesian predictions.

Examples (LSA_STORE and PMM_SRC identify the existing read-only anchor store):
    python code/transformer/refs_shared_base.py --d 3 --n-tokens 33 --out pilot.json
    python code/transformer/refs_shared_base.py --d 4 --sequence 0,1,0,2,3,0 --out path.json

The predictor retains the original independent priors: alpha uniform on
[0.5,1.5], depth uniform on 0..8, and all label permutations equally likely.
Exponent and weighted-row integrals are numerical; permutations and depths
are summed explicitly. Use validate_shared_base.py to check accuracy.
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
import scipy

from shared_base import SharedBaseLSA
from source import Source
from weighted_lsa import WeightedLSARowEvidence


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--sequence", help="comma-separated observed token labels in 0..d-1")
    parser.add_argument("--n-tokens", type=int, default=33, help="length of a generated test sequence")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--alpha-order", type=int, default=16)
    parser.add_argument("--u-lo", type=float, default=-32.0)
    parser.add_argument("--u-hi", type=float, default=32.0)
    parser.add_argument("--step", type=float, default=1 / 32)
    parser.add_argument("--prob-tol", type=float, default=1e-7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n_tokens < 2 or args.prob_tol <= 0 or not np.isfinite(args.prob_tol):
        parser.error("require at least two generated tokens and finite positive probability tolerance")
    generated = None
    if args.sequence is not None:
        x = np.array([int(token) for token in args.sequence.split(",")], dtype=np.int64)
        if len(x) < 2:
            parser.error("supply at least two observed tokens")
    else:
        source = Source(args.d, args.n_tokens, alpha=(0.5, 1.5), ls=(0, 8), permute=True)
        tokens, parameters = source.batch(1, np.random.default_rng(args.seed))
        x = tokens[0]
        generated = dict(seed=args.seed, parameters={key: value[0].tolist()
                                                    for key, value in parameters.items()},
                         note="Latent draw recorded for reproducibility; not supplied to the predictor.")
    kernel = WeightedLSARowEvidence(u_lo=args.u_lo, u_hi=args.u_hi, step=args.step)
    reference = SharedBaseLSA(args.d, kernel, alpha_order=args.alpha_order)
    start = time.monotonic()
    result = reference.run(x, full_probs=True, posteriors=True)
    seconds = time.monotonic() - start
    probability_error = float(np.max(np.abs(result["prob_sum_error"])))
    if not np.isfinite(result["probs"]).all() or (result["probs"] < 0).any() or probability_error > args.prob_tol:
        raise FloatingPointError(f"prediction normalization failed: {probability_error:g}")
    batch_error = abs(reference.log_joint(x) - result["log_evidence"][-1]) / np.log(2)
    if batch_error > 1e-7:
        raise FloatingPointError(f"sequential/batch evidence differs by {batch_error:g} bits")
    base_posterior = result["base_posterior"]
    prior = dict(alpha_interval=[0.5, 1.5], alpha_prior="continuous uniform",
                 depth_values=list(range(9)), depth_prior="uniform",
                 rank_prior="uniform over all permutations",
                 first_token_law="base q", depth_scope="shared by all rows")
    here = Path(__file__).resolve().parent
    store = os.environ.get("LSA_STORE")
    store_manifest = Path(store).expanduser() / "manifest.json" if store else None
    backend_src = os.environ.get("PMM_SRC")
    backend_file = Path(backend_src).expanduser() / "product_model_with_memory/universal_tables.py" if backend_src else None
    report = dict(
        schema="small_alphabet_shared_base_v1", d=args.d, prior=prior,
        tokens=x.tolist(), generated_source=generated,
        first_token_scored=False, loss_bits=result["loss"][1:].tolist(),
        scored_loss_total_bits=float(result["loss"][1:].sum()),
        prefix_lengths=result["positions"].tolist(),
        predictive_probs=result["probs"].tolist(),
        posterior_depth_before_token=result["depth_posterior"].tolist(),
        posterior_mean_alpha_before_token=(base_posterior @ reference.alphas).tolist(),
        posterior_mean_base_before_token=(base_posterior @ reference.base_laws).tolist(),
        quadrature=dict(alpha_order=args.alpha_order, alpha_nodes=reference.alpha_nodes.tolist(),
                        alpha_weights=reference.alpha_weights.tolist(),
                        u_lo=args.u_lo, u_hi=args.u_hi, step=args.step,
                        base_hypotheses=reference.n_hypotheses,
                        joint_base_depth_hypotheses=reference.n_hypotheses * len(reference.depths)),
        checks=dict(max_probability_sum_error=probability_error,
                    sequential_batch_error_bits=batch_error,
                    note="Numerical consistency checks; use the separate refinement validation for accuracy."),
        elapsed_seconds=seconds, cache=kernel.cache_info(),
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        provenance=dict(code_sha256={name: sha_file(here / name) for name in
                                     ("weighted_lsa.py", "shared_base.py", "shared_depth.py",
                                      "phi_source.py", "source.py", "refs_shared_base.py")},
                        store=store, source=backend_src,
                        store_manifest_sha256=sha_file(store_manifest) if store_manifest else None,
                        backend_sha256=sha_file(backend_file) if backend_file else None))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(dict(out=str(args.out), d=args.d, tokens=len(x),
                          seconds=seconds, normalization_error=probability_error,
                          scored_loss_bits=report["scored_loss_total_bits"])), flush=True)


if __name__ == "__main__":
    main()
