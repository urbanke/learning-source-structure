"""Compare the permutation sampler with enumerated small-alphabet Bayes.

The sampler and exact reference use the same 16-node exponent quadrature.
This isolates sampling error from exponent-quadrature error. Independent
continuous-prior and row-integral checks live in validate_shared_base.py.
Passing these fixed-prefix comparisons is not a convergence certificate
for arbitrary prefixes or for the d=64 application.

Set LSA_STORE and PMM_SRC to the existing read-only anchor store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy

from shared_base import SharedBaseLSA
from source import Source
from weighted_lsa import WeightedLSARowEvidence


HERE = Path(__file__).resolve().parent


def table_provenance():
    store = Path(os.environ["LSA_STORE"]).expanduser().resolve()
    source = Path(os.environ["PMM_SRC"]).expanduser().resolve()
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(store=str(store), source=str(source),
                store_files_sha256={name: digest(store / name)
                                    for name in ("manifest.json", "anchors.json")},
                backend_files_sha256={name: digest(source / "product_model_with_memory" / name)
                                      for name in ("production_coding.py", "universal_tables.py")})


def json_safe(value):
    """Represent undefined diagnostics as null, never as nonstandard JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_case(d, seed):
    source = Source(d, 32, alpha=(0.5, 1.5), ls=(0, 8), permute=True)
    tokens, parameters = source.batch(1, np.random.default_rng(seed))
    return dict(name=f"d{d}_source", d=d, tokens=tokens[0],
                origin="actual full-prior source draw", source_seed=seed,
                generated_latent_parameters={key: value[0].tolist()
                                             for key, value in parameters.items()})


def balanced_case(d):
    motif = ([0, 0, 1, 0, 2, 0, 1, 1, 2, 0, 2, 2, 0, 1, 0, 0]
             if d == 3 else
             [0, 0, 1, 0, 2, 0, 3, 3, 0, 1, 2, 3, 0, 0, 2, 1])
    return dict(name=f"d{d}_balanced", d=d,
                tokens=np.resize(np.asarray(motif, dtype=np.int64), 32),
                origin="fixed motif containing every label")


def exact_prediction(prefix, reference):
    """Append a dummy token and retain only its before-token prediction."""
    augmented = np.append(prefix, 0)
    out = reference.run(augmented, full_probs=True,
                        positions=[len(prefix)], posteriors=True)
    # All permutations are enumerated for each alpha in consecutive blocks.
    alpha = out["base_posterior"][-1].reshape(
        len(reference.alpha_nodes), math.factorial(reference.d)).sum(axis=1)
    return dict(predictive_probs=out["probs"][0],
                posterior_depth=out["depth_posterior"][-1],
                posterior_alpha=alpha)


def check_probability(vector, length, tolerance=1e-7):
    vector = np.asarray(vector, dtype=float)
    finite = vector.shape == (length,) and bool(np.isfinite(vector).all())
    nonnegative = finite and bool((vector >= 0).all())
    error = abs(float(vector.sum()) - 1.0) if finite else math.inf
    return dict(passed=nonnegative and error <= tolerance,
                finite=finite, nonnegative=nonnegative,
                sum_error=error, tolerance=tolerance)


def run_case(case, args, index):
    # Imports are deferred so the help text does not need the sampler backend.
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_mcmc import SharedBaseMCMC

    d, prefix = case["d"], case["tokens"]
    exact_kernel = WeightedLSARowEvidence()
    reference = SharedBaseLSA(d, exact_kernel, alpha_order=16)
    kernel = RankedLSARowEvidence(d, reference.alpha_nodes, depths=range(9))
    # Configure and check production mode before the independent evaluator
    # imports the same read-only anchor backend.
    kernel._get_backend()
    start = time.monotonic()
    exact = exact_prediction(prefix, reference)
    exact_seconds = time.monotonic() - start
    sampler = SharedBaseMCMC(d, kernel, reference.alpha_weights)
    seed = args.seed + 1009 * index
    estimate = sampler.sample(prefix, chains=4, burn=args.burn,
                              draws=args.draws, thin=args.thin, seed=seed)
    p = np.asarray(estimate["predictive_probs"], dtype=float)
    se = np.asarray(estimate["mcse_probs"], dtype=float)
    p_exact = exact["predictive_probs"]
    valid_se = (se.shape == (d,) and np.isfinite(se).all()
                and (se >= 0).all())
    valid_p = p.shape == (d,) and np.isfinite(p).all()
    error = np.abs(p - p_exact) if valid_p else np.full(d, np.inf)
    tolerance = 5.0 * se + 0.0002 if valid_se else np.zeros(d)
    cap = 0.04 if args.smoke else 0.015
    checks = dict(
        exact_probability=check_probability(p_exact, d),
        sampled_probability=check_probability(p, d),
        posterior_depth=check_probability(estimate["posterior_depth"], 9),
        posterior_alpha=check_probability(estimate["posterior_alpha"], 16),
        mcse_finite_nonnegative=dict(passed=bool(valid_se)),
        prediction_absolute_cap=dict(passed=bool(np.all(error <= cap)),
                                     max_absolute_error=float(error.max()),
                                     tolerance=cap),
        prediction_mcse_comparison=dict(passed=bool(valid_se and np.all(error <= tolerance)),
                                        absolute_error=error,
                                        tolerance=tolerance,
                                        multiplier=5.0, numerical_allowance=0.0002),
    )
    for quantity, se_name in (("posterior_depth", "mcse_depth"),
                             ("posterior_alpha", "mcse_alpha")):
        value = np.asarray(estimate[quantity], dtype=float)
        uncertainty = np.asarray(estimate[se_name], dtype=float)
        valid = (value.shape == uncertainty.shape == exact[quantity].shape
                 and np.isfinite(value).all() and np.isfinite(uncertainty).all()
                 and (uncertainty >= 0).all())
        discrepancy = np.abs(value - exact[quantity])
        bound = 5.0 * uncertainty + 0.0002
        checks[quantity + "_mcse_comparison"] = dict(
            passed=bool(valid and np.all(discrepancy <= bound)),
            absolute_error=discrepancy, tolerance=bound,
            multiplier=5.0, numerical_allowance=0.0002)
    passed = all(check["passed"] for check in checks.values())
    result = dict(
        **case, passed=passed, seed=seed, prefix_length=len(prefix),
        exact=exact,
        estimate={name: estimate[name] for name in
                  ("predictive_probs", "mcse_probs", "posterior_depth", "mcse_depth",
                   "posterior_alpha", "mcse_alpha")},
        errors=dict(max_probability_absolute=float(error.max()),
                    probability_total_variation=float(error.sum() / 2),
                    max_depth_absolute=float(np.max(np.abs(
                        np.asarray(estimate["posterior_depth"]) - exact["posterior_depth"]))),
                    alpha_total_variation=float(np.sum(np.abs(
                        np.asarray(estimate["posterior_alpha"]) - exact["posterior_alpha"])) / 2)),
        checks=checks, diagnostics=estimate["diagnostics"],
        elapsed_seconds=dict(exact=exact_seconds, sampler=estimate["elapsed_seconds"]),
    )
    print(json.dumps(dict(case=case["name"], passed=passed,
                          max_probability_error=float(error.max()),
                          max_probability_mcse=float(se.max()) if valid_se else None,
                          sampler_seconds=estimate["elapsed_seconds"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--burn", type=int)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.draws is None:
        args.draws = 256 if args.smoke else 1024
    if args.burn is None:
        args.burn = 256 if args.smoke else 512
    if args.draws < 32 or args.burn < 0 or args.thin < 1:
        parser.error("require draws >= 32, burn >= 0, and thin >= 1")
    cases = [source_case(3, 20260914), source_case(4, 20260914)]
    if not args.smoke:
        cases.extend([balanced_case(3), balanced_case(4)])
    start = time.monotonic()
    records = [run_case(case, args, index) for index, case in enumerate(cases)]
    code_names = ("ranked_lsa.py", "shared_base_mcmc.py", "shared_base.py",
                  "weighted_lsa.py", "lsa_seq.py", "phi_source.py", "source.py",
                  "validate_shared_base_mcmc.py")
    report = dict(
        schema="shared_base_mcmc_validation_v1", passed=all(row["passed"] for row in records),
        settings=dict(smoke=args.smoke, chains=4, burn=args.burn, draws=args.draws,
                      thin=args.thin, seed=args.seed, source_prefix_tokens=32,
                      alpha_order=16, alpha_interval=[0.5, 1.5], depths=list(range(9)),
                      rank_prior="uniform over all label permutations",
                      unknown_shared_depth=True),
        scope="Small-alphabet sampler comparisons to exact permutation/depth sums at the same exponent quadrature.",
        limitations=[
            "MCSE and agreement among chains are diagnostics, not convergence certificates.",
            "Passing these fixed cases does not establish accuracy at d=64 or on longer prefixes.",
            "This comparison does not assess exponent quadrature error; both methods use the same nodes.",
            "Generated latent parameters are recorded for reproducibility and never supplied to either predictor.",
            "Nonfinite diagnostics, if present, are encoded as null; probability checks require finite values.",
        ],
        cases=records, elapsed_seconds=time.monotonic() - start,
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        provenance=dict(code_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                     for name in code_names},
                        **table_provenance()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    rows = ["case\td\tprefix_tokens\tmax_probability_error\tmax_mcse\tprobability_tv\tmax_depth_error\talpha_tv\tsampler_seconds\tpassed"]
    for row in records:
        rows.append("\t".join(str(value) for value in (
            row["name"], row["d"], row["prefix_length"], row["errors"]["max_probability_absolute"],
            max(row["estimate"]["mcse_probs"]), row["errors"]["probability_total_variation"],
            row["errors"]["max_depth_absolute"], row["errors"]["alpha_total_variation"],
            row["elapsed_seconds"]["sampler"], row["passed"])))
    args.out.with_suffix(".tsv").write_text("\n".join(rows) + "\n")
    print(json.dumps(dict(out=str(args.out), passed=report["passed"],
                          cases=len(records), seconds=report["elapsed_seconds"])), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
