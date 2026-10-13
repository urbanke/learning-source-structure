"""Predictive backoff components of the existing product-family Bayes rule.

For each coordinate, p = (1-lambda) uniform + lambda C. The uniform
floor is the virtual zero-count LSA evidence increment, including when
every successor label has appeared. Depths are averaged with the GLOBAL
coordinate-chain depth posterior. No iid hypothesis is added to the prior.

The resulting UU/CU/UC/CC weights are predictive backoff weights, NOT
posterior probabilities of fixed iid/Markov hypotheses. C is arbitrary
uniform when lambda=0, and context_defined marks that non-identifiability.
Even a defined C may equal the uniform component; their separation is
reported separately and no behavioral identifiability is asserted.
Raw probabilities and weights are never clipped or renormalized.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from shared_depth import _positions, _tokens

COMPONENT_NAMES = ("UU", "CU", "UC", "CC")


def _check_probabilities(values, name, tolerance):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise FloatingPointError(f"{name} contains nonfinite or negative entries")
    error = float(np.max(np.abs(values.sum(axis=-1) - 1))) if values.size else 0.0
    if error > tolerance:
        raise FloatingPointError(f"{name} is not normalized: raw error {error:.3g}")
    return error


def row_components(reference, table, log_depth_weights, *, prob_tol=1e-7):
    """Decompose one current row without changing the count table.

    table=None denotes an empty context. For positive lambda, C is evaluated
    from the weighted per-depth excesses p_L-a_L. This is algebraically
    (p-a)/lambda and avoids cancellation at genuinely unseen labels.
    Depth-zero and depth-one predictions/floors use their exact formulas.
    Higher depths reuse the frozen row kernel's evidence ratios.
    """
    if not np.isfinite(prob_tol) or prob_tol <= 0:
        raise ValueError("prob_tol must be finite and positive")
    d, depths = reference.d, np.asarray(reference.depths)
    log_weights = np.asarray(log_depth_weights, dtype=float)
    if log_weights.shape != depths.shape or np.isnan(log_weights).any() or np.isposinf(log_weights).any():
        raise ValueError("log depth weights must match reference.depths")
    weights = np.exp(log_weights)
    _check_probabilities(weights, "global depth weights", prob_tol)
    uniform = np.full(d, 1.0 / d)
    counts = np.zeros(d, dtype=int) if table is None else np.asarray(table.m)
    total = 0 if table is None else table.k
    if counts.shape != (d,) or counts.dtype.kind not in "iu" or np.any(counts < 0) or counts.sum() != total:
        raise ValueError("invalid current-row count table")
    if total == 0:
        probability = uniform.copy()
        depth_probs = np.tile(uniform, (len(depths), 1))
        floors = np.full(len(depths), 1 / d)
        depth_lambda = np.zeros(len(depths))
        floor, weight, context, defined = 1 / d, 0.0, uniform.copy(), False
    else:
        before = reference._correct_evidence(table.logQ, counts, total)
        depth_probs = np.empty((len(depths), d))
        for count in np.unique(counts):
            mask = counts == count
            candidate_counts = counts.copy()
            candidate_counts[np.flatnonzero(mask)[0]] += 1
            after = reference._correct_evidence(table._logQ_after(int(count)), candidate_counts, total + 1)
            depth_probs[:, mask] = np.exp(after - before)[:, None]
        # The virtual increment is NOT evidence for a fabricated count vector.
        # It is the current integrand multiplied by phi_1/phi_0 at every L.
        floors = np.exp(np.asarray(table._logQ_after(0)) - before)
        zero, one = depths == 0, depths == 1
        floors[zero] = 1 / d
        floors[one] = 1 / (total + d)
        depth_probs[zero] = uniform
        depth_probs[one] = (counts + 1) / (total + d)
        depth_lambda = 1 - d * floors
        depth_lambda[zero] = 0.0
        _check_probabilities(depth_probs, "per-depth row predictions", prob_tol)
        if (not np.isfinite(floors).all() or np.any(floors < 0)
                or np.any(depth_lambda < 0) or np.any(depth_lambda > 1)):
            raise FloatingPointError("virtual floor does not give convex per-depth backoff weights")
        excess = depth_probs - floors[:, None]
        if np.any(excess < 0):
            raise FloatingPointError("virtual floor exceeds a per-depth successor probability")
        probability = weights @ depth_probs
        floor = float(weights @ floors)
        weight = float(1 - d * floor)
        if weight < 0 or weight > 1:
            raise FloatingPointError("mixed virtual floor does not give a convex backoff weight")
        defined = weight > 0
        context = (weights @ excess) / weight if defined else uniform.copy()
    reconstructed = (1 - weight) * uniform + weight * context
    probability_error = _check_probabilities(probability, "coordinate prediction", prob_tol)
    context_error = _check_probabilities(context, "context component", prob_tol)
    reconstruction_error = float(np.max(np.abs(reconstructed - probability)))
    if reconstruction_error > prob_tol:
        raise FloatingPointError("coordinate component reconstruction failed")
    return dict(uniform_probs=uniform, context_probs=context, context_weight=weight,
                uniform_floor=floor, prediction_probs=probability, context_defined=defined,
                depth_probs=depth_probs, depth_floors=floors, depth_context_weights=depth_lambda,
                reconstructed_probs=reconstructed, reconstruction_error=reconstruction_error,
                prediction_sum_error=probability_error, context_sum_error=context_error,
                uniform_context_tv=float(np.sum(np.abs(context - uniform)) / 2),
                row_counts=counts.copy(), row_total=int(total))


def coordinate_components(reference, x, positions=None, *, prob_tol=1e-7):
    """Components at observed prefixes; default positions are 1..len(x)-1.

    Position t predicts x[t] from x[:t]. The frozen SharedDepthLSA run
    supplies the whole-chain depth posterior BEFORE each token. A second
    pass reconstructs only the current row's integrand and decomposition.
    """
    x = _tokens(x, reference.d)
    selected = _positions(len(x), True, np.arange(1, len(x)) if positions is None else positions)
    result = reference.run(x, full_probs=True, positions=selected, depth_posteriors=True)
    wanted = set(selected.tolist())
    tables, rows = [None] * reference.d, []
    for position, symbol in enumerate(x):
        table = None if position == 0 else tables[int(x[position - 1])]
        if position in wanted:
            log_weights = np.full(len(reference.depths), -np.inf)
            weights = result["depth_posterior"][position]
            np.log(weights, out=log_weights, where=weights > 0)
            rows.append(row_components(reference, table, log_weights, prob_tol=prob_tol))
        if position:
            context = int(x[position - 1])
            if table is None:
                table = reference.kernel.new_table()
                tables[context] = table
            table.observe(int(symbol))
            table.logQ = reference._correct_evidence(table.logQ, table.m, table.k)
    if rows:
        out = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    else:
        prototype = row_components(reference, None, reference.log_prior, prob_tol=prob_tol)
        out = {key: np.empty((0,) + np.asarray(value).shape, dtype=np.asarray(value).dtype)
               for key, value in prototype.items()}
    out.update(positions=selected, reference_probs=result["probs"],
               depth_posterior=result["depth_posterior"][selected], depths=np.asarray(reference.depths))
    error = float(np.max(np.abs(out["prediction_probs"] - result["probs"]))) if len(selected) else 0.0
    if error > prob_tol:
        raise FloatingPointError(f"decomposition disagrees with frozen SharedDepthLSA: {error:.3g}")
    out["reference_error"] = error
    return out


def product_components(first, second, x, positions=None, *, prob_tol=1e-7):
    """Full product prediction and its four predictive backoff weights."""
    x = _tokens(x, first.d * second.d)
    first_out = coordinate_components(first, x // second.d, positions, prob_tol=prob_tol)
    second_out = coordinate_components(second, x % second.d, positions, prob_tol=prob_tol)
    l1, l2 = first_out["context_weight"], second_out["context_weight"]
    weights = np.stack(((1 - l1) * (1 - l2), l1 * (1 - l2), (1 - l1) * l2, l1 * l2), axis=-1)
    _check_probabilities(weights, "joint backoff weights", prob_tol)

    def outer(left, right):
        return (left[:, :, None] * right[:, None, :]).reshape(len(left), first.d * second.d)

    u1, u2 = first_out["uniform_probs"], second_out["uniform_probs"]
    c1, c2 = first_out["context_probs"], second_out["context_probs"]
    components = np.stack((outer(u1, u2), outer(c1, u2), outer(u1, c2), outer(c1, c2)), axis=1)
    reconstructed = np.sum(weights[:, :, None] * components, axis=1)
    joint = outer(first_out["reference_probs"], second_out["reference_probs"])
    _check_probabilities(components, "joint component distributions", prob_tol)
    _check_probabilities(joint, "joint reference", prob_tol)
    error = float(np.max(np.abs(reconstructed - joint))) if len(joint) else 0.0
    if error > prob_tol:
        raise FloatingPointError(f"joint component reconstruction failed: {error:.3g}")
    return dict(positions=first_out["positions"], coordinate1=first_out, coordinate2=second_out,
                component_names=COMPONENT_NAMES, weights=weights, component_probs=components,
                joint_probs=joint, reconstructed_probs=reconstructed, reconstruction_error=error,
                interpretation="predictive backoff weights, not posterior iid/Markov hypothesis probabilities")


def main(argv=None):
    """Bounded reproducible validation; no transformer or full cohort run."""
    import importlib.metadata
    import sys
    from eval_conditional_crossing import array_hash, configure_backend, sha_file
    from families import ProductFamily
    from lsa_seq import SeqLSA
    from shared_depth import SharedDepthLSA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/transformer/product_components/validation.json"))
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args(argv)
    if args.n_ctx < 1:
        parser.error("--n-ctx must be positive")
    store, src = configure_backend(args.store, args.pmm_src)
    start = time.monotonic()
    kernel = SeqLSA(8, depths=range(1, 9), rmax=args.n_ctx + 1, u_lo=-30, u_hi=30)
    first, second = SharedDepthLSA(kernel), SharedDepthLSA(kernel)
    tokens, parameters = ProductFamily(8, 8, args.n_ctx + 1, ls=(1, 8)).batch(1, np.random.default_rng(args.seed))
    result = product_components(first, second, tokens[0])
    coordinate_checks = {}
    for name in ("coordinate1", "coordinate2"):
        value = result[name]
        counts, totals = value["row_counts"], value["row_total"]
        expected = (counts + 1) / (totals[:, None] + 8)
        floor = 1 / (totals + 8)
        error = max(float(np.max(np.abs(value["depth_probs"][:, 0] - expected))),
                    float(np.max(np.abs(value["depth_floors"][:, 0] - floor))))
        coordinate_checks[name] = dict(reference_error=value["reference_error"],
            maximum_component_sum_error=float(np.max(value["context_sum_error"])),
            maximum_reconstruction_error=float(np.max(value["reconstruction_error"])),
            depth_one_closed_form_error=error,
            minimum_context_probability=float(np.min(value["context_probs"])),
            minimum_context_weight=float(value["context_weight"].min()),
            maximum_context_weight=float(value["context_weight"].max()),
            maximum_uniform_context_tv=float(value["uniform_context_tv"].max()),
            all_successors_observed_prefixes=int(np.sum((counts > 0).all(axis=1))),
            empty_context_prefixes=int(np.sum(totals == 0)))
    here = Path(__file__).resolve().parent
    paths = {name: here / name for name in ("product_components.py", "eval_conditional_crossing.py",
             "shared_depth.py", "lsa_seq.py", "phi_source.py", "source.py", "families.py")}
    paths.update({"pmm/" + name: src / "product_model_with_memory" / name for name in
                 ("production_coding.py", "universal_tables.py", "mellin.py", "kernel.py")})
    paths.update({"store/" + name: store / name for name in ("anchors.json", "manifest.json")})
    report = dict(schema="product-backoff-components-validation-v1", passed=True,
                  seed=args.seed, n_ctx=args.n_ctx, coordinate_sizes=[8, 8], source_depth_prior=list(range(1, 9)),
                  source_depths=[int(parameters["ls1"][0]), int(parameters["ls2"][0])],
                  bayes_depths=list(range(1, 9)), posterior_scope="one depth shared by all rows of each coordinate",
                  tokens_sha256=array_hash(tokens[0]), weights_sha256=array_hash(result["weights"]),
                  joint_reconstruction_error=result["reconstruction_error"], coordinate_checks=coordinate_checks,
                  grid=dict(lower=-30, upper=30, step=kernel.h, rmax=kernel.rmax),
                  runtime=dict(python=list(sys.version_info[:3]), numpy=np.__version__,
                               scipy=importlib.metadata.version("scipy")),
                  backend=dict(read_only=True, settings={name: os.environ[name] for name in
                               ("PMM_UNIVERSAL_TABLES", "PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE",
                                "PMM_PHI_SADDLE_MIN_L")}),
                  provenance={name: dict(path=str(path), sha256=sha_file(path)) for name, path in paths.items()},
                  interpretation=result["interpretation"], elapsed_seconds=time.monotonic() - start,
                  limitations=["One long sampled product path; independent small-case tests are separate.",
                               "The decomposition checks predictive identities, not a learned transformer implementation."])
    if any(value["depth_one_closed_form_error"] > 1e-13 for value in coordinate_checks.values()):
        raise FloatingPointError("depth-one closed form check failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(out=str(args.out), passed=True, joint_error=result["reconstruction_error"],
                          coordinate_checks=coordinate_checks, elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()
