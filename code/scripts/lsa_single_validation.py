"""Deterministic single-distribution LSA checks using twelve saved comparisons.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_single_validation.py

No source or count samples are generated. Exact multiplicity covariance is
inserted into the existing finite-depth-mixture curvature calculation. The
concavity lower bound uses profile-polytope vertices and an LP dual. Both
bound endpoints use numerical quadrature, not interval-certified integrals.
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
from scipy.optimize import linprog
from scipy.special import gammaln, logsumexp

from lsa_expected_profile import fractional_cost, count_profile_mean
from lsa_mean_profile_crossing_check import fractional_batch
from lsa_single_covariance import count_multiplicity_moments, self_test
from tilted_row_reference import DEFAULT_LSA, DEFAULT_STORE, configure_store


ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def profile_vertices(d, k):
    """All vertices of v>=0, sum(v)=d, sum(r*v_r)=k."""
    vertices, supports = [], []
    if k % d == 0:
        r = k//d
        vertex = np.zeros(k+1)
        vertex[r] = d
        vertices.append(vertex)
        supports.append([r])
    for r in range(k//d+1):
        if r*d >= k:
            continue
        for s in range(k//d+1, k+1):
            if s*d <= k:
                continue
            vertex = np.zeros(k+1)
            vertex[r] = (s*d-k)/(s-r)
            vertex[s] = (k-r*d)/(s-r)
            vertices.append(vertex)
            supports.append([r, s])
    vertices = np.asarray(vertices)
    assert np.max(abs(vertices.sum(1)-d)) < 1e-10
    assert np.max(abs(vertices @ np.arange(k+1)-k)) < 1e-10
    return vertices, supports


def evaluate_vertices(d, k, cap, tables):
    vertices, supports = profile_vertices(d, k)
    arguments = (vertices, np.full(len(vertices), float(k)),
                 np.full(len(vertices), gammaln(k)), cap, tables)
    costs, one, checks, _ = fractional_batch(*arguments, step=.05)
    refinements = []
    for step in (.025, .0125):
        if checks["maximum_nested_grid_difference_bits"] <= 2e-5:
            break
        refined, refined_one, refined_checks, _ = fractional_batch(*arguments, step=step)
        refinements.append(dict(step=step,
            maximum_cost_change_bits=float(abs(refined-costs).max()),
            maximum_depth_one_change_bits=float(abs(refined_one-one).max())))
        costs, one, checks = refined, refined_one, refined_checks
    assert checks["maximum_nested_grid_difference_bits"] < 2e-4, checks
    closed_one = (gammaln(k+d)-gammaln(d)-vertices @ gammaln(np.arange(k+1)+1))/LN2
    assert float(abs(one-closed_one).max()) < 2e-6
    return dict(d=d, k=k, coding_depth_cap=cap, vertices=vertices,
                supports=supports, costs=costs, depth_one=one,
                depth_one_closed=closed_one, quadrature=checks, refinements=refinements)


def lower_bound(nu, vertex_data):
    """Dual lower certificate for the vertex convex envelope at nu.

    For any feasible lambda, sum(lambda)=1. If y violates any vertex
    inequality by at most e, E c(Phi)>=nu@y-e. The implementation checks
    that inequality at every vertex; numerical vertex integration error
    remains separate from this finite-dimensional certificate.
    """
    d, k = vertex_data["d"], vertex_data["k"]
    vertices, costs = vertex_data["vertices"], vertex_data["costs"]
    if k <= d:
        weights = np.zeros(len(vertices))
        dual = np.zeros(k+1)
        for j, support in enumerate(vertex_data["supports"]):
            r = support[-1]
            weights[j] = r*nu[r]/k
            dual[r] = r*costs[j]/k
        method = "Explicit simplex decomposition and dual"
        solver_message = None
    else:
        solution = linprog(costs, A_eq=vertices.T/d, b_eq=nu/d,
                           bounds=(0., None), method="highs",
                           options=dict(primal_feasibility_tolerance=1e-10,
                                        dual_feasibility_tolerance=1e-10))
        if not solution.success:
            raise RuntimeError(solution.message)
        weights = solution.x
        dual = solution.eqlin.marginals/d
        method = "Vertex LP with explicit dual feasibility correction"
        solver_message = solution.message
    primal_residual = float(abs(weights @ vertices-nu).max())
    weight_sum_error = abs(float(weights.sum())-1.)
    violation = max(0., float(np.max(vertices @ dual-costs)))
    dual_raw = float(nu @ dual)
    lower = dual_raw-violation
    primal = float(weights @ costs)
    assert primal_residual < 1e-7, primal_residual
    assert weight_sum_error < 1e-9, weight_sum_error
    assert float(weights.min()) >= -1e-12
    assert lower <= primal+2e-6, (lower, primal)
    return dict(lower_bits=lower, raw_dual_objective_bits=dual_raw,
                dual_maximum_violation_bits=violation, dual_multipliers=dual.tolist(),
                primal_value_bits=primal, primal_maximum_profile_residual=primal_residual,
                primal_weight_sum_error=weight_sum_error,
                primal_minimum_weight=float(weights.min()),
                primal_minus_corrected_dual_bits=primal-lower,
                method=method, solver_message=solver_message,
                depth_one_value_from_same_weights_bits=float(weights @ vertex_data["depth_one_closed"]))


def direct_trace_correction(nu, covariance, k, cap, tables, step=.025):
    """Use the full covariance in the Hessian trace, without an eigen cutoff."""
    rs = np.arange(k+1)
    grid = np.r_[np.arange(-220., -75., 4*step), np.arange(-75., 35.+step/2, step)]
    widths = np.empty_like(grid)
    widths[1:-1] = (grid[2:]-grid[:-2])/2
    widths[0], widths[-1] = (grid[1]-grid[0])/2, (grid[-1]-grid[-2])/2
    covariance = covariance[2:, 2:]
    logz, first, second = [], [], []
    for depth in range(1, cap+1):
        if depth == 1:
            phi = gammaln(rs[:, None]+1)-(rs[:, None]+1)*np.logaddexp(0., grid)
        else:
            phi = tables.log_phi_matrix(depth, list(map(int, rs)), grid)
        kernel = k*grid+nu @ phi+np.log(widths)
        z = logsumexp(kernel)
        weight = np.exp(kernel-z)
        b = phi[2:]-rs[2:, None]*phi[1]+(rs[2:, None]-1)*phi[0]
        mean = b @ weight
        quadratic = np.einsum("ig,ig->g", b, covariance @ b)
        logz.append(z)
        first.append(mean)
        second.append(float(quadratic @ weight))
    depth_weights = np.exp(np.asarray(logz)-logsumexp(logz))
    mean = depth_weights @ np.asarray(first)
    value = (depth_weights @ np.asarray(second)-mean @ covariance @ mean)/(2*LN2)
    return float(value)


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/mechanism/lsa_expected_profile.json")
    parser.add_argument("--out", type=Path, default=ROOT/"results/mechanism/lsa_single_validation.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    assert source["complete"] and len(source["cells"]) == 12
    cells = source["cells"]
    if args.smoke:
        cells = [cells[0], next(c for c in cells if (c["d"],c["alpha"],c["k"]) == (64, .5, 128))]
    result = dict(schema_version=1, complete=False, smoke=args.smoke,
        source=relative(args.source), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        script_sha256=sha(__file__), command_argv=sys.argv,
        dependencies={name:sha(Path(__file__).with_name(name)) for name in
                      ("lsa_single_covariance.py", "lsa_expected_profile.py", "lsa_mean_profile_crossing_check.py")},
        exhaustive_covariance_validation=self_test(),
        qualifications=[
            "The twelve source probability vectors and measured controlled means/SEs are reused unchanged. No new counts or sources are drawn.",
            "The exact finite coding-depth mixture stays inside the logarithm; true source depth is separate from the coding-depth cap.",
            "The covariance uses finite binomial/multinomial formulas, not sampled counts. Its second-order cost correction is an approximation, not a bound.",
            "The concavity bracket is mathematically valid for exact integrals. Reported endpoints use numerical quadrature; nested-grid differences are not certified integration-error bounds.",
            "The lower endpoint is a dual value reduced by the maximum vertex-constraint violation. The primal LP value is reported only as a solver diagnostic.",
            "The polytope allows fractional multiplicities. Its lower bound can be much farther from the measured cost than the covariance approximation.",
            "The saved controlled SE measures sampling uncertainty at the fixed source p. Source-depth labels do not alter the analyzed LSA code.",
        ], vertex_sets=[], cells=[])
    configure_store(DEFAULT_STORE)
    if str(DEFAULT_LSA) not in sys.path:
        sys.path.insert(0, str(DEFAULT_LSA))
    from lsa.universal_tables import UniversalTables
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    vertex_cache = {}
    try:
        for cell in cells:
            cell_started = time.monotonic()
            d, k, cap = (cell[key] for key in ("d", "k", "coding_depth_cap"))
            p = np.asarray(cell["source_probabilities"])
            moments = count_multiplicity_moments(p, k)
            eigenvalues = np.linalg.eigvalsh(moments.covariance[2:, 2:])
            threshold = max(1e-13, eigenvalues.max(initial=0.)*1e-12)
            eigen_diagnostics = dict(
                symmetry_error=float(abs(moments.covariance-moments.covariance.T).max()),
                fractional_cost_eigen_threshold=float(threshold),
                omitted_positive_eigenvalue_sum=float(eigenvalues[(eigenvalues > 0)&(eigenvalues <= threshold)].sum()),
                discarded_negative_eigenvalue_sum=float(eigenvalues[eigenvalues < 0].sum()),
                retained_covariance_modes=int(np.sum(eigenvalues > threshold)))
            original_mean = count_profile_mean(p, k)
            mean_difference = float(abs(moments.mean-original_mean).max())
            assert mean_difference < 1e-10
            prediction = fractional_cost(moments.mean, k, cap, tables, step=.025,
                                         covariance=moments.covariance)
            if (d, cell["alpha"], k) == (256, .5, 128):
                direct = direct_trace_correction(moments.mean, moments.covariance, k, cap, tables)
                trace_error = abs(direct-prediction["covariance_correction_bits"])
                assert trace_error < 2e-6, trace_error
                eigen_diagnostics.update(full_covariance_direct_trace_correction_bits=direct,
                                         direct_trace_minus_thresholded_absolute_bits=trace_error)
            upper_difference = abs(prediction["jensen_bits"]-cell["prediction"]["jensen_bits"])
            assert upper_difference < 1e-7, upper_difference
            key = (d, k, cap)
            if key not in vertex_cache:
                vertex_cache[key] = evaluate_vertices(d, k, cap, tables)
                data = vertex_cache[key]
                result["vertex_sets"].append(dict(d=d, k=k, coding_depth_cap=cap,
                    number_of_vertices=len(data["vertices"]), supports=data["supports"],
                    vertex_costs_bits=data["costs"].tolist(),
                    depth_one_closed_costs_bits=data["depth_one_closed"].tolist(),
                    quadrature=data["quadrature"], refinements=data["refinements"]))
            bound = lower_bound(moments.mean, vertex_cache[key])
            one = fractional_cost(moments.mean, k, 1, tables, step=.025,
                                  covariance=moments.covariance)
            one_closed = float((gammaln(k+d)-gammaln(d)
                                -moments.mean @ gammaln(np.arange(k+1)+1))/LN2)
            one_error = max(abs(one["jensen_bits"]-one_closed),
                            abs(bound["depth_one_value_from_same_weights_bits"]-one_closed))
            assert one_error < 2e-6, one_error
            assert one["covariance_correction_bits"] < 2e-6, one
            controlled = cell["comparison"]["controlled_mean_bits"]
            controlled_se = cell["comparison"]["controlled_SE_bits"]
            corrected_difference = prediction["corrected_bits"]-controlled
            width = prediction["jensen_bits"]-bound["lower_bits"]
            assert width >= -2e-6
            checked = dict(d=d, alpha=cell["alpha"], source_depth=cell["source_depth"],
                k=k, coding_depth_cap=cap, exact_mean_profile=moments.mean.tolist(),
                covariance_diagnostics=moments.diagnostics,
                covariance_eigenfactor_diagnostics=eigen_diagnostics,
                deterministic_prediction=prediction, concavity_lower=bound,
                concavity_upper_bits=prediction["jensen_bits"], concavity_width_bits=width,
                saved_comparison=cell["comparison"],
                saved_sampled_covariance_prediction=cell["prediction"],
                jensen_minus_controlled_mean_bits=prediction["jensen_bits"]-controlled,
                corrected_minus_controlled_mean_bits=corrected_difference,
                corrected_difference_in_controlled_SEs=corrected_difference/controlled_se,
                exact_minus_sampled_covariance_corrected_cost_bits=prediction["corrected_bits"]-cell["prediction"]["corrected_bits"],
                controlled_mean_minus_lower_bits=controlled-bound["lower_bits"],
                validation=dict(mean_profile_agreement=mean_difference,
                    old_upper_agreement_bits=upper_difference,
                    depth_one_exact_expected_cost_bits=one_closed,
                    maximum_depth_one_collapse_error_bits=one_error,
                    depth_one_covariance_correction_bits=one["covariance_correction_bits"]),
                elapsed_seconds=time.monotonic()-cell_started)
            result["cells"].append(checked)
            result["complete"] = len(result["cells"]) == len(cells)
            result["elapsed_seconds"] = time.monotonic()-started
            assert args.source.read_bytes() == source_bytes, "Saved comparison changed during validation"
            write_json(args.out, result)
            print(json.dumps({key:checked[key] for key in ("d", "alpha", "source_depth", "k", "corrected_minus_controlled_mean_bits", "concavity_width_bits", "elapsed_seconds")}), flush=True)
    finally:
        tables.close()


if __name__ == "__main__":
    main()
