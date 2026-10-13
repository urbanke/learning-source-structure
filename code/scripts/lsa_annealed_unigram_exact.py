"""Tiny-case exact reference for the annealed unigram LSA codelength.

The source is the random Markov table from ``mechanisms.tex``.  Each row is

    T_ab = q_b G_ab / sum_c q_c G_ac,

where G_ab is a product of ``source_depth`` unit exponentials.  Integrating a
row gives, for ``k=sum_b r_b>0``, its ordered-sequence probability

    R(r) = prod_b q_b**r_b / Gamma(k)
           * integral_0^inf s**(k-1) prod_b phi[Ls,r_b](s*q_b) ds.

For a transition-count matrix M, start i, and end j, row independence makes
the annealed path mass ``q_i * prod_a R(M[a])``.  This script evaluates the
remaining finite expectation in two independent ways:

1. enumerate all token paths;
2. enumerate transition-count matrices and use the directed matrix-tree /
   Euler-trail multiplicity.

Both routes are exponential and are intentionally restricted to tiny cases.
They provide a sampling-free reference for checking approximations, not a
large-alphabet solver.

Example
-------
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  code/scripts/lsa_annealed_unigram_exact.py --self-test
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

from tilted_row_reference import (
    DEFAULT_LSA,
    DEFAULT_STORE,
    TiltedRowReference,
    configure_store,
    zipf,
)


LN2 = math.log(2.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compositions(total: int, parts: int):
    """Yield weak compositions of ``total`` into ``parts`` entries."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def bareiss_determinant(matrix) -> int:
    """Exact determinant of a small integer matrix."""
    a = [list(map(int, row)) for row in matrix]
    n = len(a)
    if n == 0:
        return 1
    sign = 1
    previous = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            pivot = next((i for i in range(k + 1, n) if a[i][k]), None)
            if pivot is None:
                return 0
            a[k], a[pivot] = a[pivot], a[k]
            sign = -sign
        diagonal = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * diagonal - a[i][k] * a[k][j]
                if numerator % previous:
                    raise ArithmeticError("Bareiss division was not exact")
                a[i][j] = numerator // previous
        previous = diagonal
    return sign * a[-1][-1]


def in_arborescence_count(counts: np.ndarray, root: int, active) -> int:
    """Directed spanning trees whose edges point toward ``root``."""
    active = tuple(sorted(active))
    if len(active) == 1:
        return 1
    keep = [a for a in active if a != root]
    laplacian = []
    for a in keep:
        row = []
        for b in keep:
            if a == b:
                row.append(int(counts[a].sum() - counts[a, a]))
            else:
                row.append(-int(counts[a, b]))
        laplacian.append(row)
    return bareiss_determinant(laplacian)


def euler_trail_multiplicity(counts: np.ndarray, start: int, end: int) -> int:
    """Number of vertex strings with fixed transitions and endpoints.

    Parallel edges with the same ordered endpoints are indistinguishable.
    The formula is the open-trail form of the BEST theorem.  A zero tree
    cofactor also removes disconnected transition matrices.
    """
    counts = np.asarray(counts, dtype=np.int64)
    d = len(counts)
    departures = counts.sum(axis=1)
    arrivals = counts.sum(axis=0)
    required = np.zeros(d, dtype=np.int64)
    required[start] += 1
    required[end] -= 1
    if not np.array_equal(departures - arrivals, required):
        return 0
    active = set(np.flatnonzero(departures + arrivals))
    if not active:
        return int(start == end)
    if start not in active or end not in active:
        return 0
    trees = in_arborescence_count(counts, end, active)
    if trees == 0:
        return 0
    numerator = trees
    for a in active:
        numerator *= math.factorial(int(departures[a]) - 1 + (a == end))
    denominator = math.prod(math.factorial(int(value)) for value in counts.ravel())
    if numerator % denominator:
        raise ArithmeticError("Euler-trail multiplicity was not integral")
    return numerator // denominator


class ExactAnnealedUnigram:
    """Numerical scalar row integrals plus exact finite combinatorics."""

    def __init__(self, d: int, alpha: float, source_depth: int,
                 coding_cap: int, store: Path = DEFAULT_STORE):
        if d < 2 or alpha < 0 or source_depth < 0 or coding_cap < 1:
            raise ValueError("require d>=2, alpha>=0, source_depth>=0, coding_cap>=1")
        self.d = d
        self.q = zipf(d, alpha)
        self.source_depth = source_depth
        self.coding_cap = coding_cap
        self.source = (None if source_depth == 0 else
                       TiltedRowReference(self.q, source_depth, store=store))
        uniform = np.full(d, 1.0 / d)
        self.coding = [TiltedRowReference(uniform, depth, store=store)
                       for depth in range(1, coding_cap + 1)]
        self.row_cache = {(0,) * d: 0.0}
        self.cost_cache = {}

    def close(self):
        if self.source is not None:
            self.source.close()
        for evaluator in self.coding:
            evaluator.close()

    def row_log_moment(self, row_counts) -> float:
        """log E[prod_b T_b**row_counts_b], with the source row integrated."""
        key = tuple(map(int, row_counts))
        if key not in self.row_cache:
            if self.source_depth == 0:
                value = sum(r * math.log(p) for r, p in zip(key, self.q))
            else:
                value, _ = self.source.log_probability(np.asarray(key))
            self.row_cache[key] = float(value)
        return self.row_cache[key]

    def lsa_cost(self, token_counts) -> float:
        """Codelength of the unchanged symmetric coding-depth mixture."""
        # Uniform coding priors make this invariant to token labels.
        key = tuple(sorted(map(int, token_counts), reverse=True))
        if key not in self.cost_cache:
            log_probabilities = []
            argument = np.asarray(key, dtype=np.int64)
            for evaluator in self.coding:
                value, _ = evaluator.log_probability(argument)
                log_probabilities.append(value)
            peak = max(log_probabilities)
            log_mixture = (peak + math.log(math.fsum(
                math.exp(value - peak) for value in log_probabilities))
                - math.log(self.coding_cap))
            self.cost_cache[key] = -log_mixture / LN2
        return self.cost_cache[key]

    def path_log_probability(self, path) -> float:
        matrix = np.zeros((self.d, self.d), dtype=np.int64)
        for a, b in zip(path, path[1:]):
            matrix[a, b] += 1
        return (math.log(self.q[path[0]])
                + math.fsum(self.row_log_moment(row) for row in matrix))

    def direct_paths(self, n: int):
        total_mass = expected_cost = 0.0
        type_counts = Counter()
        for path in itertools.product(range(self.d), repeat=n):
            matrix = np.zeros((self.d, self.d), dtype=np.int64)
            for a, b in zip(path, path[1:]):
                matrix[a, b] += 1
            log_probability = (math.log(self.q[path[0]])
                + math.fsum(self.row_log_moment(row) for row in matrix))
            probability = math.exp(log_probability)
            token_counts = np.bincount(path, minlength=self.d)
            total_mass += probability
            expected_cost += probability * self.lsa_cost(token_counts)
            type_counts[(path[0], path[-1], tuple(map(int, matrix.ravel())))] += 1
        return dict(probability_mass=total_mass, expected_cost_bits=expected_cost,
                    paths=self.d ** n, distinct_types=len(type_counts),
                    type_multiplicities=type_counts)

    def markov_types(self, n: int):
        if n == 1:
            profile = (self.d - 1, 1)
            return dict(probability_mass=1.0,
                        expected_cost_bits=self.lsa_cost((1,) + (0,) * (self.d - 1)),
                        candidate_matrices=1, nonzero_endpoint_types=self.d,
                        profile_mass={profile: 1.0})
        transitions = n - 1
        total_mass = expected_cost = 0.0
        candidates = endpoint_types = 0
        profile_mass = Counter()
        for flat in compositions(transitions, self.d * self.d):
            candidates += 1
            matrix = np.asarray(flat, dtype=np.int64).reshape(self.d, self.d)
            row_log_probability = math.fsum(
                self.row_log_moment(row) for row in matrix)
            arrivals = matrix.sum(axis=0)
            for start in range(self.d):
                for end in range(self.d):
                    multiplicity = euler_trail_multiplicity(matrix, start, end)
                    if multiplicity == 0:
                        continue
                    endpoint_types += 1
                    log_probability = (math.log(self.q[start])
                                       + math.log(multiplicity)
                                       + row_log_probability)
                    probability = math.exp(log_probability)
                    token_counts = arrivals.copy()
                    token_counts[start] += 1
                    profile = tuple(map(int, np.bincount(token_counts, minlength=n + 1)))
                    total_mass += probability
                    expected_cost += probability * self.lsa_cost(token_counts)
                    profile_mass[profile] += probability
        return dict(probability_mass=total_mass, expected_cost_bits=expected_cost,
                    candidate_matrices=candidates,
                    nonzero_endpoint_types=endpoint_types,
                    profile_mass=profile_mass)

    def iid_check(self, n: int) -> float:
        """Exact iid-q expectation, used only when source_depth is zero."""
        result = 0.0
        for counts in compositions(n, self.d):
            multiplicity = math.factorial(n) / math.prod(math.factorial(k) for k in counts)
            probability = multiplicity * math.prod(p ** k for p, k in zip(self.q, counts))
            result += probability * self.lsa_cost(counts)
        return result

    def uniform_dirichlet_row_error(self, max_total: int) -> float:
        """Check q uniform, source depth one against Dirichlet(1,...,1)."""
        if self.source_depth != 1 or not np.allclose(self.q, 1.0 / self.d):
            raise ValueError("this check requires alpha=0 and source_depth=1")
        maximum = 0.0
        for total in range(max_total + 1):
            for counts in compositions(total, self.d):
                got = self.row_log_moment(counts)
                expected = (math.lgamma(self.d)
                            + math.fsum(math.lgamma(k + 1) for k in counts)
                            - math.lgamma(self.d + total))
                maximum = max(maximum, abs(got - expected) / LN2)
        return maximum


def profile_curvature_approximation(profile_mass, n: int, coding_cap: int,
                                    store: Path):
    """Apply the existing mean-profile/Hessian formula to an exact count law."""
    mass = math.fsum(profile_mass.values())
    profiles = [(weight / mass, np.asarray(profile, dtype=float))
                for profile, weight in profile_mass.items()]
    mean = sum(weight * profile for weight, profile in profiles)
    covariance = sum(weight * np.outer(profile - mean, profile - mean)
                     for weight, profile in profiles)
    configure_store(store)
    if str(DEFAULT_LSA) not in sys.path:
        sys.path.insert(0, str(DEFAULT_LSA))
    from lsa.universal_tables import UniversalTables
    from lsa_expected_profile import fractional_cost
    from lsa_single_validation import direct_trace_correction

    tables = UniversalTables(store, read_only=True)
    try:
        upper = fractional_cost(mean, n, coding_cap, tables)
        correction = direct_trace_correction(mean, covariance, n, coding_cap, tables)
    finally:
        tables.close()
    return dict(jensen_bits=upper["jensen_bits"],
                covariance_correction_bits=correction,
                corrected_bits=upper["jensen_bits"] - correction,
                exact_profile_probability_mass=mass,
                profile_count=len(profiles),
                mean_profile=mean.tolist())


def evaluate_case(d: int, alpha: float, source_depth: int, n: int,
                  coding_cap: int, store: Path):
    if n < 1:
        raise ValueError("require n>=1")
    evaluator = ExactAnnealedUnigram(d, alpha, source_depth, coding_cap, store)
    try:
        direct = evaluator.direct_paths(n)
        grouped = evaluator.markov_types(n)
        profile_approximation = profile_curvature_approximation(
            grouped["profile_mass"], n, coding_cap, store)
        result = dict(d=d, alpha=alpha, source_depth=source_depth, n=n,
                      coding_cap=coding_cap, direct=direct, markov_types=grouped,
                      mass_difference=grouped["probability_mass"]-direct["probability_mass"],
                      cost_difference_bits=(grouped["expected_cost_bits"]
                                            - direct["expected_cost_bits"]),
                      direct_mass_error=direct["probability_mass"]-1.0,
                      grouped_mass_error=grouped["probability_mass"]-1.0,
                      row_moments_evaluated=len(evaluator.row_cache),
                      lsa_profiles_evaluated=len(evaluator.cost_cache),
                      profile_curvature_approximation=profile_approximation,
                      profile_corrected_minus_exact_bits=(
                          profile_approximation["corrected_bits"]
                          - grouped["expected_cost_bits"]))
        maximum_type_error = 0
        for start, end, flat in direct["type_multiplicities"]:
            matrix = np.asarray(flat, dtype=np.int64).reshape(d, d)
            maximum_type_error = max(maximum_type_error, abs(
                euler_trail_multiplicity(matrix, start, end)
                - direct["type_multiplicities"][(start, end, flat)]))
        result["maximum_euler_multiplicity_error"] = maximum_type_error
        # Counter keys are useful internally but not JSON serializable.
        del result["direct"]["type_multiplicities"]
        del result["markov_types"]["profile_mass"]
        if source_depth == 0:
            iid = evaluator.iid_check(n)
            result["iid_count_expectation_bits"] = iid
            result["iid_minus_markov_type_bits"] = iid - grouped["expected_cost_bits"]
        if source_depth == 1 and alpha == 0:
            result["maximum_uniform_dirichlet_row_error_bits"] = (
                evaluator.uniform_dirichlet_row_error(n - 1))
        return result
    finally:
        evaluator.close()


def self_test(store: Path):
    cases = [
        dict(d=2, alpha=0.7, source_depth=0, n=6, coding_cap=2),
        dict(d=2, alpha=0.0, source_depth=1, n=6, coding_cap=2),
        dict(d=3, alpha=0.8, source_depth=2, n=5, coding_cap=3),
        dict(d=3, alpha=1.2, source_depth=3, n=6, coding_cap=4),
    ]
    results = [evaluate_case(store=store, **case) for case in cases]
    configure_store(store)
    if str(DEFAULT_LSA) not in sys.path:
        sys.path.insert(0, str(DEFAULT_LSA))
    from lsa.universal_tables import UniversalTables
    from unigram_analytic_approximation import evaluate as fast_evaluate

    tables = UniversalTables(store, read_only=True)
    try:
        for row in results:
            fast, _ = fast_evaluate(
                row["d"], row["alpha"], row["source_depth"], row["n"],
                row["coding_cap"], tables, source_step=1 / 32, cost_step=0.04)
            exact = row["markov_types"]["expected_cost_bits"]
            row["fast_approximation"] = fast
            row["fast_minus_exact_bits"] = (
                fast["predicted_codelength_bits"] - exact)
            row["fast_absolute_relative_error"] = (
                abs(row["fast_minus_exact_bits"]) / exact)
    finally:
        tables.close()
    maximum_mass_error = max(abs(row["grouped_mass_error"]) for row in results)
    maximum_cost_error = max(abs(row["cost_difference_bits"]) for row in results)
    maximum_euler_error = max(row["maximum_euler_multiplicity_error"] for row in results)
    maximum_iid_error = max(abs(row.get("iid_minus_markov_type_bits", 0.0))
                            for row in results)
    maximum_dirichlet_error = max(row.get("maximum_uniform_dirichlet_row_error_bits", 0.0)
                                  for row in results)
    maximum_profile_error = max(abs(row["profile_corrected_minus_exact_bits"])
                                for row in results)
    maximum_fast_relative_error = max(
        row["fast_absolute_relative_error"] for row in results)
    if maximum_mass_error > 2e-7 or maximum_cost_error > 2e-7:
        raise AssertionError((maximum_mass_error, maximum_cost_error))
    if maximum_euler_error != 0 or maximum_iid_error > 2e-7:
        raise AssertionError((maximum_euler_error, maximum_iid_error))
    if maximum_dirichlet_error > 2e-7:
        raise AssertionError(maximum_dirichlet_error)
    if maximum_profile_error > 2e-3:
        raise AssertionError(maximum_profile_error)
    if maximum_fast_relative_error > 0.06:
        raise AssertionError(maximum_fast_relative_error)
    return dict(schema_version=1, complete=True,
                target=("Expected codelength of the actual unigram LSA "
                        "predictor under the paper's random Markov source."),
                method=("Exact path enumeration versus the independent "
                        "Markov-type formula, plus exact-profile and fast "
                        "analytic approximations."),
                sampling="No source table, path, or count sample is drawn.",
                argv=sys.argv, script_sha256=sha256(Path(__file__)),
                cases=results, maximum_grouped_mass_error=maximum_mass_error,
                maximum_path_vs_type_cost_difference_bits=maximum_cost_error,
                maximum_euler_multiplicity_error=maximum_euler_error,
                maximum_iid_check_difference_bits=maximum_iid_error,
                maximum_uniform_dirichlet_row_error_bits=maximum_dirichlet_error,
                maximum_profile_curvature_error_bits=maximum_profile_error,
                maximum_fast_absolute_relative_error=maximum_fast_relative_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--source-depth", type=int, default=1)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--coding-cap", type=int, default=2)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test(args.store)
    else:
        result = evaluate_case(args.d, args.alpha, args.source_depth, args.n,
                               args.coding_cap, args.store)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered + "\n")
        temporary.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
