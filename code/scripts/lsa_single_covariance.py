"""Exact mean and covariance of iid count multiplicities, without simulation.

Phi[r] is the number of alphabet symbols occurring exactly r times among k
iid draws from p. All arrays include r=0. The formulas are finite binomial
and multinomial sums, evaluated in floating point; no probabilities are
truncated. Only numpy and scipy are required.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_single_covariance.py --self-test --benchmark
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time

import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import binom


@dataclass
class MultiplicityMoments:
    mean: np.ndarray
    covariance: np.ndarray
    second_moment: np.ndarray
    diagnostics: dict


def count_multiplicity_moments(p, k: int, *, memory_mb: float = 32.) -> MultiplicityMoments:
    """Return E[Phi], Cov(Phi), and E[Phi Phi^T] for arbitrary finite p.

    Zero probabilities are allowed. The input must sum to one within 1e-12;
    it is then normalized to remove floating-point summation error. Work
    arrays for the multinomial kernels use at most approximately memory_mb
    MiB, in addition to O(d^2+k^2) bookkeeping and output arrays.

    With b_a(r)=P{Bin(k,p_a)=r}, the exact identities are

      E Phi_r = sum_a b_a(r),
      E Phi_r Phi_s = 1{r=s} E Phi_r + sum_{a!=b} j_ab(r,s),

    where j_ab(r,s)=k! p_a^r p_b^s (1-p_a-p_b)^(k-r-s)
    /(r! s! (k-r-s)!) if r+s<=k, and is zero otherwise.

    The two fixed-sum identities remove Phi_0 and one positive coordinate
    from the covariance calculation. This reduces cancellation when p is
    concentrated. The returned covariance is not clipped or projected.
    """
    started = time.perf_counter()
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("p must be a nonempty, finite, nonnegative probability vector")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if not math.isfinite(memory_mb) or memory_mb < 1:
        raise ValueError("memory_mb must be finite and at least one")
    total = math.fsum(map(float, p))
    if not math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("p must sum to one")
    d = len(p)
    p = np.sort(p[p > 0]/total)[::-1]
    positive_symbols = len(p)
    r_all = np.arange(k+1)
    means = binom.pmf(r_all[:, None], k, p[None, :]).sum(1)
    means[0] += d-positive_symbols
    covariance = np.zeros((k+1, k+1))
    pivot = int(1+np.argmax(means[1:])) if k else None
    groups = 0
    maximum_kernel_bytes = 0
    batch_size = 0
    if k >= 2 and positive_symbols >= 2:
        core = np.asarray([r for r in range(1, k+1) if r != pivot])
        rows, columns = np.meshgrid(np.arange(len(core)), np.arange(len(core)), indexing="ij")
        valid = core[rows]+core[columns] <= k
        rows, columns = rows[valid], columns[valid]
        r, s = core[rows], core[columns]
        remaining = k-r-s
        log_coefficient = gammaln(k+1)-gammaln(r+1)-gammaln(s+1)-gammaln(remaining+1)

        a_index, b_index = np.triu_indices(positive_symbols, 1)
        pa, pb = p[a_index], p[b_index]
        # Subtracting the two largest probabilities from one can erase a
        # very small but nonzero remainder. Sum that remainder directly.
        remainder = math.fsum(map(float, p))-pa-pb
        includes_largest = a_index == 0
        remainder[includes_largest] = math.fsum(map(float, p[1:]))-pb[includes_largest]
        largest_two = includes_largest & (b_index == 1)
        remainder[largest_two] = math.fsum(map(float, p[2:]))
        assert np.all(remainder >= 0)
        pairs, first, multiplicity = np.unique(
            np.column_stack((pa, pb)), axis=0, return_index=True, return_counts=True)
        remainder = remainder[first]
        groups = len(pairs)
        # An unordered symbol pair contributes J plus its transpose. Equal
        # probabilities can be grouped exactly, including the uniform case.
        joint_one_order = np.zeros((len(core), len(core)))
        accumulated = np.zeros(len(r), dtype=np.longdouble)
        bytes_per_pair = max(1, 4*len(r)*np.dtype(float).itemsize)
        batch_size = max(1, min(groups, int(memory_mb*1024**2)//bytes_per_pair))
        for start in range(0, groups, batch_size):
            stop = min(groups, start+batch_size)
            log_prob = log_coefficient[None, :]+np.log(pairs[start:stop, 0, None])*r
            log_prob += np.log(pairs[start:stop, 1, None])*s
            log_prob += xlogy(remaining[None, :], remainder[start:stop, None])
            np.exp(log_prob, out=log_prob)
            maximum_kernel_bytes = max(maximum_kernel_bytes, log_prob.nbytes)
            accumulated += np.asarray(multiplicity[start:stop] @ log_prob, dtype=np.longdouble)
        joint_one_order[rows, columns] = accumulated
        core_means = np.asarray(means[core], dtype=np.longdouble)
        second_core = np.asarray(joint_one_order, dtype=np.longdouble)
        second_core = second_core+second_core.T+np.diag(core_means)
        core_covariance = second_core-core_means[:, None]*core_means[None, :]

        # Phi_pivot=(k-sum_{r in core}r Phi_r)/pivot, and Phi_0 is
        # d minus all positive multiplicities. These are exact identities.
        transform = np.zeros((k+1, len(core)), dtype=np.longdouble)
        transform[core, np.arange(len(core))] = 1
        transform[pivot] = -core/float(pivot)
        transform[0] = core/float(pivot)-1
        covariance = np.asarray(transform @ core_covariance @ transform.T, dtype=float)
        covariance = (covariance+covariance.T)/2

    second = covariance+means[:, None]*means[None, :]
    null_mass = float(abs(covariance.sum(1)).max())
    null_count = float(abs(covariance @ r_all).max())
    diagnostics = dict(
        d=d, k=int(k), positive_symbols=positive_symbols,
        covariance_pivot=pivot, distinct_probability_pairs=groups,
        pair_batch_size=batch_size, largest_kernel_array_bytes=maximum_kernel_bytes,
        kernel_working_budget_bytes=int(memory_mb*1024**2),
        input_probability_sum=total,
        mean_symbol_sum_error=float(abs(means.sum()-d)),
        mean_sample_sum_error=float(abs(means @ r_all-k)),
        covariance_symbol_null_error=null_mass,
        covariance_sample_null_error=null_count,
        minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
        elapsed_seconds=time.perf_counter()-started,
        qualification="Finite exact formulas evaluated in floating point; no simulation, tail truncation, or covariance projection.",
    )
    return MultiplicityMoments(means, covariance, second, diagnostics)


def compositions(k, d):
    if d == 1:
        yield (k,)
    else:
        for first in range(k+1):
            for rest in compositions(k-first, d-1):
                yield (first,)+rest


def enumerate_moments(p, k):
    """Independent exhaustive multinomial count-vector calculation."""
    records = []
    for counts in compositions(k, len(p)):
        coefficient = math.factorial(k)/math.prod(math.factorial(r) for r in counts)
        probability = coefficient*math.prod(float(a)**r for a, r in zip(p, counts))
        phi = np.bincount(counts, minlength=k+1).astype(float)
        records.append((probability, phi))
    mean = np.asarray([math.fsum(probability*phi[r] for probability, phi in records) for r in range(k+1)])
    covariance = sum(probability*np.outer(phi-mean, phi-mean) for probability, phi in records)
    second = sum(probability*np.outer(phi, phi) for probability, phi in records)
    return mean, covariance, second, math.fsum(probability for probability, _ in records), len(records)


def self_test():
    cases = [[1.], [1., 0., 0.], [.5, .5], [.6, .3, .1],
             [.999999, .0000008, .0000002], [.4, .3, .2, .1], [.8, 0., .2]]
    maximum_mean = maximum_covariance = maximum_second = maximum_mass = 0.
    count_vectors = 0
    for p in cases:
        for k in range(9):
            actual = count_multiplicity_moments(p, k)
            mean, covariance, second, mass, count = enumerate_moments(p, k)
            maximum_mean = max(maximum_mean, float(abs(actual.mean-mean).max()))
            maximum_covariance = max(maximum_covariance, float(abs(actual.covariance-covariance).max()))
            maximum_second = max(maximum_second, float(abs(actual.second_moment-second).max()))
            maximum_mass = max(maximum_mass, abs(mass-1))
            count_vectors += count
    # A two-symbol skewed case at the target sample size remains enumerable.
    p = [1.-1e-8, 1e-8]
    actual = count_multiplicity_moments(p, 128)
    mean, covariance, second, mass, count = enumerate_moments(p, 128)
    maximum_mean = max(maximum_mean, float(abs(actual.mean-mean).max()))
    maximum_covariance = max(maximum_covariance, float(abs(actual.covariance-covariance).max()))
    maximum_second = max(maximum_second, float(abs(actual.second_moment-second).max()))
    maximum_mass = max(maximum_mass, abs(mass-1))
    count_vectors += count
    assert maximum_mean < 1e-11, maximum_mean
    assert maximum_covariance < 1e-10, maximum_covariance
    assert maximum_second < 1e-10, maximum_second
    assert maximum_mass < 1e-12, maximum_mass
    return dict(status="passed", small_cases=len(cases)*9,
                extra_case="d=2, k=128, p=(1-1e-8,1e-8)", enumerated_count_vectors=count_vectors,
                maximum_mean_absolute_error=maximum_mean,
                maximum_covariance_absolute_error=maximum_covariance,
                maximum_second_moment_absolute_error=maximum_second,
                maximum_probability_mass_error=maximum_mass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--memory-mb", type=float, default=32.)
    args = parser.parse_args()
    result = {}
    if args.self_test or not args.benchmark:
        result["exhaustive_validation"] = self_test()
    if args.benchmark:
        result["benchmarks"] = []
        for alpha in (0., .75, 2.):
            p = np.arange(1, 257, dtype=float)**(-alpha)
            p /= p.sum()
            actual = count_multiplicity_moments(p, 128, memory_mb=args.memory_mb)
            result["benchmarks"].append(dict(alpha=alpha, **actual.diagnostics))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
