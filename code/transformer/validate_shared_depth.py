"""Independent batch and quadrature checks for shared-depth LSA references.

Requires the read-only anchor store via LSA_STORE and PMM_SRC. Example:
    python validate_shared_depth.py --smoke --out /tmp/shared_depth_smoke.json

Batch evidence is rebuilt from count multiplicities and queried phi rows,
not from the sequential integrand updates. Refinement queries only the
needed count rows, so it does not allocate a large refined SeqLSA kernel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, logsumexp

import phi_source
from lsa_seq import SeqLSA
from shared_depth import SharedDepthLSA
from source import Source

LN2 = np.log(2.0)


class BatchEvidence:
    """Sparse phi queries and independently reconstructed row integrals."""

    def __init__(self, depths, bounds, step):
        previous_step = phi_source.H
        try:
            phi_source.H = step
            self.tables = phi_source.open_tables(max(depths), u_lo=bounds[0], u_hi=bounds[1])
        finally:
            phi_source.H = previous_step
        self.depths = tuple(depths)
        self.cache = {}

    def _rows(self, depth, counts):
        missing = [int(r) for r in counts if (depth, int(r)) not in self.cache]
        if missing:
            rows = self.tables.rows(depth, missing)
            for r, row in zip(missing, rows):
                self.cache[(depth, r)] = row
        return np.stack([self.cache[(depth, int(r))] for r in counts])

    def row(self, counts, depth, exact_cases=True):
        counts = np.asarray(counts, dtype=np.int64)
        d = len(counts)
        k = int(counts.sum())
        if k == 0:
            return 0.0
        if exact_cases:
            if k == 1:
                return -np.log(d)
            if depth == 0:
                return -k * np.log(d)
            if depth == 1:
                return float(gammaln(d) - gammaln(d + k) + gammaln(counts + 1).sum())
        unique, multiplicity = np.unique(counts, return_counts=True)
        integrand = k * self.tables.u + multiplicity @ self._rows(depth, unique)
        return float(logsumexp(integrand) + np.log(self.tables.h) - gammaln(k))

    def chain(self, tokens, d):
        """Return log joint evidence and posterior after the entire prefix."""
        if not len(tokens):
            return 0.0, np.full(len(self.depths), 1.0 / len(self.depths))
        counts = np.zeros((d, d), dtype=np.int64)
        np.add.at(counts, (tokens[:-1], tokens[1:]), 1)
        nonempty = counts[counts.sum(axis=1) > 0]
        log_by_depth = np.array([-np.log(d) + sum(self.row(row, depth) for row in nonempty)
                                 for depth in self.depths])
        normalizer = logsumexp(log_by_depth)
        return float(normalizer - np.log(len(self.depths))), np.exp(log_by_depth - normalizer)


def _update_worst(summary, name, value, case):
    if name not in summary or abs(value) > summary[name]["max_abs"]:
        summary[name] = {"max_abs": float(abs(value)), "signed_value": float(value), "case": case}


def _profiles(d, totals):
    seen = set()
    for total in totals:
        balanced = np.full(d, total // d, dtype=np.int64)
        balanced[:total % d] += 1
        concentrated = np.zeros(d, dtype=np.int64)
        concentrated[0] = total
        mixed = np.zeros(d, dtype=np.int64)
        mixed[0] = (3 * total) // 4
        remaining = total - mixed[0]
        mixed[1:] = remaining // (d - 1)
        mixed[1:1 + remaining % (d - 1)] += 1
        for name, counts in [("balanced", balanced), ("concentrated", concentrated), ("mixed", mixed)]:
            key = tuple(counts)
            if key not in seen:
                seen.add(key)
                yield name, counts


def validate(smoke=False, seed=17092026, bit_tolerance=1e-6, probability_tolerance=1e-8):
    if not os.environ.get("LSA_STORE") or not os.environ.get("PMM_SRC"):
        raise ValueError("set LSA_STORE and PMM_SRC to the read-only anchor store and package source")
    started = time.monotonic()
    depths = list(range(9))
    sizes = [8] if smoke else [8, 64]
    totals = [1, 8, 64] if smoke else [1, 8, 64, 1024]
    production = BatchEvidence(depths, (-30, 30), 1 / 64)
    expanded = BatchEvidence(depths, (-36, 36), 1 / 64)
    refined = BatchEvidence(depths, (-36, 36), 1 / 128)
    refinement = {}
    cases = 0
    for d in sizes:
        for profile, counts in _profiles(d, totals):
            total = int(counts.sum())
            for depth in depths:
                case = {"d": d, "total": total, "profile": profile, "depth": depth}
                a = production.row(counts, depth, exact_cases=False)
                b = expanded.row(counts, depth, exact_cases=False)
                c = refined.row(counts, depth, exact_cases=False)
                _update_worst(refinement, "bounds_error_bits", (a - b) / LN2, case)
                _update_worst(refinement, "grid_error_bits", (b - c) / LN2, case)
                if depth in (0, 1) or total == 1:
                    exact = production.row(counts, depth, exact_cases=True)
                    _update_worst(refinement, "analytic_error_bits", (a - exact) / LN2, case)
                cases += 1

    sequential = {}
    prefix_checks = 0
    for d in sizes:
        n = 33 if smoke else 129
        src = Source(d, n, alpha=0.0, ls=4, permute=False)
        tokens, _ = src.batch(1, np.random.default_rng(seed + d))
        x = tokens[0]
        counts = np.zeros((d, d), dtype=int)
        np.add.at(counts, (x[:-1], x[1:]), 1)
        # SeqLSA uses rmax for both individual counts and its Gamma(k)
        # table, so it must cover the largest row TOTAL as well.
        kernel = SeqLSA(d, depths=depths, rmax=int(counts.sum(axis=1).max()) + 2, u_lo=-30, u_hi=30)
        reference = SharedDepthLSA(kernel)
        positions = sorted(set([0, 1, 2, 3, 8, 16, n // 2, n - 1]))
        result = reference.run(x, full_probs=True, positions=positions, depth_posteriors=True)
        for row, t in enumerate(positions):
            before, posterior = production.chain(x[:t], d)
            after, _ = production.chain(x[:t + 1], d)
            case = {"d": d, "prefix_length": t}
            _update_worst(sequential, "joint_error_bits", (result["log_evidence"][t] - after) / LN2, case)
            _update_worst(sequential, "predictive_error_bits", result["loss"][t] + (after - before) / LN2, case)
            _update_worst(sequential, "posterior_error", float(np.max(np.abs(result["depth_posterior"][t] - posterior))), case)
            _update_worst(sequential, "probability_sum_error", result["prob_sum_error"][row], case)
            _update_worst(sequential, "observed_probability_error",
                          result["probs"][row, x[t]] - np.exp(after - before), case)
            prefix_checks += 1
    passed = all(item["max_abs"] <= bit_tolerance for item in refinement.values())
    for name, item in sequential.items():
        tolerance = bit_tolerance if name.endswith("bits") else probability_tolerance
        passed = passed and item["max_abs"] <= tolerance
    return {"schema": "shared_depth_numerical_validation_v1", "passed": bool(passed),
            "smoke": smoke, "seed": seed, "depths": depths, "alphabet_sizes": sizes,
            "profile_totals": totals, "row_cases": cases, "prefix_checks": prefix_checks,
            "production_bounds": [-30, 30], "expanded_bounds": [-36, 36],
            "production_step": 1 / 64, "refined_step": 1 / 128,
            "tolerances": {"bits": bit_tolerance, "probabilities": probability_tolerance},
            "row_refinement": refinement, "batch_vs_sequential": sequential,
            "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=17092026)
    parser.add_argument("--bit-tolerance", type=float, default=1e-6)
    parser.add_argument("--probability-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.bit_tolerance <= 0 or args.probability_tolerance <= 0:
        parser.error("tolerances must be positive")
    result = validate(args.smoke, args.seed, args.bit_tolerance, args.probability_tolerance)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "row_cases": result["row_cases"],
                      "prefix_checks": result["prefix_checks"], "elapsed_seconds": result["elapsed_seconds"],
                      "out": str(output)}, allow_nan=False), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
