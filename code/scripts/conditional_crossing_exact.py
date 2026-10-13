"""Deterministic short-prefix B/U comparison for the uniform-base LSA source.

The source has fixed depth 2 and d=64. B keeps its training prior, uniform
over one shared depth in 0..8; U keeps the original coding depths 1..20.
X1 is supplied free and fixed to label 0 by symmetry. We enumerate every
restricted-growth equality pattern, with multiplicity (d-1)_(k-1).

A and D below are divergences from the depth-known *marginal sequence law*,
not regrets relative to the realized transition matrix. Their difference G
is exactly the paper's cumulative regret/loss gap. Integrals are numerical;
enumeration introduces no source-sampling error and no probability is
renormalized or pruned.

Set LSA_STORE and PMM_SRC to the existing sealed production store/source.
    python code/scripts/conditional_crossing_exact.py
    python code/scripts/conditional_crossing_exact.py --smoke \
        --out output/conditional_crossing/exact_smoke.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER = ROOT / "code/transformer"
sys.path.insert(0, str(TRANSFORMER))
LN2 = math.log(2.0)
BELLS = (1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975)


def restricted_growth_patterns(n, d):
    """Canonical labelled-sequence orbits, with the first label fixed to 0."""
    if n < 1 or d < 2:
        raise ValueError("n must be positive and d at least two")
    prefix = [0]

    def visit(largest):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for label in range(min(largest + 1, d - 1) + 1):
            prefix.append(label)
            yield from visit(max(largest, label))
            prefix.pop()

    yield from visit(0)


def multiplicity(d, k):
    return math.prod(range(d - k + 1, d))


def partitions(total, largest=None):
    """All positive count profiles, in decreasing order, including empty."""
    if total == 0:
        yield ()
        return
    limit = total if largest is None else min(total, largest)
    for first in range(limit, 0, -1):
        for rest in partitions(total - first, first):
            yield (first,) + rest


def pattern_profiles(pattern):
    counts = Counter(pattern)
    rows = {}
    for a, b in zip(pattern[:-1], pattern[1:]):
        if a not in rows:
            rows[a] = Counter()
        rows[a][b] += 1
    return (tuple(sorted(counts.values(), reverse=True)),
            tuple(tuple(sorted(row.values(), reverse=True)) for row in rows.values()))


def horizon_summary(n, d, profile_evidence, *, source_depth=2,
                    bayes_depths=tuple(range(9)), unigram_depths=tuple(range(1, 21))):
    """Sum complete orbit probabilities using cached per-depth log evidence.

    profile_evidence(profile) returns an array indexed by depth. The generic
    interface also permits independently computed rational evidence in tests.
    """
    source_weights, bayes_weights, unigram_weights = [], [], []
    a_terms, d_terms, g_terms = [], [], []
    by_depth = {}
    unigram = {}
    logd = math.log(d)
    b_index = np.asarray(bayes_depths, dtype=int)
    u_index = np.asarray(unigram_depths, dtype=int)
    for pattern in restricted_growth_patterns(n, d):
        token_profile, rows = pattern_profiles(pattern)
        if token_profile not in unigram:
            values = profile_evidence(token_profile)
            unigram[token_profile] = logd + float(np.logaddexp.reduce(values[u_index])) - math.log(len(u_index))
        logu = unigram[token_profile]
        logs = 0.0
        logb_depths = np.zeros(len(b_index))
        for row in rows:
            if row not in by_depth:
                values = profile_evidence(row)
                by_depth[row] = (float(values[source_depth]), values[b_index])
            source_row, bayes_row = by_depth[row]
            logs += source_row
            logb_depths += bayes_row
        logb = float(np.logaddexp.reduce(logb_depths)) - math.log(len(b_index))
        count = multiplicity(d, len(token_profile))
        weight = count * math.exp(logs)
        if not math.isfinite(weight) or weight <= 0:
            raise FloatingPointError("An orbit probability is nonfinite or underflowed")
        source_weights.append(weight)
        bayes_weights.append(count * math.exp(logb))
        unigram_weights.append(count * math.exp(logu))
        a_terms.append(weight * (logs - logu) / LN2)
        d_terms.append(weight * (logs - logb) / LN2)
        g_terms.append(weight * (logb - logu) / LN2)
    a, divergence, direct = map(math.fsum, (a_terms, d_terms, g_terms))
    return dict(observed_tokens=n, scored_predictions=n - 1,
                equality_patterns=len(source_weights),
                source_probability_mass=math.fsum(source_weights),
                bayes_probability_mass=math.fsum(bayes_weights),
                unigram_probability_mass=math.fsum(unigram_weights),
                smallest_source_orbit_probability=min(source_weights),
                A_marginal_divergence_bits=a, D_marginal_divergence_bits=divergence,
                cumulative_gap_bits=a - divergence,
                direct_cumulative_gap_bits=direct)


class ProfileEvidence:
    """One frozen SeqLSA kernel, cached once per symmetric count profile."""

    def __init__(self, d, max_prefix):
        from lsa_seq import SeqLSA, lmax_of
        from ranked_lsa import RankedLSARowEvidence
        from shared_depth import SharedDepthLSA

        if not os.environ.get("LSA_STORE") or not os.environ.get("PMM_SRC"):
            raise ValueError("Set LSA_STORE and PMM_SRC to the existing sealed production store and source")
        guard = RankedLSARowEvidence(d, [0.0], depths=(2,))
        guard._get_backend()  # The existing guard locks and checks production mode.
        self.d = d
        self.depths = tuple(range(lmax_of(d) + 1))
        self.kernel = SeqLSA(d, depths=self.depths, rmax=max_prefix + 1,
                             u_lo=-36.0, u_hi=36.0)
        self.corrector = SharedDepthLSA(self.kernel)
        self.cache = {}
        self.raw_cache = {}

    def __call__(self, profile):
        if profile not in self.cache:
            table = self.kernel.new_table()
            for label, count in enumerate(profile):
                for _ in range(count):
                    table.observe(label)
            raw = np.zeros(len(self.depths)) if table.k == 0 else table.logQ.copy()
            # Analytic empty/singleton, depth-zero and depth-one anchors are
            # exactly the same priors; no predictor or depth range is changed.
            corrected = self.corrector._correct_evidence(raw, table.m, table.k)
            self.raw_cache[profile] = raw
            self.cache[profile] = corrected
        return self.cache[profile]


def refine_profiles(evidence, profiles):
    """Independently reconstruct every used profile on expanded/finer grids."""
    from validate_shared_depth import BatchEvidence

    expanded = BatchEvidence(evidence.depths, (-42.0, 42.0), 1 / 64)
    refined = BatchEvidence(evidence.depths, (-42.0, 42.0), 1 / 128)
    checks = []
    refined_cache = {}
    for profile in profiles:
        base = evidence(profile)
        counts = np.zeros(evidence.d, dtype=np.int64)
        counts[:len(profile)] = profile
        wider = np.array([expanded.row(counts, depth) for depth in evidence.depths])
        finer = np.array([refined.row(counts, depth) for depth in evidence.depths])
        refined_cache[profile] = finer
        checks.append(dict(profile=list(profile),
                           wider_bounds_error_bits=float(np.max(np.abs(base - wider))) / LN2,
                           finer_grid_error_bits=float(np.max(np.abs(wider - finer))) / LN2))
    return refined_cache, checks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(max_prefix=10):
    if max_prefix < 2 or max_prefix > 10:
        raise ValueError("This bounded calculation supports maximum prefixes 2 through 10")
    started = time.monotonic()
    d = 64
    evidence = ProfileEvidence(d, max_prefix)
    print("One production kernel loaded; caching symmetric profiles", flush=True)
    profiles = [profile for n in range(max_prefix + 1) for profile in partitions(n)]
    for profile in profiles:
        evidence(profile)
    refined, profile_checks = refine_profiles(evidence, profiles)
    print(f"Checked {len(profiles)} profiles on the wider/finer grids", flush=True)
    probability_tolerance = 1e-8
    bit_tolerance = 1e-7
    bound = math.log2(9)
    horizons = []
    checks = []
    first_bound = None
    for n in range(1, max_prefix + 1):
        value = horizon_summary(n, d, evidence)
        finer = horizon_summary(n, d, refined.__getitem__)
        value["refined"] = finer
        for key in ("source_probability_mass", "bayes_probability_mass", "unigram_probability_mass"):
            checks.append(abs(value[key] - 1) <= probability_tolerance)
            checks.append(abs(finer[key] - 1) <= probability_tolerance)
        checks.extend([
            value["equality_patterns"] == BELLS[n],
            value["A_marginal_divergence_bits"] >= -bit_tolerance,
            -bit_tolerance <= value["D_marginal_divergence_bits"] <= bound + bit_tolerance,
            abs(value["cumulative_gap_bits"] - value["direct_cumulative_gap_bits"]) < 1e-12,
        ])
        for key in ("A_marginal_divergence_bits", "D_marginal_divergence_bits", "cumulative_gap_bits"):
            checks.append(abs(value[key] - finer[key]) <= bit_tolerance)
        horizons.append(value)
        print(f"n={n}: patterns={value['equality_patterns']}, A={value['A_marginal_divergence_bits']:.9g}, "
              f"D={value['D_marginal_divergence_bits']:.9g}, G={value['cumulative_gap_bits']:.9g}", flush=True)
        if min(value["A_marginal_divergence_bits"], finer["A_marginal_divergence_bits"]) > bound + bit_tolerance:
            first_bound = n
            break
    instantaneous = [dict(observed_prefix=a["observed_tokens"],
                          next_token=a["observed_tokens"] + 1,
                          gap_bits=b["cumulative_gap_bits"] - a["cumulative_gap_bits"],
                          refined_gap_bits=b["refined"]["cumulative_gap_bits"] - a["refined"]["cumulative_gap_bits"])
                     for a, b in zip(horizons[:-1], horizons[1:])]
    # This uses only the unigram's pair collision moment and exact uniformity
    # of the source's first continuation, independently of orbit enumeration.
    repeat = d * float(np.exp(evidence((2,))[1:]).mean())
    other = (1 - repeat) / (d - 1)
    analytic_g1 = -(math.log2(d * repeat) + (d - 1) * math.log2(d * other)) / d
    checks.extend([repeat > 1 / d,
                   abs(instantaneous[0]["gap_bits"] - analytic_g1) <= bit_tolerance,
                   abs(horizons[1]["D_marginal_divergence_bits"]) <= bit_tolerance,
                   abs(horizons[0]["cumulative_gap_bits"]) <= bit_tolerance])
    checks.extend(c["wider_bounds_error_bits"] <= bit_tolerance and c["finer_grid_error_bits"] <= bit_tolerance
                  for c in profile_checks)
    for previous, current in zip(horizons[:-1], horizons[1:]):
        for key in ("A_marginal_divergence_bits", "D_marginal_divergence_bits"):
            checks.append(current[key] >= previous[key] - bit_tolerance)
    code_paths = [Path(__file__), TRANSFORMER / "lsa_seq.py", TRANSFORMER / "shared_depth.py",
                  TRANSFORMER / "phi_source.py", TRANSFORMER / "ranked_lsa.py",
                  TRANSFORMER / "weighted_lsa.py", TRANSFORMER / "validate_shared_depth.py"]
    store = Path(os.environ["LSA_STORE"]).expanduser().resolve()
    src = Path(os.environ["PMM_SRC"]).expanduser().resolve()
    code_paths.extend([src / "product_model_with_memory/universal_tables.py",
                       src / "product_model_with_memory/production_coding.py"])
    return dict(
        schema="conditional_crossing_exact_prefix_v1", passed=bool(all(checks)),
        parameters=dict(d=d, alpha=0.0, source_depth=2, bayes_depths=list(range(9)),
                        bayes_depth_prior="uniform, one shared depth across all rows",
                        unigram_depths=list(range(1, 21)), unigram_depth_prior="uniform",
                        initial_token="X1 fixed to label 0 by simultaneous label symmetry; supplied free",
                        requested_max_prefix=max_prefix, computed_max_prefix=horizons[-1]["observed_tokens"]),
        method="Complete restricted-growth equality patterns; multiplicity falling factorial (d-1)_(k-1); all positive orbit probabilities retained without normalization",
        quantities=dict(A="KL from fixed-depth marginal sequence law P2 to unigram conditional code U",
                        D="KL from fixed-depth marginal sequence law P2 to family conditional code B",
                        G="A-D, equal to expected cumulative U-minus-B loss and regret",
                        g="G(t+1)-G(t), expected next-token U-minus-B loss after t observed tokens",
                        regret_caveat="A and D omit the common information cost of learning the realized transition matrix; they are not matrix-oracle regrets"),
        numerical_grids=dict(production_bounds=[-36, 36], production_step=1 / 64,
                             expanded_bounds=[-42, 42], expanded_step=1 / 64, refined_step=1 / 128),
        horizons=horizons, instantaneous=instantaneous,
        first_step=dict(unigram_repeat_probability=repeat, unigram_other_probability=other,
                        analytic_gap_bits=analytic_g1, enumeration_gap_bits=instantaneous[0]["gap_bits"]),
        cumulative_bound=dict(depth_mismatch_bound_bits=bound, first_numerically_passing_prefix=first_bound,
                              condition="A_n > log2(9) implies G_m > 0 for all m>=n",
                              qualification="The inequality is mathematical; integral accuracy is supported by finite grid checks, not a rigorous interval bound"),
        validation=dict(checks=len(checks), probability_tolerance=probability_tolerance,
                        bit_tolerance=bit_tolerance, profiles_checked=len(profile_checks),
                        maximum_wider_bounds_error_bits=max(c["wider_bounds_error_bits"] for c in profile_checks),
                        maximum_finer_grid_error_bits=max(c["finer_grid_error_bits"] for c in profile_checks),
                        profile_checks=profile_checks),
        provenance=dict(code_sha256={str(p.resolve()): sha(p) for p in code_paths},
                        store=str(store), store_manifest_sha256=sha(store / "manifest.json"),
                        store_anchors_sha256=sha(store / "anchors.json"),
                        python=sys.version.split()[0], numpy=np.__version__),
        elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-prefix", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/conditional_crossing/exact_prefix.json")
    args = parser.parse_args()
    result = run(min(args.max_prefix, 4) if args.smoke else args.max_prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"],
                      "bound": result["cumulative_bound"], "out": str(args.out)}), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
