"""Finite count-based comparison of the deployed symmetric mechanism codes.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/crossing_count_prediction.py

Default: a bounded two-table pilot at d=64,256, alpha=1, source depth3.
The archived crossing selects the evaluation window only. No coefficient
is fitted, no chain is generated, and two draws do not estimate a crossing
precisely. Each checkpoint has an exact multinomial stationary-pair count
law; increments couple the checkpoints without changing that marginal law.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from context_stationary_check import stationary_checked
from takeover_experiment import engage_anchor_mode
from tilted_row_reference import DEFAULT_STORE


ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def moments(values):
    return dict(mean=np.mean(values, axis=0).tolist(),
                SE=(np.std(values, axis=0, ddof=1)/math.sqrt(len(values))).tolist())


def independent_l1_bits(profile, d):
    return (math.lgamma(d+sum(profile))-math.lgamma(d)
            -math.fsum(math.lgamma(c+1) for c in profile))/LN2


def run_cell(args, d):
    from lsa.codelength import default_l_max, depth_averaged_codelength_profiles, profile_of

    started = time.monotonic()
    L, alpha = args.L, args.alpha
    archive_path = args.archive/f"d{d}_a{alpha}_L{L}"/f"summary_d{d}_a{alpha}_L{L}.json"
    archive = json.loads(archive_path.read_text())
    assert (archive["d"], archive["alpha"], archive["L"]) == (d, alpha, L)
    center = archive["mean_lambda_crossing"]
    assert center is not None and math.isfinite(center) and center > 1
    ns = np.array(sorted({max(2, int(round(center*f))) for f in args.factors}), dtype=np.int64)
    ts = ns-1
    q = np.arange(1, d+1, dtype=float)**(-alpha)
    q /= q.sum()
    row_profiles, frequency_profiles = [], []
    all_profiles = {(), (1,), (2,), (1, 1)}
    saved_counts = np.zeros((args.reps, len(ts), d, d), dtype=np.int64)
    frequency_counts = np.zeros((args.reps, len(ts), d), dtype=np.int64)
    matrices = np.zeros((args.reps, d, d))
    stationary = np.zeros((args.reps, d))
    S = np.zeros((args.reps, d))
    initials = np.zeros(args.reps, dtype=np.int64)
    diagnostics = []
    for rep in range(args.reps):
        source_rng = np.random.default_rng([args.seed, d, L, round(alpha*100), rep, 77])
        count_rng = np.random.default_rng([args.seed, d, L, round(alpha*100), rep, 78])
        G = source_rng.exponential(size=(d, d, L)).prod(axis=2)
        normalizer = G@q
        T = G*q[None, :]/normalizer[:, None]
        pi, diag = stationary_checked(T)
        diagnostics.append(diag)
        assert np.all(T > 0) and abs(T.sum(axis=1)-1).max() < 1e-12
        matrices[rep], stationary[rep], S[rep] = T, pi, normalizer
        pair_law = (pi[:, None]*T).ravel()
        pair_law /= pair_law.sum()
        initial = int(count_rng.choice(d, p=q))
        initials[rep] = initial
        counts = np.zeros((d, d), dtype=np.int64)
        previous = 0
        fp_rep, rp_rep = [], []
        for j, (n, t) in enumerate(zip(ns, ts)):
            counts += count_rng.multinomial(int(t-previous), pair_law).reshape(d, d)
            previous = int(t)
            fcounts = counts.sum(axis=0)
            fcounts[initial] += 1
            assert counts.sum() == t and fcounts.sum() == n
            saved_counts[rep, j] = counts
            frequency_counts[rep, j] = fcounts
            fp = profile_of(fcounts)
            row_mult = Counter(profile_of(row) for row in counts if row.sum())
            fp_rep.append(fp)
            rp_rep.append(row_mult)
            all_profiles.add(fp)
            all_profiles.update(row_mult)
        frequency_profiles.append(fp_rep)
        row_profiles.append(rp_rep)
    preparation_seconds = time.monotonic()-started
    print(f"d={d}: prepared {args.reps} tables, {len(ns)} lengths, {len(all_profiles)} unique profiles in {preparation_seconds:.3f}s; evaluating depths1..{default_l_max(d)}", flush=True)
    evaluation_started = time.monotonic()
    evaluated = depth_averaged_codelength_profiles(
        {p:p for p in all_profiles}, d=d, l_max=default_l_max(d), jobs=args.jobs,
        tables_source="universal", universal_path=args.store)
    evaluation_seconds = time.monotonic()-evaluation_started
    # Check normalization using the production mixture, then every profile's
    # depth-one component against an independent gamma-function expression.
    zero_error = abs(evaluated[()].log2_q_avg)
    single_error = abs(-evaluated[(1,)].log2_q_avg-math.log2(d))
    two_mass = d*2**evaluated[(2,)].log2_q_avg+d*(d-1)*2**evaluated[(1,1)].log2_q_avg
    l1_error = max(abs(-value.log2_q_by_depth[0]-independent_l1_bits(p,d))
                   for p,value in evaluated.items())
    assert zero_error < 1e-7 and single_error < 1e-7 and abs(two_mass-1) < 1e-7
    assert l1_error < 1e-7
    f = np.zeros((args.reps, len(ns)))
    c = np.zeros_like(f)
    for rep in range(args.reps):
        for j in range(len(ns)):
            f[rep,j] = -evaluated[frequency_profiles[rep][j]].log2_q_avg
            c[rep,j] = math.log2(d)-math.fsum(mult*evaluated[p].log2_q_avg
                                            for p,mult in row_profiles[rep][j].items())
    gap = f-c
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir/f"d{d}_a{alpha:g}_L{L}.npz"
    np.savez_compressed(raw_path, transition_matrices=matrices,
        stationary_laws=stationary, normalized_row_sums=S, initial_states=initials,
        transition_counts=saved_counts, frequency_counts=frequency_counts,
        frequency_codelength=f, context_codelength=c, gap=gap,
        q=q, tokens=ns, transitions=ts)
    result = dict(d=d, alpha=alpha, L=L, repetitions=args.reps, seed=args.seed,
        l_max=default_l_max(d), tokens=ns.tolist(), transitions=ts.tolist(),
        archived_mean_curve_crossing=center, archive_path=relative(archive_path),
        archive_sha256=sha(archive_path), grid_factors=args.factors,
        frequency_codelength=moments(f), context_codelength=moments(c), gap=moments(gap),
        se_ddof=1, raw_path=relative(raw_path), raw_sha256=sha(raw_path),
        profile_count=len(all_profiles), preparation_seconds=preparation_seconds,
        evaluation_seconds=evaluation_seconds, elapsed_seconds=time.monotonic()-started,
        maximum_stationary_residual_l1=max(x["residual_l1"] for x in diagnostics),
        validation=dict(zero_count_log_probability_error_bits=zero_error,
            single_token_cost_error_bits=single_error, two_token_total_probability=two_mass,
            maximum_independent_depth_one_error_bits=l1_error))
    raw_path.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"d={d}: evaluator {evaluation_seconds:.2f}s; total {result['elapsed_seconds']:.2f}s; mean gaps {np.round(gap.mean(0),2).tolist()}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ds", nargs="+", type=int, default=[64,256])
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--seed", type=int, default=202609071)
    p.add_argument("--factors", nargs="+", type=float, default=[.25,.5,.75,1.,1.5,2.,4.])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--lsa-src", type=Path, default=ROOT.parent/"layered-simplex-architecture/src")
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--archive", type=Path, default=ROOT/"output/takeover")
    p.add_argument("--raw-dir", type=Path, default=ROOT/"output/crossing_count_prediction_pilot")
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/crossing_count_prediction_pilot.json")
    args = p.parse_args()
    assert args.reps >= 2 and args.L >= 1 and all(d >= 2 for d in args.ds)
    assert all(f > 0 and math.isfinite(f) for f in args.factors)
    args.raw_dir = args.raw_dir.resolve()
    engage_anchor_mode(str(args.store))
    sys.path.insert(0, str(args.lsa_src.resolve()))
    start = time.monotonic()
    cells = []
    for d in args.ds:
        cells.append(run_cell(args,d))
        result = dict(schema_version=1, complete=len(cells)==len(args.ds),
            target="Expected cost gap of the deployed symmetric depth-mixture codes evaluated on independent stationary-transition-pair counts.",
            formula="M~Multinomial(t,vec(pi_T*T)); X0~q independently; N_b=sum_a M_ab+1{X0=b}; gap=ell_est(N)-log2(d)-sum_a ell_est(M_a).",
            qualifications=["This is a count-law approximation to a chain, not a Bayes code for independent pairs.",
                "Each checkpoint's counts have the exact multinomial law; independent increments couple the checkpoints.",
                "Stationary pi is solved for the same sampled table. No replacement pi=q or row-normalizer concentration is used.",
                "Initial q matches the archived chain protocol; the transition pairs themselves have stationary marginals.",
                "Pair counts need not satisfy a chain's endpoint-flow identity; this is part of the approximation being tested.",
                "Archived mean crossings select only the pilot evaluation window; no coefficient is fitted.",
                "Two tables are a runtime and implementation pilot, not an accurate crossing estimate.",
                "All statistical errors are sample standard errors with ddof=1; no asymptotic or last-crossing claim is made."],
            seed_protocol="default_rng([seed,d,L,round(100*alpha),rep,77]) for latent weights; independent stream ending78 for initial state and count increments.",
            evaluator="lsa.codelength.depth_averaged_codelength_profiles; uniform depth mixture1..default_l_max(d); sealed universal store.",
            script_sha256=sha(Path(__file__)), command_argv=sys.argv,
            cells=cells, elapsed_seconds=time.monotonic()-start)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Completed count pilot in {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
