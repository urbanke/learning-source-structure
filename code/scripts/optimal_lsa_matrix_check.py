"""Evaluate known-q,L Bayes costs on eighteen saved LSA experiments.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/optimal_lsa_matrix_check.py

No source, path, or auxiliary-count samples are generated. The known-prior
Bayes code and finite scalar expression are distinct from both LSA codes.
The finite knowledge allowance is added to measured Bayes costs only.
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

from context_finite_prediction_check import ScalarCurve, self_test
from crossing_finite_count_prediction import Occupations
from crossing_count_prediction import ROOT, moments, relative
from iid_depth_prediction import ScalarInformation, scalar_validation, sealed_probability_validation, LN2
from tilted_row_reference import TiltedRowReference, DEFAULT_STORE


GRID = [(L, a, d) for d in (256, 128, 512) for L in (2, 3) for a in (.75, 1., 1.25)]
PRIOR = [(L, a) for L in (2, 3) for a in (.75, 1., 1.25)]
BASELINE = ROOT / "results/mechanism/optimal_context_figure_check.json"
BASELINE_AUDIT = ROOT / "results/mechanism/optimal_context_figure_audit.json"
IID_SCALARS = ROOT / "results/mechanism/iid_depth_prediction.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validated_scalars():
    baseline = json.loads(BASELINE.read_text())
    audit = json.loads(BASELINE_AUDIT.read_text())
    assert audit["status"] == "passed" and audit["source_sha256"] == sha(BASELINE)
    assert baseline["raw_sha256"] == sha(ROOT / baseline["raw_path"])
    assert audit["raw_sha256"] == baseline["raw_sha256"]
    iid = json.loads(IID_SCALARS.read_text())
    master = json.loads((ROOT / "results/mechanism/master_curves.json").read_text())
    scalar_checks = scalar_validation(master, [2, 3])
    sealed_checks = sealed_probability_validation([2, 3])
    records = {2: baseline["scalar_curve"], 3: iid["scalar_curves"]["3"]}
    out = {}
    for L, record in records.items():
        x, y = np.asarray(record["x"]), np.asarray(record["r_bits"])
        assert len(x) == 1025 and np.all(np.diff(x) > 0)
        assert np.all(y > 0) and np.all(np.diff(y) > 0)
        calc = ScalarInformation(L)
        checks = []
        for j in (0, 256, 384, 512, 640, 768, 1024):
            fresh = calc(float(x[j]))
            checks.append(dict(index=j, x=float(x[j]), stored=float(y[j]),
                               fresh=fresh, absolute_difference_bits=abs(fresh-y[j])))
        assert max(r["absolute_difference_bits"] for r in checks) < 1e-8
        source = BASELINE if L == 2 else IID_SCALARS
        out[str(L)] = dict(record=record, source=relative(source), source_sha256=sha(source),
                           fresh_knot_checks=checks)
    return out, dict(scalar_checks=scalar_checks, sealed_probability_checks=sealed_checks,
                     baseline_audit=relative(BASELINE_AUDIT), baseline_audit_sha256=sha(BASELINE_AUDIT))


def run_cell(args, L, alpha, d, scalar, scalars_sha):
    started = time.monotonic()
    stem = f"L{L}_a{alpha:g}_d{d}"
    source_path = ROOT / "output/crossing_finite_count_prediction" / (stem + ".npz")
    source_json = source_path.with_suffix(".json")
    out_json = args.raw_dir / (stem + ".json")
    out_npz = out_json.with_suffix(".npz")
    source = json.loads(source_json.read_text())
    config = dict(L=L, alpha=alpha, d=d, repetitions=16, step=1/128,
                  source_sha256=sha(source_path), source_summary_sha256=sha(source_json),
                  scalar_sha256=scalars_sha, producer_sha256=sha(Path(__file__)))
    assert (source["L"], source["alpha"], source["d"], source["repetitions"]) == (L, alpha, d, 16)
    assert source["raw_sha256"] == config["source_sha256"]
    if out_json.exists() and out_npz.exists() and not args.force:
        saved = json.loads(out_json.read_text())
        if saved.get("configuration") == config and saved.get("complete"):
            assert saved["raw_sha256"] == sha(out_npz)
            print(f"{stem}: reuse completed evaluation", flush=True)
            return saved
    with np.load(source_path) as inp:
        T, pi, S, q = (inp[k] for k in ("transition_matrices", "stationary_laws", "normalized_row_sums", "q"))
        counts = inp["chain_transition_counts"]
        saved_r = inp["expected_occupation_laws"]
        initial, paths = inp["initial_states"], inp["chain_paths"]
        ns, ts = inp["tokens"], inp["transitions"]
        f, c = inp["chain_frequency_codelength"], inp["chain_context_codelength"]
        ff, cf = inp["count_frequency_codelength"], inp["count_context_codelength"]
        fcounts = inp["chain_frequency_counts"]
    shape = (16, len(ns))
    assert len(ns) == 10 and counts.shape == (*shape, d, d)
    assert np.array_equal(ts, ns-1) and np.array_equal(ns, source["tokens"])
    assert np.array_equal(counts.sum((2, 3)), np.broadcast_to(ts, shape))
    expected_q = np.arange(1, d+1, dtype=float)**(-alpha)
    expected_q /= expected_q.sum()
    assert np.max(abs(q-expected_q)) < 1e-15
    assert np.all(T > 0) and np.max(abs(T.sum(2)-1)) < 1e-12
    for rep in range(16):
        assert paths[rep, 0] == initial[rep]
        replay = np.zeros((d, d), dtype=np.int64)
        previous = 0
        for j, t in enumerate(ts):
            source_labels = paths[rep, previous:int(t)]
            target_labels = paths[rep, previous+1:int(t)+1]
            replay += np.bincount(source_labels.astype(np.int64)*d+target_labels, minlength=d*d).reshape(d, d)
            assert np.array_equal(replay, counts[rep, j])
            destination_counts = replay.sum(0)
            destination_counts[initial[rep]] += 1
            assert np.array_equal(destination_counts, fcounts[rep, j])
            previous = int(t)
    for kind, values in (("chain", (f, c)), ("count", (ff, cf))):
        for name, array in zip(("frequency_codelength", "context_codelength"), values):
            fresh = moments(array)
            assert np.allclose(fresh["mean"], source[kind][name]["mean"], rtol=1e-13, atol=1e-9)
            assert np.allclose(fresh["SE"], source[kind][name]["SE"], rtol=1e-13, atol=1e-9)
    del paths, fcounts
    initial_bits = -np.log2(q[initial])
    Hq = -float(q@np.log2(q))
    entropies = -(T*np.log2(T)).sum(2)
    occupations = np.empty_like(saved_r)
    expected_known = np.zeros(shape)
    known_realized = np.zeros(shape)
    occupation_error, flow_error, direct_error = 0., 0., 0.
    for rep in range(16):
        op = Occupations(T[rep], int(ts[-1]))
        for j, t in enumerate(ts):
            r, end = op.at(q, int(t))
            occupations[rep, j] = r
            occupation_error = max(occupation_error, float(abs(r-saved_r[rep, j]).max()))
            flow_error = max(flow_error, float(abs(r@T[rep]-r-(end-q)/t).max()))
            expected_known[rep, j] = Hq+t*(r@entropies[rep])
            known_realized[rep, j] = initial_bits[rep]-float((counts[rep, j]*np.log2(T[rep])).sum())
        if rep in (0, 15):
            state, total = q.copy(), np.zeros(d)
            for _ in range(7):
                total += state
                state = state@T[rep]
            binary, _ = op.at(q, 7)
            direct_error = max(direct_error, float(abs(binary-total/7).max()))
    assert occupation_error < 1e-12 and flow_error < 1e-12 and direct_error < 1e-12
    curve = ScalarCurve(scalar)
    coarse = ScalarCurve(dict(scalar, x=scalar["x"][::2], r_bits=scalar["r_bits"][::2]))
    J, Js = np.zeros(shape), np.zeros(shape)
    interpolation = np.zeros((*shape, 2))
    min_rate, max_rate = math.inf, 0.
    for rep in range(16):
        weights = q[None, :]/S[rep, :, None]
        for j, t in enumerate(ts):
            rates = t*occupations[rep, j, :, None]*weights
            stationary = t*pi[rep, :, None]*weights
            min_rate, max_rate = min(min_rate, float(rates.min())), max(max_rate, float(rates.max()))
            J[rep, j], Js[rep, j] = curve(rates).sum(), curve(stationary).sum()
            interpolation[rep, j] = (abs(coarse(rates).sum()-J[rep, j]), abs(coarse(stationary).sum()-Js[rep, j]))
    bayes_rows, refinement = np.zeros((*shape, d)), np.zeros((*shape, d))
    baseline_reuse = (L, alpha, d) == (2, 1., 256)
    baseline_error = None
    if baseline_reuse:
        baseline = json.loads(BASELINE.read_text())
        assert baseline["source_sha256"] == config["source_sha256"]
        with np.load(ROOT/baseline["raw_path"]) as original:
            assert np.array_equal(original["tokens"], ns)
            bayes_rows[:] = original["bayes_row_costs"]
            refinement[:] = original["quadrature_refinement_rows_bits"]
            baseline_error = max(float(abs(expected_known-original["known_table_expected_full_cost"]).max()),
                                 float(abs(known_realized-original["known_table_realized_full_cost"]).max()),
                                 float(abs(J-original["J_finite_occupation"]).max()),
                                 float(abs(f-original["lsa_frequency_full_cost"]).max()),
                                 float(abs(c-original["lsa_context_full_cost"]).max()))
            assert baseline_error < 1e-8
    cache = {}
    evaluation_started = time.monotonic()
    ref = TiltedRowReference(q, L, store=args.store, step=1/128)
    try:
        if not baseline_reuse:
            for rep in range(16):
                for j in range(len(ns)):
                    for a, row in enumerate(counts[rep, j]):
                        labels = np.flatnonzero(row)
                        if not len(labels):
                            continue
                        key = tuple(zip(map(int, labels), map(int, row[labels])))
                        if key not in cache:
                            log_probability, diag = ref.log_probability(row)
                            cache[key] = (-log_probability/LN2, diag["coarse_fine_difference_bits"])
                        bayes_rows[rep, j, a], refinement[rep, j, a] = cache[key]
                if rep in (0, 7, 15):
                    print(f"{stem}: {rep+1}/16 Bayes paths, {len(cache)} labelled profiles, {time.monotonic()-evaluation_started:.1f}s", flush=True)
        spot_checks = []
        for rep in (0, 15):
            for j in (0, 4, 9):
                a = int(np.argmax(counts[rep, j].sum(1)))
                fresh, diag = ref.log_probability(counts[rep, j, a])
                difference = abs(-fresh/LN2-bayes_rows[rep, j, a])
                assert difference < 1e-8
                spot_checks.append(dict(rep=rep, checkpoint=j, row=a, k=int(counts[rep, j, a].sum()), baseline_difference_bits=difference))
    finally:
        ref.close()
    fine_ref = TiltedRowReference(q, L, store=args.store, step=1/256, lower=-68., upper=34.)
    try:
        for spot in spot_checks:
            rep, j, a = (spot[key] for key in ("rep", "checkpoint", "row"))
            log_probability, diag = fine_ref.log_probability(counts[rep, j, a])
            spot["wider_finer_difference_bits"] = abs(-log_probability/LN2-bayes_rows[rep, j, a])
            assert spot["wider_finer_difference_bits"] < 1e-7
    finally:
        fine_ref.close()
    bayes = initial_bits[:, None]+bayes_rows.sum(2)
    redundancy = bayes-known_realized
    prediction = expected_known+J
    allowance = math.lgamma(d+1)/LN2+math.log2(len(PRIOR))
    raw = dict(tokens=ns, transitions=ts, q=q, initial_bits=initial_bits,
        expected_occupation_laws=occupations, row_entropies=entropies,
        bayes_row_costs=bayes_rows, bayes_full_cost=bayes,
        known_table_realized_full_cost=known_realized, known_table_expected_full_cost=expected_known,
        bayes_redundancy=redundancy, bayes_control_variate_full_cost=expected_known+redundancy,
        J_finite_occupation=J, J_stationary=Js, predicted_finite_full_cost=prediction,
        predicted_stationary_full_cost=expected_known+Js,
        lsa_frequency_full_cost=f, lsa_context_full_cost=c,
        lsa_frequency_formula_full_cost=ff, lsa_context_formula_full_cost=cf,
        bayes_plus_knowledge_allowance_full_cost=bayes+allowance,
        quadrature_refinement_rows_bits=refinement, scalar_interpolation_refinement_bits=interpolation)
    differences = dict(redundancy_minus_J_finite_occupation=redundancy-J,
        bayes_raw_minus_finite_prediction=bayes-prediction,
        lsa_frequency_minus_bayes=f-bayes, lsa_context_minus_bayes=c-bayes,
        lsa_frequency_minus_finite_prediction=f-prediction, lsa_context_minus_finite_prediction=c-prediction,
        lsa_frequency_actual_minus_formula=f-ff, lsa_context_actual_minus_formula=c-cf,
        lsa_actual_gap=f-c, lsa_formula_gap=ff-cf,
        lsa_frequency_minus_hierarchy_upper_endpoint=f-bayes-allowance,
        lsa_context_minus_hierarchy_upper_endpoint=c-bayes-allowance)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **raw)
    cost_summary = {name: moments(value) for name, value in raw.items() if np.asarray(value).shape == shape}
    # The allowance is deterministic: retain the actual Bayes SE exactly.
    cost_summary["bayes_plus_knowledge_allowance_full_cost"]["SE"] = cost_summary["bayes_full_cost"]["SE"]
    result = dict(complete=True, configuration=config, d=d, alpha=alpha, L=L, repetitions=16,
        tokens=ns.tolist(), transitions=ts.tolist(), source=relative(source_path), source_sha256=config["source_sha256"],
        source_summary=relative(source_json), source_summary_sha256=config["source_summary_sha256"],
        raw_path=relative(out_npz), raw_sha256=sha(out_npz), source_coding_cutoff=source["l_max"], se_ddof=1,
        knowledge_allowance_bits=allowance, costs=cost_summary,
        paired_differences={name: moments(value) for name, value in differences.items()},
        lsa_crossings=dict(actual=source["chain"]["mean_curve_crossing"], formula=source["count"]["mean_curve_crossing"], archive=source["archived_mean_curve_crossing"]),
        baseline_reused=baseline_reuse, baseline_source=relative(BASELINE) if baseline_reuse else None,
        baseline_source_sha256=sha(BASELINE) if baseline_reuse else None,
        validation=dict(replayed_paths=16, replayed_count_matrices=160,
            maximum_saved_occupation_error=occupation_error, maximum_occupation_mean_flow_error=flow_error,
            maximum_independent_seven_step_occupation_error=direct_error,
            baseline_shared_array_maximum_difference_bits=baseline_error, row_spot_checks=spot_checks,
            maximum_summed_quadrature_refinement_bits=float(refinement.sum(2).max()),
            maximum_half_grid_scalar_sum_difference_bits=float(interpolation.max()),
            minimum_finite_rate=min_rate, maximum_finite_rate=max_rate),
        unique_evaluated_labelled_profiles=len(cache), elapsed_seconds=time.monotonic()-started)
    write_json(out_json, result)
    print(f"{stem}: COMPLETE in {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"output/optimal_lsa_matrix_check")
    parser.add_argument("--out", type=Path, default=ROOT/"results/mechanism/optimal_lsa_matrix_check.json")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--ds", type=int, nargs="+", default=[256, 128, 512])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    assert set(args.ds) <= {128, 256, 512}
    if args.self_test:
        print(json.dumps(self_test(args.store), indent=2))
        return
    started = time.monotonic()
    scalars, checks = validated_scalars()
    scalar_digest = hashlib.sha256(json.dumps(scalars, sort_keys=True).encode()).hexdigest()
    selected = [(L, a, d) for L, a, d in GRID if d in args.ds]
    result = dict(schema_version=1, complete=False, requested_grid=[dict(L=L, alpha=a, d=d) for L, a, d in selected],
        repetitions=16, se_ddof=1, scalar_curves=scalars, scalar_validation=checks,
        knowledge_prior=dict(source_depths=[2, 3], alphas=[.75, 1., 1.25], joint_weights="Uniform over six (alpha,L) pairs; uniform over all d! label permutations.",
                             allowance="log2(d!)+log2(6) bits, added to actual known-q,L Bayes costs."),
        qualifications=[
            "All source tables, initial states, actual paths and LSA auxiliary-count samples are reused unchanged; no new simulations or LSA evaluations.",
            "Every code cost counts all n tokens. The known-q,L Bayes code charges -log2q(X0); both saved symmetric LSA mechanisms use their original conventions.",
            "The Bayes formula uses only latent source weights, row normalizers and exact finite expected occupations; it does not use path counts or measured costs.",
            "The finite scalar expression is a chain approximation, not an exact identity or a rigorous bound.",
            "The shifted actual Bayes mean estimates a rigorous upper endpoint for expected hierarchical cost under the stated discrete prior. Its SE is the Bayes SE. The shifted scalar prediction is not used as a rigorous endpoint.",
            "Hierarchy comparisons in each fixed parameter cell use the mixture-component allowance; full hierarchical Bayes optimality holds in the common-prior average.",
            "Sampling SEs and numerical refinement diagnostics are separate; refinement checks are not certified quadrature bounds.",
            "Saved LSA formula samples approximate transition-count dependence. Their crossing windows were selected using archived mean crossings; no crossing or cost coefficient is fitted."],
        script_sha256=sha(Path(__file__)),
        evaluator_hashes={name:sha(Path(__file__).with_name(name)) for name in ("tilted_row_reference.py", "iid_depth_prediction.py", "context_finite_prediction_check.py", "crossing_finite_count_prediction.py")},
        command_argv=sys.argv, cells=[])
    for L, alpha, d in selected:
        result["cells"].append(run_cell(args, L, alpha, d, scalars[str(L)]["record"], scalar_digest))
        result["elapsed_seconds"] = time.monotonic()-started
        result["complete"] = len(result["cells"]) == len(selected)
        write_json(args.out, result)
    print(json.dumps(dict(complete=result["complete"], cells=len(result["cells"]), elapsed_seconds=result["elapsed_seconds"]), indent=2))


if __name__ == "__main__":
    main()
