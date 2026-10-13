"""LSA count approximation at larger source depths and concentrated tilts.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/crossing_lsa_extension.py \
    --out output/crossing_lsa_extension/pilot32_summary.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/crossing_lsa_extension.py \
    --ds 512 --settings 6:1.5 10:1.5 6:1 --reps 256 \
    --raw-dir output/crossing_lsa_extension/refined256 \
    --out output/crossing_lsa_extension/refined256/summary.json
python code/scripts/crossing_lsa_extension.py --collect \
    output/crossing_lsa_extension/pilot32_summary.json \
    output/crossing_lsa_extension/refined256/summary.json
python code/scripts/crossing_lsa_extension.py --self-test

Without an archived window, for a separate alpha=2 campaign:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/crossing_lsa_extension.py \
    --no-archive --methods count chain --settings 3:2 6:2 --ds 256 512 \
    --raw-dir output/crossing_lsa_alpha2 \
    --out results/mechanism/crossing_lsa_alpha2.json

Default: 32 draws at d=256,512, for (L,alpha)=(6,1.5),(10,1.5),
(6,1),(6,.5). The deployed frequency/context codes and default coding
depth range are unchanged. Independent Markov blocks of 4 and 16
transitions test refinements of the independent-pair count approximation.
All methods share only the source table and initial token. No Bayes costs,
new moment tables, fitted coefficients, or stationary visit substitutions.

The finite-count baseline draws independent counts at each checkpoint.
Block and actual-chain checkpoints share their respective prefixes. Later
blocks start independently from the exact marginal q T^s; their first block
starts at the common initial token. Thus every method has the correct mean
transition counts, but their joint count laws differ. Blocks are sampled
once to the final horizon and can be truncated at any checkpoint.
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

from crossing_count_prediction import ROOT, independent_l1_bits, moments, relative, sha
from crossing_finite_count_prediction import Occupations, chain_from_uniforms, last_mean_crossing
from context_stationary_check import stationary_checked
from takeover_experiment import engage_anchor_mode
from tilted_row_reference import DEFAULT_STORE


METHODS = ("count", "block4", "block16", "chain")
FACTORS = (.125, .25, .5, .75, .875, 1., 1.125, 1.25, 1.5, 2., 4., 8.)


def archive_for(directory, d, alpha, L):
    matches = []
    for path in directory.glob(f"d{d}_a*_L{L}/summary*.json"):
        item = json.loads(path.read_text())
        if (item["d"], item["alpha"], item["L"]) == (d, alpha, L):
            matches.append((path, item))
    if len(matches) != 1:
        raise ValueError(f"Expected one archived summary for {(d, alpha, L)}, found {len(matches)}")
    return matches[0]


def checkpoint_grid(archive, d):
    center = archive.get("mean_lambda_crossing")
    if center is not None:
        ns = sorted({2, 3, 4, 6, 8} | {max(2, round(center*f)) for f in FACTORS})
        return np.asarray(ns, dtype=np.int64), "archived_mean_root_times_predeclared_factors"
    # A missing archived root is not replaced by a path-crossing median.
    ns = sorted({2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
                 192, 256, 384, 512, 768, 1024, 1536, 2048, 4*d})
    return np.asarray(ns, dtype=np.int64), "predeclared_early_grid_without_archived_mean_root"


def blocked_edges(T, q, initial, total, block_size, rng, occupations):
    """Sample independent Markov blocks with exact time-specific starts."""
    d = len(q)
    starts = np.arange(0, total, block_size, dtype=np.int64)
    origins = np.empty(len(starts), dtype=np.int32)
    origins[0] = initial
    step = occupations.powers[int(math.log2(block_size))]
    marginal = q.copy()
    for j in range(1, len(starts)):
        marginal = marginal@step
        probability = np.maximum(marginal, 0.)
        probability /= probability.sum()
        origins[j] = rng.choice(d, p=probability)
    cdf = np.cumsum(T, axis=1)
    cdf[:, -1] = 1.
    source = np.empty(total, dtype=np.int32)
    target = np.empty(total, dtype=np.int32)
    state = origins.copy()
    for offset in range(block_size):
        indices = starts+offset
        active = indices < total
        positions, current = indices[active], state[active]
        u = rng.random(len(current))
        nxt = np.sum(u[:, None] >= cdf[current], axis=1).astype(np.int32)
        source[positions], target[positions] = current, nxt
        state[active] = nxt
    return source, target


def bootstrap(ns, gaps, draws, seed):
    reps = len(gaps["chain"])
    weights = np.random.default_rng(seed).multinomial(reps, np.full(reps, 1/reps), size=draws)/reps
    roots, summaries = {}, {}
    for name in gaps:
        averages = weights@gaps[name]
        values = np.full(draws, np.nan)
        statuses = Counter()
        for i, curve in enumerate(averages):
            result = last_mean_crossing(ns, curve[None, :])
            statuses[result["status"]] += 1
            if result["value"] is not None:
                values[i] = result["value"]
        located = np.isfinite(values)
        roots[name] = values
        summaries[name] = dict(draws=draws, located=int(located.sum()), null=int((~located).sum()),
            status_counts=dict(statuses), conditional_on_located_quantiles=(
                np.quantile(values[located], [.025, .5, .975]).tolist() if located.any() else None))
    for name in gaps:
        if name == "chain":
            continue
        both = np.isfinite(roots[name]) & np.isfinite(roots["chain"])
        ratio = roots[name][both]/roots["chain"][both]-1
        summaries[name+"_over_chain_minus_one"] = dict(draws=draws, located=int(both.sum()),
            null=int((~both).sum()), conditional_on_located_quantiles=(
                np.quantile(ratio, [.025, .5, .975]).tolist() if both.any() else None))
    return summaries, roots


def run_cell(args, L, alpha, d):
    from lsa.codelength import default_l_max, depth_averaged_codelength_profiles, profile_of
    started = time.monotonic()
    methods = tuple(args.methods)
    archive_path, archive = (None, {}) if args.no_archive else archive_for(args.archive, d, alpha, L)
    ns, window_rule = checkpoint_grid(archive, d)
    if args.no_archive:
        window_rule = "predeclared_early_grid_without_archive"
    ts = ns-1
    prefix_signature = dict(L=L, alpha=alpha, d=d, reps=args.reps, seed=args.seed,
        tokens=ns.tolist(), bootstrap=args.bootstrap, methods=list(methods), no_archive=args.no_archive,
        script_sha256=sha(Path(__file__)),
        evaluator_sha256=sha(args.lsa_src/"lsa/codelength.py"))
    signature = hashlib.sha256(json.dumps(prefix_signature, sort_keys=True).encode()).hexdigest()
    raw_path = args.raw_dir/f"L{L}_a{alpha:g}_d{d}.npz"
    summary_path = raw_path.with_suffix(".json")
    if args.resume and summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved["run_signature"] == signature and raw_path.exists() and saved["raw_sha256"] == sha(raw_path):
            print(f"Reuse complete L{L} a{alpha:g} d{d}", flush=True)
            return saved
        raise ValueError(f"Existing cell does not match this run; choose a separate raw directory: {summary_path}")
    q = np.arange(1, d+1, dtype=float)**(-alpha)
    q /= q.sum()
    shape = (args.reps, len(ns), d)
    stored = {name+"_frequency_counts": np.zeros(shape, dtype=np.int32) for name in methods}
    stored.update(transition_matrices=np.zeros((args.reps, d, d)),
        stationary_laws=np.zeros((args.reps, d)), normalized_row_sums=np.zeros((args.reps, d)),
        expected_occupation_laws=np.zeros(shape), initial_states=np.zeros(args.reps, dtype=np.int32),
        chain_paths=np.zeros((args.reps, int(ns[-1])), dtype=np.int32))
    for name in ("block4", "block16"):
        if name not in methods:
            continue
        stored[name+"_sources"] = np.zeros((args.reps, int(ts[-1])), dtype=np.int32)
        stored[name+"_targets"] = np.zeros((args.reps, int(ts[-1])), dtype=np.int32)
    sparse = {name: dict(pointers=[0], indices=[], values=[]) for name in methods}
    pending = {name: [] for name in methods}
    all_profiles = {(), (1,), (2,), (1, 1)}
    max_direct = max_flow = max_stationarity = 0.
    for rep in range(args.reps):
        prefix = [args.seed, d, L, round(alpha*100), rep]
        G = np.random.default_rng(prefix+[77]).exponential(size=(d, d, L)).prod(axis=2)
        S = G@q
        T = G*q[None, :]/S[:, None]
        pi, diagnostic = stationary_checked(T)
        max_stationarity = max(max_stationarity, diagnostic["residual_l1"])
        initial = int(np.random.default_rng(prefix+[78]).choice(d, p=q))
        occupation = Occupations(T, max(int(ts[-1]), 31))
        max_direct = max(max_direct, occupation.check(q))
        path = chain_from_uniforms(T, initial, np.random.default_rng(prefix+[79]).random(int(ts[-1])))
        edges = {"chain": (path[:-1], path[1:])}
        for size in [4, 16]:
            name = f"block{size}"
            if name not in methods:
                continue
            edges[name] = blocked_edges(T, q, initial, int(ts[-1]), size,
                np.random.default_rng(prefix+[90, size]), occupation)
            stored[name+"_sources"][rep], stored[name+"_targets"][rep] = edges[name]
        stored["transition_matrices"][rep], stored["stationary_laws"][rep] = T, pi
        stored["normalized_row_sums"][rep], stored["initial_states"][rep] = S, initial
        stored["chain_paths"][rep] = path
        cumulative = {name: np.zeros((d, d), dtype=np.int64) for name in edges}
        rep_profiles = {name: [] for name in methods}
        previous = 0
        for j, (n, t) in enumerate(zip(ns, ts)):
            r, end = occupation.at(q, int(t))
            assert r.min() > 0 and abs(r.sum()-1) < 1e-10
            max_flow = max(max_flow, float(abs(t*(r@T-r)-(end-q)).max()))
            stored["expected_occupation_laws"][rep, j] = r
            pair = (r[:, None]*T).ravel()
            pair /= pair.sum()
            independent = np.random.default_rng(prefix+[80, int(t)]).multinomial(int(t), pair).reshape(d, d)
            for name, (origin, destination) in edges.items():
                flat = origin[previous:int(t)].astype(np.int64)*d+destination[previous:int(t)]
                cumulative[name] += np.bincount(flat, minlength=d*d).reshape(d, d)
            previous = int(t)
            for name in methods:
                counts = independent if name == "count" else cumulative[name]
                fcounts = counts.sum(0)
                fcounts[initial] += 1
                assert counts.sum() == t and fcounts.sum() == n
                if name == "chain":
                    assert np.array_equal(fcounts, np.bincount(path[:int(n)], minlength=d))
                    balance = np.bincount([initial], minlength=d)-np.bincount([path[int(t)]], minlength=d)
                    assert np.array_equal(counts.sum(1)-counts.sum(0), balance)
                stored[name+"_frequency_counts"][rep, j] = fcounts
                occupied = np.flatnonzero(counts.ravel()).astype(np.int32)
                sp = sparse[name]
                sp["indices"].append(occupied)
                sp["values"].append(counts.ravel()[occupied].astype(np.int32))
                sp["pointers"].append(sp["pointers"][-1]+len(occupied))
                fp = profile_of(fcounts)
                rows = Counter(profile_of(row) for row in counts if row.sum())
                rep_profiles[name].append((fp, rows))
                all_profiles.add(fp)
                all_profiles.update(rows)
        for name in methods:
            pending[name].append(rep_profiles[name])
    preparation_seconds = time.monotonic()-started
    print(f"L{L} a{alpha:g} d{d}: prepared {args.reps} tables, {len(ns)} lengths, "
          f"{len(all_profiles)} profiles in {preparation_seconds:.2f}s; evaluating depths1..{default_l_max(d)}", flush=True)
    eval_start = time.monotonic()
    evaluated = depth_averaged_codelength_profiles({p: p for p in all_profiles}, d=d,
        l_max=default_l_max(d), jobs=args.jobs, tables_source="universal", universal_path=args.store)
    evaluation_seconds = time.monotonic()-eval_start
    zero_error = abs(evaluated[()].log2_q_avg)
    single_error = abs(-evaluated[(1,)].log2_q_avg-math.log2(d))
    two_mass = d*2**evaluated[(2,)].log2_q_avg+d*(d-1)*2**evaluated[(1, 1)].log2_q_avg
    l1_error = max(abs(-v.log2_q_by_depth[0]-independent_l1_bits(p, d)) for p, v in evaluated.items())
    assert max(zero_error, single_error, abs(two_mass-1), l1_error) < 1e-7
    assert max_direct < 1e-12 and max_flow < 1e-7 and max_stationarity < 1e-10
    summaries, gaps = {}, {}
    for name in methods:
        f = np.zeros((args.reps, len(ns)))
        c = np.zeros_like(f)
        for rep in range(args.reps):
            for j, (fp, rows) in enumerate(pending[name][rep]):
                f[rep, j] = -evaluated[fp].log2_q_avg
                c[rep, j] = math.log2(d)-math.fsum(mult*evaluated[p].log2_q_avg for p, mult in rows.items())
        gap = f-c
        stored[name+"_frequency_codelength"], stored[name+"_context_codelength"], stored[name+"_gap"] = f, c, gap
        gaps[name] = gap
        summaries[name] = dict(frequency_codelength=moments(f), context_codelength=moments(c),
            gap=moments(gap), mean_curve_crossing=last_mean_crossing(ns, gap))
        sp = sparse[name]
        stored[name+"_count_indptr"] = np.asarray(sp["pointers"], dtype=np.int64)
        stored[name+"_count_indices"] = np.concatenate(sp["indices"])
        stored[name+"_count_values"] = np.concatenate(sp["values"])
    paired = {}
    for name in methods:
        if name == "chain":
            continue
        paired[name+"_minus_chain"] = {field: moments(stored[name+"_"+field]-stored["chain_"+field])
            for field in ("frequency_codelength", "context_codelength", "gap")}
    boot, root_draws = bootstrap(ns, gaps, args.bootstrap, [args.seed, d, L, round(alpha*100), 700])
    for name, values in root_draws.items():
        stored[name+"_bootstrap_roots"] = values
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **stored, q=q, tokens=ns, transitions=ts)
    result = dict(complete=True, run_signature=signature, d=d, alpha=alpha, L=L,
        repetitions=args.reps, seed=args.seed, l_max=default_l_max(d), tokens=ns.tolist(), transitions=ts.tolist(),
        archived_mean_curve_crossing=archive.get("mean_lambda_crossing"),
        archive_path=relative(archive_path) if archive_path is not None else None,
        archive_sha256=sha(archive_path) if archive_path is not None else None,
        window_rule=window_rule, grid_factors=list(FACTORS) if archive_path is not None else None,
        methods=summaries, paired_differences=paired, bootstrap=boot, se_ddof=1,
        raw_path=relative(raw_path), raw_sha256=sha(raw_path), sparse_counts_shape=[args.reps, len(ns), d, d],
        sparse_counts_order="row-major rep/checkpoint; each stored index is a*d+b",
        profile_count=len(all_profiles), preparation_seconds=preparation_seconds,
        evaluation_seconds=evaluation_seconds, elapsed_seconds=time.monotonic()-started,
        validation=dict(maximum_direct_occupation_error=max_direct, maximum_expected_flow_error=max_flow,
            maximum_stationary_residual_l1=max_stationarity, zero_count_error_bits=zero_error,
            single_count_error_bits=single_error, two_token_total_probability=two_mass,
            maximum_independent_depth_one_error_bits=l1_error))
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"L{L} a{alpha:g} d{d}: total {result['elapsed_seconds']:.2f}s; roots "
          +str({name: summaries[name]["mean_curve_crossing"] for name in methods}), flush=True)
    return result


def self_test():
    T = np.array([[.81, .19], [.27, .73]])
    q = np.array([.67, .33])
    obj = Occupations(T, 64)
    assert obj.check(q) < 1e-14
    # A block covering the entire prefix has the full Markov path law.
    for initial in [0, 1]:
        for b in [4, 16]:
            seed = 137+initial+b
            source, target = blocked_edges(T, q, initial, b, b, np.random.default_rng(seed), obj)
            direct = chain_from_uniforms(T, initial, np.random.default_rng(seed).random(b))
            assert np.array_equal(source, direct[:-1]) and np.array_equal(target, direct[1:])
    ns = np.array([2, 4, 8])
    assert last_mean_crossing(ns, np.array([[1., 2., 3.]]))["value"] is None
    assert abs(last_mean_crossing(ns, np.array([[-1., -1., 1.]]))["value"]-math.sqrt(32)) < 1e-12
    return dict(status="passed", checks=["Occupation sums versus direct matrix iteration",
        "A single block reproduces the exact same Markov path for the same random uniforms",
        "Positive-start null root and hand-computed log-length root"])


def collect_existing(paths, out):
    """Collect complete campaigns, retaining pilot values and provenance.

    No random samples or code evaluations are performed. The selected cells
    use the greatest repetition count; complete input summaries are also
    retained, so the effect of the refinement can be checked directly.
    """
    campaigns, selected = [], {}
    for path in paths:
        path = path.resolve()
        summary = json.loads(path.read_text())
        if not summary["complete"] or summary.get("schema_version") != 1:
            raise ValueError(f"Expected a complete, uncollected campaign: {path}")
        campaigns.append(dict(summary_path=relative(path), summary_sha256=sha(path),
                              summary=summary))
        for cell in summary["cells"]:
            if not cell["complete"]:
                raise ValueError(f"Incomplete cell in {path}")
            raw = ROOT/cell["raw_path"]
            if not raw.is_file() or sha(raw) != cell["raw_sha256"]:
                raise ValueError(f"Raw data missing or hash mismatch: {raw}")
            key = (cell["d"], cell["L"], cell["alpha"])
            old = selected.get(key)
            if old is not None:
                for field in ("seed", "l_max", "tokens", "transitions", "archive_sha256"):
                    if old[field] != cell[field]:
                        raise ValueError(f"Incompatible refinement {key}: {field}")
                if old["repetitions"] == cell["repetitions"] and old["raw_sha256"] != cell["raw_sha256"]:
                    raise ValueError(f"Ambiguous repeated campaign for {key}")
            if old is None or cell["repetitions"] > old["repetitions"]:
                selected[key] = cell
    result = dict(campaigns[0]["summary"])
    result.update(schema_version=2, cells=list(selected.values()), campaigns=campaigns,
        selected_independent_table_draws=sum(c["repetitions"] for c in selected.values()),
        collection=dict(script=relative(Path(__file__)), script_sha256=sha(Path(__file__)),
            command_argv=sys.argv,
            selection_rule="For each parameter cell use the complete campaign with most repetitions; preserve all complete input campaigns.",
            repeated_prefix_rule="Replicate-index seeds preserve earlier samples. Refinements are not independent of their pilot prefixes."),
        elapsed_seconds=sum(c["summary"]["elapsed_seconds"] for c in campaigns))
    result["qualifications"] = result["qualifications"] + [
        "Repetition counts differ across cells: use each cell's repetitions and uncertainty fields. The 256-draw refinements include their original 32 draws; the complete pilot summaries remain in campaigns."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Collected {len(campaigns)} campaigns, {len(selected)} cells, "
          f"{result['selected_independent_table_draws']} distinct table/path draws: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", default=["6:1.5", "10:1.5", "6:1", "6:.5"])
    parser.add_argument("--ds", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--reps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=202609074)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--lsa-src", type=Path, default=ROOT.parent/"layered-simplex-architecture/src")
    parser.add_argument("--archive", type=Path, default=ROOT/"output/takeover")
    parser.add_argument("--no-archive", action="store_true",
        help="Use the predeclared early grid through max(2048,4d), without an archived center.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
        help="Methods to evaluate; count and chain are required. Defaults retain both block refinements.")
    parser.add_argument("--previous-driver", type=Path,
        help="Optional preserved driver snapshot, recorded by path and hash for provenance.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"output/crossing_lsa_extension")
    parser.add_argument("--out", type=Path, default=ROOT/"results/mechanism/crossing_lsa_extension.json")
    parser.add_argument("--collect", nargs="+", type=Path,
        help="Collect existing complete campaign summaries without drawing samples or evaluating codes.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect_existing(args.collect, args.out)
        return
    validation = self_test()
    if args.self_test:
        print(json.dumps(validation, indent=2)); return
    assert args.reps >= 2 and args.bootstrap > 0
    assert {"count", "chain"}.issubset(args.methods) and len(set(args.methods)) == len(args.methods)
    args.methods = [name for name in METHODS if name in args.methods]
    if args.previous_driver is not None:
        assert args.previous_driver.is_file()
    args.raw_dir = args.raw_dir.resolve()
    engage_anchor_mode(str(args.store))
    sys.path.insert(0, str(args.lsa_src.resolve()))
    grid = [(int(L), float(alpha), d) for d in args.ds for L, alpha in (s.split(":") for s in args.settings)]
    cells = []
    started = time.monotonic()
    for L, alpha, d in grid:
        cells.append(run_cell(args, L, alpha, d))
        result = dict(schema_version=1, complete=len(cells) == len(grid), requested_cells=grid,
            target="Expected cumulative LSA frequency/context costs: finite-occupation pairs, independent Markov blocks, and actual chains.",
            methods=list(args.methods), no_archive=args.no_archive,
            source_model="Independent product-of-L-exponential weights; normalized Zipf tilt; X0~q independently of the table.",
            numerical_method="Unchanged symmetric depth-mixture evaluator and default_l_max(d), reading the existing sealed anchor store.",
            seed_protocol="default_rng([seed,d,L,round(100alpha),rep,stream]);77 source;78 initial;79 chain;80,t independent pairs;90,B independent blocks.",
            qualifications=["The source depth is not the coding depth. Both LSA codes retain the same default coding range.",
                "Known table draws are used to integrate the approximation, never supplied to the LSA evaluator.",
                "All methods have the exact mean transition counts. Blocks also preserve dependence within each block and the true first-block/initial-token dependence.",
                "Methods share table and initial token only; their other random streams are independent. Blocks and chains share prefixes over checkpoints; pair-count checkpoints are conditionally independent.",
                "An absent archived mean root triggers a predeclared early grid, not a substituted median or fitted crossing.",
                "Roots are last observed sign changes of mean cumulative gaps on the stated finite grids. A positive gap at every check is unlocated; no later signs are certified.",
                "Bootstrap resamples independent table indices jointly across methods and lengths. Quantiles condition on located roots; all null counts are retained.",
                "Sampling uncertainty does not cover finite-grid root interpolation or approximation bias. No optimal-Bayes quantities are computed."],
            script_sha256=sha(Path(__file__)), evaluator_sha256=sha(args.lsa_src/"lsa/codelength.py"),
            dependency_sha256={f: sha(Path(__file__).with_name(f)) for f in
                ["crossing_finite_count_prediction.py", "crossing_count_prediction.py", "takeover_experiment.py", "context_stationary_check.py"]},
            command_argv=sys.argv, validation=validation, cells=cells, elapsed_seconds=time.monotonic()-started)
        if args.previous_driver is not None:
            result["driver_extension_provenance"] = dict(
                previous_driver_path=relative(args.previous_driver.resolve()),
                previous_driver_sha256=sha(args.previous_driver),
                changes="Added explicit no-archive and method-selection CLI options; source draws, count laws, evaluator, depth range, and previous default choices are unchanged.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
