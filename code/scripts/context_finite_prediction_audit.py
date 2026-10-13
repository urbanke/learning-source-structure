"""Independent normalization and saved-data checks for the actual-chain experiment.

python code/scripts/context_finite_prediction_audit.py --self-test
python code/scripts/context_finite_prediction_audit.py

No new transition tables or sample paths are generated. Tiny path laws are
enumerated and integrated using the independent adaptive depth-one route.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from context_row_check import log_mixture_adaptive
from tilted_row_reference import TiltedRowReference


ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    values = [float(v) for v in values]
    mean = math.fsum(values)/len(values)
    se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1)/len(values))
    return mean, se


def close(a, b, tol=1e-8):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a, b, tol)


def tiny_paths():
    checks = []
    for q in (np.array([.5, .5]), np.array([.7, .3])):
        ref = TiltedRowReference(q, 1, step=1/128)
        cache = {}
        try:
            for n in range(1, 6):
                adaptive_mass = production_mass = 0.
                maximum_error = 0.
                for path in itertools.product(range(2), repeat=n):
                    counts = np.zeros((2, 2), dtype=np.int64)
                    for a, b in zip(path[:-1], path[1:]):
                        counts[a, b] += 1
                    adaptive_log = production_log = math.log(q[path[0]])
                    for row in counts:
                        key = tuple(row)
                        if key not in cache:
                            cache[key] = (log_mixture_adaptive(row, q)[0], ref.log_probability(row)[0])
                        a, p = cache[key]
                        adaptive_log += a
                        production_log += p
                    adaptive_mass += math.exp(adaptive_log)
                    production_mass += math.exp(production_log)
                    maximum_error = max(maximum_error, abs(adaptive_log-production_log)/LN2)
                close(adaptive_mass, 1., 1e-10)
                close(production_mass, 1., 1e-10)
                assert maximum_error < 1e-10
                checks.append(dict(q=q.tolist(), tokens=n, enumerated_paths=2**n,
                    adaptive_total_probability=adaptive_mass,
                    production_total_probability=production_mass,
                    maximum_adaptive_production_log_difference_bits=maximum_error))
        finally:
            ref.close()
    return checks


def curve_values(x, curve):
    x = np.asarray(x)
    out = np.zeros_like(x)
    good = x > 0
    knots = np.asarray(curve["x"])
    out[good] = np.interp(np.log(x[good]), np.log(knots), curve["r_bits"])
    low, high = good & (x < knots[0]), x > knots[-1]
    out[low] = curve["slope0_bits"]*x[low]
    out[high] = .5*np.log2(x[high])+curve["tail_constant_bits"]
    return out


def audit(source, tiny):
    data = json.loads(source.read_text())
    assert data["complete"] and len(data["cells"]) == 24
    grid = {(c["configuration"]["L"], c["configuration"]["alpha"], c["configuration"]["d"])
            for c in data["cells"]}
    assert grid == set(itertools.product((1, 3, 6), (.5, 1., 1.5, 2.), (64, 256)))
    scalar_path = ROOT/data["scalar_path"]
    assert digest(scalar_path) == data["scalar_sha256"]
    experiment_hash = digest(Path(__file__).with_name("context_finite_prediction_check.py"))
    evaluator_hash = digest(Path(__file__).with_name("tilted_row_reference.py"))
    curves = json.loads(scalar_path.read_text())["scalar_curves"]
    checks, independent_paths = [], []
    for cell in data["cells"]:
        config = cell["configuration"]
        d, L, alpha, n = config["d"], config["L"], config["alpha"], config["repetitions"]
        assert n == 16 and cell["complete"] and cell["se_ddof"] == 1
        assert config["seed"] == 202609070 and config["step"] == 1/128
        assert config["scalar_sha256"] == data["scalar_sha256"]
        assert config["experiment_sha256"] == experiment_hash
        assert cell["evaluator_sha256"] == evaluator_hash
        raw_path = ROOT/cell["raw_path"]
        assert digest(raw_path) == cell["raw_sha256"]
        with np.load(raw_path) as raw:
            ts = raw["transitions"]
            assert ts.tolist() == [d*d//4, d*d, 4*d*d]
            a = np.arange(1, d+1, dtype=float)**(-alpha)
            q = a/a.sum()
            assert np.max(abs(q-raw["q"])) < 1e-14
            cost_error = prediction_error = stationary_error = 0.
            for rep in range(n):
                T, pi, Z = raw["transition_matrices"][rep], raw["stationary_laws"][rep], raw["unnormalized_row_sums"][rep]
                stationary_error = max(stationary_error, float(abs(pi@T-pi).sum()))
                assert np.all(Z > 0) and np.max(abs(T.sum(1)-1)) < 1e-12
                assert np.max(abs(raw["normalized_row_sums"][rep]-Z/a.sum())) < 1e-12
                close(raw["initial_cost"][rep], -math.log2(q[raw["initial_states"][rep]]))
                for j, t in enumerate(ts):
                    counts = raw["transition_counts"][rep, j]
                    K = counts.sum(1)
                    assert K.sum() == t and np.array_equal(K, raw["visit_counts"][rep, j])
                    flow = np.bincount([raw["initial_states"][rep]], minlength=d)-np.bincount([raw["terminal_states"][rep, j]], minlength=d)
                    assert np.array_equal(K-counts.sum(0), flow)
                    oracle = -math.fsum((counts*np.log2(T)).ravel())
                    cost_error = max(cost_error, abs(oracle-raw["oracle"][rep, j]))
                    close(oracle, raw["oracle"][rep, j])
                    close(raw["bayes"][rep, j]-raw["oracle"][rep, j], raw["redundancy"][rep, j])
                    rate_vectors = {"old_stationary":t*pi[:, None]*q[None, :],
                        "joint_stationary":(t*pi/Z)[:, None]*a[None, :],
                        "joint_actual_visits":(K/Z)[:, None]*a[None, :]}
                    for name, rates in rate_vectors.items():
                        value = math.fsum(curve_values(rates, curves[str(L)]).ravel())
                        prediction_error = max(prediction_error, abs(value-raw[name][rep, j]))
                        close(value, raw[name][rep, j])
            assert stationary_error < 1e-10
            for j in range(len(ts)):
                r = raw["redundancy"][:, j]
                mean, se = moments(r)
                close(mean, cell["redundancy"]["mean"][j])
                close(se, cell["redundancy"]["SE"][j])
                for name, record in cell["comparisons"].items():
                    p = raw[name][:, j]
                    pm, ps = moments(p)
                    dm, ds = moments(r-p)
                    covariance = math.fsum((r-mean)*(p-pm))/(n-1)
                    close(pm, record["prediction"]["mean"][j])
                    close(ps, record["prediction"]["SE"][j])
                    close(dm, record["difference"]["mean"][j])
                    close(ds, record["difference"]["SE"][j])
                    close(covariance, record["measured_prediction_sample_covariance"][j], 1e-7)
                    close(mean/pm-1, record["measured_over_prediction_minus_one"][j], 1e-12)
                    ratio_variance = (se**2+(mean/pm)**2*ps**2
                                      -2*(mean/pm)*covariance/n)/pm**2
                    close(math.sqrt(max(0., ratio_variance)),
                          record["relative_difference_delta_method_SE"][j], 1e-12)
            # Re-evaluate one full nonuniform L1 path independently.
            if L == 1 and alpha == 1.5 and d == 64:
                counts = raw["transition_counts"][0, -1]
                adaptive = -math.fsum(log_mixture_adaptive(row, q)[0]/LN2 for row in counts)
                stored = float(raw["bayes"][0, -1])
                close(adaptive, stored, 1e-7)
                independent_paths.append(dict(L=L, d=d, alpha=alpha, repetition=0,
                    transitions=int(ts[-1]), adaptive_Bayes_cost_bits=adaptive,
                    stored_Bayes_cost_bits=stored, difference_bits=adaptive-stored))
            checks.append(dict(L=L, d=d, alpha=alpha, repetitions=n, checkpoints=len(ts),
                maximum_oracle_recomputation_error_bits=cost_error,
                maximum_prediction_recomputation_error_bits=prediction_error,
                maximum_stationarity_residual_l1=stationary_error,
                raw_path=cell["raw_path"], raw_sha256=cell["raw_sha256"]))
    return dict(status="passed", source=str(source.relative_to(ROOT)), source_sha256=digest(source),
        checked_cells=len(checks), checked_means=sum(c["checkpoints"] for c in checks),
        scope="Independent numerical/aggregation audit only; no new random paths and no assertion of a chain asymptotic.",
        tiny_path_checks=tiny, independent_full_path_checks=independent_paths, cell_checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/mechanism/context_finite_prediction_check.json")
    parser.add_argument("--out", type=Path, default=ROOT/"results/mechanism/context_finite_prediction_audit.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    tiny = tiny_paths()
    if args.self_test:
        print(json.dumps(dict(status="passed", checks=tiny), indent=2))
        return
    result = audit(args.source, tiny)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status":result["status"], "checked_means":result["checked_means"], "independent_full_path_checks":result["independent_full_path_checks"]}, indent=2))


if __name__ == "__main__":
    main()
