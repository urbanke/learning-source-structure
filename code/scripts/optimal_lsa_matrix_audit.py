"""Independently check the saved optimal/LSA comparison, without simulation.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/optimal_lsa_matrix_audit.py

This reproduces the arithmetic and 36 resolvent checks in the read-only
review. It does not import the producer, evaluate a Bayes row integral, or
generate a source, path, or auxiliary count sample. Input hashes identify
the saved data checked. Numerical sensitivities are not certified bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def audit(source_path, smoke=False):
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    assert source["complete"] and len(source["cells"]) == 18
    assert source["repetitions"] == 16 and source["se_ddof"] == 1
    producer = ROOT / "code/scripts/optimal_lsa_matrix_check.py"
    assert sha(producer) == source["script_sha256"]
    for name, expected in source["evaluator_hashes"].items():
        assert sha(producer.with_name(name)) == expected
    cells = source["cells"]
    if smoke:
        cells = [next(c for c in cells if (c["d"], c["L"], c["alpha"]) == (256, 2, .75))]

    maximum_summary_error = 0.
    summary_entries = 0
    raw_files, results, resolvent_sources = [], [], []
    resolvent_errors = dict(occupation_absolute=0., expected_known_bits=0.,
                            scalar_sum_bits=0., initial_cost_bits=0.)
    resolvent_checks = 0
    for cell in cells:
        raw_path = ROOT / cell["raw_path"]
        digest = sha(raw_path)
        assert digest == cell["raw_sha256"]
        raw_files.append(dict(path=relative(raw_path), sha256=digest))
        with np.load(raw_path) as raw:
            n = raw["tokens"]
            known = raw["known_table_expected_full_cost"]
            realized = raw["known_table_realized_full_cost"]
            bayes = raw["bayes_full_cost"]
            scalar = raw["J_finite_occupation"]
            f, c = raw["lsa_frequency_full_cost"], raw["lsa_context_full_cost"]
            ff, cf = raw["lsa_frequency_formula_full_cost"], raw["lsa_context_formula_full_cost"]
            allowance = math.lgamma(cell["d"] + 1) / math.log(2.) + math.log2(6)
            assert abs(allowance - cell["knowledge_allowance_bits"]) < 1e-11
            identities = [
                bayes - raw["initial_bits"][:, None] - raw["bayes_row_costs"].sum(2),
                raw["bayes_redundancy"] - (bayes - realized),
                raw["predicted_finite_full_cost"] - (known + scalar),
                raw["bayes_plus_knowledge_allowance_full_cost"] - bayes - allowance,
            ]
            assert max(float(abs(value).max()) for value in identities) < 1e-8
            differences = dict(
                redundancy_minus_J_finite_occupation=bayes-realized-scalar,
                bayes_raw_minus_finite_prediction=bayes-known-scalar,
                lsa_frequency_minus_bayes=f-bayes, lsa_context_minus_bayes=c-bayes,
                lsa_frequency_minus_finite_prediction=f-known-scalar,
                lsa_context_minus_finite_prediction=c-known-scalar,
                lsa_frequency_actual_minus_formula=f-ff,
                lsa_context_actual_minus_formula=c-cf,
                lsa_actual_gap=f-c, lsa_formula_gap=ff-cf,
                lsa_frequency_minus_hierarchy_upper_endpoint=f-bayes-allowance,
                lsa_context_minus_hierarchy_upper_endpoint=c-bayes-allowance,
            )
            for section, arrays in (
                ("costs", {key: raw[key] for key in cell["costs"]}),
                ("paired_differences", differences),
            ):
                assert arrays.keys() == cell[section].keys()
                for key, values in arrays.items():
                    mean = values.mean(0)
                    se = np.sqrt(((values-mean)**2).sum(0) / (len(values)*(len(values)-1)))
                    for field, truth in (("mean", mean), ("SE", se)):
                        error = float(abs(truth - np.asarray(cell[section][key][field])).max())
                        maximum_summary_error = max(maximum_summary_error, error)
                        summary_entries += len(truth)
                        assert error < 1e-8, (section, key, field, error)

            if cell["d"] == 256:
                original_path = ROOT / cell["source"]
                original_sha = sha(original_path)
                assert original_sha == cell["source_sha256"]
                resolvent_sources.append(dict(path=relative(original_path), sha256=original_sha))
                with np.load(original_path) as original:
                    q, tables = original["q"], original["transition_matrices"]
                    sums, stationary = original["normalized_row_sums"], original["stationary_laws"]
                    initial = original["initial_states"]
                    record = source["scalar_curves"][str(cell["L"])]["record"]
                    x, y = np.asarray(record["x"]), np.asarray(record["r_bits"])
                    error = float(abs(raw["initial_bits"] + np.log2(q[initial])).max())
                    resolvent_errors["initial_cost_bits"] = max(resolvent_errors["initial_cost_bits"], error)
                    assert error < 1e-12
                    for rep in (0, 15):
                        table, pi = tables[rep], stationary[rep]
                        # Sum the transient part through the fundamental matrix,
                        # independently of the producer's binary average recursion.
                        fundamental = np.linalg.inv(np.eye(len(q))-table+np.ones((len(q), 1))*pi[None, :])
                        row_entropy = -(table*np.log2(table)).sum(1)
                        for checkpoint in (0, 4, 9):
                            t = int(raw["transitions"][checkpoint])
                            end = q @ np.linalg.matrix_power(table, t)
                            occupation = pi + (q-end) @ fundamental / t
                            error = float(abs(occupation-raw["expected_occupation_laws"][rep, checkpoint]).max())
                            resolvent_errors["occupation_absolute"] = max(resolvent_errors["occupation_absolute"], error)
                            expected = -float(q @ np.log2(q)) + t*(occupation @ row_entropy)
                            error = abs(expected-known[rep, checkpoint])
                            resolvent_errors["expected_known_bits"] = max(resolvent_errors["expected_known_bits"], error)
                            rates = t*occupation[:, None]*q[None, :]/sums[rep, :, None]
                            assert rates.min() > x[0] and rates.max() < x[-1]
                            predicted = float(np.interp(np.log(rates), np.log(x), y).sum())
                            error = abs(predicted-scalar[rep, checkpoint])
                            resolvent_errors["scalar_sum_bits"] = max(resolvent_errors["scalar_sum_bits"], error)
                            resolvent_checks += 1

            learning_relative = (bayes-realized).mean(0)/scalar.mean(0)-1
            total_relative = bayes.mean(0)/(known+scalar).mean(0)-1
            difference = bayes-realized-scalar
            paired_z = difference.mean(0)/(difference.std(0, ddof=1)/4)
            hierarchy_difference = c-bayes-allowance
            results.append(dict(
                d=cell["d"], L=cell["L"], alpha=cell["alpha"],
                tokens_range=[int(n[0]), int(n[-1])],
                learning_relative_difference_range=[float(learning_relative.min()), float(learning_relative.max())],
                maximum_absolute_total_relative_difference=float(abs(total_relative).max()),
                learning_difference_paired_SE_units_range=[float(paired_z.min()), float(paired_z.max())],
                central_checkpoint=dict(tokens=int(n[4]),
                    learning_relative_difference=float(learning_relative[4]),
                    total_relative_difference=float(total_relative[4]),
                    lsa_context_minus_bayes_bits=float((c-bayes).mean(0)[4]),
                    lsa_context_minus_hierarchy_upper_endpoint_bits=float(hierarchy_difference.mean(0)[4]),
                    lsa_context_minus_hierarchy_upper_endpoint_SE_bits=float(hierarchy_difference.std(0, ddof=1)[4]/4)),
                producer_summed_row_refinement_bits=cell["validation"]["maximum_summed_quadrature_refinement_bits"],
                producer_half_grid_scalar_sum_difference_bits=cell["validation"]["maximum_half_grid_scalar_sum_difference_bits"],
            ))

    assert resolvent_errors["occupation_absolute"] < 1e-11
    assert resolvent_errors["expected_known_bits"] < 1e-6
    assert resolvent_errors["scalar_sum_bits"] < 1e-6
    assert source_path.read_bytes() == source_bytes, "Producer summary changed during the audit"
    return dict(
        schema_version=1, status="passed", smoke=smoke,
        source=relative(source_path), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        script=relative(Path(__file__)), script_sha256=sha(Path(__file__)),
        producer_sha256=source["script_sha256"], evaluator_hashes=source["evaluator_hashes"],
        command_argv=sys.argv, raw_files=raw_files, resolvent_source_files=resolvent_sources,
        checks=dict(cells=len(cells), independently_recomputed_summary_entries=summary_entries,
                    maximum_summary_absolute_error=maximum_summary_error,
                    independent_resolvent_calculations=resolvent_checks,
                    resolvent_scope="d=256, all six (L,alpha) cells; replicates0,15 and checkpoints0,4,9" if not smoke else "One d256 cell; replicates0,15 and checkpoints0,4,9",
                    maximum_resolvent_errors=resolvent_errors),
        reviewed_calculation=[
            "Matched Bayes uses the labeled q and fixed true source depth; row probabilities are ordered-sequence probabilities, with no multinomial coefficient.",
            "Bayes includes -log2 q(X0). Expected known-table cost includes H(q) and finite expected occupations.",
            "The scalar prediction uses latent tables and normalizers, not observed counts or codelengths. Its actual-chain approximation error is not bounded by this audit.",
            "The hierarchy endpoint adds log2(d!)+log2(6) to measured Bayes cost; the deterministic shift leaves its SE unchanged. This is not a bound on the LSA algorithm.",
            "SEs use sixteen independent replicates and sample variance with ddof1; differences are paired before calculating SE. Checkpoints are correlated and their SEs are pointwise.",
            "A negative LSA-minus-hierarchy-upper-endpoint difference does not establish that LSA beats the hierarchical predictor.",
            "No Bayes evaluations or simulations are run here. The Bayes evaluator was checked separately; quoted producer refinement diagnostics are numerical sensitivities, not certified error bounds.",
        ],
        cells=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/mechanism/optimal_lsa_matrix_check.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out or (Path("/private/tmp/optimal_lsa_matrix_audit_smoke.json") if args.smoke
                       else ROOT/"results/mechanism/optimal_lsa_matrix_audit.json")
    result = audit(args.source, args.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"], **result["checks"], out=relative(out)), indent=2))


if __name__ == "__main__":
    main()
