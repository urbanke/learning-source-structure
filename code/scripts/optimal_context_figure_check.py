"""Matched Bayes costs on the saved L2,alpha1,d256 actual-chain paths.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/optimal_context_figure_check.py

No new latent tables, initial states or paths are sampled. The prediction
uses an independently calculated scalar information curve and exact finite
expected occupations. It does not use observed transition counts or costs.
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

from context_finite_prediction_check import ScalarCurve
from crossing_finite_count_prediction import Occupations
from crossing_count_prediction import ROOT,relative,moments,sha
from iid_depth_prediction import ScalarInformation,scalar_validation,sealed_probability_validation,GAMMA,LN2
from tilted_row_reference import TiltedRowReference,DEFAULT_STORE


def run(args):
    started=time.monotonic()
    source_json=args.source.with_suffix(".json")
    source=json.loads(source_json.read_text())
    assert (source["L"],source["alpha"],source["d"],source["repetitions"])==(2,1.,256,16)
    assert source["raw_sha256"]==sha(args.source)
    with np.load(args.source) as inp:
        T=inp["transition_matrices"]
        pi=inp["stationary_laws"]
        S=inp["normalized_row_sums"]
        q=inp["q"]
        counts=inp["chain_transition_counts"]
        saved_r=inp["expected_occupation_laws"]
        initial=inp["initial_states"]
        ns=inp["tokens"]
        ts=inp["transitions"]
        f=inp["chain_frequency_codelength"]
        c=inp["chain_context_codelength"]
    assert counts.shape==(16,10,256,256) and np.array_equal(ts,ns-1)
    assert np.all(counts.sum((2,3))==ts[None,:])
    Hq=-float(q@np.log2(q))
    initial_bits=-np.log2(q[initial])
    row_entropies=-(T*np.log2(T)).sum(axis=2)
    expected_known=np.zeros((16,10))
    occupations=np.empty_like(saved_r)
    occupation_error=0.
    for rep in range(16):
        op=Occupations(T[rep],int(ts[-1]))
        for j,t in enumerate(ts):
            r,_=op.at(q,int(t))
            occupation_error=max(occupation_error,float(abs(r-saved_r[rep,j]).max()))
            occupations[rep,j]=r
            expected_known[rep,j]=Hq+t*(r@row_entropies[rep])
    assert occupation_error<1e-12
    # Scalar information is calculated from the product-exponential prior,
    # independently of both the Bayes row evaluator and these sample paths.
    scalar_start=time.monotonic()
    master=json.loads((ROOT/"results/mechanism/master_curves.json").read_text())
    scalar_checks=scalar_validation(master,[2])
    sealed_checks=sealed_probability_validation([2])
    calc=ScalarInformation(2)
    xs=np.geomspace(1e-8,1e8,1025)
    ys=np.asarray([calc(float(x)) for x in xs])
    assert np.all(np.diff(ys)>0) and np.all(ys>0)
    record=dict(x=xs.tolist(),r_bits=ys.tolist(),slope0_bits=2*(1-GAMMA)/LN2,
                tail_constant_bits=calc.tail_constant(),diagnostics=calc.diagnostics())
    curve=ScalarCurve(record)
    coarse=ScalarCurve(dict(record,x=xs[::2].tolist(),r_bits=ys[::2].tolist()))
    scalar_seconds=time.monotonic()-scalar_start
    print(f"L2 scalar curve and independent checks completed in {scalar_seconds:.2f}s",flush=True)
    joint_stationary=np.zeros((16,10))
    joint_finite=np.zeros((16,10))
    interpolation_error=np.zeros((16,10,2))
    for rep in range(16):
        weights=q[None,:]/S[rep,:,None]
        for j,t in enumerate(ts):
            stationary_rates=t*pi[rep,:,None]*weights
            finite_rates=t*occupations[rep,j,:,None]*weights
            joint_stationary[rep,j]=curve(stationary_rates).sum()
            joint_finite[rep,j]=curve(finite_rates).sum()
            interpolation_error[rep,j]=[abs(coarse(stationary_rates).sum()-joint_stationary[rep,j]),
                                      abs(coarse(finite_rates).sum()-joint_finite[rep,j])]
    assert interpolation_error.max()<5.
    bayes_rows=np.zeros((16,10,256))
    refinement_rows=np.zeros_like(bayes_rows)
    cache={}
    ref=TiltedRowReference(q,2,store=args.store,step=1/128)
    evaluation_start=time.monotonic()
    try:
        for rep in range(16):
            for j in range(10):
                for a,row in enumerate(counts[rep,j]):
                    labels=np.flatnonzero(row)
                    if not len(labels):
                        continue
                    # Labelled counts are required for the nonuniform q prior.
                    key=tuple((int(b),int(row[b])) for b in labels)
                    if key not in cache:
                        logq,diag=ref.log_probability(row)
                        cache[key]=(-logq/LN2,diag["coarse_fine_difference_bits"])
                    bayes_rows[rep,j,a],refinement_rows[rep,j,a]=cache[key]
            if rep in [0,3,7,15]:
                print(f"Bayes rows: {rep+1}/16 tables, {len(cache)} labelled patterns, {time.monotonic()-evaluation_start:.2f}s",flush=True)
    finally:
        ref.close()
    bayes_full=initial_bits[:,None]+bayes_rows.sum(2)
    known_realized=initial_bits[:,None]-(counts*np.log2(T)[:,None,:,:]).sum((2,3))
    redundancy=bayes_full-known_realized
    # A fixed, prespecified coefficient-one control variate. Both this and
    # the raw Bayes code cost are unbiased estimates of the same mean.
    bayes_control=expected_known+redundancy
    predicted_finite=expected_known+joint_finite
    predicted_stationary=expected_known+joint_stationary
    raw=dict(tokens=ns,transitions=ts,q=q,initial_bits=initial_bits,
        expected_occupation_laws=occupations,row_entropies=row_entropies,
        bayes_row_costs=bayes_rows,bayes_full_cost=bayes_full,
        known_table_realized_full_cost=known_realized,known_table_expected_full_cost=expected_known,
        bayes_redundancy=redundancy,bayes_control_variate_full_cost=bayes_control,
        J_finite_occupation=joint_finite,J_stationary=joint_stationary,
        predicted_finite_full_cost=predicted_finite,predicted_stationary_full_cost=predicted_stationary,
        lsa_frequency_full_cost=f,lsa_context_full_cost=c,
        quadrature_refinement_rows_bits=refinement_rows,scalar_interpolation_refinement_bits=interpolation_error)
    args.raw.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.raw,**raw)
    fields=[name for name,value in raw.items() if np.asarray(value).shape==(16,10)]
    summary={name:moments(raw[name]) for name in fields}
    differences={
        "redundancy_minus_J_finite_occupation":redundancy-joint_finite,
        "redundancy_minus_J_stationary":redundancy-joint_stationary,
        "bayes_raw_minus_finite_prediction":bayes_full-predicted_finite,
        "lsa_frequency_minus_bayes":f-bayes_full,
        "lsa_context_minus_bayes":c-bayes_full,
        "lsa_frequency_minus_bayes_control_variate":f-bayes_control,
        "lsa_context_minus_bayes_control_variate":c-bayes_control}
    relative=(redundancy.mean(0)/joint_finite.mean(0)-1)
    result=dict(schema_version=1,complete=True,d=256,alpha=1.,L=2,repetitions=16,
        source=relative_path(args.source),source_sha256=sha(args.source),
        source_summary=relative_path(source_json),source_summary_sha256=sha(source_json),
        raw_path=relative_path(args.raw),raw_sha256=sha(args.raw),tokens=ns.tolist(),transitions=ts.tolist(),
        source_coding_cutoff=source["l_max"],se_ddof=1,
        initialization="The saved initial state is drawn from known q independently of the table. Every full Bayes/known-T cost includes its q initial factor; the analytic expected initial cost is H(q).",
        formula="Predicted full Bayes cost = H(q)+t*sum_a r_t(a)H(T_a)+sum_ab r_2(t*r_t(a)*q_b/S_a); averages are over the saved sixteen latent tables. Stationary pi replaces r_t only in the separate diagnostic.",
        qualifications=["No new source, initial-state or path draws; this reuses the same samples as the LSA comparison.",
            "Finite expected occupations are used by preselection, not chosen after examining the prediction error.",
            "The scalar curve and predicted full costs do not use observed counts, path likelihoods, measured Bayes costs or fitted coefficients.",
            "The finite-occupation scalar expression remains an approximation for an actual chain.",
            "The control-variate cost equals analytic expected known-T cost plus the measured Bayes/known-T likelihood ratio; it is an unbiased estimator using a fixed coefficient-one control variate. Raw Bayes costs are retained separately. In this sample, subtracting the prediction from this estimator reduces the paired residual's standard error, but the total-cost estimator itself has a slightly larger standard error.",
            "Numerical refinement comparisons are separate from sampling uncertainty and are not certified total-error bounds.",
            "The matched Bayes expected code cost is no larger than either proper deployed LSA code under this source prior. No optimal-versus-actual-code crossing is inferred."],
        costs=summary,paired_differences={name:moments(value) for name,value in differences.items()},
        measured_redundancy_over_finite_prediction_minus_one=relative.tolist(),
        scalar_curve=record,validation=dict(scalar_checks=scalar_checks,sealed_probability_checks=sealed_checks,
            maximum_occupation_recomputation_error=occupation_error,
            maximum_summed_quadrature_refinement_bits=float(refinement_rows.sum(2).max()),
            maximum_single_row_quadrature_refinement_bits=float(refinement_rows.max()),
            maximum_half_grid_scalar_sum_difference_bits=float(interpolation_error.max())),
        scalar_seconds=scalar_seconds,bayes_evaluation_seconds=time.monotonic()-evaluation_start,
        unique_labelled_count_patterns=len(cache),elapsed_seconds=time.monotonic()-started,
        script_sha256=sha(Path(__file__)),row_evaluator_sha256=sha(Path(__file__).with_name("tilted_row_reference.py")),
        scalar_evaluator_sha256=sha(Path(__file__).with_name("iid_depth_prediction.py")),command_argv=sys.argv)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"elapsed_seconds":result["elapsed_seconds"],"relative_redundancy_errors":relative.tolist(),"maximum_quadrature_refinement_bits":result["validation"]["maximum_summed_quadrature_refinement_bits"]},indent=2))


def relative_path(path):
    return relative(path.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"output/crossing_finite_count_prediction/L2_a1_d256.npz")
    parser.add_argument("--store",type=Path,default=DEFAULT_STORE)
    parser.add_argument("--raw",type=Path,default=ROOT/"output/optimal_context_figure_check.npz")
    parser.add_argument("--out",type=Path,default=ROOT/"results/mechanism/optimal_context_figure_check.json")
    run(parser.parse_args())


if __name__=="__main__":
    main()
