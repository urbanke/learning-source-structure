"""One full-family Bayesian reference for a fixed-source evaluation cohort.

Preserve the old per-sequence scalar precision decision, but separately export
the numerical information needed to validate a mean across all declared cases.
No cohort mean is produced by this worker and no failed case is dropped.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"code/transformer"))
from shared_base_conditional_cohort import inputs
from shared_base_precision_cohort import case_precision
from shared_base_curve_runner import CurveChainRun
from shared_base_surrogate import SharedBaseSurrogateMCMC
from shared_base_efficiency import SharedBaseEfficiencyMCMC, SharedBaseEfficiencyRowMCMC
from ranked_lsa import RankedLSARowEvidence
from eval_shared_base_reference import selected_grid_checks
from eval_shared_base_models import atomic_json, atomic_npz, fingerprint, sha_file
from full_context_statistics import compare_prediction, diagnostic_gates
from stable_chain_statistics import stable_predictive_moments
from source import seed_of
from validate_shared_base_mcmc import json_safe

SCHEMA = "shared-base-curve-reference-v1"


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("reference already running") from error
        yield


def run_job(args):
    tokens, oracle, models, input_identity = inputs(args.input, args.index, args.prefix)
    nodes, weights = leggauss(args.alpha_order)
    kernel = RankedLSARowEvidence(64, 1+nodes/2, term_cache_size=16384)
    settings = dict(surrogate_steps=args.steps)
    sampler_class = {"iid": SharedBaseSurrogateMCMC, "tempered": SharedBaseEfficiencyMCMC,
                     "row": SharedBaseEfficiencyRowMCMC}[args.proposal]
    if args.proposal == "tempered": settings["auxiliary_power"] = args.power
    if args.proposal == "row": settings["concentration"] = args.concentration
    sampler = sampler_class(64, kernel, weights/2, **settings)
    seed = seed_of(SCHEMA, args.seed, args.input.name, args.index, args.prefix, args.alpha_order)
    stem = f"seq_{args.index:06d}_t{args.prefix:04d}"
    folder = args.out/args.input.name/f"alpha{args.alpha_order}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint, report_path = folder/(stem+"_chains.npz"), folder/(stem+".json")
    with exclusive(folder/(stem+".lock")):
        run = CurveChainRun.load(sampler, tokens, checkpoint) if checkpoint.exists() else CurveChainRun(
            sampler, tokens, chains=4, burn=args.burn, thin=1, seed=seed)
        if (run.chains, run.burn, run.thin, run.seed) != (4,args.burn,1,seed):
            raise ValueError("saved sampling settings differ")
        code = {name: sha_file(Path(__file__).with_name(name)) for name in (
            "shared_base_curve_reference.py", "shared_base_conditional_cohort.py",
            "shared_base_precision_cohort.py")}
        identity = dict(inputs=input_identity, target=run.fingerprint,
                        proposal=run.proposal_descriptor, master_sampling_seed=args.seed,
                        sampling_seed=seed, burn=args.burn,
                        scoring_code=code)
        old = json.loads(report_path.read_text()) if report_path.exists() else None
        if old and old["identity"] != identity:
            raise ValueError("reference identity changed")
        history = [] if old is None else old["history"]
        budget = max(args.initial_draws, int(run.draw_counts.max()))
        if budget > args.max_draws: raise ValueError("cannot discard saved draws")
        def progress(event):
            print(json.dumps(dict(setting=args.input.name,index=args.index,time=args.prefix,
                                  order=args.alpha_order,**event)),flush=True)
            if event["phase"] == "burn" or event["draws"] == event["requested_draws"]:
                run.save(checkpoint)
        while True:
            estimate = run.extend(budget, block_draws=2048, progress=progress)
            samples = run.samples["predictive_probs"]
            moments = stable_predictive_moments(samples)
            precision = case_precision(samples, oracle)
            if not np.allclose(moments["mean"],precision["predictive_probs"],rtol=0,atol=2e-14):
                raise ValueError("independent predictive mean disagreement")
            metrics = {name: compare_prediction(moments["mean"],v["prediction"],oracle,moments)
                       for name,v in models.items()}
            gates = diagnostic_gates(estimate)
            mixing = all(gates.values()) and precision["scalar_mixing_passed"]
            usable = mixing and precision["jensen_local_check_passed"]
            grids = selected_grid_checks(sampler,tokens,estimate) if usable or budget>=args.max_draws else []
            row_passed = bool(grids) and all(v["passed"] for v in grids)
            old_passed = all(gates.values()) and all(v["metric_precision_passed"] for v in metrics.values()) and row_passed
            record = dict(case_id=args.index, sampling_seed=seed, precision=precision,
                          full_prior_valid=True, provenance_valid=True,
                          mixing_passed=bool(mixing), row_grid_passed=bool(row_passed),
                          per_sequence_precision_passed=bool(old_passed))
            current = dict(draws=budget,seconds=run.elapsed_seconds,
                           ready_for_cohort_check=bool(usable and row_passed),
                           per_sequence_precision_passed=bool(old_passed),
                           mcse_bits=precision["mcse_bits"],jensen_bits=precision["jensen_proxy_bits"],
                           gates=gates,scalar_ess=precision["scalar_ess"])
            if history and history[-1]["draws"]==budget: history[-1]=current
            else: history.append(current)
            moments_path=folder/(stem+"_moments.npz")
            atomic_npz(moments_path,**moments)
            result = dict(schema=SCHEMA,reference_id=fingerprint(identity),identity=identity,
                          setting=args.input.name,index=args.index,time=args.prefix,order=args.alpha_order,
                          estimate=estimate,record=record,model_metrics=metrics,
                          fine_grid_checks=grids,history=history,checkpoint=str(checkpoint.resolve()),
                          checkpoint_sha256=sha_file(checkpoint),moments_sha256=sha_file(moments_path),
                          ready_for_cohort_check=bool(usable and row_passed),
                          interpretation="Not a validated cohort mean. Every case and both exponent rules must pass the separate cohort audit.")
            atomic_json(report_path,json_safe(result))
            print(json.dumps(json_safe(current)),flush=True)
            if usable and row_passed or budget>=args.max_draws: break
            budget=min(2*budget,args.max_draws)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--index",type=int,required=True)
    parser.add_argument("--prefix",type=int,required=True)
    parser.add_argument("--alpha-order",type=int,default=16)
    parser.add_argument("--proposal",choices=["iid","tempered","row"],default="iid")
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--power",type=float,default=.5)
    parser.add_argument("--concentration",type=float,default=1.)
    parser.add_argument("--burn",type=int,default=16384)
    parser.add_argument("--initial-draws",type=int,default=4096)
    parser.add_argument("--max-draws",type=int,default=65536)
    parser.add_argument("--seed",type=int,default=20261006)
    args=parser.parse_args()
    if args.initial_draws<16 or args.max_draws<args.initial_draws or args.index<0 or args.prefix<1:
        parser.error("invalid reference budget or case")
    result=run_job(args)
    if not result["ready_for_cohort_check"]: raise SystemExit(2)


if __name__=="__main__": main()
