"""Separate resumable surrogate-block proposal evaluation on the frozen cohort.

This prototype writes to its own output folder and binds its proposal,
implementation and distinct seed protocol into every reference identity.
It does not replace the original full-context reference calculation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from eval_shared_base_models import HERE, atomic_json, atomic_npz, fingerprint, sha_file
from eval_shared_base_reference import load_inputs, selected_grid_checks
from full_context_statistics import compare_prediction, diagnostic_gates
from stable_chain_statistics import stable_predictive_moments as predictive_moments
from surrogate_chain_runner import SurrogateChainRun
from ranked_lsa import RankedLSARowEvidence
from shared_base_surrogate import SharedBaseSurrogateMCMC
from source import seed_of
from validate_shared_base_mcmc import json_safe


SEED_TAG = "shared-base-full-context-surrogate-block-v1"
CODE_FILES = (
    "eval_shared_base_surrogate.py", "surrogate_chain_runner.py", "shared_base_surrogate.py",
    "eval_shared_base_reference.py", "full_context_statistics.py", "eval_shared_base_models.py",
    "validate_shared_base_mcmc.py", "stable_chain_statistics.py",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=Path("output/shared_base_full_context"))
    result.add_argument("--out", type=Path, default=Path("output/shared_base_surrogate_context"))
    result.add_argument("--index", type=int, required=True)
    result.add_argument("--prefix", type=int, default=1024)
    result.add_argument("--alpha-order", type=int, default=64)
    result.add_argument("--seed", type=int, default=20260924)
    result.add_argument("--chains", type=int, default=4)
    result.add_argument("--burn", type=int, default=512)
    result.add_argument("--thin", type=int, default=1)
    result.add_argument("--initial-draws", type=int, default=256)
    result.add_argument("--max-draws", type=int, default=65536)
    result.add_argument("--block-draws", type=int, default=2048)
    result.add_argument("--term-cache-size", type=int, default=16384)
    result.add_argument("--surrogate-steps", type=int, default=64)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    if args.alpha_order < 2 or args.initial_draws < 16 or args.max_draws < args.initial_draws:
        arguments.error("invalid exponent order or draw budget")
    if args.out.resolve() == args.input.resolve():
        arguments.error("surrogate-block references require a separate output folder")
    _, tokens, oracle, models, inputs = load_inputs(args.input, args.index, args.prefix)
    nodes, weights = leggauss(args.alpha_order)
    nodes, weights = 1 + nodes / 2, weights / 2
    kernel = RankedLSARowEvidence(64, nodes, term_cache_size=args.term_cache_size)
    sampler = SharedBaseSurrogateMCMC(64, kernel, weights, surrogate_steps=args.surrogate_steps)
    seed = seed_of(SEED_TAG, args.seed, args.index, args.prefix, args.alpha_order)
    stem = f"seq_{args.index:06d}_t{args.prefix:04d}"
    folder = args.out / "refs" / f"alpha{args.alpha_order}"
    report_path, moments_path = folder / (stem + ".json"), folder / (stem + ".npz")
    chain_path = args.out / "chains" / f"alpha{args.alpha_order}" / (stem + ".npz")
    run = SurrogateChainRun.load(sampler, tokens, chain_path) if chain_path.exists() else SurrogateChainRun(
        sampler, tokens, chains=args.chains, burn=args.burn, thin=args.thin, seed=seed)
    if (run.seed, run.chains, run.burn, run.thin) != (seed, args.chains, args.burn, args.thin):
        raise ValueError("saved chain configuration differs from request")
    code_hashes = {name: sha_file(HERE / name) for name in CODE_FILES}
    identity = dict(inputs=inputs, numerical_problem=run.fingerprint, code_sha256=code_hashes,
                    proposal_descriptor=run.proposal_descriptor, sampling_seed_tag=SEED_TAG)
    old = json.loads(report_path.read_text()) if report_path.exists() else None
    if old is not None and old["identity"] != identity:
        raise ValueError("comparison inputs or proposal code changed; use another output folder")
    history = [] if old is None else old["sampling_rounds"]

    def progress(event):
        print(json.dumps(dict(sequence=args.index, alpha_order=args.alpha_order, **event)), flush=True)
        if event["phase"] == "burn" or event["draws"] == event["requested_draws"]:
            run.save(chain_path)

    budget = max(args.initial_draws, int(run.draw_counts.max()))
    if budget > args.max_draws:
        raise ValueError("maximum budget cannot discard existing retained draws")
    while True:
        estimate = run.extend(budget, block_draws=args.block_draws, progress=progress)
        moments = predictive_moments(run.samples["predictive_probs"])
        if (not np.allclose(moments["mean"], estimate["predictive_probs"], rtol=0, atol=2e-14)
                or not np.allclose(moments["mcse_probs"], estimate["mcse_probs"], rtol=1e-11, atol=1e-15)):
            raise ValueError("independent moment and sampler summaries disagree")
        metrics = {name: compare_prediction(estimate["predictive_probs"], item["prediction"], oracle, moments)
                   for name, item in models.items()}
        gates = diagnostic_gates(estimate)
        passed = all(gates.values()) and all(m["metric_precision_passed"] for m in metrics.values())
        round_report = dict(draws_per_chain=budget, diagnostic_gates=gates,
            model_metrics=metrics, sampling_precision_passed=passed,
            max_rhat=max(v for k in ("rhat_probs", "rhat_depth", "rhat_alpha_and_loglik")
                         for v in estimate["diagnostics"][k] if v is not None),
            min_probability_ess=min(estimate["diagnostics"]["ess_probs"]),
            max_probability_mcse=float(np.max(estimate["mcse_probs"])),
            elapsed_seconds=run.elapsed_seconds)
        if not history or history[-1]["draws_per_chain"] != budget:
            history.append(round_report)
        else:
            history[-1] = round_report
        atomic_npz(moments_path, **moments)
        grid_checks = selected_grid_checks(sampler, tokens, estimate) if passed or budget >= args.max_draws else []
        report = dict(schema="shared-base-full-context-reference-v1", identity=identity,
            reference_id=fingerprint(identity), sequence_index=args.index, prefix_length=args.prefix,
            source_id=inputs["source_id"], tokens_sha256=inputs["tokens_sha256"],
            prefix_tokens_sha256=inputs["prefix_tokens_sha256"], estimate=estimate,
            settings=dict(alpha_order=args.alpha_order, alpha_nodes=nodes, alpha_weights=weights,
                depth_prior=sampler.depth_prior, depths=sampler.depths,
                row_grid=dict(lower=kernel.u_lo, upper=kernel.u_hi, step=kernel.step),
                term_cache_size=args.term_cache_size, sampling_master_seed=args.seed,
                sampling_seed_tag=SEED_TAG, surrogate_steps=args.surrogate_steps, seed=seed, chains=args.chains, burn=args.burn, thin=args.thin),
            model_metrics=metrics, diagnostic_gates=gates,
            sampling_precision_passed=passed, final_state_grid_checks=grid_checks,
            local_validation_passed=passed and bool(grid_checks) and all(c["passed"] for c in grid_checks),
            sampling_rounds=history, moments_file=moments_path.name,
            moments_sha256=sha_file(moments_path), chain_checkpoint=str(chain_path.resolve()),
            cache=kernel.cache_info(),
            limitation="Separate surrogate-block proposal evaluation. Exponent refinement is checked separately; chain diagnostics do not certify convergence.")
        atomic_json(report_path, json_safe(report))
        print(json.dumps(json_safe(dict(sequence=args.index, alpha_order=args.alpha_order,
                                       **round_report))), flush=True)
        if passed or budget >= args.max_draws:
            break
        budget = min(args.max_draws, 2 * budget)
    if not report["local_validation_passed"]:
        raise SystemExit("reference remains unresolved at current budget; retained for extension")


if __name__ == "__main__":
    main()
