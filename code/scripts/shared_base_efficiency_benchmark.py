"""Bounded comparison of block lengths for the unchanged shared-base target.

This is an algorithm benchmark on declared source index 0, not source-population
evidence. Each configuration has four fresh, independently initialized chains.
Every draw and proposal descriptor is retained under a new output directory;
existing reference chains are neither read as initial states nor overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
sys.path.insert(0, str(ROOT / 'code/scripts'))
from conditional_shared_base_benchmark import inputs
from eval_shared_base_models import atomic_json, atomic_npz, sha_file
from full_context_statistics import compare_prediction, diagnostic_gates
from ranked_lsa import RankedLSARowEvidence
from shared_base_surrogate import SharedBaseSurrogateMCMC
from stable_chain_statistics import stable_predictive_moments
from surrogate_chain_runner import SurrogateChainRun
from validate_shared_base_mcmc import json_safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'output/conditional_shared_base_benchmark_2026-09-15/a1_L4')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prefix', type=int, default=64)
    p.add_argument('--alpha-order', type=int, default=16)
    p.add_argument('--steps', type=int, nargs='+', default=[4, 16, 64])
    p.add_argument('--burn', type=int, default=16384)
    p.add_argument('--draws', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20261003)
    args = p.parse_args()
    tokens, oracle, predictions, input_identity = inputs(args.input, 0, args.prefix)
    nodes, weights = leggauss(args.alpha_order)
    nodes, weights = 1 + nodes/2, weights/2
    # Shared immutable row cache: same prefix/grid/target for all proposals.
    kernel = RankedLSARowEvidence(64, nodes, term_cache_size=16384)
    summaries = []
    for steps in args.steps:
        sampler = SharedBaseSurrogateMCMC(64, kernel, weights, surrogate_steps=steps)
        folder = args.out / f'alpha{args.alpha_order}_t{args.prefix}_steps{steps}'
        checkpoint = folder / 'chains.npz'
        if checkpoint.exists():
            run = SurrogateChainRun.load(sampler, tokens, checkpoint)
        else:
            run = SurrogateChainRun(sampler, tokens, chains=4, burn=args.burn,
                                    thin=1, seed=args.seed)
        if (run.chains, run.burn, run.thin, run.seed) != (4, args.burn, 1, args.seed):
            raise ValueError('saved chain configuration differs')
        def progress(event):
            print(json.dumps(dict(steps=steps, **event)), flush=True)
            run.save(checkpoint)
        estimate = run.extend(args.draws, block_draws=2048, progress=progress)
        samples = run.samples['predictive_probs']
        moments = stable_predictive_moments(samples)
        metrics = {name: compare_prediction(estimate['predictive_probs'], entry['prediction'], oracle, moments)
                   for name, entry in predictions.items()}
        gates = diagnostic_gates(estimate)
        scalar_mcse = metrics['d64']['gap_mcse_bits']
        report = dict(
            schema='shared-base-efficiency-benchmark-v1',
            inputs=input_identity, numerical_problem=run.fingerprint,
            proposal_descriptor=run.proposal_descriptor,
            script_sha256=sha_file(__file__), estimate=estimate,
            metrics=metrics, diagnostic_gates=gates,
            scientific_reference_validated=False,
            note='Algorithm cost comparison only. Independent exponent refinement and row-grid audit are still required.',
            efficiency=dict(variance_times_sampling_seconds=scalar_mcse**2 * run.elapsed_seconds,
                            mcse_gap_bits=scalar_mcse, seconds=run.elapsed_seconds,
                            min_probability_ess=min(estimate['diagnostics']['ess_probs'])))
        atomic_npz(folder / 'moments.npz', **moments)
        atomic_json(folder / 'report.json', json_safe(report))
        summary = dict(steps=steps, **report['efficiency'],
                       gap=metrics['d64']['true_loss_gap_bits'],
                       rhat_max=max(estimate['diagnostics']['rhat_probs']),
                       gates=gates,
                       block_acceptance=float(np.sum(run._accepts[:, 0]) / np.sum(run._attempts[:, 0])))
        summaries.append(summary)
        atomic_json(args.out / 'summary.json', summaries)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
