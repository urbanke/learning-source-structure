"""Bounded, nonresumable auxiliary-power benchmark on one declared prefix."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
sys.path.insert(0, str(ROOT / 'code/scripts'))
from conditional_shared_base_benchmark import inputs
from eval_shared_base_models import atomic_json, atomic_npz, sha_file
from full_context_statistics import compare_prediction, diagnostic_gates
from ranked_lsa import RankedLSARowEvidence
from shared_base_chain_runner import _fingerprint
from shared_base_efficiency import SharedBaseEfficiencyMCMC, SharedBaseEfficiencyRowMCMC
from stable_chain_statistics import stable_predictive_moments, stable_chain_summary
from validate_shared_base_mcmc import json_safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'output/conditional_shared_base_benchmark_2026-09-15/a1_L4')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prefix', type=int, default=64)
    p.add_argument('--alpha-order', type=int, default=16)
    p.add_argument('--power', type=float, default=.5)
    p.add_argument('--auxiliary', choices=['tempered', 'row'], default='tempered')
    p.add_argument('--concentration', type=float, default=1.0)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--burn', type=int, default=16384)
    p.add_argument('--draws', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20261003)
    args = p.parse_args()
    if (args.out / 'report.json').exists() or (args.out / 'samples.npz').exists():
        raise ValueError('refusing to overwrite previous benchmark')
    tokens, oracle, predictions, input_identity = inputs(args.input, 0, args.prefix)
    nodes, weights = leggauss(args.alpha_order)
    kernel = RankedLSARowEvidence(64, 1+nodes/2, term_cache_size=16384)
    sampler = (SharedBaseEfficiencyMCMC(64, kernel, weights/2,
                                       surrogate_steps=args.steps, auxiliary_power=args.power)
               if args.auxiliary == 'tempered' else
               SharedBaseEfficiencyRowMCMC(64, kernel, weights/2,
                                          surrogate_steps=args.steps, concentration=args.concentration))
    target_fingerprint = _fingerprint(sampler, tokens)
    estimate = sampler.sample(tokens, chains=4, burn=args.burn, draws=args.draws,
                              thin=1, seed=args.seed, retain_samples=True)
    samples = estimate.pop('samples')
    moments = stable_predictive_moments(samples['predictive_probs'])
    # Replace inherited classic summaries with the established stable ones.
    pred = stable_chain_summary(samples['predictive_probs'])
    depth = stable_chain_summary(samples['posterior_depth'])
    scalar = stable_chain_summary(samples['alpha_and_loglik'])
    estimate['predictive_probs'], estimate['mcse_probs'] = pred['mean'], pred['mcse']
    estimate['posterior_depth'], estimate['mcse_depth'] = depth['mean'], depth['mcse']
    diag = estimate['diagnostics']
    diag.update(rhat_probs=pred['rhat'].tolist(), ess_probs=pred['ess'].tolist(),
                rhat_depth=depth['rhat'].tolist(), rhat_alpha_and_loglik=scalar['rhat'].tolist(),
                chain_mean_probs=pred['chain_means'].tolist())
    metrics = {name: compare_prediction(estimate['predictive_probs'], entry['prediction'], oracle, moments)
               for name, entry in predictions.items()}
    atomic_npz(args.out / 'samples.npz', **samples)
    scalar_mcse = metrics['d64']['gap_mcse_bits']
    summary = dict(auxiliary=args.auxiliary, power=args.power, concentration=args.concentration,
                   steps=args.steps,
                   seconds=estimate['elapsed_seconds'],
                   variance_times_sampling_seconds=scalar_mcse**2*estimate['elapsed_seconds'],
                   mcse_gap_bits=scalar_mcse, gap=metrics['d64']['true_loss_gap_bits'],
                   rhat_max=float(max(pred['rhat'])), min_probability_ess=float(min(pred['ess'])),
                   gates=diagnostic_gates(estimate),
                   block_acceptance=float(np.sum(np.array(diag['move_accepts'])[:, 0])
                                          /np.sum(np.array(diag['move_attempts'])[:, 0])))
    report = dict(schema='shared-base-efficiency-tempered-v1', inputs=input_identity,
                  numerical_problem=target_fingerprint, proposal_spec=sampler.proposal_spec,
                  script_sha256=sha_file(__file__), estimate=estimate, metrics=metrics,
                  summary=summary, samples_sha256=sha_file(args.out / 'samples.npz'),
                  scientific_reference_validated=False,
                  note='One-prefix algorithm comparison only; no source-conditioned mean or acceptance as a Bayesian reference.')
    atomic_json(args.out / 'report.json', json_safe(report))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
