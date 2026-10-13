"""Approximate shared-base Bayes predictions at one observed prefix.

Uses the existing sealed anchor store in production mode. This is a fixed-
prefix posterior average, not a sequential codelength or evidence estimator.
Use independent seeds, longer chains and exponent/grid refinement to assess
accuracy. A passing diagnostic is not a convergence certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy

from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import PrefixPosterior, SharedBaseMCMC
from source import Source
from validate_shared_base_mcmc import json_safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--n-tokens', type=int, default=128)
    parser.add_argument('--tokens-json', type=Path, help='JSON token array or object containing tokens')
    parser.add_argument('--source-seed', type=int, default=20260914)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--alpha-order', type=int, default=16)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--burn', type=int, default=2048)
    parser.add_argument('--draws', type=int, default=2048)
    parser.add_argument('--thin', type=int, default=2)
    parser.add_argument('--u-lo', type=float, default=-32.)
    parser.add_argument('--u-hi', type=float, default=32.)
    parser.add_argument('--step', type=float, default=1/32)
    parser.add_argument('--term-cache-size', type=int, default=4096, help='Bounded shifted-term cache; 4096 uses about 512 MiB at the default grid')
    parser.add_argument('--check-grid', action='store_true', help='Reevaluate final chain states on wider/finer row grid')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.n_tokens < 1 or args.alpha_order < 1 or args.d < 2:
        parser.error('require d>=2, n-tokens>=1, alpha-order>=1')
    source_info = None
    if args.tokens_json:
        loaded = json.loads(args.tokens_json.read_text())
        tokens = np.asarray(loaded['tokens'] if isinstance(loaded, dict) else loaded)
        if len(tokens) < args.n_tokens:
            parser.error('token file is shorter than n-tokens')
        tokens = tokens[:args.n_tokens]
    else:
        tokens, params = Source(args.d, args.n_tokens, alpha=(.5, 1.5), ls=(0, 8), permute=True).batch(1, np.random.default_rng(args.source_seed))
        tokens = tokens[0]
        source_info = dict(seed=args.source_seed, parameters={k: v[0].tolist() for k, v in params.items()},
                           note='Recorded only for reproducibility; never passed to the predictor.')
    nodes, weights = leggauss(args.alpha_order)
    nodes, weights = 1 + nodes/2, weights/2
    kernel = RankedLSARowEvidence(args.d, nodes, u_lo=args.u_lo, u_hi=args.u_hi, step=args.step,
                                 term_cache_size=args.term_cache_size)
    sampler = SharedBaseMCMC(args.d, kernel, weights)
    estimate = sampler.sample(tokens, chains=args.chains, burn=args.burn, draws=args.draws, thin=args.thin, seed=args.seed)
    diagnostic = estimate['diagnostics']
    rhat = diagnostic['rhat_probs'] + diagnostic['rhat_depth'] + diagnostic['rhat_alpha_and_loglik']
    threshold_checks = dict(
        split_rhat_below_1_05=all(v is not None and v < 1.05 for v in rhat),
        probability_ess_at_least_200=min(diagnostic['ess_probs']) >= 200,
        maximum_probability_mcse_at_most_0_005=max(estimate['mcse_probs']) <= .005)
    grid_checks = []
    if args.check_grid:
        refined_kernel = RankedLSARowEvidence(args.d, nodes, u_lo=min(args.u_lo, -36),
                                             u_hi=max(args.u_hi, 34), step=args.step/2)
        fine = PrefixPosterior(SharedBaseMCMC(args.d, refined_kernel, weights), tokens)
        original = PrefixPosterior(sampler, tokens)
        for values in diagnostic['final_states']:
            coarse_state = original.evaluate(values['alpha_index'], values['ranks'])
            fine_state = fine.evaluate(values['alpha_index'], values['ranks'])
            cp, _, ce = original.predict(coarse_state)
            fp, _, fe = fine.predict(fine_state)
            error_bits = abs(coarse_state.log_likelihood - fine_state.log_likelihood)/np.log(2)
            error_p = float(np.max(np.abs(cp-fp)))
            grid_checks.append(dict(log_likelihood_difference_bits=error_bits,
                                    max_prediction_difference=error_p,
                                    coarse_normalization_error=ce, fine_normalization_error=fe,
                                    passed=error_bits < 1e-5 and error_p < 1e-7))
    here = Path(__file__).resolve().parent
    code_names = ('ranked_lsa.py', 'weighted_lsa.py', 'phi_source.py', 'lsa_seq.py', 'shared_base_mcmc.py',
                  'refs_shared_base_mcmc.py', 'source.py', 'validate_shared_base_mcmc.py')
    store = Path(os.environ['LSA_STORE']).expanduser()
    backend = Path(os.environ['PMM_SRC']).expanduser()/'product_model_with_memory/universal_tables.py'
    report = dict(schema='shared_base_mcmc_prediction_v1', d=args.d, prefix_length=len(tokens),
                  tokens=tokens, generated_source=source_info, estimate=estimate,
                  settings=dict(alpha_prior='uniform on [0.5,1.5]', alpha_order=args.alpha_order,
                                alpha_nodes=nodes, alpha_weights=weights,
                                depth_prior='one shared uniform depth in 0,...,8',
                                rank_prior='uniform over all label permutations',
                                row_grid=dict(lower=args.u_lo, upper=args.u_hi, step=args.step),
                                term_cache_size=args.term_cache_size),
                  diagnostic_threshold_checks=threshold_checks,
                  diagnostic_thresholds_passed=all(threshold_checks.values()),
                  final_state_grid_checks=grid_checks, cache=kernel.cache_info(),
                  limitation='Selected-prefix approximation; diagnostic thresholds and selected-state quadrature checks do not certify convergence or arbitrary-prefix accuracy.',
                  runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
                  provenance=dict(code_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in code_names},
                                  store=str(store), source=str(backend.parent.parent),
                                  store_manifest_sha256=hashlib.sha256((store/'manifest.json').read_bytes()).hexdigest(),
                                  anchors_sha256=hashlib.sha256((store/'anchors.json').read_bytes()).hexdigest(),
                                  backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
                                  production_config_sha256=hashlib.sha256((backend.parent/'production_coding.py').read_bytes()).hexdigest()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name+'.tmp')
    temporary.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps(dict(out=str(args.out), d=args.d, n=len(tokens), seconds=estimate['elapsed_seconds'],
                          max_mcse=float(max(estimate['mcse_probs'])),
                          max_rhat=max(v for v in rhat if v is not None),
                          diagnostics_passed=report['diagnostic_thresholds_passed'],
                          grid_passed=all(c['passed'] for c in grid_checks))), flush=True)
    if grid_checks and not all(c['passed'] for c in grid_checks):
        raise SystemExit('row grid refinement failed')


if __name__ == '__main__':
    main()
