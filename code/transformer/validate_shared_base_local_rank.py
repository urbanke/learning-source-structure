"""Validate the local-rank proposal against existing explicit long controls.

The exact permutation/rank sums were already independently evaluated and
refined. Verify their complete numerical provenance before reusing them;
only the new Markov-chain proposal is sampled here. Existing kernels and
stored tables remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from eval_shared_base_models import atomic_json, sha_file
from full_context_statistics import diagnostic_gates
from ranked_lsa import RankedLSARowEvidence
from shared_base_local_rank import SharedBaseLocalRankMCMC, PROPOSAL_SPEC
from validate_shared_base_long import fixed_state_checks
from validate_shared_base_mcmc import json_safe, table_provenance


HERE = Path(__file__).resolve().parent


def verified_controls(path):
    source = json.loads(Path(path).read_text())
    if not source['passed'] or not source['diagnostic_thresholds_passed']:
        raise ValueError('saved exact-control validation did not pass')
    if source['settings']['prefix_tokens'] != 1024 or source['settings']['alpha_orders'] != [16, 32]:
        raise ValueError('expected long controls with exponent refinement')
    provenance = source['provenance']
    for name, expected in provenance['code_sha256'].items():
        if sha_file(HERE/name) != expected:
            raise ValueError(f'exact-control numerical code changed: {name}')
    for name, expected in table_provenance().items():
        if provenance[name] != expected:
            raise ValueError(f'exact-control store/backend provenance changed: {name}')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path,
        default=Path('results/transformer/shared_base_full_context/long_validation.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='only the faster d3 long control')
    parser.add_argument('--burn', type=int, default=4096)
    parser.add_argument('--draws', type=int, default=8192)
    parser.add_argument('--thin', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260926)
    args = parser.parse_args()
    if args.burn < 0 or args.draws < 16 or args.thin < 1:
        parser.error('invalid sampling budget')
    controls = verified_controls(args.controls)
    records = []
    started = time.monotonic()
    for original in controls['cases']:
        d = original['d']
        if args.smoke and d != 3:
            continue
        exact = original['exact']
        nodes = np.asarray(exact['alpha_nodes'])
        weights = np.asarray(exact['alpha_weights'])
        kernel = RankedLSARowEvidence(d, nodes, term_cache_size=4096)
        sampler = SharedBaseLocalRankMCMC(d, kernel, weights)
        tokens = np.asarray(original['tokens'], dtype=int)
        print(json.dumps(dict(stage='sample', d=d, draws=args.draws)), flush=True)
        estimate = sampler.sample(tokens, chains=4, burn=args.burn, draws=args.draws,
                                  thin=args.thin, seed=args.seed+d)
        checks = {}
        for name, se_name in (('predictive_probs','mcse_probs'),
                              ('posterior_depth','mcse_depth'),
                              ('posterior_alpha','mcse_alpha')):
            actual, expected = np.asarray(estimate[name]), np.asarray(exact[name])
            se = np.asarray(estimate[se_name])
            error = np.abs(actual-expected)
            normalized = np.isfinite(actual).all() and np.all(actual >= 0) and abs(actual.sum()-1) < 1e-7
            checks[name] = dict(passed=bool(normalized and np.isfinite(se).all()
                and np.all(se >= 0) and np.all(error <= 5*se+.0002)
                and (name != 'predictive_probs' or error.max() <= .01)),
                max_absolute_error=float(error.max()), maximum_mcse=float(se.max()),
                absolute_error=error, mcse=se)
        grids = fixed_state_checks(tokens, sampler, estimate)
        gates = diagnostic_gates(estimate)
        passed = all(c['passed'] for c in checks.values()) and all(gates.values()) and all(g['passed'] for g in grids)
        record = dict(d=d, prefix_length=len(tokens), token_sha256=original['token_sha256'],
            estimate=estimate, exact_checks=checks, diagnostic_gates=gates,
            independent_kernel_grid_checks=grids, passed=passed)
        records.append(record)
        print(json.dumps(json_safe(dict(d=d, passed=passed, diagnostic_gates=gates,
              max_prediction_error=checks['predictive_probs']['max_absolute_error'],
              seconds=estimate['elapsed_seconds']))), flush=True)
    names = ('validate_shared_base_local_rank.py', 'shared_base_local_rank.py',
             'eval_shared_base_models.py', 'full_context_statistics.py')
    report = dict(schema='shared-base-local-rank-validation-v1',
        passed=bool(records) and all(r['passed'] for r in records), cases=records,
        proposal=PROPOSAL_SPEC, settings=dict(smoke=args.smoke, chains=4,burn=args.burn,
            draws=args.draws,thin=args.thin,seed=args.seed,seed_rule='seed+d'),
        elapsed_seconds=time.monotonic()-started,
        provenance=dict(controls_file=str(args.controls),controls_sha256=sha_file(args.controls),
            code_sha256={n:sha_file(HERE/n) for n in names},
            exact_control_provenance=controls['provenance']),
        limitation='Numerical controls for the new proposal; no general convergence guarantee.')
    atomic_json(args.out, json_safe(report))
    if not report['passed']:
        raise SystemExit('local-rank validation failed')


if __name__ == '__main__':
    main()
