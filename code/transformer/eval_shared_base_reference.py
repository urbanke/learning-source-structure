"""Resumable full-context Bayes estimates on the frozen transformer cohort.

Only observed tokens enter inference. Stored transformer and oracle laws
are used afterwards to assess the precision of the comparison metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from eval_shared_base_models import (HERE, atomic_json, atomic_npz, fingerprint,
    sha_array, sha_file, source_identity, validate_source_record)
from full_context_statistics import (compare_prediction, diagnostic_gates,
                                     predictive_moments)
from ranked_lsa import RankedLSARowEvidence
from shared_base_chain_runner import ChainRun
from shared_base_mcmc import PrefixPosterior, SharedBaseMCMC
from source import seed_of
from validate_shared_base_mcmc import json_safe


def load_inputs(folder, index, prefix):
    folder = Path(folder)
    cohort = json.loads((folder/'cohort.json').read_text())
    if index not in range(cohort['n_sequences']) or prefix not in cohort['prefixes']:
        raise ValueError('requested source/prefix is outside declared cohort')
    path = folder/'data'/f'seq_{index:06d}.json'
    source = json.loads(path.read_text())
    identity, _ = source_identity(index, cohort['master_seed'])
    validate_source_record(source, identity)
    declared = cohort['records'][index]
    if declared['source_id'] != source['source_id'] or declared['file_sha256'] != sha_file(path):
        raise ValueError('source differs from frozen cohort')
    tokens = np.asarray(source['tokens'], dtype=int)[:prefix]
    prefix_hash = sha_array(tokens, '<i2')
    oracle = np.asarray(source['oracle']['transition_probs'])[tokens[-1]]
    models = {}
    for name in ('d64', 'd64_seed1'):
        model_path = folder/'models'/name/f'seq_{index:06d}.json'
        model = json.loads(model_path.read_text())
        if model['cache_id'] != fingerprint(model['identity']):
            raise ValueError('model identity checksum failed')
        if (model['source_id'] != source['source_id']
                or model['tokens_sha256'] != source['tokens_sha256']
                or model['prefix_tokens_sha256'][str(prefix)] != prefix_hash):
            raise ValueError('model and reference tokens differ')
        j = model['prefixes'].index(prefix)
        if (model['target_indices'][j] != prefix
                or model['targets'][j] != source['tokens'][prefix]):
            raise ValueError('model target alignment differs')
        array_path = model_path.with_name(model['arrays_file'])
        if sha_file(array_path) != model['arrays_sha256']:
            raise ValueError('model arrays checksum failed')
        with np.load(array_path, allow_pickle=False) as arrays:
            prediction = arrays['predictive_probs'][j]
            if (not np.array_equal(prediction, model['predictive_probs'][j])
                    or not np.array_equal(oracle, arrays['oracle_probs'][j])):
                raise ValueError('model probability/oracle readback failed')
        models[name] = dict(prediction=prediction, path=str(model_path.resolve()),
                            file_sha256=sha_file(model_path), cache_id=model['cache_id'])
    return source, tokens, oracle, models, dict(
        cohort_sha256=sha_file(folder/'cohort.json'), source_file_sha256=sha_file(path),
        source_id=source['source_id'], tokens_sha256=source['tokens_sha256'],
        prefix_tokens_sha256=prefix_hash, sequence_index=index, prefix_length=prefix,
        model_files={k:dict(path=v['path'], file_sha256=v['file_sha256'], cache_id=v['cache_id'])
                     for k,v in models.items()})


def selected_grid_checks(sampler, tokens, estimate):
    kernel = sampler.kernel
    fine_kernel = RankedLSARowEvidence(sampler.d, kernel.alpha_nodes,
        depths=sampler.depths, u_lo=min(-36, kernel.u_lo),
        u_hi=max(34, kernel.u_hi), step=kernel.step/2, term_cache_size=256)
    coarse = PrefixPosterior(sampler, tokens)
    fine = PrefixPosterior(SharedBaseMCMC(sampler.d, fine_kernel,
                          sampler.alpha_weights, depth_prior=sampler.depth_prior), tokens)
    checks = []
    for state in estimate['diagnostics']['final_states']:
        c = coarse.evaluate(state['alpha_index'], state['ranks'])
        f = fine.evaluate(state['alpha_index'], state['ranks'])
        cp, _, ce = coarse.predict(c)
        fp, _, fe = fine.predict(f)
        delta = abs(c.log_likelihood-f.log_likelihood)/np.log(2)
        error = float(np.max(np.abs(cp-fp)))
        checks.append(dict(log_likelihood_difference_bits=delta,
            max_prediction_difference=error, coarse_normalization_error=ce,
            fine_normalization_error=fe,
            passed=bool(delta < 1e-5 and error < 1e-7 and max(ce, fe) < 1e-7)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('output/shared_base_full_context'))
    parser.add_argument('--out', type=Path, default=Path('output/shared_base_full_context'))
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--prefix', type=int, default=1024)
    parser.add_argument('--alpha-order', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--burn', type=int, default=8192)
    parser.add_argument('--thin', type=int, default=4)
    parser.add_argument('--initial-draws', type=int, default=4096)
    parser.add_argument('--max-draws', type=int, default=65536)
    parser.add_argument('--block-draws', type=int, default=2048)
    parser.add_argument('--term-cache-size', type=int, default=8192)
    args = parser.parse_args()
    if args.alpha_order < 2 or args.initial_draws < 16 or args.max_draws < args.initial_draws:
        parser.error('invalid exponent order or draw budget')
    _, tokens, oracle, models, inputs = load_inputs(args.input, args.index, args.prefix)
    nodes, weights = leggauss(args.alpha_order)
    nodes, weights = 1+nodes/2, weights/2
    kernel = RankedLSARowEvidence(64, nodes, term_cache_size=args.term_cache_size)
    sampler = SharedBaseMCMC(64, kernel, weights)
    seed = seed_of('shared-base-full-context-reference-v1', args.seed, args.index,
                   args.prefix, args.alpha_order)
    stem = f'seq_{args.index:06d}'
    folder = args.out/'refs'/f'alpha{args.alpha_order}'
    report_path, moments_path = folder/(stem+'.json'), folder/(stem+'.npz')
    chain_path = args.out/'chains'/f'alpha{args.alpha_order}'/(stem+'.npz')
    run = ChainRun.load(sampler, tokens, chain_path) if chain_path.exists() else ChainRun(
        sampler, tokens, chains=args.chains, burn=args.burn, thin=args.thin, seed=seed)
    if (run.seed, run.chains, run.burn, run.thin) != (seed, args.chains, args.burn, args.thin):
        raise ValueError('saved chain configuration differs from request')
    code_hashes = {name:sha_file(HERE/name) for name in (
        'eval_shared_base_reference.py', 'full_context_statistics.py', 'eval_shared_base_models.py')}
    identity = dict(inputs=inputs, numerical_problem=run.fingerprint, code_sha256=code_hashes)
    old = json.loads(report_path.read_text()) if report_path.exists() else None
    if old is not None and old['identity'] != identity:
        raise ValueError('comparison inputs or code changed; use another output folder')
    history = [] if old is None else old['sampling_rounds']

    def progress(event):
        print(json.dumps(dict(sequence=args.index, alpha_order=args.alpha_order, **event)), flush=True)
        # Save the first completed burn and each completed chain in a round.
        # Raw samples are retained outside git; unequal counts can resume.
        if event['phase'] == 'burn' or event['draws'] == event['requested_draws']:
            run.save(chain_path)

    budget = max(args.initial_draws, int(run.draw_counts.max()))
    if budget > args.max_draws:
        raise ValueError('maximum budget cannot discard existing retained draws')
    while True:
        estimate = run.extend(budget, block_draws=args.block_draws, progress=progress)
        moments = predictive_moments(run.samples['predictive_probs'])
        if (not np.allclose(moments['mean'], estimate['predictive_probs'], rtol=0, atol=2e-14)
                or not np.allclose(moments['mcse_probs'], estimate['mcse_probs'], rtol=1e-11, atol=1e-15)):
            raise ValueError('independent moment and sampler summaries disagree')
        metrics = {name:compare_prediction(estimate['predictive_probs'], item['prediction'], oracle, moments)
                   for name,item in models.items()}
        gates = diagnostic_gates(estimate)
        passed = all(gates.values()) and all(m['metric_precision_passed'] for m in metrics.values())
        round_report = dict(draws_per_chain=budget, diagnostic_gates=gates,
            model_metrics=metrics, sampling_precision_passed=passed,
            max_rhat=max(v for k in ('rhat_probs','rhat_depth','rhat_alpha_and_loglik')
                         for v in estimate['diagnostics'][k] if v is not None),
            min_probability_ess=min(estimate['diagnostics']['ess_probs']),
            max_probability_mcse=float(np.max(estimate['mcse_probs'])),
            elapsed_seconds=run.elapsed_seconds)
        if not history or history[-1]['draws_per_chain'] != budget:
            history.append(round_report)
        else:
            history[-1] = round_report
        atomic_npz(moments_path, **moments)
        grid_checks = selected_grid_checks(sampler, tokens, estimate) if passed or budget >= args.max_draws else []
        report = dict(schema='shared-base-full-context-reference-v1', identity=identity,
            reference_id=fingerprint(identity), sequence_index=args.index, prefix_length=args.prefix,
            source_id=inputs['source_id'], tokens_sha256=inputs['tokens_sha256'],
            prefix_tokens_sha256=inputs['prefix_tokens_sha256'], estimate=estimate,
            settings=dict(alpha_order=args.alpha_order, alpha_nodes=nodes, alpha_weights=weights,
                depth_prior=sampler.depth_prior, depths=sampler.depths,
                row_grid=dict(lower=kernel.u_lo, upper=kernel.u_hi, step=kernel.step),
                term_cache_size=args.term_cache_size, sampling_master_seed=args.seed,
                seed=seed, chains=args.chains, burn=args.burn, thin=args.thin),
            model_metrics=metrics, diagnostic_gates=gates,
            sampling_precision_passed=passed, final_state_grid_checks=grid_checks,
            local_validation_passed=passed and bool(grid_checks) and all(c['passed'] for c in grid_checks),
            sampling_rounds=history, moments_file=moments_path.name,
            moments_sha256=sha_file(moments_path), chain_checkpoint=str(chain_path.resolve()),
            cache=kernel.cache_info(), limitation='Selected-prefix pilot. Exponent refinement is checked separately; chain diagnostics do not certify convergence.')
        atomic_json(report_path, json_safe(report))
        print(json.dumps(json_safe(dict(sequence=args.index, alpha_order=args.alpha_order,
                                       **round_report))), flush=True)
        if passed or budget >= args.max_draws:
            break
        budget = min(args.max_draws, 2*budget)
    if not report['local_validation_passed']:
        raise SystemExit('reference remains unresolved at current budget; retained for extension')


if __name__ == '__main__':
    main()
