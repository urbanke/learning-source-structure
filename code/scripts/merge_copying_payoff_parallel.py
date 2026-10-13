"""Complete-cohort diagnostics for the bounded parallel copying waves, t6--8.

No new precision budget, scientific mean release or crossing decision. Shared
canonical references and common prior banks retain their numerical covariance.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from statistics import NormalDist
import numpy as np
import merge_copying_payoff_dense as old

dense = old.dense
ROOT, require, read, sha = old.ROOT, old.require, old.read, old.sha
TIMES = (6, 7, 8)
FAMILY_PAIRS = sum(dense.COUNTS[t] for t in TIMES)
FAMILY_COMPARISONS = 64*FAMILY_PAIRS
FAMILY_ALPHA = .05
Z_MULTIPLIER = NormalDist().inv_cdf(1-FAMILY_ALPHA/(2*FAMILY_COMPARISONS))


def paired_order_check(left, right):
    a, b = (np.asarray(m['mean'], dtype=float) for m in (left, right))
    sa, sb = (np.asarray(m['mcse_probs'], dtype=float) for m in (left, right))
    require(all(v.shape == (64,) and np.isfinite(v).all() for v in (a,b,sa,sb))
            and min(a.min(),b.min()) > 0 and min(sa.min(),sb.min()) >= 0
            and max(abs(a.sum()-1),abs(b.sum()-1)) <= 1e-7,
            'Invalid paired predictive moments')
    error = abs(a-b); se = np.hypot(sa,sb)
    limit = Z_MULTIPLIER*se+2*old.dense.accuracy.DETERMINISTIC_ALLOWANCE
    return dict(passed=bool(np.all(error <= limit)),
        failing_coordinates=np.flatnonzero(error > limit).tolist(),
        maximum_absolute_probability_error=float(error.max()),
        maximum_combined_mcse=float(se.max()),
        maximum_excess_over_allowance=float((error-limit).max()),
        z_multiplier=Z_MULTIPLIER, family_pairs=FAMILY_PAIRS,
        family_comparisons=FAMILY_COMPARISONS, family_alpha=FAMILY_ALPHA,
        deterministic_probability_allowance=2*old.dense.accuracy.DETERMINISTIC_ALLOWANCE)


def campaign_binding(campaign, folder):
    # The separate dispatcher validates allocation and immutable scientific
    # wave identities. It is intentionally not imported by the frozen sampler.
    import copying_payoff_parallel as parallel
    manifest = parallel.validate_wave(campaign, folder)
    paths = (Path(campaign)/'campaign.json',Path(parallel.__file__))
    return manifest, {str(p.resolve()):sha(p) for p in paths}


def load_references(folder, campaign):
    manifest, bindings = campaign_binding(campaign, folder)
    plan = dense.load_plan(folder); t = plan['observed']
    require(dense.pilot.runtime() == plan['runtime'], 'Frozen numerical runtime differs')
    require(t in TIMES and len(plan['tasks']) == 2*dense.COUNTS[t], 'Wrong full wave scope')
    require(len({task['seed'] for task in plan['tasks']}) == len(plan['tasks']),
            'Canonical runs must have distinct seeds')
    bindings[str(Path(folder)/'plan.json')] = sha(Path(folder)/'plan.json')
    missing = [task['id'] for task in plan['tasks'] if not dense.paths(folder,task)['report'].exists()]
    result = dict(complete=not missing, expected_tasks=len(plan['tasks']), original_histories=768,
        missing_task_ids=missing, individual_audits_passed=False, paired_orders_passed=False,
        diagnostic_arrays_validated=False, aggregate_precision_checked=False,
        aggregate_precision_passed=False, population_reference_approved=False,
        payoff_forecast_approved=False, timing_assessment_pending=True,
        finite_shuffle_exact=False, published_mean_gain_bits=None)
    if missing:
        return plan,result,{},bindings
    refs = {16:{},32:{}}; failed = []
    for task in plan['tasks']:
        paths = dense.paths(folder,task)
        report, audit, identity = (read(paths[k]) for k in ('report','audit','identity'))
        require(report['plan_id'] == identity['plan_id'] == plan['plan_id']
                and report['task'] == identity['task'] == task and report['complete'],
                'Reference identity changed')
        for key in ('audit','identity','checkpoint','moments'):
            digest = sha(paths[key])
            require(digest == report[key+'_sha256'], 'Raw reference binding changed')
            bindings[str(paths[key])] = digest
        bindings[str(paths['report'])] = sha(paths['report'])
        require(report['audit_passed'] == (audit['passed'] and all(
            g['passed'] for g in audit['final_state_grid_checks'])), 'Audit acceptance changed')
        if not report['audit_passed']:
            failed.append(task['id'])
        with np.load(paths['moments'],allow_pickle=False) as z:
            moments = {k:z[k].copy() for k in z}
        old.moments_checked(moments,report['estimate'],len(set(task['tokens'])))
        require(task['reference_key'] not in refs[task['order']], 'Duplicate canonical reference')
        refs[task['order']][task['reference_key']] = moments
    require(set(refs[16]) == set(refs[32]) and len(refs[16]) == dense.COUNTS[t],
            'Missing paired integration orders')
    pairs = [dict(reference_key=k,**paired_order_check(refs[16][k],refs[32][k]))
             for k in sorted(refs[16])]
    result.update(failed_audit_ids=failed,individual_audits_passed=not failed,
        paired_order_checks=pairs,paired_orders_passed=all(p['passed'] for p in pairs),
        paired_order_scope='Normal-MCSE Bonferroni diagnostic over all488 canonical pairs '
            'and64 coordinates across t6,t7,t8; estimated errors do not certify convergence.')
    return plan,result,refs,bindings


def map_history(tokens, transition, use, task, observed):
    require(observed in TIMES and len(tokens) >= observed, 'Invalid observed context')
    prefix = np.asarray(tokens[:observed],dtype='<i8')
    pattern,labels = old.canonicalize(prefix)
    require(old.reference_key(pattern) == use['reference_key'] and list(pattern) == task['tokens']
            and list(labels) == use['canonical_to_original_seen']
            and hashlib.sha256(prefix.tobytes()).hexdigest() == use['original_prefix_sha256']
            and int(prefix[-1]) == use['queried_original_label'],
            'Original ordered history or alphabet map changed')
    mapping = old.alphabet_map(labels); oracle = transition[int(prefix[-1]),mapping]
    require(oracle.shape == (64,) and np.isfinite(oracle).all() and oracle.min() >= 0
            and abs(oracle.sum()-1) < 1e-7, 'Invalid true query row')
    return mapping,oracle


def model_loss(name, seed, role, source, bindings, observed):
    folder = old.SCORES/'eval'/f'{role}_seed{seed}'; path = folder/(name+'.npz')
    protocol,status = read(folder/'protocol.json'),read(folder/'status.json')
    require(status['complete'] and status['scientific_results'] and protocol['role'] == role
        and protocol['seed'] == seed
        and protocol['evaluation_code_sha256'] == sha(ROOT/'code/transformer/shared_base_shuffle_eval.py')
        and protocol['model_code_sha256'] == sha(ROOT/'code/transformer/model.py'),
        'Neural evaluation identity changed')
    with np.load(path,allow_pickle=False) as z:
        require(str(z['protocol_sha256']) == sha(folder/'protocol.json')
            and str(z['source_tokens_sha256']) == source['tokens_sha256']
            and np.array_equal(z['times'],np.arange(1,1025))
            and z['expected_loss_bits'].shape == (256,1024), 'Neural cohort/time differs')
        value = z['expected_loss_bits'][:,observed-1].copy()
    require(np.isfinite(value).all(), 'Nonfinite neural losses')
    for p in (path,folder/'protocol.json',folder/'status.json'):
        bindings[str(p)] = sha(p)
    return value


def merge(folder, campaign):
    plan,result,refs,bindings = load_references(folder,campaign); t = plan['observed']
    result.update(schema='copying-payoff-parallel-diagnostics-v1',plan_id=plan['plan_id'],time=t,
        interpretation='Complete-cohort diagnostic only. Aggregate precision, timing definition '
            'and total uncertainty remain unapproved; no crossing inferred.',
        comparator='Source-induced stationary-frequency surrogate, not exact finite-shuffle Bayes.',
        informed_bayes='Original full shared-base prior; source parameters used only for scoring.',
        uncertainty='Shared-prefix MCMC covariance, local log-bias proxies and common-bank '
            'jackknife errors. Observed paired sequence SE is separate and contains numerical noise.')
    if not result['complete'] or not result['individual_audits_passed'] or not result['paired_orders_passed']:
        return result,{}
    extra,_ = old.frozen_inputs(plan); bindings.update(extra); arrays = {}; groups = []
    tasks = {task['reference_key']:task for task in plan['tasks']}
    require(set(plan['source_bindings']) == {'a1_L0','a1_L2','a1_L3'}, 'Incomplete source conditions')
    for name,source in plan['source_bindings'].items():
        require(sha(source['path']) == source['sha256'], 'Original source binding changed')
        bindings[source['path']] = source['sha256']
        with np.load(source['path'],allow_pickle=False) as z:
            tokens,transition = z['tokens'],z['transition']
        require(tokens.shape == (256,1025) and transition.shape == (256,64,64), 'Incomplete cohort')
        uses = sorted((u for u in plan['uses'] if u['condition'] == name),key=lambda u:u['index'])
        require([u['index'] for u in uses] == list(range(256)), 'Selected or duplicate histories')
        mapped = [map_history(tokens[i],transition[i],u,tasks[u['reference_key']],t)
                  for i,u in enumerate(uses)]
        oracles = np.stack([v[1] for v in mapped]); keys = [u['reference_key'] for u in uses]
        with np.load(old.FREQ/'refined'/(name+'.npz'),allow_pickle=False) as z:
            require(z['stationary_loss'].shape == (256,8)
                    and z['stationary_leave_one_bank_out_loss'].shape == (16,256,8),
                    'Stationary coverage changed')
            stationary = z['stationary_loss'][:,t-1]; leave = z['stationary_leave_one_bank_out_loss'][:,:,t-1]
        with np.load(old.FREQ/'stratified'/(name+'.npz'),allow_pickle=False) as z:
            require(z['stationary_loss'].shape == (256,8), 'Coarse stationary coverage changed')
            coarse = z['stationary_loss'][:,t-1]
        require(all(np.isfinite(v).all() for v in (stationary,leave,coarse)), 'Invalid stationary losses')
        bank = old.bank_error(stationary,leave)
        bank['coarse_refined_mean_shift_bits'] = float((stationary-coarse).mean())
        group = dict(condition=name,n_sequences=256,stationary_numerics=bank,rules={})
        arrays[name+'_stationary_loss'] = stationary
        for order in (16,32):
            prediction = np.stack([refs[order][k]['mean'] for k in keys])
            loss = -np.sum(oracles*np.log2(prediction),axis=1)
            error = old.shared_error(oracles,keys,refs[order]); gain = stationary-loss
            row = dict(bayes_loss=old.stats(loss),stationary_minus_bayes=old.stats(gain),**error)
            row['stationary_minus_bayes'].update(
                combined_numerical_mcse_bits=float(np.hypot(error['numerical_mcse_bits'],bank['prior_bank_mcse_bits'])),
                combined_log_bias_proxy_bits=error['log_bias_proxy_bits']+bank['prior_bank_jackknife_bias_proxy_bits'])
            group['rules'][str(order)] = row; arrays[name+f'_bayes_loss_a{order}'] = loss
            original = np.empty_like(prediction)
            for i,(mapping,_) in enumerate(mapped):
                original[i,mapping] = prediction[i]
            arrays[name+f'_original_label_probs_a{order}'] = original
        difference = arrays[name+'_bayes_loss_a16']-arrays[name+'_bayes_loss_a32']
        group['order_loss_difference'] = dict(**old.stats(difference),numerical_mcse_bits=float(
            np.hypot(*[group['rules'][str(o)]['numerical_mcse_bits'] for o in (16,32)])))
        group['neural'] = {}
        for seed in (0,1):
            ordered,shuffled = [model_loss(name,seed,role,source,bindings,t) for role in ('ordered','shuffled')]
            for role,value in (('ordered',ordered),('shuffled',shuffled)):
                arrays[name+f'_{role}_seed{seed}'] = value
            group['neural'][str(seed)] = dict(gain=old.stats(shuffled-ordered),
                ordered_minus_bayes=old.stats(ordered-arrays[name+'_bayes_loss_a32']),
                shuffled_minus_stationary=old.stats(shuffled-stationary))
        groups.append(group)
    for path in (Path(__file__),Path(old.__file__)):
        bindings[str(path.resolve())] = sha(path)
    result.update(diagnostic_arrays_validated=True,diagnostic_groups=groups,input_sha256=bindings,
        merger_sha256=sha(__file__),wave_plan_sha256=sha(Path(folder)/'plan.json'))
    return result,arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','campaign','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    require(not args.out.exists(), 'Use a new diagnostics folder; preserve previous results')
    result,arrays = merge(args.input,args.campaign); args.out.mkdir(parents=True)
    if arrays:
        np.savez_compressed(args.out/'arrays.npz',**arrays)
        result['arrays_sha256'] = sha(args.out/'arrays.npz')
    dense.write(args.out/'summary.json',result)
    print({k:result[k] for k in ('time','complete','individual_audits_passed','paired_orders_passed',
                                'diagnostic_arrays_validated','payoff_forecast_approved')})
    return old.exit_status(result)


if __name__ == '__main__':
    raise SystemExit(main())
