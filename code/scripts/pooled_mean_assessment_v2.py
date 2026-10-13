"""Revision 2: cancellation-aware full-cohort pooled-regret diagnostics.

The random unit is a complete SMC repetition for a canonical reference.
Histories that reuse that reference share its numerical error. All input
vectors use the same full canonical alphabet; label mapping and source/code
provenance are the caller's responsibility. No sampler, file I/O or plotting.

The frozen v1 module is retained unchanged. Revision 2 changes bias proxies
for log ratios and paired differences, not samples, estimates or thresholds.
All component ESS/share and relative-MCSE guards remain mandatory.

Uncertainty and curvature are estimated numerical diagnostics, not rigorous
confidence bounds or certificates against missed posterior modes.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

FAMILIES = ('product', 'shared_base', 'uniform_base')
INDICES = tuple(range(5, 69))
REPETITIONS = 32
LN2 = math.log(2.)
CRITERIA = dict(minimum_replicates=32, minimum_effective_replica_count=8.,
                maximum_likelihood_share=.2, normalization_tolerance=1e-7,
                maximum_relative_mcse=.1, cohort_mcse_multiplier=2.,
                cohort_bias_multiplier=2., refinement_mcse_multiplier=3.,
                cohort_absolute_budget_bits=.01/3, cohort_se_fraction=.25)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _probs(value, *, oracle=False):
    p = np.asarray(value, dtype=float)
    require(p.ndim == 1 and len(p) > 1 and np.isfinite(p).all()
            and (np.min(p) >= 0 if oracle else np.min(p) > 0)
            and abs(float(p.sum())-1) <= CRITERIA['normalization_tolerance'],
            'Invalid probabilities; no clipping or renormalization is permitted')
    return p


def _component(value):
    p = _probs(value['predictive_probs'])
    z = float(value['log_evidence'])
    require(math.isfinite(z) and z <= 1e-12, 'Invalid absolute evidence')
    if 'log_joint_next' in value:
        joint = np.asarray(value['log_joint_next'], dtype=float)
        require(joint.shape == p.shape and np.isfinite(joint).all()
                and np.max(np.abs(joint-z-np.log(p))) <= 1e-9,
                'Joint evidence and prediction disagree')
    if 'maximum_conditional_normalization_error' in value:
        error = value['maximum_conditional_normalization_error']
        require(math.isfinite(error) and 0 <= error <= CRITERIA['normalization_tolerance'],
                'Conditional normalization failed')
    return z, p


def _ordered_histories(histories):
    require(len(histories) == 192
            and {(r['family'], r['index']) for r in histories}
            == {(f, i) for f in FAMILIES for i in INDICES},
            'All 192 distinct fixed histories are required')
    return sorted(histories, key=lambda r: (FAMILIES.index(r['family']), r['index']))


def _stats(values):
    x = np.asarray(values, dtype=float).reshape(3, 64)
    require(np.isfinite(x).all(), 'Nonfinite cohort score')
    return dict(mean=float(x.mean()),
                se=float(np.sqrt(np.sum(x.var(axis=1, ddof=1)/64))/3),
                family_means=x.mean(axis=1).tolist(),
                family_se=(x.std(axis=1, ddof=1)/8).tolist())


def _reference(records):
    require(len(records) == REPETITIONS, 'Exactly 32 independent repetitions are required')
    require(all(isinstance(r.get('seed'), int) and not isinstance(r['seed'], bool)
                and r['seed'] >= 0 for r in records), 'Every repetition needs its recorded seed')
    records = sorted(records, key=lambda r: r['seed'])
    seeds = [r['seed'] for r in records]
    require(len(set(seeds)) == len(seeds), 'Repeated repetition seed')
    parts = [_component(r) for r in records]
    logs = np.array([v[0] for v in parts])
    predictions = np.stack([v[1] for v in parts])
    z = np.exp(logs-logs.max())
    ess = float(z.sum()**2/(z@z))
    share = float(z.max()/z.sum())
    pooled_p = _probs((z[:, None]*predictions).mean(axis=0)/z.mean())
    pooled_log_z = float(logs.max()+np.log(z.mean()))
    checks = dict(effective_replicas=ess >= 8., maximum_likelihood_share=share <= .2)
    return dict(logs=logs, predictions=predictions, seeds=seeds,
                records_sha256=_digest(records), effective_replica_count=ess,
                maximum_single_replica_likelihood_share=share, checks=checks,
                pooled=dict(log_evidence=pooled_log_z, predictive_probs=pooled_p.tolist(),
                            log_joint_next=(pooled_log_z+np.log(pooled_p)).tolist()))


def _history(reference, history):
    q = _probs(history['oracle_probs'], oracle=True)
    require(len(history['fixed_components']) == 2, 'Product and uniform components required')
    fixed = [_component(v) for v in history['fixed_components']]
    fixed_logs = np.array([v[0] for v in fixed])
    fixed_p = np.stack([v[1] for v in fixed])
    p = reference['predictions']
    require(p.shape[1] == len(q) and fixed_p.shape == (2, len(q)),
            'Canonical alphabet shapes differ')
    shift = max(float(reference['logs'].max()), float(fixed_logs.max()))
    z = np.exp(reference['logs']-shift)/3
    joint = z[:, None]*p
    fixed_z = np.exp(fixed_logs-shift)/3
    fixed_joint = fixed_z@fixed_p
    a = fixed_joint+joint.mean(axis=0)
    d = float(fixed_z.sum()+z.mean())
    mixture = _probs(a/d)
    positive = q > 0
    entropy_term = float(q[positive]@np.log2(q[positive]))
    regret = entropy_term-float(q@np.log2(mixture))
    influence = -((joint-joint.mean(axis=0))/a
                  -(z-z.mean())[:, None]/d)@q/LN2
    n = len(z)
    joint_var = joint.var(axis=0, ddof=1)/n
    z_var = float(z.var(ddof=1)/n)
    # Bias of -sum(q*log A) + log D: common scale fluctuations cancel.
    curvature = float((q@(joint_var/a**2)-z_var/d**2)/(2*LN2))
    absolute_curvature = float((q@(joint_var/a**2)+z_var/d**2)/(2*LN2))
    leave_a = fixed_joint+(joint.sum(axis=0)-joint)/(n-1)
    leave_d = fixed_z.sum()+(z.sum()-z)/(n-1)
    leave_p = leave_a/leave_d[:, None]
    require(np.isfinite(leave_p).all() and np.min(leave_p) > 0
            and np.max(np.abs(leave_p.sum(axis=1)-1)) <= 1e-7,
            'Invalid leave-one-repetition probabilities')
    leave_regret = entropy_term-np.log2(leave_p)@q
    jackknife = float((n-1)*(leave_regret.mean()-regret))
    relative = float(max(np.max(np.sqrt(joint_var)/a), math.sqrt(z_var)/d))
    variance = float(influence.var(ddof=1)/n)
    return dict(regret=regret, mixture=mixture, influence=influence,
                variance=variance, leave_regret=leave_regret,
                curvature=curvature, absolute_curvature=absolute_curvature,
                jackknife=jackknife, relative=relative,
                checks=dict(relative_mcse=relative <= .1,
                            jackknife_curvature=abs(jackknife) <= 2*absolute_curvature+1e-12))


def analyze_arm(references, histories):
    """Analyze 32 repetitions/reference and all 192 paired canonical histories.

    ``references`` maps a canonical key to raw SMC estimate dictionaries.
    Each history has family, index, reference_key, oracle_probs and two
    fixed_components in product/uniform order. Optional transformer_regret_bits
    is a model-name/value dictionary and must appear consistently throughout.
    """
    histories = _ordered_histories(histories)
    require(set(references) == {h['reference_key'] for h in histories},
            'Missing or extraneous canonical reference')
    refs = {key: _reference(records) for key, records in references.items()}
    seeds = [s for r in refs.values() for s in r['seeds']]
    require(len(set(seeds)) == len(seeds), 'Independent references reuse random seeds')
    calculations = [_history(refs[h['reference_key']], h) for h in histories]
    rows = []
    for h, c in zip(histories, calculations):
        row = dict(family=h['family'], index=h['index'], reference_key=h['reference_key'],
                   bayes_regret_bits=c['regret'], mixture_probs=c['mixture'].tolist(),
                   mcse_bits=math.sqrt(c['variance']), relative_mcse_max=c['relative'],
                   signed_second_order_bias_bits=c['curvature'],
                   legacy_absolute_curvature_bits=c['absolute_curvature'],
                   signed_jackknife_bias_bits=c['jackknife'],
                   checks=c['checks'])
        if 'transformer_regret_bits' in h:
            row['transformer_regret_bits'] = h['transformer_regret_bits']
        rows.append(row)
    reference_results = {}
    numerical_variance = bias = curvature = jackknife = 0.
    for key, ref in refs.items():
        group = [c for h, c in zip(histories, calculations) if h['reference_key'] == key]
        influence = np.sum([c['influence'] for c in group], axis=0)/192
        curve = float(sum(c['curvature'] for c in group)/192)
        full = float(sum(c['regret'] for c in group)/192)
        leave = np.sum([c['leave_regret'] for c in group], axis=0)/192
        jack = float((REPETITIONS-1)*(leave.mean()-full))
        variance = float(influence.var(ddof=1)/REPETITIONS)
        numerical_variance += variance
        curvature += abs(curve)
        jackknife += abs(jack)
        bias += max(abs(curve), abs(jack))
        reference_results[key] = dict(seeds=ref['seeds'], records_sha256=ref['records_sha256'],
            uses=len(group), pooled=ref['pooled'], effective_replica_count=ref['effective_replica_count'],
            maximum_single_replica_likelihood_share=ref['maximum_single_replica_likelihood_share'],
            checks=ref['checks'], mean_influence_bits=influence.tolist(),
            numerical_variance_bits2=variance, signed_second_order_bias_bits=curve,
            signed_jackknife_bias_bits=jack,
            bias_allowance_bits=max(abs(curve), abs(jack)),
            mean_regret_contribution_bits=full,
            leave_one_regret_contributions_bits=leave.tolist())
    bayes = _stats([c['regret'] for c in calculations])
    denoised_variances = []
    numerical_sample_variances = []
    for family in FAMILIES:
        positions = [i for i, h in enumerate(histories) if h['family'] == family]
        diagonal = sum(calculations[i]['variance'] for i in positions)
        variance_of_sum = 0.
        for key in refs:
            selected = [calculations[i]['influence'] for i in positions
                        if histories[i]['reference_key'] == key]
            if selected:
                variance_of_sum += float(np.sum(selected, axis=0).var(ddof=1)/REPETITIONS)
        # E[sample variance of numerical noise] includes shared-reference covariance.
        noise = max(0., (diagonal-variance_of_sum/64)/63)
        observed = float(np.var([calculations[i]['regret'] for i in positions], ddof=1))
        numerical_sample_variances.append(noise)
        denoised_variances.append(max(0., observed-noise))
    source_se = math.sqrt(sum(denoised_variances)/64)/3
    budget = min(.01/3, .25*source_se)
    mcse = math.sqrt(max(0., numerical_variance))
    allowance = 2*mcse+2*bias
    basic = all(all(r['checks'].values()) for r in reference_results.values())
    basic = basic and all(all(r['checks'].values()) for r in rows)
    models = set(histories[0].get('transformer_regret_bits', {}))
    require(all(set(h.get('transformer_regret_bits', {})) == models for h in histories),
            'Incomplete transformer pairing')
    summary = dict(n_total=192, n_per_family=64, n_references=len(refs), repetitions=32,
                   bayes=bayes, numerical_mcse_bits=mcse, bias_allowance_bits=bias,
                   absolute_second_order_bias_sum_bits=curvature, absolute_jackknife_bias_sum_bits=jackknife,
                   numerical_allowance_bits=allowance, numerical_budget_bits=budget,
                   denoised_source_se_bits=source_se,
                   within_family_numerical_sample_variance_bits2=numerical_sample_variances,
                   basic_passed=basic, precision_passed=basic and allowance <= budget,
                   transformers={name: _stats([h['transformer_regret_bits'][name] for h in histories])
                                 for name in sorted(models)},
                   paired={name: _stats([h['transformer_regret_bits'][name]-c['regret']
                                        for h, c in zip(histories, calculations)])
                           for name in sorted(models)})
    cohort_signature = _digest([dict(family=h['family'], index=h['index'],
                                    reference_key=h['reference_key'], oracle_probs=h['oracle_probs'],
                                    transformer_regret_bits=h.get('transformer_regret_bits', {}))
                               for h in histories])
    fixed_signature = _digest([h['fixed_components'] for h in histories])
    return dict(schema='pooled-mean-arm-analysis-v2', cohort_sha256=cohort_signature,
                fixed_components_sha256=fixed_signature, rows=rows, summary=summary,
                references=reference_results,
                scope='Independent SMC repetitions; canonical reuse and joint Z,Zp covariance retained. '
                      'Numerical diagnostics are not confidence guarantees or missed-mode bounds.')


def _seeds(analysis):
    return {s for r in analysis['references'].values() for s in r['seeds']}


def compare_arms(left, right, *, shared_records=False):
    """Uncertainty of a full-mean difference, with explicit seed relationship."""
    require(left['cohort_sha256'] == right['cohort_sha256'], 'Compared cohorts differ')
    if shared_records:
        require(set(left['references']) == set(right['references']), 'Shared references differ')
        variance = 0.
        bias = 0.
        for key, a in left['references'].items():
            b = right['references'][key]
            require(a['records_sha256'] == b['records_sha256'] and a['seeds'] == b['seeds'],
                    'Deterministic refinement must use identical shared records')
            diff = np.asarray(a['mean_influence_bits'])-np.asarray(b['mean_influence_bits'])
            variance += float(diff.var(ddof=1)/REPETITIONS)
            # Difference-specific bias; identical stochastic inputs must cancel.
            second_order = (a['signed_second_order_bias_bits']
                            - b['signed_second_order_bias_bits'])
            jackknife = a['signed_jackknife_bias_bits'] - b['signed_jackknife_bias_bits']
            bias += max(abs(second_order), abs(jackknife))
        se = math.sqrt(max(0., variance))
    else:
        require(not (_seeds(left) & _seeds(right)), 'Refinement arms must have independent seeds')
        se = math.hypot(left['summary']['numerical_mcse_bits'], right['summary']['numerical_mcse_bits'])
        bias = left['summary']['bias_allowance_bits']+right['summary']['bias_allowance_bits']
    difference = abs(left['summary']['bayes']['mean']-right['summary']['bayes']['mean'])
    return dict(absolute_difference_bits=difference, difference_mcse_bits=se,
                bias_allowance_bits=bias, allowance_bits=difference+3*se+bias,
                shared_records=shared_records)


def validate_campaign(arms, cheap_refined):
    """One total budget for final-D error and A/B/C/D/inexpensive refinements.

    A/B compare particle counts, B/C alpha integration, C/D row integration.
    The cheap-family comparison reuses D's exact same stochastic records.
    Both the default and refined cheap rules are required. The publication
    estimate is D with the refined cheap rules.
    """
    require(set(arms) == set('ABCD'), 'All four complete refinement arms are required')
    seen = set()
    for name in 'ABCD':
        seeds = _seeds(arms[name])
        require(not (seeds & seen), 'All refinement arms must have independent seeds')
        seen |= seeds
    require(len({a['fixed_components_sha256'] for a in arms.values()}) == 1,
            'Stochastic refinement arms must use the same fixed components')
    comparisons = {a+'_'+b: compare_arms(arms[a], arms[b])
                   for a, b in (('A', 'B'), ('B', 'C'), ('C', 'D'))}
    comparisons['D_cheap_refined'] = compare_arms(arms['D'], cheap_refined, shared_records=True)
    summaries = [a['summary'] for a in arms.values()]+[cheap_refined['summary']]
    # Use the smallest denoised source budget so the choice of rule cannot loosen it.
    budget = min(s['numerical_budget_bits'] for s in summaries)
    final = cheap_refined['summary']
    refinement = sum(c['allowance_bits'] for c in comparisons.values())
    total = final['numerical_allowance_bits']+refinement
    basic = all(s['basic_passed'] for s in summaries)
    return dict(schema='pooled-mean-campaign-decision-v2', comparisons=comparisons,
                publication_arm='D_with_refined_cheap_components', final_summary=final,
                stochastic_allowance_bits=final['numerical_allowance_bits'],
                refinement_allowance_bits=refinement, total_numerical_allowance_bits=total,
                numerical_budget_bits=budget, basic_passed=basic,
                passed=basic and total <= budget,
                exact_short_validation_required=True,
                scope='Cancellation-aware mean diagnostics; all v1 stability guards retained. '
                      'Not a posterior-convergence certificate.')


def exact_short_validation(analyses, exact_rows, *, cheap_refined=None):
    """Test each new arm's complete mapped mean against exact short evidence.

    exact_rows contain family, index and bayes_regret_bits. This is separate
    from the preserved older componentwise calibration and changes no flag.
    """
    require(set(analyses) == set('ABCD'), 'Exact-short validation requires all four arms')
    exact_rows = _ordered_histories(exact_rows)
    exact = _stats([r['bayes_regret_bits'] for r in exact_rows])
    budget = min(.01/3, .25*exact['se'])
    checks = {}
    seen = set()
    signatures = {a['cohort_sha256'] for a in analyses.values()}
    require(len(signatures) == 1, 'Exact-short arm cohorts differ')
    for name, analysis in analyses.items():
        seeds = _seeds(analysis)
        require(not (seen & seeds), 'Exact-short arms must have independent seeds')
        seen |= seeds
        rows = _ordered_histories(analysis['rows'])
        require([(r['family'], r['index']) for r in rows]
                == [(r['family'], r['index']) for r in exact_rows], 'Exact-short pairing differs')
        s = analysis['summary']
        error = s['bayes']['mean']-exact['mean']
        allowance = abs(error)+3*s['numerical_mcse_bits']+2*s['bias_allowance_bits']
        checks[name] = dict(mean_error_bits=error, allowance_bits=allowance,
                            passed=s['basic_passed'] and allowance <= budget)
    if cheap_refined is not None:
        # The deterministic comparison must share the complete D record set.
        compare_arms(analyses['D'], cheap_refined, shared_records=True)
        s = cheap_refined['summary']
        error = s['bayes']['mean']-exact['mean']
        allowance = abs(error)+3*s['numerical_mcse_bits']+2*s['bias_allowance_bits']
        checks['D_cheap_refined'] = dict(mean_error_bits=error, allowance_bits=allowance,
            passed=s['basic_passed'] and allowance <= budget)
    return dict(schema='pooled-mean-exact-short-decision-v2', exact=exact,
                numerical_budget_bits=budget, arms=checks,
                passed=all(c['passed'] for c in checks.values()),
                old_component_calibration_unchanged=True)
