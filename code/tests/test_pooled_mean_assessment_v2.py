"""Independent bias identities, paired differences, and unchanged safety gates."""
import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from test_pooled_mean_validation import component, histories, records

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

v = load('assessment_v2', ROOT/'code/scripts/pooled_mean_assessment_v2.py')
v1 = load('assessment_v1', ROOT/'code/scripts/pooled_mean_validation.py')


def analyze(offset=0, hs=None, recs=None):
    hs = histories() if hs is None else hs
    keys = sorted({h['reference_key'] for h in hs})
    refs = {k: records(offset+i*1000) for i, k in enumerate(keys)} if recs is None else recs
    return v.analyze_arm(refs, hs)


def test_common_evidence_scale_has_zero_actual_prediction_error():
    hs = histories()
    for h in hs:
        h['fixed_components'] = [component(-3, [.7, .3]), component(-4, [.7, .3])]
    keys = sorted({h['reference_key'] for h in hs})
    refs = {k: [dict(component(-3+(.4 if j % 2 else -.4), [.7, .3]), seed=i*1000+j)
                for j in range(32)] for i, k in enumerate(keys)}
    a = v.analyze_arm(refs, hs)
    old = v1.analyze_arm(refs, hs)
    assert a['summary']['bayes'] == old['summary']['bayes']
    assert a['summary']['numerical_mcse_bits'] < 1e-15
    assert a['summary']['bias_allowance_bits'] < 1e-13
    assert old['summary']['bias_allowance_bits'] > 1e-5
    assert a['summary']['basic_passed'] == old['summary']['basic_passed']


def test_signed_delta_bias_matches_independent_directional_hessian():
    hs = histories()
    recs = records(scale=.08)
    ref = v._reference(recs)
    result = v._history(ref, hs[0])
    # Independent calculation in unscaled probability units and natural coordinates.
    samples = np.array([[math.exp(r['log_evidence'])*p/3 for p in r['predictive_probs']]
                        + [math.exp(r['log_evidence'])/3] for r in recs])
    fixed = hs[0]['fixed_components']
    const_a = sum(math.exp(r['log_evidence'])*np.array(r['predictive_probs'])/3 for r in fixed)
    const_d = sum(math.exp(r['log_evidence'])/3 for r in fixed)
    q = np.array(hs[0]['oracle_probs'])
    def f(x):
        return -q@np.log2((const_a+x[:-1])/(const_d+x[-1]))
    mean = samples.mean(axis=0)
    eps = .01
    second = sum((f(mean+eps*d)+f(mean-eps*d)-2*f(mean))/eps**2
                 for d in samples-mean)/(2*32*31)
    assert result['curvature'] == pytest.approx(second, rel=2e-4, abs=2e-9)


def test_identical_paired_rules_have_zero_uncertainty_and_bias():
    a = analyze()
    assert a['summary']['bias_allowance_bits'] > 0
    comparison = v.compare_arms(a, copy.deepcopy(a), shared_records=True)
    assert comparison['allowance_bits'] == 0
    assert comparison['bias_allowance_bits'] == 0
    assert comparison['difference_mcse_bits'] == 0


def test_changed_paired_rules_use_delete_one_difference_bias():
    hs = histories()
    a = analyze(hs=hs)
    for h in hs:
        h['fixed_components'][0] = component(-2.7, [.61, .39])
    b = analyze(hs=hs)
    c = v.compare_arms(a, b, shared_records=True)
    independent_bias = 0
    for key, ar in a['references'].items():
        br = b['references'][key]
        leave = np.array(ar['leave_one_regret_contributions_bits'])-np.array(br['leave_one_regret_contributions_bits'])
        full = ar['mean_regret_contribution_bits']-br['mean_regret_contribution_bits']
        jack = 31*(leave.mean()-full)
        delta2 = ar['signed_second_order_bias_bits']-br['signed_second_order_bias_bits']
        independent_bias += max(abs(jack), abs(delta2))
    assert c['bias_allowance_bits'] == pytest.approx(independent_bias, abs=1e-13)
    assert c['difference_mcse_bits'] > 0


def test_all_numerical_thresholds_and_tail_checks_remain_in_force():
    assert v.CRITERIA == v1.CRITERIA
    hs = histories(); keys = sorted({h['reference_key'] for h in hs})
    refs = {k: records(i*1000) for i, k in enumerate(keys)}
    bad = refs[keys[0]][0]
    bad.update(component(-.001, bad['predictive_probs']))
    a = v.analyze_arm(refs, hs)
    old = v1.analyze_arm(refs, hs)
    assert not a['summary']['basic_passed']
    assert [r['checks'] for r in a['rows']] == [r['checks'] for r in old['rows']]
    assert {k:r['checks'] for k,r in a['references'].items()} == {k:r['checks'] for k,r in old['references'].items()}


def test_seed_and_complete_cohort_guards_cannot_be_bypassed():
    a = analyze()
    with pytest.raises(ValueError, match='independent seeds'):
        v.compare_arms(a, a)
    bad = copy.deepcopy(a)
    next(iter(bad['references'].values()))['records_sha256'] = 'changed'
    with pytest.raises(ValueError, match='identical shared records'):
        v.compare_arms(a, bad, shared_records=True)
    with pytest.raises(ValueError, match='192'):
        analyze(hs=histories()[:-1])


def test_biased_refinement_still_fails_fixed_total_budget():
    arms = {name: analyze(j*10000) for j, name in enumerate('ABCD')}
    assert v.validate_campaign(arms, arms['D'])['passed']
    altered = copy.deepcopy(arms)
    altered['A']['summary']['bayes']['mean'] += .02
    result = v.validate_campaign(altered, altered['D'])
    assert not result['passed']
    assert result['numerical_budget_bits'] <= .01/3
