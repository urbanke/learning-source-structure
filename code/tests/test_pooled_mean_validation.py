"""Independent algebra, reference reuse, complete coverage and decision checks."""
import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/pooled_mean_validation.py'
SPEC = importlib.util.spec_from_file_location('mean_validation', PATH)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def component(logz, p):
    p = np.asarray(p)
    return dict(log_evidence=float(logz), predictive_probs=p.tolist(),
                log_joint_next=(logz+np.log(p)).tolist())


def records(offset=0, scale=.003):
    result = []
    for i in range(32):
        theta = 2*np.pi*i/32
        p = .42+scale*np.cos(theta)
        r = component(-3+scale*np.sin(theta), [p, 1-p])
        r['seed'] = offset+i
        result.append(r)
    return result


def histories(*, distinct=False, constant_oracle=False):
    result = []
    for f in v.FAMILIES:
        for i in v.INDICES:
            q = .8 if constant_oracle else .1+.8*(i-5)/63
            result.append(dict(family=f, index=i,
                reference_key=f'{f}:{i}' if distinct else 'shared',
                oracle_probs=[q, 1-q],
                fixed_components=[component(-3.2, [.7, .3]), component(-3.1, [.5, .5])],
                transformer_regret_bits={'seed0': .02+i/100, 'seed1': .025+i/100}))
    return result


def analyze(offset=0, scale=.003, hs=None):
    hs = histories() if hs is None else hs
    refs = {key: records(offset+100*j, scale)
            for j, key in enumerate(dict.fromkeys(h['reference_key'] for h in hs))}
    return v.analyze_arm(refs, hs)


def direct_loss(raw, h, weights=None):
    """Direct unscaled likelihood algebra, separate from implementation."""
    w = np.ones(len(raw)) if weights is None else np.asarray(weights)
    z = np.array([math.exp(r['log_evidence']) for r in raw])
    p = np.array([r['predictive_probs'] for r in raw])
    numerator = np.sum(w[:, None]*z[:, None]*p, axis=0)/w.sum()
    denominator = float(w@z/w.sum())
    for c in h['fixed_components']:
        fixed = math.exp(c['log_evidence'])
        numerator += fixed*np.array(c['predictive_probs'])
        denominator += fixed
    mixture = numerator/denominator
    q = np.array(h['oracle_probs'])
    positive = q > 0
    return float(q[positive]@np.log2(q[positive]/mixture[positive])), mixture


def test_weighted_evidence_and_joint_ratios_match_direct_algebra():
    hs = histories()
    raw = records(scale=.03)
    a = v.analyze_arm({'shared': raw}, hs)
    for row, h in zip(a['rows'], hs):
        expected, p = direct_loss(raw, h)
        assert row['bayes_regret_bits'] == pytest.approx(expected, abs=1e-14)
        np.testing.assert_allclose(row['mixture_probs'], p, rtol=0, atol=1e-14)
    z = np.exp([r['log_evidence'] for r in raw])
    p = np.array([r['predictive_probs'] for r in raw])
    np.testing.assert_allclose(a['references']['shared']['pooled']['predictive_probs'], z@p/z.sum())
    assert a['references']['shared']['pooled']['log_evidence'] == pytest.approx(np.log(z.mean()))


def test_covariance_influence_matches_independent_finite_difference():
    hs = histories()
    raw = records(scale=.03)
    a = v.analyze_arm({'shared': raw}, hs)
    expected = []
    eps = 1e-5
    for j in range(32):
        lo, hi = np.ones(32), np.ones(32)
        lo[j] -= eps
        hi[j] += eps
        lower = np.mean([direct_loss(raw, h, lo)[0] for h in hs])
        upper = np.mean([direct_loss(raw, h, hi)[0] for h in hs])
        expected.append(32*(upper-lower)/(2*eps))
    np.testing.assert_allclose(a['references']['shared']['mean_influence_bits'], expected,
                               rtol=2e-6, atol=2e-9)
    assert a['summary']['numerical_mcse_bits'] == pytest.approx(np.std(expected, ddof=1)/np.sqrt(32), rel=2e-6)


def test_reusing_one_reference_does_not_manufacture_192_independent_draws():
    shared = analyze(hs=histories(constant_oracle=True))
    independent = analyze(hs=histories(distinct=True, constant_oracle=True))
    assert shared['summary']['numerical_mcse_bits'] == pytest.approx(
        independent['summary']['numerical_mcse_bits']*math.sqrt(192), rel=1e-11)
    # A common error adds no within-family sample variation.
    assert max(shared['summary']['within_family_numerical_sample_variance_bits2']) < 1e-22
    assert min(independent['summary']['within_family_numerical_sample_variance_bits2']) > 0


def test_aggregate_jackknife_deletes_shared_repetition_for_every_use():
    hs = histories()
    raw = records(scale=.03)
    a = v.analyze_arm({'shared': raw}, hs)
    full = np.mean([direct_loss(raw, h)[0] for h in hs])
    leaves = [np.mean([direct_loss(raw[:j]+raw[j+1:], h)[0] for h in hs]) for j in range(32)]
    expected = 31*abs(np.mean(leaves)-full)
    assert a['summary']['jackknife_bias_bits'] == pytest.approx(expected, abs=3e-14)
    assert a['summary']['bias_allowance_bits'] >= expected-3e-14


def test_covariance_denoising_matches_explicit_covariance_matrix():
    hs = histories()
    raw = records(scale=.03)
    a = v.analyze_arm({'shared': raw}, hs)
    ref = v._reference(raw)
    influences = np.stack([v._history(ref, h)['influence'] for h in hs[:64]])
    covariance = np.cov(influences, ddof=1)/32
    noise = (np.trace(covariance)-covariance.sum()/64)/63
    assert a['summary']['within_family_numerical_sample_variance_bits2'][0] == pytest.approx(noise)


@pytest.mark.parametrize('count', [0, 16, 31, 33])
def test_requires_exactly_32_repetitions(count):
    raw = records()
    if count == 33:
        raw = raw+[dict(raw[0], seed=999)]
    else:
        raw = raw[:count]
    with pytest.raises(ValueError, match='32'):
        v.analyze_arm({'shared': raw}, histories())


def test_rejects_missing_duplicated_or_extra_histories():
    for hs in (histories()[:-1], histories()[:-1]+histories()[:1], histories()+histories()[:1]):
        with pytest.raises(ValueError, match='192'):
            v.analyze_arm({'shared': records()}, hs)


def test_rejects_seed_reuse_within_or_across_references():
    raw = records()
    raw[1]['seed'] = raw[0]['seed']
    with pytest.raises(ValueError, match='seed'):
        v.analyze_arm({'shared': raw}, histories())
    hs = histories()
    hs[0]['reference_key'] = 'other'
    with pytest.raises(ValueError, match='seeds'):
        v.analyze_arm({'shared': records(), 'other': records()}, hs)


@pytest.mark.parametrize('kind', ['probability', 'joint', 'conditional', 'oracle'])
def test_does_not_repair_invalid_kernel_or_oracle(kind):
    raw, hs = records(), histories()
    if kind == 'probability':
        raw[0]['predictive_probs'][0] += .01
    elif kind == 'joint':
        raw[0]['log_joint_next'][0] += .01
    elif kind == 'conditional':
        raw[0]['maximum_conditional_normalization_error'] = 1e-5
    else:
        hs[0]['oracle_probs'] = [-.1, 1.1]
    with pytest.raises(ValueError):
        v.analyze_arm({'shared': raw}, hs)


def test_concentrated_evidence_fails_without_omitting_reference():
    raw = records()
    raw[0].update(component(-.01, raw[0]['predictive_probs']))
    for r in raw[1:]:
        r.update(component(-30., r['predictive_probs']))
    a = v.analyze_arm({'shared': raw}, histories())
    assert not a['summary']['basic_passed']
    assert len(a['rows']) == 192
    assert a['references']['shared']['effective_replica_count'] < 8
    assert a['references']['shared']['maximum_single_replica_likelihood_share'] > .2


def test_source_and_numerical_uncertainty_stay_separate():
    a = analyze()
    s = a['summary']
    assert s['bayes']['se'] > s['numerical_mcse_bits']
    assert s['numerical_budget_bits'] == pytest.approx(min(.01/3, .25*s['denoised_source_se_bits']))
    assert s['numerical_allowance_bits'] == pytest.approx(2*s['numerical_mcse_bits']+2*s['bias_allowance_bits'])
    assert set(s['transformers']) == set(s['paired']) == {'seed0', 'seed1'}


def test_four_arm_decision_uses_one_sum_not_separate_per_arm_passes():
    arms = {name: analyze(j*10000, scale=.0001) for j, name in enumerate('ABCD')}
    report = v.validate_campaign(arms, arms['D'])
    expected = arms['D']['summary']['numerical_allowance_bits']
    expected += sum(x['allowance_bits'] for x in report['comparisons'].values())
    assert report['total_numerical_allowance_bits'] == pytest.approx(expected)
    assert report['passed']
    # Each contrast alone fits the budget, but their sum must fail.
    changed = copy.deepcopy(arms)
    budget = report['numerical_budget_bits']
    for j, name in enumerate('ABCD'):
        changed[name]['summary']['bayes']['mean'] += j*.6*budget
    failed = v.validate_campaign(changed, changed['D'])
    assert all(c['allowance_bits'] < budget for c in failed['comparisons'].values())
    assert failed['total_numerical_allowance_bits'] > budget
    assert not failed['passed']


def test_all_stochastic_arms_need_independent_seeds():
    arms = {name: analyze(j*10000) for j, name in enumerate('ABCD')}
    arms['C'] = arms['A']
    with pytest.raises(ValueError, match='independent seeds'):
        v.validate_campaign(arms, arms['D'])


def test_deterministic_refinement_retains_same_record_covariance():
    base = analyze(scale=.02)
    identical = v.compare_arms(base, base, shared_records=True)
    assert identical['difference_mcse_bits'] == 0.
    hs = histories()
    for h in hs:
        h['fixed_components'][0] = component(-3.2, [.7001, .2999])
    refined = analyze(scale=.02, hs=hs)
    paired = v.compare_arms(base, refined, shared_records=True)
    assert paired['difference_mcse_bits'] < .001*base['summary']['numerical_mcse_bits']
    independent = analyze(offset=10000, scale=.02, hs=hs)
    with pytest.raises(ValueError, match='identical shared records'):
        v.compare_arms(base, independent, shared_records=True)


def test_refinement_cannot_change_cohort_or_fixed_components_silently():
    a = analyze()
    hs = histories()
    hs[0]['oracle_probs'] = [.3, .7]
    with pytest.raises(ValueError, match='cohorts differ'):
        v.compare_arms(a, analyze(10000, hs=hs))
    arms = {name: analyze(j*10000) for j, name in enumerate('ABCD')}
    arms['C']['fixed_components_sha256'] = 'changed'
    with pytest.raises(ValueError, match='same fixed components'):
        v.validate_campaign(arms, arms['D'])


def test_exact_short_checks_new_mean_and_keeps_old_flags_unchanged():
    hs = histories()
    truth = [dict(family=h['family'], index=h['index'], bayes_regret_bits=direct_loss(records(scale=0), h)[0])
             for h in hs]
    arms = {name: analyze(j*10000, scale=.0001) for j, name in enumerate('ABCD')}
    good = v.exact_short_validation(arms, truth, cheap_refined=arms['D'])
    assert good['passed'] and good['old_component_calibration_unchanged']
    assert set(good['arms']) == {'A', 'B', 'C', 'D', 'D_cheap_refined'}
    bad_truth = copy.deepcopy(truth)
    for row in bad_truth:
        row['bayes_regret_bits'] += .01
    assert not v.exact_short_validation(arms, bad_truth)['passed']


def test_zero_oracle_mass_is_allowed_without_nan():
    hs = histories()
    hs[0]['oracle_probs'] = [0., 1.]
    a = analyze(hs=hs)
    assert math.isfinite(a['rows'][0]['bayes_regret_bits'])
