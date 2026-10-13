"""New predictive export checked against independent finite and saved exact cases."""
from itertools import permutations
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/transformer'))
sys.path.insert(0, str(ROOT/'code/scripts'))
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC
from shared_base_evidence_smc import estimate_log_evidence
from shared_base_predictive_smc import (estimate, pool_repetitions, combine_components,
                                       component_prediction, checked_probs, mixture_regret_mcse)


def sampler():
    return SharedBaseMCMC(3, RankedLSARowEvidence(3, [.6, 1.3], [0]), [.3, .7])


def exact_iid_joint(mc, tokens):
    # Independent enumeration, with no row kernel or posterior predictor.
    return sum(aw*np.prod(mc.kernel.base_laws[ai, list(ranks)][tokens])
               for ai, aw in enumerate(mc.alpha_weights)
               for ranks in permutations(range(mc.d)))/math.factorial(mc.d)


@pytest.mark.parametrize('seed,resample_fraction', [(3, .01), (7, .75), (401, 1.)])
def test_new_loop_has_identical_original_seeded_normalizer_and_moves(seed, resample_fraction):
    kwargs = dict(particles=16, temperatures=8, moves=2, seed=seed,
                  resample_fraction=resample_fraction)
    tokens = [0, 0, 1, 0, 2, 1]
    original = estimate_log_evidence(sampler(), tokens, **kwargs)
    new = estimate(sampler(), tokens, **kwargs)
    for key in ('log_evidence', 'move_attempts', 'move_accepts', 'effective_sample_sizes',
                'resampled', 'initial_particle_ancestry_counts'):
        assert new[key] == original[key]
    np.testing.assert_allclose(np.exp(new['log_joint_next']).sum(),
                               math.exp(new['log_evidence']), atol=1e-15)


def test_unnormalized_terminal_moments_match_independent_successor_evidence():
    mc = sampler()
    x = [0, 0, 1, 2]
    results = [estimate(mc, x, particles=32, temperatures=8, moves=1, seed=i) for i in range(32)]
    z = np.exp([r['log_evidence'] for r in results])
    joint = z[:, None]*np.array([r['predictive_probs'] for r in results])
    wanted = np.array([exact_iid_joint(mc, x+[j]) for j in range(3)])
    se = joint.std(axis=0, ddof=1)/math.sqrt(len(joint))
    assert np.all(abs(joint.mean(axis=0)-wanted) < 5*se)
    np.testing.assert_allclose(joint.mean(axis=0), wanted, rtol=.05)
    pooled = pool_repetitions(results)
    np.testing.assert_allclose(np.exp(pooled['log_joint_next']), joint.mean(axis=0), atol=1e-16)
    np.testing.assert_allclose(pooled['predictive_probs'], joint.mean(axis=0)/z.mean(), atol=1e-15)


def test_first_token_evidence_exact_but_prediction_need_not_be_uniform():
    result = estimate(sampler(), [0], particles=256, temperatures=1, moves=1, seed=8)
    assert result['log_evidence'] == pytest.approx(-math.log(3), abs=1e-14)
    assert result['predictive_probs'][0] > 1/3


def test_pooling_uses_evidence_weights_and_stable_logscale():
    records = [dict(log_evidence=-1000, predictive_probs=[.9, .1]),
               dict(log_evidence=-1000+math.log(3), predictive_probs=[.1, .9])]
    pooled = pool_repetitions(records)
    np.testing.assert_allclose(pooled['predictive_probs'], [.3, .7], atol=1e-14)
    assert not np.allclose(pooled['predictive_probs'], [.5, .5])
    assert pooled['log_evidence'] == pytest.approx(-1000+math.log(2))


def test_family_weights_and_successor_identity_use_past_evidence():
    logs = np.log([.01, .04, .02])
    p = np.array([[.9, .1], [.2, .8], [.6, .4]])
    prior = np.array([.2, .3, .5])
    answer = combine_components(logs, p, prior)
    joint = (prior*np.exp(logs))@p
    np.testing.assert_allclose(np.exp(answer['log_joint_next']), joint, atol=1e-16)
    np.testing.assert_allclose(answer['predictive_probs'], joint/joint.sum(), atol=1e-15)
    np.testing.assert_allclose(answer['posterior_family_weights'],
                               prior*np.exp(logs)/sum(prior*np.exp(logs)), atol=1e-15)
    assert not np.allclose(answer['predictive_probs'], prior@p)


def test_actual_mixture_mcse_preserves_z_prediction_covariance():
    records = [dict(log_evidence=math.log(z), predictive_probs=p)
               for z, p in ((.01, [.8, .2]), (.03, [.2, .8]), (.02, [.4, .6]))]
    fixed = [dict(log_evidence=math.log(z), predictive_probs=p)
             for z, p in ((.02, [.5, .5]), (.04, [.6, .4]))]
    oracle = np.array([.7, .3])
    se = mixture_regret_mcse(records, fixed, oracle)
    # Independent directional differentiation of the complete ratio.
    z = np.exp([r['log_evidence'] for r in records])
    joint = z[:, None]*np.array([r['predictive_probs'] for r in records])
    fixed_z = np.exp([r['log_evidence'] for r in fixed])
    fixed_joint = fixed_z@np.array([r['predictive_probs'] for r in fixed])
    def loss(zz, jj):
        pred = (fixed_joint+jj)/(fixed_z.sum()+zz)
        return -np.dot(oracle, np.log2(pred))
    eps = 1e-5
    changes = []
    for zi, ji in zip(z, joint):
        plus = loss(z.mean()+eps*(zi-z.mean()), joint.mean(axis=0)+eps*(ji-joint.mean(axis=0)))
        minus = loss(z.mean()-eps*(zi-z.mean()), joint.mean(axis=0)-eps*(ji-joint.mean(axis=0)))
        changes.append((plus-minus)/(2*eps))
    assert se == pytest.approx(np.std(changes, ddof=1)/math.sqrt(3), rel=1e-8)


@pytest.mark.parametrize('value', [[.2, .3], [0., 1.], [-.1, 1.1], [float('nan'), .5]])
def test_invalid_probabilities_rejected_without_renormalizing(value):
    with pytest.raises(ValueError):
        checked_probs(value)


def test_exact_short_all_families_on_original_frozen_histories(monkeypatch):
    # These existing short-prefix numerical references are already validated.
    # No SMC inference is run here. This verifies the new pooling adapter and
    # the inexpensive component reconstruction against saved independent work.
    raw = ROOT/'output/multifamily_results_2026-09-16'
    saved = ROOT/'output/multifamily_bayes_2026-09-16/next_token_regret'
    store = Path.home()/'Projects/product_model_with_memory/tables/anchors_prod'
    src = Path.home()/'Projects/product_model_with_memory/src'
    if not (store/'manifest.json').exists() or not raw.exists() or not saved.exists():
        pytest.skip('requires existing sealed store and frozen scientific short-reference fixtures')
    for key, value in dict(LSA_STORE=str(store), PMM_SRC=str(src), PMM_UNIVERSAL_TABLES=str(store),
        PMM_PHI_LADDER_EVERY='1', PMM_PHI_LADDER_DEGREE='11', PMM_PHI_SADDLE_MIN_L='54').items():
        monkeypatch.setenv(key, value)
    from mixture_family_evidence import FamilyEvidence
    from shared_base_short_evidence import SharedBaseShortEvidence
    providers = [FamilyEvidence('product'), SharedBaseShortEvidence(), FamilyEvidence('uniform_base')]
    for family in ('product', 'shared_base', 'uniform_base'):
        with np.load(raw/f'heldout/{family}_prior.npz', allow_pickle=False) as a:
            tokens = a['tokens'][5].copy()
        with np.load(saved/f'{family}_prior.npz', allow_pickle=False) as a:
            wanted = a['mixture_probs'][5].copy()
        for column, observed in enumerate((1, 4)):
            components = [component_prediction(ref, tokens[:observed]) for ref in providers]
            mixture = combine_components([r['log_evidence'] for r in components],
                                         [r['predictive_probs'] for r in components])
            np.testing.assert_allclose(mixture['predictive_probs'], wanted[column], atol=2e-10, rtol=0)
            assert sum(mixture['predictive_probs']) == pytest.approx(1., abs=1e-9)


def test_cli_freezes_all_families_without_sampling(tmp_path):
    from pooled_predictive_pilot import freeze_plan, load_case
    raw = ROOT/'output/multifamily_results_2026-09-16'
    if not raw.exists():
        pytest.skip('requires existing frozen scientific inputs')
    plan = freeze_plan(raw, tmp_path/'plan.json')
    assert len(plan['cases']) == 192
    assert plan['observation_times'] == [16, 64]
    assert len(plan['development_cases']) == 12
    for family in ('product', 'shared_base', 'uniform_base'):
        assert [c['index'] for c in plan['cases'] if c['family'] == family] == list(range(5, 69))
        x, oracle, models, _ = load_case(raw, plan, family, 5, 16)
        assert len(x) == 16 and oracle.shape == (64,) and len(models) == 2
        dx, _, _, _ = load_case(raw, plan, family, 1, 64, development=True)
        assert len(dx) == 64
    with pytest.raises(ValueError, match='outside frozen'):
        load_case(raw, plan, 'product', 0, 16)
    with pytest.raises(ValueError, match='already exists'):
        freeze_plan(raw, tmp_path/'plan.json')
    plan['input_sha256']['heldout/product_prior.npz'] = 'wrong'
    with pytest.raises(ValueError, match='frozen file'):
        load_case(raw, plan, 'product', 5, 16)


def test_single_case_timeout_reuses_saved_repetition_without_sampling(tmp_path, monkeypatch):
    import pooled_predictive_pilot as cli
    raw = ROOT/'output/multifamily_results_2026-09-16'
    store = Path.home()/'Projects/product_model_with_memory/tables/anchors_prod'
    src = Path.home()/'Projects/product_model_with_memory/src'
    if not raw.exists() or not store.exists():
        pytest.skip('requires existing provenance fixtures')
    plan_path = tmp_path/'selection.json'
    cli.freeze_plan(raw, plan_path)
    monkeypatch.setattr(cli, 'RankedLSARowEvidence', lambda *a, **kw: None)
    monkeypatch.setattr(cli, 'SharedBaseMCMC', lambda *a, **kw: None)
    class UniformFixture:
        def __init__(self, *a, **kw):
            pass
        def log_joint(self, tokens):
            return -len(tokens)*math.log(64)
    monkeypatch.setattr(cli, 'FamilyEvidence', UniformFixture)
    calls = []
    def fake_estimate(sampler, tokens, **kw):
        calls.append(kw['seed'])
        if len(calls) == 2:
            raise TimeoutError('intentional bounded-smoke interruption')
        return dict(log_evidence=-len(tokens)*math.log(64), predictive_probs=[1/64]*64,
            seed=kw['seed'], n_observed=len(tokens), sampler_seconds=2., prediction_seconds=3.,
            synthetic_test_fixture=True)
    monkeypatch.setattr(cli, 'estimate', fake_estimate)
    args = SimpleNamespace(raw=raw, plan=plan_path, family='product', index=1,
        observed=16, development=True, particles=2, temperatures=1, moves=1,
        replicates=2, seconds_per_replica=1., alpha_order=16, seed=4,
        store=store, src=src, out=tmp_path/'smoke')
    with pytest.raises(TimeoutError):
        cli.run(args)
    reports = list(args.out.glob('*/replica*.json'))
    assert len(reports) == 1
    before = reports[0].read_bytes()
    folder = cli.run(args)
    assert reports[0].read_bytes() == before
    assert len(calls) == 3 and calls[1] == calls[2] and calls[0] != calls[2]
    result = __import__('json').loads((folder/'result.json').read_text())
    assert result['complete'] and not result['scientific_reference_validated']
    assert len(result['replica_sha256']) == 2
    reports[0].write_text(reports[0].read_text()+'\n')
    with pytest.raises(ValueError, match='saved replica changed'):
        cli.run(args)
