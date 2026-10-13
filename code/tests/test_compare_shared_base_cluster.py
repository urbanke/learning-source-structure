"""Paired mixture arithmetic and fail-closed pilot receipt handling."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
sys.path.insert(0, str(ROOT / 'code/scripts'))
import compare_shared_base_cluster as comparison
import shared_base_cluster as cluster
from shared_base_short_evidence import SharedBaseShortEvidence


@pytest.fixture(scope='module')
def catalog():
    return SharedBaseShortEvidence()


@pytest.fixture
def plan(tmp_path, monkeypatch, catalog):
    value = json.loads(comparison.PLAN.read_text())
    value['stages']['pilot16']['requires'] = []
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(value))
    monkeypatch.setattr(comparison, 'SharedBaseShortEvidence', lambda: catalog)
    return path


def receipt_for(plan, logs=None, partial=False):
    context = cluster.read_plan(plan, 'pilot16')
    shards = []
    ready = dict(compiled_kernel_available=True, store_sealed=True, store_read_only=True)
    for ci, cid in enumerate(context['config']['case_ids']):
        shard = cluster.new_shard(context, cid, dict(compiled_kernel_available=True), 0, 16, ready)
        for rep in range(16):
            if partial and ci == 11 and rep == 15:
                continue
            logz = -60.-ci if logs is None else logs(ci, rep)
            estimate = dict(log_evidence=logz, seed=cluster.expected_seed(context, cid, rep),
                            sampler_class='SharedBaseMCMC', elapsed_seconds=1.,
                            **{key: context['config'][key] for key in ('particles', 'temperatures', 'moves', 'n_tokens')})
            cluster.append_record(shard, context, rep, estimate)
        shard.update(complete=not (partial and ci == 11), status='time_limit' if partial and ci == 11 else 'completed')
        shards.append(shard)
    return cluster.merge_reports(context, shards)[0]


def save_receipt(plan, value, name='receipt.json'):
    path = plan.parent/name
    path.write_text(json.dumps(value))
    return path


def test_supplied_first_token_and_equal_mixture_arithmetic():
    first = -math.log(64)
    zero, weights = comparison.mixture_loss([first]*3, [1/3]*3)
    assert zero == pytest.approx(0., abs=1e-14)
    loss, weights = comparison.mixture_loss([first+math.log(p) for p in (.1, .2, .3)], [1/3]*3)
    assert loss == pytest.approx(-math.log2(.2))
    np.testing.assert_allclose(weights, [1/6, 1/3, 1/2])
    with pytest.raises(ValueError):
        comparison.mixture_loss([first]*3, [1/3, 1/3, float('nan')])


def test_exact_short_points_have_full_balanced_pairing(plan):
    result = comparison.compare(plan, [])
    assert [p['time'] for p in result['points']] == [1, 4]
    assert [p['time'] for p in result['pending']] == [16, 64, 256, 1024]
    assert len(result['sequence_records']) == 24
    rows = [r for r in result['sequence_records'] if r['time'] == 4]
    for model in comparison.MODELS:
        families = [[r['transformer_minus_mixture_bits'][model] for r in rows if r['family'] == f]
                    for f in comparison.FAMILIES]
        point = result['points'][1]['balanced'][model]
        assert point['mean_bits'] == pytest.approx(np.mean(families))
        expected_se = math.sqrt(sum(np.var(v, ddof=1)/4 for v in families))/3
        assert point['sequence_se_bits'] == pytest.approx(expected_se)
        assert point['numerical_mc_se_bits'] == 0.
    assert result['production_reference_validated'] is False


def test_likelihood_mean_before_log_and_mixture_sensitivity(plan):
    logs = lambda ci, rep: -60.-ci + (math.log(.25) if rep%2 else 0.)
    receipt = receipt_for(plan, logs)
    assert receipt['gate']['precision_passed'] is False
    path = save_receipt(plan, receipt)
    # One failed precision screen suppresses the entire time in ordinary output.
    result = comparison.compare(plan, [path])
    assert 16 not in [p['time'] for p in result['points']]
    diagnostic = comparison.compare(plan, [path], diagnostic=True)
    point = next(p for p in diagnostic['points'] if p['time'] == 16)
    assert point['numerical_status'] == 'unresolved_numerical_precision'
    first = next(r for r in diagnostic['sequence_records'] if r['time'] == 16)
    expected_shared = -60+math.log(.625)
    assert first['shared_log_likelihood'] == pytest.approx(expected_shared)
    assert first['shared_log_likelihood'] != pytest.approx(-60+math.log(.25)/2)
    data = json.loads(plan.read_text())['comparison']['cases'][first['case_id']]
    z = (math.exp(data['product_log_marginals'][2])+math.exp(expected_shared)
         +math.exp(data['uniform_base_log_marginals'][2]))/3
    assert first['mixture_loss_bits'] == pytest.approx(-math.log2(z*64))
    stat = receipt['case_summaries'][first['case_id']]
    expected_mc = first['posterior_family_weights'][1]*stat['loss_mc_se_bits_delta_method']
    assert first['mixture_mc_se_bits_delta_method'] == pytest.approx(expected_mc)


def test_complete_precise_receipt_and_partial_receipt_scope(plan):
    full = receipt_for(plan)
    path = save_receipt(plan, full)
    result = comparison.compare(plan, [path])
    assert [p['time'] for p in result['points']] == [1, 4, 16]
    assert result['points'][-1]['numerical_status'] == 'precision_check_only'
    assert result['receipt_sha256'][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    incomplete = save_receipt(plan, receipt_for(plan, partial=True), 'partial.json')
    result = comparison.compare(plan, [incomplete], diagnostic=True)
    assert [p['time'] for p in result['points']] == [1, 4]
    assert result['pending'][0]['reason'] == 'missing or incomplete Bayesian calculation'
    with pytest.raises(ValueError, match='Duplicate stage'):
        comparison.compare(plan, [path, path])


@pytest.mark.parametrize('field', ['nested_mean', 'nested_se', 'top_mean', 'gate', 'raw_value',
                                   'seed_rehashed', 'coverage', 'missing_case', 'duplicate_replica', 'plan_hash'])
def test_forged_or_inconsistent_receipt_rejected(plan, field):
    receipt = receipt_for(plan)
    cid = next(iter(receipt['cases']))
    case = receipt['cases'][cid]
    if field == 'nested_mean':
        # Break the alias created by the merger, as JSON storage does.
        case['statistics'] = copy.deepcopy(case['statistics'])
        case['statistics']['log_mean_evidence_nats'] += .01
    elif field == 'nested_se':
        case['statistics'] = copy.deepcopy(case['statistics'])
        case['statistics']['loss_mc_se_bits_delta_method'] += .01
    elif field == 'top_mean':
        receipt['case_summaries'][cid]['log_mean_evidence_nats'] += .01
    elif field == 'gate':
        receipt['gate']['precision_passed'] = False
    elif field == 'raw_value':
        case['records'][0]['estimate']['log_evidence'] += .1
    elif field == 'seed_rehashed':
        record = case['records'][0]
        record['seed'] += 1
        record['estimate']['seed'] += 1
        record['record_sha256'] = cluster.fingerprint({k:v for k,v in record.items() if k!='record_sha256'})
    elif field == 'coverage':
        receipt['coverage'][cid].pop()
    elif field == 'missing_case':
        del receipt['cases'][cid]
    elif field == 'duplicate_replica':
        case['records'][-1] = copy.deepcopy(case['records'][0])
    else:
        receipt['plan_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        comparison.compare(plan, [save_receipt(plan, receipt)], diagnostic=True)


def test_forged_precision_pass_cannot_promote_noisy_result(plan):
    receipt = receipt_for(plan, lambda ci, rep: -60.-ci + (1. if rep%2 else -1.))
    receipt['gate'].update(precision_passed=True, precision_met=True)
    with pytest.raises(ValueError, match='raw repetitions'):
        comparison.compare(plan, [save_receipt(plan, receipt)])


@pytest.mark.parametrize('field', ['weight', 'ordering', 'time', 'convention', 'coverage',
                                   'stage_length', 'short_catalog', 'transformer_shape'])
def test_wrong_pairing_or_source_prior_is_rejected(plan, field):
    value = json.loads(plan.read_text())
    c = value['comparison']
    cid = next(iter(c['cases']))
    if field == 'weight': c['family_prior'] = [.2, .3, .5]
    elif field == 'ordering': c['family_order'].reverse()
    elif field == 'time': c['prefixes'][1] = 5
    elif field == 'convention': c['loss_convention'] = 'score the first token'
    elif field == 'coverage': del c['cases'][cid]
    elif field == 'stage_length': value['stages']['pilot16']['n_tokens'] = 16
    elif field == 'short_catalog': c['cases'][cid]['shared_base_short_log_marginals'][0] += .1
    else: c['cases'][cid]['transformer_cumulative_loss_bits']['pooled_seed0'].pop()
    plan.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        comparison.compare(plan, [])
