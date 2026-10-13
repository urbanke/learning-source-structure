"""Controlled scaling evaluation: fixed inputs, strict checkpoints and pairing."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'transformer'))
import eval_hidden_factor as original
import hidden_factor_scaling_eval as scaling
import train_hidden_factor as training
from model import GPT


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config(seed=0, steps=1, width=8, layers=1):
    return training.validate_config(dict(schema='hidden-factor-training-v1', seed=seed,
        source_dimensions=[2, 2], n_ctx=4, batch=2, steps=steps, warmup=1, lr=.0003,
        wd=.1, clip=1., n_layer=layers, n_head=1, d_model=width, val_n=2,
        log_every=1, val_every=1, ckpt_every=1, weight_steps=[steps], validation_seed=456, val_chunk=1))


def template(cfg):
    return {key: value for key, value in cfg.items() if key not in ('seed', 'source_dimensions')}


def weights(cfg, provenance=None, continuation=False):
    cfg = copy.deepcopy(cfg)
    if provenance is not None:
        cfg['scaling_provenance'] = provenance
    torch.manual_seed(90+cfg['seed']+cfg['d_model'])
    model = GPT(4, cfg['n_ctx'], cfg['n_layer'], cfg['n_head'], cfg['d_model'])
    contract = training.source_contract(cfg['source_dimensions'])
    receipt = dict(config=cfg, config_hash=original.fingerprint(cfg), model=model.cfg,
        n_params=model.n_params(), source_contract=contract, source_contract_hash=original.fingerprint(contract),
        implementation_hashes=training.implementation_hashes(), validation_hashes=scaling.monitoring_hashes((2, 2), 4, 2, 456),
        precision='float32', runtime={'device_type': 'cpu'}, torch_version=str(torch.__version__))
    return dict(schema='hidden-factor-scaling-weights-v1' if continuation else 'hidden-factor-weights-v1',
                model=model.state_dict(), step=cfg['steps'], cfg=receipt, config_hash=receipt['config_hash'],
                implementation_hashes=receipt['implementation_hashes'], completed_sequences=cfg['steps']*cfg['batch'])


@pytest.fixture
def experiment(tmp_path):
    baseline = tmp_path/'baseline'
    original.create_cohort(baseline, (2, 2), n_seq=8, length=5, times=[1, 4], seed=937)
    original.references(baseline, max_sequences=4)
    checkpoints = {}
    for seed in (0, 1):
        ckpt = weights(config(seed))
        path = tmp_path/f'baseline_seed{seed}.pt'
        torch.save(ckpt, path)
        original.evaluate_model(baseline, path, batch_size=2, max_sequences=4)
        checkpoints[seed] = path
    summary = original.summarize(baseline, max_sequences=4)
    cohort = summary['cohort']
    plan = dict(schema='hidden-factor-scaling-plan-v1', source_dimensions=[2, 2], seeds=[0, 1],
        implementation_sha256=training.implementation_hashes(),
        baseline=dict(config=template(config()), final_step=1, completed_sequences=2,
            checkpoint_sha256={str(seed): original.file_hash(path) for seed, path in checkpoints.items()},
            model_records_sha256=summary['model_records_sha256'], reference_records_sha256=summary['reference_records_sha256'],
            summary_sha256=original.file_hash(baseline/'summary_prefix_000004.json')),
        arms=dict(longsmall=dict(mode='continuation', config=template(config(steps=4)),
                    parent_config=template(config()), parent_step=1, parent_completed_sequences=2,
                    continuation_lr=.00003, additional_sequences=6, n_params=weights(config())['cfg']['n_params'], interpretation='Additional optimization.'),
                  large=dict(mode='fresh', config=template(config(width=16, layers=2)),
                    n_params=weights(config(width=16, layers=2))['cfg']['n_params'], interpretation='Joint architecture change.')),
        test=dict(cohort_id=cohort['cohort_id'], data_sha256=cohort['data_sha256'], n_evaluate=4, n_planned=8,
                  length=5, times=[1, 4], role='development'),
        primary=dict(quantity='Next-token regret gap', time=4, steps=dict(longsmall=4, large=1)))
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan))
    plan_hash = original.file_hash(plan_path)
    arms = {}
    for arm in plan['arms']:
        for seed in (0, 1):
            parent = None
            if arm == 'longsmall':
                parent = dict(schema='hidden-factor-checkpoint-v1', checkpoint_sha256='b'*64,
                    weights_checkpoint_sha256=plan['baseline']['checkpoint_sha256'][str(seed)],
                    config_hash=original.fingerprint(config(seed)), step=1, completed_sequences=2,
                    implementation_hashes=training.implementation_hashes(), runtime={'device_type': 'cpu'},
                    validation_hashes=scaling.monitoring_hashes((2, 2), 4, 2, 456))
            provenance = dict(plan_sha256=plan_hash, arm=arm, seed=seed,
                trainer_sha256=original.file_hash(scaling.HERE/'train_hidden_factor_scaling.py'), parent=parent)
            ckpt = weights(scaling.planned_config(plan, arm, seed), provenance, continuation=(arm == 'longsmall'))
            if arm == 'longsmall':
                ckpt['cfg']['learning_rate_schedule'] = dict(kind='constant continuation', value=.00003, starts_after_step=1)
            path = tmp_path/f'{arm}_seed{seed}.pt'
            torch.save(ckpt, path)
            arms[arm, seed] = path
    return dict(plan=plan, plan_path=plan_path, plan_hash=plan_hash, baseline=baseline,
                arm_checkpoints=arms, output=tmp_path/'new-evaluations')


def tree_hash(path):
    return {str(file.relative_to(path)): original.file_hash(file) for file in path.rglob('*') if file.is_file()}


def test_scoring_reuses_frozen_data_without_reference_calls_and_pairs_correctly(experiment, monkeypatch):
    e = experiment
    before = tree_hash(e['baseline'])
    def forbidden(*args, **kwargs):
        raise AssertionError('New source/reference computation is forbidden')
    monkeypatch.setattr(original, 'references', forbidden)
    monkeypatch.setattr(original, 'reference_one', forbidden)
    monkeypatch.setattr(original, 'create_cohort', forbidden)
    for (arm, seed), path in e['arm_checkpoints'].items():
        manifest = scaling.evaluate(e['plan_path'], e['baseline'], arm, path, e['output'], batch_size=2)
        assert manifest['complete_for_requested_pilot']
        assert manifest['identity']['name'] == f'seed{seed}'
        assert manifest['identity']['test_role'] == 'development'
    assert tree_hash(e['baseline']) == before
    assert not (e['output']/'longsmall/cohort.npz').exists()
    output = e['output']/'comparison.json'
    result = scaling.summarize(e['plan_path'], e['baseline'], e['output'], output)
    assert result['complete_for_requested_comparison'] and not result['new_bayesian_computation']
    assert len(result['model_provenance']) == 4 and result['test_role'] == 'development'
    group = result['groups'][0]
    left = scaling.read_complete(e['output']/'large', 'models/seed0', 4, e['plan']['test']['cohort_id'], 8)[0]['next_kl_bits']
    right = scaling.read_complete(e['baseline'], 'models/seed0', 4, e['plan']['test']['cohort_id'], 8)[0]['next_kl_bits']
    saved = group['paired']['large_seed0_minus_baseline_seed0']['next_kl_bits']
    np.testing.assert_allclose(saved['mean'], (left-right).mean(0), atol=1e-15)
    np.testing.assert_allclose(saved['se'], (left-right).std(0, ddof=1)/2, atol=1e-15)
    assert tree_hash(e['baseline']) == before
    scaling.evaluate(e['plan_path'], e['baseline'], 'large', e['arm_checkpoints']['large', 0], e['output'], batch_size=2)
    assert tree_hash(e['baseline']) == before


@pytest.mark.parametrize('change', ['unfinished', 'model', 'source', 'arm', 'parent', 'test_leak'])
def test_checkpoint_drift_or_nonfinal_refused(experiment, change):
    e = experiment
    ckpt = torch.load(e['arm_checkpoints']['longsmall', 0], weights_only=False)
    if change == 'unfinished':
        ckpt['step'] -= 1
    elif change == 'model':
        ckpt['cfg']['model']['d_model'] *= 2
    elif change == 'source':
        ckpt['implementation_hashes']['hidden_factor.py'] = '0'*64
    elif change == 'arm':
        ckpt['cfg']['config']['scaling_provenance']['arm'] = 'large'
    elif change == 'parent':
        ckpt['cfg']['config']['scaling_provenance']['parent']['weights_checkpoint_sha256'] = '0'*64
    elif change == 'test_leak':
        ckpt['cfg']['validation_hashes'] = {'hidden_factor': e['plan']['test']['data_sha256']}
    with pytest.raises(ValueError):
        scaling.validate_checkpoint(ckpt, e['plan'], e['plan_hash'], 'longsmall')


def test_missing_inference_and_modified_frozen_input_refused(experiment):
    e = experiment
    with pytest.raises(ValueError, match='Missing final inference'):
        scaling.summarize(e['plan_path'], e['baseline'], e['output'], e['output']/'comparison.json')
    assert not (e['output']/'comparison.json').exists()
    plan = copy.deepcopy(e['plan'])
    plan['baseline']['reference_records_sha256'] = '0'*64
    with pytest.raises(ValueError, match='references changed'):
        scaling.load_frozen_baseline(e['baseline'], plan)
    with pytest.raises(ValueError, match='separate'):
        scaling.stage_inputs(e['baseline'], e['baseline']/'large')


def test_cached_scores_cannot_switch_checkpoint(experiment):
    e = experiment
    path = e['arm_checkpoints']['large', 0]
    scaling.evaluate(e['plan_path'], e['baseline'], 'large', path, e['output'], batch_size=2)
    ckpt = torch.load(path, weights_only=False)
    ckpt['model']['head.weight'][0, 0] += .1
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='different checkpoint'):
        scaling.evaluate(e['plan_path'], e['baseline'], 'large', path, e['output'], batch_size=2)


def test_frozen_entropy_audit_allows_roundoff_but_not_changed_groups(experiment, monkeypatch):
    e = experiment
    actual = scaling.entropy_strata
    def roundoff(matrices):
        entropy, medians, strata = actual(matrices)
        return entropy+2e-16, medians+2e-16, strata
    monkeypatch.setattr(scaling, 'entropy_strata', roundoff)
    scaling.load_frozen_baseline(e['baseline'], e['plan'])
    def changed_group(matrices):
        entropy, medians, strata = actual(matrices)
        strata[0] = (int(strata[0])+1) % 4
        return entropy, medians, strata
    monkeypatch.setattr(scaling, 'entropy_strata', changed_group)
    with pytest.raises(ValueError, match='entropy strata changed'):
        scaling.load_frozen_baseline(e['baseline'], e['plan'])
