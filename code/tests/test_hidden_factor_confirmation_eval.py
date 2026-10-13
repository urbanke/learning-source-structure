"""Confirmation suffix isolation, exact model binding, and paired scoring."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'transformer'))
import hidden_factor_confirmation_eval as confirmation
import eval_hidden_factor as original
import hidden_factor_scaling_eval as scaling
from test_hidden_factor_scaling_eval import experiment, one_thread, tree_hash


@pytest.fixture
def confirmation_experiment(experiment, tmp_path):
    e = experiment
    baseline_root, scaling_root = tmp_path/'original', tmp_path/'scaling'
    baseline_root.mkdir()
    shutil.copytree(e['baseline'], baseline_root/'eval')
    for seed in (0, 1):
        path = baseline_root/f'train/seed{seed}/model_1.pt'
        path.parent.mkdir(parents=True)
        shutil.copy2(tmp_path/f'baseline_seed{seed}.pt', path)
    for (arm, seed), checkpoint in e['arm_checkpoints'].items():
        path = scaling_root/f'train/{arm}/seed{seed}'/f"model_{e['plan']['primary']['steps'][arm]}.pt"
        path.parent.mkdir(parents=True)
        shutil.copy2(checkpoint, path)
        scaling.evaluate(e['plan_path'], baseline_root/'eval', arm, path, scaling_root/'eval', batch_size=2)
    # This fixture creates references before the consumer is invoked. The
    # consumer tests then forbid any reference or held-out source calculation.
    original.references(baseline_root/'eval')
    return dict(e, baseline_root=baseline_root, scaling_root=scaling_root, out=tmp_path/'confirmation')


def test_token_time_and_cumulative_scoring_have_independent_expected_values():
    tokens = np.array([[3, 0, 1, 2, 0]])
    probabilities = np.array([[[.4, .2, .2, .2], [.1, .6, .2, .1],
                               [.1, .1, .7, .1], [.8, .1, .05, .05]]])
    times = np.array([1, 4])
    oracle = np.array([[[.25, .25, .25, .25], [.5, .25, .125, .125]]])
    s = confirmation.score_log_probabilities(np.log2(probabilities), tokens, oracle, times)
    np.testing.assert_allclose(s['cumulative_loss_bits'][0], [-np.log2(.4), -np.log2(.4*.6*.7*.8)])
    np.testing.assert_allclose(s['next_loss_bits'][0], -np.log2([.4, .8]))
    expected_kl = (oracle*np.log2(oracle/probabilities[:, [0, 3]])).sum(-1)
    np.testing.assert_allclose(s['next_kl_bits'], expected_kl)


def test_end_to_end_uses_only_untouched_suffix_and_saved_bayes(confirmation_experiment, monkeypatch):
    e = confirmation_experiment
    before = {key: tree_hash(e[key]) for key in ('baseline_root', 'scaling_root')}
    def forbidden(*args, **kwargs):
        raise AssertionError('Confirmation must not generate sources or compute Bayesian references')
    monkeypatch.setattr(original, 'create_cohort', forbidden)
    monkeypatch.setattr(original, 'reference_one', forbidden)
    monkeypatch.setattr(original, 'references', forbidden)
    result = confirmation.run(e['plan_path'], e['baseline_root'], e['scaling_root'], e['out'], device='cpu', batch_size=2)
    assert result['complete'] and result['test_role'] == 'untouched_confirmation'
    assert result['evaluated_n_sequences'] == 4 and result['sequence_start'] == 4 and result['sequence_stop'] == 8
    assert result['development_n_sequences'] == 4 and not result['first_token_scored']
    assert not result['new_training'] and not result['new_bayesian_computation']
    assert len(result['model_receipts']) == 6
    assert all(tree_hash(e[key]) == before[key] for key in before)
    inputs = confirmation.load_inputs(e['plan_path'], e['baseline_root'], e['scaling_root'])
    large, _ = confirmation._load_model_result(inputs, 'large_seed0', e['out'])
    baseline, _ = confirmation._load_model_result(inputs, 'baseline_seed0', e['out'])
    np.testing.assert_array_equal(large['sequence_indices'], [4, 5, 6, 7])
    paired = result['groups'][0]['paired']['large_seed0_minus_baseline_seed0']['next_kl_bits']
    difference = large['next_kl_bits']-baseline['next_kl_bits']
    np.testing.assert_allclose(paired['mean'], difference.mean(0))
    np.testing.assert_allclose(paired['se'], difference.std(0, ddof=1)/2)
    expected_bayes = inputs['references']['bayes_hidden_next_kl_bits'][4:].mean(0)
    np.testing.assert_allclose(result['groups'][0]['predictors']['bayes_hidden']['next_kl_bits']['mean'], expected_bayes)
    # A rerun verifies and reuses all compact scores; it does not run inference.
    monkeypatch.setattr(confirmation, 'GPT', forbidden)
    again = confirmation.run(e['plan_path'], e['baseline_root'], e['scaling_root'], e['out'], device='cpu', batch_size=2)
    assert again['groups'] == result['groups']


def test_modified_final_weights_are_rejected_even_with_valid_metadata(confirmation_experiment):
    e = confirmation_experiment
    inputs = confirmation.load_inputs(e['plan_path'], e['baseline_root'], e['scaling_root'])
    path = inputs['checkpoints']['large_seed1']
    weights = torch.load(path, weights_only=False)
    weights['model']['head.weight'][0, 0] += .1
    torch.save(weights, path)
    with pytest.raises(ValueError, match='differs from the model evaluated in the pilot'):
        confirmation.evaluate_model(inputs, 'large_seed1', e['out'], batch_size=2)


def test_incomplete_references_or_nested_output_are_rejected(confirmation_experiment):
    e = confirmation_experiment
    with pytest.raises(ValueError, match='separate'):
        confirmation.run(e['plan_path'], e['baseline_root'], e['scaling_root'], e['scaling_root']/'confirmation')
    path = e['baseline_root']/'eval/references/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['complete'] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete or mismatched'):
        confirmation.load_inputs(e['plan_path'], e['baseline_root'], e['scaling_root'])


@pytest.mark.parametrize('change', ['pairing', 'metric'])
def test_score_replay_rejects_wrong_pairing_or_metrics(confirmation_experiment, change):
    e = confirmation_experiment
    inputs = confirmation.load_inputs(e['plan_path'], e['baseline_root'], e['scaling_root'])
    confirmation.evaluate_model(inputs, 'baseline_seed0', e['out'], batch_size=2)
    path = e['out']/'models/baseline_seed0.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if change == 'pairing':
        arrays['sequence_indices'][0] = 0
    else:
        arrays['next_kl_bits'][0, 0] += .1
    original.save_npz(path, arrays)
    receipt = json.loads(path.with_suffix('.json').read_text())
    receipt['data_sha256'] = original.file_hash(path)
    original.save_json(path.with_suffix('.json'), receipt)
    with pytest.raises(ValueError, match='pairing/times changed|metric replay failed'):
        confirmation._load_model_result(inputs, 'baseline_seed0', e['out'])


def test_summary_requires_all_six_fixed_models(confirmation_experiment):
    e = confirmation_experiment
    inputs = confirmation.load_inputs(e['plan_path'], e['baseline_root'], e['scaling_root'])
    confirmation.evaluate_model(inputs, 'baseline_seed0', e['out'], batch_size=2)
    with pytest.raises(ValueError, match='Missing confirmation scores'):
        confirmation.summarize(inputs, e['out'])
    assert not (e['out']/'summary.json').exists()
