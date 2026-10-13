"""Alignment and marginal-probability checks for the early payoff consumer."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/scripts'))
from shared_base_shuffle_early_reference import score_sequence, statistics, validate_cohort_arrays, array_sha
from shared_base_short_evidence import SharedBaseShortEvidence


class LaplaceTable:
    def __init__(self):
        self.counts = np.zeros(64)

    def predictive(self):
        return (self.counts+1)/(self.counts.sum()+64)

    def observe(self, token):
        loss = -np.log2(self.predictive()[token])
        self.counts[token] += 1
        return loss


class LaplaceKernel:
    def new_table(self):
        return LaplaceTable()


def test_first_token_updates_unigram_and_oracle_uses_current_context():
    tokens = np.array([17, 2, 17, 3, 2])
    matrix = np.ones((64,64))
    matrix[np.arange(64), np.arange(64)] = 65
    matrix /= matrix.sum(1, keepdims=True)
    result = score_sequence(tokens, matrix, SharedBaseShortEvidence(), LaplaceKernel())
    np.testing.assert_array_equal(result['oracle_probs'], matrix[tokens[:4]])
    assert result['unigram_probs'][0,17] == 2/65
    assert result['unigram_probs'][0,2] == 1/65
    assert result['unigram_probs'][1,17] == 2/66
    assert result['unigram_probs'][1,2] == 2/66
    np.testing.assert_allclose(result['unigram_expected_loss_bits'],
        -np.sum(matrix[tokens[:4]] * np.log2(result['unigram_probs']), axis=1), rtol=0, atol=0)
    np.testing.assert_allclose(result['cumulative_gap_bits'],
        np.cumsum(result['unigram_expected_loss_bits']-result['bayes_expected_loss_bits']))


def test_catalog_lookup_matches_absolute_evidence_for_actual_labels():
    catalog = SharedBaseShortEvidence()
    for prefix in ([17], [17,2], [17,2,17], [17,2,17,3]):
        predictions = catalog.next_probs(prefix)
        direct = np.array([np.exp(catalog.log_joint(prefix+[a])-catalog.log_joint(prefix))
                           for a in range(64)])
        np.testing.assert_allclose(predictions, direct, atol=2e-15, rtol=0)
        assert abs(predictions.sum()-1) < 1e-12


def test_standard_error_uses_paired_sequence_values_before_cumulative_average():
    gaps = np.array([[1.,-1.], [2.,-2.], [3.,-3.]])
    values = statistics(np.cumsum(gaps, axis=1))
    np.testing.assert_allclose(values['mean'], [2.,0.])
    np.testing.assert_allclose(values['se'], [1/np.sqrt(3),0.])


def test_changed_frozen_history_or_true_row_is_rejected():
    data = dict(tokens=np.zeros((256,1025),dtype=int), base=np.full((256,64),1/64),
                transition=np.full((256,64,64),1/64))
    condition = dict(arrays_sha256={k:array_sha(v,'<i8' if k == 'tokens' else '<f8') for k,v in data.items()})
    validate_cohort_arrays(data, condition)
    data['tokens'][0,0] = 1
    with pytest.raises(ValueError, match='checksum mismatch: tokens'):
        validate_cohort_arrays(data, condition)
    data['tokens'][0,0] = 0
    data['transition'][0,0,:2] += [.001,-.001]
    with pytest.raises(ValueError, match='checksum mismatch: transition'):
        validate_cohort_arrays(data, condition)
