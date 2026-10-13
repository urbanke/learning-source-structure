"""Resumption preserves transitions and binds the accelerated proposal."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transformer'))
from ranked_lsa import RankedLSARowEvidence
from shared_base_surrogate import SharedBaseSurrogateMCMC
from surrogate_chain_runner import SurrogateChainRun
from stable_chain_statistics import stable_predictive_moments
from eval_shared_base_surrogate import parser


TOKENS = np.array([0, 2, 0, 1, 0, 2, 0, 0, 2])


def sampler(steps=4):
    return SharedBaseSurrogateMCMC(3, RankedLSARowEvidence(3, [.5, 1.5], [0, 1]),
                                   [.3, .7], surrogate_steps=steps)


def test_resumption_is_bit_exact_and_summaries_agree(tmp_path):
    config = dict(chains=4, burn=19, thin=2, seed=20260925)
    whole = SurrogateChainRun(sampler(), TOKENS, **config)
    expected = whole.extend(48, progress=lambda event: None)
    split = SurrogateChainRun(sampler(), TOKENS, **config)
    checkpoint = tmp_path / 'run.npz'

    class Interrupt(Exception):
        pass

    def stop(event):
        if event['phase'] == 'draws':
            split.save(checkpoint)
            raise Interrupt()

    with pytest.raises(Interrupt):
        split.extend(48, block_draws=5, progress=stop)
    resumed = SurrogateChainRun.load(sampler(), TOKENS, checkpoint)
    actual = resumed.extend(48, block_draws=7, progress=lambda event: None)
    for name in whole.samples:
        np.testing.assert_array_equal(resumed.samples[name], whole.samples[name])
    assert actual['diagnostics'] == expected['diagnostics']
    moments = stable_predictive_moments(resumed.samples['predictive_probs'])
    np.testing.assert_allclose(moments['mean'], actual['predictive_probs'], atol=2e-14, rtol=0)
    np.testing.assert_allclose(moments['mcse_probs'], actual['mcse_probs'], atol=1e-15, rtol=1e-11)


def test_checkpoint_rejects_changed_block_length(tmp_path):
    run = SurrogateChainRun(sampler(), TOKENS, chains=2, burn=0, thin=1, seed=4)
    checkpoint = tmp_path / 'run.npz'
    run.save(checkpoint)
    with pytest.raises(ValueError, match='proposal descriptor'):
        SurrogateChainRun.load(sampler(8), TOKENS, checkpoint)


def test_driver_has_finer_quadrature_and_unique_prefix_paths():
    args = parser().parse_args(['--index', '3'])
    assert args.alpha_order == 64 and args.surrogate_steps == 64
    assert args.out != args.input
