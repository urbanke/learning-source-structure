"""Cross-check symmetry reduction against full finite-rank enumeration."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/scripts'));sys.path.insert(0,str(ROOT/'code/transformer'))
from conditional_shared_base_first_step import prediction
from shared_base import SharedBaseLSA
from weighted_lsa import WeightedLSARowEvidence


def test_first_step_matches_full_enumerated_joint_ratio():
    reduced=prediction(3,8,depths=[0,1])
    exact=SharedBaseLSA(3,WeightedLSARowEvidence([0,1]),alpha_order=8)
    p=np.asarray([exact.joint([0,j])/exact.joint([0]) for j in range(3)])
    assert np.allclose(reduced['probs'],p,rtol=0,atol=2e-13)
    for first in (1,2):
        expected=np.asarray([exact.joint([first,j])/exact.joint([first]) for j in range(3)])
        permuted=np.full(3,reduced['unseen_probability']);permuted[first]=reduced['seen_probability']
        assert np.allclose(permuted,expected,rtol=0,atol=2e-13)
