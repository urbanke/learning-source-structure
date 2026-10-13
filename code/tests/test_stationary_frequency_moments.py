import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'transformer'))
from stationary_frequency_moments import frequency_type, stationary_joint_type
from shared_base_frequency_reference import stationary_iid_joint


def brute(word, q, nu):
    labels = list(dict.fromkeys(word))
    terms = []
    for ranks in itertools.permutations(range(len(q)), len(labels)):
        mapping = dict(zip(labels, ranks))
        terms.append(q[mapping[word[0]]] * np.prod([nu[mapping[x]] for x in word[1:]]))
    return np.mean(terms)


@pytest.mark.parametrize('word', [[0], [0,0], [0,1,0,1], [0,1,2,1,2,0,3],
                                 [0,1,1,1,2,2,3,3,3,3], [0]*17])
def test_independent_brute(word):
    rng = np.random.default_rng(17)
    q, nu = rng.dirichlet(np.ones(5), 2)
    value = stationary_joint_type(frequency_type(word, 5), q[None], nu[None])[0]
    assert value == pytest.approx(brute(word,q,nu), rel=3e-13, abs=1e-25)


def test_short_independent_signed_moments():
    rng = np.random.default_rng(65)
    q, nu = rng.dirichlet(np.ones(64), 14).reshape(2,7,64)
    for word in itertools.product(range(3), repeat=5):
        actual = stationary_joint_type(frequency_type(word),q,nu,block_size=3)
        assert np.allclose(actual,stationary_iid_joint(word,q,nu),rtol=3e-13,atol=1e-25)


def test_long_joint_child_normalization_and_tail_invariance():
    rng = np.random.default_rng(15)
    q, nu = rng.dirichlet(np.ones(7), 6).reshape(2,3,7)
    word = [0,1,2,1,0,1,3,3,1,0,4,5,4,1,6,5]
    parent = stationary_joint_type(frequency_type(word,7),q,nu)
    children = sum(stationary_joint_type(frequency_type(word+[x],7),q,nu) for x in range(7))
    assert np.allclose(parent,children,rtol=3e-13,atol=1e-30)
    assert frequency_type(word,7) == frequency_type(word[:1]+word[1:][::-1],7)


def test_uniform_long_joint():
    q = np.full((2,64),1/64)
    for word in [list(range(17)),[0]*17,[0,1,2]*5+[3,4]]:
        assert np.allclose(stationary_joint_type(frequency_type(word),q,q),64.**-17,rtol=3e-13,atol=0)


def test_reject_invalid_inputs():
    q = np.full((1,4),.25)
    with pytest.raises(ValueError): frequency_type([0]*18)
    with pytest.raises(ValueError): frequency_type([.5])
    with pytest.raises(ValueError): stationary_joint_type((0,(1,)),q*2,q)
