"""Independent finite-alphabet checks of the three-law short references."""
from collections import Counter
import itertools
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'transformer'))
import shared_base_frequency_reference as frequency


def original_joint(tokens):
    """A two-source nonsymmetric Markov mixture independent of the implementation."""
    initial=np.array([[.65,.25,.10],[.15,.35,.50]])
    rows=np.array([[[.80,.15,.05],[.15,.70,.15],[.35,.15,.50]],
                   [[.10,.40,.50],[.60,.10,.30],[.20,.30,.50]]])
    if not tokens:return 1.
    result=0.
    for mixture,q,transition in zip((.4,.6),initial,rows):
        value=mixture*q[tokens[0]]
        for a,b in zip(tokens[:-1],tokens[1:]):value*=transition[a,b]
        result+=value
    return result


def brute_restricted_prediction(tokens):
    """Enumerate all histories and condition on the observed statistic directly."""
    histories=[h for h in itertools.product(range(3),repeat=len(tokens))
               if h[0]==tokens[0] and Counter(h[1:])==Counter(tokens[1:])]
    denominator=sum(original_joint(h) for h in histories)
    return np.array([sum(original_joint(h+(y,)) for h in histories)/denominator for y in range(3)])


def test_count_only_evidence_weighting_matches_statistic_conditioning_for_every_short_word():
    for n in range(1,5):
        for tokens in itertools.product(range(3),repeat=n):
            actual=frequency.count_only_original_probs(tokens,lambda x:math.log(original_joint(x)),d=3)
            np.testing.assert_allclose(actual,brute_restricted_prediction(tokens),atol=2e-15,rtol=2e-14)
            assert abs(actual.sum()-1)<2e-15


def test_tail_permutation_keeps_first_distinguished_and_successor_outside_shuffle():
    joint=lambda x:math.log(original_joint(x))
    p=frequency.count_only_original_probs([0,1,1,2],joint,d=3)
    q=frequency.count_only_original_probs([0,2,1,1],joint,d=3)
    np.testing.assert_array_equal(p,q)
    moved=frequency.count_only_original_probs([1,0,1,2],joint,d=3)
    assert np.max(abs(p-moved))>.01
    tokens=(0,1,2)
    histories=[tokens,(0,2,1)]
    average_of_conditionals=np.mean([[original_joint(h+(y,))/original_joint(h) for y in range(3)]
                                    for h in histories],axis=0)
    correct=frequency.count_only_original_probs(tokens,joint,d=3)
    assert np.max(abs(correct-average_of_conditionals))>.001
    # Shuffling all augmented tail tokens would incorrectly include the future
    # token in the observed statistic. Even renormalizing that law cannot fix it.
    wrong=np.array([np.mean([original_joint((0,)+tail) for tail in set(itertools.permutations((1,2,y)))])
                    for y in range(3)])
    wrong/=wrong.sum()
    assert np.max(abs(correct-wrong))>.01


def test_first_unobserved_and_first_observed_boundary_are_distinct():
    log_joint=lambda x:math.log(original_joint(x))
    p=frequency.count_only_original_probs([],log_joint,d=3)
    np.testing.assert_allclose(p,[original_joint((y,)) for y in range(3)],atol=1e-15)
    observed=frequency.count_only_original_probs([0],log_joint,d=3)
    np.testing.assert_allclose(observed,[original_joint((0,y))/original_joint((0,)) for y in range(3)],atol=1e-15)
    assert np.max(abs(p-observed))>.01


def brute_rank_average(q,nu,powers):
    terms=[]
    for assignment in itertools.permutations(range(len(q)),len(powers)):
        terms.append(math.prod(q[r]**a*nu[r]**b for r,(a,b) in zip(assignment,powers)))
    return math.fsum(terms)/len(terms)


@pytest.mark.parametrize('d',[2,3,5])
def test_mobius_injective_average_equals_brute_rank_enumeration(d):
    rng=np.random.default_rng(916+d)
    q=rng.dirichlet(np.ones(d),size=3);nu=rng.dirichlet(np.ones(d),size=3)
    for n in range(1,6):
        # All equality types realized by finite-alphabet words, including repeats.
        patterns=set()
        for tokens in itertools.product(range(min(d,n)),repeat=n):
            labels=list(dict.fromkeys(tokens))
            patterns.add(tuple((int(label==tokens[0]),tokens[1:].count(label)) for label in labels))
        for powers in patterns:
            actual=frequency.injective_rank_average(q,nu,powers)
            expected=[brute_rank_average(a,b,powers) for a,b in zip(q,nu)]
            np.testing.assert_allclose(actual,expected,rtol=4e-13,atol=2e-16)


def test_two_label_inclusion_exclusion_uses_falling_factorial():
    q=np.array([.65,.25,.10]);nu=np.array([.2,.3,.5])
    expected=(q.sum()*nu.sum()-q@nu)/(3*2)
    actual=frequency.injective_rank_average(q,nu,[(1,0),(0,1)])
    assert abs(actual-expected)<1e-16
    assert abs(actual-(1-q@nu)/9)>.01


def test_stationary_joint_first_q_then_nu_matches_brute_and_marginalizes():
    q=np.array([.7,.2,.1]);nu=np.array([.05,.35,.6])
    assert frequency.stationary_iid_joint([],q,nu)==1.
    for n in range(5):
        for prefix in itertools.product(range(3),repeat=n):
            joint=frequency.stationary_iid_joint(prefix,q,nu)
            children=sum(frequency.stationary_iid_joint(prefix+(y,),q,nu) for y in range(3))
            np.testing.assert_allclose(joint,children,atol=3e-16,rtol=1e-13)
    repeated=frequency.stationary_iid_joint([0,0],q,nu)
    assert abs(repeated-q@nu/3)<1e-16
    assert abs(repeated-(nu@nu)/3)>.01


def test_stationary_joint_tail_and_label_symmetry_preserve_distinguished_first():
    q=np.array([.7,.2,.1]);nu=np.array([.05,.35,.6])
    x=frequency.stationary_iid_joint([0,1,0,2,2],q,nu)
    assert x==frequency.stationary_iid_joint([0,2,2,1,0],q,nu)
    assert x==frequency.stationary_iid_joint([2,0,2,1,1],q,nu)
    assert abs(x-frequency.stationary_iid_joint([1,0,0,2,2],q,nu))>1e-5


def test_bank_average_must_precede_predictive_division():
    q=np.array([[.8,.15,.05],[.2,.3,.5]])
    nu=np.array([[.9,.05,.05],[.1,.3,.6]])
    weights=np.array([.3,.7]);prefix=(0,0,1)
    joint=lambda x:float(weights@frequency.stationary_iid_joint(x,q,nu))
    prediction=frequency.count_only_original_probs(prefix,lambda x:math.log(joint(x)),d=3)
    expected=np.array([joint(prefix+(y,))/joint(prefix) for y in range(3)])
    np.testing.assert_allclose(prediction,expected,atol=3e-15)
    wrong=sum(w*np.array([frequency.stationary_iid_joint(prefix+(y,),a,b)
                         /frequency.stationary_iid_joint(prefix,a,b) for y in range(3)])
              for w,a,b in zip(weights,q,nu))
    assert np.max(abs(prediction-wrong))>.01


def test_nearly_degenerate_rank_laws_are_nonnegative_without_clipping():
    q=np.array([[1.,0,0,0,0],[1-1e-10,1e-10,0,0,0]])
    nu=np.array([[1.,0,0,0,0],[1-4e-8,1e-8,1e-8,1e-8,1e-8]])
    powers=[(1,0)]+[(0,1)]*4
    actual=frequency.injective_rank_average(q,nu,powers)
    expected=np.array([brute_rank_average(a,b,powers) for a,b in zip(q,nu)])
    assert actual[0]==0 and actual[1]>0
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=0)


def test_partition_inventory_is_complete_and_unique():
    for n,count in enumerate([1,1,2,5,15,52]):
        partitions=frequency.set_partitions(n)
        canonical={tuple(sorted(tuple(sorted(block)) for block in p)) for p in partitions}
        assert len(partitions)==len(canonical)==count
        assert all(sorted(itertools.chain.from_iterable(p))==list(range(n)) for p in partitions)


@pytest.mark.parametrize('tokens',[[-1],[3],[0.,1.],[[0,1]],[0]*6])
def test_stationary_rejects_invalid_or_unavailable_words(tokens):
    with pytest.raises(ValueError):frequency.stationary_iid_joint(tokens,[.2,.3,.5],[.1,.4,.5])


def test_no_catalog_lookahead_or_hidden_renormalization():
    with pytest.raises(ValueError,match='At most 4'):
        frequency.count_only_original_probs([0]*5,lambda x:0,d=3)
    with pytest.raises(ValueError,match='normalization'):
        frequency.count_only_original_probs([0],lambda x:0.,d=3)
    with pytest.raises(ValueError,match='first token'):
        frequency.injective_rank_average([.2,.8],[.3,.7],[(1,0),(1,0)])
    assert set(frequency.LAWS)=={'count_only_original','stationary_iid','finite_shuffle'}
