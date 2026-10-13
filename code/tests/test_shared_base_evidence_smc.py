"""Independent normalization and finite-state checks for experimental SMC."""
from itertools import permutations, product
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base_evidence_smc import (TemperedTransitionTarget, estimate_log_evidence,
                                      systematic_resample)
from shared_base_mcmc import PrefixPosterior, SharedBaseMCMC
from shared_base_efficiency import SharedBaseEfficiencyRowMCMC
from test_shared_base_mcmc import ScriptedRNG


def sampler(d=3, depths=(0,), alpha=(.6, 1.3), weights=(.3, .7)):
    return SharedBaseMCMC(d, RankedLSARowEvidence(d, alpha, depths,
                            u_lo=-36, u_hi=36, step=1/32), weights)


def independent_evidence(mc, tokens):
    """Enumerate ranks; depth-one moments use independent adaptive integration.

    For independent exponential E_j, a depth-one row is q_j E_j / sum qE.
    Its moment is the one-dimensional Laplace identity integrated directly
    on [0,infinity], independent of the production log-grid row kernel.
    """
    d = mc.d
    counts = np.zeros((d, d), dtype=int)
    np.add.at(counts, (tokens[:-1], tokens[1:]), 1)
    total = 0.
    for ai, aw in enumerate(mc.alpha_weights):
        for ranks in permutations(range(d)):
            q = mc.kernel.base_laws[ai, list(ranks)]
            depth_likelihood = []
            for depth in mc.depths:
                rows = []
                for row in counts:
                    k = int(row.sum())
                    if not k:
                        value = 1.
                    elif depth == 0:
                        value = float(np.prod(q**row))
                    elif depth == 1:
                        constant = float(np.prod(q**row) * math.prod(math.factorial(int(c)) for c in row)
                                         / math.gamma(k))
                        value, error = quad(lambda s: constant * s**(k-1)
                                            / np.prod((1+s*q)**(row+1)),
                                            0, np.inf, epsabs=1e-13, epsrel=1e-11)
                        assert error < 1e-10
                    else:
                        raise ValueError("independent test supports depths zero and one")
                    rows.append(value)
                depth_likelihood.append(math.prod(rows))
            total += aw * q[tokens[0]] * (mc.depth_prior @ depth_likelihood) / math.factorial(d)
    return float(total)


@pytest.mark.parametrize("beta", [0., .2, 1.])
def test_tempered_moves_obey_detailed_balance_for_independent_iid_target(beta):
    mc = sampler()
    tokens = [2, 0, 2, 1, 2]
    target = TemperedTransitionTarget(PrefixPosterior(mc, tokens), beta)
    states = [(a, r) for a in range(2) for r in permutations(range(3))]
    index = {state: i for i, state in enumerate(states)}
    frequency = np.bincount(tokens[1:], minlength=3)
    logpi = np.array([np.log(mc.alpha_weights[a]) + mc.kernel.log_base_laws[a, r[tokens[0]]]
                      + beta * (frequency @ mc.kernel.log_base_laws[a, list(r)])
                      for a, r in states])
    pi = np.exp(logpi-logsumexp(logpi))
    transition = np.zeros((len(states), len(states)))
    for i, (ai, ranks) in enumerate(states):
        state = target.evaluate(ai, ranks)
        assert state.log_target == pytest.approx(logpi[i], abs=1e-13)
        proposals = []
        for left in target.seen:
            for right in range(3):
                changed = list(ranks)
                changed[left], changed[right] = changed[right], changed[left]
                proposals.append((0, int(left), right, (ai, tuple(changed)), .4/len(target.seen)/3, 0.))
        for left in range(3):
            for right in range(3):
                a, b = ranks.index(left), ranks.index(right)
                changed = list(ranks)
                changed[a], changed[b] = changed[b], changed[a]
                proposals.append((1, left, right, (ai, tuple(changed)),
                                  .4*mc.rank_proposal_weights[left]/3, 0.))
        for other in range(2):
            proposals.append((2, other, None, (other, ranks), .1*mc.alpha_weights[other],
                              math.log(mc.alpha_weights[ai]/mc.alpha_weights[other])))
        for direction in (-1, 1):
            other = ai + direction
            other = other if 0 <= other < 2 else ai
            proposals.append((3, direction, None, (other, ranks), .05, 0.))
        for move, choice, integer, destination, probability, correction in proposals:
            j = index[destination]
            acceptance = min(1., math.exp(logpi[j]-logpi[i]+correction))
            for u in (acceptance/2, (1+acceptance)/2):
                actual = mc._step(target, state, ScriptedRNG(move, choice, integer, u),
                                  np.zeros(4, dtype=int), np.zeros(4, dtype=int))
                assert (actual.alpha_index, tuple(actual.ranks)) == (destination if u < acceptance else (ai, ranks))
            transition[i, j] += probability*acceptance
            transition[i, i] += probability*(1-acceptance)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-15)
    flow = pi[:, None]*transition
    np.testing.assert_allclose(flow, flow.T, atol=2e-16)


@pytest.mark.parametrize("depths", [(0,), (1,), (0, 1)])
def test_uniform_base_anchors_are_exact_without_monte_carlo_variation(depths):
    mc = sampler(d=4, depths=depths, alpha=(0.,), weights=(1.,))
    tokens = np.array([0, 1, 1, 2, 0, 1, 3, 3])
    counts = np.zeros((4, 4), dtype=int)
    np.add.at(counts, (tokens[:-1], tokens[1:]), 1)
    l0 = -(len(tokens)-1)*math.log(4)
    l1 = float(sum(gammaln(4)-gammaln(4+row.sum())+gammaln(row+1).sum() for row in counts))
    exact = -math.log(4)+float(logsumexp([l0 if depth == 0 else l1 for depth in depths]))-math.log(len(depths))
    for seed in [3, 7]:
        result = estimate_log_evidence(mc, tokens, particles=8, temperatures=7, moves=1, seed=seed)
        assert result["log_evidence"] == pytest.approx(exact, abs=1e-12)
        np.testing.assert_allclose(result["effective_sample_sizes"], 8, atol=1e-12)
    assert estimate_log_evidence(mc, [])['log_evidence'] == 0
    assert estimate_log_evidence(mc, [3])['log_evidence'] == -math.log(4)


@pytest.mark.parametrize("d,depths,resample_fraction", [
    (3, (0,), .01), (3, (0,), 1.), (3, (0, 1), .75), (4, (0, 1), .75)])
def test_independent_smc_replicas_match_enumerated_normalized_evidence(d, depths, resample_fraction):
    mc = sampler(d=d, depths=depths)
    tokens = np.array([0, 0, 1, 0, 2, 1])
    exact = independent_evidence(mc, tokens)
    estimates = np.array([math.exp(estimate_log_evidence(mc, tokens, particles=32,
                          temperatures=12, moves=2, seed=seed,
                          resample_fraction=resample_fraction)["log_evidence"])
                          for seed in range(20)])
    se = estimates.std(ddof=1)/math.sqrt(len(estimates))
    assert abs(estimates.mean()-exact) < 5*se
    assert abs(estimates.mean()/exact-1) < .06


def test_retargeting_and_swapping_do_not_mutate_shared_particle_states():
    mc = sampler(depths=(0, 1))
    target = TemperedTransitionTarget(PrefixPosterior(mc, [0, 0, 1, 2]), .2)
    state = target.evaluate(0, [1, 0, 2])
    rows, ranks, old_target = state.rows.copy(), state.ranks.copy(), state.log_target
    target.beta = .8
    changed = target.swap(state, 0, 2)
    target.retarget(state)
    np.testing.assert_array_equal(state.rows, rows)
    np.testing.assert_array_equal(state.ranks, ranks)
    assert state.log_target == old_target and changed is not state


def test_systematic_resampling_has_correct_mean_offspring_counts():
    rng = np.random.default_rng(6013)
    weights = np.array([.01, .06, .18, .25, .5])
    counts = sum((np.bincount(systematic_resample(weights, rng), minlength=5)
                  for _ in range(10000)), start=np.zeros(5)) / 10000
    np.testing.assert_allclose(counts, len(weights)*weights, atol=.01, rtol=0)


def test_subclass_with_unaudited_moves_is_rejected():
    class ModifiedSampler(SharedBaseMCMC):
        pass
    mc = sampler()
    changed = ModifiedSampler(mc.d, mc.kernel, mc.alpha_weights)
    with pytest.raises(TypeError):
        estimate_log_evidence(changed, [0, 1])


def row_sampler(d=3, depths=(0,), steps=3):
    return SharedBaseEfficiencyRowMCMC(d,
        RankedLSARowEvidence(d, [.6, 1.3], depths, u_lo=-36, u_hi=36, step=1/32),
        [.3, .7], surrogate_steps=steps, concentration=1.7)


def independent_auxiliary_logs(mc, tokens, states):
    """Full normalized Dirichlet row evidences, including omitted row constants."""
    counts = np.zeros((mc.d, mc.d), dtype=int)
    np.add.at(counts, (tokens[:-1], tokens[1:]), 1)
    logs = []
    for ai, ranks in states:
        q = mc.kernel.base_laws[ai, list(ranks)]
        cq = mc.concentration*q
        rows = sum(gammaln(mc.concentration)-gammaln(mc.concentration+row.sum())
                   + (gammaln(cq+row)-gammaln(cq)).sum() for row in counts)
        logs.append(math.log(mc.alpha_weights[ai])+math.log(q[tokens[0]])+rows)
    return np.array(logs)


class AuxiliaryRNG:
    """Deterministically select a path, checking its actual Gibbs probabilities."""
    def __init__(self, *, alpha=None, alpha_probs=None, matching=None, uniforms=None):
        self.alpha, self.alpha_probs = alpha, alpha_probs
        self.matching, self.uniforms = matching, uniforms

    def random(self, size=None):
        if size is None:
            return .1 if self.alpha is not None else .5
        assert len(self.uniforms) == size
        return np.array(self.uniforms)

    def choice(self, count, p):
        np.testing.assert_allclose(p, self.alpha_probs, rtol=0, atol=1e-13)
        assert count == len(p)
        return self.alpha

    def permutation(self, d):
        assert len(self.matching) == d
        return np.array(self.matching)


def independent_auxiliary_kernel(mc, tokens):
    """Enumerate every auxiliary outcome and check the executed move for each."""
    states = [(a, ranks) for a in range(2) for ranks in permutations(range(mc.d))]
    index = {state: i for i, state in enumerate(states)}
    logs = independent_auxiliary_logs(mc, tokens, states)
    target = TemperedTransitionTarget(PrefixPosterior(mc, tokens), .2)
    table = mc.auxiliary_table(target)
    actual_logs = np.array([mc.surrogate_log_target(table, a, np.array(ranks)) for a, ranks in states])
    # The implementation legitimately omits row constants independent of state.
    np.testing.assert_allclose(actual_logs-actual_logs[0], logs-logs[0], atol=1e-13, rtol=0)
    transition = np.zeros((len(states), len(states)))
    for i, (ai, ranks) in enumerate(states):
        destinations = [index[(a, ranks)] for a in range(2)]
        p_alpha = np.exp(logs[destinations]-logsumexp(logs[destinations]))
        for new_alpha in range(2):
            copy = np.array(ranks)
            actual = mc._surrogate_step(table, ai, copy,
                AuxiliaryRNG(alpha=new_alpha, alpha_probs=p_alpha))
            assert actual == new_alpha and tuple(copy) == ranks
            transition[i, index[(new_alpha, ranks)]] += .2*p_alpha[new_alpha]
        for matching in permutations(range(mc.d)):
            pairs = list(zip(matching[0:2*(mc.d//2):2], matching[1:2*(mc.d//2):2]))
            probabilities = []
            for left, right in pairs:
                changed = list(ranks)
                changed[left], changed[right] = changed[right], changed[left]
                delta = logs[index[(ai, tuple(changed))]]-logs[i]
                probabilities.append(1/(1+math.exp(-delta)))
            for choices in product((False, True), repeat=len(pairs)):
                destination = list(ranks)
                probability = .8/math.factorial(mc.d)
                uniforms = []
                for (left, right), chosen, p in zip(pairs, choices, probabilities):
                    probability *= p if chosen else 1-p
                    uniforms.append(p/2 if chosen else (1+p)/2)
                    if chosen:
                        destination[left], destination[right] = destination[right], destination[left]
                copy = np.array(ranks)
                actual = mc._surrogate_step(table, ai, copy,
                    AuxiliaryRNG(matching=matching, uniforms=uniforms))
                assert actual == ai and tuple(copy) == tuple(destination)
                transition[i, index[(ai, tuple(destination))]] += probability
    return states, logs, transition


@pytest.mark.parametrize("d", [3, 4])
def test_auxiliary_block_kernel_and_corrected_tempered_target_have_exact_detailed_balance(d):
    mc = row_sampler(d=d)
    tokens = [0, 0, 1, 2, 0, 1]
    states, auxiliary_logs, one_step = independent_auxiliary_kernel(mc, tokens)
    auxiliary = np.exp(auxiliary_logs-logsumexp(auxiliary_logs))
    np.testing.assert_allclose(one_step.sum(axis=1), 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(auxiliary[:, None]*one_step,
                               (auxiliary[:, None]*one_step).T, atol=1e-14, rtol=0)
    block = np.linalg.matrix_power(one_step, mc.surrogate_steps)
    frequencies = np.bincount(tokens[1:], minlength=d)
    for beta in (0., .2, 1.):
        # Independent normalized iid target, conditional on its first token.
        log_target = np.array([math.log(mc.alpha_weights[a])
            + mc.kernel.log_base_laws[a, ranks[tokens[0]]]
            + beta*(frequencies @ mc.kernel.log_base_laws[a, list(ranks)])
            for a, ranks in states])
        target = np.exp(log_target-logsumexp(log_target))
        log_acceptance = (log_target[None, :]-log_target[:, None]
                          + auxiliary_logs[:, None]-auxiliary_logs[None, :])
        corrected = block*np.exp(np.minimum(0., log_acceptance))
        corrected[np.diag_indices_from(corrected)] += 1-corrected.sum(axis=1)
        np.testing.assert_allclose(corrected.sum(axis=1), 1, atol=1e-14, rtol=0)
        np.testing.assert_allclose(target[:, None]*corrected,
                                   (target[:, None]*corrected).T, atol=1e-14, rtol=0)
        np.testing.assert_allclose(target @ corrected, target, atol=1e-14, rtol=0)


class OuterBlockRNG(AuxiliaryRNG):
    def __init__(self, acceptance_u, **kwargs):
        super().__init__(**kwargs)
        self.acceptance_u = acceptance_u
        self.outer_choice = True
        self.branch_chosen = False

    def choice(self, count, p):
        if self.outer_choice:
            self.outer_choice = False
            np.testing.assert_array_equal(p, [.5, .4, .05, .05])
            return 0
        return super().choice(count, p)

    def random(self, size=None):
        if size is not None:
            return super().random(size)
        if not self.branch_chosen:
            self.branch_chosen = True
            return super().random()
        return self.acceptance_u


@pytest.mark.parametrize("beta", [0., .2, 1.])
def test_executed_block_hastings_correction_uses_tempered_target_and_untempered_auxiliary(beta):
    mc = row_sampler(steps=1)
    tokens = [0, 0, 1, 2, 0, 1]
    target = TemperedTransitionTarget(PrefixPosterior(mc, tokens), beta)
    states = [(a, ranks) for a in range(2) for ranks in permutations(range(3))]
    index = {state: i for i, state in enumerate(states)}
    aux = independent_auxiliary_logs(mc, tokens, states)
    frequency = np.bincount(tokens[1:], minlength=3)
    pi = np.array([math.log(mc.alpha_weights[a])+mc.kernel.log_base_laws[a, r[tokens[0]]]
                   + beta*(frequency @ mc.kernel.log_base_laws[a, list(r)]) for a, r in states])
    for i, (ai, ranks) in enumerate(states):
        state = target.evaluate(ai, ranks)
        for kind in ("alpha", "rank"):
            if kind == "alpha":
                destination = (1-ai, ranks)
                ids = [index[(a, ranks)] for a in range(2)]
                settings = dict(alpha=1-ai, alpha_probs=np.exp(aux[ids]-logsumexp(aux[ids])))
            else:
                destination = (ai, (ranks[1], ranks[0], ranks[2]))
                j = index[destination]
                swap_probability = 1/(1+math.exp(-(aux[j]-aux[i])))
                settings = dict(matching=[0, 1, 2], uniforms=[swap_probability/2])
            j = index[destination]
            wanted_correction = aux[i]-aux[j]
            proposal = mc.surrogate_proposal(target, state, AuxiliaryRNG(**settings))
            assert (proposal[0], tuple(proposal[1])) == destination
            assert proposal[2] == pytest.approx(wanted_correction, abs=1e-13)
            acceptance = min(1., math.exp(pi[j]-pi[i]+wanted_correction))
            for u in (acceptance/2, (1+acceptance)/2):
                attempts, accepts = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
                actual = mc._step(target, state, OuterBlockRNG(u, **settings), attempts, accepts)
                assert (actual.alpha_index, tuple(actual.ranks)) == (destination if u < acceptance else (ai, ranks))
                assert attempts.tolist() == [1, 0, 0, 0]
                assert accepts.tolist() == [int(u < acceptance), 0, 0, 0]
            np.testing.assert_array_equal(state.ranks, ranks)


def test_row_block_smc_matches_independent_evidence_and_reports_bounded_diagnostics():
    mc = row_sampler(depths=(0, 1), steps=3)
    tokens = np.array([0, 0, 1, 0, 2, 1])
    exact = independent_evidence(mc, tokens)
    results = [estimate_log_evidence(mc, tokens, particles=32, temperatures=12,
                                    moves=2, seed=seed) for seed in range(20)]
    estimates = np.exp([r["log_evidence"] for r in results])
    se = estimates.std(ddof=1)/math.sqrt(len(estimates))
    assert abs(estimates.mean()-exact) < 5*se
    assert abs(estimates.mean()/exact-1) < .06
    for result in results:
        assert result["sampler_class"] == "SharedBaseEfficiencyRowMCMC"
        assert result["move_names"][0] == "dirichlet_row_surrogate_block"
        assert len(result["move_attempts"]) == len(result["move_accepts"]) == 4
        assert sum(result["move_attempts"]) == 32*12*2
        assert result["proposal_spec"]["surrogate_steps"] == 3
        ancestry = result["initial_particle_ancestry_counts"]
        assert len(ancestry) == 13 and ancestry[0] == 32
        assert all(1 <= after <= before for before, after in zip(ancestry[:-1], ancestry[1:]))
        for i, resampled in enumerate(result["resampled"]):
            if not resampled:
                assert ancestry[i+1] == ancestry[i]
        ranks = result["leading_rank_diagnostics"]
        assert ranks["tracked_ranks"] == [0, 1, 2]
        assert np.asarray(ranks["distinct_labels_by_temperature"]).shape == (13, 3)
        assert len(ranks["distinct_assignments_by_temperature"]) == 13
        assert 0 < sum(ranks["label_changes_during_mh"]) <= 3*sum(result["move_accepts"])
        probabilities = np.asarray(ranks["final_weighted_label_probabilities"])
        assert probabilities.shape == (3, 3) and np.all(probabilities >= 0)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14)


def test_original_sampler_keeps_identical_seeded_estimate_and_move_counts():
    mc = sampler(depths=(0, 1))
    result = estimate_log_evidence(mc, [0, 0, 1, 0, 2, 1], particles=16,
                                   temperatures=8, moves=2, seed=401)
    assert result["log_evidence"] == -7.246210909268235
    assert result["move_attempts"] == [96, 102, 29, 29]
    assert result["move_accepts"] == [77, 82, 27, 25]


def test_unaudited_row_sampler_subclass_is_rejected():
    class ModifiedRowSampler(SharedBaseEfficiencyRowMCMC):
        pass
    mc = row_sampler()
    changed = ModifiedRowSampler(mc.d, mc.kernel, mc.alpha_weights)
    with pytest.raises(TypeError):
        estimate_log_evidence(changed, [0, 1])
