"""CPU-executable audit of the experimental batched float64 sampler.

These test algorithm composition and actual row targets. Passing does not
validate CUDA execution, long contexts, or real-source Monte Carlo precision.
"""
from itertools import permutations
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.special import logsumexp

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from shared_base_evidence_gpu import (MomentTables, ShortTarget, checked_device,
                                      estimate_log_evidence, row_descriptors, systematic_indices)
from shared_base_mcmc import SharedBaseMCMC, PrefixPosterior
from ranked_lsa import RankedLSARowEvidence
from test_shared_base_evidence_smc import independent_evidence


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def make_sampler(d=3, alpha=(.6, 1.3), weights=(.3, .7), depths=(0, 1), depth_weights=(.2, .8)):
    kernel = RankedLSARowEvidence(d, alpha, depths, u_lo=-36, u_hi=36, step=1/32)
    return SharedBaseMCMC(d, kernel, weights, depth_weights)


@pytest.fixture(scope="module")
def small():
    mc = make_sampler()
    return mc, MomentTables.from_sampler(mc)


def test_actual_tables_match_original_target_all_small_states(small):
    mc, tables = small
    states = [(a, r) for a in range(2) for r in permutations(range(3))]
    ai = torch.tensor([a for a, r in states])
    ranks = torch.tensor([r for a, r in states])
    for tokens in ([0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 2, 0],
                   [0, 0, 1, 0, 1], [2, 1, 0, 2, 1]):
        target = ShortTarget(tables, tokens)
        prior, likelihood = target.terms(ai, ranks)
        original = PrefixPosterior(mc, tokens)
        expected = [original.evaluate(a, np.array(r)).log_target for a, r in states]
        np.testing.assert_allclose((prior+likelihood).numpy(), expected, rtol=0, atol=1e-12)
        # Independent adaptive depth-one integration, summed over all ranks.
        evidence = np.exp((prior+likelihood).numpy()).sum()/math.factorial(3)
        assert evidence == pytest.approx(independent_evidence(mc, np.array(tokens)), rel=2e-11)
        assert prior.dtype == likelihood.dtype == torch.float64


@pytest.mark.parametrize("beta", [0., .2, 1.])
@pytest.mark.parametrize("tokens", [[0, 1, 0, 1, 0], [0, 1, 0, 2, 0]])
def test_four_actual_proposals_and_mh_obey_detailed_balance(small, beta, tokens):
    mc, tables = small
    target = ShortTarget(tables, tokens)
    states = [(a, r) for a in range(2) for r in permutations(range(3))]
    lookup = {state: i for i, state in enumerate(states)}
    ai = torch.tensor([a for a, r in states])
    ranks = torch.tensor([r for a, r in states])
    prior, likelihood = target.terms(ai, ranks)
    logpi = (prior+beta*likelihood).numpy()
    pi = np.exp(logpi-logsumexp(logpi))
    transition = np.zeros((len(states), len(states)))
    seen = sorted(set(tokens))
    for si, (a, r) in enumerate(states):
        draws, probability = [], []
        # Draw columns: move, seen index, uniform label, head rank,
        # uniform rank, prior alpha, adjacent direction.
        for i in range(len(seen)):
            for j in range(3):
                draws.append((0, i, j, 0, 0, 0, 1)); probability.append(.4/len(seen)/3)
        for i in range(3):
            for j in range(3):
                draws.append((1, 0, 0, i, j, 0, 1)); probability.append(.4*mc.rank_proposal_weights[i]/3)
        for b in range(2):
            draws.append((2, 0, 0, 0, 0, b, 1)); probability.append(.1*mc.alpha_weights[b])
        for direction in (-1, 1):
            draws.append((3, 0, 0, 0, 0, 0, direction)); probability.append(.05)
        n = len(draws)
        old_a, old_r = ai[si].expand(n), ranks[si].expand(n, -1)
        old_p, old_l = prior[si].expand(n), likelihood[si].expand(n)
        columns = torch.tensor(draws).T
        proposed = target.propose_from_draws(old_a, old_r, *columns)
        new_a, new_r, correction = proposed
        destination = [lookup[(int(aa), tuple(rr))] for aa, rr in zip(new_a, new_r.tolist())]
        expected_correction = [math.log(mc.alpha_weights[a]/mc.alpha_weights[int(aa)]) if d[0] == 2 else 0.
                               for aa, d in zip(new_a, draws)]
        np.testing.assert_allclose(correction.numpy(), expected_correction, atol=1e-15)
        acceptance = np.minimum(1., np.exp(logpi[destination]-logpi[si]+correction.numpy()))
        for u, expect in ((acceptance/2, np.ones(n, dtype=bool)),
                          ((1+acceptance)/2, acceptance == 1)):
            # Uniform exactly one at acceptance one is outside RNG support;
            # replace it by a legal value strictly below one.
            u = np.minimum(u, np.nextafter(1., 0.))
            changed = target.accept_proposal(old_a, old_r, old_p, old_l, beta,
                                             proposed, torch.tensor(np.log(u)))
            np.testing.assert_array_equal(changed[4].numpy(), expect)
            for j, accepted in enumerate(expect):
                actual = (int(changed[0][j]), tuple(changed[1][j].tolist()))
                assert lookup[actual] == (destination[j] if accepted else si)
        for j, probability_j, accepted in zip(destination, probability, acceptance):
            transition[si, j] += probability_j*accepted
            transition[si, si] += probability_j*(1-accepted)
    np.testing.assert_allclose(transition.sum(1), 1, atol=1e-15)
    np.testing.assert_allclose(pi[:, None]*transition, (pi[:, None]*transition).T, atol=1e-15)


def test_normalizer_uniform_base_anchor_has_no_sampling_variance():
    mc = make_sampler(d=4, alpha=(0.,), weights=(1.,))
    target = ShortTarget(MomentTables.from_sampler(mc), [0, 1, 0, 2, 0])
    exact = math.log(independent_evidence(mc, np.array(target.tokens)))
    for seed in (3, 7):
        result = estimate_log_evidence(target, particles=16, temperatures=9, moves=2, seed=seed)
        assert result["log_evidence"] == pytest.approx(exact, abs=1e-11)
        assert sum(result["move_attempts"]) == 16*9*2
        np.testing.assert_allclose(result["effective_sample_sizes"], 16, atol=1e-11)
        assert result["final_distinct_initial_ancestors"] == 16


def test_nonuniform_normalizer_against_independent_integral(small):
    mc, tables = small
    tokens = np.array([0, 1, 0, 2, 0])
    exact = independent_evidence(mc, tokens)
    target = ShortTarget(tables, tokens)
    values = np.array([math.exp(estimate_log_evidence(target, particles=128,
        temperatures=12, moves=2, seed=700+i)["log_evidence"]) for i in range(12)])
    # A stochastic implementation check, not the stricter scientific gate.
    assert abs(values.mean()-exact) <= 6*values.std(ddof=1)/math.sqrt(len(values)) + 1e-12


def test_resampling_offspring_and_copy_independence():
    weights = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    generator = torch.Generator().manual_seed(42)
    frequencies = np.zeros(4)
    for _ in range(2000):
        selected = systematic_indices(weights, generator)
        frequencies += np.bincount(selected.numpy(), minlength=4)
    np.testing.assert_allclose(frequencies/(2000*4), weights.numpy(), atol=.006)
    ranks = torch.arange(12).reshape(4, 3)
    copied = ranks[torch.tensor([0, 0, 0, 1])]
    copied[0, 0] = 99
    assert copied[1, 0] == 0 and ranks[0, 0] == 0


def test_scope_and_cuda_fail_closed(monkeypatch):
    for tokens in ([0, 0, 0, 0, 1], [0, 0, 1, 0, 2], [0]*6):
        with pytest.raises(ValueError):
            row_descriptors(tokens, 64)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        checked_device("cuda")
    with pytest.raises(ValueError):
        checked_device("mps")


@pytest.mark.parametrize("code", [0, 75])
def test_benchmark_cli_propagates_completed_or_partial_exit_code(monkeypatch, code):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark_shared_base_evidence_gpu as script
    monkeypatch.setattr(script, "execute", lambda args: code)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--device", "cpu", "--case", "0,0,0,0,0",
                                      "--store", "unused", "--src", "unused", "--out", "unused"])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == code


def test_benchmark_deadline_writes_partial_receipt_and_returns_75(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark_shared_base_evidence_gpu as script
    dependency = Path("/Users/ruediger/Projects/product_model_with_memory")
    if not (dependency / "tables/anchors_prod").exists():
        pytest.skip("external sealed dependency unavailable")
    def expired(*args, **kwargs):
        raise TimeoutError("simulated bounded sampling deadline")
    monkeypatch.setattr(script, "estimate_log_evidence", expired)
    args = SimpleNamespace(device="cpu", out=tmp_path / "partial.json", cpu_threads=1,
        src=dependency/"src", store=dependency/"tables/anchors_prod",
        dependency_manifest=script.ROOT/"code/transformer/configs/shared_base_dependency_manifest_2026-09-16.json",
        cases=None, case=["0,0,0,0,0"], alpha_order=1, particles=2, temperatures=1,
        moves=1, replicates=1, cpu_replicates=0, torch_cpu_replicates=0, seed=44, seconds=30)
    assert script.execute(args) == 75
    receipt = json.loads(args.out.read_text())
    assert receipt["status"].startswith("partial:")
    assert receipt["records"] == [] and receipt["active"] is not None
    assert receipt["scientific_reference_validated"] is False


def test_actual_64_symbol_sealed_store_target():
    store = Path("/Users/ruediger/Projects/product_model_with_memory/tables/anchors_prod")
    src = Path("/Users/ruediger/Projects/product_model_with_memory/src")
    if not store.exists() or not src.exists():
        pytest.skip("real sealed-store integration requires the external numerical dependency")
    mc = SharedBaseMCMC(64, RankedLSARowEvidence(64, (.7, 1.2), (0, 1, 2),
                       store=store, src=src, u_lo=-32, u_hi=32, step=1/32), (.4, .6), (.2, .3, .5))
    tables = MomentTables.from_sampler(mc)
    rng = np.random.default_rng(404)
    ranks = np.stack([rng.permutation(64) for _ in range(8)])
    ai = np.arange(8)%2
    for tokens in ([24, 22, 19, 17, 24], [31, 23, 29, 17, 27], [50, 62, 41, 40, 61],
                   [0]*5, [0, 1, 0, 1, 0], [0, 1, 0, 2, 0]):
        target = ShortTarget(tables, tokens)
        p, l = target.terms(torch.tensor(ai), torch.tensor(ranks))
        original = PrefixPosterior(mc, tokens)
        expected = [original.evaluate(int(a), r).log_target for a, r in zip(ai, ranks)]
        np.testing.assert_allclose((p+l).numpy(), expected, atol=1e-10, rtol=0)
        result = estimate_log_evidence(target, particles=16, temperatures=3, moves=1, seed=59)
        assert np.isfinite(result["log_evidence"])
        assert result["scientific_reference_validated"] is False
