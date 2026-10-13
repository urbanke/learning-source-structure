"""Bounded prototype: normalized shared-base evidence AND next-token prediction.

The annealing loop preserves the original SMC target, RNG order, proposals and
resampling. Existing hash-locked modules are imported unchanged. Terminal
particles additionally supply Rao--Blackwellized predictions via the existing
PrefixPosterior.predict. This is an experimental estimator, not a certified
population reference. No oracle/source parameters enter the computation.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import logsumexp

from shared_base_evidence_smc import TemperedTransitionTarget, systematic_resample
from shared_base_mcmc import PrefixPosterior, SharedBaseMCMC
from shared_depth import _tokens


def checked_probs(value, d=None, tolerance=1e-7):
    p = np.asarray(value, dtype=np.float64)
    if (p.ndim != 1 or not len(p) or (d is not None and len(p) != d)
            or not np.isfinite(p).all() or np.any(p <= 0)
            or abs(float(p.sum())-1) > tolerance):
        raise ValueError("invalid predictive probabilities; no renormalization is applied")
    return p


def estimate(sampler, tokens, *, particles=32, temperatures=8, moves=1,
             seed=2026091801, resample_fraction=.75, seconds_limit=300.):
    """One finite-particle repetition, including an absolute prefix marginal.

    The seed/normalizer/move-count parity test compares this loop directly with
    the unchanged original implementation. Prediction uses no random draws.
    Time limits are checked between stages/particles; one kernel call can
    exceed the deadline. A surrounding batch allocation remains necessary.
    """
    if type(sampler) is not SharedBaseMCMC:
        raise TypeError("prototype supports only the original audited four-move sampler")
    for name, value, low in (("particles", particles, 2), ("temperatures", temperatures, 1),
                             ("moves", moves, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"invalid {name}")
    if not 0 < resample_fraction <= 1 or not np.isfinite(seconds_limit) or seconds_limit <= 0:
        raise ValueError("invalid resampling fraction or time limit")
    x = _tokens(tokens, sampler.d)
    if not len(x):
        raise ValueError("a first observed token is required")
    started = time.monotonic()

    def check_time():
        if time.monotonic()-started > seconds_limit:
            raise TimeoutError("bounded SMC/prediction repetition exceeded its time limit")

    rng = np.random.default_rng(seed)
    target = PrefixPosterior(sampler, x)
    tempered = TemperedTransitionTarget(target, 0.)
    states = []
    for _ in range(particles):
        check_time()
        ai = int(rng.choice(len(sampler.alpha_weights), p=sampler.alpha_weights))
        first_rank = int(rng.choice(sampler.d, p=sampler.kernel.base_laws[ai]))
        ranks = rng.permutation(sampler.d)
        other_label = int(np.flatnonzero(ranks == first_rank)[0])
        ranks[x[0]], ranks[other_label] = ranks[other_label], ranks[x[0]]
        states.append(tempered.evaluate(ai, ranks))
    log_weights = np.full(particles, -np.log(particles))
    log_evidence = -float(np.log(sampler.d))
    attempts, accepts = np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
    history, resampled = [], []
    ancestry = np.arange(particles)
    ancestry_history = [particles]
    betas = np.linspace(0., 1., temperatures+1)**3
    for previous, beta in zip(betas[:-1], betas[1:]):
        check_time()
        loglik = np.array([tempered.transition_loglik(s) for s in states])
        incremental = log_weights+(beta-previous)*loglik
        normalizer = float(logsumexp(incremental))
        log_evidence += normalizer
        log_weights = incremental-normalizer
        weights = np.exp(log_weights)
        ess = float(1/(weights@weights))
        history.append(ess)
        should_resample = ess < resample_fraction*particles
        resampled.append(should_resample)
        if should_resample:
            indices = systematic_resample(weights, rng)
            states = [states[i] for i in indices]
            ancestry = ancestry[indices]
            log_weights.fill(-np.log(particles))
        ancestry_history.append(len(np.unique(ancestry)))
        tempered.beta = float(beta)
        for i, state in enumerate(states):
            check_time()
            state = tempered.retarget(state)
            for _ in range(moves):
                state = sampler._step(tempered, state, rng, attempts, accepts)
            states[i] = state
    sampler_seconds = time.monotonic()-started
    # Duplicate terminal particles are scored once. This cache only merges
    # identical latent states; it does not discard particles or change weights.
    predictions = {}
    p = np.zeros(sampler.d, dtype=np.float64)
    max_raw_error = 0.
    for weight, state in zip(np.exp(log_weights), states):
        check_time()
        key = (state.alpha_index, tuple(map(int, state.ranks)))
        if key not in predictions:
            vector, _, error = target.predict(state)
            predictions[key] = checked_probs(vector, sampler.d)
            max_raw_error = max(max_raw_error, float(error))
        p += weight*predictions[key]
    checked_probs(p, sampler.d)
    return dict(schema="shared-base-predictive-smc-repetition-v1", n_observed=len(x),
        log_evidence=float(log_evidence), predictive_probs=p.tolist(),
        log_joint_next=(log_evidence+np.log(p)).tolist(),
        particles=particles, temperatures=temperatures, moves=moves, seed=seed,
        resample_fraction=resample_fraction, beta_schedule="(k/K)^3; fixed before sampling",
        effective_sample_sizes=history, resampled=resampled,
        initial_particle_ancestry_counts=ancestry_history,
        move_attempts=attempts.tolist(), move_accepts=accepts.tolist(),
        distinct_terminal_states=len(predictions),
        maximum_conditional_normalization_error=max_raw_error,
        sampler_seconds=sampler_seconds,
        prediction_seconds=time.monotonic()-started-sampler_seconds,
        elapsed_seconds=time.monotonic()-started, scientific_reference_validated=False)


def pool_repetitions(records):
    """Average unnormalized Z and Z*p, never the logarithms or bare p vectors.

    The ratio is finite-sample biased in general. Independent repetitions and
    refinement are still needed; reported MCSE is a delta-method diagnostic.
    """
    if not records:
        raise ValueError("no completed repetitions")
    logs = np.array([r["log_evidence"] for r in records], dtype=np.float64)
    p = np.stack([checked_probs(r["predictive_probs"]) for r in records])
    if not np.isfinite(logs).all() or np.any(logs > 1e-12):
        raise ValueError("invalid absolute log evidence")
    if p.shape[0] != len(logs):
        raise ValueError("mismatched repetitions")
    shift = float(logs.max())
    z = np.exp(logs-shift)
    mean_z = z.mean()
    mean_joint = np.mean(z[:, None]*p, axis=0)
    mean_p = checked_probs(mean_joint/mean_z)
    mean_log_z = shift+math.log(mean_z)
    se = (np.std(z[:, None]*(p-mean_p), axis=0, ddof=1)
          / (math.sqrt(len(z))*mean_z)) if len(z) > 1 else None
    return dict(log_evidence=mean_log_z, predictive_probs=mean_p.tolist(),
        log_joint_next=(mean_log_z+np.log(mean_p)).tolist(), repetitions=len(z),
        probability_mcse_delta_method=None if se is None else se.tolist(),
        effective_replica_count=float(z.sum()**2/(z@z)),
        maximum_single_replica_likelihood_share=float(z.max()/z.sum()),
        pooling="mean(Z*p)/mean(Z); mean likelihood before logarithm",
        scientific_reference_validated=False)


def combine_components(log_evidences, probabilities, prior=(1/3, 1/3, 1/3)):
    """Posterior family weights and mixture vector on one common history."""
    logs, prior = np.asarray(log_evidences, float), np.asarray(prior, float)
    p = np.stack([checked_probs(row) for row in probabilities])
    if (logs.shape != prior.shape or logs.shape != (len(p),)
            or not np.isfinite(logs).all() or np.any(logs > 1e-12)
            or not np.isfinite(prior).all() or np.any(prior <= 0)
            or abs(prior.sum()-1) > 1e-12):
        raise ValueError("invalid component evidence or prior")
    log_z = float(logsumexp(logs+np.log(prior)))
    weights = np.exp(logs+np.log(prior)-log_z)
    vector = checked_probs(weights@p)
    return dict(log_evidence=log_z, posterior_family_weights=weights.tolist(),
                predictive_probs=vector.tolist(),
                log_joint_next=(log_z+np.log(vector)).tolist())


def component_prediction(reference, tokens, d=64):
    """Existing inexpensive evidence adapter, evaluated at every successor."""
    x = _tokens(tokens, d)
    parent = float(reference.log_joint(x))
    children = np.array([reference.log_joint(np.append(x, j)) for j in range(d)])
    if not np.isfinite(parent) or parent > 1e-12 or not np.isfinite(children).all():
        raise ValueError("invalid component marginal likelihood")
    p = checked_probs(np.exp(children-parent), d)
    return dict(log_evidence=parent, predictive_probs=p.tolist(),
                log_joint_next=children.tolist())


def mixture_regret_mcse(records, fixed_components, oracle, prior=(1/3, 1/3, 1/3)):
    """Delta-method MCSE of the actual mixture regret, shared component middle.

    Retains covariance of each replicate's Z and Z*p. Sampling error of test
    sequences, missed posterior modes and quadrature error are not included.
    """
    if len(records) < 2:
        return None
    prior = np.asarray(prior, float)
    oracle = checked_probs(oracle)
    require_shape = prior.shape == (3,) and len(fixed_components) == 2
    if not require_shape or np.any(prior <= 0) or abs(prior.sum()-1) > 1e-12:
        raise ValueError("invalid family prior or fixed components")
    logs = np.asarray([r['log_evidence'] for r in records])
    fixed_logs = np.asarray([r['log_evidence'] for r in fixed_components])
    if not np.isfinite(logs).all() or not np.isfinite(fixed_logs).all():
        raise ValueError("nonfinite evidence")
    shift = float(max(logs.max(), fixed_logs.max()))
    z = np.exp(logs-shift)
    p = np.stack([checked_probs(r['predictive_probs'], len(oracle)) for r in records])
    fixed_p = np.stack([checked_probs(r['predictive_probs'], len(oracle)) for r in fixed_components])
    fixed_z = np.exp(fixed_logs-shift)*prior[[0, 2]]
    joint = z[:, None]*p
    denominator = fixed_z.sum()+prior[1]*z.mean()
    numerator = fixed_z@fixed_p+prior[1]*joint.mean(axis=0)
    influence = -(prior[1]/math.log(2))*((joint-joint.mean(axis=0))/numerator
        -(z-z.mean())[:, None]/denominator)@oracle
    return float(influence.std(ddof=1)/math.sqrt(len(records)))
