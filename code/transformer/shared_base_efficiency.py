"""Tempered auxiliary proposals with the unchanged full posterior correction.

Only the proposal density changes: its token counts are multiplied by a fixed
positive power. The MH target remains PrefixPosterior, integrating all depths
and retaining the full exponent/rank prior. Powers are algorithm settings,
never source parameters. Auxiliary reversibility and its endpoint correction
follow the same argument as shared_base_surrogate.py.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit, gammaln, logsumexp

from shared_base_surrogate import SharedBaseSurrogateMCMC


class SharedBaseEfficiencyMCMC(SharedBaseSurrogateMCMC):
    def __init__(self, *args, auxiliary_power=0.5, **kwargs):
        if not np.isfinite(auxiliary_power) or auxiliary_power <= 0:
            raise ValueError('auxiliary_power must be positive and finite')
        super().__init__(*args, **kwargs)
        self.auxiliary_power = float(auxiliary_power)
        self.proposal_spec = deepcopy(self.proposal_spec)
        self.proposal_spec.update(
            id='shared-base-tempered-depth-zero-surrogate-v1',
            auxiliary_power=self.auxiliary_power,
            auxiliary_target='alpha quadrature weight times product of q over every prefix token, raised to fixed auxiliary_power',
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def surrogate_log_target(self, counts, alpha_index, ranks):
        return super().surrogate_log_target(self.auxiliary_power*counts, alpha_index, ranks)

    def _surrogate_step(self, counts, alpha_index, ranks, rng):
        return super()._surrogate_step(self.auxiliary_power*counts, alpha_index, ranks, rng)


class SharedBaseEfficiencyRowMCMC(SharedBaseSurrogateMCMC):
    """Dirichlet row auxiliary whose additive label terms are precomputed.

For fixed concentration kappa, auxiliary row evidence has label terms
lgamma(kappa*q_b+n_ab)-lgamma(kappa*q_b). Its omitted row normalizers
depend only on observed row totals and kappa, so cancel in every move.
This auxiliary is solely a proposal; weighted-LSA rows remain the target.
"""
    def __init__(self, *args, concentration=1.0, **kwargs):
        if not np.isfinite(concentration) or concentration <= 0:
            raise ValueError('concentration must be positive and finite')
        super().__init__(*args, **kwargs)
        self.concentration = float(concentration)
        self._auxiliary_tables = {}
        self.proposal_spec = deepcopy(self.proposal_spec)
        self.proposal_spec.update(
            id='shared-base-dirichlet-row-surrogate-v1', concentration=self.concentration,
            auxiliary_target='q(first token) times independent Dirichlet(concentration*q) row evidences and alpha quadrature weight',
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def auxiliary_table(self, target):
        key = tuple(target.x)
        if key not in self._auxiliary_tables:
            table = np.zeros((self.d, len(self.alpha_weights), self.d))
            table[target.x[0]] += self.kernel.log_base_laws
            cq = self.concentration*self.kernel.base_laws
            for label in range(self.d):
                values, frequencies = np.unique(target.counts[:, label], return_counts=True)
                for count, frequency in zip(values, frequencies):
                    if count:
                        table[label] += frequency*(gammaln(cq+count)-gammaln(cq))
            table.setflags(write=False)
            self._auxiliary_tables[key] = table
        return self._auxiliary_tables[key]

    def surrogate_log_target(self, table, alpha_index, ranks):
        return float(self.log_alpha_weights[alpha_index]
                     + table[np.arange(self.d), alpha_index, ranks].sum())

    def _surrogate_step(self, table, alpha_index, ranks, rng):
        if rng.random() < .2:
            terms = self.log_alpha_weights + table[np.arange(self.d), :, ranks].sum(axis=0)
            return int(rng.choice(len(self.alpha_weights), p=np.exp(terms-logsumexp(terms))))
        labels = rng.permutation(self.d)
        pairs = labels[:2*(self.d//2)].reshape(-1, 2)
        i, j = pairs.T
        delta = (table[i, alpha_index, ranks[j]] + table[j, alpha_index, ranks[i]]
                 -table[i, alpha_index, ranks[i]] - table[j, alpha_index, ranks[j]])
        chosen = rng.random(len(pairs)) < expit(delta)
        i, j = i[chosen], j[chosen]
        ranks[i], ranks[j] = ranks[j].copy(), ranks[i].copy()
        return alpha_index

    def surrogate_proposal(self, target, state, rng):
        table = self.auxiliary_table(target)
        a, ranks = state.alpha_index, state.ranks.copy()
        before = self.surrogate_log_target(table, a, ranks)
        for _ in range(self.surrogate_steps):
            a = self._surrogate_step(table, a, ranks, rng)
        return a, ranks, before-self.surrogate_log_target(table, a, ranks)
