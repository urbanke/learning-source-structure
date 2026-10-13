"""Evaluate explicit LSA gap formulas from source parameters, without draws.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_parameter_gap.py --smoke

The integer_interpolated_gap_bits field evaluates the full-integral formula
in notes/lsa_gap_result_2026-09-08.md. Before numerical error, that expression
has an additive o(d^2) guarantee on positive d^2 length windows for fixed
0<alpha<1, fixed source depth, and the actual O(log d) coding cap.
It has not achieved small finite crossing errors at the saved dimensions.
The leading_gap_bits field instead replaces integrals by maxima and drops
finite unigram learning; that extra reduction has not been proved here.
The full_integral_mean_profile_gap_bits field uses fractional row lengths
without integer interpolation, as a separate diagnostic. All retain the
actual allowed coding depths. No output is a certified finite-error bound.
Tail mass is reported, but
does not by itself bound truncation error in the objective. Only existing
sealed moment tables are read; no samples or caches are built.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

from tilted_row_reference import configure_store, DEFAULT_STORE

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)
EULER = 0.5772156649015329


class LeadingGap:
    def __init__(self, source_depth, coding_cap, tables, rmax=1024, step=.12):
        self.source_depth = source_depth
        self.coding_cap = coding_cap
        self.tables = tables
        self.rmax = rmax
        self.step = step
        self.u = np.arange(-24., 30.+step/2, step)
        self.r = np.arange(rmax+1)
        self.factorial = gammaln(self.r+1)
        self.kernels = []
        for level in range(1, coding_cap+1):
            phi = self.log_phi(level)
            # log(r! p_{L,lambda}(r)); the factorial cancels in the KL.
            self.kernels.append(phi+self.r[:, None]*self.u)
        source_logp = self.log_phi(source_depth)+self.r[:, None]*self.u-self.factorial[:, None]
        self.source_pmf = np.exp(source_logp)

    def integral_cost(self, profiles, sample_counts):
        """Full depth/scale integral at a mean profile; still an approximation.

        This retains gamma factors and mixture weights but does not average
        the nonlinear profile cost. The source profiles used below omit the
        source normalizer. Truncated profile tails are diagnosed separately.
        """
        profiles = np.atleast_2d(profiles)
        sample_counts = np.atleast_1d(sample_counts)
        missing_counts = sample_counts-profiles @ self.r
        logweights = np.full(len(self.u), math.log(self.step))
        logweights[[0, -1]] -= math.log(2.)
        integrals = []
        depth_one = None
        edge_mass_log = -math.inf
        for level, kernel in enumerate(self.kernels, start=1):
            exponent = profiles @ kernel+missing_counts[:, None]*self.u
            logz = logsumexp(exponent+logweights, axis=1)
            edge_mass_log = max(edge_mass_log, float(np.max(
                exponent[:, [0, -1]]-np.max(exponent,axis=1)[:,None])))
            integrals.append(logz)
            if level == 1:
                direct = (gammaln(sample_counts)-logz)/LN2
                closed = (gammaln(sample_counts+profiles.sum(1))-gammaln(profiles.sum(1))
                          -profiles @ self.factorial)/LN2
                # The analytic identity applies when the profile is complete.
                depth_one = float(np.max(abs(direct-closed)))
        result = (gammaln(sample_counts)-logsumexp(integrals,axis=0)
                  +math.log(self.coding_cap))/LN2
        return result, dict(largest_log_integrand_at_grid_edge_relative_to_peak=edge_mass_log,
                            depth_one_identity_discrepancy_bits=depth_one)

    def source_profiles(self, q, sample_counts):
        """Mean scalar count profiles, with an exact zero-length boundary."""
        sample_counts = np.asarray(sample_counts, dtype=float)
        result = np.zeros((len(sample_counts), self.rmax+1))
        result[sample_counts == 0, 0] = len(q)
        live = np.flatnonzero(sample_counts > 0)
        if not len(live):
            return result
        intensity = sample_counts[live, None]*q[None, :]
        left = np.floor((np.log(intensity)-self.u[0])/self.step).astype(int)
        if left.min() < 0 or left.max()+1 >= len(self.u):
            raise ValueError('Source count intensities exceed the explicit integration grid')
        lower = np.exp(self.u[left])
        fraction = (intensity-lower)/(np.exp(self.u[left+1])-lower)
        weights = np.zeros((len(live), len(self.u)))
        rows = np.broadcast_to(np.arange(len(live))[:, None], left.shape)
        np.add.at(weights,(rows,left),1-fraction)
        np.add.at(weights,(rows,left+1),fraction)
        result[live] = weights @ self.source_pmf.T
        return result

    def log_phi(self, level):
        if level == 1:
            return self.factorial[:, None]-(self.r[:, None]+1)*np.logaddexp(0., self.u)
        return self.tables.log_phi_matrix(level, self.r.tolist(), self.u)

    def evaluate(self, d, alpha, n):
        if n < 2 or int(n) != n:
            raise ValueError('This numerical diagnostic requires an integer n >= 2')
        q = np.arange(1, d+1, dtype=float)**(-alpha)
        q /= q.sum()
        t = n-1.
        k = t*q
        intensity = t*q[:, None]*q[None, :]
        log_intensity = np.log(intensity)
        left = np.floor((log_intensity-self.u[0])/self.step).astype(int)
        if left.min() < 0 or left.max()+1 >= len(self.u):
            raise ValueError('Source count intensities exceed the explicit integration grid')
        # Convex interpolation in lambda, not log(lambda), preserves means
        # of the untruncated mixed Poisson distributions exactly.
        lower = np.exp(self.u[left])
        fraction = (intensity-lower)/(np.exp(self.u[left+1])-lower)
        weights = np.zeros((d, len(self.u)))
        rows = np.broadcast_to(np.arange(d)[:, None], left.shape)
        np.add.at(weights, (rows, left), 1-fraction)
        np.add.at(weights, (rows, left+1), fraction)
        profiles = weights @ self.source_pmf.T
        source_gain = self.source_depth*(1-EULER)
        base = k*(np.log(k)+float(q @ np.log(q))+source_gain-1)
        best = np.full(d, -np.inf)
        best_level = np.zeros(d, dtype=int)
        best_u = np.zeros(d)
        boundary = False
        for level, kernel in enumerate(self.kernels, start=1):
            objective = profiles @ kernel
            indices = np.argmax(objective, axis=1)
            boundary |= bool(np.any((indices == 0)|(indices == len(self.u)-1)))
            values = objective[np.arange(d), indices]
            us = self.u[indices].copy()
            # Local quadratic refinement is checked by halving the grid.
            interior = np.flatnonzero((indices > 0)&(indices < len(self.u)-1))
            a = objective[interior, indices[interior]-1]
            b = objective[interior, indices[interior]]
            c = objective[interior, indices[interior]+1]
            curvature = a-2*b+c
            valid = curvature < 0
            ii = interior[valid]
            offset = np.clip((a[valid]-c[valid])/(2*curvature[valid]), -.5, .5)
            values[ii] = b[valid]+.5*(c[valid]-a[valid])*offset+.5*curvature[valid]*offset**2
            us[ii] += self.step*offset
            improve = values > best
            best[improve], best_level[improve], best_u[improve] = values[improve], level, us[improve]
        redundancy = (base-best)/LN2
        first_token_penalty = (math.log(d)+float(q @ np.log(q)))/LN2
        gap = t*source_gain/LN2-float(redundancy.sum())-first_token_penalty
        row_costs, row_diagnostics = self.integral_cost(profiles, k)
        unigram_profile = binom.pmf(self.r[:,None], int(n), q[None,:]).sum(1)
        unigram_cost, unigram_diagnostics = self.integral_cost(unigram_profile, [n])
        full_profile_gap = float(unigram_cost[0]-math.log2(d)-row_costs.sum())
        # The proved expression interpolates integer-length row costs.
        # It does not substitute a fractional total into the gamma factor.
        floors = np.floor(k).astype(int)
        lengths, inverse = np.unique(np.r_[floors, floors+1], return_inverse=True)
        positive = lengths > 0
        integer_profiles = self.source_profiles(q, lengths[positive])
        integer_costs = np.zeros(len(lengths))
        integer_costs[positive], integer_diagnostics = self.integral_cost(integer_profiles, lengths[positive])
        fraction = k-floors
        interpolated = (1-fraction)*integer_costs[inverse[:d]]+fraction*integer_costs[inverse[d:]]
        theorem_gap = float(unigram_cost[0]-math.log2(d)-interpolated.sum())
        return dict(d=d, alpha=alpha, source_depth=self.source_depth, n=float(n),
                    coding_cap=self.coding_cap, leading_gap_bits=gap,
                    gap_per_token=gap/n, row_learning_sum_bits=float(redundancy.sum()),
                    first_token_penalty_bits=first_token_penalty,
                    chosen_depths={str(j):int(np.sum(best_level==j)) for j in np.unique(best_level)},
                    minimum_row_redundancy_bits=float(redundancy.min()),
                    source_profile_missing_symbols=float(d*d-profiles.sum()),
                    source_profile_missing_counts=float(t-np.sum(profiles @ self.r)),
                    largest_missing_row_count=float(np.max(k-profiles @ self.r)),
                    boundary_mode=boundary, log_scale_range=[float(best_u.min()),float(best_u.max())],
                    rmax=self.rmax, log_grid_step=self.step,
                    full_integral_mean_profile_gap_bits=full_profile_gap,
                    integer_interpolated_gap_bits=theorem_gap,
                    integer_interpolation_change_bits=theorem_gap-full_profile_gap,
                    integer_row_integral_diagnostics=integer_diagnostics,
                    full_integral_unigram_bits=float(unigram_cost[0]),
                    full_integral_copier_bits=float(math.log2(d)+row_costs.sum()),
                    row_integral_diagnostics=row_diagnostics,
                    unigram_integral_diagnostics=unigram_diagnostics,
                    unigram_profile_missing_symbols=float(d-unigram_profile.sum()),
                    unigram_profile_missing_counts=float(n-unigram_profile @ self.r),
                    source_normalizer_variance=float((2**self.source_depth-1)*(q @ q)))


def saved_source_diagnostic(cell, indices):
    """Locate one finite-source error using retained tables, not a new law.

    The best fixed iid distribution for the expected token counts is pbar.
    The expected cost of its known probabilities is n H(pbar). The copier's
    corresponding cost is log2(d)+sum_a E[K_a|T] H(T_a). These are oracle
    costs for diagnosing the approximation, not the implemented predictors.
    """
    raw_path = ROOT/cell['raw_path']
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if digest != cell['raw_sha256']:
        raise ValueError('Retained source file differs from its recorded hash')
    entropy = lambda p: -np.sum(p*np.log2(np.maximum(p, 1e-300)), axis=-1)
    out = []
    with np.load(raw_path, allow_pickle=False) as raw:
        q = raw['q']
        transition = raw['transition_matrices']
        row_entropies = entropy(transition)
        for index in indices:
            n = cell['tokens'][index]
            t = n-1
            occupation = raw['expected_occupation_laws'][:, index, :]
            continuation = np.einsum('sa,sab->sb', occupation, transition)
            pbar = (q[None, :]+t*continuation)/n
            known_unigram = n*entropy(pbar)
            known_copier = math.log2(cell['d'])+t*np.sum(occupation*row_entropies, axis=1)
            advantage = known_unigram-known_copier
            leading_advantage = t*cell['L']*(1-EULER)/LN2-(math.log2(cell['d'])-entropy(q))
            out.append(dict(n=n, retained_sources=len(advantage),
                            known_probability_advantage_bits=float(advantage.mean()),
                            known_probability_advantage_SE_bits=float(advantage.std(ddof=1)/np.sqrt(len(advantage))),
                            leading_advantage_bits=float(leading_advantage),
                            leading_advantage_excess_bits=float(leading_advantage-advantage.mean()),
                            known_unigram_bits=float(known_unigram.mean()),
                            known_copier_bits=float(known_copier.mean()),
                            interpretation='Source-dependent diagnostic; not a parameter-only prediction.'))
    return dict(raw_path=cell['raw_path'], raw_sha256=digest, rows=out)


def geometric_self_test():
    """Independent depth-one count sum checks the scalar KL normalization."""
    from master_curve import r1_geometric
    checks = []
    d, alpha, n = 16, .5, 65
    q = np.arange(1,d+1,dtype=float)**(-alpha)
    q /= q.sum()
    g = lambda x: ((1+x)*np.log1p(x)-x*np.log(x))/LN2
    exact = 0.
    for k in (n-1)*q:
        intensities = k*q
        exact += sum(r1_geometric(float(x)) for x in intensities)
        exact += d*g(k/d)-sum(g(intensities))
    for step in (.06,.03,.015):
        calculation = LeadingGap(1,1,None,rmax=512,step=step).evaluate(d,alpha,n)
        error = calculation['row_learning_sum_bits']-exact
        checks.append(dict(log_grid_step=step, direct_geometric_row_sum_bits=float(exact),
                           calculated_row_sum_bits=calculation['row_learning_sum_bits'],
                           discrepancy_bits=float(error)))
    if abs(checks[-1]['discrepancy_bits']) >= .002:
        raise AssertionError(checks)
    if abs(checks[-1]['discrepancy_bits']) >= abs(checks[0]['discrepancy_bits'])/8:
        raise AssertionError('Scalar count interpolation did not converge as expected')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=.75)
    parser.add_argument('--source-depth', type=int, default=2)
    parser.add_argument('--rmax', type=int, default=1024)
    parser.add_argument('--step', type=float, default=.12)
    parser.add_argument('--out', type=Path, default=ROOT/'results/mechanism/lsa_parameter_gap_check.json')
    parser.add_argument('--diagnose-saved-sources', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--indices', type=int, nargs='+', help='Saved checkpoint indices to evaluate')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(geometric_self_test(),indent=2))
        return
    if not 0 < args.alpha < 1:
        raise ValueError('This candidate is being investigated only for 0<alpha<1')
    source_path = ROOT/'results/mechanism/crossing_finite_count_prediction.json'
    data = json.loads(source_path.read_text())
    cell = next(c for c in data['cells'] if (c['d'],c['alpha'],c['L']) == (args.d,args.alpha,args.source_depth))
    ix = args.indices if args.indices is not None else ([4] if args.smoke else list(range(2,8)))
    configure_store(DEFAULT_STORE)
    from lsa.universal_tables import UniversalTables
    started = time.monotonic()
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    try:
        evaluator = LeadingGap(args.source_depth, cell['l_max'], tables, args.rmax, args.step)
        rows = []
        for i in ix:
            result = evaluator.evaluate(args.d,args.alpha,cell['tokens'][i])
            result.update(saved_count_gap_bits=cell['count']['gap']['mean'][i],
                          saved_count_SE_bits=cell['count']['gap']['SE'][i],
                          saved_chain_gap_bits=cell['chain']['gap']['mean'][i],
                          saved_chain_SE_bits=cell['chain']['gap']['SE'][i])
            rows.append(result)
            print(json.dumps(result),flush=True)
    finally:
        tables.close()
    output = dict(complete=True, interpretation=__doc__,
                  source=str(source_path.relative_to(ROOT)),
                  source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  rows=rows, elapsed_seconds=time.monotonic()-started,
                  theorem_reference='notes/lsa_gap_result_2026-09-08.md, eq1 (integer_interpolated_gap_bits); asymptotic guarantee eq2',
                  evaluation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.diagnose_saved_sources:
        output['retained_source_diagnostic'] = saved_source_diagnostic(cell, ix)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
