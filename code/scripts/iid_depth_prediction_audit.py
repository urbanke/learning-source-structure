"""Bounded independent checks of the depth-six scalar information calculation.

    OPENBLAS_NUM_THREADS=1 python code/scripts/iid_depth_prediction_audit.py
    OPENBLAS_NUM_THREADS=1 python code/scripts/iid_depth_prediction_audit.py --smoke --out /tmp/iid-depth-audit-smoke.json

Check conditional entropy through the independent depth-one information
integral, sum the marginal count entropy directly, and bound its omitted
tail using geometric memorylessness. Also check the Poisson-entropy
interpolator and the retained-density cutoff separately. No measured
Bayes costs are read or fitted.
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

from iid_depth_prediction import LN2, ScalarInformation
from iid_tilt_coverage_check import scalar_r1


ROOT = Path(__file__).resolve().parents[2]


def run_checks(smoke=False):
    start = time.monotonic()
    calc = ScalarInformation(6)
    audit = dict(L=6, smoke=smoke, conditional_entropy_checks=[],
                 direct_count_entropy_checks=[], poisson_entropy_checks=[],
                 density_cutoff_checks=[])

    def alternative_conditional(x):
        # W_6=U E, with U=W_5 and E exponential. For each fixed U,
        # E_E H(Pois(x U E)) = H(Geom(x U)) - r_1(x U).
        rates = x*np.exp(calc.sh)
        geometric = np.log1p(rates)+rates*np.log1p(1/rates)
        return float(calc.mh@(geometric-scalar_r1(rates, 256)*LN2))

    xs = [.001, 1., 1e3] if smoke else [1e-8, .001, .1, 1., 10., 1e3, 1e5, 1e8]
    for x in xs:
        standard = float(calc.mw@calc.poisson_entropy(math.log(x)+calc.sw))
        alternative = alternative_conditional(x)
        audit['conditional_entropy_checks'].append(dict(
            x=x, standard_nats=standard, alternative_nats=alternative,
            difference_bits=(standard-alternative)/LN2))

    count_cases = [(.001, 128), (.1, 8192)]
    if not smoke:
        count_cases.append((1., 32768))
    for x, cutoff in count_cases:
        u = math.log(x)+calc.sh
        logq = -np.logaddexp(0., -u)
        logbase = calc.logmh-np.logaddexp(0., u)
        partial = 0.
        for first in range(0, cutoff, 128):
            n = np.arange(first, min(cutoff, first+128))
            lp = logsumexp(logbase[None, :]+n[:, None]*logq[None, :], axis=1)
            partial -= float(np.exp(lp)@lp)

        # For N>=K, each geometric component is K+Geom(x U).
        # If p=P(N>=K), its omitted entropy is at most
        # -p log(p) + p g(E[N-K | N>=K]).
        logmass = float(logsumexp(calc.logmh+cutoff*logq))
        logexcess = float(logsumexp(calc.logmh+cutoff*logq+u))
        mass, mean = math.exp(logmass), math.exp(logexcess-logmass)
        upper = mass*(-logmass+math.log1p(mean)+mean*math.log1p(1/mean))/LN2
        alternative = alternative_conditional(x)
        standard = float(calc.mw@calc.poisson_entropy(math.log(x)+calc.sw))
        lower = (partial-alternative)/LN2
        value = calc(x)
        inferred_tail = value-lower+(standard-alternative)/LN2
        audit['direct_count_entropy_checks'].append(dict(
            x=x, direct_count_cutoff=cutoff, direct_information_lower_bits=lower,
            omitted_marginal_entropy_upper_bits=upper, evaluator_bits=value,
            evaluator_minus_lower_bits=value-lower,
            inferred_marginal_tail_bits=inferred_tail))
        assert -2e-10 <= inferred_tail <= upper+2e-10

    rates = ([1e-8, .373, 499.71, 500.1, 1000.] if smoke else
             [1e-8, .000137, .0317, .373, 1.113, 4.25, 33.3, 101.5,
              499.71, 500.1, 1000., 3000.])
    for rate in rates:
        n = np.arange(int(math.ceil(rate+16*math.sqrt(rate)+100)))
        lp = n*math.log(rate)-rate-gammaln(n+1)
        direct = -float(np.exp(lp)@lp)
        approximation = float(calc.poisson_entropy(np.array([math.log(rate)]))[0])
        audit['poisson_entropy_checks'].append(dict(
            rate=rate, difference_bits=(approximation-direct)/LN2))

    for cut in ([1e-13] if smoke else [1e-14, 1e-13]):
        other = ScalarInformation(6, relative_cut=cut)
        for x in ([.001, 1., 1e3] if smoke else [1e-8, .001, 1., 1e3, 1e8]):
            audit['density_cutoff_checks'].append(dict(
                cutoff=cut, x=x, difference_bits=other(x)-calc(x)))

    for name, group in [('conditional_entropy', 'conditional_entropy_checks'),
                        ('poisson_entropy', 'poisson_entropy_checks'),
                        ('cutoff', 'density_cutoff_checks')]:
        maximum = max(abs(c['difference_bits']) for c in audit[group])
        audit[f'maximum_{name}_difference_bits'] = maximum
        assert maximum < 1e-8, (name, maximum)
    audit['diagnostics'] = calc.diagnostics()
    assert abs(audit['diagnostics']['mean_weight']-1.) < 1e-7
    audit['elapsed_seconds'] = time.monotonic()-start
    audit['methods'] = dict(
        conditional_entropy='Integrate g(x U)-r1(x U) over W_5 using the independent exact scalar_r1 integral; no Poisson-entropy spline/asymptotic.',
        marginal_entropy='Direct count sum, without Euler-Maclaurin. Bound the omitted entropy by the geometric maximum-entropy law at its conditional tail mean.',
        poisson_entropy='Direct Poisson probability summation at rates away from the interpolation grid and above the asymptotic switch.',
        density_cutoff='Hold all other settings fixed and raise the relative cutoff.')
    audit['scope'] = ('No Bayes-cost data are read or fitted. The analytic geometric tail '
                      'bound pertains to the retained grid and is evaluated numerically; '
                      'it is not a certified total bound for continuous-density truncation.')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--out', type=Path,
                        default=ROOT/'results/mechanism/iid_depth_prediction_independent_audit.json')
    args = parser.parse_args()
    result = run_checks(args.smoke)
    result['source_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['iid_depth_prediction_audit.py', 'iid_depth_prediction.py',
                     'iid_tilt_coverage_check.py']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k.startswith('maximum_') or k == 'elapsed_seconds'}, indent=2))


if __name__ == '__main__':
    main()
