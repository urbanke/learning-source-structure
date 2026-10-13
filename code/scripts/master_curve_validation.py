"""Small deterministic convergence study for Appendix B's Poisson curve.

The saved differences compare numerical settings at stated check points;
they are empirical error estimates, not rigorous error bounds over an interval.
Run with --smoke for one point instead of the four-point study.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln
from master_curve import r1_geometric


def density(depth: int, step: float, lower: float = -45.0,
            upper: float = 9.0, cutoff: float = 1e-15):
    s1 = np.arange(lower, upper, step)
    g = np.exp(s1 - np.exp(s1))
    g /= g.sum() * step
    f = g.copy()
    for _ in range(depth - 1):
        f = fftconvolve(f, g) * step
    f = np.maximum(f, 0)
    f /= f.sum() * step
    s = depth * lower + np.arange(len(f)) * step
    keep = f > f.max() * cutoff
    a, b = np.flatnonzero(keep)[[0, -1]]
    return s[a:b + 1], f[a:b + 1]


def poisson_information(x, s, f, tail=1e-12):
    mass = f * (s[1] - s[0])
    mass /= mass.sum()
    w = np.exp(s)
    high = np.flatnonzero(np.cumsum(mass[::-1])[::-1] < tail)
    whi = w[high[0]] if len(high) else w[-1]
    nmax = int(x * whi + 12 * math.sqrt(max(x * whi, 1)) + 80)
    lam = x * w
    pn = np.zeros(nmax + 1)
    conditional_negative_entropy = 0.0
    # Bounded temporary memory even for the denser integration grid.
    for a in range(0, nmax + 1, 128):
        n = np.arange(a, min(a + 128, nmax + 1), dtype=float)
        logp = n[:, None] * np.log(lam)[None, :] - lam - gammaln(n + 1)[:, None]
        p = np.exp(logp)
        pn[a:a + len(n)] = p @ mass
        conditional_negative_entropy += float(np.sum((p * logp) @ mass))
    nz = pn > 0
    return float((-np.dot(pn[nz], np.log(pn[nz]))
                  + conditional_negative_entropy) / math.log(2)), float(pn.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--master', type=Path, default=Path('results/mechanism/master_curves.json'))
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    xs = [0.1] if args.smoke else [0.001, 0.1, 1.0, 8.0]
    settings = {
        'reference': dict(step=1/128, lower=-45., upper=9., tail=1e-12),
        'finer_grid': dict(step=1/256, lower=-45., upper=9., tail=1e-12),
        'coarser_grid': dict(step=1/32, lower=-45., upper=9., tail=1e-12),
        'shorter_weight_interval': dict(step=1/128, lower=-30., upper=7., tail=1e-12),
        'shorter_count_tail': dict(step=1/128, lower=-45., upper=9., tail=1e-9),
    }
    evaluations = {}
    for name, setting in settings.items():
        s, f = density(3, setting['step'], setting['lower'], setting['upper'])
        evaluations[name] = []
        for x in xs:
            value, mass = poisson_information(x,s,f,setting['tail'])
            assert abs(1-mass) < max(5*setting['tail'],1e-10)
            evaluations[name].append(dict(x=x,r_bits=value,retained_count_mass=mass))
        print(name, [round(v['r_bits'], 9) for v in evaluations[name]], flush=True)
    reference = np.array([v['r_bits'] for v in evaluations['reference']])
    deltas = {name: float(np.max(np.abs(reference - np.array([v['r_bits'] for v in rows]))))
              for name, rows in evaluations.items() if name != 'reference'}
    s1, f1 = density(1, 1/128)
    depth_one = []
    for x in ([1.] if args.smoke else [0.1, 1., 8., 61.]):
        value, mass = poisson_information(x,s1,f1)
        depth_one.append(dict(x=x, numerical_bits=value,
            geometric_bits=r1_geometric(x), error_bits=value-r1_geometric(x), retained_count_mass=mass))
        assert abs(value-r1_geometric(x)) < 1e-8
    table = json.loads(args.master.read_text())
    lx = np.log(table['x']); values = np.asarray(table['r3_bits'])
    s3, f3 = density(3,1/128)
    interpolation = []
    for target in xs:
        i = min(max(np.searchsorted(np.exp(lx),target)-1,0),len(lx)-2)
        mid = (lx[i]+lx[i+1])/2
        x = math.exp(mid)
        exact,_ = poisson_information(x,s3,f3)
        coarse = float((values[i]+values[i+1])/2)
        left = math.exp((lx[i]+mid)/2); right=math.exp((mid+lx[i+1])/2)
        refined=(poisson_information(left,s3,f3)[0]+poisson_information(right,s3,f3)[0])/2
        interpolation.append(dict(x=x, direct_bits=exact, table_interpolation_bits=coarse,
            table_error_bits=coarse-exact, half_width_interpolation_bits=refined,
            half_width_error_bits=refined-exact))
        assert abs(refined-exact) <= abs(coarse-exact)+1e-7
    at112=[]
    if not args.smoke:
        for d in (1024,4096,16384):
            val,_=poisson_information(112/d,s3,f3)
            at112.append(dict(d=d,k=112,direct_bits=d*val,
                plotted_bits=d*float(np.interp(math.log(112/d),lx,values))))
    out=dict(depth=3,settings=settings,points=evaluations,max_differences_bits=deltas,
        depth_one_independent_check=depth_one,interpolation=interpolation,predictions_at_112=at112,
        qualification='Observed setting differences at the stated points; no interval-wide or asymptotic-extrapolation error guarantee.',
        density_method='FFT convolution of log-exponential density, nonnegative projection and relative density cutoff 1e-15',
        units='bits per symbol')
    assert deltas['finer_grid'] < 1e-8
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    rows=['setting\tmaximum_change_bits']+[f'{key}\t{v:.9g}' for key,v in deltas.items()]
    args.out.with_suffix('.tsv').write_text('\n'.join(rows)+'\n')
    print('maximum changes',deltas)
    print('interpolation',[(r['table_error_bits'],r['half_width_error_bits']) for r in interpolation])

if __name__=='__main__':
    main()
