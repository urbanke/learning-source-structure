"""Reproduce Appendix B's small analytic and stored-data checks.

--rescale-legacy applies the approved sample-SE convention exactly once to
five known legacy aggregates. Other runs are read-only apart from --out.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from master_curve import r1_geometric
from necessary_examples import log_permutation_mixture
from necessary_examples_figure import c_uniform, r_star_exact_bits


def convert_legacy(folder: Path):
    changed=[]
    for d in (64,256,1024,4096,16384):
        path=folder/f'nec_d{d}_L3.json'
        data=json.loads(path.read_text())
        if data.get('se_ddof') == 1:
            continue
        if data.get('se_ddof',0) != 0 or data['rows'] != 32:
            raise ValueError(f'unexpected legacy variance convention in {path}')
        n=data['rows'];factor=math.sqrt(n/(n-1))
        data['rho_se']=(np.asarray(data['rho_se'])*factor).tolist()
        data['se_ddof']=1
        data['se_adjustment']={'from_ddof':0,'to_ddof':1,'factor':factor,
            'method':'Exact conversion of stored standard errors; sample means unchanged.'}
        data['note']='rho(k) estimates depth-mixture redundancy from 32 source/sample draws. D_estimate is the final recorded value, not a proved plateau.'
        path.write_text(json.dumps(data,indent=1)+'\n');changed.append(str(path))
    return changed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--rescale-legacy',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    folder=Path('results/mechanism/necessary')
    if args.rescale_legacy: print('converted',convert_legacy(folder))
    rng=np.random.default_rng(123)
    permutation_errors=[]
    for d in range(2,7):
        q=rng.dirichlet(np.ones(d));counts=rng.integers(0,8,(9,d))
        perms=np.array(list(itertools.permutations(range(d))))
        direct=logsumexp(counts@np.log(q)[perms].T,axis=1)-math.log(len(perms))
        error=float(np.max(np.abs(direct-log_permutation_mixture(counts,q))))
        assert error<1e-11
        ratio=log_permutation_mixture(counts,q,np.log(q))
        ratio_error=float(np.max(abs(ratio-(direct-counts@np.log(q)))))
        assert ratio_error<1e-11
        # The true assignment is one summand in the mixture.
        assert np.min(ratio)>=-math.lgamma(d+1)-1e-12
        permutation_errors.append({'d':d,'maximum_error_nats':error,
                                   'maximum_ratio_error_nats':ratio_error})
    mc=json.loads(Path('results/mechanism/master_curves.json').read_text())
    lx=np.log(mc['x']);mr=np.array(mc['r3_bits']);c3=mc['c_bits']['3']
    def reference(x):
        x=np.asarray(x,dtype=float);a=np.interp(np.log(x),lx,mr)
        a[x>300]=.5*np.log2(x[x>300])+c3
        a[x<1e-4]=mc['slope0_bits_per_sample']['3']*x[x<1e-4]
        return a
    layered=[];allz=[]
    for d in (64,256,1024,4096,16384):
        data=json.loads((folder/f'nec_d{d}_L3.json').read_text())
        assert data['se_ddof']==1
        k=np.asarray(data['k']);y=np.asarray(data['rho_mean']);se=np.asarray(data['rho_se'])
        pred=d*reference(k/d);sel=k>=8;z=(y-pred)/se;allz.extend(z[sel])
        layered.append(dict(d=d,points=int(sum(sel)),maximum_abs_z=float(max(abs(z[sel]))),
            terminal_se_bits=float(se[-1]),terminal_mean_bits=float(y[-1]),
            last10_slope_fraction=float(np.polyfit(np.log2(k[-10:]),y[-10:],1)[0]/((d-1)/2))))
    allz=np.asarray(allz)
    examples=json.loads(Path('results/mechanism/necessary_examples.json').read_text())
    uniform=[]
    for row in examples['uniform']:
        d=row['d'];ks=np.asarray(row['k']);mean=np.asarray(row['rho_mean']);se=np.asarray(row['rho_se'])
        select=np.unique(np.linspace(0,len(ks)-1,5,dtype=int)) if args.smoke else np.arange(len(ks))
        theory=np.asarray([r_star_exact_bits(int(k),d) for k in ks[select]])
        z=(mean[select]-theory)/se[select]
        tail=(d-1)/2*np.log2(ks)+c_uniform(d)
        good=np.abs(mean-tail)/mean<=.02
        onset=next((i for i in range(len(ks)) if np.all(good[i:])),None)
        uniform.append(dict(d=d,C_bits=c_uniform(d),checked_points=len(select),maximum_abs_z=float(max(abs(z))),
            two_percent_suffix_start_k=None if onset is None else int(ks[onset]),
            two_percent_suffix_start_k_over_d=None if onset is None else float(ks[onset]/d)))
    profile=[]
    for row in examples['profile']:
        profile.append(dict(d=row['d'],draws=row['draws'],terminal_mean_bits=row['rho_mean'][-1],
            terminal_se_bits=row['rho_se'][-1],limit_bits=row['plateau_log2_dfact']))
    pair=json.loads(Path('results/mechanism/paired_depth_validation.json').read_text())
    mix=np.asarray(pair['rho_mean']);fixed=np.asarray(pair['fixed_depth_rho_mean']);excess=np.asarray(pair['paired_depth_excess_mean'])
    assert np.max(abs(mix-fixed-excess))<1e-10
    assert np.max(excess)<=math.log2(pair['l_max'])+1e-10
    paired=dict(d=pair['d'],L=pair['L'],rows=pair['rows'],kmax=pair['k'][-1],
        terminal_mixture_mean_bits=float(mix[-1]),terminal_fixed_depth_mean_bits=float(fixed[-1]),
        terminal_paired_excess_bits=float(excess[-1]),terminal_paired_excess_se_bits=pair['paired_depth_excess_se'][-1],
        upper_bound_bits=math.log2(pair['l_max']),se_ddof=pair['se_ddof'])
    accuracy=[]
    for d in [64,16384]:
        exact=r_star_exact_bits(64*d,d)/d;limit=r1_geometric(64.)
        accuracy.append(dict(d=d,k=64*d,finite_bits_per_symbol=exact,limit_bits_per_symbol=limit,
            absolute_difference_bits_per_symbol=abs(exact-limit)))
    out=dict(se_ddof=1,permanent_vs_permutation_enumeration=permutation_errors,
        layered=layered,coverage=dict(points=len(allz),within_two=int(sum(abs(allz)<=2)),within_four=int(sum(abs(allz)<=4)),
            fraction_within_two=float(np.mean(abs(allz)<=2)),maximum_abs_z=float(max(abs(allz)))),
        uniform=uniform,profile=profile,paired_validation=paired,finite_alphabet_accuracy=accuracy,
        qualification='Pointwise comparisons use sample standard errors, shared sample paths, and the same numerical interpolation/extrapolation as the figure.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('coverage','uniform','profile','paired_validation')},indent=2))

if __name__=='__main__':main()
