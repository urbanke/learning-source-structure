"""Small coupled-prefix check of stationary laws above the Zipf threshold.

Samples transition tables, not trajectories. Nested alphabets use the same
underlying weight matrix. Total variation compares adjacent finite alphabets,
with the smaller stationary law padded by zeros; it is not distance to the
unknown infinite-alphabet law.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.linalg import solve


def stationary_checked(T):
    n=len(T);pi=np.full(n,1/n)
    for it in range(10000):
        new=pi@T
        if np.max(abs(new-pi))<2e-14:
            pi=new/new.sum();method='power iteration';break
        pi=new
    else:
        A=T.T-np.eye(n);A[-1]=1;b=np.zeros(n);b[-1]=1
        pi=solve(A,b);method='linear solve after iteration limit'
        if min(pi)<-1e-10:raise AssertionError('negative stationary mass')
        pi=np.maximum(pi,0);pi/=pi.sum()
    residual=float(np.sum(abs(pi@T-pi)))
    assert residual<1e-10,residual
    return pi,dict(method=method,iterations=it+1,residual_l1=residual)


def entropy(p):
    nz=p>0;return float(-np.sum(p[nz]*np.log2(p[nz])))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    ds=[16,32] if args.smoke else [32,128,512];reps=2 if args.smoke else 8
    rows=[]
    for alpha in (1.25,1.5,2.):
        for L in (1,3):
            I=np.zeros((reps,len(ds)));tv=np.zeros_like(I);piq=np.zeros_like(I);diag=[]
            for r in range(reps):
                rng=np.random.default_rng([20260908,r,int(alpha*100),L])
                weights=np.ones((max(ds),max(ds)))
                for _ in range(L):weights*=rng.exponential(size=weights.shape)
                prev=None
                for j,d in enumerate(ds):
                    q=np.arange(1,d+1,dtype=float)**(-alpha);q/=q.sum()
                    T=weights[:d,:d]*q[None,:];T/=T.sum(1,keepdims=True)
                    pi,diagnostic=stationary_checked(T);diag.append(diagnostic)
                    I[r,j]=entropy(pi)-float(pi@np.array([entropy(row) for row in T]))
                    assert I[r,j]>=-1e-12
                    piq[r,j]=float(abs(pi-q).sum()/2)
                    if prev is not None:tv[r,j]=float(abs(pi-np.pad(prev,(0,d-len(prev)))).sum()/2)
                    prev=pi
            for j,d in enumerate(ds):
                row=dict(alpha=alpha,L=L,d=d,reps=reps,
                    I_mean_bits=float(I[:,j].mean()),I_se_bits=float(I[:,j].std(ddof=1)/math.sqrt(reps)),
                    TV_pi_to_q_mean=float(piq[:,j].mean()),
                    previous_d=None if j==0 else ds[j-1],
                    TV_successive_mean=None if j==0 else float(tv[:,j].mean()),
                    TV_successive_se=None if j==0 else float(tv[:,j].std(ddof=1)/math.sqrt(reps)),
                    maximum_stationary_residual_l1=max(x['residual_l1'] for x in diag),
                    max_iterations=max(x['iterations'] for x in diag))
                rows.append(row);print(alpha,L,d,'I',row['I_mean_bits'],'TVprevious',row['TV_successive_mean'],flush=True)
    out=dict(se_ddof=1,seed_protocol='default_rng([20260908,replication,100*alpha,L])',
        convention='Nested prefixes of a common maximum-size iid product-exponential matrix; stationary laws solved numerically; no sampled chains.',
        interpretation='Successive finite-size differences illustrate convergence; these are not error bounds to an infinite-alphabet law.',rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')
    lines=['alpha\tL\td\tI_bits\tI_SE_bits\tprevious_d\tTV_successive\tTV_SE']
    for r in rows:lines.append('\t'.join(str(r[k]) for k in ['alpha','L','d','I_mean_bits','I_se_bits','previous_d','TV_successive_mean','TV_successive_se']))
    args.out.with_suffix('.tsv').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
