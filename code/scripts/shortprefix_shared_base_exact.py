"""Deterministic full-family Bayes for shared-base prefixes of length <=4.

Integrate existing weighted-LSA row moments with the sealed production store.
Enumerate the distinct observed ranks (at most three), or use symmetric sums
for a four-distinct prefix. Unseen labels are exchangeable. Evaluation source
parameters never enter this calculation.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
from ranked_lsa import RankedLSARowEvidence
from eval_shared_base_models import atomic_json, atomic_npz, sha_file


def canonical(tokens):
    labels={}
    return tuple(labels.setdefault(int(x),len(labels)) for x in tokens)


def patterns(n):
    return sorted({canonical(x) for x in itertools.product(range(n),repeat=n)})


def row_moments(kernel,alpha_index):
    """m1..m4 and M11/M21, using batched existing quadrature terms.

    Off-diagonal M21[a,b]=E[T_a^2 T_b]. Its diagonal is E[T_a^3].
    Both matrices include the actual, unrenormalized quadrature errors.
    """
    q=kernel.base_laws[alpha_index]
    logs=kernel.log_base_laws[alpha_index]
    nk=len(kernel.depths);d=kernel.d
    result={f'm{p}':np.repeat((q**p)[None,:],nk,axis=0) for p in range(1,5)}
    result['M11']=np.repeat((q[:,None]*q[None,:])[None,:,:],nk,axis=0)
    result['M21']=np.repeat((q[:,None]**2*q[None,:])[None,:,:],nk,axis=0)
    if len(kernel._positive):
        baseline=kernel._baseline(alpha_index)
        terms={p:np.stack([kernel._term(alpha_index,r,p) for r in range(d)],axis=1) for p in range(1,5)}
        for p in range(1,5):
            exponent=baseline[:,None,:]+p*kernel.u+terms[p]+p*logs[None,:,None]
            result[f'm{p}'][kernel._positive]=np.exp(exponent).sum(axis=-1)*(kernel.step/math.gamma(p))
        for j,li in enumerate(kernel._positive):
            half=(baseline[j]+2*kernel.u+np.log(kernel.step))/2
            features=np.exp(logs[:,None]+terms[1][j]+half)
            result['M11'][li]=features@features.T
            half=(baseline[j]+3*kernel.u+np.log(kernel.step/2))/2
            left=np.exp(2*logs[:,None]+terms[2][j]+half)
            right=np.exp(logs[:,None]+terms[1][j]+half)
            result['M21'][li]=left@right.T
    diagonal=np.arange(d)
    result['M11'][:,diagonal,diagonal]=result['m2']
    result['M21'][:,diagonal,diagonal]=result['m3']
    errors=dict(m1_normalization=float(np.max(abs(result['m1'].sum(-1)-1))),
        M11_marginal=float(np.max(abs(result['M11'].sum(-1)-result['m1']))),
        M21_marginal=float(np.max(abs(result['M21'].sum(-1)-result['m2']))),
        M11_symmetry=float(np.max(abs(result['M11']-result['M11'].transpose(0,2,1)))))
    if max(errors.values())>1e-7 or any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError(f'row-moment identities failed: {errors}')
    return result,errors


def row_factor(counts,ranks,m):
    active=[(i,int(c)) for i,c in enumerate(counts) if c]
    if not active:return 1.
    if len(active)==1:
        i,p=active[0]
        return m[f'm{p}'][ranks[i]]
    if len(active)==2:
        (i,a),(j,b)=active
        if a==b==1:return m['M11'][ranks[i],ranks[j]]
        if (a,b)==(2,1):return m['M21'][ranks[i],ranks[j]]
        if (a,b)==(1,2):return m['M21'][ranks[j],ranks[i]]
    raise ValueError(f'unneeded higher-order distinct moment requested: {counts}')


def all_distinct(q,m):
    """Rank-assignment sums for A B C D, including four seen-token joints."""
    v=m['m1'];s1=v.sum();s2=np.sum(v*v);s3=np.sum(v*v*v)
    # Six times elementary symmetric e3 with first rank excluded.
    a1=s1-v;a2=s2-v*v;a3=s3-v*v*v
    ordered_triples=a1**3-3*a1*a2+2*a3
    denominator=float(q@ordered_triples)
    first=float((q*v)@ordered_triples)
    remaining1=s1-v[:,None]-v[None,:]
    remaining2=s2-v[:,None]**2-v[None,:]**2
    pairs=remaining1**2-remaining2
    np.fill_diagonal(pairs,0.)
    other=float(np.sum(q[:,None]*v[None,:]**2*pairs))
    return denominator,np.array([first,other,other,other])


def pattern_joints(pattern,q,m):
    """Sums over distinct ranks, before the uniform rank-prior factor."""
    k=max(pattern)+1;d=len(q)
    if k==4:
        if tuple(pattern)!=(0,1,2,3):raise ValueError('only length-four all-distinct pattern supported')
        return all_distinct(q,m)
    ranks=[np.arange(d).reshape((1,)*i+(d,)+(1,)*(k-i-1)) for i in range(k)]
    shape=(d,)*k
    distinct=np.ones(shape,dtype=bool)
    for i in range(k):
        for j in range(i):distinct&=ranks[i]!=ranks[j]
    counts=np.zeros((k,k),dtype=int)
    for a,b in zip(pattern[:-1],pattern[1:]):counts[a,b]+=1
    first=np.broadcast_to(q[ranks[pattern[0]]],shape)
    def joint(matrix):
        value=first.copy()
        for row in matrix:
            value*=row_factor(row,ranks,m)
        return float(value[distinct].sum())
    denominator=joint(counts)
    numerator=[]
    for b in range(k):
        after=counts.copy();after[pattern[-1],b]+=1
        numerator.append(joint(after))
    return denominator,np.asarray(numerator)


def exhaustive_joints(kernel,alpha_index,depth_index,pattern):
    """Independent small-d enumeration using full original row evidence."""
    d=kernel.d;k=max(pattern)+1
    q=kernel.base_laws[alpha_index]
    denominator=0.;numerator=np.zeros(d)
    for assignment in itertools.permutations(range(d),k):
        ranks=list(assignment)+[r for r in range(d) if r not in assignment]
        path=np.array([ranks[x] for x in pattern])
        counts=np.zeros((d,d),dtype=int)
        for a,b in zip(path[:-1],path[1:]):counts[a,b]+=1
        logs=[kernel.log_evidence(row,alpha_index)[depth_index] for row in counts]
        probability=q[path[0]]*np.exp(sum(logs))
        denominator+=probability
        last=path[-1]
        before=logs[last]
        # Unseen labels are averaged across every remaining rank here;
        # that is equivalent to enumerating the remaining (d-k)! ranks.
        predictive=np.exp(kernel.log_predictive(counts[last],alpha_index)[depth_index])
        for observed in range(k):numerator[observed]+=probability*predictive[ranks[observed]]
        if d>k:numerator[k:]+=probability*predictive[ranks[k:]].sum()/(d-k)
    return denominator,numerator


def validate_small():
    checks=[]
    for d,depths in [(4,[0,1]),(5,[0,1])]:
        kernel=RankedLSARowEvidence(d,[.6,1.3],depths=depths,step=1/64,u_lo=-36,u_hi=34,term_cache_size=512)
        error=0.
        for ai in range(2):
            batch,_=row_moments(kernel,ai)
            for li in range(len(depths)):
                m={name:values[li] for name,values in batch.items()}
                for pattern in patterns(4):
                    denominator,numerators=pattern_joints(pattern,kernel.base_laws[ai],m)
                    brute,brute_next=exhaustive_joints(kernel,ai,li,pattern)
                    k=max(pattern)+1
                    pred=numerators/denominator
                    if d>k:pred=np.r_[pred,np.repeat((1-pred.sum())/(d-k),d-k)]
                    err=max(abs(denominator-brute),np.max(abs(pred-brute_next/brute)))
                    error=max(error,float(err))
        if error>1e-9:raise ValueError(f'exhaustive small-d comparison failed: {error}')
        checks.append(dict(d=d,depths=depths,alpha_nodes=[.6,1.3],patterns=15,max_error=error,passed=True))
    return checks


def calculate(d,order,*,step=1/32,bounds=(-32.,32.),depths=range(9)):
    started=time.monotonic()
    nodes,weights=leggauss(order);nodes=1+nodes/2;weights=weights/2
    depths=list(depths)
    kernel=RankedLSARowEvidence(d,nodes,depths=depths,step=step,u_lo=bounds[0],u_hi=bounds[1],term_cache_size=512)
    pats=patterns(4)
    denominators=np.zeros(len(pats));numerators=np.zeros((len(pats),4))
    row_checks=[]
    for ai,weight in enumerate(weights):
        batched,checks=row_moments(kernel,ai);row_checks.append(checks)
        for li in range(len(depths)):
            m={name:values[li] for name,values in batched.items()}
            for pi,pattern in enumerate(pats):
                den,num=pattern_joints(pattern,kernel.base_laws[ai],m)
                denominators[pi]+=weight*den/len(depths)
                numerators[pi,:len(num)]+=weight*num/len(depths)
        print(json.dumps(dict(stage='alpha',d=d,order=order,alpha_index=ai,seconds=time.monotonic()-started)),flush=True)
    records=[]
    for i,pattern in enumerate(pats):
        k=max(pattern)+1
        seen=numerators[i,:k]/denominators[i]
        unseen=(1-seen.sum())/(d-k)
        prediction=np.r_[seen,np.repeat(unseen,d-k)]
        if np.min(prediction)<=0 or abs(prediction.sum()-1)>1e-12:raise ValueError('invalid predictive distribution')
        records.append(dict(pattern=list(pattern),n_seen=k,seen_probs=seen.tolist(),unseen_probability=float(unseen),
            predictive_probs=prediction.tolist(),prefix_probability=float(denominators[i]/math.prod(range(d-k+1,d+1)))))
    return dict(d=d,alpha_order=order,depths=depths,step=step,bounds=list(bounds),seconds=time.monotonic()-started,
        records=records,row_checks=row_checks,cache=kernel.cache_info())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'output/shortprefix_shared_base_exact_2026-09-15')
    p.add_argument('--order',type=int,default=16)
    p.add_argument('--step',type=float,default=1/32)
    p.add_argument('--bounds',type=float,nargs=2,default=[-32.,32.])
    p.add_argument('--small-only',action='store_true')
    args=p.parse_args()
    if args.small_only:
        checks=validate_small()
        atomic_json(args.out/'small_d_validation.json',dict(checks=checks,passed=True,script_sha256=sha_file(__file__)))
        print(json.dumps(checks));return
    result=calculate(64,args.order,step=args.step,bounds=args.bounds)
    result.update(schema='shared-base-shortprefix-exact-v1',time=4,
        prior=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'),
        method='Deterministic observed-rank marginalization; existing sealed-store row integrals; uniform sequence-level depth mixture and full alpha prior.',
        script_sha256=sha_file(__file__))
    name=f'alpha{args.order}_step{round(1/args.step)}_bounds{args.bounds[0]:g}_{args.bounds[1]:g}.json'
    atomic_json(args.out/name,result)
    print(json.dumps(dict(stage='complete',path=str(args.out/name),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
