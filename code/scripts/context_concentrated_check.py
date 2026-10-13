"""Finite-alphabet checks for concentrated Zipf row priors (alpha>1).

Fixed-k row redundancies use the depth-one Bayes integral with direct adaptive
quadrature. Rows and sampling uniforms are coupled across alphabet sizes.
A separate coupled simulation checks that the random normalizer does not
concentrate for alpha>1, at depths one and three. This is not a chain test.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.special import gammaln,zeta
from context_row_check import log_mixture,log_mixture_adaptive
from master_curve import r1_geometric


def summarize(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(x.mean()),'se':float(x.std(ddof=1)/math.sqrt(len(x)))}


def compositions(k,d):
    if d==1:
        yield (k,);return
    for i in range(k+1):
        for tail in compositions(k-i,d-1):yield (i,)+tail


def validate_quadrature():
    rng=np.random.default_rng(1207);checks=[];maxerr=0.;production_maxerr=0.
    for d in (16,128,1024):
        q=np.full(d,1/d);u=np.arange(-16.,27.,.004);kernel=d*np.log1p(np.exp(u)/d)
        for k in (1,8,64,512):
            counts=rng.multinomial(k,q);idx=np.flatnonzero(counts)
            exact=float(gammaln(d)+gammaln(counts+1).sum()-gammaln(k+d))
            value,diagnostic=log_mixture_adaptive(counts,q)
            old=log_mixture(idx,counts[idx],k,q,u,kernel)
            error=abs(value-exact);maxerr=max(maxerr,error);production_maxerr=max(production_maxerr,abs(old-exact))
            assert error<2e-9,(d,k,error)
            checks.append(dict(d=d,k=k,adaptive_error_nats=error,production_error_nats=old-exact,**diagnostic))
    mass=[]
    q=np.array([.6,.3,.1])
    for k in range(5):
        total=0.
        for c in compositions(k,3):
            lm,_=log_mixture_adaptive(c,q)
            total+=math.exp(gammaln(k+1)-gammaln(np.asarray(c)+1).sum()+lm)
        assert abs(total-1)<2e-10
        mass.append(dict(d=3,k=k,total_probability=total))
    return dict(uniform_closed_form=checks,maximum_adaptive_error_nats=maxerr,
        maximum_production_error_nats=production_maxerr,nonuniform_total_probability=mass)


def row_study(ds,ks,reps):
    records=[];maximum_d=max(ds);maximum_k=max(ks)
    for alpha in (1.5,2.0):
        curves=np.zeros((reps,len(ds),len(ks)))
        diagnostics=[]
        for r in range(reps):
            rng=np.random.default_rng([20260906,r,int(100*alpha),1])
            weights=rng.exponential(size=maximum_d)
            uniforms=rng.random(maximum_k)
            for i,d in enumerate(ds):
                q=np.arange(1,d+1,dtype=float)**(-alpha);q/=q.sum()
                p=q*weights[:d];p/=p.sum();cdf=np.cumsum(p);cdf[-1]=1.
                labels=np.searchsorted(cdf,uniforms,side='right')
                for j,k in enumerate(ks):
                    counts=np.bincount(labels[:k],minlength=d)
                    lm,diag=log_mixture_adaptive(counts,q)
                    observed=np.flatnonzero(counts)
                    curves[r,i,j]=(counts[observed]@np.log(p[observed])-lm)/math.log(2)
                    diagnostics.append(diag)
        for i,d in enumerate(ds):
            q=np.arange(1,d+1,dtype=float)**(-alpha);q/=q.sum()
            comparison=np.array([sum(r1_geometric(float(k*qj)) for qj in q) for k in ks])
            a=curves[:,i,:];delta=a-curves[:,-1,:]
            records.append(dict(alpha=alpha,d=d,k=ks,reps=reps,
                redundancy_mean=a.mean(0).tolist(),redundancy_se=(a.std(0,ddof=1)/math.sqrt(reps)).tolist(),
                poisson_comparison_bits=comparison.tolist(),
                paired_difference_to_largest_d_mean=delta.mean(0).tolist(),
                paired_difference_to_largest_d_se=(delta.std(0,ddof=1)/math.sqrt(reps)).tolist(),
                sum_q_squared=float(q@q),largest_d=maximum_d,
                max_relative_quadrature_error=max(x['relative_quadrature_error'] for x in diagnostics),
                max_relative_tail_bound=max(x['relative_tail_bound'] for x in diagnostics)))
            print('row',alpha,d,'terminal',a.mean(0)[-1],'SE',a.std(0,ddof=1)[-1]/math.sqrt(reps),'Poisson',comparison[-1],flush=True)
    return records


def normalizer_study(ds,draws):
    alphas=(.5,1.,1.5,2.);rows=[];rng=np.random.default_rng(20260907)
    values={(a,L,d):[] for a in alphas for L in (1,3) for d in ds}
    q={(a,d):(np.arange(1,d+1,dtype=float)**(-a)) for a in alphas for d in ds}
    q={key:val/val.sum() for key,val in q.items()}
    for start in range(0,draws,128):
        count=min(128,draws-start);w1=rng.exponential(size=(count,max(ds)))
        w3=w1*rng.exponential(size=w1.shape)*rng.exponential(size=w1.shape)
        for a in alphas:
            for d in ds:
                for L,w in ((1,w1),(3,w3)):values[(a,L,d)].extend((w[:,:d]@q[(a,d)]).tolist())
    for a in alphas:
        for L in (1,3):
            for d in ds:
                x=np.asarray(values[(a,L,d)]);variance=float(x.var(ddof=1))
                # Empirical influence-function SE for the sample variance.
                variance_se=float(((x-x.mean())**2).std(ddof=1)/math.sqrt(draws))
                exact=(2**L-1)*float(q[(a,d)]@q[(a,d)])
                limit=None if a<=1 else float((2**L-1)*zeta(2*a,1)/zeta(a,1)**2)
                record=dict(alpha=a,L=L,d=d,draws=draws,normalizer_mean=float(x.mean()),
                    normalizer_mean_se=float(x.std(ddof=1)/math.sqrt(draws)),
                    normalizer_variance=variance,normalizer_variance_se=variance_se,
                    exact_finite_variance=exact,infinite_alphabet_variance=limit)
                if a>1:
                    beta=1/a;hmass=float(np.sum(np.arange(1,d+1,dtype=float)**(-a)))
                    inverse=x**(-beta);raw_inverse=inverse/(hmass**beta)
                    record.update(dict(negative_moment_beta=beta,
                        normalized_inverse_moment_mean=float(inverse.mean()),
                        normalized_inverse_moment_se=float(inverse.std(ddof=1)/math.sqrt(draws)),
                        unnormalized_inverse_moment_mean=float(raw_inverse.mean()),
                        unnormalized_inverse_moment_se=float(raw_inverse.std(ddof=1)/math.sqrt(draws)),
                        unnormalized_mass=hmass))
                    if a==2 and L==1:
                        record['infinite_normalized_inverse_moment']=math.pi**1.5/(2*math.sqrt(6))
                        record['infinite_unnormalized_inverse_moment']=math.sqrt(math.pi)/2
                rows.append(record)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    ds=[16,64] if args.smoke else [32,128,512,2048]
    ks=[1,4,16] if args.smoke else [1,4,16,64,256,1024]
    reps=8 if args.smoke else 64;draws=256 if args.smoke else 16384
    out=dict(se_ddof=1,seed_protocol='row default_rng([20260906,replication,100*alpha,1]); normalizers default_rng(20260907)',
        coupling='The same exponential prefixes and sampling uniforms are used across d at fixed alpha; normalizers use common weight prefixes and layers.',
        scope='iid rows only; fixed finite sample counts; no Markov visits or stationary-start experiment',
        quadrature_validation=validate_quadrature(),rows=row_study(ds,ks,reps),normalizers=normalizer_study(ds,draws))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')
    lines=['alpha\td\tk\trow_redundancy_bits\tSE_bits\tpoisson_comparison_bits\tpaired_difference_to_largest_d_bits\tpaired_SE_bits']
    for row in out['rows']:
        for j,k in enumerate(row['k']):
            lines.append('\t'.join(map(str,[row['alpha'],row['d'],k,row['redundancy_mean'][j],row['redundancy_se'][j],row['poisson_comparison_bits'][j],row['paired_difference_to_largest_d_mean'][j],row['paired_difference_to_largest_d_se'][j]])))
    args.out.with_suffix('.tsv').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
