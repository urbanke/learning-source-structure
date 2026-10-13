"""Finite-occupation count predictions and paired actual-chain measurements.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/crossing_finite_count_prediction.py

The archive selects checkpoint windows only. No cost or crossing coefficient
is fitted. The predicted crossing is the last located zero of the sample
mean gap. Counts at each checkpoint are independent conditional on the latent
table and shared initial state. Actual-chain checkpoints share one path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from context_stationary_check import stationary_checked
from crossing_count_prediction import ROOT, sha, relative, moments, independent_l1_bits
from takeover_experiment import engage_anchor_mode
from tilted_row_reference import DEFAULT_STORE


class Occupations:
    """Binary matrix powers and averages; no mixing approximation."""
    def __init__(self,T,maximum):
        self.T=T
        self.powers=[T]
        self.averages=[np.eye(len(T))]
        for _ in range(1,int(maximum).bit_length()):
            P,A=self.powers[-1],self.averages[-1]
            self.averages.append(.5*(A+P@A))
            self.powers.append(P@P)

    def at(self,q,t):
        state=q.copy()
        total=np.zeros_like(q)
        for j in range(int(t).bit_length()):
            if int(t)&(1<<j):
                total+=(1<<j)*(state@self.averages[j])
                state=state@self.powers[j]
        return total/t,state

    def check(self,q):
        total=np.zeros_like(q)
        state=q.copy()
        errors=[]
        for t in range(1,32):
            total+=state
            state=state@self.T
            r,v=self.at(q,t)
            errors.extend((abs(r-total/t).max(),abs(v-state).max()))
        return float(max(errors))


def last_mean_crossing(ns,gaps):
    means=np.mean(gaps,axis=0)
    roots=[]
    for j in range(1,len(ns)):
        if means[j-1]<=0<means[j]:
            weight=-means[j-1]/(means[j]-means[j-1])
            roots.append(math.exp(math.log(ns[j-1])+weight*math.log(ns[j]/ns[j-1])))
    if means[-1]<=0:
        status="nonpositive_at_end"
        root=None
    elif not roots:
        status="positive_at_every_check"
        root=None
    else:
        status="located"
        root=roots[-1]
    return dict(value=root,status=status,observed_upcrossings=len(roots),
                horizon_tokens=int(ns[-1]),minimum_mean_gap=float(min(means)))


def chain_from_uniforms(T,initial,uniforms):
    cdf=np.cumsum(T,axis=1)
    cdf[:,-1]=1.
    x=np.empty(len(uniforms)+1,dtype=np.int32)
    x[0]=initial
    for i,u in enumerate(uniforms):
        x[i+1]=np.searchsorted(cdf[x[i]],u,side="right")
    return x


def run_cell(args,L,alpha,d):
    from lsa.codelength import default_l_max,depth_averaged_codelength_profiles,profile_of
    started=time.monotonic()
    archive_path=args.archive/f"d{d}_a{alpha}_L{L}"/f"summary_d{d}_a{alpha}_L{L}.json"
    archive=json.loads(archive_path.read_text())
    assert (archive["d"],archive["alpha"],archive["L"])==(d,alpha,L)
    center=archive["mean_lambda_crossing"]
    assert center is not None and center>1 and archive["n_censored"]==0
    ns=np.asarray(sorted({max(2,int(round(center*f))) for f in args.factors}),dtype=np.int64)
    ts=ns-1
    assert min(ts)>=31
    q=np.arange(1,d+1,dtype=float)**(-alpha)
    q/=q.sum()
    shape=(args.reps,len(ts),d,d)
    stored={name:np.zeros(shape,dtype=np.int32) for name in ("count_transition_counts","chain_transition_counts")}
    for name in ("count_frequency_counts","chain_frequency_counts"):
        stored[name]=np.zeros(shape[:3],dtype=np.int32)
    stored.update(transition_matrices=np.zeros((args.reps,d,d)),
        stationary_laws=np.zeros((args.reps,d)),normalized_row_sums=np.zeros((args.reps,d)),
        expected_occupation_laws=np.zeros(shape[:3]),initial_states=np.zeros(args.reps,dtype=np.int32),
        chain_paths=np.zeros((args.reps,int(ns[-1])),dtype=np.int32))
    pending={name:[] for name in ("count","chain")}
    all_profiles={(),(1,),(2,),(1,1)}
    max_direct=max_flow=max_stationarity=0.
    for rep in range(args.reps):
        prefix=[args.seed,d,L,round(alpha*100),rep]
        G=np.random.default_rng(prefix+[77]).exponential(size=(d,d,L)).prod(axis=2)
        S=G@q
        T=G*q[None,:]/S[:,None]
        pi,diag=stationary_checked(T)
        max_stationarity=max(max_stationarity,diag["residual_l1"])
        initial=int(np.random.default_rng(prefix+[78]).choice(d,p=q))
        occupations=Occupations(T,max(int(ts[-1]),31))
        max_direct=max(max_direct,occupations.check(q))
        x=chain_from_uniforms(T,initial,np.random.default_rng(prefix+[79]).random(int(ts[-1])))
        stored["transition_matrices"][rep]=T
        stored["stationary_laws"][rep]=pi
        stored["normalized_row_sums"][rep]=S
        stored["initial_states"][rep]=initial
        stored["chain_paths"][rep]=x
        chain_counts=np.zeros((d,d),dtype=np.int64)
        previous=0
        rep_profiles={"count":[],"chain":[]}
        for j,(n,t) in enumerate(zip(ns,ts)):
            r,end=occupations.at(q,int(t))
            assert r.min()>0 and abs(r.sum()-1)<1e-10
            max_flow=max(max_flow,float(abs(t*(r@T-r)-(end-q)).max()))
            stored["expected_occupation_laws"][rep,j]=r
            pair_law=(r[:,None]*T).ravel()
            pair_law/=pair_law.sum()
            count_counts=np.random.default_rng(prefix+[80,int(t)]).multinomial(int(t),pair_law).reshape(d,d)
            flat=x[previous:int(t)].astype(np.int64)*d+x[previous+1:int(n)]
            chain_counts+=np.bincount(flat,minlength=d*d).reshape(d,d)
            previous=int(t)
            flow=np.bincount([initial],minlength=d)-np.bincount([x[int(t)]],minlength=d)
            assert np.array_equal(chain_counts.sum(1)-chain_counts.sum(0),flow)
            for name,counts in (("count",count_counts),("chain",chain_counts)):
                fcounts=counts.sum(0)
                fcounts[initial]+=1
                assert counts.sum()==t and fcounts.sum()==n
                if name=="chain":
                    assert np.array_equal(fcounts,np.bincount(x[:int(n)],minlength=d))
                stored[name+"_transition_counts"][rep,j]=counts
                stored[name+"_frequency_counts"][rep,j]=fcounts
                fp=profile_of(fcounts)
                row_mult=Counter(profile_of(row) for row in counts if row.sum())
                rep_profiles[name].append((fp,row_mult))
                all_profiles.add(fp)
                all_profiles.update(row_mult)
        for name in pending:
            pending[name].append(rep_profiles[name])
    preparation_seconds=time.monotonic()-started
    print(f"L{L} a{alpha:g} d{d}: prepared16 paired tables/counts/chains and {len(all_profiles)} profiles in {preparation_seconds:.2f}s",flush=True)
    eval_start=time.monotonic()
    evaluated=depth_averaged_codelength_profiles({p:p for p in all_profiles},d=d,
        l_max=default_l_max(d),jobs=args.jobs,tables_source="universal",universal_path=args.store)
    evaluation_seconds=time.monotonic()-eval_start
    zero_error=abs(evaluated[()].log2_q_avg)
    single_error=abs(-evaluated[(1,)].log2_q_avg-math.log2(d))
    two_mass=d*2**evaluated[(2,)].log2_q_avg+d*(d-1)*2**evaluated[(1,1)].log2_q_avg
    l1_error=max(abs(-v.log2_q_by_depth[0]-independent_l1_bits(p,d)) for p,v in evaluated.items())
    assert max(zero_error,single_error,abs(two_mass-1),l1_error)<1e-7
    assert max_direct<1e-12 and max_flow<1e-7 and max_stationarity<1e-10
    summaries={}
    for name in pending:
        f=np.zeros((args.reps,len(ns)))
        c=np.zeros_like(f)
        for rep in range(args.reps):
            for j,(fp,row_mult) in enumerate(pending[name][rep]):
                f[rep,j]=-evaluated[fp].log2_q_avg
                c[rep,j]=math.log2(d)-math.fsum(mult*evaluated[p].log2_q_avg for p,mult in row_mult.items())
        gap=f-c
        stored[name+"_frequency_codelength"]=f
        stored[name+"_context_codelength"]=c
        stored[name+"_gap"]=gap
        summaries[name]=dict(frequency_codelength=moments(f),context_codelength=moments(c),
            gap=moments(gap),gap_by_repetition=gap.tolist(),mean_curve_crossing=last_mean_crossing(ns,gap))
    paired=stored["count_gap"]-stored["chain_gap"]
    args.raw_dir.mkdir(parents=True,exist_ok=True)
    raw_path=args.raw_dir/f"L{L}_a{alpha:g}_d{d}.npz"
    np.savez_compressed(raw_path,**stored,q=q,tokens=ns,transitions=ts)
    result=dict(d=d,alpha=alpha,L=L,repetitions=args.reps,seed=args.seed,l_max=default_l_max(d),
        tokens=ns.tolist(),transitions=ts.tolist(),archived_mean_curve_crossing=center,
        archive_path=relative(archive_path),archive_sha256=sha(archive_path),grid_factors=args.factors,
        count=summaries["count"],chain=summaries["chain"],paired_gap_difference=moments(paired),
        paired_gap_sample_covariance=[float(np.cov(stored["count_gap"][:,j],stored["chain_gap"][:,j],ddof=1)[0,1]) for j in range(len(ns))],
        se_ddof=1,raw_path=relative(raw_path),raw_sha256=sha(raw_path),profile_count=len(all_profiles),
        preparation_seconds=preparation_seconds,evaluation_seconds=evaluation_seconds,
        elapsed_seconds=time.monotonic()-started,
        validation=dict(maximum_direct_occupation_error=max_direct,maximum_expected_flow_error=max_flow,
            maximum_stationary_residual_l1=max_stationarity,zero_count_error_bits=zero_error,
            single_count_error_bits=single_error,two_token_total_probability=two_mass,
            maximum_independent_depth_one_error_bits=l1_error))
    raw_path.with_suffix(".json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"L{L} a{alpha:g} d{d}: evaluator {evaluation_seconds:.2f}s, total {result['elapsed_seconds']:.2f}s; roots count={summaries['count']['mean_curve_crossing']['value']} chain={summaries['chain']['mean_curve_crossing']['value']} archive={center}",flush=True)
    return result


def self_test():
    pi=np.array([.2,.3,.5])
    q=np.array([.6,.3,.1])
    T=np.repeat(pi[None,:],3,axis=0)
    obj=Occupations(T,1001)
    assert obj.check(q)<1e-14
    for t in [1,2,17,1001]:
        r,end=obj.at(q,t)
        assert np.max(abs(r-(q+(t-1)*pi)/t))<1e-14
    perfect=np.array([[-2.,-1.,1.,2.]])
    assert abs(last_mean_crossing(np.array([10,20,40,80]),perfect)["value"]-math.sqrt(800))<1e-12
    return dict(status="passed",checks=["Binary occupation versus direct iteration and exact rank-one formula",
        "Mean-curve log-length interpolation on a hand-calculated crossing"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths",nargs="+",type=int,default=[2,3])
    p.add_argument("--alphas",nargs="+",type=float,default=[.75,1.,1.25])
    p.add_argument("--ds",nargs="+",type=int,default=[128,256,512])
    p.add_argument("--reps",type=int,default=16)
    p.add_argument("--seed",type=int,default=202609072)
    p.add_argument("--factors",nargs="+",type=float,default=[.25,.5,.75,.875,1.,1.125,1.25,1.5,2.,4.])
    p.add_argument("--jobs",type=int,default=1)
    p.add_argument("--store",type=Path,default=DEFAULT_STORE)
    p.add_argument("--lsa-src",type=Path,default=ROOT.parent/"layered-simplex-architecture/src")
    p.add_argument("--archive",type=Path,default=ROOT/"output/takeover")
    p.add_argument("--raw-dir",type=Path,default=ROOT/"output/crossing_finite_count_prediction")
    p.add_argument("--out",type=Path,default=ROOT/"results/mechanism/crossing_finite_count_prediction.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    validation=self_test()
    if args.self_test:
        print(json.dumps(validation,indent=2));return
    assert args.reps>=2
    args.raw_dir=args.raw_dir.resolve()
    engage_anchor_mode(str(args.store))
    sys.path.insert(0,str(args.lsa_src.resolve()))
    start=time.monotonic()
    cells=[]
    for L,alpha,d in itertools.product(args.depths,args.alphas,args.ds):
        cells.append(run_cell(args,L,alpha,d))
        result=dict(schema_version=1,complete=len(cells)==len(args.depths)*len(args.alphas)*len(args.ds),
            target="Finite-occupation independent-pair count approximation to the expected gap of the deployed symmetric depth-mixture codes; paired actual-chain control.",
            formula="r_t=(1/t)sum_{s=0}^{t-1}qT^s; M~Multinomial(t,vec(r_t*T)); X0~q independently; N_b=sum_a M_ab+1{X0=b}; gap=ell_est(N)-log2(d)-sum_a ell_est(M_a).",
            qualifications=["Finite expected occupations retain the initial transient and match every actual transition-count mean; only the count dependence is approximated.",
                "The count law is unconditional on the realized initial state; count samples are independent of X0 given T. Actual paths start at that shared X0.",
                "Count checkpoints are independent conditional on the table and initial state, because their pair laws vary with time.",
                "Paired count and actual-chain samples share the same latent table and initial state; sampling streams are separate.",
                "The archived mean crossing selects ten evaluation lengths only; no coefficient is fitted. Roots interpolate the sample-mean gap linearly in log token count.",
                "Only located crossings within the evaluated window are reported; no final crossing at every later length is certified.",
                "SE uses sample variance ddof=1. Archived16-seed observations use independent tables and paths.",
                "No replacement of actual stationary probabilities by q, no Bayes/symmetric-code substitution, and no asymptotic scaling claim."],
            seed_protocol="default_rng([seed,d,L,round(100*alpha),rep,stream]); streams77 weights,78 initial,79 chain uniforms; count checkpoint has stream80 followed by t.",
            numerical_method="Binary stochastic-matrix powers and average powers compute expected occupations; exact deployed profile-mixture evaluator with sealed moment store.",
            script_sha256=sha(Path(__file__)),helper_sha256=sha(Path(__file__).with_name("crossing_count_prediction.py")),
            evaluator_sha256=sha(args.lsa_src/"lsa/codelength.py"),command_argv=sys.argv,
            validation=validation,cells=cells,elapsed_seconds=time.monotonic()-start)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Completed finite-occupation campaign in {time.monotonic()-start:.2f}s",flush=True)


if __name__=="__main__":
    main()
