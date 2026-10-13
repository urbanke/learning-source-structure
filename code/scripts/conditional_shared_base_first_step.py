"""First prediction via exact observed-rank enumeration and existing row integrals.

After one token, the existing posterior is symmetric in all unseen labels.
Enumerate the d possible ranks of the seen label and let PrefixPosterior
average the unseen assignments exactly. This removes MCMC at this time only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC,PrefixPosterior
from shared_base_chain_runner import _fingerprint
from eval_shared_base_models import atomic_json,sha_file


def prediction(d,order,*,depths=range(9),step=1/32,bounds=(-32.,32.)):
    started=time.monotonic()
    nodes,weights=leggauss(order);nodes=1+nodes/2;weights=weights/2
    kernel=RankedLSARowEvidence(d,nodes,depths=depths,step=step,u_lo=bounds[0],u_hi=bounds[1],term_cache_size=16384)
    sampler=SharedBaseMCMC(d,kernel,weights)
    target=PrefixPosterior(sampler,np.array([0]))
    result=np.zeros(d);normalizer=0.;max_error=0.
    for a,w in enumerate(weights):
        for rank in range(d):
            ranks=np.arange(d);ranks[0],ranks[rank]=ranks[rank],ranks[0]
            state=target.evaluate(a,ranks)
            predictive,posterior_depth,error=target.predict(state)
            # Every seen rank initially has probability1/d. The observed
            # token also has prior marginal1/d, so these constants cancel.
            weight=w*np.exp(state.log_likelihood)
            result+=weight*predictive;normalizer+=weight;max_error=max(max_error,error)
            if not np.allclose(posterior_depth,sampler.depth_prior,rtol=0,atol=2e-14):raise ValueError('first token altered depth prior')
    if abs(normalizer-1)>1e-12 or abs(result.sum()-1)>1e-7:raise ValueError('raw first-step normalization failed')
    if np.max(np.abs(result[1:]-result[1]))>1e-14:raise ValueError('unseen-label symmetry failed')
    return dict(probs=result.tolist(),seen_probability=float(result[0]),unseen_probability=float(result[1]),
                normalizer=normalizer,max_conditional_sum_error=max_error,seconds=time.monotonic()-started,
                d=d,alpha_order=order,depths=list(depths),row_step=step,row_bounds=list(bounds),
                numerical_problem=_fingerprint(sampler,np.array([0])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/transformer/conditional_shared_base/first_step.json');a=p.parse_args()
    reports=[prediction(64,16),prediction(64,32),prediction(64,32,step=1/64,bounds=(-36.,34.))]
    errors=dict(alpha_refinement=float(np.max(np.abs(np.asarray(reports[0]['probs'])-reports[1]['probs']))),
                row_refinement=float(np.max(np.abs(np.asarray(reports[1]['probs'])-reports[2]['probs']))))
    result=dict(schema='shared-base-first-step-exact-ranks-v1',time=1,prior=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'),
                reports=reports,checks=errors,passed=max(errors.values())<1e-7,script_sha256=sha_file(__file__),
                interpretation='Same prediction at t=1 under every evaluation condition; parameters are hidden. Deterministic rank marginalization; alpha and row integration validated by refinement.')
    atomic_json(a.out,result)
    print({k:result[k] for k in ('time','checks','passed')});print(reports[-1]['seen_probability'],reports[-1]['unseen_probability'])
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
