"""Validate deterministic t4 references and score every declared source path."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import rel_entr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
from eval_shared_base_models import atomic_json,atomic_npz,sha_file
from ranked_lsa import RankedLSARowEvidence
from shortprefix_shared_base_exact import canonical,patterns,row_moments,pattern_joints,exhaustive_joints
from shared_base_conditional_cohort import inputs,moments


def prediction_for_tokens(tokens,catalog):
    pattern=canonical(tokens)
    if len(tokens)!=4:raise ValueError('reference requires exactly four observed tokens')
    entry=catalog[pattern]
    seen=list(dict.fromkeys(map(int,tokens)))
    result=np.full(64,entry['unseen_probability'])
    result[seen]=entry['seen_probs']
    if min(result)<=0 or abs(result.sum()-1)>1e-12:raise ValueError('invalid mapped prediction')
    return result


def full_depth_small_check():
    checks=[]
    for d in [4,5]:
        kernel=RankedLSARowEvidence(d,[.6,1.3],depths=[0,1,4,8],step=1/64,u_lo=-36,u_hi=34,term_cache_size=512)
        worst=0.
        for ai in range(2):
            batched,_=row_moments(kernel,ai)
            for li in range(4):
                m={name:array[li] for name,array in batched.items()}
                for pattern in patterns(4):
                    den,num=pattern_joints(pattern,kernel.base_laws[ai],m)
                    brute,brute_num=exhaustive_joints(kernel,ai,li,pattern)
                    k=max(pattern)+1
                    pred=num/den
                    if d>k:pred=np.r_[pred,np.repeat((1-pred.sum())/(d-k),d-k)]
                    worst=max(worst,abs(den-brute),float(np.max(abs(pred-brute_num/brute))))
        if worst>1e-8:raise ValueError(f'small-d exact enumeration failed: {worst}')
        checks.append(dict(d=d,alpha_nodes=[.6,1.3],depths=[0,1,4,8],patterns=15,max_absolute_error=worst,passed=True))
    return checks


def summarize(raw,cohorts,out):
    files=[raw/'alpha16_step32_bounds-32_32.json',raw/'alpha32_step32_bounds-32_32.json',raw/'alpha32_step64_bounds-36_34.json']
    reports=[json.loads(p.read_text()) for p in files]
    code=ROOT/'code/scripts/shortprefix_shared_base_exact.py'
    for report in reports:
        if report['script_sha256']!=sha_file(code) or report['schema']!='shared-base-shortprefix-exact-v1' or report['time']!=4:
            raise ValueError('reference code or schema mismatch')
        if report['prior']!=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'):
            raise ValueError('reference prior changed')
        if [tuple(r['pattern']) for r in report['records']]!=patterns(4):raise ValueError('incomplete pattern catalog')
    discrepancies={}
    for name,a,b in [('alpha_refinement',reports[0],reports[1]),('row_refinement',reports[1],reports[2])]:
        discrepancies[name]=dict(
            max_probability_difference=max(float(np.max(abs(np.asarray(x['predictive_probs'])-y['predictive_probs']))) for x,y in zip(a['records'],b['records'])),
            max_prefix_probability_relative_difference=max(abs(x['prefix_probability']-y['prefix_probability'])/y['prefix_probability'] for x,y in zip(a['records'],b['records'])))
    normalization=[abs(sum(r['prefix_probability']*math.prod(range(64-r['n_seen']+1,65)) for r in report['records'])-1) for report in reports]
    if max(v for d in discrepancies.values() for v in d.values())>1e-7 or max(normalization)>1e-7:
        raise ValueError(f'quadrature validation failed: {discrepancies}; {normalization}')
    small=full_depth_small_check()
    catalog={tuple(r['pattern']):r for r in reports[-1]['records']}
    reference=dict(schema='shared-base-shortprefix-reference-catalog-v1',time=4,prior=reports[-1]['prior'],
        records=reports[-1]['records'],passed=True,
        validation=dict(refinements=discrepancies,all_pattern_probability_sum_errors=normalization,small_d_exhaustive=small,
            row_marginal_max_errors=[max(max(x.values()) for x in report['row_checks']) for report in reports]),
        provenance=dict(exact_script_sha256=sha_file(code),summary_script_sha256=sha_file(__file__),
            numerical_reports=[dict(path=str(p.resolve()),sha256=sha_file(p)) for p in files]))
    atomic_json(out/'shortprefix_reference.json',reference)
    rows=[];receipts=[]
    for depth in [0,1,4,8]:
        folder=cohorts/f'a1_L{depth}'
        cohort=json.loads((folder/'cohort.json').read_text())
        if cohort['alpha']!=1 or cohort['depth']!=depth:raise ValueError('condition mismatch')
        n=cohort['n_sequences']
        bayes_regret=np.zeros(n);tf_regret=np.zeros((n,2));predictions=np.zeros((n,64))
        mapped_refinement=np.zeros((n,2))
        pattern_counts={}
        for index in range(n):
            tokens,oracle,models,identity=inputs(folder,index,4)
            bayes=prediction_for_tokens(tokens,catalog)
            predictions[index]=bayes
            pattern=canonical(tokens)
            label=''.join(map(str,pattern));pattern_counts[label]=pattern_counts.get(label,0)+1
            bayes_regret[index]=np.sum(rel_entr(oracle,bayes))/np.log(2)
            for j,name in enumerate(['d64','d64_seed1']):
                tf_regret[index,j]=np.sum(rel_entr(oracle,models[name]['prediction']))/np.log(2)
            for j,report in enumerate(reports[:2]):
                compare=prediction_for_tokens(tokens,{tuple(r['pattern']):r for r in report['records']})
                mapped_refinement[index,j]=np.sum(rel_entr(oracle,compare))/np.log(2)-bayes_regret[index]
        if np.max(abs(mapped_refinement))>1e-6:raise ValueError('numerical uncertainty is material to scored regrets')
        for j,name in enumerate(['d64','d64_seed1']):
            rows.append(dict(alpha=1.,depth=depth,time=4,model=name,n_sequences=n,
                bayes_regret=moments(bayes_regret),transformer_regret=moments(tf_regret[:,j]),
                transformer_minus_bayes=moments(tf_regret[:,j]-bayes_regret),
                maximum_refinement_change_in_individual_regret=float(np.max(abs(mapped_refinement)))))
        arrays=folder/'shortprefix_t4_predictions.npz'
        atomic_npz(arrays,source_indices=np.arange(n),bayes_predictive_probs=predictions,bayes_regret_bits=bayes_regret,
            transformer_regret_bits=tf_regret,transformer_minus_bayes_bits=tf_regret-bayes_regret[:,None])
        receipts.append(dict(alpha=1.,depth=depth,n_sequences=n,pattern_counts=pattern_counts,
            cohort_sha256=sha_file(folder/'cohort.json'),array_path=str(arrays.resolve()),array_sha256=sha_file(arrays)))
    summary=dict(schema='shared-base-shortprefix-paired-summary-v1',time=4,rows=rows,
        reference='Full original training prior, deterministic rank marginalization; selected alpha/depth are hidden.',
        statistic='Mean KL(true next-token transition row || predictor), in bits; standard errors over independent whole sequences.',
        every_declared_sequence_retained=True,reference_validation_passed=True,
        provenance=dict(reference_catalog_sha256=sha_file(out/'shortprefix_reference.json'),summary_script_sha256=sha_file(__file__),cohorts=receipts))
    atomic_json(out/'time4_summary.json',summary)
    print(json.dumps(dict(refinements=discrepancies,small_d=small,rows=rows)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'output/shortprefix_shared_base_exact_2026-09-15')
    p.add_argument('--cohorts',type=Path,default=ROOT/'output/shared_base_conditional_curves_2026-09-15')
    p.add_argument('--out',type=Path,default=ROOT/'results/transformer/shared_base_conditional_curves')
    args=p.parse_args();summarize(args.raw,args.cohorts,args.out)


if __name__=='__main__':main()
