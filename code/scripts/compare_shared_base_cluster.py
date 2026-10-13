"""Build the 12-sequence pilot comparison from saved cluster merge receipts.

Uses the actual three-family sequence mixture and the existing transformer
scores. No specialist-transformer surrogate, retraining, or automatic paper
update. A complete, precise SMC calculation is still a preliminary numerical
reference until independent effort and quadrature checks have been reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
from shared_base_cluster import read_plan, validate_merge_receipt
from shared_base_short_evidence import SharedBaseShortEvidence

PLAN = ROOT/'code/transformer/configs/shared_base_cluster_plan_2026-09-16.json'
LN2 = math.log(2)
FAMILIES = ['product', 'shared_base', 'uniform_base']
TIMES = [1, 4, 16, 64, 256, 1024]
MODELS = ['pooled_seed0', 'pooled_seed1']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def mixture_loss(logs, prior):
    """Cumulative bits after the supplied first token, whose marginal is 1/64."""
    logs, prior = np.asarray(logs,dtype=float), np.asarray(prior,dtype=float)
    require(logs.shape == prior.shape == (3,) and np.isfinite(logs).all()
            and np.isfinite(prior).all() and np.all(logs <= -math.log(64)+1e-10)
            and np.all(prior>0) and abs(float(prior.sum())-1)<1e-12, 'Invalid mixture inputs')
    marginal = float(logsumexp(logs+np.log(prior)))
    return -(marginal+math.log(64))/LN2, np.exp(logs+np.log(prior)-marginal)


def stage_estimates(receipt, context):
    """Use recomputed raw-record statistics, never a saved precision boolean."""
    require(receipt.get('schema') == 'shared-base-cluster-merge-v1'
            and receipt.get('plan_sha256') == context['plan_sha256']
            and receipt.get('stage') == context['stage']
            and receipt.get('stage_config') == context['config'], 'Merge receipt identity mismatch')
    require(isinstance(receipt.get('complete'), bool), 'Receipt completeness must be Boolean')
    if not receipt['complete']:
        return None, None  # An incomplete preview supplies no numerical estimates.
    checked = validate_merge_receipt(context, receipt)
    values = {}
    for key in context['config']['case_ids']:
        stats = checked['cases'][key]['statistics']
        require(receipt['cases'][key].get('statistics') == stats,
                'Saved case statistics disagree with retained raw repetitions')
        values[key] = (stats['log_mean_evidence_nats'], stats['loss_mc_se_bits_delta_method'])
    return values, checked['gate']


def validate_comparison(plan, catalog):
    """Validate pairing, time convention, and the fixed equal-family prior."""
    comparison = plan['comparison']
    require(comparison['prefixes'] == TIMES, 'Unexpected prediction times')
    require(comparison['family_order'] == FAMILIES, 'Wrong component ordering')
    prior = np.asarray(comparison['family_prior'], dtype=float)
    require(prior.shape == (3,) and np.allclose(prior, 1/3, rtol=0, atol=1e-15),
            'The trained three-family mixture requires equal family weights')
    require(comparison['loss_convention'] == 'First token supplied/unscored. At time t, predict tokens[1:t+1].',
            'Comparison loss convention differs from supplied-first-token training')
    lookup = {c['id']: c for c in plan['cases']}
    require(len(lookup) == len(plan['cases']), 'Duplicate source case identity')
    case_ids = list(comparison['cases'])
    require(len(case_ids) == 12 and all(c in lookup for c in case_ids)
            and all(sum(lookup[c].get('family') == f for c in case_ids) == 4 for f in FAMILIES),
            'The fixed balanced 12-sequence pilot is required')
    for time in TIMES[2:]:
        stage = plan['stages'][f'pilot{time}']
        require(set(stage['case_ids']) == set(case_ids) and stage['n_tokens'] == time+1
                and stage['purpose'] == 'repeatability', 'Pilot stage is not paired with the comparison')
    for cid in case_ids:
        values = comparison['cases'][cid]
        for key in ('product_log_marginals', 'uniform_base_log_marginals'):
            array = np.asarray(values[key], dtype=float)
            require(array.shape == (len(TIMES),) and np.isfinite(array).all()
                    and np.all(array <= -math.log(64)+1e-10)
                    and np.all(np.diff(array) <= 1e-10), 'Invalid component evidence curve')
        short = np.asarray(values['shared_base_short_log_marginals'], dtype=float)
        expected = np.array([catalog.log_joint(lookup[cid]['tokens'][:time+1]) for time in TIMES[:2]])
        require(short.shape == (2,) and np.allclose(short, expected, rtol=0, atol=1e-10),
                'Saved short likelihoods disagree with the independent catalog')
        scores = values['transformer_cumulative_loss_bits']
        require(set(scores) == set(MODELS), 'Both trained transformer seeds are required')
        for values in scores.values():
            array = np.asarray(values, dtype=float)
            require(array.shape == (len(TIMES),) and np.isfinite(array).all()
                    and np.all(array >= 0) and np.all(np.diff(array) >= -1e-10),
                    'Invalid transformer cumulative loss curve')
    return lookup, case_ids


def compare(plan_path, receipt_paths, diagnostic=False):
    base_context = read_plan(plan_path, 'pilot16')
    plan = base_context['plan']
    catalog = SharedBaseShortEvidence()
    lookup, case_ids = validate_comparison(plan, catalog)
    comparison=plan['comparison']; times=comparison['prefixes']
    families=comparison['family_order'];prior=comparison['family_prior']
    receipts={}; provenance={}
    for path in receipt_paths:
        raw = Path(path).read_bytes()
        data=json.loads(raw)
        stage=data['stage']
        require(stage not in receipts,'Duplicate stage receipt')
        require(stage in plan['stages'] and plan['stages'][stage]['purpose']=='repeatability',
                'Only pilot-stage receipts belong in the comparison')
        context = read_plan(plan_path, stage)
        require(context['plan_sha256'] == base_context['plan_sha256'], 'Plan changed during comparison')
        values, gate = stage_estimates(data, context)
        receipts[stage]=(gate,values)
        provenance[str(path)]=hashlib.sha256(raw).hexdigest()
    points=[];pending=[];sequence_records=[]
    for j,t in enumerate(times):
        numerical_status='validated_short_catalog'
        if j<2:
            reference={c:(comparison['cases'][c]['shared_base_short_log_marginals'][j],0.) for c in case_ids}
        else:
            stage=f'pilot{t}'
            if stage not in receipts or receipts[stage][1] is None:
                pending.append(dict(time=t,reason='missing or incomplete Bayesian calculation'))
                continue
            gate,reference=receipts[stage]
            precise=gate['precision_passed']
            numerical_status='precision_check_only' if precise else 'unresolved_numerical_precision'
            if not precise and not diagnostic:
                pending.append(dict(time=t,reason='Monte Carlo precision gate has not passed'))
                continue
        rows=[]
        for c in case_ids:
            values=comparison['cases'][c]; shared,shared_se=reference[c]
            loss,weights=mixture_loss([values['product_log_marginals'][j],shared,
                                      values['uniform_base_log_marginals'][j]],prior)
            row=dict(case_id=c,family=lookup[c]['family'],time=t,mixture_loss_bits=loss,
                     shared_log_likelihood=shared,posterior_family_weights=weights.tolist(),
                     mixture_mc_se_bits_delta_method=float(weights[1]*shared_se),
                     transformer_minus_mixture_bits={model:float(scores[j]-loss)
                         for model,scores in values['transformer_cumulative_loss_bits'].items()})
            rows.append(row)
        sequence_records.extend(rows)
        models=MODELS; groups={}
        for family in families:
            selected=[r for r in rows if r['family']==family]
            groups[family]={}
            for model in models:
                gaps=np.asarray([r['transformer_minus_mixture_bits'][model] for r in selected])
                groups[family][model]=dict(mean_bits=float(gaps.mean()),
                    sequence_se_bits=float(gaps.std(ddof=1)/2), n_sequences=4,
                    numerical_mc_se_bits=float(np.linalg.norm([r['mixture_mc_se_bits_delta_method'] for r in selected])/4))
        balanced={model:dict(mean_bits=float(np.mean([groups[f][model]['mean_bits'] for f in families])),
                             sequence_se_bits=float(np.linalg.norm([groups[f][model]['sequence_se_bits'] for f in families])/3),
                             numerical_mc_se_bits=float(np.linalg.norm([groups[f][model]['numerical_mc_se_bits'] for f in families])/3),
                             n_sequences=12) for model in models}
        points.append(dict(time=t,numerical_status=numerical_status,families=groups,balanced=balanced))
    return dict(schema='shared-base-cluster-comparison-v1',production_reference_validated=False,
                scope='Preliminary comparison on 12 fixed sequences; no population or validated long-context claim.',
                plan_sha256=base_context['plan_sha256'],receipt_sha256=provenance,script_sha256=sha(__file__),
                diagnostic=diagnostic,points=points,pending=pending,sequence_records=sequence_records,
                uncertainty='Sequence SE uses 4 independent sequences per family; numerical SE uses independent whole SMC repetitions and a delta approximation. These are shown separately and do not bound numerical bias or missed modes.')


def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'product':'Product','shared_base':'Shared-base LSA','uniform_base':'Uniform base'}
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
        for ax,(family,title) in zip(axes,names.items()):
            ax.axhline(0,color='.5',lw=.8)
            for model,color,label in [('pooled_seed0','#2468A0','Transformer seed 0'),('pooled_seed1','#BF611C','Transformer seed 1')]:
                times=[p['time'] for p in report['points']]
                values=[p['families'][family][model]['mean_bits'] for p in report['points']]
                errors=[p['families'][family][model]['sequence_se_bits'] for p in report['points']]
                ax.errorbar(times,values,yerr=errors,fmt='o-',color=color,label=label,markersize=4,capsize=2)
            ax.set_xscale('log',base=2);ax.set_xlim(.85,1200)
            ax.set_xticks([1,4,16,64,256,1024],['1','4','16','64','256','1024'])
            ax.set_title(title);ax.set_xlabel('Number of predictions');ax.grid(alpha=.15)
        axes[0].set_ylabel('Transformer loss minus mixture Bayes (bits)')
        axes[-1].legend(fontsize=8,loc='best')
        unresolved=any(p['numerical_status']=='unresolved_numerical_precision' for p in report['points'])
        label='DIAGNOSTIC: numerical precision unresolved' if unresolved else 'PRELIMINARY: 4 held-out sequences per family'
        fig.suptitle(label+'\nBars: ±1 sequence SE; numerical uncertainty retained in JSON',fontsize=10)
        fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--receipts',type=Path,nargs='*',default=[])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true',help='Explicitly include complete but imprecise numerical points; never for manuscript evidence')
    args=parser.parse_args()
    report=compare(args.plan,args.receipts,args.diagnostic)
    args.out.mkdir(parents=True,exist_ok=True)
    plot(report,args.out/'pilot_comparison.png')
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(points=[p['time'] for p in report['points']],pending=report['pending'],production_reference_validated=False)))


if __name__=='__main__':main()
