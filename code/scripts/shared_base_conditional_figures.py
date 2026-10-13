"""Provisional shared-base curves from saved transformer and validated Bayes data.

No inference is performed. The deterministic times 1 and4 are always included.
Optional independently validated full-cohort summaries add time16 at every
depth and time64 at the depths declared in the supplied audit. There is no Bayesian connecting line or
extrapolation beyond validated points.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/transformer/shared_base_conditional_curves'
DEFAULT_OUT=SOURCE/'figures'
TIMES=[1,4,16,64,256,1024]
DEPTHS=[0,1,4,8]
STYLES={
    'd64':dict(color='#1565a8',linestyle='-',marker='D',markersize=3.4),
    'd64_seed1':dict(color='#438575',linestyle='--',marker='s',markersize=3.4),
    'bayes':dict(color='#d77719',linestyle='none',marker='o',markersize=7.5),
}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def same_moments(actual,expected):
    return (set(actual)==set(expected)=={'n','mean','standard_error'}
        and actual['n']==expected['n']
        and all(np.isclose(actual[key],expected[key],rtol=1e-12,atol=1e-13)
                for key in ['mean','standard_error']))


def validated_time16(path,tf):
    """Accept only a complete audited comparison of the same fixed cohorts."""
    summary=json.loads(Path(path).read_text())
    require(summary['schema']=='shared-base-curve-paired-summary-v1' and summary['time']==16,
        'unexpected time16 summary')
    require(summary['prior']==dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'),
        'time16 reference training prior changed')
    require(summary['reference_validation_passed'] is True
        and summary['every_declared_sequence_retained'] is True,'incomplete time16 comparison')
    require(len(summary['rows'])==8,'time16 requires every depth and training seed')
    validations=summary['validations'];cohorts=summary['provenance']['cohorts']
    require(len(validations)==len(cohorts)==4,'incomplete time16 validation provenance')
    require(sorted(v['depth'] for v in validations)==sorted(v['depth'] for v in cohorts)==DEPTHS,
        'time16 validation conditions differ')
    for depth in DEPTHS:
        cohort=next(v for v in cohorts if v['depth']==depth)
        original=next(v for v in tf['provenance']['cohorts'] if v['depth']==depth)
        require(all(cohort[key]==original[key] for key in
            ['alpha','depth','n_sequences','cohort_sha256','protocol_sha256']),
            'time16 and transformer cohort identities differ')
        digest=cohort['audit_sha256']
        require(isinstance(digest,str) and len(digest)==64
            and all(c in '0123456789abcdef' for c in digest),'missing independent audit checksum')
        validation=next(v for v in validations if v['depth']==depth)
        require(validation['orders']==[16,32],'time16 requires both exponent quadrature rules')
        require(validation['status_counts']==dict(missing=0,running=0,invalid=0,needs_more_draws=0,ready=512),
            'time16 includes missing or unresolved references')
        audit=validation['cohort_validation']
        require(audit['schema']=='shared-base-cohort-precision-v1'
            and audit['complete'] is True and audit['passed'] is True
            and audit['n_sequences']==256 and audit['expected_ids']==list(range(256)),
            'time16 full-cohort validation has not passed')
        expected_checks={f'{rule}:{name}' for rule in ['coarse','refined']
            for name in ['all_basic_checks','all_scalar_mixing','all_jensen_local_checks','cohort_numerical_budget']}
        expected_checks.update(['all_case_refinement_checks','aggregate_refinement_resolution',
            'aggregate_refinement_stochastic_agreement'])
        require(len(audit['checks'])==len(expected_checks)
            and {v['name'] for v in audit['checks']}==expected_checks
            and all(v['passed'] is True for v in audit['checks']),
            'time16 numerical checks are missing or failed')
        require([v['rule'] for v in audit['rules']]==['coarse','refined']
            and all(v['n_sequences']==256 and v['preliminary_passed'] is True
                and v['numerical_error_allowance_bits']<=v['numerical_budget_bits']+1e-14
                for v in audit['rules']),'time16 numerical precision budget has not passed')
        refinement=audit['refinement']['per_case']
        require(len(refinement)==256 and sorted(v['case_id'] for v in refinement)==list(range(256))
            and all(v['passed'] is True for v in refinement),'time16 case refinement is incomplete')
        rows=[r for r in summary['rows'] if r['depth']==depth]
        require(len(rows)==2 and {r['model'] for r in rows}=={'d64','d64_seed1'},
            'time16 requires both matched transformer comparisons')
        require(rows[0]['bayes_regret']==rows[1]['bayes_regret'],'time16 Bayesian cohorts differ by transformer seed')
        refined=audit['rules'][1]
        expected_bayes=dict(n=256,mean=audit['reported_mean_bayes_regret_bits'],
            standard_error=refined['observed_sequence_se_bits'])
        require(same_moments(rows[0]['bayes_regret'],expected_bayes)
            and np.isclose(expected_bayes['mean'],refined['candidate_mean_bayes_regret_bits'],rtol=1e-12,atol=1e-13),
            'time16 displayed Bayesian values differ from the independent audit')
        for row in rows:
            require(row['alpha']==1 and row['time']==16 and row['n_sequences']==256,
                'time16 comparison condition differs')
            original=next(r for r in tf['rows'] if r['depth']==depth and r['time']==16 and r['model']==row['model'])
            require(same_moments(row['transformer_regret'],{key:original[key] for key in ['n','mean','standard_error']}),
                'time16 and transformer summary disagree')
    return summary


def validated_time64(path,tf):
    """Bind each declared complete time64 cohort to its independent raw audit."""
    summary=json.loads(Path(path).read_text())
    if summary.get('schema')=='shared-base-corrected-time64-summary-v1':
        from export_shared_base_time64 import validate_export
        return validate_export(path,tf)
    depths=[c['depth'] for c in summary['cohorts']]
    require(1<=len(depths)<=len(DEPTHS) and len(set(depths))==len(depths)
        and set(depths)<=set(DEPTHS),'time64 requires unique declared source depths')
    if 'depths' in summary:
        require(summary['depths']==sorted(depths),'time64 declared coverage differs from its cohorts')
    require(summary['schema']=='shared-base-morning-audit-summary-v1'
        and summary['all_requested_cohorts_passed'] is True
        and summary['total_distinct_source_sequences']==256*len(depths)
        and summary['total_audited_references']==512*len(depths),'incomplete time64 summary')
    files=[Path(path)]
    for cohort in summary['cohorts']:
        depth=cohort['depth'];report_path=ROOT/cohort['report']
        raw=report_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==cohort['report_sha256'],'time64 raw audit checksum differs')
        report=json.loads(raw);files.append(report_path)
        require(report['schema']=='shared-base-curve-audit-v1'
            and report['all_requested_times_validated'] is True
            and report['alpha']==cohort['alpha']==1 and report['depth']==depth
            and report['times']==[cohort['time']]==[64]
            and report['orders']==cohort['orders']==[16,32]
            and report['n_sequences']==cohort['n_sequences']==256
            and report['n_tokens_per_sequence']==1025 and len(report['rows'])==1
            and cohort['full_cohort_passed'] is True,'time64 audited source condition differs')
        provenance=cohort['provenance']
        original=next(c for c in tf['provenance']['cohorts'] if c['depth']==depth)
        require(provenance==report['provenance']
            and all(provenance[k]==original[k] for k in ['cohort_sha256','protocol_sha256'])
            and provenance['audit_sha256']==summary['audit_script_sha256'],
            'time64 and transformer cohort identities differ')
        row=report['rows'][0];audit=row['cohort_validation']
        require(row['time']==64 and row['status_counts']==dict(missing=0,running=0,invalid=0,needs_more_draws=0,ready=512),
            'time64 includes missing or unresolved references')
        require(audit['schema']=='shared-base-cohort-precision-v1' and audit['complete'] is True
            and audit['passed'] is True and audit['n_sequences']==256
            and audit['expected_ids']==list(range(256)),'time64 full-cohort audit failed')
        expected_checks={f'{rule}:{name}' for rule in ['coarse','refined'] for name in
            ['all_basic_checks','all_scalar_mixing','all_jensen_local_checks','cohort_numerical_budget']}
        expected_checks.update(['all_case_refinement_checks','aggregate_refinement_resolution',
            'aggregate_refinement_stochastic_agreement'])
        require(audit['checks']==cohort['checks'] and len(audit['checks'])==len(expected_checks)
            and {v['name'] for v in audit['checks']}==expected_checks
            and all(v['passed'] is True for v in audit['checks']),'time64 numerical checks are missing or failed')
        require([v['rule'] for v in audit['rules']]==['coarse','refined']
            and all(v['n_sequences']==256 and v['preliminary_passed'] is True
                and v['numerical_error_allowance_bits']<=v['numerical_budget_bits']+1e-14
                for v in audit['rules']),'time64 numerical precision budget failed')
        refinement=audit['refinement']['per_case']
        require(len(refinement)==256 and sorted(v['case_id'] for v in refinement)==list(range(256))
            and all(v['passed'] is True for v in refinement),'time64 case refinement is incomplete')
        cases=row['case_audits']
        require(len(cases)==512 and {(v['index'],v['order']) for v in cases}==
            {(i,o) for i in range(256) for o in [16,32]}
            and all(v['time']==64 and v['status']=='ready' and v['ready'] is True for v in cases),
            'time64 audited reference coverage differs')
        matched=cohort['matched_comparisons']
        require(matched==row['matched_comparisons'] and set(matched)=={'d64','d64_seed1'},
            'time64 summary differs from raw matched comparisons')
        refined=audit['rules'][1]
        expected_bayes=dict(n=256,mean=audit['reported_mean_bayes_regret_bits'],
            standard_error=refined['observed_sequence_se_bits'])
        require(np.isclose(expected_bayes['mean'],refined['candidate_mean_bayes_regret_bits'],rtol=1e-12,atol=1e-13),
            'time64 Bayesian mean differs from the refined audit')
        for model,values in matched.items():
            tf_row=next(r for r in tf['rows'] if r['depth']==depth and r['time']==64 and r['model']==model)
            require(same_moments(values['transformer_regret'],{k:tf_row[k] for k in ['n','mean','standard_error']})
                and same_moments(values['bayes_regret'],expected_bayes),
                'time64 displayed moments differ from the original transformer data or refined audit')
    return files,summary


def read_panels(time16_path=None,time64_path=None):
    files=[SOURCE/'summary.json',SOURCE/'validation.json',SOURCE/'time4_summary.json',SOURCE/'shortprefix_reference.json']
    tf,validation,time4,reference=[json.loads(p.read_text()) for p in files]
    require(tf['schema']=='shared-base-conditional-cohort-v1','unexpected transformer summary')
    require(tf['alpha']==1 and tf['depths']==DEPTHS and tf['times']==TIMES and tf['n_tokens_per_sequence']==1025,'unexpected source setting or time grid')
    require(validation['passed'] and all(tf['checks'].values()),'transformer/source validation has not passed')
    require(time4['reference_validation_passed'] and time4['every_declared_sequence_retained'],'incomplete time4 comparison')
    require(reference['passed'] and reference['time']==4,'time4 reference validation has not passed')
    require(time4['provenance']['reference_catalog_sha256']==sha_file(files[3]),'reference catalog changed')
    require(reference['prior']==dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'),'reference training prior changed')
    time16=None
    if time16_path is not None:
        time16=validated_time16(time16_path,tf)
        files.append(Path(time16_path))
    time64=None
    if time64_path is not None:
        require(time16 is not None,'time64 presentation requires the validated time16 comparison')
        new_files,time64=validated_time64(time64_path,tf)
        files.extend(new_files)
    time64_by_depth={} if time64 is None else {c['depth']:c for c in time64['cohorts']}
    panels=[]
    for depth in DEPTHS:
        panel=dict(alpha=1.,depth=depth,n_sequences=256,n_tokens=1025,times=TIMES,transformers={})
        for model in ['d64','d64_seed1']:
            rows=sorted([r for r in tf['rows'] if r['depth']==depth and r['model']==model],key=lambda r:r['time'])
            require([r['time'] for r in rows]==TIMES and all(r['n']==256 for r in rows),'missing transformer points or inconsistent cohort size')
            panel['transformers'][model]=dict(mean=[r['mean'] for r in rows],se=[r['standard_error'] for r in rows])
            r4=[r for r in time4['rows'] if r['depth']==depth and r['model']==model]
            require(len(r4)==1 and r4[0]['n_sequences']==256,'missing matched time4 record')
            require(r4[0]['transformer_regret']==dict(n=256,mean=rows[1]['mean'],standard_error=rows[1]['standard_error']),'time4 and transformer summaries disagree')
        r1=[r for r in tf['first_step_paired'] if r['depth']==depth]
        r4=[r for r in time4['rows'] if r['depth']==depth]
        require(len(r1)==len(r4)==2 and r1[0]['bayes_regret']==r1[1]['bayes_regret'] and r4[0]['bayes_regret']==r4[1]['bayes_regret'],'Bayesian cohorts differ by transformer seed')
        panel['bayes']=dict(times=[1,4],mean=[r1[0]['bayes_regret']['mean'],r4[0]['bayes_regret']['mean']],
            se=[r1[0]['bayes_regret']['standard_error'],r4[0]['bayes_regret']['standard_error']])
        if time16 is not None:
            r16=next(r for r in time16['rows'] if r['depth']==depth)
            panel['bayes']['times'].append(16)
            panel['bayes']['mean'].append(r16['bayes_regret']['mean'])
            panel['bayes']['se'].append(r16['bayes_regret']['standard_error'])
            validated=next(v for v in time16['validations'] if v['depth']==depth)
            panel['time16_numerical_validation']=validated
        if depth in time64_by_depth:
            cohort=time64_by_depth[depth]
            r64=cohort['matched_comparisons']['d64']['bayes_regret']
            panel['bayes']['times'].append(64)
            panel['bayes']['mean'].append(r64['mean'])
            panel['bayes']['se'].append(r64['standard_error'])
            panel['time64_numerical_validation']=cohort
        for curve in list(panel['transformers'].values())+[panel['bayes']]:
            require(np.isfinite(curve['mean']).all() and np.isfinite(curve['se']).all(),'nonfinite displayed statistic')
            require(min(curve['mean'])>0 and min(curve['se'])>=0,'invalid mean or uncertainty')
            require(np.max(np.asarray(curve['mean'])+curve['se'])<3.85 and np.min(np.asarray(curve['mean'])-curve['se'])>0,'axis limits would clip an uncertainty interval')
        panels.append(panel)
    return files,panels


def receipt_base(files,panels):
    # The shared lists describe points available at every depth; the explicit
    # per-depth lists retain additional validated points without implying them elsewhere.
    bayes_times=[t for t in TIMES if all(t in p['bayes']['times'] for p in panels)]
    coverage={str(p['depth']):p['bayes']['times'] for p in panels}
    return dict(schema='shared-base-provisional-conditional-figure-v1',passed=True,script_sha256=sha_file(__file__),
        inputs={str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve()):sha_file(p) for p in files},panels=panels,
        quantity='Mean next-token regret in bits: KL(true current transition row || predictor).',
        uncertainty='Plus/minus one standard error over256 independent complete sequences in each panel.',
        conditions='alpha=1; source depths0,1,4,8. Predictors keep their full training prior; selected parameters are hidden.',
        display=dict(transformer_times=TIMES,bayes_times=bayes_times,bayes_pending_times=[t for t in TIMES if t not in bayes_times],
            bayes_times_scope='Times validated at every displayed depth; see bayes_times_by_depth for full coverage.',
            bayes_times_by_depth=coverage,
            bayes_pending_times_by_depth={d:[t for t in TIMES if t not in ts] for d,ts in coverage.items()},
            deterministic_bayes_times=[1,4],mcmc_bayes_times=[t for t in bayes_times if t not in [1,4]],
            bayes_connecting_lines=False,bayes_extrapolation=False,styles=STYLES,
            transformer_connecting_lines='Straight visual guides between measured times; no interpolated estimates are reported.',
            common_y_range=[0,3.85],x_axis='Observed tokens, logarithmic base2'),
        new_inference=False,new_reference_computation=False,provisional=True)


def render(panels,out):
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42}):
        fig,axes=plt.subplots(2,2,figsize=(8.,6.4),sharex=True,sharey=True)
        fig.subplots_adjust(left=.105,right=.985,bottom=.19,top=.80,wspace=.16,hspace=.25)
        for i,(ax,panel) in enumerate(zip(axes.flat,panels)):
            ax.set_xscale('log',base=2)
            ax.set_xlim(.82,1240)
            ax.set_xticks(TIMES,[str(t) for t in TIMES])
            ax.minorticks_off()
            ax.set_ylim(0,3.85)
            ax.set_yticks([0,1,2,3])
            ax.set_axisbelow(True)
            ax.grid(axis='y',color='#D8DEE5',linewidth=.65)
            ax.tick_params(labelsize=8.5,length=3)
            for model,curve in panel['transformers'].items():
                mean=np.array(curve['mean']);se=np.array(curve['se'])
                ax.fill_between(TIMES,mean-se,mean+se,color=STYLES[model]['color'],alpha=.14,linewidth=0)
                ax.plot(TIMES,mean,**STYLES[model],linewidth=1.6,markerfacecolor='white',markeredgewidth=.8,zorder=3)
            b=panel['bayes']
            ax.errorbar(b['times'],b['mean'],yerr=b['se'],**STYLES['bayes'],linewidth=0,
                markerfacecolor='none',markeredgewidth=1.25,elinewidth=1.,capsize=2.5,zorder=5)
            ax.set_title(f"({chr(97+i)}) $L={panel['depth']}$",loc='left',pad=7)
        fig.suptitle('Shared-base LSA: fixed source parameters',y=.985,fontsize=13)
        fig.text(.5,.940,r'$\alpha=1$; 256 independent sequences of length 1025 per panel',ha='center',fontsize=10)
        handles=[Line2D([],[],**STYLES[key],linewidth=1.6,markerfacecolor='none',markeredgewidth=1.)
                 for key in ['d64','d64_seed1','bayes']]
        fig.legend(handles,['Transformer, seed 0','Transformer, seed 1','Bayes with the training prior'],
            loc='upper center',bbox_to_anchor=(.54,.914),ncol=3,frameon=False,fontsize=9.,handlelength=2.5,columnspacing=1.6)
        fig.supylabel('Expected next-token regret (bits)',x=.018,fontsize=10.5)
        fig.supxlabel('Observed tokens, $t$',y=.123,fontsize=10.5)
        endpoints={p['depth']:p['bayes']['times'][-1] for p in panels}
        if len(set(endpoints.values()))==1:
            coverage=f'Bayesian reference points after $t={next(iter(endpoints.values()))}$ are still pending.'
        else:
            groups=[]
            for endpoint in sorted(set(endpoints.values()),reverse=True):
                depths=','.join(str(d) for d,t in endpoints.items() if t==endpoint)
                groups.append(f'through $t={endpoint}$ for $L={depths}$')
            coverage='Bayes validated '+'; '.join(groups)+'.'
        fig.text(.5,.068,coverage,ha='center',fontsize=10,color='#555555')
        fig.text(.5,.035,'Shading and error bars show ±1 standard error.',ha='center',fontsize=9,color='#555555')
        fig.savefig(out/'provisional_shared_base.pdf',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(out/'provisional_shared_base.png',dpi=220,metadata={'Software':'shared_base_conditional_figures.py'})
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--time16-summary',type=Path,help='Optional independently validated complete-cohort time16 summary.')
    p.add_argument('--time64-summary',type=Path,help='Optional independently validated time64 summary for declared complete depth cohorts.')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();files,panels=read_panels(args.time16_summary,args.time64_summary);receipt=receipt_base(files,panels)
    names=['provisional_shared_base.pdf','provisional_shared_base.png']
    if args.check_only:
        actual=json.loads((args.out/'receipt.json').read_text())
        hashes=actual.pop('outputs')
        require(actual==receipt,'figure receipt differs from source data or script')
        require(hashes=={name:sha_file(args.out/name) for name in names},'figure files changed')
        print('Provisional shared-base figure: source values, reference limits and hashes verified.')
        return
    args.out.mkdir(parents=True,exist_ok=True);render(panels,args.out)
    receipt['outputs']={name:sha_file(args.out/name) for name in names}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(args.out/'provisional_shared_base.pdf')


if __name__=='__main__':main()
