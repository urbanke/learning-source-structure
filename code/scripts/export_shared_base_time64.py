"""Export all four corrected time64 cohorts only after every numerical gate passes.

This is a read-only reduction of completed audits and frozen predictions. It
recomputes the unchanged cohort gates and paired moments; it neither samples
posteriors nor presents corrected audits as historical uncorrected receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='shared-base-corrected-time64-summary-v1'
DEPTHS=[0,1,4,8]
MODELS=['d64','d64_seed1']
PRIOR=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def payload_hash(summary):
    payload={k:v for k,v in summary.items() if k!='validation_receipt'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def checked_cohort_gate(value):
    checks={f'{rule}:{name}' for rule in ['coarse','refined'] for name in
            ['all_basic_checks','all_scalar_mixing','all_jensen_local_checks','cohort_numerical_budget']}
    checks.update(['all_case_refinement_checks','aggregate_refinement_resolution','aggregate_refinement_stochastic_agreement'])
    require(value['schema']=='shared-base-cohort-precision-v1' and value['complete'] is True
        and value['passed'] is True and value['n_sequences']==256
        and value['expected_ids']==list(range(256)),'Incomplete or failed whole-cohort gate')
    require(len(value['checks'])==11 and {c['name'] for c in value['checks']}==checks
        and all(c['passed'] is True for c in value['checks']),'Missing or failed numerical check')
    require([r['rule'] for r in value['rules']]==['coarse','refined']
        and all(r['n_sequences']==256 and r['preliminary_passed'] is True
        and r['numerical_error_allowance_bits']<=r['numerical_budget_bits']+1e-14
        for r in value['rules']),'Numerical precision budget failed')
    cases=value['refinement']['per_case']
    require(len(cases)==256 and sorted(c['case_id'] for c in cases)==list(range(256))
        and all(c['passed'] is True for c in cases),'Refinement coverage incomplete')


def source_rows(summary_path):
    """Check closed case receipts and their declared mandatory replacement map."""
    import shared_base_local_revalidation as local
    from shared_base_curve_audit import same
    from revalidate_shared_base_depth import adjust_depth_gate
    summary_path=Path(summary_path).resolve();summary=read(summary_path)
    require(summary['complete'] is True and summary['all_cohorts_passed'] is True
        and summary['expected_references']==summary['observed_references']==2048
        and not summary.get('failures') and not summary['missing_keys'],
        'All2048 cases and allfour full-cohort gates must pass before export')
    inputs={str(summary_path):sha(summary_path)}
    if summary['schema']=='shared-base-local-revalidation-v1':
        plan,plan_hash=local.read_plan(summary_path.parent)
        require(summary['plan_id']==plan['plan_id'] and summary['launch_sha256']==plan_hash,
                'Revalidation summary/plan mismatch')
        expected={t['key']:t for t in plan['cases']}
        launches={key:plan_hash for key in expected}
    elif summary['schema']=='shared-base-local-finish-summary-v1':
        import finish_shared_base_local as finish
        plan=finish.read_plan(summary_path.parent)
        require(summary['plan_id']==plan['plan_id'],'Continuation summary/plan mismatch')
        overlay_path=summary_path.parent/'overlay.json';overlay=read(overlay_path)
        require(sha(overlay_path)==summary['overlay_sha256'] and overlay['complete'] is True
            and overlay['schema']=='shared-base-local-finish-overlay-v1'
            and overlay['plan_id']==plan['plan_id'] and overlay['n_declared']==2048,
            'Mandatory continuation overlay incomplete or changed')
        inputs[str(overlay_path)]=sha(overlay_path)
        base,base_rows=finish.checked_summary(plan['audit_summary'])
        require(summary['base_audit_summary']==plan['audit_summary']
            and summary['base_audit_summary_sha256']==plan['audit_summary_sha256'],
            'Continuation base audit changed')
        inputs[plan['audit_summary']]=plan['audit_summary_sha256']
        expected={};launches={}
        for key,item in overlay['references'].items():
            require(item['complete'] is True,'Planned replacement is missing')
            path=Path(item['case_receipt']);row=read(path)
            require(sha(path)==item['case_receipt_sha256'],'Overlay case receipt changed')
            if key in plan['selected']:
                receipt,checked=finish.checked_result(plan,key)
                require(item['planned_replacement'] is True and same(row,checked),
                        'Continued reference differs from its checked receipt')
                launches[key]=plan['plan_id']
                for artifact in (finish.task_paths(plan,key)['receipt'],path):
                    inputs[str(artifact)]=sha(artifact)
            else:
                require(item['planned_replacement'] is False and same(row,base_rows[key]),
                        'Untouched reference was replaced')
                launches[key]=base_rows[key]['launch_sha256']
            expected[key]=item['task']
    else:raise ValueError('Unsupported corrected audit schema')
    plan_path=summary_path.parent/'plan.json';inputs[str(plan_path)]=sha(plan_path)
    require(set(expected)==set(summary['references'])==local.expected_keys(),'Fixed reference coverage differs')
    code=plan['code_sha256']
    for name,digest in code.items():
        path=ROOT/name
        require(sha(path)==digest,'Bound audit or continuation code changed')
        inputs[str(path.resolve())]=digest
    rows=[]
    for key,item in summary['references'].items():
        require(item['task']==expected[key],'Summary task differs from declared effective case')
        path=Path(item['case_receipt']);row=read(path)
        require(sha(path)==item['case_receipt_sha256'] and row['task']==expected[key]
            and row['launch_sha256']==launches[key]
            and row['row_id']==local.fingerprint({k:v for k,v in row.items() if k!='row_id'}),
            'Corrected case receipt identity differs')
        inputs[str(path)]=item['case_receipt_sha256']
        task=row['task'];reference=read(task['reference'])
        local.check_task_files(task)
        require(sha(task['reference'])==task['file_sha256'][task['reference']]
            and reference['reference_id']==task['reference_id'], 'Bound source report changed')
        inputs[task['reference']]=task['file_sha256'][task['reference']]
        corrected=adjust_depth_gate(reference,row['original_audit'],row['depth_diagnostics'])
        require(same(corrected,row['corrected']) and corrected['audit']['status']=='ready'
            and corrected['audit']['ready'] is True and item['ready'] is True,
            'Corrected arithmetic or case readiness differs')
        rows.append(row)
    return summary,plan,rows,inputs


def build(summary_path):
    from shared_base_curve_audit import aggregate_time,moments,same
    from shared_base_conditional_cohort import inputs as cohort_inputs
    from validate_shared_base_mcmc import json_safe
    summary,plan,rows,bound=source_rows(summary_path)
    tf_path=ROOT/'results/transformer/shared_base_conditional_curves/summary.json';tf=read(tf_path)
    bound[str(tf_path)]=sha(tf_path)
    require(sorted(g['depth'] for g in summary['groups'])==DEPTHS,'Missing or duplicate numerical cohort')
    cohorts=[]
    for depth in DEPTHS:
        selected=[r for r in rows if r['task']['depth']==depth]
        by_case={(r['task']['index'],r['task']['order']):r for r in selected}
        require(len(selected)==512 and set(by_case)=={(i,o) for i in range(256) for o in [16,32]},
                'Incomplete source/order coverage')
        audit=json_safe(aggregate_time([r['corrected']['audit'] for r in selected],list(range(256)),[16,32]))
        checked_cohort_gate(audit)
        stored=next(g for g in summary['groups'] if g['depth']==depth)
        require(stored['time']==64 and same(audit,stored['numerical_audit']),
                'Recomputed cohort gates differ from the completed audit')
        folders={r['task']['source_folder'] for r in selected};require(len(folders)==1,'Mixed source folders')
        folder=Path(next(iter(folders)));cohort_path=folder/'cohort.json';protocol_path=folder/'protocol.json'
        declaration=read(cohort_path);protocol=read(protocol_path)
        original=next(c for c in tf['provenance']['cohorts'] if c['depth']==depth)
        require(sha(cohort_path)==original['cohort_sha256'] and sha(protocol_path)==original['protocol_sha256']
            and declaration['n_sequences']==256 and declaration['n_tokens_per_sequence']==1025
            and declaration['depth']==depth and declaration['alpha']==1
            and protocol['reference_prior']==PRIOR,'Frozen test cohort or training prior differs')
        bound[str(cohort_path)]=sha(cohort_path);bound[str(protocol_path)]=sha(protocol_path)
        bayes=[];transformers={name:[] for name in MODELS}
        for index in range(256):
            _,oracle,predictions,identity=cohort_inputs(folder,index,64)
            refined=by_case[index,32];reference=read(refined['task']['reference'])
            require(identity==reference['identity']['inputs'],'Frozen source/model inputs changed')
            precision=refined['corrected']['audit']['record']['precision']
            q=np.asarray(precision['predictive_probs'])
            require(q.shape==(64,) and np.isfinite(q).all() and np.all(q>0)
                and abs(q.sum()-1)<1e-10,'Invalid refined predictive distribution')
            kl=float(np.sum(rel_entr(oracle,q))/np.log(2))
            require(np.isclose(kl,precision['bayes_regret_bits'],atol=1e-12,rtol=1e-12),
                    'Refined Bayesian KL differs from audited probabilities')
            bayes.append(kl)
            for name,prediction in predictions.items():
                transformers[name].append(float(np.sum(rel_entr(oracle,prediction['prediction']))/np.log(2)))
                model_path=Path(prediction['path']);bound[str(model_path)]=prediction['sha256']
                bound[str(model_path.with_suffix('.npz'))]=read(model_path)['arrays_sha256']
            bound[str(folder/'data'/f'seq_{index:06d}.json')]=identity['source_file_sha256']
        matched={name:dict(transformer_regret=moments(values),bayes_regret=moments(bayes),
            transformer_minus_bayes=moments(np.asarray(values)-bayes)) for name,values in transformers.items()}
        require(np.isclose(np.mean(bayes),audit['reported_mean_bayes_regret_bits'],atol=1e-12,rtol=1e-12),
                'Mean Bayesian regret differs from cohort audit')
        for name,values in matched.items():
            old=next(r for r in tf['rows'] if r['depth']==depth and r['time']==64 and r['model']==name)
            require(same(values['transformer_regret'],{k:old[k] for k in ['n','mean','standard_error']}),
                    'Matched transformer statistics differ from frozen figure data')
        cohorts.append(dict(depth=depth,alpha=1,time=64,n_sequences=256,n_tokens=1025,orders=[16,32],
            full_cohort_passed=True,numerical_audit=audit,matched_comparisons=matched,
            provenance=dict(cohort_sha256=sha(cohort_path),protocol_sha256=sha(protocol_path)),
            case_receipts={r['task']['key']:summary['references'][r['task']['key']]['case_receipt_sha256'] for r in selected}))
    result=dict(schema=SCHEMA,depths=DEPTHS,prior=PRIOR,all_requested_cohorts_passed=True,
        total_distinct_source_sequences=1024,total_audited_references=2048,cohorts=cohorts,
        source_summary_schema=summary['schema'],source_summary=str(Path(summary_path).resolve()),
        source_plan_id=plan['plan_id'],correction='Scale-invariant depth split-Rhat; all other numerical gates unchanged.',
        uncertainty='One standard error across256 independent complete sequences per depth; differences paired persequence. No training-run pooling.',
        limitations='Numerical precision and mixing checks do not prove global posterior convergence. Historical audit files remain unchanged.')
    validation=dict(schema='shared-base-corrected-time64-export-validation-v1',passed=True,
        exporter_sha256=sha(__file__),summary_payload_sha256=payload_hash(result),inputs=bound,
        raw_file_sha256={path:digest for row in rows for path,digest in row['task']['file_sha256'].items()},
        raw_files_rehashed_at_export=True,
        correction_code_sha256={name:digest for name,digest in bound.items() if name.endswith('.py')},
        raw_audit_reexecuted=False,source_sampling=False,posterior_sampling=False,model_inference=False,
        recomputed='All2048 correction receipts, allfour unchanged11-gate aggregates and all256 refined-probability/matched-model moments perdepth.')
    require(all(sha(path)==digest for path,digest in bound.items()),
            'Bound audit or prediction files changed during export')
    return result,validation


def validate_export(path,tf):
    """Lightweight plotting gate; corrected receipts remain a distinct schema."""
    path=Path(path);summary=read(path);receipt_path=ROOT/summary['validation_receipt']['path']
    require(sha(receipt_path)==summary['validation_receipt']['sha256'],'Corrected export validation changed')
    validation=read(receipt_path)
    require(summary['schema']==SCHEMA and validation['schema']=='shared-base-corrected-time64-export-validation-v1'
        and validation['passed'] is True and validation['exporter_sha256']==sha(__file__)
        and validation['summary_payload_sha256']==payload_hash(summary),'Corrected export identity differs')
    require(summary['depths']==DEPTHS and summary['prior']==PRIOR
        and summary['all_requested_cohorts_passed'] is True
        and summary['total_distinct_source_sequences']==1024 and summary['total_audited_references']==2048
        and sorted(c['depth'] for c in summary['cohorts'])==DEPTHS,'Incomplete corrected export')
    for name,digest in validation['inputs'].items():
        require(sha(name)==digest,'Bound corrected audit/model input changed: '+name)
    for cohort in summary['cohorts']:
        require(cohort['alpha']==1 and cohort['time']==64 and cohort['n_sequences']==256
            and cohort['n_tokens']==1025 and cohort['orders']==[16,32]
            and cohort['full_cohort_passed'] is True,'Corrected source condition differs')
        checked_cohort_gate(cohort['numerical_audit'])
        original=next(c for c in tf['provenance']['cohorts'] if c['depth']==cohort['depth'])
        require(all(cohort['provenance'][k]==original[k] for k in ['cohort_sha256','protocol_sha256']),
                'Corrected and original transformer cohorts differ')
        require(set(cohort['matched_comparisons'])==set(MODELS),'Missing training run')
        expected={f"a1_L{cohort['depth']}/seq_{i:06d}/t0064/alpha{o}"
                  for i in range(256) for o in [16,32]}
        require(set(cohort['case_receipts'])==expected,'Missing corrected case receipt')
        audit=cohort['numerical_audit']
        for name,value in cohort['matched_comparisons'].items():
            row=next(r for r in tf['rows'] if r['depth']==cohort['depth'] and r['time']==64 and r['model']==name)
            require(value['transformer_regret']['n']==256 and value['bayes_regret']['n']==256
                and value['transformer_minus_bayes']['n']==256
                and all(np.isclose(value['transformer_regret'][k],row[k],atol=1e-13,rtol=1e-12)
                for k in ['mean','standard_error']),'Matched transformer values differ')
            require(np.isclose(value['bayes_regret']['mean'],audit['reported_mean_bayes_regret_bits'],atol=1e-13,rtol=1e-12)
                and np.isclose(value['bayes_regret']['standard_error'],audit['rules'][1]['observed_sequence_se_bits'],atol=1e-13,rtol=1e-12)
                and all(np.isfinite(v[k]) and (k!='standard_error' or v[k]>=0)
                        for v in value.values() for k in ['mean','standard_error']),
                'Matched Bayesian statistics differ from refined audit')
    return [path,receipt_path],summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),'Use a new export directory; historical records remain unchanged')
    result,validation=build(args.summary)
    args.out.mkdir(parents=True)
    receipt=args.out/'validation.json';receipt.write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    result['validation_receipt']=dict(path=str(receipt.resolve()),sha256=sha(receipt))
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(args.out),depths=DEPTHS,complete=True,all_cohorts_passed=True)))


if __name__=='__main__':main()
