"""Independent raw-sample audit of a fixed-setting full-family Bayes pair."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from conditional_shared_base_benchmark import (ROOT,HERE,TAG,inputs,identity,draw,sha_array,sha_file,fingerprint,atomic_json,CODE_FILES)
from shared_base_surrogate_summary import raw_checkpoint,load_models,_close
from shared_base_full_context_summary import (Audit,comparison_metrics,metric_refinement,probability_refinement,numerical_array_hash,NUMERICAL_CODE)
from full_context_statistics import diagnostic_gates


def run(folder,index,prefix,orders):
    audit=Audit();tokens,oracle,models,input_identity=inputs(folder,index,prefix)
    cohort=audit.read('cohort',folder/'cohort.json')
    source=audit.read('source',folder/'data'/f'seq_{index:06d}.json')
    regenerated=draw(cohort['alpha'],cohort['depth'],source['identity']['sequence_seed'])
    audit.require('source_replay',np.array_equal(regenerated[0],source['tokens'])
        and np.array_equal(regenerated[1],source['oracle']['initial_probs'])
        and np.array_equal(regenerated[2],source['oracle']['transition_probs']))
    model_probs=load_models(audit,folder,dict(record=source,tokens=np.asarray(source['tokens'])),index,prefix)
    estimates=[];allmetrics=[];summaries=[]
    for order in orders:
        name=f'alpha{order}';stem=f'seq_{index:06d}_t{prefix:04d}'
        path=folder/'refs'/name/(stem+'.json');r=audit.read(name,path)
        audit.require(name+':identity',r['schema']==TAG and r['reference_id']==fingerprint(r['identity'])
            and r['identity']['inputs']==input_identity and r['sequence_index']==index and r['prefix_length']==prefix)
        hashes=r['identity']['code_sha256']
        audit.require(name+':complete_code',set(hashes)==set(CODE_FILES)|{'conditional_shared_base_benchmark.py'})
        for file,h in hashes.items():audit.require(name+':code:'+file,sha_file(Path(__file__).with_name(file) if file=='conditional_shared_base_benchmark.py' else HERE/file)==h)
        problem=r['identity']['numerical_problem']
        audit.require(name+':observations',problem['d']==64 and problem['prefix_length']==prefix and problem['token_hash']==numerical_array_hash(np.asarray(tokens,dtype='<i8')))
        audit.require(name+':numerical_files',set(problem['numerical_code_hashes'])==NUMERICAL_CODE)
        for file,h in problem['numerical_code_hashes'].items():
            if file.startswith('store/'):p=ROOT.parent/'product_model_with_memory/tables/anchors_prod'/file.split('/',1)[1]
            elif file.startswith('pmm/'):p=ROOT.parent/'product_model_with_memory/src/product_model_with_memory'/file.split('/',1)[1]
            else:p=HERE/file
            audit.require(name+':numerical_hash:'+file,sha_file(p)==h)
        nodes,weights=np.polynomial.legendre.leggauss(order);nodes=1+nodes/2;weights=weights/2
        audit.require(name+':full_prior',all(_close(item['alpha_nodes'],nodes,atol=1e-14) and _close(item['alpha_weights'],weights,atol=1e-14)
            and item['depths']==list(range(9)) and _close(item['depth_prior'],np.ones(9)/9,atol=1e-14) for item in (r['settings'],problem)))
        audit.require(name+':row_grid',all(problem['kernel_grid'][k]==v for k,v in dict(lower=-32.,upper=32.,step=1/32).items()) and problem['prob_tol']==1e-7)
        moments,fresh=raw_checkpoint(audit,name,folder/'chains'/name/(stem+'.npz'),r)
        moment_path=path.with_name(r['moments_file'])
        audit.require(name+':moments_hash',sha_file(moment_path)==r['moments_sha256'])
        with np.load(moment_path,allow_pickle=False) as saved:
            audit.require(name+':moments_keys',set(saved.files)==set(moments))
            for key,v in moments.items():audit.require(name+':moments:'+key,_close(v,saved[key],rtol=1e-12))
        gates=diagnostic_gates(fresh);audit.require(name+':gates_readback',gates==r['diagnostic_gates'])
        for key,passed in gates.items():audit.check(name+':'+key,passed)
        grids=r['final_state_grid_checks'];audit.check(name+':four_fine_grid_checks',len(grids)==4)
        for j,g in enumerate(grids):
            passed=(0<=g['log_likelihood_difference_bits']<1e-5 and 0<=g['max_prediction_difference']<1e-7
                and 0<=max(g['coarse_normalization_error'],g['fine_normalization_error'])<1e-7)
            audit.require(name+f':fine_grid_gate{j}',passed==g['passed']);audit.check(name+f':fine_grid{j}',passed)
        metrics={k:comparison_metrics(moments['mean'],v,oracle,moments['cov_batch'],moments['cov_chain']) for k,v in model_probs.items()}
        for k,m in metrics.items():
            old=r['model_metrics'][k]
            for key,value in [('kl_bits',m['kl_bits']),('true_loss_gap_bits',m['true_expected_loss_gap_bits']),('kl_mcse_bits',m['kl_uncertainty']['mcse_bits']),('gap_mcse_bits',m['gap_uncertainty']['mcse_bits'])]:
                audit.require(name+':metric_readback:'+k+':'+key,np.isclose(old[key],value,rtol=1e-10,atol=1e-13))
            audit.check(name+':scalar_precision:'+k,max(m['kl_numerical_scale_bits'],m['gap_numerical_scale_bits'])<=.01/3)
        estimates.append(fresh);allmetrics.append(metrics)
        summaries.append(dict(order=order,draws_per_chain=fresh['diagnostics']['draws'],seconds=fresh['elapsed_seconds'],
            maximum_probability_mcse=float(max(fresh['mcse_probs'])),minimum_probability_ess=float(min(fresh['diagnostics']['ess_probs'])),
            local_validation_passed=r['local_validation_passed'],model_metrics=r['model_metrics']))
    audit.require('independent_seeds',estimates[0]['diagnostics']['seed']!=estimates[1]['diagnostics']['seed'])
    vector=probability_refinement(estimates[0]['predictive_probs'],estimates[1]['predictive_probs'],estimates[0]['mcse_probs'],estimates[1]['mcse_probs'])
    audit.check('probability_refinement',vector['passed'])
    refinement={k:metric_refinement(allmetrics[0][k],allmetrics[1][k]) for k in model_probs}
    for k,m in refinement.items():audit.check('metric_refinement:'+k,m['passed'])
    result=dict(schema='conditional-shared-base-audit-v1',setting=dict(alpha=cohort['alpha'],depth=cohort['depth']),
        sequence_index=index,time=prefix,passed=all(c['passed'] for c in audit.checks),checks=audit.checks,inputs=audit.inputs,
        references=summaries,probability_refinement=vector,metric_refinement=refinement,
        interpretation='Single declared computational benchmark. No population average; no selected-subset mean.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'output/conditional_shared_base_benchmark_2026-09-15/a1_L4')
    p.add_argument('--index',type=int,default=0);p.add_argument('--prefix',type=int,default=16);p.add_argument('--orders',type=int,nargs=2,default=[16,32])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=run(a.input,a.index,a.prefix,a.orders);atomic_json(a.out,result)
    print(json.dumps(dict(passed=result['passed'],checks=len(result['checks']),failures=[x['name'] for x in result['checks'] if not x['passed']]),indent=2))
    if not result['passed']:raise SystemExit(1)
