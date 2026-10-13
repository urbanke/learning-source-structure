"""Independent saved-data audit of the fixed hidden-factor scaling pilot.

Uses NumPy and the earlier independent baseline audit, never the evaluator,
transformer, trainer or Bayesian enumeration. No input artifact is modified.
Full tokenwise transformer losses were not archived: cumulative statistics
are recomputed from the saved cumulative scores, not reconstructed by inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import re
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json'
BASELINE=ROOT/'output/hidden_factor_cluster_pilot_2026-09-16'
INPUT=ROOT/'output/hidden_factor_scaling_results_2026-09-17'
OUT=ROOT/'results/transformer/hidden_factor_scaling_2026-09-17'
BASE_AUDITOR=ROOT/'results/transformer/hidden_factor_pilot_2026-09-16/validate_pilot.py'
METRICS=('next_kl_bits','next_loss_bits','cumulative_loss_bits')


def require(value,message):
    if not value:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def fingerprint(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


class Audit:
    def __init__(self):self.hashes={};self.errors={};self.checks=0
    def check(self,value,message):self.checks+=1;require(value,message)
    def raw(self,path):
        path=Path(path).resolve();raw=path.read_bytes();self.hashes[str(path)]=sha(raw);return raw
    def json(self,path):return json.loads(self.raw(path))
    def npz(self,path,digest):
        raw=self.raw(path);self.check(sha(raw)==digest,'Array checksum mismatch: '+str(path))
        with np.load(BytesIO(raw),allow_pickle=False) as saved:return {k:saved[k] for k in saved.files}
    def close(self,name,actual,expected,tolerance=2e-9):
        a,b=np.asarray(actual),np.asarray(expected)
        self.check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name+': invalid shape/value')
        error=float(np.abs(a-b).max()) if a.size else 0.
        self.errors[name]=max(error,self.errors.get(name,0.))
        self.check(error<=tolerance,f'{name}: error {error} > {tolerance}')


def checkpoint_metadata(audit,identity,plan,plan_sha,arm,seed,code_hashes):
    receipt=identity['checkpoint_receipt'];config=receipt['config'];spec=plan['arms'][arm]
    expected=dict(spec['config'],seed=seed,source_dimensions=plan['source_dimensions'])
    audit.check({k:v for k,v in config.items() if k!='scaling_provenance'}==expected,'Checkpoint configuration differs from frozen plan')
    audit.check(identity['config_hash']==receipt['config_hash']==fingerprint(config),'Configuration hash differs')
    provenance=config['scaling_provenance'];parent=provenance['parent']
    audit.check(identity['scaling_provenance']==provenance and provenance['plan_sha256']==plan_sha
        and provenance['arm']==arm and provenance['seed']==seed
        and provenance['trainer_sha256']==code_hashes['train_hidden_factor_scaling.py'],'Scaling provenance differs')
    audit.check(identity['checkpoint_step']==expected['steps'] and identity['completed_sequences']==expected['steps']*expected['batch'],
        'Incomplete or intermediate checkpoint')
    audit.check(re.fullmatch('[0-9a-f]{64}',identity['checkpoint_sha256']) is not None,'Missing checkpoint digest')
    model=dict(vocab=16,n_ctx=expected['n_ctx'],n_layer=expected['n_layer'],n_head=expected['n_head'],
               d_model=expected['d_model'],mlp=True,dropout=0.)
    audit.check(identity['model']==receipt['model']==model and identity['n_params']==receipt['n_params']==spec['n_params'],
        'Architecture differs from plan')
    source=dict(version='hidden-factor-dirichlet1-v1',dimensions=[2,8],
        rows='independent Dirichlet(1,...,1) for each coordinate',initial_states='independent uniform coordinates',
        encoding='fresh uniform joint-pair-to-token permutation per sequence',parameters='fresh independent transitions and encoding per sequence',
        model_input='encoded tokens only',sampling='indexed-batch-v1')
    audit.check(receipt['source_contract']==source and receipt['source_contract_hash']==fingerprint(source),'Source contract differs')
    audit.check(identity['original_implementation_sha256']==receipt['implementation_hashes']==plan['implementation_sha256'],
        'Source/model/helper identity differs')
    audit.check(receipt['precision']=='float32' and receipt['runtime']['device']=='cuda'
        and identity['device']=='cuda' and identity['inference_precision']=='float32; log-softmax float64','Unexpected numerical precision/device')
    monitoring=receipt['validation_hashes']
    audit.check(set(monitoring)=={'hidden_factor'} and re.fullmatch('[0-9a-f]{64}',monitoring['hidden_factor']) is not None
        and plan['test']['data_sha256'] not in monitoring.values(),'Monitoring data identity missing or equals test archive')
    if arm=='longsmall':
        original=dict(spec['parent_config'],seed=seed,source_dimensions=[2,8])
        audit.check(identity['checkpoint_schema']=='hidden-factor-scaling-weights-v1' and isinstance(parent,dict),
            'Continuation checkpoint schema/ancestry differs')
        audit.check(parent['schema']=='hidden-factor-checkpoint-v1' and parent['config_hash']==fingerprint(original)
            and parent['step']==spec['parent_step'] and parent['completed_sequences']==spec['parent_completed_sequences']
            and parent['weights_checkpoint_sha256']==plan['baseline']['checkpoint_sha256'][str(seed)],'Wrong continuation parent')
        audit.check(re.fullmatch('[0-9a-f]{64}',parent['checkpoint_sha256']) is not None
            and parent['implementation_hashes']==plan['implementation_sha256']
            and parent['runtime']==receipt['runtime'] and parent['validation_hashes']==monitoring,'Parent runtime/source/monitoring drift')
        audit.check(receipt['learning_rate_schedule']==dict(kind='constant continuation',value=spec['continuation_lr'],
            starts_after_step=spec['parent_step']),'Continuation learning-rate schedule differs')
    else:audit.check(identity['checkpoint_schema']=='hidden-factor-weights-v1' and parent is None,'Fresh large-model ancestry differs')


def read_records(audit,folder,n,identity):
    rows=[];hashes=[]
    audit.check({p.name for p in folder.glob('seq_*.json')}=={f'seq_{i:06d}.json' for i in range(n)},'Missing/extra saved sequence records')
    for i in range(n):
        path=folder/f'seq_{i:06d}';record=audit.json(path.with_suffix('.json'))
        audit.check(record['schema']=='hidden-factor-evaluation-v1' and record['identity']==dict(identity,sequence_index=i),
            'Per-sequence identity differs')
        rows.append(audit.npz(path.with_suffix('.npz'),record['data_sha256']));hashes.append(record['data_sha256'])
    return rows,fingerprint(hashes)


def run(directory=INPUT,baseline=BASELINE,plan_path=PLAN):
    started=time.monotonic();directory=Path(directory);baseline=Path(baseline);audit=Audit()
    plan=audit.json(plan_path);plan_sha=audit.hashes[str(Path(plan_path).resolve())]
    summary=audit.json(directory/'comparison.json');summary_sha=audit.hashes[str((directory/'comparison.json').resolve())]
    audit.check(plan['schema']=='hidden-factor-scaling-plan-v1' and plan['source_dimensions']==[2,8]
        and plan['seeds']==[0,1] and set(plan['arms'])=={'longsmall','large'},'Wrong planned experiment')
    test=plan['test'];n=test['n_evaluate'];times=np.asarray(test['times'])
    audit.check(n==256 and test['n_planned']==4096 and test['length']==1025 and test['role']=='development'
        and times.tolist()==[1,4,16,64,256,1024],'Wrong test scope')
    audit.check(summary['schema']=='hidden-factor-scaling-comparison-v1' and summary['plan_sha256']==plan_sha
        and summary['test_role']=='development' and summary['evaluated_n_sequences']==n
        and summary['planned_n_sequences']==4096 and summary['sequence_length']==1025
        and summary['times']==test['times'] and summary['dimensions']==[2,8]
        and summary['cohort_id']==test['cohort_id'] and summary['cohort_data_sha256']==test['data_sha256']
        and summary['complete_for_requested_comparison'] is True and summary['full_cohort_complete'] is False
        and summary['new_bayesian_computation'] is False and summary['baseline']==plan['baseline']
        and summary['primary']==plan['primary'],'Comparison plan/scope differs')
    code_hashes={name:sha(audit.raw(ROOT/'code/transformer'/name)) for name in
        (*plan['implementation_sha256'],'train_hidden_factor_scaling.py','hidden_factor_scaling_eval.py')}
    audit.check(all(code_hashes[k]==v for k,v in plan['implementation_sha256'].items()),'Frozen implementation changed')
    # Reuse the independently implemented oracle/reference/stratum/paired audit.
    audit.raw(BASE_AUDITOR);module_spec=importlib.util.spec_from_file_location('hidden_factor_baseline_independent',BASE_AUDITOR)
    baseline_audit=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(baseline_audit)
    baseline_receipt=baseline_audit.run(baseline)
    audit.check(baseline_receipt['passed'] and baseline_receipt['summary_sha256']==plan['baseline']['summary_sha256'],
        'Frozen baseline independent audit failed')
    cohort=audit.json(baseline/'cohort.json');data=audit.npz(baseline/'cohort.npz',test['data_sha256'])
    audit.check(cohort['cohort_id']==test['cohort_id'] and fingerprint(cohort['identity'])==test['cohort_id'],
        'Frozen cohort identity differs')
    ref_manifest=audit.json(baseline/'references/manifest.json')
    references,ref_hash=read_records(audit,baseline/'references',n,ref_manifest['identity'])
    audit.check(ref_hash==plan['baseline']['reference_records_sha256']==summary['reference_records_sha256'],'Reference data changed')
    predictors={'bayes_hidden':{m:np.asarray([r['bayes_hidden_'+m] for r in references]) for m in METRICS}}
    for seed in (0,1):
        folder=baseline/'models'/f'seed{seed}';manifest=audit.json(folder/'manifest.json')
        rows,digest=read_records(audit,folder,n,manifest['identity'])
        audit.check(digest==plan['baseline']['model_records_sha256'][f'seed{seed}'],'Baseline model records changed')
        predictors[f'baseline_seed{seed}']={m:np.asarray([r[m] for r in rows]) for m in METRICS}
    identities={};pairs=[(f'baseline_seed{s}','bayes_hidden') for s in (0,1)]
    for arm in ('longsmall','large'):
        for seed in (0,1):
            label=f'{arm}_seed{seed}';folder=directory/'eval'/arm/'models'/f'seed{seed}'
            manifest=audit.json(folder/'manifest.json');ident=manifest['identity']
            audit.check(manifest['schema']=='hidden-factor-scaling-evaluation-v1'
                and manifest['n_seq']==manifest['evaluated_n_sequences']==manifest['requested_n_sequences']==n
                and manifest['planned_n_sequences']==4096 and manifest['complete_for_requested_pilot'] is True
                and manifest['partial_evaluation'] is True and manifest['complete'] is False,'Incomplete arm/seed evaluation')
            audit.check(ident['schema']=='hidden-factor-scaling-evaluation-v1' and ident['arm']==arm
                and ident['name']==f'seed{seed}' and ident['plan_sha256']==plan_sha and ident['test_role']=='development'
                and ident['cohort_id']==test['cohort_id'] and ident['reference_records_sha256']==ref_hash
                and ident['evaluator_sha256']==code_hashes['hidden_factor_scaling_eval.py'],'Model evaluator/input identity differs')
            causal=manifest['causality_check']
            audit.check(causal['passed'] is True and causal['max_absolute_logit_difference']<=causal['tolerance']<=2e-5,
                'Missing/failed recorded causal alignment check')
            checkpoint_metadata(audit,ident,plan,plan_sha,arm,seed,code_hashes)
            rows,digest=read_records(audit,folder,n,ident)
            audit.check(summary['model_records_sha256'][label]==digest and summary['model_provenance'][label]==ident,
                'Summary model provenance or record hash differs')
            for i,row in enumerate(rows):
                audit.close('record times',row['times'],times,0)
                p=row['predictive'];r=references[i]['oracle_probs']
                audit.check(p.shape==(6,16) and np.isfinite(p).all() and np.all(p>0) and np.all(p<=1+1e-12),
                    'Invalid predictive distribution')
                audit.close('prediction normalization',p.sum(1),np.ones(6),1e-11)
                kl=np.maximum(0,np.sum(r*(np.log2(r)-np.log2(p)),axis=1))
                realized=-np.log2(p[np.arange(6),data['tokens'][i,times]])
                audit.close('recomputed next-token KL',row['next_kl_bits'],kl,2e-11)
                audit.close('recomputed next-token loss',row['next_loss_bits'],realized,2e-11)
                cumulative=np.asarray(row['cumulative_loss_bits'])
                audit.check(cumulative.shape==(6,) and np.isfinite(cumulative).all() and np.all(cumulative>=0)
                    and np.all(np.diff(cumulative)>=realized[1:]-2e-9),'Invalid cumulative score or interval lower bound')
                audit.close('first cumulative score',cumulative[:1],realized[:1],2e-11)
            predictors[label]={m:np.asarray([r[m] for r in rows]) for m in METRICS};identities[label]=ident
            pairs.extend([(label,'bayes_hidden'),(label,f'baseline_seed{seed}')])
    audit.check(len({v['checkpoint_sha256'] for v in identities.values()}|set(plan['baseline']['checkpoint_sha256'].values()))==6,
        'Model checkpoint identities are not distinct')
    audit.check(len({fingerprint(v['checkpoint_receipt']['validation_hashes']) for v in identities.values()})==1,
        'Monitoring stream changed between arms/seeds')
    audit.check([g['id'] for g in summary['groups']]==['full_prior','entropy_0','entropy_1','entropy_2','entropy_3'],
        'Missing/extra source group')
    for group in summary['groups']:
        mask=np.ones(n,dtype=bool) if group['id']=='full_prior' else data['stratum'][:n]==int(group['id'][-1])
        expected_label=None if group['id']=='full_prior' else cohort['stratum_labels'][int(group['id'][-1])]
        audit.check(group['status']=='complete' and group['n_seq']==int(mask.sum()) and group['entropy_label']==expected_label,
            'Group membership/count/label differs')
        audit.check(set(group['predictors'])==set(predictors) and set(group['paired'])=={a+'_minus_'+b for a,b in pairs},
            'Missing/extra predictor or paired comparison')
        for kind in ('predictors','paired'):
            for name,statistics in group[kind].items():
                left,right=name.split('_minus_') if kind=='paired' else (name,None)
                audit.check(set(statistics)==set(METRICS),'Missing/extra metric')
                for metric in METRICS:
                    values=predictors[left][metric][mask]
                    if right is not None:values=values-predictors[right][metric][mask]
                    saved=statistics[metric];audit.check(saved['n_seq']==int(mask.sum()),'Statistic sequence count differs')
                    audit.close('summary means',saved['mean'],values.mean(0))
                    audit.close('summary paired/ordinary SE',saved['se'],values.std(0,ddof=1)/math.sqrt(len(values)))
    audit.check(all(sha(Path(p).read_bytes())==h for p,h in audit.hashes.items()),'An input changed during audit')
    full=next(g for g in summary['groups'] if g['id']=='full_prior')
    receipt=dict(schema='hidden-factor-scaling-independent-audit-v1',passed=True,validator_sha256=sha(Path(__file__).read_bytes()),
        plan_sha256=plan_sha,summary_sha256=summary_sha,baseline_validator_sha256=sha(BASE_AUDITOR.read_bytes()),
        baseline_input_tree_sha256=baseline_receipt['input_tree_sha256'],input_tree_sha256=fingerprint(audit.hashes),
        hashed_files=len(audit.hashes),checks_passed=audit.checks,prediction_records_checked=4*n,baseline_records_rechecked=3*n,
        evaluated_sequences=n,planned_sequences=4096,sequence_length=1025,dimensions=[2,8],times=times.tolist(),groups_checked=5,
        cohort_id=test['cohort_id'],cohort_data_sha256=test['data_sha256'],
        maximum_absolute_errors=audit.errors,numpy_version=np.__version__,full_prior=full,
        recorded_checkpoints={name:{k:v[k] for k in ('checkpoint_sha256','checkpoint_step','completed_sequences','model','n_params')}
                              for name,v in identities.items()},
        no_inference_or_reference_recomputation=True,test_role='development',elapsed_seconds=time.monotonic()-started,
        limitations=['Checkpoint bytes were not transferred; source, configuration, ancestry, final-step and recorded checkpoint identities were checked, not model weights.',
            'Transformer probabilities are saved at six times only. Next-token metrics are independently recomputed. Cumulative aggregation and paired standard errors are recomputed from saved cumulative scores; full tokenwise sums cannot be reconstructed without new inference.',
            'The original 256-sequence pilot motivated these arms and is development evidence, not untouched confirmation. Training seeds are reported separately; sequence standard errors do not estimate training-seed variability.',
            'Recorded causal-alignment and monitoring-stream identities are checked; transformer inference and monitoring token generation are not rerun.'])
    return receipt,summary


def corruption_checks(directory,baseline,plan_path):
    """Bounded in-memory faults; no transferred artifact is changed."""
    from unittest.mock import patch
    original=Audit.json;passed=[]
    for kind in ('paired_se','mean','checkpoint_step','source_config','sequence_identity','array_hash'):
        def altered(self,path):
            value=original(self,path);path=Path(path)
            if path.name=='comparison.json' and kind in ('mean','paired_se'):
                group=value['groups'][0]
                if kind=='mean':group['predictors']['large_seed0']['next_kl_bits']['mean'][-1]+=.01
                else:group['paired']['large_seed1_minus_baseline_seed1']['cumulative_loss_bits']['se'][-1]+=.1
            if path==Path(directory)/'eval/longsmall/models/seed0/manifest.json':
                if kind=='checkpoint_step':value['identity']['checkpoint_step']-=1
                if kind=='source_config':value['identity']['checkpoint_receipt']['config']['source_dimensions']=[4,4]
            if path==Path(directory)/'eval/longsmall/models/seed0/seq_000000.json':
                if kind=='sequence_identity':value['identity']['sequence_index']=1
                if kind=='array_hash':value['data_sha256']='0'*64
            return value
        with patch.object(Audit,'json',altered):
            try:run(directory,baseline,plan_path)
            except ValueError:passed.append(kind)
            else:raise ValueError('Corruption was not rejected: '+kind)
    return passed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=INPUT)
    p.add_argument('--baseline',type=Path,default=BASELINE);p.add_argument('--plan',type=Path,default=PLAN)
    p.add_argument('--out',type=Path,help='Write the passed receipt and byte-identical checked comparison here')
    p.add_argument('--check-corruption',action='store_true');args=p.parse_args()
    receipt,_=run(args.input,args.baseline,args.plan)
    if args.check_corruption:receipt['passed_corruption_checks']=corruption_checks(args.input,args.baseline,args.plan)
    if args.out:
        out=args.out.resolve();require(not out.is_relative_to(args.input.resolve()) and not out.is_relative_to(args.baseline.resolve()),
            'Output must be separate from scientific inputs')
        raw=(args.input/'comparison.json').read_bytes();require(sha(raw)==receipt['summary_sha256'],'Comparison changed before release')
        out.mkdir(parents=True,exist_ok=True);(out/'comparison.json').write_bytes(raw)
        (out/'validation.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('passed','prediction_records_checked','groups_checked','checks_passed','elapsed_seconds')}))


if __name__=='__main__':main()
