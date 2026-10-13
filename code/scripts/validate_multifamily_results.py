"""Independent post-retrieval audit; no model inference or training is rerun."""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path.cwd(); RAW=ROOT/'output/multifamily_results_2026-09-16'
OUT=ROOT/'results/transformer/multifamily_2026-09-16/validation.json'
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(test,msg):
    if not test: raise AssertionError(msg)
def close(x,y,msg,tol=1e-10): require(np.allclose(x,y,atol=tol,rtol=0),msg)
summary=load(RAW/'comparison.json'); source=load(RAW/'heldout/manifest.json')
require(fp(source['spec'])==source['plan_id']==summary['source_plan_id'],'source plan identity')
require(load(RAW/'heldout/plan.json')==source['spec'],'source plan file')
spec=source['spec']; times=np.array(spec['prefixes']); metrics=['next_kl_bits','next_loss_bits','cumulative_loss_bits','cumulative_realized_regret_bits']
require(spec['n_tokens']==1025 and spec['alphabet_size']==64,'cohort dimensions')
for key,file in [('source_sha256','source.py'),('families_sha256','families.py'),('evaluator_sha256','eval_mixture.py')]:
    require(spec[key]==sha(ROOT/'code/transformer'/file),'source implementation hash')
all_seeds=[]; oracle_cumulative={}; conditions={}; max_row_error=0.; file_counts={'source_files':0,'evaluation_files':0,'models':0,'evaluated_model_cohorts':0}
sys.path.insert(0,str(ROOT/'code/transformer'))
from eval_mixture import draw_source
for e in source['cohorts']:
    c=e['condition']; cid=c['id']; conditions[cid]=c
    for k,h in [('data','data_sha256'),('record','record_sha256')]:
        require(sha(RAW/'heldout'/e[k])==e[h],f'{cid} source {k} checksum');file_counts['source_files']+=1
    record=load(RAW/'heldout'/e['record']); require(record['plan_id']==source['plan_id'] and record['condition']==c,'source record identity')
    with np.load(RAW/'heldout'/e['data'],allow_pickle=False) as z:
        x=z['tokens']; matrix=z['transition_probs']; q=z['initial_probs']; rows=z['oracle_rows']; seeds=z['sequence_seed']; n=c['n_seq']
        require(x.shape==(n,1025) and matrix.shape==(n,64,64) and rows.shape==(n,6,64),'source shape')
        require(np.all(x>=0) and np.all(x<64),'source tokens range')
        require(np.all(matrix>0) and np.all(q>0),'positive source probabilities')
        close(matrix.sum(-1),1.,'transition normalization',1e-12);close(q.sum(-1),1.,'initial normalization',1e-12)
        max_row_error=max(max_row_error,float(np.max(np.abs(matrix.sum(-1)-1))))
        idx=np.arange(n)[:,None];actual_rows=matrix[idx,x[:,times-1]]
        close(rows,actual_rows,'oracle next-row alignment',0)
        loss=-np.log2(matrix[idx,x[:,:-1],x[:,1:]])
        close(z['oracle_next_loss_bits'],loss[:,times-1],'oracle next-loss indexing')
        close(z['oracle_cumulative_loss_bits'],np.cumsum(loss,axis=1)[:,times-1],'oracle cumulative loss indexing')
        close(z['source_entropy_bits'],-np.sum(rows*np.log2(rows),axis=-1),'source entropy')
        oracle_cumulative[cid]=z['oracle_cumulative_loss_bits'].copy()
        for i,seed in enumerate(seeds):
            identity=[spec['seed_namespace'],spec['master_seed'],cid,i]
            expected=np.frombuffer(hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).digest()[:16],dtype='<u4')
            require(np.array_equal(seed,expected),'independent seed namespace')
            all_seeds.append(tuple(seed))
        for i in (0,n-1):
            replay=draw_source(c,1025,seeds[i].tolist())
            require(np.array_equal(replay['tokens'],x[i]),'source generator replay tokens')
            close(replay['initial_probs'],q[i],'source generator replay initial')
            close(replay['transition_probs'],matrix[i],'source generator replay transitions')
            require(replay['parameters']==record['parameters'][i],'source parameter replay')
require(len(all_seeds)==8192 and len(set(all_seeds))==8192,'sequence independence seeds')
require(len(conditions)==23 and sum(c['kind']=='prior' for c in conditions.values())==3,'cohort plan')

model_arrays={}; model_receipts={}; max_stat_error=0.; causal_max=0.
def check_stats(raw,reported,msg):
    global max_stat_error
    require(raw.ndim==2 and raw.shape[1]==6 and np.isfinite(raw).all(),msg+' finite scores')
    mean=raw.mean(0); se=raw.std(0,ddof=1)/math.sqrt(len(raw))
    require(reported['n_seq']==len(raw),msg+' sample count')
    for value,key in [(mean,'mean'),(se,'se')]:
        max_stat_error=max(max_stat_error,float(np.max(np.abs(value-np.array(reported[key])))))
        close(value,reported[key],msg+' '+key)

specialist_paths={'product_specialist':'prod_8x8','uniform_specialist':'lsa_a0_d64','shared_specialist_seed0':'d64','shared_specialist_seed1':'d64_seed1'}
for name in summary['models']:
    directory=RAW/'eval'/name; manifest=load(directory/'manifest.json'); ident=manifest['identity']; checkpoint=manifest['checkpoint_record']; file_counts['models']+=1
    require(manifest['run_id']==fp(ident)==summary['models'][name]['run_id'],'model run identity')
    require(ident['source_plan_id']==source['plan_id'],'model source identity')
    require(ident['evaluator_sha256']==spec['evaluator_sha256'],'model evaluator hash')
    require(ident['model_code_sha256']==sha(ROOT/'code/transformer/model.py'),'model implementation hash')
    require(ident['checkpoint_sha256']==summary['models'][name]['checkpoint_sha256'],'summary checkpoint hash')
    expected={cid for cid,c in conditions.items() if c['family'] in manifest['evaluated_families']}
    require({e['condition']['id'] for e in manifest['cohorts']}==expected,'complete model cohorts')
    require(checkpoint['step']==checkpoint['predeclared_final_step'],'predeclared final checkpoint')
    model_arrays[name]={}
    for e in manifest['cohorts']:
        cid=e['condition']['id']; c=conditions[cid]; file_counts['evaluated_model_cohorts']+=1
        for k,h in [('data','data_sha256'),('record','record_sha256')]:
            require(sha(directory/e[k])==e[h],name+' '+cid+' evaluation '+k+' checksum');file_counts['evaluation_files']+=1
        rec=load(directory/e['record']); require(rec['condition']==c and rec['run_id']==manifest['run_id'] and rec['identity']==ident,'evaluation record identity')
        require(rec['source_data_sha256']==next(a['data_sha256'] for a in source['cohorts'] if a['condition']['id']==cid),'evaluation source checksum')
        require(rec['checkpoint_record']==checkpoint,'evaluation checkpoint record')
        causal=rec['causality_check'];require(causal['passed'] and causal['max_absolute_logit_difference']<=causal['tolerance'],'causality check')
        causal_max=max(causal_max,causal['max_absolute_logit_difference'])
        with np.load(directory/e['data'],allow_pickle=False) as z, np.load(RAW/'heldout'/f'{cid}.npz',allow_pickle=False) as source_z:
            arrays={k:z[k].copy() for k in metrics+['sequence_seed','prefixes']}
            require(np.array_equal(z['prefixes'],times),'prediction times')
            require(np.array_equal(z['sequence_seed'],source_z['sequence_seed']),'model uses source sequence identities')
        for metric in metrics: check_stats(arrays[metric],summary['models'][name]['cohorts'][cid]['metrics'][metric],name+' '+cid+' '+metric)
        close(arrays['cumulative_realized_regret_bits'],arrays['cumulative_loss_bits']-oracle_cumulative[cid],'regret subtraction')
        require(np.all(arrays['next_kl_bits']>=0) and np.all(arrays['next_loss_bits']>=0),'nonnegative KL and loss')
        model_arrays[name][cid]=arrays
    if name.startswith('pooled'):
        seed=name[-1];runpath=RAW/'train'/f'seed{seed}';weight_path=runpath/'model_300000.pt';cfg=load(runpath/'config.json');status=load(runpath/'status.json')
        require(cfg==checkpoint['config'],'pooled config/checkpoint agreement')
        require(fp(cfg['config'])==cfg['config_hash']==status['config_hash'],'training config hash')
        require(status['complete'] and status['reason']=='completed' and status['step']==status['production_steps']==300000,'training completed final budget')
        require(status['completed_counts']==dict(product=6400000,shared_base=6400000,uniform_base=6400000),'equal family budgets')
        training_summary=status
    else:
        weight_path=ROOT/'results/transformer/runs'/specialist_paths[name]/'model_100000.pt';cfg=checkpoint['config']; require(cfg['steps']==100000 and cfg['batch']==64,'specialist 6.4m budget')
        training_summary={'step':cfg['steps'],'sequences':cfg['steps']*cfg['batch'],'seed':cfg['seed']}
    require(sha(weight_path)==ident['checkpoint_sha256'],'checkpoint weight file checksum')
    import torch
    weights=torch.load(weight_path,map_location='cpu',weights_only=False)
    require(weights['step']==checkpoint['step'] and fp(weights.get('cfg',weights.get('config')))==fp(checkpoint['config']),'loaded checkpoint identity')
    if name.startswith('pooled'):require(weights['completed_counts']==status['completed_counts'],'checkpoint completed training counts')
    model_receipts[name]={'run_id':manifest['run_id'],'checkpoint_sha256':ident['checkpoint_sha256'],'model_config':checkpoint['model_config'],'training':training_summary,'evaluation_manifest_sha256':sha(directory/'manifest.json')}
require(len({json.dumps(x['model_config'],sort_keys=True) for x in model_receipts.values()})==1,'identical model architectures')
for pair,cohorts in summary['paired_differences'].items():
    left,right=pair.split('_minus_');require(set(cohorts)==set(model_arrays[left])&set(model_arrays[right]),'paired cohort completeness')
    for cid,stats in cohorts.items():
        a,b=model_arrays[left][cid],model_arrays[right][cid]
        require(np.array_equal(a['sequence_seed'],b['sequence_seed']),'paired sequence identity')
        for metric in metrics: check_stats(a[metric]-b[metric],stats[metric],pair+' '+cid+' '+metric)
receipt={'schema':'multifamily-independent-validation-v1','date':'2026-09-16','status':'passed','summary_path':'results/transformer/multifamily_2026-09-16/comparison.json','summary_sha256':sha(RAW/'comparison.json'),'raw_root':'output/multifamily_results_2026-09-16','source_plan_id':source['plan_id'],'source_manifest_sha256':sha(RAW/'heldout/manifest.json'),'cluster_job':{'id':4234451,'state':'COMPLETED','elapsed':'00:02:34','exit_code':'0:0'},'counts':dict(file_counts,independent_sequences=8192,conditions=23,prior_sequences_per_family=1024,conditional_sequences_per_condition=256,sequence_length=1025,source_replayed_sequences=46),'times':times.tolist(),'checks':['All source and evaluation data/record hashes match their manifests.','Source plan, model run IDs and current source/evaluation implementation hashes match.','All 8192 sequence seeds are unique and match the frozen independent namespace.','All oracle rows, normalization, entropy, next losses and cumulative losses independently recomputed from saved transition matrices.','First and last source sequence per condition replay exactly in token identity and parameters; probabilities agree within 1e-10.','All 82 model-cohort records contain complete scores and valid saved causal-alignment checks.','All summary means and standard errors, including paired differences computed before averaging, independently recomputed from per-sequence records.','All six weight-file hashes and loaded final checkpoint metadata match the evaluation records.','Both pooled checkpoints and status files verify 300000 updates and 6400000 examples from each family; each specialist used 100000 updates and 6400000 examples.','All six model architectures are identical.'],'maximum_summary_statistic_absolute_difference':max_stat_error,'maximum_transition_normalization_error':max_row_error,'maximum_recorded_causal_logit_difference':causal_max,'models':model_receipts,'limitations':['The audit verifies saved prediction scores and metadata; it does not rerun all transformer inference.','Training-seed variability is not included in sequence standard errors; the two seeds are kept separate.','Product and uniform-base specialists each have one training seed; both pooled seeds use that same specialist.','Family-aware Bayesian and Bayesian-mixture references remain pending; pooled-minus-specialist loss is not the Bayesian family-uncertainty cost.','Equal examples per family means pooled training used three times the total examples and optimizer steps, so gains do not isolate transfer from extra training.']}
OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['status','counts','maximum_summary_statistic_absolute_difference','maximum_recorded_causal_logit_difference']},indent=2))
