"""Replay complete-cohort statistics and provenance without new model inference."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'output/kuma_completion_2026-09-23/raw'
DEST=ROOT/'results/transformer/larger_models_2026-09-23'
sys.path[:0]=[str(ROOT/'code/scripts'),str(ROOT/'code/transformer')]
from pooled_transformer_large import configuration
from eval_mixture import fingerprint, load_manifest

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return dict(z)
def stats(x):return dict(mean=x.mean(0),se=x.std(0,ddof=1)/np.sqrt(len(x)))
def same(a,b):
 assert np.allclose(a,b,atol=2e-12,rtol=0),float(np.max(np.abs(np.asarray(a)-b)))
def causality(d):
 assert d['passed'] is True and d['tolerance']==2e-5 and np.isfinite(d['max_absolute_logit_difference']) and d['max_absolute_logit_difference']<=2e-5
receipt=read(RAW/'completion_receipt.json')
for name,digest in receipt['file_sha256'].items():assert sha(RAW/name)==digest,name
assert len(receipt['accounting'].strip().splitlines())==6
for line in receipt['accounting'].strip().splitlines():
 parts=line.split('|');assert parts[2]=='COMPLETED' and parts[4]=='0:0'
result={};audit=dict(passed=False,independent_statistical_replay=True,source_hashes={},models={},endpoints=[])
sp=RAW/'specialist_transformer_large_2026-09-22';plan=ROOT/'code/transformer/configs/specialist_transformer_large_plan_2026-09-22.json'
cohorts=ROOT/'output/specialist_transformer_large_2026-09-22/cohorts';manifest=read(cohorts/'manifest.json')
launch=read(sp/'train_submission.json');assert launch['plan_sha256']==sha(plan) and launch['cohort_manifest_sha256']==sha(cohorts/'manifest.json')
for name,digest in read(plan)['implementation_sha256'].items():assert sha(ROOT/name)==digest
for name,digest in launch['implementation_sha256'].items():assert sha(ROOT/name)==digest
for family in ('product','uniform_base'):
 entry=manifest['families'][family];path=cohorts/entry['file'];assert sha(path)==entry['sha256'];data=arrays(path);times=data['times'];ng=len(entry['groups'])
 assert np.array_equal(times,2**np.arange(11)) and len(data['tokens'])==1024
 original=ROOT/f'results/transformer/conditional_accuracy/{family}/summary.json';assert sha(original)==entry['original_summary_sha256']
 old=read(original);oldgroups=old['cases'] if family=='product' else old['panels']
 values={key:[] for key in ('mean','se','gap_mean','gap_se','improvement_mean','improvement_se')}
 smalls=[];bayes=[];counts=[]
 for g in entry['groups']:
  mask=data['groups']==g['id'];assert mask.sum()==g['n_sequences']==oldgroups[g['id']]['n_sequences'];counts.append(int(mask.sum()))
  small=stats(data['small_regret'][mask]);bay=stats(data['bayes_regret'][mask]);smalls.append(small['mean']);bayes.append(bay['mean'])
  for stat in ('mean','se'):
   same(small[stat],oldgroups[g['id']]['transformer'][stat]);same(bay[stat],oldgroups[g['id']]['informed_bayes'][stat])
 for seed in (0,1):
  cfg=read(sp/f'train/{family}/seed{seed}/config.json');status=read(sp/f'train/{family}/seed{seed}/status.json')
  assert {k:cfg[k] for k in ('n_layer','n_head','d_model','steps','batch','seed')}==dict(n_layer=4,n_head=8,d_model=256,steps=100000,batch=64,seed=seed)
  assert status['passed'] is True and status['complete'] is True and status['completed_steps']==100000 and status['plan_sha256']==sha(plan) and status['launch_binding']==launch
  out=sp/f'eval/{family}/seed{seed}';summary=read(out/'summary.json');scores=arrays(out/'scores.npz')
  assert summary['complete'] is True and summary['family']==family and summary['seed']==seed and summary['cohort_sha256']==entry['sha256'] and summary['plan_sha256']==sha(plan)
  assert summary['checkpoint_sha256']==receipt['checkpoint_sha256'][f'{sp.name}/train/{family}/seed{seed}/model_100000.pt']
  causality(summary['causality']);assert np.array_equal(scores['times'],times) and np.array_equal(scores['groups'],data['groups'])
  regret=scores['regret'];assert regret.shape==(1024,11) and np.isfinite(regret).all() and regret.min()>=-1e-8 and len(summary['groups'])==ng
  per={k:[] for k in values}
  for g in summary['groups']:
   mask=data['groups']==g['id'];assert g['condition']==entry['groups'][g['id']]['condition'] and g['n_sequences']==int(mask.sum())
   quantities=dict(transformer=regret[mask],informed_bayes=data['bayes_regret'][mask],transformer_minus_bayes=regret[mask]-data['bayes_regret'][mask],improvement_over_small=data['small_regret'][mask]-regret[mask])
   for name,x in quantities.items():
    st=stats(x)
    for stat in ('mean','se'):same(st[stat],g[name][stat])
   for name,target in (('transformer',''),('transformer_minus_bayes','gap_'),('improvement_over_small','improvement_')):
    for stat in ('mean','se'):per[target+stat].append(np.asarray(g[name][stat]))
   for t in (32,128,1024):
    j=list(times).index(t);audit['endpoints'].append(dict(family=family,seed=seed,condition=g['condition'],time=t,small_mean=float(data['small_regret'][mask,j].mean()),bayes_mean=g['informed_bayes']['mean'][j],large_mean=g['transformer']['mean'][j],large_se=g['transformer']['se'][j],gap=g['transformer_minus_bayes']['mean'][j],gap_se=g['transformer_minus_bayes']['se'][j],improvement=g['improvement_over_small']['mean'][j],improvement_se=g['improvement_over_small']['se'][j]))
  for k in values:values[k].append(np.asarray(per[k]))
  audit['models'][f'{family}_seed{seed}']=dict(checkpoint_sha256=summary['checkpoint_sha256'],config=cfg,causality=summary['causality'],source_relationship=status['source_relationship'])
 result.update({family+'_'+k:np.asarray(v) for k,v in values.items()})
 result.update({family+'_times':times,family+'_small_mean':np.asarray(smalls),family+'_bayes_mean':np.asarray(bayes),family+'_counts':np.asarray(counts)})
 audit['source_hashes'][str(path.relative_to(ROOT))]=sha(path)
pooled=RAW/'pooled_transformer_large_2026-09-22';plan=ROOT/'code/transformer/configs/pooled_transformer_large_plan_2026-09-22.json';launch=read(pooled/'train_submission.json')
assert launch['plan_sha256']==sha(plan)
sources=ROOT/'output/multifamily_results_2026-09-16/heldout';source_manifest=load_manifest(sources)
assert sha(sources/'manifest.json')==launch['source_manifest_sha256']==sha(RAW/'baseline/heldout_manifest.json')
source_by_id={e['condition']['id']:e for e in source_manifest['cohorts']}
original=read(ROOT/'results/transformer/multifamily_bayes_2026-09-16/next_token_regret.json')
assert original['source_plan_id']==source_manifest['plan_id']
values={k:[] for k in ('mean','se','small_mean','small_se','improvement_mean','improvement_se','family_mean','family_se')}
for seed in (0,1):
 cfg=read(pooled/f'train/seed{seed}/config.json');status=read(pooled/f'train/seed{seed}/status.json');runtime=read(pooled/f'train/runtime_seed{seed}.json')
 assert cfg['config']==configuration(read(plan),seed) and cfg['n_params']==3454464
 assert status['complete'] is True and status['step']==300000 and status['completed_counts']==dict(product=6400000,shared_base=6400000,uniform_base=6400000)
 assert runtime['return_code']==0 and runtime['plan_sha256']==sha(plan) and runtime['implementation_sha256']==launch['implementation_sha256']
 directory=pooled/f'eval/pooled_large_seed{seed}';m=read(directory/'manifest.json');identity=m['identity']
 assert identity['checkpoint_sha256']==receipt['checkpoint_sha256'][f'{pooled.name}/train/seed{seed}/model_300000.pt'] and m['run_id']==fingerprint(identity) and identity['source_plan_id']==source_manifest['plan_id']
 complete=read(pooled/f'eval/complete_seed{seed}.json');assert complete['complete'] is True and complete['run_id']==m['run_id'] and complete['cohorts']==len(source_by_id)==23
 assert {e['condition']['id'] for e in m['cohorts']}==set(source_by_id)
 scored={}
 for e in m['cohorts']:
  condition=e['condition'];name=condition['id'];source=source_by_id[name]
  assert condition==source['condition'] and sha(directory/e['data'])==e['data_sha256'] and sha(directory/e['record'])==e['record_sha256']
  rec=read(directory/e['record']);assert rec['run_id']==m['run_id'] and rec['identity']==identity and rec['source_data_sha256']==source['data_sha256'];causality(rec['causality_check'])
  saved=arrays(directory/e['data'])
  with np.load(sources/source['data'],allow_pickle=False) as z:assert np.array_equal(saved['sequence_seed'],z['sequence_seed'])
  assert np.array_equal(saved['prefixes'],[1,4,16,64,256,1024]) and saved['next_kl_bits'].shape==(condition['n_seq'],6)
  assert np.isfinite(saved['next_kl_bits']).all() and saved['next_kl_bits'].min()>=0
  scored[name]=saved
 large=[];small=[]
 for family in ('product','shared_base','uniform_base'):
  new=scored[family+'_prior'];p=RAW/f'baseline/pooled_seed{seed}/{family}_prior.npz';rec=read(p.with_suffix('.json'));assert sha(p)==rec['data_sha256'] and rec['source_data_sha256']==source_by_id[family+'_prior']['data_sha256']
  old=arrays(p);assert np.array_equal(old['sequence_seed'],new['sequence_seed']) and np.array_equal(old['prefixes'],new['prefixes'])
  large.append(new['next_kl_bits']);small.append(old['next_kl_bits'])
 large=np.asarray(large);small=np.asarray(small);assert large.shape==small.shape==(3,1024,6)
 for name,x in (('',large),('small_',small),('improvement_',small-large)):
  mean=x.mean(axis=(0,1));se=np.sqrt(np.sum(x.var(axis=1,ddof=1)/1024,axis=0))/3
  values[name+'mean'].append(mean);values[name+'se'].append(se)
 for stat in ('mean','se'):same(values['small_'+stat][-1],original['models'][f'pooled_seed{seed}']['next_token_regret'][stat])
 values['family_mean'].append(large.mean(1));values['family_se'].append(large.std(1,ddof=1)/32)
 audit['models'][f'pooled_seed{seed}']=dict(checkpoint_sha256=identity['checkpoint_sha256'],config=cfg,cohorts=23,source_relationship='Matched indexed streams, same300000 updates and19.2M sequences; only architecture changes.')
 for j,t in enumerate([1,4,16,64,256,1024]):audit['endpoints'].append(dict(family='pooled',seed=seed,time=t,small_mean=float(values['small_mean'][-1][j]),large_mean=float(values['mean'][-1][j]),large_se=float(values['se'][-1][j]),improvement=float(values['improvement_mean'][-1][j]),improvement_se=float(values['improvement_se'][-1][j])))
result.update({'pooled_'+k:np.asarray(v) for k,v in values.items()});result['pooled_times']=np.array([1,4,16,64,256,1024])
if '--check-only' in sys.argv:
 saved=arrays(DEST/'curves.npz');assert set(saved)==set(result)
 for key,value in result.items():assert np.array_equal(saved[key],value),key
 with gzip.open(DEST/'manifest.json.gz','rt') as f:release=json.load(f)
 assert release['passed'] is True
 for name,digest in release['files'].items():assert sha(DEST/name)==digest,name
 with gzip.open(DEST/'provenance.json.gz','rt') as f:previous=json.load(f)
 assert previous['passed'] is True and previous['models']==audit['models'] and previous['endpoints']==audit['endpoints'] and previous['validation_script']==Path(__file__).read_text()
 print('PASS: independent replay and committed release hashes match.');sys.exit(0)
DEST.mkdir(parents=True,exist_ok=True);np.savez_compressed(DEST/'curves.npz',**result)
audit.update(passed=True,completion=receipt,input_bundle_sha256=sha(RAW.parent/'raw_bundle.tar.gz'),validation_script=Path(__file__).read_text(),curve_sha256=sha(DEST/'curves.npz'))
with gzip.GzipFile(filename=str(DEST/'provenance.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(audit,sort_keys=True,allow_nan=False).encode())
manifest=dict(schema='larger-transformer-release-v1',passed=True,files={n:sha(DEST/n) for n in ('curves.npz','provenance.json.gz')},seeds=[0,1],architecture=dict(layers=4,width=256,heads=8,parameters=3454464),training=dict(pooled_updates=300000,pooled_sequences=19200000,specialist_updates=100000,specialist_sequences=6400000),cohorts=dict(product=1024,uniform_base=1024,pooled=3072),uncertainty='Independent complete test sequences; paired differences, stratified across three families for pooled; training runs never pooled.')
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with gzip.GzipFile(filename=str(DEST/'manifest.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(manifest,sort_keys=True).encode())
(RAW.parent/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('PASS: all6 models,46 pooled evaluations,4 specialist scores, original cohort statistics and paired comparisons replayed.')
for row in audit['endpoints']:
 if row['seed']==0 and ((row['family']=='pooled') or(row['family']=='product' and row['time']==32) or(row['family']=='uniform_base' and row['condition']['source_depth'] in (0,1,4,8) and row['time']==1024)):print(json.dumps(row))
