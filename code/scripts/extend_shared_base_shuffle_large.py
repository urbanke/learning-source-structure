"""Extend the frozen paired copying evaluation, retaining all 1024 old histories.

Only new source indices are sampled and scored. Every source/scoring chunk is
hash-bound and reusable; a pilot must fit the fixed resource bound before the
remaining work begins. No training, Bayes calculation, smoothing or selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
import shared_base_shuffle_eval as old

PLAN=ROOT/'code/transformer/configs/shared_base_shuffle_extension_2026-09-23.json'
TRAINING=ROOT/'code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json'
METRICS=('expected_loss_bits','regret_bits','realized_loss_bits')


def need(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def announce(event,**fields):print(json.dumps(dict(event=event,**fields)),flush=True)


def tasks(plan,pilot=False):
    starts=[plan['retained_sequences']] if pilot else range(plan['retained_sequences'],plan['n_sequences'],plan['chunk'])
    return [(d,left,min(left+plan['chunk'],plan['n_sequences'])) for d in plan['depths'] for left in starts]


def score_pair(previous,new,retained):
    merged={}
    for name in METRICS:
        need(previous[name].shape==(retained,1024) and np.isfinite(previous[name]).all(),'Incomplete retained score cohort')
        need(new[name].ndim==2 and new[name].shape[1]==1024 and np.isfinite(new[name]).all(),'Incomplete new score cohort')
        merged[name]=np.concatenate((previous[name],new[name]))
        need(np.array_equal(merged[name][:retained],previous[name]),'Retained scores changed')
    return merged


def validate_plan(path):
    p=read(path)
    need(p['schema']=='shared-base-shuffle-large-extension-v1' and p['n_sequences']==16384
         and p['retained_sequences']==1024 and p['depths']==[0,2,3] and p['seeds']==[0]
         and p['master_seed']==20260930 and p['dtype']=='float64' and p['d']==64 and p['n_tokens']==1025
         and p['chunk']==64 and p['bootstrap']==1000,'Production declaration changed')
    for name,digest in p['dependencies_sha256'].items():need(old.sha(ROOT/name)==digest,'Frozen dependency changed: '+name)
    return p


def validate_parent(p,parent):
    m,s,status=[read(parent/name) for name in ('manifest.json','summary.json','status.json')]
    need(old.sha(parent/'manifest.json')==p['parent_manifest_sha256']
         and old.sha(parent/'summary.json')==p['parent_summary_sha256']==status['summary_sha256']
         and status['complete'] and s['complete'] and s['scientific_reference_validated'],'Parent extension is not approved')
    for d in p['depths']:
        key=f'a1_L{d}';c=next(c for c in m['conditions'] if c['id']==key)
        need(c['indices']==list(range(1024)),'Parent sources reordered/incomplete')
        a=load(parent/'cohorts'/(key+'.npz'))
        need(str(a['manifest_sha256'])==p['parent_manifest_sha256']
             and old.checksums(a)==c['arrays_sha256'],'Parent source arrays changed')
        for seed in p['seeds']:
            for role in ('ordered','shuffled'):
                name=f'{role}_seed{seed}';f=parent/'eval'/name/(key+'.npz');r=read(f.parent/'protocol.json')
                need(old.sha(f)==s['input_sha256'][name+'/'+key]
                     and r['checkpoint_sha256']==p['checkpoint_sha256'][name],'Parent model scores changed')
                v=load(f)
                need(str(v['protocol_sha256'])==old.sha(f.parent/'protocol.json')
                     and str(v['source_tokens_sha256'])==c['arrays_sha256']['tokens']
                     and np.array_equal(v['times'],np.arange(1,1025)),'Parent score provenance changed')
                for metric in METRICS:need(v[metric].shape==(1024,1024) and np.isfinite(v[metric]).all(),'Parent scores incomplete')
    return m,s


def source_path(out,task):
    d,left,right=task;return out/'chunks'/f'a1_L{d}'/f'source_{left:06d}.npz'


def score_path(out,task,role,seed):
    d,left,right=task;return out/'chunks'/f'a1_L{d}'/f'{role}_seed{seed}_{left:06d}.npz'


def check_chunk(path,identity):
    r=read(path.with_suffix('.json'))
    need(r['identity']==identity and r['sha256']==old.sha(path),'Chunk identity or hash changed')
    return r


def draw_source(args):
    out,task,plan_sha,master_seed=args;path=source_path(Path(out),task)
    identity=dict(plan_sha256=plan_sha,task=list(task))
    if path.exists():return check_chunk(path,identity)
    began=time.monotonic();d,left,right=task
    a=old.source_arrays(d,range(left,right),master_seed,64,1025)
    old.save_npz(path,**a,indices=np.arange(left,right))
    r=dict(identity=identity,sha256=old.sha(path),array_sha256=old.checksums(a),seconds=time.monotonic()-began)
    old.write_json(path.with_suffix('.json'),r,immutable=True)
    return r


def prepare_chunks(p,out,plan_sha,pilot=False):
    work=tasks(p,pilot)
    with ProcessPoolExecutor(max_workers=p['source_workers'],mp_context=mp.get_context('spawn')) as pool:
        reports=[]
        for count,record in enumerate(pool.map(draw_source,[(str(out),task,plan_sha,p['master_seed']) for task in work]),1):
            reports.append(record)
            if pilot or count%12==0:announce('sources_progress',complete_chunks=count,total_chunks=len(work),pilot=pilot)
    announce('sources_ready',chunks=len(work),histories=sum(t[2]-t[1] for t in work),pilot=pilot)
    return reports


def evaluate_chunks(p,out,plan_sha,ordered,shuffled,pilot=False):
    import torch
    need(torch.cuda.is_available(),'GPU allocation required')
    torch.set_num_threads(p['threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reports=[]
    for seed in p['seeds']:
        for role,root in (('ordered',ordered),('shuffled',shuffled)):
            checkpoint=root/f'seed{seed}'/'model_100000.pt'
            need(old.sha(checkpoint)==p['checkpoint_sha256'][f'{role}_seed{seed}'],'Checkpoint differs')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True);cfg=read(checkpoint.parent/'config.json')
            old.validate_checkpoint(saved,cfg,seed,role,False,checkpoint,TRAINING)
            model=old.GPT(**old.model_config(cfg)).to(device='cuda',dtype=torch.float64).eval();model.load_state_dict(saved['model']);del saved
            protocol=dict(plan_sha256=plan_sha,checkpoint_sha256=old.sha(checkpoint),config_sha256=old.sha(checkpoint.parent/'config.json'),
                role=role,seed=seed,dtype='float64',device='cuda',torch=str(torch.__version__),numpy=np.__version__,
                model_code_sha256=p['dependencies_sha256']['code/transformer/model.py'],tf32=False)
            old.write_json(out/'eval'/f'{role}_seed{seed}'/'protocol.json',protocol,immutable=True)
            for count,task in enumerate(tasks(p,pilot)):
                path=score_path(out,task,role,seed);source=source_path(out,task)
                r=check_chunk(source,dict(plan_sha256=plan_sha,task=list(task)))
                identity=dict(protocol=protocol,source_sha256=r['sha256'],task=list(task))
                if path.exists():reports.append(check_chunk(path,identity));continue
                a=load(source);need(np.array_equal(a['indices'],np.arange(task[1],task[2])) and old.checksums(a)==r['array_sha256'],'Chunk source changed')
                began=time.monotonic();values={k:[] for k in METRICS};causal=0.
                for left in range(0,len(a['tokens']),p['batch']):
                    right=min(left+p['batch'],len(a['tokens']));x=torch.from_numpy(a['tokens'][left:right,:-1]).cuda()
                    with torch.inference_mode():
                        logits=model(x)
                        if left==0:
                            causal=float((model(x[:,:32])[:,-1]-logits[:,31]).abs().max())
                        logp=logits.log_softmax(-1).cpu().numpy()
                    scored=old.expected_losses(logp,a['tokens'][left:right],a['transition'][left:right])
                    for k in METRICS:values[k].append(scored[k])
                merged={k:np.concatenate(v) for k,v in values.items()}
                need(causal<=1e-9 and all(np.isfinite(v).all() and v.shape==(task[2]-task[1],1024) for v in merged.values())
                     and merged['regret_bits'].min()>=-1e-9,'Causality or score validation failed')
                old.save_npz(path,**merged,indices=np.arange(task[1],task[2]),times=np.arange(1,1025))
                record=dict(identity=identity,sha256=old.sha(path),seconds=time.monotonic()-began,causal_error=causal)
                old.write_json(path.with_suffix('.json'),record,immutable=True);reports.append(record)
                if pilot or count%12==0:announce('scores_progress',role=role,seed=seed,task=list(task),seconds=record['seconds'],pilot=pilot)
            del model;torch.cuda.empty_cache()
    return reports


def projection(p,sources,scores):
    multiplier=(p['n_sequences']-p['retained_sequences'])/p['chunk']
    generation=sum(r['seconds'] for r in sources)*multiplier/p['source_workers']
    inference=sum(r['seconds'] for r in scores)*multiplier
    total=1.5*(generation+inference)+600
    return dict(source_seconds=sum(r['seconds'] for r in sources),paired_inference_seconds=sum(r['seconds'] for r in scores),
        projected_generation_seconds=generation,projected_inference_seconds=inference,projected_total_seconds_with_margin=total,
        rate_chf_per_gpu_hour=p['gpu_hour_rate_chf'],projected_cost_chf=total/3600*p['gpu_hour_rate_chf'],
        maximum_gpu_hours=p['scheduler']['wall_minutes']/60,maximum_cost_chf=p['scheduler']['wall_minutes']/60*p['gpu_hour_rate_chf'],
        rate_based_projection_not_invoice=True,passed=bool(total<=p['maximum_projection_seconds']))


def summarize(p,out,parent,parent_manifest,parent_summary,plan_sha):
    manifest=dict(schema='shared-base-shuffle-large-cohort-v1',plan_sha256=plan_sha,
        parent_manifest_sha256=p['parent_manifest_sha256'],retained_sequences=1024,n_sequences=p['n_sequences'],
        master_seed=p['master_seed'],source_code_sha256=p['dependencies_sha256']['code/transformer/source.py'],conditions=[])
    for d in p['depths']:
        key=f'a1_L{d}';oldsource=load(parent/'cohorts'/(key+'.npz'));source_parts=[oldsource];chunk_hashes={}
        for task in (t for t in tasks(p) if t[0]==d):
            path=source_path(out,task);r=check_chunk(path,dict(plan_sha256=plan_sha,task=list(task)))
            source_parts.append(load(path));chunk_hashes[str(path.relative_to(out))]=r['sha256']
        merged={name:np.concatenate([a[name] for a in source_parts]) for name in ('tokens','base','transition')}
        indices=np.concatenate([a['indices'] for a in source_parts])
        need(np.array_equal(indices,np.arange(p['n_sequences'])),'Source coverage incomplete/reordered')
        for k in merged:need(np.array_equal(merged[k][:1024],oldsource[k]),'Retained source arrays changed')
        old.save_npz(out/'cohorts'/(key+'.npz'),**merged,indices=indices)
        manifest['conditions'].append(dict(id=key,depth=d,alpha=1.,indices=indices.tolist(),
            arrays_sha256=old.checksums(merged),retained_arrays_sha256=old.checksums(oldsource),
            file_sha256=old.sha(out/'cohorts'/(key+'.npz')),source_chunks=chunk_hashes))
        del source_parts,merged
    old.write_json(out/'manifest.json',manifest,immutable=True)
    summary=dict(schema='shared-base-shuffle-large-extension-summary-v1',complete=False,scientific_reference_validated=False,
        manifest_sha256=old.sha(out/'manifest.json'),plan_sha256=plan_sha,groups=[],input_sha256={},
        no_retraining=True,no_new_bayesian_references=True,retained_scores_exact=True,
        uncertainty='Pointwise95% paired whole-source bootstrap; no training-seed uncertainty; no smoothing.')
    for seed in p['seeds']:
        for condition in manifest['conditions']:
            d,key=condition['depth'],condition['id'];roles={}
            for role in ('ordered','shuffled'):
                name=f'{role}_seed{seed}';previous=load(parent/'eval'/name/(key+'.npz'));parts=[]
                protocol=read(out/'eval'/name/'protocol.json')
                for task in (t for t in tasks(p) if t[0]==d):
                    path=score_path(out,task,role,seed)
                    check_chunk(path,dict(protocol=protocol,source_sha256=old.sha(source_path(out,task)),task=list(task)))
                    a=load(path);need(np.array_equal(a['indices'],np.arange(task[1],task[2])) and np.array_equal(a['times'],np.arange(1,1025)),'Scored chunk identity differs')
                    parts.append(a)
                merged=score_pair(previous,{k:np.concatenate([a[k] for a in parts]) for k in METRICS},1024)
                need(all(v.shape==(p['n_sequences'],1024) for v in merged.values()),'Full score coverage incomplete')
                path=out/'eval'/name/(key+'.npz')
                old.save_npz(path,**merged,indices=np.arange(p['n_sequences']),times=np.arange(1,1025),
                    protocol_sha256=old.sha(path.parent/'protocol.json'),source_tokens_sha256=condition['arrays_sha256']['tokens'])
                summary['input_sha256'][name+'/'+key]=old.sha(path);roles[role]=merged['expected_loss_bits']
                del parts,merged
            gap=roles['shuffled']-roles['ordered'];del roles
            for label,start,stop in ((f"all{p['n_sequences']}",0,p['n_sequences']),('retained1024',0,1024),('new15360',1024,p['n_sequences'])):
                values=gap[start:stop];rng_label='all1024' if label=='retained1024' else label
                rng=np.random.default_rng(old.seed_of('shuffle-extension-bootstrap-v1',seed,d,rng_label))
                counts=rng.multinomial(len(values),np.full(len(values),1/len(values)),size=p['bootstrap']).astype(float)
                group=dict(seed=seed,condition=key,depth=d,cohort=label,n_sequences=len(values),
                    instantaneous=old.paired_summary(values,counts),cumulative=old.paired_summary(values.cumsum(1),counts))
                if label=='retained1024':
                    previous=next(g for g in parent_summary['groups'] if (g['seed'],g['depth'],g['cohort'])==(seed,d,'all1024'))
                    for metric in ('instantaneous','cumulative'):
                        for stat in ('mean','se','lower95','upper95'):
                            need(np.max(abs(np.asarray(group[metric][stat])-previous[metric][stat]))<2e-12,'Retained1024 aggregate differs')
                summary['groups'].append(group)
            announce('condition_complete',depth=d,seed=seed,sequences=p['n_sequences'])
    summary.update(complete=True,scientific_reference_validated=True,independent_replay_passed=False)
    old.write_json(out/'summary.json',summary,immutable=True)
    return summary


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--plan',type=Path,default=PLAN)
    q.add_argument('--out',type=Path,required=True);q.add_argument('--parent',type=Path,required=True)
    q.add_argument('--ordered-root',type=Path,required=True);q.add_argument('--shuffled-root',type=Path,required=True)
    q.add_argument('--pilot-only',action='store_true');a=q.parse_args();p=validate_plan(a.plan)
    announce('starting',job=os.environ.get('SLURM_JOB_ID'),n_sequences=p['n_sequences'],retained_sequences=1024,
             depths=p['depths'],seeds=p['seeds'],scope='Paired frozen-checkpoint evaluation; no training or Bayesian computation')
    need(a.out.resolve()!=a.parent.resolve() and a.parent.resolve() not in a.out.resolve().parents,'Fresh separate output required')
    a.out.mkdir(parents=True,exist_ok=True);lock=(a.out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();plan_sha=old.sha(a.plan)
    identity=dict(plan_sha256=plan_sha,driver_sha256=old.sha(__file__),parent=str(a.parent.resolve()))
    old.write_json(a.out/'identity.json',identity,immutable=True)
    announce('checking_parent');pm,ps=validate_parent(p,a.parent);announce('parent_verified')
    sources=prepare_chunks(p,a.out,plan_sha,True)
    scores=evaluate_chunks(p,a.out,plan_sha,a.ordered_root,a.shuffled_root,True)
    report=projection(p,sources,scores);old.write_json(a.out/'projection.json',report,immutable=True);announce('pilot_projection',**report)
    if a.pilot_only:return
    need(report['passed'],'Measured projection exceeds the fixed allocation: retain pilot and stop')
    prepare_chunks(p,a.out,plan_sha);evaluate_chunks(p,a.out,plan_sha,a.ordered_root,a.shuffled_root)
    summarize(p,a.out,a.parent,pm,ps,plan_sha)
    old.write_json(a.out/'status.json',dict(complete=True,elapsed_seconds=time.monotonic()-started,
        job=os.environ.get('SLURM_JOB_ID'),summary_sha256=old.sha(a.out/'summary.json'),retained_sequences=1024,n_sequences=p['n_sequences']))
    announce('complete',seconds=time.monotonic()-started)


if __name__=='__main__':main()
