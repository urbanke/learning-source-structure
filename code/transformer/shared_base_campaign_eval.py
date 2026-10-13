"""Export and score the existing shared-base diagnostic cohorts at saved milestones.

No reference inference or source regeneration is performed. Export reads each
pinned source record; scoring uses its true transition row after t observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from model import GPT
from train_mixture import atomic_save, write_json, digest_json
from shared_base_campaign_train import read_plan, require, file_hash, ROOT, PLAN


def export(plan_path, source_root, out):
    plan,_=read_plan(plan_path); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'code/scripts'))
    from shared_base_conditional_cohort import load_source
    records={}; files={}; times=np.asarray(plan['evaluation']['times'])
    for depth in plan['evaluation']['conditions']:
        folder=Path(source_root)/f'a1_L{depth}'
        for name in ('cohort.json','protocol.json'):
            key=f'output/shared_base_conditional_curves_2026-09-15/a1_L{depth}/{name}'
            require(file_hash(folder/name)==plan['evaluation']['frozen_cohort_sha256'][key], 'frozen cohort changed')
        tokens=[]; rows=[]; ids=[]; hashes=[]
        for index in range(256):
            source=load_source(folder,index); x=np.asarray(source['tokens'],dtype=np.int64)
            transition=np.asarray(source['oracle']['transition_probs'],dtype=np.float64)
            require(x.shape==(1025,) and transition.shape==(64,64),'source shape changed')
            tokens.append(x); rows.append(transition[x[times-1]])
            ids.append(source['source_id']); hashes.append(file_hash(folder/'data'/f'seq_{index:06d}.json'))
        name=f'a1_L{depth}.npz'; path=out/name
        atomic_save(path,lambda f:np.savez_compressed(f,tokens=np.asarray(tokens),oracle_probs=np.asarray(rows),
                    source_ids=np.asarray(ids),source_indices=np.arange(256),times=times))
        files[name]=file_hash(path); records[str(depth)]=dict(source_ids=ids,record_sha256=hashes,
                cohort_sha256=file_hash(folder/'cohort.json'),protocol_sha256=file_hash(folder/'protocol.json'))
    manifest=dict(schema='shared-base-campaign-inputs-v1',files=files,source_records=records,
        times=times.tolist(),n_sequences=256,n_tokens=1025,source='Previously frozen alpha1, depths0/1/4/8; same sequences as earlier comparisons.')
    write_json(out/'manifest.json',manifest)
    print(json.dumps(dict(out=str(out),manifest_sha256=file_hash(out/'manifest.json'),n_sources=1024)))
    return manifest


def inputs(plan, folder):
    folder=Path(folder); manifest=json.loads((folder/'manifest.json').read_text())
    require(file_hash(folder/'manifest.json')==plan['evaluation']['inputs_manifest_sha256'],'input manifest changed')
    result={}
    for depth in plan['evaluation']['conditions']:
        name=f'a1_L{depth}.npz'; path=folder/name
        require(file_hash(path)==manifest['files'][name],'frozen compact inputs changed')
        with np.load(path,allow_pickle=False) as f: a={k:f[k] for k in f.files}
        require(a['tokens'].shape==(256,1025) and np.array_equal(a['times'],plan['evaluation']['times']),'input dimensions changed')
        require(a['oracle_probs'].shape==(256,6,64) and np.isfinite(a['oracle_probs']).all()
                and (a['oracle_probs']>=0).all() and np.max(abs(a['oracle_probs'].sum(-1)-1))<1e-12,'invalid oracle probabilities')
        require(np.array_equal(a['source_indices'],np.arange(256)) and a['source_ids'].tolist()==manifest['source_records'][str(depth)]['source_ids'],'source alignment changed')
        require((a['tokens']>=0).all() and (a['tokens']<64).all(),'token outside alphabet')
        result[depth]=a
    return result


def score(plan_path, run, cohort_root, out, *, device='cuda', batch=8):
    plan,plan_hash=read_plan(plan_path); run=Path(run); out=Path(out); frozen=inputs(plan,cohort_root)
    require(device in ('cpu','cuda') and batch>0,'invalid evaluation settings')
    torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    receipt=json.loads((run/'config.json').read_text()); cfg=receipt['config']
    require(receipt['plan_sha256']==plan_hash and cfg['arm'] in plan['arms'] and cfg['seed'] in [0,1],'run is not this campaign')
    require(receipt['config_hash']==digest_json(cfg) and receipt['implementation_hashes']==plan['implementation_sha256'],'run contract changed')
    arm=plan['arms'][cfg['arm']]
    expected_cfg=dict(schema='shared-base-campaign-training-v1',**arm['config'],**plan['source'],seed=cfg['seed'],
                      arm=cfg['arm'],parent=cfg['parent'],continuation_lr=arm.get('continuation_lr'))
    require(cfg==expected_cfg,'configuration or source prior changed')
    if cfg['arm']=='continue_large':
        require(cfg['parent']['step']==100000 and cfg['parent']['weights_sha256']==
                plan['parent']['seeds'][str(cfg['seed'])]['model_100000.pt'],'parent identity differs')
    else:
        require(cfg['parent'] is None,'fresh arm has a parent')
    expected_model=dict(vocab=64,n_ctx=1024,n_layer=cfg['n_layer'],n_head=cfg['n_head'],d_model=cfg['d_model'],mlp=True,dropout=0.)
    require(receipt['model']==expected_model and receipt['n_params']==arm['n_params'] and receipt['precision']=='float32',
            'saved model architecture or precision differs')
    out.mkdir(parents=True,exist_ok=True); completed=[]; missing=[]
    for step in cfg['weight_steps']:
        path=run/f'model_{step}.pt'
        if not path.is_file(): missing.append(step); continue
        target=out/f'step{step}'; target.mkdir(exist_ok=True)
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        for key,value in [('schema','shared-base-campaign-weights-v1'),('step',step),('cfg',receipt),('config_hash',receipt['config_hash']),
                          ('plan_sha256',plan_hash),('implementation_hashes',plan['implementation_sha256'])]:
            require(ckpt.get(key)==value,'checkpoint identity differs: '+key)
        model=GPT(**receipt['model']).to(device).eval(); model.load_state_dict(ckpt['model'])
        summaries=[]
        for depth,a in frozen.items():
            values=[]; max_error=0.
            with torch.no_grad():
                for start in range(0,256,batch):
                    x=torch.from_numpy(a['tokens'][start:start+batch,:-1]).to(device)
                    logits=model(x); selected=logits[:,a['times']-1,:]
                    logp=torch.log_softmax(selected.double(),dim=-1).cpu().numpy()
                    require(np.isfinite(logp).all(),'nonfinite predictions')
                    max_error=max(max_error,float(abs(np.exp(logp).sum(-1)-1).max()))
                    if start==0:
                        for t in (1,16,256):
                            prefix=model(x[:,:t])[:,-1,:]
                            require(torch.allclose(prefix,logits[:,t-1,:],atol=2e-5,rtol=0),'causal prefix alignment failed')
                    values.append(logp)
            logp=np.concatenate(values); oracle=a['oracle_probs']; positive=oracle>0
            terms=np.zeros_like(oracle); terms[positive]=oracle[positive]*(np.log(oracle[positive])-logp[positive])/np.log(2)
            regret=terms.sum(-1); expected=-(oracle*logp).sum(-1)/np.log(2)
            require(max_error<1e-12 and np.min(regret)>-1e-10,'invalid normalized regret')
            dest=target/f'a1_L{depth}.npz'
            atomic_save(dest,lambda f:np.savez_compressed(f,source_ids=a['source_ids'],source_indices=a['source_indices'],
                times=a['times'],log_predictive_probs=logp,expected_loss_bits=expected,regret_bits=regret))
            summaries.append(dict(depth=depth,mean_regret_bits=regret.mean(0).tolist(),se_regret_bits=(regret.std(0,ddof=1)/16).tolist(),
                                  max_normalization_error=max_error,file=dest.name,sha256=file_hash(dest)))
        write_json(target/'summary.json',dict(schema='shared-base-campaign-evaluation-v1',plan_sha256=plan_hash,arm=cfg['arm'],seed=cfg['seed'],
            step=step,weights_sha256=file_hash(path),inputs_manifest_sha256=plan['evaluation']['inputs_manifest_sha256'],
            evaluation_code_sha256=file_hash(__file__),
            times=plan['evaluation']['times'],n_sequences=256,role=plan['evaluation']['role'],groups=summaries,
            bayesian_reference_computed=False,baseline='Compare by source_id with the original100k model scores; do not select a seed.'))
        completed.append(step)
    result=dict(arm=cfg['arm'],seed=cfg['seed'],evaluated_steps=completed,missing_steps=missing,complete=not missing)
    write_json(out/'status.json',result); print(json.dumps(result)); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',type=Path,default=PLAN)
    sub=p.add_subparsers(dest='mode',required=True)
    e=sub.add_parser('export'); e.add_argument('--source-root',type=Path,required=True); e.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('score'); s.add_argument('--run',type=Path,required=True); s.add_argument('--cohorts',type=Path,required=True)
    s.add_argument('--out',type=Path,required=True); s.add_argument('--device',choices=['cpu','cuda'],default='cuda'); s.add_argument('--batch',type=int,default=8)
    a=p.parse_args()
    if a.mode=='export': export(a.plan,a.source_root,a.out)
    else: score(a.plan,a.run,a.cohorts,a.out,device=a.device,batch=a.batch)

if __name__=='__main__': main()
