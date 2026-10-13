"""Frozen paired evaluation for transformers trained on a hidden product source.

Source entropy strata are fixed from the true matrices before any prediction
is scored. All models and exact references receive the same observed tokens.
Raw cohort and per-sequence results belong in ignored output/, not git.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from hidden_factor import draw_batch, known_encoding_curve, flat_curve, true_curve
from hidden_factor_gpu import exact_selected, dimensions_arg, TIMES

SCHEMA = 'hidden-factor-evaluation-v1'
NAMESPACE = 'hidden-factor-heldout-20260917-v1'
HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_npz(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def entropy_strata(matrices):
    """Split each coordinate's normalized mean row entropy at its source median."""
    entropies = []
    for matrix in matrices:
        dimension = matrix.shape[-1]
        if (matrix.ndim != 3 or matrix.shape[1] != dimension or np.any(matrix <= 0)
                or not np.allclose(matrix.sum(-1),1,rtol=0,atol=1e-12)):
            raise ValueError('invalid positive normalized transition matrices')
        entropies.append(-(matrix*np.log2(matrix)).sum(-1).mean(-1)/math.log2(dimension))
    entropy = np.column_stack(entropies)
    medians = np.median(entropy, axis=0)
    # Equality is assigned to the high group. No outcome-dependent balancing.
    strata = (entropy[:,0] >= medians[0]).astype(np.int8)*2 + (entropy[:,1] >= medians[1]).astype(np.int8)
    return entropy, medians, strata


def source_spec(dimensions, n_seq, length, times, seed):
    if n_seq < 2 or length < 2:
        raise ValueError('at least two sequences and two tokens required')
    if list(times) != sorted(set(times)) or min(times)<1 or max(times)!=length-1:
        raise ValueError('times must be increasing, positive, and include the final prediction')
    return dict(schema=SCHEMA, dimensions=list(dimensions), n_seq=n_seq, length=length,
        times=list(times), seed=seed, seed_namespace=NAMESPACE,
        seed_rule='first128bits SHA256(canonicalJSON([namespace,seed,dimensions,n_seq,length])) as four little-endian uint32 words',
        source_sha256=file_hash(HERE/'hidden_factor.py'), numpy_version=np.__version__,
        source='Independent Dirichlet(1,...,1) rows in each coordinate; fresh uniform joint permutation; uniform initial symbol.',
        stratum_rule='Median split of each coordinate mean row entropy divided by log2(coordinate dimension); equal-to-median is high. Membership is fixed from source matrices before scoring models.',
        first_token_scored=False, cumulative_rule='At time t sum realized losses predicting tokens2 through t+1.',
        production_size=(n_seq==4096 and length==1025 and list(times)==list(TIMES)))


def create_cohort(out, dimensions, n_seq=4096, length=1025, times=TIMES, seed=2026091702):
    out = Path(out); spec=source_spec(dimensions,n_seq,length,times,seed)
    ident=fingerprint(spec); plan=out/'cohort.json'; path=out/'cohort.npz'
    if plan.exists() and path.exists():
        record=json.loads(plan.read_text())
        if record['identity']!=spec or record['cohort_id']!=ident or record['data_sha256']!=file_hash(path):
            raise ValueError('existing held-out cohort differs or is corrupt; use another directory')
        return record
    if plan.exists() or path.exists():
        raise ValueError('incomplete cohort write; inspect before removing')
    digest=hashlib.sha256(canonical([NAMESPACE,seed,list(dimensions),n_seq,length]).encode()).digest()
    random_seed=np.frombuffer(digest[:16],dtype='<u4')
    draw=draw_batch(np.random.default_rng(random_seed),n_seq,length,dimensions=dimensions)
    entropy,medians,strata=entropy_strata(draw['matrices'])
    arrays=dict(tokens=draw['xs'],first_matrix=draw['matrices'][0],second_matrix=draw['matrices'][1],
        encoding=draw['encoding'], normalized_row_entropy=entropy,stratum=strata,times=np.array(times,dtype=np.int64))
    save_npz(path,arrays)
    record=dict(schema=SCHEMA,cohort_id=ident,identity=spec,data_sha256=file_hash(path),
        entropy_medians=medians.tolist(),stratum_counts=np.bincount(strata,minlength=4).tolist(),
        stratum_labels=['low / low','low / high','high / low','high / high'])
    save_json(plan,record)
    return record


def load_cohort(out):
    out=Path(out);record=json.loads((out/'cohort.json').read_text())
    if (record['schema']!=SCHEMA or fingerprint(record['identity'])!=record['cohort_id']
            or file_hash(out/'cohort.npz')!=record['data_sha256']):
        raise ValueError('held-out cohort checksum/identity mismatch')
    with np.load(out/'cohort.npz',allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    if arrays['tokens'].shape!=(record['identity']['n_seq'],record['identity']['length']):
        raise ValueError('held-out cohort shape mismatch')
    entropy,medians,strata=entropy_strata((arrays['first_matrix'],arrays['second_matrix']))
    if (not np.array_equal(entropy,arrays['normalized_row_entropy'])
            or not np.array_equal(strata,arrays['stratum']) or medians.tolist()!=record['entropy_medians']):
        raise ValueError('source-only entropy strata changed')
    return record,arrays


def checked_existing(path, identity, recover_orphan=False):
    path=Path(path);meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if record['identity']!=identity or record['data_sha256']!=file_hash(path):
            raise ValueError(f'existing result input/hash mismatch: {path}')
        with np.load(path,allow_pickle=False) as data:
            return {key:data[key] for key in data.files}
    if path.exists() or meta.exists():
        if recover_orphan:
            # A wall-time kill can happen between the two atomic replacements.
            # Preserve the orphan for inspection and recompute this sequence.
            for orphan in (path,meta):
                if orphan.exists():
                    orphan.replace(orphan.with_name(orphan.name+f'.orphan.{os.getpid()}'))
            return None
        raise ValueError(f'incomplete result write: {path}')
    return None


def write_result(path, identity, arrays):
    save_npz(path,arrays)
    save_json(Path(path).with_suffix('.json'),dict(schema=SCHEMA,identity=identity,data_sha256=file_hash(path)))


def metrics(probability, oracle, tokens, times, cumulative):
    probability,oracle=np.asarray(probability),np.asarray(oracle)
    if (probability.shape!=oracle.shape or probability.shape[0]!=len(times)
            or not np.isfinite(probability).all() or np.any(probability<=0)
            or not np.allclose(probability.sum(-1),1,rtol=0,atol=4e-6)):
        raise ValueError('invalid predicted distribution')
    kl=np.sum(oracle*(np.log2(oracle)-np.log2(probability)),axis=-1)
    if kl.min() < -2e-6:
        raise ArithmeticError('negative true-distribution KL beyond rounding tolerance')
    return dict(next_kl_bits=np.maximum(0,kl),
        next_loss_bits=-np.log2(probability[np.arange(len(times)),tokens[np.asarray(times)]]),
        cumulative_loss_bits=np.asarray(cumulative))


def reference_one(data,index,dimensions,times,device,chunk_size,encoding_cache=None):
    xs=data['tokens'][index];encoding=data['encoding'][index]
    matrices=(data['first_matrix'][index],data['second_matrix'][index])
    truth=true_curve(xs,matrices,encoding,times,dimensions)
    unknown=exact_selected(xs,dimensions,times,device=device,chunk_size=chunk_size,encoding_cache=encoding_cache)
    known=known_encoding_curve(xs,encoding,times,dimensions)
    flat=flat_curve(xs,times,dimensions)
    curves={'bayes_hidden':unknown,'bayes_encoding_given':known,'flat_dirichlet':flat,'oracle':truth}
    arrays=dict(times=np.asarray(times),oracle_probs=truth['predictive'])
    initial=-math.log2(math.prod(dimensions))
    for name,curve in curves.items():
        cumulative=(curve['cumulative_loss_bits'] if name=='bayes_hidden'
            else -curve['log_evidence_bits'][np.asarray(times)+1]+initial)
        values=metrics(curve['predictive'],truth['predictive'],xs,times,cumulative)
        for key,value in values.items():arrays[f'{name}_{key}']=value
        arrays[f'{name}_predictive']=curve['predictive']
    cost=arrays['bayes_hidden_cumulative_loss_bits']-arrays['bayes_encoding_given_cumulative_loss_bits']
    if np.any(cost > math.log2(unknown['encoding_count'])+1e-8):
        raise ArithmeticError('exact encoding-mixture cumulative bound failed')
    arrays['encoding_uncertainty_cost_bits']=cost
    arrays['hidden_evidence_times']=unknown['evidence_times']
    arrays['hidden_log_evidence_bits']=unknown['log_evidence_bits']
    return arrays


def evaluation_count(planned, max_sequences=None):
    """A cap selects a prefix of the frozen cohort; it never changes its identity."""
    if max_sequences is None:
        return planned
    if max_sequences < 2 or max_sequences > planned:
        raise ValueError('max_sequences must be between two and the planned cohort size')
    return max_sequences


def write_progress_manifest(path,identity,evaluated,planned,**extra):
    """Record checked contiguous results without downgrading a longer prior run."""
    path=Path(path)
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['identity']!=identity:
            raise ValueError('existing evaluation manifest identity mismatch')
        evaluated=max(evaluated,previous.get('evaluated_n_sequences',previous['n_seq']))
    save_json(path,dict(schema=SCHEMA,identity=identity,n_seq=evaluated,
        evaluated_n_sequences=evaluated,planned_n_sequences=planned,
        partial_evaluation=(evaluated<planned),complete=(evaluated==planned),**extra))


def references(out,device='cpu',chunk_size=65536,cache_root=None,max_sequences=None):
    import torch
    out=Path(out);record,data=load_cohort(out);spec=record['identity']
    n=evaluation_count(spec['n_seq'],max_sequences)
    from hidden_factor_cache import prepare_encoding_cache,open_encoding_cache
    cache_root=Path(cache_root) if cache_root is not None else out/'encoding_cache'
    cache_path=cache_root/f"{spec['dimensions'][0]}x{spec['dimensions'][1]}.npy"
    cache_receipt=prepare_encoding_cache(cache_path,tuple(spec['dimensions']),chunk_size=chunk_size)
    cached=open_encoding_cache(cache_path,tuple(spec['dimensions']))
    identity=dict(schema=SCHEMA,cohort_id=record['cohort_id'],kind='reference',
        encoding_cache_sha256=cache_receipt['sha256_payload'],
        device=device,chunk_size=chunk_size,torch_version=str(torch.__version__),
        evaluator_sha256=file_hash(__file__),source_sha256=file_hash(HERE/'hidden_factor.py'),
        reference_sha256=file_hash(HERE/'hidden_factor_gpu.py'))
    if identity['source_sha256']!=spec['source_sha256']:
        raise ValueError('source implementation changed since cohort generation')
    for i in range(n):
        path=out/'references'/f'seq_{i:06d}.npz'
        current=dict(identity,sequence_index=i)
        if checked_existing(path,current,recover_orphan=True) is None:
            values=reference_one(data,i,tuple(spec['dimensions']),spec['times'],device,chunk_size,cached)
            write_result(path,current,values)
        if (i+1)%16==0 or i+1==n:
            write_progress_manifest(out/'references'/'manifest.json',identity,i+1,spec['n_seq'])
            print(json.dumps(dict(stage='reference',completed=i+1,total=spec['n_seq'],
                requested=n,partial_evaluation=(n<spec['n_seq']))),flush=True)


def validate_checkpoint(ckpt,spec):
    from train_hidden_factor import source_contract
    if ckpt.get('schema') not in ('hidden-factor-weights-v1','hidden-factor-checkpoint-v1'):
        raise ValueError('expected a hidden-factor checkpoint')
    receipt=ckpt['cfg'];config=receipt['config'];model_config=receipt['model']
    if (fingerprint(config)!=ckpt.get('config_hash') or receipt.get('config_hash')!=ckpt.get('config_hash')):
        raise ValueError('checkpoint configuration hash mismatch')
    contract=source_contract(config['source_dimensions'])
    if receipt.get('source_contract')!=contract or receipt.get('source_contract_hash')!=fingerprint(contract):
        raise ValueError('checkpoint source contract mismatch')
    hashes=ckpt.get('implementation_hashes')
    if hashes!=receipt.get('implementation_hashes') or not isinstance(hashes,dict):
        raise ValueError('checkpoint implementation receipt mismatch')
    for name in ['hidden_factor.py','model.py']:
        if hashes.get(name)!=file_hash(HERE/name):
            raise ValueError(f'checkpoint {name} differs from the current evaluated source/model')
    if (list(config['source_dimensions'])!=spec['dimensions'] or ckpt['step']!=config['steps']
            or ckpt.get('completed_sequences')!=config['batch']*ckpt['step']
            or model_config['vocab']!=math.prod(spec['dimensions']) or model_config['n_ctx']<spec['length']-1):
        raise ValueError('checkpoint source, final step, sample count, or model support mismatch')
    if spec['production_size'] and (ckpt['completed_sequences']!=6400000 or config['batch']!=64
            or config['steps']!=100000 or config['n_ctx']!=1024
            or (model_config['n_layer'],model_config['n_head'],model_config['d_model'])!=(2,4,128)):
        raise ValueError('production evaluation requires the declared 6.4-million-sequence training budget and model')
    if config['seed'] not in (0,1):
        raise ValueError('held-out experiment expects training seed0 or seed1')
    return config,model_config


def evaluate_model(out, checkpoint, device='cpu', batch_size=16,max_sequences=None):
    import torch
    from model import GPT
    from eval_mixture import check_causality
    out=Path(out);record,data=load_cohort(out);spec=record['identity']
    n=evaluation_count(spec['n_seq'],max_sequences)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config,model_config=validate_checkpoint(ckpt,spec)
    name=f"seed{config['seed']}"
    model=GPT(**model_config);model.load_state_dict(ckpt['model'],strict=True);model.to(device).eval()
    identity=dict(schema=SCHEMA,cohort_id=record['cohort_id'],kind='transformer',name=name,
        checkpoint_sha256=file_hash(checkpoint),checkpoint_step=ckpt['step'],device=device,
        torch_version=str(torch.__version__),model_sha256=file_hash(HERE/'model.py'),
        evaluator_sha256=file_hash(__file__),batch_size=batch_size)
    times=np.asarray(spec['times']);causality=None
    for start in range(0,n,batch_size):
        stop=min(start+batch_size,n);indices=list(range(start,stop))
        missing=[i for i in indices if checked_existing(out/'models'/name/f'seq_{i:06d}.npz',dict(identity,sequence_index=i),recover_orphan=True) is None]
        if not missing:continue
        tokens=data['tokens'][missing];tensor=torch.as_tensor(tokens[:,:-1],dtype=torch.long,device=device)
        if causality is None:
            causality=check_causality(model,torch.as_tensor(tokens,dtype=torch.long,device=device),times)
        with torch.inference_mode():
            logp=torch.log_softmax(model(tensor).double(),dim=-1).cpu().numpy()/math.log(2)
        realized=-np.take_along_axis(logp,tokens[:,1:,None],axis=-1).squeeze(-1)
        cumulative=np.cumsum(realized,axis=-1)[:,times-1]
        for j,i in enumerate(missing):
            truth=true_curve(data['tokens'][i],(data['first_matrix'][i],data['second_matrix'][i]),
                             data['encoding'][i],times,tuple(spec['dimensions']))['predictive']
            probabilities=np.exp2(logp[j,times-1])
            values=metrics(probabilities,truth,data['tokens'][i],times,cumulative[j])
            values['predictive']=probabilities;values['times']=times
            write_result(out/'models'/name/f'seq_{i:06d}.npz',dict(identity,sequence_index=i),values)
    # A resumed complete evaluation still performs a causality check.
    if causality is None:
        causality=check_causality(model,torch.as_tensor(data['tokens'][:1],dtype=torch.long,device=device),times)
    write_progress_manifest(out/'models'/name/'manifest.json',identity,n,spec['n_seq'],
        causality_check=causality)


def moments(values):
    values=np.asarray(values)
    if values.ndim!=2 or not np.isfinite(values).all() or len(values)<2:
        raise ValueError('at least two finite independent sequence records required for uncertainty')
    return dict(n_seq=len(values),mean=values.mean(0).tolist(),se=(values.std(0,ddof=1)/math.sqrt(len(values))).tolist())


def read_complete(out, subdirectory, n_seq, cohort_id,planned_n_seq=None):
    """Read a checked full cohort, or an explicitly requested evaluated prefix.

    Omitting planned_n_seq retains the strict full-evaluation API. Partial
    consumers must supply the original cohort size as well as their prefix.
    """
    path=Path(out)/subdirectory;manifest=json.loads((path/'manifest.json').read_text())
    planned=n_seq if planned_n_seq is None else planned_n_seq
    partial=n_seq<planned
    available=manifest.get('evaluated_n_sequences',manifest['n_seq'])
    if (manifest['identity']['cohort_id']!=cohort_id or available<n_seq
            or manifest.get('planned_n_sequences',manifest['n_seq'])!=planned
            or (not partial and (not manifest['complete'] or available!=n_seq))):
        raise ValueError('incomplete or mismatched evaluation manifest')
    rows=[];hashes=[]
    for i in range(n_seq):
        file=path/f'seq_{i:06d}.npz'
        value=checked_existing(file,dict(manifest['identity'],sequence_index=i))
        if value is None:raise ValueError('missing required sequence result')
        rows.append(value);hashes.append(file_hash(file))
    return {k:np.stack([row[k] for row in rows]) for k in rows[0]},fingerprint(hashes)


def summarize(out,bayes_only=False,max_sequences=None):
    out=Path(out);record,data=load_cohort(out);spec=record['identity'];planned=spec['n_seq']
    n=evaluation_count(planned,max_sequences);partial=n<planned
    reference,reference_hash=read_complete(out,'references',n,record['cohort_id'],planned)
    predictors={}
    for name in ['bayes_hidden','bayes_encoding_given','flat_dirichlet','oracle']:
        predictors[name]={metric:reference[f'{name}_{metric}'] for metric in ('next_kl_bits','next_loss_bits','cumulative_loss_bits')}
    model_hashes={}
    if not bayes_only:
        manifests=sorted((out/'models').glob('*/manifest.json'))
        if not manifests:raise ValueError('no complete transformer evaluations; use --bayes-only explicitly')
        for manifest_path in manifests:
            name=manifest_path.parent.name
            arrays,digest=read_complete(out,Path('models')/name,n,record['cohort_id'],planned)
            predictors[name]={k:arrays[k] for k in ('next_kl_bits','next_loss_bits','cumulative_loss_bits')}
            model_hashes[name]=digest
    groups=[]
    definitions=[('full_prior',np.ones(n,dtype=bool),None)]
    definitions += [(f'entropy_{i}',data['stratum'][:n]==i,label) for i,label in enumerate(record['stratum_labels'])]
    for name,mask,label in definitions:
        if mask.sum()<2:
            groups.append(dict(id=name,entropy_label=label,n_seq=int(mask.sum()),status='insufficient_sample'))
            continue
        row=dict(id=name,entropy_label=label,n_seq=int(mask.sum()),status='complete',predictors={},paired={})
        for predictor,values in predictors.items():
            row['predictors'][predictor]={metric:moments(value[mask]) for metric,value in values.items()}
        pairs=[('bayes_hidden','bayes_encoding_given'),('flat_dirichlet','bayes_hidden')]
        pairs += [(name,'bayes_hidden') for name in model_hashes]
        pairs += [(name,'bayes_encoding_given') for name in model_hashes]
        for left,right in pairs:
            row['paired'][f'{left}_minus_{right}']={metric:moments((predictors[left][metric]-predictors[right][metric])[mask])
                for metric in ('next_kl_bits','next_loss_bits','cumulative_loss_bits')}
        groups.append(row)
    result=dict(schema=SCHEMA,cohort=record,times=spec['times'],groups=groups,
        reference_records_sha256=reference_hash,model_records_sha256=model_hashes,
        uncertainty='One standard error across independent complete sequences; differences use paired per-sequence values. Seeds remain separate.',
        caveat='Conditional costs can be negative; posterior ignorance costs are nonnegative only in expectation over the full prior. Entropy bins are source-only descriptive strata; no model-outcome selection.',
        bayes_only=bayes_only,production_size=spec['production_size'],
        partial_evaluation=partial,evaluated_n_sequences=n,planned_n_sequences=planned,
        evaluation_complete=not partial,
        expected_training_seeds=[0,1],present_training_seeds=[int(name[4:]) for name in model_hashes],
        missing_training_seeds=[seed for seed in [0,1] if f'seed{seed}' not in model_hashes],
        two_repeat_comparison_complete=(not partial and set(model_hashes)=={'seed0','seed1'}))
    name=f'summary_prefix_{n:06d}.json' if partial else 'summary.json'
    save_json(out/name,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','sources','references','models','summary'],default='all')
    parser.add_argument('--dimensions',type=dimensions_arg,required=True)
    parser.add_argument('--n-seq',type=int,default=4096)
    parser.add_argument('--max-sequences',type=int,
        help='Evaluate only the first N sequences of the unchanged full --n-seq cohort; resume without this cap to finish.')
    parser.add_argument('--length',type=int,default=1025)
    parser.add_argument('--times',type=int,nargs='+',default=list(TIMES))
    parser.add_argument('--seed',type=int,default=2026091702)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--chunk-size',type=int,default=65536)
    parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--checkpoint',type=Path,action='append',default=[])
    parser.add_argument('--bayes-only',action='store_true')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.batch_size<1:parser.error('positive model batch size required')
    try:evaluation_count(args.n_seq,args.max_sequences)
    except ValueError as error:parser.error(str(error))
    if args.stage in ('all','models') and not args.bayes_only and not args.checkpoint:
        parser.error('checkpoint required unless --bayes-only; no expensive computation was started')
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.evaluation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('another evaluation is writing this output directory') from error
        if args.stage in ('all','sources'):
            create_cohort(args.out,args.dimensions,args.n_seq,args.length,args.times,args.seed)
        else:
            record,_=load_cohort(args.out)
            if record['identity']!=source_spec(args.dimensions,args.n_seq,args.length,args.times,args.seed):
                parser.error('declared cohort arguments differ from the frozen source plan')
        if args.stage in ('all','references'):references(args.out,args.device,args.chunk_size,args.cache_root,args.max_sequences)
        if args.stage in ('all','models') and not args.bayes_only:
            if not args.checkpoint:parser.error('checkpoint required unless --bayes-only')
            for checkpoint in args.checkpoint:evaluate_model(args.out,checkpoint,args.device,args.batch_size,args.max_sequences)
        if args.stage in ('all','summary'):summarize(args.out,args.bayes_only,args.max_sequences)


if __name__=='__main__':main()
