"""Declared fixed-parameter benchmark using the unchanged full-family posterior.

Only source generation is conditioned on alpha/depth. Reference inference sees
observed tokens and retains alpha Uniform[.5,1.5], rank permutation, and depth
Uniform{0,...,8}; the oracle and frozen model enter numerical scoring afterwards.
This is a cost/validation benchmark, not a selected-source population estimate.
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'code/transformer'
sys.path.insert(0, str(HERE))
from eval_shared_base_models import (atomic_json, atomic_npz, fingerprint, sha_array,
    sha_file, models, validate_source_record, SOURCE_SCHEMA)
from eval_shared_base_reference import selected_grid_checks
from eval_shared_base_surrogate import CODE_FILES
from full_context_statistics import compare_prediction, diagnostic_gates
from stable_chain_statistics import stable_predictive_moments
from ranked_lsa import RankedLSARowEvidence
from shared_base_surrogate import SharedBaseSurrogateMCMC
from surrogate_chain_runner import SurrogateChainRun
from source import Source, seed_of
from validate_shared_base_mcmc import json_safe

TAG = 'conditional-shared-base-benchmark-v1'


def draw(alpha, depth, seed):
    source = Source(64, 1025, alpha=alpha, ls=depth, permute=True)
    rng = np.random.default_rng(seed)
    a, l = source.draw_params(1, rng)
    q, t = source.matrices(a, l, rng)
    original = source.sequences(q, t, rng)[0]
    permutation = rng.permutation(64)
    inverse = np.argsort(permutation)
    return permutation[original], q[0, inverse], t[0][inverse][:, inverse], permutation


def identity(alpha, depth, index, master_seed):
    return dict(schema=TAG, source_alpha=float(alpha), source_depth=int(depth),
                source_index=index, master_seed=master_seed,
                sequence_seed=seed_of(TAG, master_seed, f'{alpha:g}', depth, index),
                d=64, n_tokens=1025, source_sha256=sha_file(HERE / 'source.py'),
                construction_sha256=fingerprint(inspect.getsource(draw)),
                numpy_version=np.__version__)


def prepare(args):
    records=[]
    for index in range(args.n_seq):
        ident=identity(args.alpha,args.depth,index,args.seed)
        path=args.out/'data'/f'seq_{index:06d}.json'
        if path.exists():
            record=validate_source_record(json.loads(path.read_text()), ident)
        else:
            tokens,q,transition,permutation=draw(args.alpha,args.depth,ident['sequence_seed'])
            record=dict(schema=SOURCE_SCHEMA, identity=ident, source_id=fingerprint(ident),
                tokens=tokens.tolist(), tokens_sha256=sha_array(tokens,'<i2'),
                generated_source=dict(parameters=dict(alpha=args.alpha,ls=args.depth,perm=permutation.tolist())),
                oracle=dict(initial_probs=q.tolist(),transition_probs=transition.tolist(),
                    initial_probs_sha256=sha_array(q,'<f8'),transition_probs_sha256=sha_array(transition,'<f8')))
            validate_source_record(record,ident)
            atomic_json(path,record)
        records.append(dict(index=index,path=str(path.resolve()),source_id=record['source_id'],file_sha256=sha_file(path)))
    cohort=dict(schema=TAG,n_sequences=args.n_seq,master_seed=args.seed,alpha=args.alpha,depth=args.depth,
                prefixes=args.times,n_tokens_per_sequence=1025,records=records,
                selection='Fixed source setting; independent source matrices, paths, and relabelings. All benchmark indices retained.',
                reference_prior=dict(alpha=[.5,1.5],depths=list(range(9)),rank='uniform permutation'),
                purpose='Bounded computational benchmark; no cohort mean is a paper estimate.')
    path=args.out/'cohort.json'
    if path.exists() and json.loads(path.read_text()) != cohort: raise ValueError('cohort already frozen with different inputs')
    atomic_json(path,cohort)
    print(json.dumps(dict(prepared=args.n_seq,alpha=args.alpha,depth=args.depth,times=args.times)),flush=True)


def inputs(folder,index,prefix):
    cohort=json.loads((folder/'cohort.json').read_text())
    if cohort['schema'] != TAG or index not in range(cohort['n_sequences']) or prefix not in cohort['prefixes']:
        raise ValueError('outside declared conditional benchmark')
    entry=cohort['records'][index]
    path=folder/'data'/f'seq_{index:06d}.json'
    if sha_file(path)!=entry['file_sha256']: raise ValueError('source file changed')
    source=validate_source_record(json.loads(path.read_text()),identity(cohort['alpha'],cohort['depth'],index,cohort['master_seed']))
    if entry['source_id']!=source['source_id']: raise ValueError('source identity changed')
    tokens=np.asarray(source['tokens'],dtype=int)[:prefix]
    oracle=np.asarray(source['oracle']['transition_probs'])[tokens[-1]]
    predictions={}
    for name in ['d64','d64_seed1']:
        path=folder/'models'/name/f'seq_{index:06d}.json'
        report=json.loads(path.read_text())
        if report['cache_id']!=fingerprint(report['identity']) or report['source_id']!=source['source_id']:
            raise ValueError('model identity failed')
        if report['prefix_tokens_sha256'][str(prefix)]!=sha_array(tokens,'<i2'): raise ValueError('model prefix mismatch')
        j=report['prefixes'].index(prefix)
        if report['targets'][j]!=source['tokens'][prefix] or report['target_indices'][j]!=prefix:
            raise ValueError('model target mismatch')
        array_path=path.with_suffix('.npz')
        if sha_file(array_path)!=report['arrays_sha256']: raise ValueError('model array changed')
        with np.load(array_path,allow_pickle=False) as a:
            prediction=a['predictive_probs'][j]
            if not np.array_equal(prediction,report['predictive_probs'][j]) or not np.array_equal(oracle,a['oracle_probs'][j]):
                raise ValueError('model/oracle readback failed')
        predictions[name]=dict(prediction=prediction,path=str(path.resolve()),sha256=sha_file(path))
    ident=dict(cohort_sha256=sha_file(folder/'cohort.json'),source_id=source['source_id'],
               tokens_sha256=source['tokens_sha256'],prefix_tokens_sha256=sha_array(tokens,'<i2'),
               index=index,prefix=prefix,model_files={k:dict(path=v['path'],sha256=v['sha256']) for k,v in predictions.items()})
    return tokens,oracle,predictions,ident


def reference(args):
    tokens,oracle,predictions,input_identity=inputs(args.out,args.index,args.prefix)
    nodes,weights=leggauss(args.alpha_order);nodes=1+nodes/2;weights=weights/2
    kernel=RankedLSARowEvidence(64,nodes,term_cache_size=args.term_cache_size)
    sampler=SharedBaseSurrogateMCMC(64,kernel,weights,surrogate_steps=64)
    seed=seed_of(TAG,args.sample_seed,args.index,args.prefix,args.alpha_order)
    stem=f'seq_{args.index:06d}_t{args.prefix:04d}'
    folder=args.out/'refs'/f'alpha{args.alpha_order}'
    path=folder/(stem+'.json');moments_path=folder/(stem+'.npz')
    chain_path=args.out/'chains'/f'alpha{args.alpha_order}'/(stem+'.npz')
    run=SurrogateChainRun.load(sampler,tokens,chain_path) if chain_path.exists() else SurrogateChainRun(
        sampler,tokens,chains=4,burn=args.burn,thin=1,seed=seed)
    if (run.seed,run.burn,run.thin,run.chains)!=(seed,args.burn,1,4): raise ValueError('existing chain configuration differs')
    code={name:sha_file(HERE/name) for name in CODE_FILES}
    code['conditional_shared_base_benchmark.py']=sha_file(__file__)
    ident=dict(inputs=input_identity,numerical_problem=run.fingerprint,code_sha256=code,
               proposal_descriptor=run.proposal_descriptor,sampling_seed_tag=TAG)
    old=json.loads(path.read_text()) if path.exists() else None
    if old is not None and old['identity']!=ident: raise ValueError('reference identity changed')
    history=[] if old is None else old['sampling_rounds']
    def progress(event):
        print(json.dumps(dict(index=args.index,prefix=args.prefix,order=args.alpha_order,**event)),flush=True)
        if event['phase']=='burn' or event['draws']==event['requested_draws']: run.save(chain_path)
    budget=max(args.initial_draws,int(run.draw_counts.max()))
    if budget>args.max_draws: raise ValueError('cannot discard existing draws')
    while True:
        estimate=run.extend(budget,block_draws=2048,progress=progress)
        moments=stable_predictive_moments(run.samples['predictive_probs'])
        if not np.allclose(moments['mean'],estimate['predictive_probs'],rtol=0,atol=2e-14): raise ValueError('independent mean disagreement')
        metrics={name:compare_prediction(estimate['predictive_probs'],p['prediction'],oracle,moments) for name,p in predictions.items()}
        gates=diagnostic_gates(estimate)
        passed=all(gates.values()) and all(v['metric_precision_passed'] for v in metrics.values())
        grid=selected_grid_checks(sampler,tokens,estimate) if passed or budget>=args.max_draws else []
        local=passed and bool(grid) and all(v['passed'] for v in grid)
        summary=dict(draws_per_chain=budget,elapsed_seconds=run.elapsed_seconds,diagnostic_gates=gates,
                     sampling_precision_passed=passed,local_validation_passed=local,
                     max_rhat=max(v for k in ('rhat_probs','rhat_depth','rhat_alpha_and_loglik') for v in estimate['diagnostics'][k] if v is not None),
                     min_probability_ess=min(estimate['diagnostics']['ess_probs']),
                     max_probability_mcse=float(np.max(estimate['mcse_probs'])),model_metrics=metrics)
        if history and history[-1]['draws_per_chain']==budget: history[-1]=summary
        else: history.append(summary)
        atomic_npz(moments_path,**moments)
        report=dict(schema=TAG,identity=ident,reference_id=fingerprint(ident),estimate=estimate,
                    sequence_index=args.index,prefix_length=args.prefix,source_id=input_identity['source_id'],
                    settings=dict(alpha_order=args.alpha_order,alpha_nodes=nodes,alpha_weights=weights,
                        depths=sampler.depths,depth_prior=sampler.depth_prior,chains=4,burn=args.burn,thin=1,seed=seed),
                    model_metrics=metrics,diagnostic_gates=gates,local_validation_passed=local,
                    sampling_precision_passed=passed,final_state_grid_checks=grid,sampling_rounds=history,
                    moments_file=moments_path.name,moments_sha256=sha_file(moments_path),chain_checkpoint=str(chain_path.resolve()),
                    cache=kernel.cache_info(),limitation='Fixed-setting computational benchmark. Independent exponent and raw-sample audit required; no population mean.')
        atomic_json(path,json_safe(report));print(json.dumps(json_safe(summary)),flush=True)
        if local or budget>=args.max_draws:break
        budget=min(2*budget,args.max_draws)
    if not local: raise SystemExit('unresolved at declared budget')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','models','reference'],required=True)
    p.add_argument('--out',type=Path,default=ROOT/'output/conditional_shared_base_benchmark_2026-09-15/a1_L4')
    p.add_argument('--alpha',type=float,default=1.0);p.add_argument('--depth',type=int,default=4)
    p.add_argument('--n-seq',type=int,default=8);p.add_argument('--seed',type=int,default=20260930)
    p.add_argument('--times',nargs='+',type=int,default=[16,64,256,1024])
    p.add_argument('--index',type=int,default=0);p.add_argument('--prefix',type=int,default=16)
    p.add_argument('--alpha-order',type=int,default=16);p.add_argument('--burn',type=int,default=16384)
    p.add_argument('--initial-draws',type=int,default=8192);p.add_argument('--max-draws',type=int,default=65536)
    p.add_argument('--sample-seed',type=int,default=20260930);p.add_argument('--term-cache-size',type=int,default=16384)
    args=p.parse_args()
    if not .5<=args.alpha<=1.5 or args.depth not in range(9) or args.n_seq<1: p.error('invalid fixed source setting')
    if args.times!=sorted(set(args.times)) or min(args.times)<1 or max(args.times)>1024:p.error('invalid times')
    if args.stage=='prepare':prepare(args)
    elif args.stage=='models':
        cohort=json.loads((args.out/'cohort.json').read_text());records=[]
        for entry in cohort['records']:
            path=Path(entry['path']);record=json.loads(path.read_text())
            if sha_file(path)!=entry['file_sha256']:raise ValueError('frozen source changed')
            validate_source_record(record)
            records.append(dict(path=path,record=record,file_sha256=entry['file_sha256']))
        models(args.out,records,cohort['prefixes'],[ROOT/'results/transformer/runs/d64',ROOT/'results/transformer/runs/d64_seed1'],device='cpu',batch=4,torch_threads=1)
    else:reference(args)

if __name__=='__main__':main()
