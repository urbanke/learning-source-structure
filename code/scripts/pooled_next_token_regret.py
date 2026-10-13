"""Recover pooled next-token regret from frozen scores and validated short references.

Transformer scores are read, never inferred again. At times1/4 the original
reference providers reconstruct every next-token probability using the existing
sealed row tables and short-prefix catalog; no new posterior sampler or numerical
rule is introduced. The original saved family weights and realized-child
likelihoods provide independent alignment/normalization checks.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp, xlogy

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
DEST=ROOT/'results/transformer/multifamily_bayes_2026-09-16'
RAW=ROOT/'output/multifamily_results_2026-09-16'
SHORT=ROOT/'output/multifamily_bayes_2026-09-16/short_comparison'
OUT=ROOT/'output/multifamily_bayes_2026-09-16/next_token_regret'
FAMILIES=['product','shared_base','uniform_base']
TIMES=[1,4,16,64,256,1024]


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(tokens):
    labels={}
    return tuple(labels.setdefault(int(x),len(labels)) for x in tokens)


def moments(values):
    return dict(n_seq=len(values),mean=np.mean(values,axis=0).tolist(),
                se=(np.std(values,axis=0,ddof=1)/np.sqrt(len(values))).tolist())


def balanced(values):
    return dict(n_seq=3072,mean=np.mean([v['mean'] for v in values],axis=0).tolist(),
                se=(np.sqrt(np.sum(np.square([v['se'] for v in values]),axis=0))/3).tolist())


def run():
    started=time.monotonic()
    from mixture_family_evidence import FamilyEvidence, chain_log_marginal
    from shared_base_short_evidence import SharedBaseShortEvidence
    old_path=DEST/'short_validation.json';old=json.loads(old_path.read_text())
    original_path=ROOT/'results/transformer/multifamily_2026-09-16/validation.json'
    original=json.loads(original_path.read_text())
    require(old['passed'] is True and old['checks']['passed'] is True
            and original['status']=='passed','existing evaluation audits must pass')
    verified={str(old_path):sha(old_path),str(original_path):sha(original_path)}
    def checked(path,expected):
        require(sha(path)==expected,f'validated input changed: {path}')
        verified[str(path)]=expected
    def checked_old(path):
        checked(path,old['verified_file_sha256'][str(path)])
    numerical_path=DEST/'family_evidence_validation.json'
    checked(numerical_path,old['numerical_validations'][str(numerical_path)])
    numerical=json.loads(numerical_path.read_text());require(numerical['passed'] is True,'family reference audit failed')
    for name,digest in numerical['code_sha256'].items():checked(ROOT/'code/transformer'/name,digest)
    for name in ['PMM_PHI_BIAS','PMM_PHI_WAVE']:
        require(float(os.environ.get(name,'0'))==0,f'nonzero numerical perturbation: {name}')
    store=Path(os.environ['LSA_STORE']);source=Path(os.environ['PMM_SRC'])
    backend=source/'product_model_with_memory/universal_tables.py'
    reference_configs=[]
    for ref_family in ['product','uniform_base']:
        for family in FAMILIES:
            config_path=SHORT.parent/'family_evidence'/f'{ref_family}_on_{family}_prior.json'
            checked_old(config_path)
            config=json.loads(config_path.read_text());reference_configs.append(config)
            require(config['bounds']==[-30.,30.] and config['grid_step']==1/64,
                    'saved numerical rule changed')
            checked(backend,config['backend_sha256'])
            for name,digest in config['store_file_sha256'].items():
                if str(store/name) not in verified: checked(store/name,digest)
                require(verified[str(store/name)]==digest,'reference store identities differ')
    shared=SharedBaseShortEvidence()
    verified.update(shared.validation['verified_file_sha256'])
    product=FamilyEvidence('product');uniform=FamilyEvidence('uniform_base')
    for config in reference_configs:
        require(all(os.environ.get(name)==value for name,value in config['backend_environment'].items()),
                'reference backend environment changed')
    @lru_cache(None)
    def coordinate_joint(pattern):
        return chain_log_marginal(np.asarray(pattern,dtype=np.int64),8,product.kernel)
    @lru_cache(None)
    def uniform_joint(pattern):
        return uniform.log_joint(np.asarray(pattern,dtype=np.int64))
    @lru_cache(None)
    def shared_joint(pattern):
        return shared.log_joint(np.asarray(pattern,dtype=np.int64))
    def joints(tokens):
        x=np.asarray(tokens)
        return np.array([coordinate_joint(canonical(x//8))+coordinate_joint(canonical(x%8)),
                         shared_joint(canonical(x)),uniform_joint(canonical(x))])
    checks=dict(component_normalization=0.,mixture_normalization=0.,saved_parent_nats=0.,
        saved_child_nats=0.,saved_weights=0.,direct_vs_weighted=0.,oracle_row=0.,oracle_entropy=0.)
    cohorts={};raw_hashes={};OUT.mkdir(parents=True,exist_ok=True)
    for family in FAMILIES:
        name=f'{family}_prior';source_path=RAW/'heldout'/f'{name}.npz';checked_old(source_path)
        short_path=SHORT/f'{name}.npz';checked(short_path,old['raw_output_sha256'][str(short_path)])
        with np.load(source_path,allow_pickle=False) as f:
            tokens=f['tokens'];oracle=f['oracle_rows'];seeds=f['sequence_seed'];prefixes=f['prefixes']
            transitions=f['transition_probs'];entropy=f['source_entropy_bits']
        require(tokens.shape==(1024,1025) and oracle.shape==(1024,6,64)
                and prefixes.tolist()==TIMES,'source shape/time grid mismatch')
        expected=transitions[np.arange(1024)[:,None],tokens[:,np.asarray(TIMES)-1],:]
        checks['oracle_row']=max(checks['oracle_row'],float(np.max(abs(expected-oracle))))
        checks['oracle_entropy']=max(checks['oracle_entropy'],float(np.max(abs(-np.sum(xlogy(oracle,oracle),axis=-1)/np.log(2)-entropy))))
        require(np.isfinite(oracle).all() and np.all(oracle>=0)
                and np.max(abs(oracle.sum(axis=-1)-1))<1e-12,'invalid oracle distribution')
        with np.load(short_path,allow_pickle=False) as f:
            require(np.array_equal(f['sequence_seed'],seeds) and f['prefixes'].tolist()==[1,4]
                    and f['family_names'].tolist()==FAMILIES,'short reference alignment changed')
            parents=f['log_marginals_before_prediction'];children_saved=f['log_marginals_after_prediction']
            weights=f['posterior_family_weights_before_prediction']
        probabilities=np.empty((1024,2,64))
        for i in range(1024):
            for j,t in enumerate([1,4]):
                parent=joints(tokens[i,:t])
                children=np.stack([joints(np.r_[tokens[i,:t],b]) for b in range(64)],axis=1)
                component=np.exp(children-parents[i,j,:,None])
                q=weights[i,j]@component
                direct=np.exp(logsumexp(children,axis=0)-logsumexp(parents[i,j]))
                checks['component_normalization']=max(checks['component_normalization'],float(np.max(abs(component.sum(axis=1)-1))))
                checks['mixture_normalization']=max(checks['mixture_normalization'],float(abs(q.sum()-1)))
                checks['saved_parent_nats']=max(checks['saved_parent_nats'],float(np.max(abs(parent-parents[i,j]))))
                checks['saved_child_nats']=max(checks['saved_child_nats'],float(np.max(abs(children[:,tokens[i,t]]-children_saved[i,j]))))
                checks['saved_weights']=max(checks['saved_weights'],float(np.max(abs(np.exp(parents[i,j]-logsumexp(parents[i,j]))-weights[i,j]))))
                checks['direct_vs_weighted']=max(checks['direct_vs_weighted'],float(np.max(abs(direct-q))))
                probabilities[i,j]=q
        require(np.isfinite(probabilities).all() and np.all(probabilities>0),'invalid mixture probabilities')
        p=oracle[:,:2]
        regret=np.sum(xlogy(p,p)-xlogy(p,probabilities),axis=-1)/np.log(2)
        require(np.min(regret)>-1e-9,'negative mixture KL')
        models={};arrays=dict(prefixes=np.asarray([1,4]),sequence_seed=seeds,
            mixture_probs=probabilities,oracle_probs=p,mixture_next_kl_bits=regret)
        for seed in [0,1]:
            model=f'pooled_seed{seed}';path=RAW/'eval'/model/f'{name}.npz';checked_old(path)
            record_path=path.with_suffix('.json');checked_old(record_path)
            record=json.loads(record_path.read_text())
            require(record['identity']['source_plan_id']==old['source_plan_id']
                    and record['source_data_sha256']==sha(source_path)
                    and record['data_sha256']==sha(path),'model/cohort identity changed')
            with np.load(path,allow_pickle=False) as f:
                require(np.array_equal(f['sequence_seed'],seeds) and f['prefixes'].tolist()==TIMES,'model time/sequence mismatch')
                values=f['next_kl_bits']
            require(values.shape==(1024,6) and np.isfinite(values).all() and values.min()>-1e-9,'invalid transformer KL')
            models[model]=dict(next_token_regret=moments(values),transformer_minus_mixture=moments(values[:,:2]-regret),
                run_id=record['run_id'],checkpoint_sha256=record['identity']['checkpoint_sha256'])
            arrays[model+'_next_kl_bits']=values
        target=OUT/f'{name}.npz';np.savez_compressed(target,**arrays);raw_hashes[str(target)]=sha(target)
        cohorts[family]=dict(n_seq=1024,bayes=moments(regret),models=models)
        print(json.dumps(dict(family=family,bayes=cohorts[family]['bayes'])),flush=True)
    require(all(v<1e-9 for v in checks.values()),'short mixture probability/alignment check failed')
    summary=dict(schema='pooled-next-token-regret-v1',source_plan_id=old['source_plan_id'],
        family_order=FAMILIES,family_prior_weights=[1/3]*3,n_sequences=3072,n_sequences_per_family=1024,
        n_tokens=1025,transformer_times=TIMES,bayes_times=[1,4],cohorts=cohorts,
        bayes=balanced([c['bayes'] for c in cohorts.values()]),models={},
        quantity='Expected next-token regret: sum over all64 tokens of true next-token probability times log2(true probability/predictive probability).',
        uncertainty='One SE over independent sequences, stratified by family: sqrt(sum(SE_f^2))/3. Both training runs remain separate.',
        recovery='Saved transformer next_kl scores; mixture predictions reconstructed from original short-reference providers and saved posterior family weights.',
        limitation='Transformer probability vectors were not retained; original independently audited next_kl scores are reused without re-inference.')
    for seed in [0,1]:
        model=f'pooled_seed{seed}'
        summary['models'][model]={key:balanced([c['models'][model][key] for c in cohorts.values()])
            for key in ['next_token_regret','transformer_minus_mixture']}
    validation=dict(schema='pooled-next-token-validation-v1',passed=True,source_plan_id=old['source_plan_id'],
        summary_payload_sha256=hashlib.sha256(json.dumps(summary,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        independent_sequences=3072,checks=checks,inputs=verified,raw_output_sha256=raw_hashes,
        script_sha256=sha(__file__),reference_adapter_sha256=sha(ROOT/'code/transformer/shared_base_short_evidence.py'),
        family_prior_weights=[1/3]*3,reference_times=[1,4],new_transformer_inference=False,
        new_source_sampling=False,new_posterior_sampling=False,new_numerical_rule=False,
        numerical_backend=dict(store=os.environ.get('LSA_STORE'),source=os.environ.get('PMM_SRC'),
            backend_sha256=sha(backend),store_file_sha256=reference_configs[0]['store_file_sha256'],
            environment=reference_configs[0]['backend_environment'],
            product_cache=product.kernel.cache_info(),uniform_cache=uniform.kernel.cache_info()))
    validation['runtime_seconds']=time.monotonic()-started
    DEST.mkdir(parents=True,exist_ok=True)
    validation_path=DEST/'next_token_validation.json';validation_path.write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    summary['validation_receipt']=dict(path=str(validation_path.relative_to(ROOT)),sha256=sha(validation_path))
    path=DEST/'next_token_regret.json';path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(bayes=summary['bayes'],models=summary['models'],checks=checks)),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
