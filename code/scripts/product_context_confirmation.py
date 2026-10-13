"""Frozen, fresh-sequence context-length test using the existing product APIs.

Stages: prepare, benchmark, references, forecast, model, summary. The model
stage requires a sealed complete Bayesian forecast. Earlier experiments and
their numerical code are never overwritten. Raw records stay in output/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.special import xlogy

import conditional_product_accuracy as previous
from section4_product_accuracy import first_tested_context

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'output/product_context_confirmation_2026-09-17'
DEFAULT_RESULTS = ROOT / 'results/transformer/product_context_confirmation_2026-09-17'
DEVELOPMENT = ROOT / 'results/transformer/conditional_accuracy/product'
PAIRS = [[1, 1], [1, 8], [4, 4], [8, 8]]
TARGETS = [.1, .2, .5]
SCHEMA = 'product-context-confirmation-v1'
BACKEND_DEFAULTS = dict(PMM_SERIES_TAIL='40', PMM_PHI_WAVE='0', PMM_PHI_WAVE_SCALE='4',
    PMM_PHI_BIAS='0', PMM_PHI_SADDLE_MIN_L='54', PMM_PHI_LADDER='0',
    PMM_PHI_LADDER_EVERY='1', PMM_PHI_LADDER_DEGREE='11', PMM_INTERP_KERNEL='1')
existing = previous.existing
require = previous.require
sha = existing._sha_file


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def backend_environment():
    """Bind effective sealed-store settings before any backend import."""
    effective = {name:os.environ.get(name,default) for name,default in BACKEND_DEFAULTS.items()}
    require(effective == BACKEND_DEFAULTS, 'Nonstandard or experimental LSA backend settings')
    for name,value in effective.items():os.environ.setdefault(name,value)
    return effective


def dense_grid(development):
    """Design depends only on the pre-existing Bayesian development means."""
    times = {2**j for j in range(11)}
    forecasts = []
    require([c['depth_pair'] for c in development['cases']] == PAIRS,
            'Development conditions differ')
    for c in development['cases']:
        for target in TARGETS:
            n = first_tested_context(c['times'], c['informed_bayes']['mean'], target)
            forecasts.append(dict(depth_pair=c['depth_pair'], target_bits=target,
                                  development_first_tested_context=n))
            if n is not None:
                times.update(int(np.clip(np.ceil(n * 2**(j/4)), 1, 1024))
                             for j in range(-4, 5))
    return sorted(times), forecasts


def seed_audit(protocol, development_protocol):
    """Check the actual generator seeds, including cross-condition collisions."""
    def seeds(spec):
        return [previous.seed_of(previous.SCHEMA, spec['master_seed'], *pair, i)
                for pair in spec['depth_pairs']
                for i in range(spec['n_sequences_per_pair'])]
    fresh, old = seeds(protocol), seeds(development_protocol)
    require(len(set(fresh)) == len(fresh), 'Duplicate confirmation sequence seed')
    require(set(fresh).isdisjoint(old), 'Confirmation reuses a development sequence seed')
    return dict(confirmation_seeds=len(fresh), development_seeds=len(old),
                unique_confirmation_seeds=True, disjoint_from_development=True,
                confirmation_seed_list_sha256=existing._fingerprint(fresh))


def numerical_args(out, times, *, n_seq=1024, seed=2026091704,
                   device='cpu', batch=8, torch_threads=4):
    return argparse.Namespace(out=str(out), results=str(Path(out)/'numerical_validation'),
        n_seq=n_seq, n_ctx=1024, seed=seed, prefixes=np.asarray(times, dtype=int),
        product_run=str(ROOT/'results/transformer/runs/prod_8x8'), step=100000,
        u_lo=-30., u_hi=30., table_dir=str(ROOT/'code/transformer'), prob_tol=2e-6,
        device=device, batch=batch, torch_threads=torch_threads)


def prepare(out):
    require(not (out/'plan.json').exists(), 'Plan already exists; use the existing stages')
    environment = backend_environment()
    out.mkdir(parents=True,exist_ok=True)
    development = read(DEVELOPMENT/'summary.json')
    receipt = read(DEVELOPMENT/'receipt.json')
    require(receipt['passed'] and receipt['validation']['passed'], 'Development audit failed')
    for name, digest in receipt['files_sha256'].items():
        require(sha(ROOT/name) == digest, 'Changed development input: '+name)
    times, forecasts = dense_grid(development)
    args = numerical_args(out, times)
    case, protocol = previous.setup(args)
    require(protocol['master_seed'] != development['protocol']['master_seed'],
            'Confirmation seed must differ from development')
    seeds = seed_audit(protocol, development['protocol'])
    files = [Path(__file__), ROOT/'code/scripts/product_context_statistics.py',
             Path(previous.__file__), ROOT/'code/scripts/section4_product_accuracy.py',
             ROOT/'code/transformer/eval_shared_depth.py', ROOT/'code/transformer/model.py',
             ROOT/'code/transformer/shared_depth.py', ROOT/'code/transformer/lsa_seq.py',
             ROOT/'code/transformer/phi_source.py', ROOT/'code/transformer/source.py',
             ROOT/'code/transformer/families.py', DEVELOPMENT/'summary.json',
             DEVELOPMENT/'receipt.json', out/'protocol.json', case['checkpoint'], case['config_path']]
    refspec = previous.reference_spec(case, args)
    require(refspec['store_manifest_sha256'] and refspec['backend_sha256'],
            'Existing sealed LSA store is required')
    plan = dict(schema=SCHEMA, prepared_utc=utc(), out=str(out.resolve()),
        depth_pairs=PAIRS, targets_bits=TARGETS, times=times, n_sequences_per_pair=1024,
        n_tokens=1025, master_seed=args.seed, benchmark_indices=list(range(8)),
        benchmark_sequences=32, bootstrap_replicates=4096, bootstrap_seed=2026091705,
        confidence=.95, quantity='Expected next-token KL regret in bits',
        crossing='First tested time whose population mean regret is at most target; no interpolation',
        nonattainment='Keep unreached outcomes in estimates and bootstrap samples',
        grid_rule='Union of powers of two and ceil(n_B*2^(j/4)),j=-4..4,clipped1..1024; development Bayes only',
        development_forecasts=forecasts, seed_audit=seeds, numerical_reference=refspec,
        backend_environment=environment,
        model=existing._model_spec(case, args.prefixes),
        inputs={str(p.resolve()):sha(p) for p in files},
        model_scope='One existing fixed trained checkpoint; no training-seed generalization',
        stop_rule='All1024 sequences per condition; do not stop for favorable agreement')
    plan['plan_id'] = existing._fingerprint(plan)
    write(out/'plan.json', plan)
    print(json.dumps(dict(stage='prepared', plan_id=plan['plan_id'], times=times,
                          sequences=4096, new_training=False)), flush=True)


def load_plan(out, cli=None):
    plan = read(out/'plan.json')
    require(backend_environment()==plan['backend_environment'], 'Numerical environment changed')
    require(plan['schema']==SCHEMA and plan['out']==str(out.resolve()), 'Plan location/schema differs')
    require(plan['plan_id']==existing._fingerprint({k:v for k,v in plan.items() if k!='plan_id'}),
            'Frozen plan changed')
    require(plan['depth_pairs']==PAIRS and plan['targets_bits']==TARGETS
            and plan['n_sequences_per_pair']==1024 and plan['n_tokens']==1025,
            'Scientific design changed')
    for name,digest in plan['inputs'].items():
        require(sha(name)==digest, 'Frozen input changed: '+name)
    args = numerical_args(out, plan['times'], seed=plan['master_seed'],
        device=getattr(cli,'device','cpu'), batch=getattr(cli,'batch',8),
        torch_threads=getattr(cli,'torch_threads',4))
    case, protocol = previous.setup(args)
    require(previous.reference_spec(case,args)==plan['numerical_reference'],
            'Reference backend/store/settings differ')
    require(existing._model_spec(case,args.prefixes)==plan['model'], 'Model differs')
    return plan,args,case,protocol


def cases_and_data(case, protocol, args):
    return [(dict(case,name=f'L{a}_{b}'), [a,b], previous.dataset([a,b],protocol,args))
            for a,b in PAIRS]


def source_risk(data, prediction):
    p = np.asarray(data['true_probs'],dtype=float)
    prediction = np.asarray(prediction,dtype=float)
    require(p.shape==prediction.shape and p.ndim==2 and p.shape[1]==64,'Prediction dimensions differ')
    existing._normalization_error(p,'source',2e-12)
    existing._normalization_error(prediction,'prediction',2e-6)
    require(np.min(prediction)>0, 'Predictor support missing')
    result = np.sum(xlogy(p,p)-xlogy(p,prediction),axis=-1)/np.log(2.)
    require(np.isfinite(result).all() and result.min()>-1e-7,'Invalid expected regret')
    return result


def benchmark(out, cli):
    plan,args,case,_ = load_plan(out,cli)
    require(not (out/'benchmark.json').exists(),'Benchmark already sealed')
    started=time.monotonic(); args.out=str(out/'benchmark_raw')
    args.results=str(out/'benchmark_raw/validation'); args.n_seq=8
    development=read(DEVELOPMENT/'summary.json')
    spec=dict(development['protocol'],times=plan['times'])
    cases=[]
    for a,b in PAIRS:
        rows=[]; name=f'L{a}_{b}'
        for i in plan['benchmark_indices']:
            record=previous.source_record([a,b],i,spec)
            with np.load(ROOT/f'output/conditional_accuracy/product/{name}/data/seq_{i:06d}.npz',allow_pickle=False) as old:
                require(np.array_equal(record['tokens'],old['tokens']), 'Benchmark must reuse old sequences')
            existing._atomic_npz(Path(args.out)/name/'data'/f'seq_{i:06d}.npz',**record)
            rows.append(record)
        cases.append((dict(case,name=name),[a,b],rows))
    start=time.monotonic(); reference=previous.make_reference(args)
    setup_seconds=time.monotonic()-start
    start=time.monotonic()
    for c,_,data in cases:previous.references(c,data,args,reference)
    ref_seconds=time.monotonic()-start
    # Match all old times and independent expanded-range/Dirichlet controls.
    old_max=0.
    common=[plan['times'].index(t) for t in development['protocol']['times']]
    for c,_,data in cases:
        for i in range(8):
            new=existing._read_cache(Path(args.out)/c['name']/'references'/f'seq_{i:06d}.npz',
                 existing._reference_id(data[i],previous.reference_spec(c,args)))
            with np.load(ROOT/f"output/conditional_accuracy/product/{c['name']}/references/seq_{i:06d}.npz",allow_pickle=False) as old:
                old_max=max(old_max,float(np.max(np.abs(new['bayes_probs'][common]-old['bayes_probs']))))
    require(old_max<2e-12,'Common-time Bayes predictions changed')
    Path(args.results).mkdir(parents=True,exist_ok=True)
    previous.validate(cases,args)
    start=time.monotonic()
    for c,_,data in cases:existing._models(c,data,args,args.prefixes)
    model_seconds=time.monotonic()-start
    model_max=0.
    for c,_,data in cases:
        for i in range(8):
            new=existing._read_cache(Path(args.out)/c['name']/'model'/f'seq_{i:06d}.npz',
                 existing._model_id(data[i],existing._model_spec(c,args.prefixes)))
            with np.load(ROOT/f"output/conditional_accuracy/product/{c['name']}/model/seq_{i:06d}.npz",allow_pickle=False) as old:
                model_max=max(model_max,float(np.max(np.abs(np.exp(new['logp'][common].astype(float))-np.exp(old['logp'].astype(float))))))
    require(model_max<2e-6,'Model prediction replay differs')
    report=dict(schema=SCHEMA+'-benchmark',passed=True,plan_id=plan['plan_id'],created_utc=utc(),
        development_sequences=32, reference_setup_seconds=setup_seconds,
        reference_seconds=ref_seconds, model_seconds=model_seconds,
        projected_serial_reference_hours=ref_seconds/32*4096/3600,
        projected_model_hours=model_seconds/32*4096/3600,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        common_time_bayes_max_error=old_max,common_time_model_max_error=model_max,
        numerical_validation=read(Path(args.results)/'validation.json'),
        elapsed_seconds=time.monotonic()-started)
    write(out/'benchmark.json',report);print(json.dumps(report),flush=True)


_WORKER=None


def worker_init(out):
    global _WORKER
    plan,args,case,protocol=load_plan(Path(out))
    _WORKER=(plan,args,case,protocol,previous.make_reference(args))


def reference_task(pair,index):
    _,args,case,protocol,reference=_WORKER
    c=dict(case,name=f'L{pair[0]}_{pair[1]}')
    sequence_seed=previous.seed_of(previous.SCHEMA,protocol['master_seed'],*pair,index)
    metadata=dict(protocol=protocol,index=index,sequence_seed=sequence_seed,depth_pair=pair)
    record=existing._read_cache(Path(args.out)/c['name']/'data'/f'seq_{index:06d}.npz',existing._fingerprint(metadata))
    require(record is not None and existing._sha_tokens(record['tokens'])==str(record['tokens_sha256']), 'Source missing or corrupt')
    path=Path(args.out)/c['name']/'references'/f'seq_{index:06d}.npz'
    spec=previous.reference_spec(c,args); ident=existing._reference_id(record,spec)
    start=time.monotonic(); cached=existing._read_cache(path,ident)
    if cached is None:
        result=reference.run(record['tokens'].astype(int),full_probs=True,positions=args.prefixes)
        p=result['probs'];error=existing._normalization_error(p,'Bayes',args.prob_tol)
        require(np.max(np.abs(-np.log2(p[np.arange(len(args.prefixes)),record['tokens'][args.prefixes]])-result['loss'][args.prefixes]))<2e-6,
                'Selected-time Bayes loss differs')
        source_risk(record,p)
        existing._atomic_npz(path,cache_id=ident,metadata_json=existing._json(dict(reference=spec,
            tokens_sha256=str(record['tokens_sha256']),runtime=existing._runtime_versions())),
            prefixes=args.prefixes,bayes_probs=p,bayes_loss=result['loss'][1:],
            log_evidence=result['log_evidence'],max_normalization_error=error)
    return dict(case=c['name'],index=index,seconds=time.monotonic()-start,cached=cached is not None)


def references(out,cli):
    plan,args,case,protocol=load_plan(out,cli)
    bench=read(out/'benchmark.json')
    require(bench['passed'] and bench['plan_id']==plan['plan_id'],'Successful matching benchmark required')
    require(1<=cli.workers<=6,'At most six bounded single-thread workers on the laptop')
    cases_and_data(case,protocol,args)
    jobs=[]
    for pair in PAIRS:
        jobs.extend((pair,i) for i in range(args.n_seq))
    start=time.monotonic(); count=0; failures=[]
    write(out/'status.json',dict(stage='references',status='running',completed=0,total=4096,updated_utc=utc()))
    with ProcessPoolExecutor(max_workers=cli.workers,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(str(out),)) as pool:
        futures={pool.submit(reference_task,*job):job for job in jobs}
        for f in as_completed(futures):
            try:f.result();count+=1
            except Exception as exc:failures.append(dict(task=futures[f],error=str(exc)))
            if (count+len(failures))%32==0 or count+len(failures)==4096:
                status=dict(stage='references',status='running',completed=count,total=4096,
                    failures=failures,elapsed_seconds=time.monotonic()-start,updated_utc=utc())
                write(out/'status.json',status);print(json.dumps(status),flush=True)
    write(out/'status.json',dict(stage='references',status='complete' if not failures else 'failed',
        completed=count,total=4096,failures=failures,elapsed_seconds=time.monotonic()-start,updated_utc=utc()))
    require(not failures and count==4096,'Incomplete references; every source is required')


def forecast(out,cli):
    plan,args,case,protocol=load_plan(out,cli)
    require(not (out/'forecast.json').exists(),'Forecast already sealed')
    require(not any(out.glob('L*/model/seq_*.npz')),'New model predictions already exist before forecast')
    cases=cases_and_data(case,protocol,args)
    Path(args.results).mkdir(parents=True,exist_ok=True)
    previous.validate(cases,args)
    rows=[]; bindings={str(out/'plan.json'):sha(out/'plan.json')}
    for c,pair,data in cases:
        risks=[]
        require(len(list((out/c['name']/'references').glob('seq_*.npz')))==1024,'Reference coverage differs')
        for i,record in enumerate(data):
            path=out/c['name']/'references'/f'seq_{i:06d}.npz'
            ref=existing._read_cache(path,existing._reference_id(record,previous.reference_spec(c,args)))
            require(ref is not None and np.array_equal(ref['prefixes'],args.prefixes),'Missing reference/time identity')
            risks.append(source_risk(record,ref['bayes_probs']))
            for p in (path,out/c['name']/'data'/f'seq_{i:06d}.npz'):bindings[str(p)]=sha(p)
        moments=previous.stats(np.asarray(risks))
        rows.append(dict(id=c['name'],depth_pair=pair,n_sequences=1024,times=plan['times'],
            informed_bayes=moments,forecasts=[dict(target_bits=t,first_tested_context=first_tested_context(plan['times'],moments['mean'],t)) for t in TARGETS]))
    validation=out/'numerical_validation/validation.json';bindings[str(validation)]=sha(validation)
    result=dict(schema=SCHEMA+'-forecast',plan_id=plan['plan_id'],sealed_utc=utc(),
        complete=True,new_model_predictions_present=False,cases=rows,inputs=bindings,
        numerical_validation=read(validation))
    write(out/'forecast.json',result)
    write(out/'forecast_seal.json',dict(sha256=sha(out/'forecast.json'),created_utc=utc()))
    print(json.dumps(dict(stage='forecast_sealed',cases=rows)),flush=True)


def verify_forecast(out,plan):
    value=read(out/'forecast.json');seal=read(out/'forecast_seal.json')
    require(sha(out/'forecast.json')==seal['sha256'] and value['complete'] is True
        and value['plan_id']==plan['plan_id'] and value['new_model_predictions_present'] is False,
        'Complete frozen Bayesian forecast required')
    for name,digest in value['inputs'].items():require(sha(name)==digest,'Forecast input changed: '+name)
    return value


def model(out,cli):
    plan,args,case,protocol=load_plan(out,cli);verify_forecast(out,plan)
    if (out/'model_started.json').exists():
        require(read(out/'model_started.json')['forecast_sha256']==sha(out/'forecast.json'),
                'Resumed model evaluation refers to another forecast')
    else:
        write(out/'model_started.json',dict(started_utc=utc(),forecast_sha256=sha(out/'forecast.json'),
                                          device=args.device,threads=args.torch_threads,batch=args.batch))
    for c,_,data in cases_and_data(case,protocol,args):existing._models(c,data,args,args.prefixes)
    write(out/'model_complete.json',dict(completed_utc=utc(),plan_id=plan['plan_id'],sequences=4096))


def summary(out,cli):
    from product_context_statistics import analyze_context_requirements
    plan,args,case,protocol=load_plan(out,cli);forecast_value=verify_forecast(out,plan)
    require(read(out/'model_complete.json')['plan_id']==plan['plan_id'],'Model evaluation incomplete')
    curves=[];risks={};inputs={str(out/'plan.json'):sha(out/'plan.json'),str(out/'forecast.json'):sha(out/'forecast.json')}
    for c,pair,data in cases_and_data(case,protocol,args):
        require(len(list((out/c['name']/'model').glob('seq_*.npz')))==1024,'Model cohort coverage differs')
        row=previous.collect(c,pair,data,args);row['id']=c['name'];curves.append(row)
        expected=next(r for r in forecast_value['cases'] if r['id']==c['name'])
        require(np.allclose(row['informed_bayes']['mean'],expected['informed_bayes']['mean'],atol=1e-13,rtol=0),'Forecast means changed')
        path=out/c['name']/'per_sequence_risks.npz'
        with np.load(path,allow_pickle=False) as a:risks[c['name']]=np.stack([a['informed_bayes'],a['transformer']],axis=-1)
        inputs[str(path)]=sha(path)
    statistics=analyze_context_requirements(risks,plan['times'],TARGETS,
        bootstrap_replicates=plan['bootstrap_replicates'],seed=plan['bootstrap_seed'],confidence=plan['confidence'])
    result=dict(schema=SCHEMA+'-summary',plan_id=plan['plan_id'],complete=True,
        confirmation=True,created_utc=utc(),plan=plan,curves=curves,statistics=statistics,
        forecast_sha256=sha(out/'forecast.json'),inputs=inputs,
        uncertainty='Whole-sequence paired bootstrap; one fixed trained model, no training-seed variation')
    write(Path(cli.results)/'summary.json',result)
    print(json.dumps(dict(stage='summary',complete=True,out=str(cli.results))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','benchmark','references','forecast','model','summary'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--results',type=Path,default=DEFAULT_RESULTS)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','mps','cuda','auto'],default='cpu')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--torch-threads',type=int,default=4)
    args=parser.parse_args();args.out=args.out.resolve()
    if args.stage=='prepare':prepare(args.out)
    else:globals()[args.stage](args.out,args)


if __name__=='__main__':main()
