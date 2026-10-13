"""Bayes-only development pilot for the matched unigram/copier mixture.

Reuses SeqLSA and its read-only sealed store, plus Source's existing
normalized-exponential row generator and sequence sampler. The copier has
independent row depths. No transformer is trained or evaluated here.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.special import expit, gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/transformer'))
from source import Source, seed_of
from lsa_seq import SeqLSA
from phi_source import open_tables

SCHEMA = 'section4-copying-pilot-v1'
DEFAULT_OUT = ROOT/'output/section4_copying_pilot_2026-09-17'
DEFAULT_RESULTS = ROOT/'results/transformer/section4_copying_pilot_2026-09-17'
D, N_TOKENS, N_PER_BRANCH = 64, 1025, 256
TIMES = np.array([2**i for i in range(11)], dtype=np.int64)
PRIORS = (.25, .125)
SEED = 2026091711
BACKEND_DEFAULTS = dict(PMM_SERIES_TAIL='40', PMM_PHI_WAVE='0', PMM_PHI_WAVE_SCALE='4',
    PMM_PHI_BIAS='0', PMM_PHI_SADDLE_MIN_L='54', PMM_PHI_LADDER='0',
    PMM_PHI_LADDER_EVERY='1', PMM_PHI_LADDER_DEGREE='11', PMM_INTERP_KERNEL='1')
KERNELS = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def setup_environment():
    os.environ.setdefault('LSA_STORE', str(Path.home()/'Projects/product_model_with_memory/tables/anchors_prod'))
    os.environ.setdefault('PMM_SRC', str(Path.home()/'Projects/product_model_with_memory/src'))
    for name, value in BACKEND_DEFAULTS.items():
        require(os.environ.get(name, value) == value, f'Nonstandard backend setting {name}')
        os.environ.setdefault(name, value)
    store, src = Path(os.environ['LSA_STORE']), Path(os.environ['PMM_SRC'])
    require((store/'manifest.json').is_file(), 'Existing sealed LSA store missing')
    require((src/'product_model_with_memory/universal_tables.py').is_file(), 'Store implementation missing')
    return store, src


def plan():
    store, src = setup_environment()
    paths = [Path(__file__), ROOT/'code/transformer/lsa_seq.py', ROOT/'code/transformer/phi_source.py',
             ROOT/'code/transformer/source.py', store/'manifest.json',
             src/'product_model_with_memory/universal_tables.py']
    value = dict(schema=SCHEMA, d=D, n_tokens=N_TOKENS, n_sequences_per_branch=N_PER_BRANCH,
        depth_maxima=[4,8], depth_priors='uniform on integers 1,...,depth_max',
        branches=['iid', 'copy'], copier_depth_dependence='independent across all 64 rows',
        branch_priors=list(PRIORS), times=TIMES.tolist(), master_seed=SEED,
        distinctness_threshold_tv=.05, quadrature=dict(u_lo=-30.,u_hi=30.,step=1/64),
        wide_quadrature=dict(u_lo=-45.,u_hi=40.,step=1/64),
        backend_environment=BACKEND_DEFAULTS, lsa_store=str(store), pmm_src=str(src),
        input_sha256={str(path):sha(path) for path in paths},
        inference='Bayes-only candidate-selection pilot; not final confirmation',
        posterior_crossing='first tested time with posterior copy weight strictly above 1/2',
        distinguishable_crossing='first tested time with weight > 1/2 and component TV >= .05',
        denominator='all 256 sequences per generating branch; no filtering on outcome',
        first_token='iid sampled from its random P; copier uniform over 64 tokens',
        training_started=False)
    value['plan_id'] = fingerprint(value)
    return value


def initialize():
    global KERNELS
    setup_environment()
    base = SeqLSA(D, depths=range(1,9), rmax=N_TOKENS, u_lo=-30., u_hi=30.)
    small = copy.copy(base)
    small.depths, small.Lmax = list(range(1,5)), 4
    small.LN0, small.D = base.LN0[:4], base.D[:,:4]
    KERNELS = {4:small,8:base}


def source_record(depth_max, branch, index):
    """Reuse independent rows from Source.matrices; do not share their depths."""
    seed = seed_of(SCHEMA, SEED, depth_max, branch, index)
    rng = np.random.default_rng(seed)
    src = Source(D, N_TOKENS, alpha=0., ls=(1,depth_max), permute=False)
    if branch == 'copy':
        depths = rng.integers(1,depth_max+1,size=D)
        matrix = np.empty((D,D))
        # Source.matrices draws independent normalized products of Exp(1)
        # variables. Select only rows whose independent depth equals L.
        for depth in np.unique(depths):
            _, candidate = src.matrices(np.array([0.]),np.array([depth]),rng)
            select = depths == depth
            matrix[select] = candidate[0,select]
        initial = np.full(D,1/D)
    else:
        require(branch == 'iid','Unknown generating branch')
        depths = rng.integers(1,depth_max+1,size=1)
        _, candidate = src.matrices(np.array([0.]),depths,rng)
        initial = candidate[0,0].copy()
        matrix = np.repeat(initial[None,:],D,axis=0)
    tokens = src.sequences(initial[None,:],matrix[None,:,:],rng)[0]
    require(np.max(np.abs(matrix.sum(axis=1)-1))<2e-14,'Source row not normalized')
    return dict(tokens=tokens, source_depths=depths, source_initial=initial,
                source_matrix=matrix, sequence_seed=np.array(seed,dtype=np.uint64))


def predictions(kernel, source):
    answer = kernel.run(source['tokens'], full_probs=True, positions=TIMES)
    # Array index t-1 means all observed tokens X_1,...,X_t are scored.
    li, lc = np.cumsum(answer['lF']), np.cumsum(answer['lC'])
    qi, qc = answer['pF'], answer['pC']
    error = max(float(np.max(abs(qi.sum(axis=1)-1))),float(np.max(abs(qc.sum(axis=1)-1))))
    require(error<2e-6, f'Normalization error {error}')
    require(np.isfinite(qi).all() and np.isfinite(qc).all() and (qi>0).all() and (qc>0).all(),
            'Nonpositive or nonfinite component probabilities')
    actual = source['tokens'][TIMES]
    observed_error = max(float(np.max(abs(-np.log2(qi[np.arange(len(TIMES)),actual])-answer['lF'][TIMES]))),
                         float(np.max(abs(-np.log2(qc[np.arange(len(TIMES)),actual])-answer['lC'][TIMES]))))
    require(observed_error<2e-6,'Next-token probabilities disagree with component code increments')
    gap = li[:-1]-lc[:-1]
    visits=np.zeros(D,dtype=int);row_count=[]
    for token in source['tokens'][:-1]:
        row_count.append(visits[token]);visits[token]+=1
    row_count=np.asarray(row_count,dtype=int)
    return dict(times=TIMES, q_iid=qi, q_copy=qc,
        log_evidence_iid=-li[:-1]*np.log(2), log_evidence_copy=-lc[:-1]*np.log(2),
        evidence_gap_bits=gap, component_tv=.5*np.abs(qi-qc).sum(axis=1),
        current_row_counts_all=row_count,current_row_count=row_count[TIMES-1],
        tv_copy_uniform=.5*np.abs(qc-1/D).sum(axis=1),
        loss_iid=answer['lF'],loss_copy=answer['lC'],
        normalization_error=np.array(error), observed_probability_error=np.array(observed_error))


def raw_path(out, depth_max, branch, index):
    return Path(out)/f'depths1_{depth_max}'/branch/f'seq_{index:06d}.npz'


def run_one(out, plan_id, depth_max, branch, index):
    target = raw_path(out,depth_max,branch,index)
    start = time.monotonic()
    if target.exists():
        with np.load(target,allow_pickle=False) as saved:
            require(str(saved['plan_id'])==plan_id,'Cached record plan differs')
        return dict(depth_max=depth_max,branch=branch,index=index,cached=True,seconds=0.)
    source = source_record(depth_max,branch,index)
    result = predictions(KERNELS[depth_max],source)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,**source,**result,plan_id=plan_id,depth_max=depth_max,branch=branch,index=index)
    target.with_suffix('.tmp').replace(target)
    return dict(depth_max=depth_max,branch=branch,index=index,cached=False,seconds=time.monotonic()-start)


def analytic_depth_one(tokens):
    """Independent conjugate Dirichlet(1,...,1) validation of both codes."""
    unigram = np.zeros(D,dtype=int)
    rows = np.zeros((D,D),dtype=int)
    losses_i, losses_c, qi, qc = [], [], [], []
    for t, b in enumerate(tokens):
        p_i=(unigram+1)/(t+D)
        p_c=np.full(D,1/D) if t==0 else (rows[tokens[t-1]]+1)/(rows[tokens[t-1]].sum()+D)
        if t in TIMES:
            qi.append(p_i.copy());qc.append(p_c.copy())
        losses_i.append(-np.log2(p_i[b]));losses_c.append(-np.log2(p_c[b]))
        unigram[b]+=1
        if t:rows[tokens[t-1],b]+=1
    # Full count formulas independently check accumulated sequential losses.
    log_i=gammaln(D)-gammaln(D+len(tokens))+gammaln(unigram+1).sum()
    log_c=-np.log(D)+sum(gammaln(D)-gammaln(D+row.sum())+gammaln(row+1).sum() for row in rows)
    require(abs(log_i/np.log(2)+sum(losses_i))<2e-8,'Analytic iid count code failed')
    require(abs(log_c/np.log(2)+sum(losses_c))<2e-8,'Analytic copier count code failed')
    return np.array(losses_i),np.array(losses_c),np.array(qi),np.array(qc)


def direct_count_codes(tokens,depth_max,tables):
    """Independent count-profile integration, without sequential integrands."""
    cache={}
    def row_code(counts):
        unique,multiplicity=np.unique(counts,return_counts=True)
        key=tuple(zip(unique.tolist(),multiplicity.tolist()))
        if key in cache:return cache[key]
        k=int(counts.sum())
        if k==0:return 0.
        raw=[]
        for depth in range(1,depth_max+1):
            log_integrand=k*tables.u+(multiplicity[:,None]*tables.rows(depth,unique)).sum(axis=0)
            raw.append(logsumexp(log_integrand)+np.log(tables.h)-gammaln(k))
        log_evidence=logsumexp(raw)-np.log(depth_max)
        cache[key]=-log_evidence/np.log(2)
        return cache[key]
    counts=np.bincount(tokens,minlength=D)
    rows=np.zeros((D,D),dtype=int)
    np.add.at(rows,(tokens[:-1],tokens[1:]),1)
    return row_code(counts),np.log2(D)+sum(row_code(row) for row in rows)


def benchmark(out):
    protocol=plan()
    if (out/'plan.json').exists():require(read(out/'plan.json')==protocol,'Existing plan differs')
    else:write(out/'plan.json',protocol)
    start=time.monotonic();initialize();setup_seconds=time.monotonic()-start
    checks=[]
    for depth_max in (4,8):
        for branch in ('iid','copy'):
            checks.append(run_one(out,protocol['plan_id'],depth_max,branch,0))
    measured=sum(c['seconds'] for c in checks)
    report=dict(schema=SCHEMA+'-benchmark',plan_id=protocol['plan_id'],passed=True,
        setup_seconds=setup_seconds,sequence_timings=checks,measured_seconds=measured,
        projected_serial_hours=measured/len(checks)*1024/3600,
        projected_four_worker_hours=measured/len(checks)*1024/3600/4,
        runtime_scope='steady sequence evaluation; startup, validation and I/O add overhead',utc=utc())
    write(out/'benchmark.json',report);print(json.dumps(report),flush=True)


def run(out,workers):
    protocol=read(out/'plan.json')
    require(protocol==plan(),'Plan/code changed since benchmark')
    bench=read(out/'benchmark.json')
    require(bench['passed'] and bench['plan_id']==protocol['plan_id'],'Matching benchmark required')
    require(1<=workers<=6,'Use one to six single-thread workers')
    initialize();start=time.monotonic();completed=0
    tasks=[(depth_max,branch,index) for depth_max in (4,8) for branch in ('iid','copy') for index in range(N_PER_BRANCH)]
    # The tables are immutable; fork shares this ~250 MB read-only kernel.
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        pending=[pool.submit(run_one,out,protocol['plan_id'],*task) for task in tasks]
        for future in as_completed(pending):
            future.result();completed+=1
            if completed%32==0 or completed==len(tasks):
                status=dict(stage='references',completed=completed,total=len(tasks),elapsed_seconds=time.monotonic()-start,
                            complete=completed==len(tasks),utc=utc())
                write(out/'status.json',status);print(json.dumps(status),flush=True)


def validate(out):
    protocol=read(out/'plan.json');require(protocol==plan(),'Plan/code changed')
    initialize()
    # Rebuild the very same tables over a wider integration interval.
    wide=SeqLSA(D,depths=range(1,9),rmax=N_TOKENS,u_lo=-45.,u_hi=40.)
    exact=SeqLSA(D,depths=[1],rmax=N_TOKENS,u_lo=-30.,u_hi=30.)
    direct_tables=open_tables(8,u_lo=-30.,u_hi=30.)
    singleton=[]
    for depth in range(1,9):
        phi=direct_tables.rows(depth,[0,1])
        logq=logsumexp(direct_tables.u+(D-1)*phi[0]+phi[1])+np.log(direct_tables.h)
        singleton.append(float(abs(logq+np.log(D))))
    require(max(singleton)<2e-10,'Per-depth singleton evidence is not uniform')
    rows=[]
    for depth_max in (4,8):
        kernel=copy.copy(wide);kernel.depths=list(range(1,depth_max+1))
        kernel.LN0=wide.LN0[:depth_max];kernel.D=wide.D[:,:depth_max]
        for branch in ('iid','copy'):
            for index in (0,N_PER_BRANCH-1):
                source=source_record(depth_max,branch,index)
                with np.load(raw_path(out,depth_max,branch,index),allow_pickle=False) as saved:
                    require(np.array_equal(source['tokens'],saved['tokens']),'Source replay changed')
                    result=predictions(kernel,source)
                    gap_error=float(np.max(abs(result['evidence_gap_bits']-saved['evidence_gap_bits'])))
                    p_error=max(float(np.max(abs(result[k]-saved[k]))) for k in ('q_iid','q_copy'))
                    require(gap_error<2e-5 and p_error<2e-6,'Expanded integration range changed probabilities or evidence threshold')
                    for prior in PRIORS:
                        threshold=np.log2((1-prior)/prior)
                        require(np.array_equal(result['evidence_gap_bits']>threshold,saved['evidence_gap_bits']>threshold),
                                'Quadrature changes posterior crossing')
                a=predictions(exact,source);li,lc,qi,qc=analytic_depth_one(source['tokens'])
                exact_p=max(float(np.max(abs(a['q_iid']-qi))),float(np.max(abs(a['q_copy']-qc))))
                exact_loss=max(float(np.max(abs(a['loss_iid']-li))),float(np.max(abs(a['loss_copy']-lc))))
                require(exact_p<2e-8 and exact_loss<2e-7,'Depth-one closed-form control failed')
                # Prefix predictions are unchanged after future tokens change.
                altered=dict(source,tokens=source['tokens'].copy());altered['tokens'][65:]=(altered['tokens'][65:]+1)%D
                changed=predictions(KERNELS[depth_max],altered)
                original=predictions(KERNELS[depth_max],source)
                causal=max(float(np.max(abs(changed[k][TIMES<=64]-original[k][TIMES<=64]))) for k in ('q_iid','q_copy'))
                require(causal==0.,'Future token changes affect earlier predictions')
                direct_i,direct_c=direct_count_codes(source['tokens'][:1024],depth_max,direct_tables)
                direct_error=max(abs(direct_i+original['log_evidence_iid'][-1]/np.log(2)),
                                 abs(direct_c+original['log_evidence_copy'][-1]/np.log(2)))
                require(direct_error<2e-7,'Sequential evidence differs from count-profile integration')
                rows.append(dict(depth_max=depth_max,branch=branch,index=index,wide_gap_error_bits=gap_error,
                    wide_probability_error=p_error,depth_one_probability_error=exact_p,depth_one_loss_error_bits=exact_loss,
                    causality_error=causal,direct_count_code_error_bits=direct_error,posterior_crossings_unchanged=True))
    report=dict(schema=SCHEMA+'-validation',plan_id=protocol['plan_id'],passed=True,checks=rows,
                per_depth_singleton_log_evidence_errors=singleton,utc=utc())
    write(out/'validation.json',report);print(json.dumps(report),flush=True)


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=values.mean(axis=0).tolist(),se=(values.std(axis=0,ddof=1)/np.sqrt(len(values))).tolist(),
        q10=np.quantile(values,.1,axis=0).tolist(),median=np.median(values,axis=0).tolist(),
        q90=np.quantile(values,.9,axis=0).tolist())


def crossing_records(mask):
    mask=np.asarray(mask,dtype=bool)
    first=np.where(mask.any(axis=1),TIMES[mask.argmax(axis=1)],-1)
    recross=((mask[:,:-1])&(~mask[:,1:])).sum(axis=1)
    reached=first[first>=0]
    return dict(first_tested_times=first.tolist(),recrossings_per_sequence=recross.tolist(),
        first_crossing_counts={str(t):int((first==t).sum()) for t in TIMES},
        no_crossing_count=int((first<0).sum()),no_crossing_fraction=float((first<0).mean()),
        any_recrossing_count=int((recross>0).sum()),
        median_first_crossing_among_reached=float(np.median(reached)) if len(reached) else None,
        grid_only=True,unreached_value=-1)


def summarize(out,results):
    protocol=read(out/'plan.json');require(protocol==plan(),'Plan/code changed')
    validation=read(out/'validation.json');require(validation['passed'],'Numerical validation missing')
    groups=[];raw_hashes={};csv_rows=[]
    import csv
    for depth_max in (4,8):
        for branch in ('iid','copy'):
            source=[]
            for index in range(N_PER_BRANCH):
                path=raw_path(out,depth_max,branch,index)
                raw_hashes[str(path.relative_to(ROOT))]=sha(path)
                with np.load(path,allow_pickle=False) as r:
                    require(str(r['plan_id'])==protocol['plan_id'],'Record plan differs')
                    require(np.array_equal(r['times'],TIMES),'Record time grid differs')
                    source.append({k:r[k].copy() for k in r.files})
            gap=np.stack([r['evidence_gap_bits'][TIMES-1] for r in source])
            tv=np.stack([r['component_tv'] for r in source])
            row_counts=np.stack([r['current_row_count'] for r in source])
            tv_uniform=np.stack([r['tv_copy_uniform'] for r in source])
            for prior in PRIORS:
                weights=expit(gap*np.log(2)+np.log(prior/(1-prior)))
                raw=weights>.5;dist=tv>=.05;dist_copy=raw&dist
                preference=(2*weights-1)*tv
                reconstructed=np.stack([((1-w[:,None])*r['q_iid']+w[:,None]*r['q_copy']) for w,r in zip(weights,source)])
                qi=np.stack([r['q_iid'] for r in source]);qc=np.stack([r['q_copy'] for r in source])
                sign_check=.5*(abs(reconstructed-qi).sum(axis=2)-abs(reconstructed-qc).sum(axis=2))
                error=float(np.max(abs(sign_check-preference)))
                require(error<1e-12,'TV mixture identity failed')
                group=dict(depth_max=depth_max,prior_copy=prior,branch=branch,n_sequences=N_PER_BRANCH,
                    times=TIMES.tolist(),weight=stats(weights),component_tv=stats(tv),evidence_gap_bits=stats(gap),
                    preference=stats(preference),posterior_copy_fraction=raw.mean(axis=0).tolist(),
                    distinguishable_copy_fraction=dist_copy.mean(axis=0).tolist(),
                    distinguishable_copy_seen_row_fraction=(dist_copy&(row_counts>0)).mean(axis=0).tolist(),
                    distinguishable_nonuniform_copy_fraction=(dist_copy&(tv_uniform>=.05)).mean(axis=0).tolist(),
                    current_row_seen_fraction=(row_counts>0).mean(axis=0).tolist(),
                    tv_copy_uniform=stats(tv_uniform),
                    indistinguishable_fraction=(~dist).mean(axis=0).tolist(),
                    distinguishable_iid_fraction=((~raw)&dist).mean(axis=0).tolist(),
                    posterior_crossing=crossing_records(raw),distinguishable_crossing=crossing_records(dist_copy),
                    mixture_tv_identity_max_error=error)
                groups.append(group)
                for index in range(N_PER_BRANCH):
                    for j,t in enumerate(TIMES):
                        csv_rows.append(dict(depth_max=depth_max,prior_copy=prior,branch=branch,index=index,time=int(t),
                            evidence_gap_bits=float(gap[index,j]),weight_copy=float(weights[index,j]),
                            component_tv=float(tv[index,j]),preference=float(preference[index,j]),
                            log_evidence_iid=float(source[index]['log_evidence_iid'][t-1]),
                            log_evidence_copy=float(source[index]['log_evidence_copy'][t-1]),
                            current_row_count=int(row_counts[index,j]),tv_copy_uniform=float(tv_uniform[index,j]),
                            posterior_copy=bool(raw[index,j]),distinguishable=bool(dist[index,j]),
                            distinguishable_copy=bool(dist_copy[index,j])))
    # The identity E[w_C]=p_C is a useful branch-balanced Monte Carlo check,
    # not a test requiring each noisy empirical mean to equal p_C exactly.
    calibration=[]
    for depth_max in (4,8):
        for prior in PRIORS:
            match={g['branch']:g for g in groups if g['depth_max']==depth_max and g['prior_copy']==prior}
            mean=(1-prior)*np.array(match['iid']['weight']['mean'])+prior*np.array(match['copy']['weight']['mean'])
            se=np.sqrt((1-prior)**2*np.array(match['iid']['weight']['se'])**2+prior**2*np.array(match['copy']['weight']['se'])**2)
            calibration.append(dict(depth_max=depth_max,prior_copy=prior,mean=mean.tolist(),se=se.tolist(),
                                    deviation_from_prior=(mean-prior).tolist()))
    summary=dict(schema=SCHEMA+'-summary',plan=protocol,n_sequences=1024,n_records=len(csv_rows),
        validation=validation,groups=groups,prior_calibration=calibration,raw_sha256=raw_hashes,
        full_probabilities='Raw NPZ records contain q_iid and q_copy at all tested times.',
        scope='Development feasibility only; no transformer trained or evaluated; first crossings are grid crossings.',utc=utc())
    results.mkdir(parents=True,exist_ok=True)
    write(results/'summary.json',summary);write(results/'validation.json',validation)
    with (results/'records.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write(results/'provenance.json',dict(plan_id=protocol['plan_id'],input_sha256=protocol['input_sha256'],
        raw_sha256=raw_hashes,summary_sha256=sha(results/'summary.json'),records_sha256=sha(results/'records.csv')))
    print(json.dumps(dict(stage='summary',complete=True,groups=len(groups),sequences=1024,records=len(csv_rows))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['benchmark','run','validate','summary'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--results',type=Path,default=DEFAULT_RESULTS)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='benchmark':benchmark(args.out)
    elif args.stage=='run':run(args.out,args.workers)
    elif args.stage=='validate':validate(args.out)
    else:summarize(args.out,args.results)


if __name__=='__main__':main()
