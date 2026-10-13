"""Independently replay compact early frequency-calibration statistics."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/copying_frequency_calibration_2026-09-19'
DEST=ROOT/'results/transformer/copying_frequency_calibration_2026-09-19'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(ok,message):
    if not ok:raise ValueError(message)


def main():
    reports={};verified={};maximum_readback_error=0.;first_errors=[]
    for stage in ('stratified','refined'):
        folder=BASE/stage;p=folder/'summary.json';s=json.loads(p.read_text())
        plan=json.loads((folder/'plan.json').read_text())
        check(s['complete'] and s['times']==[1,2,3,4] and not s['scientific_timing_validated'], 'Unexpected result scope')
        check(s['plan_sha256']==sha(folder/'plan.json'),'Plan binding changed')
        for name,digest in {**plan['code_sha256'],**s['scoring_inputs_sha256']}.items():
            check(sha(ROOT/name)==digest,'Changed input: '+name)
        check(sha(folder/'bank.npz')==s['reference']['bank_sha256'],'Bank changed')
        with np.load(folder/'bank.npz',allow_pickle=False) as z:
            q=z['q'];nu=z['nu'];banks=z['joint_banks']
        n_banks,order,n_draws=plan['settings'];nodes,weights=np.polynomial.legendre.leggauss(order)
        cursor=0;means=[]
        for _ in range(n_banks):
            total=0.
            for w in weights:
                for depth in range(9):
                    n=1 if depth==0 else n_draws
                    total+=w/18*np.mean(np.sum(q[cursor:cursor+n]*nu[cursor:cursor+n],axis=1));cursor+=n
            means.append(total)
        check(cursor==len(q),'Source quadrature readback lost cases')
        error=abs(float(np.mean(means))-s['first_repeat_probability']);first_errors.append(error)
        check(error<1e-12,'Independent first prediction differs')
        for group in s['groups']:
            raw_path=folder/(group['id']+'.npz');verified[str(raw_path.relative_to(ROOT))]=sha(raw_path)
            with np.load(raw_path,allow_pickle=False) as z:
                raw={k:z[k] for k in z.files}
            check(raw['stationary'].shape==(256,4),'Incomplete group')
            for name,value in group['comparisons'].items():
                if name.startswith('neural_'):
                    seed=int(name[-1]);values=raw[f'shuffled_seed{seed}']-raw[f'ordered_seed{seed}']
                    loo=None
                else:
                    left,right=name.split('_minus_',1);right=right.removesuffix('_cumulative')
                    values=raw[left]-raw[right]
                    loo=raw['stationary_leave_one_bank_out']-raw[right] if left=='stationary' else None
                if 'cumulative' in name:
                    values=np.cumsum(values,axis=1)
                    if loo is not None:loo=np.cumsum(loo,axis=2)
                for expected,actual in ((value['mean'],values.mean(0)),
                        (value['paired_sequence_se'],values.std(0,ddof=1)/16)):
                    error=float(np.max(abs(np.asarray(expected)-actual)))
                    maximum_readback_error=max(error,maximum_readback_error)
                    check(error<1e-12,'Mean/paired standard error readback differs')
                if loo is not None:
                    x=loo.mean(1);mean=x.mean(0)
                    mc=np.sqrt((len(x)-1)/len(x)*np.sum((x-mean)**2,axis=0))
                    error=float(np.max(abs(mc-value['prior_bank_mcse'])))
                    maximum_readback_error=max(error,maximum_readback_error)
                    check(error<1e-12,'Prior uncertainty readback differs')
        reports[stage]=s;verified[str(p.relative_to(ROOT))]=sha(p)
    final=reports['refined'];coarse=reports['stratified']
    max_shift=max(abs(a-b) for c,f in zip(coarse['groups'],final['groups'])
        for a,b in zip(c['comparisons']['stationary_minus_bayes']['mean'],f['comparisons']['stationary_minus_bayes']['mean']))
    max_mc=max(v for g in final['groups'] for v in g['comparisons']['stationary_minus_bayes']['prior_bank_mcse'])
    max_gap=max(abs(v) for g in final['groups'] for seed in (0,1)
        for v in g['comparisons'][f'stationary_minus_shuffled_seed{seed}']['mean'])
    summary=dict(schema='copying-frequency-calibration-publication-v1',times=[1,2,3,4],
        first_repeat_probability=final['first_repeat_probability'],groups=final['groups'],
        maximum_absolute_surrogate_minus_shuffled_mean_bits=max_gap,
        maximum_sampling_mcse_of_mean_bits=max_mc,maximum_coarse_fine_change_bits=max_shift,
        prior=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'),
        numerical_method=final['reference'],coarse_numerical_method=coarse['reference'],
        scope=final['interpretation'],limitations=final['uncertainty'],
        scientific_timing_validated=False,independent_confirmation=False)
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    receipt=dict(schema='copying-frequency-calibration-readback-v1',passed=True,
        summary_sha256=sha(DEST/'summary.json'),script_sha256=sha(__file__),input_sha256=verified,
        maximum_statistic_error=maximum_readback_error,maximum_first_probability_error=max(first_errors),
        scientific_timing_validated=False,raw_arrays_committed=False)
    (DEST/'validation.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(readback_passed=True,max_shuffled_gap_bits=max_gap,max_mcse_bits=max_mc,
        coarse_fine_change_bits=max_shift,scientific_timing_validated=False)))


if __name__=='__main__':main()
