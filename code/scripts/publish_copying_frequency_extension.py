"""Independent readback of the completed through-eight surrogate calibration."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/copying_frequency_extension_2026-09-19'
SCORES=ROOT/'output/shared_base_shuffle_download_2026-09-18'
DEST=ROOT/'results/transformer/copying_frequency_extension_2026-09-19'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def kind(tokens):
    tail=Counter(map(int,tokens[1:]));first=tail.pop(int(tokens[0]),0)
    return json.dumps([first,sorted(tail.values())],separators=(',',':'))


def main():
    verified={};reports=[];maximum_replay=0.
    for stage in ('stratified','refined'):
        folder=BASE/stage;plan=json.loads((folder/'plan.json').read_text());report=json.loads((folder/'summary.json').read_text())
        require(report['complete'] and report['plan_sha256']==sha(folder/'plan.json'),'Incomplete or changed plan')
        for name,digest in {**plan['code_sha256'],**plan['inputs_sha256'],**plan['original_scoring_inputs_sha256']}.items():
            require(sha(ROOT/name)==digest,'Frozen input changed: '+name)
        require(not report['scientific_timing_validated'] and not report['finite_shuffle_exact'],'Overstated scientific scope')
        moments={k:np.asarray(v) for k,v in json.loads((folder/'moments.json').read_text()).items()}
        for group in report['groups']:
            name=group['id'];rawpath=folder/(name+'.npz')
            with np.load(rawpath,allow_pickle=False) as z:loss=z['stationary_loss'];loo=z['stationary_leave_one_bank_out_loss']
            with np.load(SCORES/'cohorts'/(name+'.npz'),allow_pickle=False) as z:tokens=z['tokens'];transitions=z['transition']
            require(loss.shape==(256,8) and loo.shape==(len(next(iter(moments.values()))),256,8),'Wrong population coverage')
            require(np.isfinite(loss).all() and np.isfinite(loo).all(),'Nonfinite saved losses')
            for i in range(256):
                for n in range(1,9):
                    prefix=list(map(int,tokens[i,:n]));den=moments[kind(prefix)]
                    num=np.stack([moments[kind(prefix+[y])] for y in range(64)],axis=1)
                    mean=num.mean(0)/den.mean();leave=(num.sum(0)-num)/(den.sum()-den[:,None])
                    require(np.isfinite(mean).all() and np.isfinite(leave).all() and np.min(mean)>0 and np.min(leave)>0,'Invalid raw prediction')
                    require(abs(mean.sum()-1)<1e-10 and np.max(abs(leave.sum(1)-1))<1e-10,'Raw conditional normalization failed')
                    oracle=transitions[i,tokens[i,n-1]]
                    errors=[abs(loss[i,n-1]+oracle@np.log2(mean)),np.max(abs(loo[:,i,n-1]+np.log2(leave)@oracle))]
                    maximum_replay=max(maximum_replay,*map(float,errors))
            for seed,comparison in group['comparisons'].items():
                with np.load(SCORES/'eval'/('shuffled_seed'+seed)/(name+'.npz'),allow_pickle=False) as z:model=z['expected_loss_bits'][:,:8]
                require(model.shape==(256,8) and np.isfinite(model).all(),'Invalid paired model scores')
                delta=loss-model;bank=(loo-model).mean(1);b=len(bank)
                checks={'mean':delta.mean(0),'paired_sequence_se':delta.std(0,ddof=1)/16,
                    'prior_bank_mcse':np.sqrt((b-1)/b*np.sum((bank-bank.mean(0))**2,axis=0))}
                for field,value in checks.items():maximum_replay=max(maximum_replay,float(np.max(abs(value-comparison[field]))))
            verified[str(rawpath.relative_to(ROOT))]=sha(rawpath)
        require(maximum_replay<1e-11,'Independent loss/statistic replay differs')
        for name in ('plan.json','summary.json','moments.json'):
            path=folder/name;verified[str(path.relative_to(ROOT))]=sha(path)
        reports.append(report)
    coarse,refined=reports
    gap=max(abs(v) for g in refined['groups'] for r in g['comparisons'].values() for v in r['mean'])
    mc=max(v for g in refined['groups'] for r in g['comparisons'].values() for v in r['prior_bank_mcse'])
    shift=max(abs(x-y) for a,b in zip(coarse['groups'],refined['groups']) for seed in ('0','1')
              for x,y in zip(a['comparisons'][seed]['mean'],b['comparisons'][seed]['mean']))
    DEST.mkdir(parents=True,exist_ok=True)
    result=dict(refined,maximum_absolute_mean_gap_bits=gap,maximum_prior_mcse_bits=mc,
                maximum_coarse_fine_shift_bits=shift)
    (DEST/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    validation=dict(passed=True,input_sha256=verified,script_sha256=sha(__file__),
        summary_sha256=sha(DEST/'summary.json'),maximum_readback_error=maximum_replay,
        scientific_timing_validated=False,independent_confirmation=False)
    (DEST/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(dict(passed=True,maximum_mean_gap=gap,maximum_prior_mcse=mc,
                         maximum_coarse_fine_shift=shift,maximum_readback_error=maximum_replay)))


if __name__=='__main__':main()
