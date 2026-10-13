"""Recompute descriptive checks for the stored Appendix C experiments.

The chain predictions are random across tables. Legacy files do not retain
prediction variance or its covariance with measured redundancy; their reported
standardized discrepancies use only the measured curve's SE and are not a
paired test of the prediction error. New runs retain the paired differences.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from context_crossing_figure import last_zero

FOLDER=Path('results/mechanism')
FILES={'context_row_check.json':32,'context_chain_check.json':8,'context_crossing_check.json':8,'one_step_info.json':8}


def rescale():
    changed=[]
    for name,n in FILES.items():
        p=FOLDER/name;d=json.loads(p.read_text())
        if d.get('se_ddof')==1:continue
        if d.get('se_ddof',0)!=0:raise ValueError('unknown SE convention')
        if d.get('seeds',d.get('reps',n))!=n:raise ValueError('unexpected replication count')
        factor=math.sqrt(n/(n-1))
        for row in d['alphas']:
            for k,v in row.items():
                if k.endswith('_se'):row[k]=(np.asarray(v)*factor).tolist()
            if name=='one_step_info.json':
                row['I_sd']*=factor
                row['I_se']=row['I_sd']/math.sqrt(n)
        d['se_ddof']=1;d['reps']=n
        d['se_adjustment']={'from_ddof':0,'to_ddof':1,'factor':factor,'method':'Exact rescaling of legacy sample SE; means unchanged.'}
        if name=='one_step_info.json':
            d['sd_ddof']=1
            d['convention']='stationary gain of sampled transition tables; I_sd is sample SD and I_se is standard error of mean; no sampled trajectories'
        elif 'row_' in name:
            d['L']=1;d['sample_count_unit']='iid row samples'
            d['code_convention']='Bayes sequence mixture for one tilted iid row, evaluated by interpolated log-scale quadrature.'
            d['seed']=23
        else:
            d['sample_count_unit']='transitions'
            d['initialization']='sampled from numerically solved stationary law of each transition table'
            d['code_convention']='transition-only unconditioned row-product reference; initial token omitted, not the full stationary-start Bayes mixture'
            d['legacy_uncertainty_limitation']='prediction variance and covariance with measured redundancy not retained; standardized gaps are expressed in measured-curve SE only'
        p.write_text(json.dumps(d,indent=2)+'\n');changed.append(name)
    return changed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--rescale-legacy',action='store_true');args=ap.parse_args()
    if args.rescale_legacy:print('converted',rescale())
    result={'se_ddof':1,'units':'bits; sample sizes count iid draws or transitions as marked','row':[],'chain':[],'crossing':[]}
    for kind,name in [('row','context_row_check.json'),('chain','context_chain_check.json')]:
        data=json.loads((FOLDER/name).read_text());assert data['se_ddof']==1
        for row in data['alphas']:
            y=np.asarray(row['rho_mean' if kind=='row' else 'red_mean'])
            se=np.asarray(row['rho_se' if kind=='row' else 'red_se'])
            pred=np.asarray(row['pred_sum_r1' if kind=='row' else 'pred_pi_mean'])
            z=(y-pred)/se
            r=dict(alpha=row['alpha'],d=data['d'],reps=data['reps'],points=len(y),
                maximum_abs_measured_SE_units=float(max(abs(z))),points_beyond_two_measured_SE=int(sum(abs(z)>2)),
                terminal_percent_difference=float(100*(y[-1]/pred[-1]-1)),terminal_measured_bits=float(y[-1]),
                terminal_comparison_bits=float(pred[-1]),terminal_measured_SE_bits=float(se[-1]))
            if kind=='chain':
                pq=np.asarray(row['pred_q']);r['max_percent_prediction_change_pi_to_q']=float(max(abs(pq/pred-1))*100)
                r['terminal_percent_prediction_change_pi_to_q']=float(100*(pq[-1]/pred[-1]-1))
            result[kind].append(r)
    data=json.loads((FOLDER/'context_crossing_check.json').read_text());assert data['se_ddof']==1
    n=np.asarray(data['n'])
    for row in data['alphas']:
        diff=np.asarray(row['diff_mean']);balance=np.asarray(row['G_mean'])-np.asarray(row['P_mean'])
        actual=last_zero(n,diff);reconstructed=last_zero(n,balance)
        # Algebraic consistency of all seven stored mean curves.
        identity=float(max(abs(diff-(np.asarray(row['RF_dep_mean'])-np.asarray(row['RC_dep_mean'])))))
        assert identity<1e-8
        result['crossing'].append(dict(alpha=row['alpha'],d=data['d'],reps=data['reps'],
            last_interpolated_mean_crossing=actual,reconstruction_using_measured_P=reconstructed,
            relative_crossing_difference_percent=100*(reconstructed/actual-1),
            count_unit='transitions; initial token omitted',mean_identity_error_bits=identity,
            crossing_convention='last observed sign change in mean curves, interpolated linearly in log transition count; finite horizon',
            interpretation='The reconstruction uses the measured premium and is not an independent forecast.'))
    result['chain_uncertainty_limitation']='Legacy standardized differences use measured-curve SE only, not the SE of the paired measured-minus-predicted difference.'
    information=json.loads((FOLDER/'one_step_info.json').read_text())
    result['one_step_information']={'d':information['d'],'L':information['L'],'reps':information['reps'],'alphas':information['alphas']}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
