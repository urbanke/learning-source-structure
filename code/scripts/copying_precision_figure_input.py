"""Export and replay the approved complete-cohort copying precision release.

The refined t1--8 estimates and256-bank comparator supersede their earlier
estimates; source histories and neural scores must still match exactly.
This adapter performs no inference and makes no payoff-crossing claim.
"""
from pathlib import Path
import argparse,gzip,hashlib,json
import numpy as np

CONDITIONS=('a1_L0','a1_L2','a1_L3')
ROOT=Path(__file__).resolve().parents[2]
SCHEMA='copying-precision-figure-input-v1'

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def check_release(summary,plan,selection):
    require(summary['schema']=='copying-precision-independent-release-v1' and all(summary[k] is True for k in
        ('complete','population_reference_approved','aggregate_precision_passed','independent_frequency_replay_passed'))
        and summary['publication_status']=='approved','Copying full-cohort release is not approved')
    require(summary['reference_count']==11224 and summary['original_histories']==768 and summary['times']==list(range(1,17))
        and not any(summary[k] for k in ('missing','failed_audits','failed_paired_orders')),'Copying cohort is incomplete')
    require(summary['payoff_forecast_approved'] is False and 'surrogate' in summary['comparator'].lower()
        and 'not exact finite-shuffle' in summary['comparator'].lower(),'Copying comparator or approved scope changed')
    require(plan['plan_id']==summary['plan_id']==digest({k:v for k,v in plan.items() if k!='plan_id'})
        and len(plan['expected_reference_ids'])==len(set(plan['expected_reference_ids']))==11224
        and plan['original_histories']==768 and plan['times']==list(range(1,17))
        and plan['numerical_allowance_bits']==.01/3 and plan['source_se_fraction']==.25,'Copying release plan differs')
    expected=selection['expected_repair_ids'];selected=selection['selected_repair_ids']
    require(selection['schema']=='copying-evening-release-selection-v1' and selection['complete'] is True
        and selection['sampling_performed'] is False and len(expected)==len(set(expected))==len(selected)==len(set(selected))
        and set(expected)==set(selected) and len(expected)==summary['timeout_replacement_reference_count']+summary['evening_replacement_reference_count']
        and set(expected)<=set(plan['expected_reference_ids']),'Copying repair selection did not validate exactly once')
    require(len(summary['groups'])==3 and {g['condition'] for g in summary['groups']}==set(CONDITIONS),'Copying source conditions differ')

def replay(summary,arrays):
    result={};checks={}
    require(np.array_equal(arrays['times'],np.arange(1,17)),'Copying coordinate grid changed')
    for group in summary['groups']:
        condition=group['condition'];depth=int(condition.split('L')[-1]);score={}
        for name in ('stationary_loss','bayes_loss_a16','bayes_loss_a32','ordered_seed0','shuffled_seed0','ordered_seed1','shuffled_seed1'):
            values=np.asarray(arrays[condition+'_'+name]);require(values.shape==(256,16) and np.isfinite(values).all(),'Copying must retain256 complete paired histories')
            score[name]=values
        gains={'bayes':score['stationary_loss']-score['bayes_loss_a32'],
               'neural_seed0':score['shuffled_seed0']-score['ordered_seed0']}
        result[depth]={}
        for metric in ('instantaneous','cumulative'):
            gate=group[metric];require(gate['passed'] is True,'Copying coordinate precision gate failed');record={}
            for name,values in gains.items():
                z=values.cumsum(1) if metric=='cumulative' else values
                record[name+'_mean']=z.mean(0);record[name+'_source_se']=z.std(0,ddof=1)/16
            mean,se=record['bayes_mean'],record['bayes_source_se']
            saved_mean,saved_se,allowance,budget=(np.asarray(gate[k]) for k in ('mean_bits','source_se_bits','numerical_allowance_bits','numerical_budget_bits'))
            require(all(v.shape==(16,) and np.isfinite(v).all() for v in (saved_mean,saved_se,allowance,budget)),'Copying metric summary is incomplete')
            error=max(float(np.max(abs(mean-saved_mean))),float(np.max(abs(se-saved_se))))
            require(error<2e-12 and np.all(allowance>=0) and np.all(allowance<=budget)
                and np.allclose(budget,np.minimum(.01/3,.25*se),rtol=0,atol=2e-14),'Copying mean, uncertainty or precision budget differs')
            result[depth][metric]=record;checks[f'precision_copying_{depth}_{metric}_max_error']=error
    return result,checks

def export(source,neural_source,out):
    source,neural_source,out=(Path(x).resolve() for x in (source,neural_source,out));out.mkdir(parents=True,exist_ok=True)
    summary,plan,selection=(read(source/name) for name in ('summary.json','plan.json','evening_selection.json'))
    check_release(summary,plan,selection)
    require(selection['release_summary']['sha256']==sha(source/'summary.json'),'Selected copying summary changed')
    require(summary['arrays_sha256']==sha(source/'arrays.npz') and summary['input_bindings_sha256']==sha(source/'input_bindings.json'),'Copying release artifacts changed')
    for name,value in plan['code_sha256'].items():require(sha(ROOT/name)==value,'Frozen copying implementation changed: '+name)
    full_bindings=read(source/'input_bindings.json')
    require(len(full_bindings)>=11224*5,'Copying raw-reference provenance is incomplete')
    compact={'times':np.arange(1,17)};neural_bindings={}
    with np.load(source/'arrays.npz',allow_pickle=False) as arrays:
        for condition in CONDITIONS:
            for name in ('stationary_loss','bayes_loss_a16','bayes_loss_a32','ordered_seed0','shuffled_seed0','ordered_seed1','shuffled_seed1'):
                compact[condition+'_'+name]=arrays[condition+'_'+name].copy()
            for role in ('ordered','shuffled'):
                for seed in (0,1):
                    path=neural_source/f'{role}_seed{seed}'/(condition+'.npz')
                    with np.load(path,allow_pickle=False) as raw:
                        require(np.array_equal(raw['times'],np.arange(1,1025)) and raw['expected_loss_bits'].shape==(256,1024)
                            and np.array_equal(compact[f'{condition}_{role}_seed{seed}'],raw['expected_loss_bits'][:,:16]),'Copying release changed original source order or neural scores')
                    neural_bindings[str(path.relative_to(ROOT))]=sha(path)
    _,checks=replay(summary,compact)
    np.savez_compressed(out/'paired_scores.npz',**compact)
    provenance=dict(schema=SCHEMA,passed=True,adapter_sha256=sha(__file__),source_summary=summary,source_plan=plan,source_selection=selection,
        source_artifact_sha256={n:sha(source/n) for n in ('summary.json','plan.json','evening_selection.json','arrays.npz','input_bindings.json')},
        remote_source=str(Path(selection['release_summary']['path']).parent),input_bindings_count=len(full_bindings),paired_scores_sha256=sha(out/'paired_scores.npz'),
        neural_input_sha256=neural_bindings,checks=checks,cohort_retained=True,early_scores_refined=True,
        scope='Instantaneous and cumulative gains at t1--16; source-induced stationary-frequency surrogate; no payoff crossing approval.')
    with gzip.GzipFile(filename=str(out/'provenance.json.gz'),mode='wb',mtime=0) as handle:handle.write((json.dumps(provenance,indent=2)+'\n').encode())
    return load(out)

def load(source):
    source=Path(source);paths=[source/'paired_scores.npz',source/'provenance.json.gz']
    with gzip.open(paths[1],'rt') as handle:p=json.load(handle)
    require(p['schema']==SCHEMA and p['passed'] is True and p['cohort_retained'] is True
        and p['early_scores_refined'] is True and p['paired_scores_sha256']==sha(paths[0]),'Copying compact release changed')
    require(p['adapter_sha256']==sha(__file__),'Copying adapter differs from independent export')
    check_release(p['source_summary'],p['source_plan'],p['source_selection'])
    require(p['source_selection']['release_summary']['sha256']==p['source_artifact_sha256']['summary.json']
        and p['source_summary']['arrays_sha256']==p['source_artifact_sha256']['arrays.npz']
        and p['source_summary']['input_bindings_sha256']==p['source_artifact_sha256']['input_bindings.json'],
        'Copying compact provenance no longer binds the source release')
    with np.load(paths[0],allow_pickle=False) as arrays:result,checks=replay(p['source_summary'],arrays)
    require(checks.keys()==p['checks'].keys() and all(abs(v-p['checks'][k])<2e-12 for k,v in checks.items()),'Copying aggregate replay changed')
    return result,checks,paths

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,required=True);a.add_argument('--neural-source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args()
    _,checks,_=export(v.source,v.neural_source,v.out);print(json.dumps(dict(passed=True,checks=checks),indent=2))
if __name__=='__main__':main()
