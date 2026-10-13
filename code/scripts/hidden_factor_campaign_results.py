"""Independently audit and plot the campaign on the reused 3840-sequence cohort.

Consumes saved arrays only: no training, model inference, or Bayesian inference.
Both runs and all older models stay in the comparison. Final 400k checkpoints
are the default; the historical 200k comparison remains explicitly selectable.
The previously examined cohort is not fresh confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import hidden_factor_confirmation_results as old

ROOT = old.ROOT
PLAN = ROOT/'code/transformer/configs/hidden_factor_campaign_plan_2026-09-18.json'
DOWNLOAD = ROOT/'output/kuma_results_2026-09-20'
SOURCE = DOWNLOAD/'hidden_factor_campaign_comparison_2026-09-19'
DEVELOPMENT = DOWNLOAD/'hidden_factor_campaign_2026-09-18'
OUTPUT = ROOT/'results/transformer/hidden_factor_campaign_2026-09-19'
REQUIRED = {'continue_large': 400000, 'xlarge': 400000}
ARMS = ('baseline', 'continue_large', 'xlarge')
COLORS = {'baseline': '#2468A0', 'continue_large': '#288167', 'xlarge': '#8A4E98', 'large': '#6B727B'}
LABELS = {'baseline': '2 layers, 6.4M sequences', 'large': '4 layers, 6.4M sequences',
          'longsmall': '2 layers, 25.6M sequences'}
TIMES, METRICS = old.TIMES, old.METRICS
read, arrays, sha, fingerprint, require = old.read, old.arrays, old.sha, old.fingerprint, old.require
REMOTE_SCHEMA = 'hidden-factor-campaign-comparison-v1'
SCHEMA = 'hidden-factor-campaign-publication-v1'
CODE_FILES = ('code/transformer/hidden_factor_campaign_comparison_eval.py',
              'code/transformer/hidden_factor_campaign_eval.py',
              'code/transformer/hidden_factor_confirmation_eval.py',
              'code/transformer/eval_mixture.py', 'code/scripts/campaign_job_guard.py')


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve(); actual = sha(path)
        require(expected is None or actual == expected, 'Input checksum differs: '+str(path))
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        self.hashes[name] = actual
        return actual


def selection_metadata(plan, steps):
    """Derive publication labels only from a declared, validated checkpoint selection."""
    require(set(steps) == set(REQUIRED) and steps['continue_large'] == 400000
            and type(steps['xlarge']) is int and steps['xlarge'] in (200000,400000),
            'Unsupported campaign publication checkpoint selection')
    final = {}; labels = dict(LABELS); descriptions = []
    for arm in REQUIRED:
        spec = plan['arms'][arm]; step = steps[arm]
        require(type(step) is int and step in spec['milestones']
                and step <= plan['primary']['final_steps'][arm], 'Unplanned publication milestone')
        final[arm] = step == plan['primary']['final_steps'][arm]
        layers = spec['config']['n_layer']; sequences = step*spec['config']['batch']/1e6
        labels[arm] = f'{layers} layers, {sequences:g}M sequences'+('' if final[arm] else ' (interim)')
        state = 'final' if final[arm] else 'intermediate'
        descriptions.append(f'The {layers}-layer model is {state} at {step:,} updates ({sequences:g} million sequences).')
    return final, labels, ' '.join(descriptions)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and len(values) >= 2 and values.shape[1] == len(TIMES)
            and np.isfinite(values).all(), 'Invalid finite score array')
    n = len(values); mean = np.sum(values, axis=0)/n
    se = np.sqrt(np.sum((values-mean)**2, axis=0)/(n*(n-1)))
    return dict(n_seq=n, mean=mean.tolist(), se=se.tolist())


def check_stats(values, saved, label):
    actual = stats(values)
    require(saved['n_seq'] == actual['n_seq'], label+': sample size changed')
    for key in ('mean', 'se'):
        require(np.shape(saved[key]) == (6,) and np.allclose(saved[key], actual[key], rtol=0, atol=2e-10),
                label+': '+key+' differs from paired raw arrays')


def check_scores(scored, oracle, tokens, label):
    require(np.array_equal(scored['sequence_indices'], np.arange(256,4096))
            and np.array_equal(scored['times'], TIMES), label+': sequence pairing or times differ')
    p = scored['predictive']
    require(p.shape == oracle.shape == (3840,6,16) and np.isfinite(p).all() and (p > 0).all()
            and np.max(abs(p.sum(-1)-1)) < 1e-11, label+': predictive probabilities do not normalize')
    old.check_predictions(p, oracle, tokens, scored, label)
    return {metric: scored[metric] for metric in METRICS}


def validate_training_receipt(plan, receipt, arm, seed):
    cfg = receipt['config']; spec = plan['arms'][arm]
    key = 'scaling_provenance' if arm == 'continue_large' else 'campaign_provenance'
    require({k:v for k,v in cfg.items() if k != key} == dict(spec['config'],seed=seed,source_dimensions=[2,8]),
            'Training configuration or source differs')
    provenance = cfg[key]
    require(provenance['plan_sha256'] == sha(PLAN) and provenance['arm'] == arm and provenance['seed'] == seed
            and provenance['trainer_sha256'] == sha(ROOT/'code/transformer/hidden_factor_campaign_train.py'),
            'Training campaign provenance differs')
    require(receipt['config_hash'] == fingerprint(cfg) and receipt['implementation_hashes'] == plan['implementation_sha256'],
            'Training configuration or implementation hash differs')
    expected_model = dict(vocab=16,n_ctx=1024,n_layer=spec['config']['n_layer'],n_head=8,
                          d_model=spec['config']['d_model'],mlp=True,dropout=0.)
    require(receipt['model'] == expected_model and receipt['n_params'] == spec['n_params'], 'Training architecture differs')
    contract = dict(version='hidden-factor-dirichlet1-v1',dimensions=[2,8],
        rows='independent Dirichlet(1,...,1) for each coordinate',initial_states='independent uniform coordinates',
        encoding='fresh uniform joint-pair-to-token permutation per sequence',
        parameters='fresh independent transitions and encoding per sequence',model_input='encoded tokens only',
        sampling='indexed-batch-v1')
    require(receipt['source_contract'] == contract and receipt['source_contract_hash'] == fingerprint(contract),
            'Training source contract differs')
    require(receipt['precision'] == 'float32' and receipt['device'] == 'cuda'
            and receipt['runtime']['device'] == 'cuda' and receipt['runtime']['float32_matmul_precision'] == 'highest'
            and receipt['runtime']['cuda_matmul_allow_tf32'] is False and receipt['runtime']['num_threads'] == 1,
            'Training arithmetic contract differs')
    if arm == 'continue_large':
        parent = provenance['parent']
        require(parent['weights_checkpoint_sha256'] == plan['parent']['checkpoint_sha256'][str(seed)]
                and parent['step'] == 100000 and parent['completed_sequences'] == 6400000
                and parent['validation_hashes'] == receipt['validation_hashes']
                and parent['runtime'] == receipt['runtime'], 'Continuation ancestry differs')
    else:
        require(provenance['parent'] is None, 'Fresh model has an unexpected parent')


def check_run_envelope(folder, arm, step, plan, baseline, hashes):
    summary_path = folder/'summary.json'; summary = read(summary_path)
    status, selection = read(folder/'status.json'), read(folder/'selection.json')
    summary_hash = hashes.bind(summary_path)
    hashes.bind(folder/'status.json'); hashes.bind(folder/'selection.json')
    expected_code = {name: hashes.bind(ROOT/name) for name in CODE_FILES}
    binding = dict(schema=REMOTE_SCHEMA,arm=arm,step=step,plan_sha256=sha(PLAN),test_role='reused_comparison',
                   implementation_sha256=expected_code)
    require(selection == binding, 'Comparison selection or code identity differs')
    require(status['schema'] == REMOTE_SCHEMA and status['status'] == 'completed' and status['complete'] is True
            and status['arm'] == arm and status['step'] == step and status['evaluated_n_sequences'] == 3840
            and status['fresh_confirmation'] is False and status['summary_sha256'] == summary_hash,
            'Comparison status is incomplete or its summary hash differs')
    require(summary['schema'] == REMOTE_SCHEMA and summary['complete'] is True and summary['scientific_results'] is True
            and summary['arm'] == arm and summary['step'] == step
            and summary['final_checkpoint'] is (step == plan['primary']['final_steps'][arm])
            and summary['test_role'] == 'reused_comparison' and summary['fresh_confirmation'] is False
            and summary['new_training'] is False and summary['new_bayesian_computation'] is False
            and summary['plan_sha256'] == sha(PLAN) and summary['source_dimensions'] == [2,8]
            and summary['sequence_start'] == 256 and summary['sequence_stop'] == 4096
            and summary['evaluated_n_sequences'] == 3840 and summary['times'] == TIMES.tolist()
            and summary['first_token_scored'] is False and summary['implementation_sha256'] == expected_code
            and summary['interpretation'] == plan['interpretation'], 'Comparison scientific scope differs')
    for key in ('cohort_id','cohort_data_sha256','reference_records_sha256'):
        require(summary[key] == baseline[key], 'Original and campaign comparison differ: '+key)
    expected_parents = {f'large_seed{s}': baseline['model_receipts'][f'large_seed{s}'] for s in (0,1)}
    require(summary['parent_receipts'] == expected_parents, 'Original 100k model receipts differ')
    require(set(summary['model_receipts']) == {f'{arm}_seed{s}' for s in (0,1)}, 'A training seed is omitted or substituted')
    return summary


def load_campaign_model(folder, arm, seed, step, summary, baseline, development, plan, hashes, oracle, tokens):
    name = f'{arm}_seed{seed}'
    model_path = folder/'models'/f'{name}.npz'
    receipt_path = model_path.with_suffix('.json'); receipt = read(receipt_path); identity = receipt['identity']
    hashes.bind(receipt_path); hashes.bind(model_path, receipt['data_sha256'])
    require(receipt == summary['model_receipts'][name] and receipt['complete'] is True and receipt['n_sequences'] == 3840,
            name+': model receipt differs or is incomplete')
    config_path = development/f'train/{arm}/seed{seed}/config.json'
    config = read(config_path); hashes.bind(config_path)
    validate_training_receipt(plan, config, arm, seed)
    dev = development/f'eval/{arm}/seed{seed}/step{step}'
    dev_receipt = read(dev/'summary.json'); di = dev_receipt['identity']
    require(dev_receipt['complete'] is True and di['plan_sha256'] == sha(PLAN)
            and di['arm'] == arm and di['seed'] == seed and di['step'] == step
            and di['n_sequences'] == 256 and di['test_role'] == 'existing_development_prefix'
            and di['cohort_id'] == baseline['cohort_id'] and di['times'] == TIMES.tolist()
            and di['evaluator_sha256'] == sha(ROOT/'code/transformer/hidden_factor_campaign_eval.py')
            and di['reference_records_sha256'] == read(old.PLAN)['baseline']['reference_records_sha256'],
            name+': development checkpoint identity differs')
    require(dev_receipt['causality_check']['passed'] is True
            and dev_receipt['causality_check']['max_absolute_logit_difference'] <= 2e-5,
            name+': development causal check failed')
    provenance = dict(checkpoint_sha256=di['checkpoint_sha256'],
                      development_summary_sha256=hashes.bind(dev/'summary.json'),
                      development_scores_sha256=hashes.bind(dev/'scores.npz',dev_receipt['data_sha256']))
    require(len(provenance['checkpoint_sha256']) == 64
            and all(c in '0123456789abcdef' for c in provenance['checkpoint_sha256']), 'Invalid checkpoint hash')
    parent = baseline['model_receipts'][f'large_seed{seed}']
    expected = dict(schema=REMOTE_SCHEMA,test_role='reused_comparison',fresh_confirmation=False,
        arm=arm,seed=seed,step=step,final_checkpoint=step==plan['primary']['final_steps'][arm],
        plan_sha256=sha(PLAN),original_plan_sha256=sha(old.PLAN),cohort_id=baseline['cohort_id'],
        cohort_data_sha256=baseline['cohort_data_sha256'],sequence_start=256,sequence_stop=4096,
        times=TIMES.tolist(),first_token_scored=False,reference_records_sha256=baseline['reference_records_sha256'],
        checkpoint=provenance,parent_score_identity=parent['identity'],parent_scores_sha256=parent['data_sha256'],
        implementation_sha256=summary['implementation_sha256'])
    require(identity == expected, name+': comparison model binding differs')
    require(receipt['causality_check']['passed'] is True
            and receipt['causality_check']['max_absolute_logit_difference'] <= 2e-5,
            name+': comparison causal check failed')
    require(receipt['inference']['device'] == 'cuda'
            and receipt['inference']['precision'] == 'float32; log-softmax float64'
            and receipt['inference']['torch_version'] == config['runtime']['torch_version'],
            name+': inference arithmetic differs from the validated checkpoint runtime')
    scored = arrays(model_path)
    return check_scores(scored,oracle,tokens,name),receipt


def compare_remote_summary(summary, predictors, strata, labels):
    arm = summary['arm']
    names = {f'{arm}_seed{s}' for s in (0,1)} | {'large_seed0','large_seed1'} | set(old.REFERENCE_NAMES)
    pairs = {('bayes_hidden','bayes_encoding_given')}
    for seed in (0,1):
        pairs |= {(f'{arm}_seed{seed}','bayes_hidden'),(f'{arm}_seed{seed}',f'large_seed{seed}'),
                  (f'large_seed{seed}','bayes_hidden')}
    groups = {g['id']: g for g in summary['groups']}
    require(len(summary['groups']) == len(groups) == 5 and set(groups) == {'full_prior',*[f'entropy_{i}' for i in range(4)]},
            'Comparison entropy groups differ')
    count = 0
    for key,g in groups.items():
        mask = np.ones(3840,dtype=bool) if key=='full_prior' else strata==int(key[-1])
        require(g['n_seq'] == int(mask.sum()) and g['status'] == 'complete'
                and g['entropy_label'] == (None if key=='full_prior' else labels[int(key[-1])]), 'Comparison group membership differs')
        require(set(g['predictors']) == names and set(g['paired']) == {a+'_minus_'+b for a,b in pairs},
                'Comparison predictors or paired contrasts were selected')
        for name in names:
            for metric in METRICS:
                check_stats(predictors[name][metric][mask],g['predictors'][name][metric],key+'/'+name+'/'+metric);count+=1
        for left,right in pairs:
            for metric in METRICS:
                check_stats((predictors[left][metric]-predictors[right][metric])[mask],
                            g['paired'][left+'_minus_'+right][metric],key+'/'+left+'-'+right+'/'+metric);count+=1
    return count


def rebuild_groups(predictors, strata, labels):
    names = set(predictors)-set(old.REFERENCE_NAMES)
    pairs = {(name,'bayes_hidden') for name in names}
    pairs |= {('bayes_hidden','bayes_encoding_given'),('flat_dirichlet','bayes_hidden')}
    for arm in REQUIRED:
        for seed in (0,1):
            pairs |= {(f'{arm}_seed{seed}',f'{previous}_seed{seed}') for previous in old.ARMS}
    for seed in (0,1):
        pairs.add((f'xlarge_seed{seed}',f'continue_large_seed{seed}'))
        pairs |= {(f'{arm}_seed{seed}',f'baseline_seed{seed}') for arm in ('longsmall','large')}
    groups=[]
    for key,mask,label in [('full_prior',np.ones(3840,dtype=bool),None)]+[(f'entropy_{i}',strata==i,label) for i,label in enumerate(labels)]:
        groups.append(dict(id=key,n_seq=int(mask.sum()),status='complete',entropy_label=label,
            predictors={name:{m:stats(values[m][mask]) for m in METRICS} for name,values in sorted(predictors.items())},
            paired={left+'_minus_'+right:{m:stats((predictors[left][m]-predictors[right][m])[mask]) for m in METRICS}
                    for left,right in sorted(pairs)}))
    return groups


def validate(source=SOURCE, development=DEVELOPMENT, original_source=old.SOURCE, cohort=old.COHORT,
             references=old.REFERENCES, xlarge_step=400000):
    source,development,original_source,cohort,references=map(Path,(source,development,original_source,cohort,references))
    hashes=Inputs();plan=read(PLAN);hashes.bind(PLAN);hashes.bind(old.PLAN);hashes.bind(old.__file__)
    selected_steps=dict(REQUIRED,xlarge=xlarge_step)
    final_checkpoint,_,checkpoint_scope=selection_metadata(plan,selected_steps)
    require(plan['source_dimensions']==[2,8] and plan['seeds']==[0,1], 'Campaign source or seeds changed')
    for field in ('implementation_sha256','adapter_dependencies_sha256'):
        for name,digest in plan[field].items():hashes.bind(ROOT/'code/transformer'/name,digest)
    require(plan['parent']['plan_sha256']==sha(old.PLAN), 'Original campaign plan changed')
    baseline,baseline_audit=old.validate(original_source,cohort,references)
    require(baseline_audit['passed'] and baseline_audit['full_reference_checked'], 'Original comparison failed independent audit')
    for p in (original_source/'summary.json',original_source/'status.json',cohort/'cohort.json',cohort/'cohort.npz',references/'manifest.json'):
        hashes.bind(p)
    data=arrays(cohort/'cohort.npz');indices=np.arange(256,4096);oracle=old.oracle_rows(data)[indices]
    tokens=data['tokens'][indices];strata=data['stratum'][indices];labels=read(cohort/'cohort.json')['stratum_labels']
    predictors={}
    for arm in old.ARMS:
        for seed in (0,1):
            name=f'{arm}_seed{seed}';p=original_source/'models'/f'{name}.npz'
            hashes.bind(p,baseline['model_receipts'][name]['data_sha256']);hashes.bind(p.with_suffix('.json'))
            a=arrays(p);predictors[name]={m:a[m] for m in METRICS}
    # The old independent audit above has already checked every record hash,
    # direct source transition row and probability/loss identity.
    rows=[arrays(references/f'seq_{i:06d}.npz') for i in indices]
    for name in old.REFERENCE_NAMES:
        predictors[name]={m:np.stack([a[name+'_'+m] for a in rows]) for m in METRICS}
    summaries={};receipts={};checked=0
    for arm,step in selected_steps.items():
        folder=source/arm/f'step{step}'
        summary=check_run_envelope(folder,arm,step,plan,baseline,hashes);summaries[arm]=summary
        for seed in (0,1):
            name=f'{arm}_seed{seed}'
            predictors[name],receipts[name]=load_campaign_model(folder,arm,seed,step,summary,baseline,development,
                plan,hashes,oracle,tokens)
        checked+=compare_remote_summary(summary,predictors,strata,labels)
    result=dict(schema=SCHEMA,complete=True,scientific_results=True,test_role='reused_comparison',fresh_confirmation=False,
        plan_sha256=sha(PLAN),original_plan_sha256=sha(old.PLAN),cohort_id=baseline['cohort_id'],
        cohort_data_sha256=baseline['cohort_data_sha256'],reference_records_sha256=baseline['reference_records_sha256'],
        sequence_start=256,sequence_stop=4096,evaluated_n_sequences=3840,excluded_development_sequences=256,
        planned_n_sequences=4096,dimensions=[2,8],sequence_length=1025,times=TIMES.tolist(),first_token_scored=False,
        selected_steps=selected_steps,final_checkpoint=final_checkpoint,
        groups=rebuild_groups(predictors,strata,labels),model_receipts={**baseline['model_receipts'],**receipts},
        new_training=False,new_inference=False,new_bayesian_computation=False,
        interpretation=plan['interpretation'],uncertainty='One paired standard error across sequences; training seeds stay separate.',
        scope='Previously examined 3840-case comparison. '+checkpoint_scope)
    audit=dict(schema='hidden-factor-campaign-independent-audit-v1',passed=True,full_reference_checked=True,
        model_checks_passed=True,evaluated_sequences=3840,groups_checked=5,remote_metric_summaries_checked=checked,
        consumer_sha256=sha(__file__),original_consumer_sha256=sha(old.__file__),plan_sha256=sha(PLAN),
        input_sha256=hashes.hashes,original_comparison_audit=baseline_audit,
        limitations=['Weight binaries are not transferred; checkpoint hashes match their validated development receipts.',
                     'Saved cumulative model losses are aggregated, not rerun by tokenwise model inference.',
                     'Exact Bayesian enumeration is not repeated; all saved reference records are independently checked.',
                     'The cohort was previously examined, and two seeds do not establish a model scaling law.'])
    return result,audit


def curve(values):
    mean,se=np.asarray(values['mean']),np.asarray(values['se'])
    require(values['n_seq']==3840 and mean.shape==se.shape==(6,) and np.isfinite(mean).all() and np.isfinite(se).all()
            and (mean>=0).all() and (se>=0).all(), 'Invalid publication curve')
    return dict(times=TIMES.tolist(),mean=mean.tolist(),se=se.tolist())


def panel(data,plan):
    final,labels,checkpoint_scope=selection_metadata(plan,data['selected_steps'])
    require(data['final_checkpoint']==final,'Publication final-checkpoint flags differ from selected steps')
    predictors=next(g for g in data['groups'] if g['id']=='full_prior')['predictors'];cases=[]
    for arm in ARMS:
        for seed in (0,1):
            name=f'{arm}_seed{seed}';receipt=data['model_receipts'][name]
            if arm=='baseline':
                step=100000;params=531968;digest=receipt['identity']['pilot']['checkpoint_sha256']
            else:
                step=data['selected_steps'][arm];params=plan['arms'][arm]['n_params'];digest=receipt['identity']['checkpoint']['checkpoint_sha256']
                require(receipt['identity']['step']==step and receipt['identity']['final_checkpoint'] is final[arm],
                        'Publication receipt and selected checkpoint differ')
            cases.append(dict(label=labels[arm],arm=arm,color=COLORS[arm],condition=dict(dimensions=[2,8],arm=arm,training_seed=seed),
                checkpoint_sha256=digest,checkpoint_step=step,n_params=params,training_sequences=step*64,
                n_sequences=3840,n_tokens=1025,interim=arm!='baseline' and not final[arm],transformer=curve(predictors[name]['next_kl_bits'])))
    require(len({c['checkpoint_sha256'] for c in cases})==6, 'Repeated publication checkpoint')
    return dict(id='c',title='Product source: hidden encoding',cases=cases,bayes=curve(predictors['bayes_hidden']['next_kl_bits']),
        bayes_style='points only',cohort_id=data['cohort_id'],pilot=False,test_role='reused_comparison',fresh_confirmation=False,
        evaluated_sequences=3840,planned_sequences=4096,excluded_development_sequences=256,
        quantity='Full-prior expected next-token regret on the same previously examined 3840 sequences.',
        seed_rule='Main text uses fixed training run 1; both runs appear separately in the appendix. No seed pooling or selection.',
        checkpoint_scope=checkpoint_scope)


def draw(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    group=next(g for g in data['groups'] if g['id']=='full_prior')
    _,labels,_=selection_metadata(read(PLAN),data['selected_steps'])
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8.,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(5.5,3.05),sharey=True)
        fig.subplots_adjust(left=.11,right=.975,bottom=.20,top=.64,wspace=.17)
        display=('baseline','large','continue_large','xlarge')
        styles={'baseline':'-','large':'--','continue_large':'-.','xlarge':(0,(7,2))}
        for seed,ax in enumerate(axes):
            for arm in display:
                v=group['predictors'][f'{arm}_seed{seed}']['next_kl_bits'];mean,se=np.asarray(v['mean']),np.asarray(v['se'])
                ax.plot(TIMES,mean,color=COLORS[arm],linestyle=styles[arm],linewidth=1.15)
                ax.fill_between(TIMES,np.maximum(0,mean-se),mean+se,color=COLORS[arm],alpha=.13,linewidth=0)
            v=group['predictors']['bayes_hidden']['next_kl_bits'];mean,se=np.asarray(v['mean']),np.asarray(v['se'])
            ax.errorbar(TIMES,mean,yerr=se,color='#222222',linestyle='none',marker='o',markersize=3.5,
                        markerfacecolor='white',markeredgewidth=.8,elinewidth=.75,capsize=1.5,zorder=5)
            ax.set_xscale('log',base=2);ax.set_xlim(.85,1250);ax.set_xticks(TIMES,[str(t) for t in TIMES]);ax.minorticks_off()
            ax.set_ylim(0,.9);ax.set_yticks([0,.3,.6,.9]);ax.tick_params(labelsize=7,length=2.5,pad=2)
            ax.grid(axis='y',color='#D8DEE5',linewidth=.5);ax.set_axisbelow(True)
            ax.set_title(f'({chr(97+seed)}) Training run {seed+1}',loc='left',fontsize=8.5,pad=5)
        handles=[Line2D([],[],color=COLORS[a],linestyle=styles[a],linewidth=1.3) for a in display]
        handles+=[Line2D([],[],color='#222222',linestyle='none',marker='o',markerfacecolor='white',markersize=3.5)]
        fig.legend(handles,[labels[a] for a in display]+['Informed Bayes'],loc='upper center',bbox_to_anchor=(.54,1.015),
                   ncol=1,frameon=False,fontsize=7.1,handlelength=2.,labelspacing=.3)
        fig.supylabel('Expected next-token regret (bits)',x=.008,fontsize=8.)
        fig.supxlabel('Observed tokens, $t$',y=.075,fontsize=8.)
        fig.text(.54,.012,'Same 3,840 reused sequences; shading shows ±1 standard error.',ha='center',fontsize=7.)
        for ext in ('pdf','png'):
            kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {'dpi':260}
            fig.savefig(out/f'next_token_regret.{ext}',**kw)
        plt.close(fig)


def write_results(data,audit,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    audit['summary_sha256']=sha(out/'comparison.json')
    figure=out/'figures';figure.mkdir(exist_ok=True);draw(data,figure)
    audit['figures']={str(p.relative_to(out)):sha(p) for p in sorted(figure.glob('next_token_regret.*'))}
    group=next(g for g in data['groups'] if g['id']=='full_prior')
    final,labels,checkpoint_scope=selection_metadata(read(PLAN),data['selected_steps'])
    with (out/'regret_at_1024.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['predictor','regret_bits','regret_se','minus_bayes_bits','paired_se'])
        for name,metrics in sorted(group['predictors'].items()):
            value=metrics['next_kl_bits'];gap=group['paired'].get(name+'_minus_bayes_hidden',{}).get('next_kl_bits')
            writer.writerow([name,value['mean'][-1],value['se'][-1],gap['mean'][-1] if gap else '',gap['se'][-1] if gap else ''])
    audit['table_sha256']=sha(out/'regret_at_1024.tsv')
    (out/'validation.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    lines=['# Hidden-encoding campaign: reused comparison','',
        'All models use the same 3,840 previously examined sequences, excluding the first 256 development cases. '
        +checkpoint_scope+' This comparison supplies no fresh independent confirmation.','',
        'Expected next-token regret after 1,024 observed tokens; mean ± one standard error across sequences:','',
        '| Model | Training run 1 | Training run 2 |','|---|---:|---:|']
    for arm in ('baseline','longsmall','large','continue_large','xlarge'):
        label=labels[arm]
        values=[group['predictors'][f'{arm}_seed{s}']['next_kl_bits'] for s in (0,1)]
        lines.append('| '+label+' | '+' | '.join(f"{v['mean'][-1]:.5f} ± {v['se'][-1]:.5f}" for v in values)+' |')
    bayes=group['predictors']['bayes_hidden']['next_kl_bits']
    lines+=['',f"Informed Bayes: {bayes['mean'][-1]:.5f} ± {bayes['se'][-1]:.5f} bits.",'',
        'The main panel uses fixed training run 1: two-layer 100,000-update model in blue, four-layer continuation in green, '
        'and eight-layer '+('final' if final['xlarge'] else 'intermediate')+' model in purple. The appendix figure shows both runs separately and adds the '
        'four-layer 100,000-update model in gray. The two-layer transformer is solid, '
        'all larger transformers use distinct dashed variants, and informed Bayes is shown by unconnected '
        'black open circles with error bars. '
        'No seed is selected or averaged. All six older models remain in the checked data and table.','',
        'The consumer reconstructs true transition rows from the source matrices and encoding, verifies all old model '
        'and reference records, checks complete remote status/selection receipts and checkpoint hashes against their '
        'development evaluations, replays next-token probabilities/losses, and independently recomputes all means, '
        'paired contrasts and standard errors. Cumulative losses are aggregated from saved arrays; weight binaries '
        'and exact Bayesian enumeration are not rerun. Different training schedules and joint depth/width changes '
        'prevent these observations from identifying a capacity-only effect or establishing convergence.','',
        'Reproduce with `python code/scripts/hidden_factor_campaign_results.py --xlarge-step '
        +str(data['selected_steps']['xlarge'])+'`. Historical comparisons also accept explicit input and development directories.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def load_campaign(source=OUTPUT):
    source=Path(source);paths=[source/'comparison.json',source/'validation.json',PLAN,old.PLAN]
    data,audit,plan,_=[read(p) for p in paths]
    final,_,_=selection_metadata(plan,data['selected_steps'])
    require(audit['schema']=='hidden-factor-campaign-independent-audit-v1' and audit['passed'] is True
            and audit['full_reference_checked'] is True and audit['model_checks_passed'] is True
            and audit['summary_sha256']==sha(paths[0]) and audit['consumer_sha256']==sha(__file__)
            and audit['original_consumer_sha256']==sha(old.__file__), 'Campaign lacks an unchanged independent audit')
    require(data['schema']==SCHEMA and data['complete'] is True and data['scientific_results'] is True
            and data['test_role']=='reused_comparison' and data['fresh_confirmation'] is False
            and data['plan_sha256']==audit['plan_sha256']==sha(PLAN) and data['original_plan_sha256']==sha(old.PLAN)
            and data['final_checkpoint']==final
            and data['evaluated_n_sequences']==audit['evaluated_sequences']==3840
            and data['sequence_start']==256 and data['sequence_stop']==4096 and data['times']==TIMES.tolist(),
            'Campaign result scope changed')
    return paths,data,panel(data,plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in [('input',SOURCE),('development',DEVELOPMENT),('original',old.SOURCE),('cohort',old.COHORT),('references',old.REFERENCES),('out',OUTPUT)]:
        parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--xlarge-step',type=int,choices=(200000,400000),default=400000)
    parser.add_argument('--check-only',action='store_true');a=parser.parse_args()
    data,audit=validate(a.input,a.development,a.original,a.cohort,a.references,a.xlarge_step)
    if not a.check_only:write_results(data,audit,a.out)
    print(json.dumps({k:audit[k] for k in ('passed','full_reference_checked','evaluated_sequences','remote_metric_summaries_checked')}))


if __name__=='__main__':main()
