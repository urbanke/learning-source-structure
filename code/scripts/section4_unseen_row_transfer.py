"""Exploratory unseen-row predictions from already audited shared-base results.

No inference, sampling, source generation or training. For each fixed t and
depth, keep a sequence iff its final observed label has not appeared earlier.
All uncertainty bars use independent whole-sequence losses on that subset.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/transformer'))
from shared_base_conditional_cohort import inputs
from shared_base_precision_cohort import validate_cohort
from eval_shared_base_models import fingerprint
from shortprefix_shared_base_summary import prediction_for_tokens

RAW = ROOT/'output/shared_base_conditional_curves_2026-09-15'
RELEASE = ROOT/'results/transformer/shared_base_conditional_curves'
OUT = ROOT/'results/transformer/section4_unseen_row_transfer'
DEPTHS, TIMES = (0, 1, 4, 8), (4, 16)
PREDICTORS = ('bayes', 'transformer_seed0', 'transformer_seed1', 'kt_unigram', 'one_token_bayes', 'uniform')
LABELS = dict(bayes='Informed Bayes', transformer_seed0='Transformer 0',
              transformer_seed1='Transformer 1', kt_unigram='Count baseline',
              one_token_bayes='One-token Bayes', uniform='Uniform')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else value


class Provenance:
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, f'Input changed: {path}')
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        require(name not in self.files or self.files[name] == digest, f'Input changed during read: {path}')
        self.files[name] = digest
        return raw

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def arrays(self, path, expected=None):
        with np.load(BytesIO(self.read(path, expected)), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}


def unseen_row(tokens):
    tokens = np.asarray(tokens)
    require(tokens.ndim == 1 and len(tokens) > 0, 'Expected a nonempty observed sequence')
    # Outgoing transitions already observed have origins X_1,...,X_{t-1}.
    return bool(not np.any(tokens[:-1] == tokens[-1]))


def loss(probabilities, oracle):
    p, r = np.asarray(probabilities), np.asarray(oracle)
    require(p.shape == r.shape == (64,) and np.isfinite(p).all() and np.min(p) > 0
            and abs(p.sum()-1) < 1e-10 and np.isfinite(r).all() and np.min(r) > 0
            and abs(r.sum()-1) < 1e-10, 'Invalid predictive/true row')
    return float(-r@np.log2(p))


def moments(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), 'At least two independent finite cases required')
    return dict(n=len(values), mean=float(values.mean()),
                standard_error=float(values.std(ddof=1)/math.sqrt(len(values))))


def build():
    tracked = Provenance()
    release16 = tracked.json(RELEASE/'time16_summary.json')
    release4 = tracked.json(RELEASE/'time4_summary.json')
    require(release16['reference_validation_passed'] and release4['reference_validation_passed'], 'Released references have not passed')
    require(release16['every_declared_sequence_retained'] and release4['every_declared_sequence_retained'], 'Original cohort release was selective')
    for name, expected in release16['provenance']['script_sha256'].items():
        tracked.read(ROOT/'code/scripts'/name, expected)
    catalog_record = tracked.json(RELEASE/'shortprefix_reference.json', release4['provenance']['reference_catalog_sha256'])
    require(catalog_record['passed'] and catalog_record['time'] == 4 and catalog_record['prior'] == release16['prior'], 'Invalid deterministic four-token catalog')
    catalog = {tuple(row['pattern']): row for row in catalog_record['records']}
    for numerical in catalog_record['provenance']['numerical_reports']:
        tracked.read(numerical['path'], numerical['sha256'])
    first = tracked.json(ROOT/'results/transformer/conditional_shared_base/first_step.json')
    require(first['passed'] and first['time'] == 1 and first['prior'] == release16['prior'], 'One-token comparator has not passed')
    tracked.read(ROOT/'code/scripts/conditional_shared_base_first_step.py', first['script_sha256'])
    first_probs = np.asarray(first['reports'][-1]['probs'])
    require(np.max(np.abs(first_probs[1:]-first_probs[1])) < 1e-12, 'One-token symmetry failed')
    rows, groups = [], []
    model_hashes = {}
    for depth in DEPTHS:
        folder = RAW/f'a1_L{depth}'
        receipt16 = next(row for row in release16['provenance']['cohorts'] if row['depth'] == depth)
        receipt4 = next(row for row in release4['provenance']['cohorts'] if row['depth'] == depth)
        audit = tracked.json(receipt16['audit_path'], receipt16['audit_sha256'])
        require(audit['all_requested_times_validated'] is True and audit['times'] == [16]
                and audit['orders'] == [16, 32] and audit['n_sequences'] == 256
                and audit['depth'] == depth and audit['alpha'] == 1., 'Wrong audited cohort')
        audited = {(row['index'], row['order']): row for row in audit['rows'][0]['case_audits']}
        require(set(audited) == {(i, order) for i in range(256) for order in (16, 32)}, 'Audit coverage incomplete')
        cohort = tracked.json(folder/'cohort.json', receipt16['cohort_sha256'])
        require(sha(folder/'cohort.json') == receipt4['cohort_sha256'], 'Different four-token cohort')
        tracked.read(folder/'protocol.json', receipt16['protocol_sha256'])
        require(cohort['n_sequences'] == 256 and [c['index'] for c in cohort['records']] == list(range(256)), 'Wrong or reordered cohort')
        four = tracked.arrays(receipt4['array_path'], receipt4['array_sha256'])
        require(np.array_equal(four['source_indices'], np.arange(256)), 'Four-token reference indices changed')
        source_records = []
        for i in range(256):
            source_records.append(tracked.json(folder/'data'/f'seq_{i:06d}.json', cohort['records'][i]['file_sha256']))
        for t in TIMES:
            selected = [i for i, source in enumerate(source_records) if unseen_row(source['tokens'][:t])]
            require(len(selected) >= 2, 'Too few selected sources')
            if t == 16:
                numeric = native(validate_cohort(
                    [audited[i, 16]['record'] for i in selected],
                    [audited[i, 32]['record'] for i in selected], selected))
                require(numeric['passed'], f'Unseen-row subset fails numerical precision at L={depth}')
            else:
                numeric = dict(passed=True, method='Existing deterministic rank marginalization with exponent/row refinement.',
                               maximum_individual_loss_refinement_bits=max(row['maximum_refinement_change_in_individual_regret']
                                   for row in release4['rows'] if row['depth'] == depth))
            group_rows = []
            for i in selected:
                source = source_records[i]
                observed, oracle, models, identity = inputs(folder, i, t)
                require(np.array_equal(observed, source['tokens'][:t]) and unseen_row(observed), 'Selection or sequence changed')
                references = {}
                for order in (16, 32):
                    checked = audited[i, order]
                    require(checked['status'] == 'ready' and checked['ready']
                            and checked['recomputed_final_states'] and checked['recomputed_finer_grid'], 'Unvalidated selected reference')
                    reference = tracked.json(checked['report'], checked['report_sha256'])
                    require(reference['reference_id'] == checked['reference_id'] == fingerprint(reference['identity'])
                            and reference['record'] == checked['record']
                            and reference['setting'] == folder.name and reference['index'] == i
                            and reference['time'] == 16 and reference['order'] == order, 'Selected reference mismatch')
                    original_inputs = reference['identity']['inputs']
                    for key in ('protocol_sha256', 'source_id', 'source_file_sha256', 'tokens_sha256', 'model_files'):
                        require(identity[key] == original_inputs[key], f'Frozen model/source changed: {key}')
                    references[order] = reference
                for name, model in models.items():
                    metadata = tracked.json(model['path'], model['sha256'])
                    tracked.read(Path(model['path']).with_suffix('.npz'), metadata['arrays_sha256'])
                    spec = metadata['identity']['model']
                    require(name not in model_hashes or model_hashes[name] == spec['checkpoint_sha256'], 'Mixed model checkpoints')
                    model_hashes[name] = spec['checkpoint_sha256']
                bayes = four['bayes_predictive_probs'][i] if t == 4 else np.asarray(audited[i,32]['record']['precision']['predictive_probs'])
                if t == 4:
                    require(np.allclose(bayes, prediction_for_tokens(observed, catalog), rtol=0, atol=1e-15), 'Four-token catalog mapping differs')
                else:
                    require(np.allclose(bayes, references[32]['estimate']['predictive_probs'], rtol=0, atol=2e-14), 'Audited predictive mean differs')
                    regret = float(np.sum(rel_entr(oracle, bayes))/math.log(2))
                    require(abs(regret-audited[i,32]['record']['precision']['bayes_regret_bits']) < 2e-12, 'Bayes loss replay differs')
                counts = np.bincount(observed, minlength=64)
                one = np.full(64, first_probs[1])
                one[int(observed[-1])] = first_probs[0]
                predictions = dict(bayes=bayes, transformer_seed0=models['d64']['prediction'],
                    transformer_seed1=models['d64_seed1']['prediction'], kt_unigram=(counts+.5)/(t+32),
                    one_token_bayes=one, uniform=np.ones(64)/64)
                losses = {name: loss(p, oracle) for name, p in predictions.items()}
                record = dict(depth=depth, time=t, index=i, source_id=source['source_id'], query=int(observed[-1]),
                              n_observed_tokens=t, prior_outgoing_transitions=0,
                              entropy_bits=float(-oracle@np.log2(oracle)), loss_bits=losses)
                rows.append(record)
                group_rows.append(record)
            predictors = {}
            for name in PREDICTORS:
                values = [row['loss_bits'][name] for row in group_rows]
                predictors[name] = dict(loss_bits=moments(values),
                    regret_bits=moments([value-row['entropy_bits'] for value,row in zip(values,group_rows)]),
                    gain_over_uniform_bits=moments([6-value for value in values]),
                    gain_over_one_token_bayes_bits=moments([row['loss_bits']['one_token_bayes']-row['loss_bits'][name] for row in group_rows]))
            paired = {name+'_minus_bayes': moments([row['loss_bits'][name]-row['loss_bits']['bayes'] for row in group_rows])
                      for name in ('transformer_seed0','transformer_seed1','kt_unigram')}
            groups.append(dict(depth=depth,time=t,n_selected=len(selected),n_original=256,
                               selected_indices=selected,predictors=predictors,paired_loss_bits=paired,
                               subset_numerical_validation=numeric))
    result = dict(schema='section4-unseen-row-exploratory-v1', alpha=1., d=64, depths=list(DEPTHS),times=list(TIMES),
        status='Exploratory post hoc selection from frozen validated cohorts; no new experiment.',
        selection='At time t retain exactly X_t not in {X_1,...,X_(t-1)}. Hence the prefix contains zero completed transitions out of X_t.',
        score='Expected next-token log loss in bits, averaged over the true transition row separately for each source sequence.',
        uncertainty='One standard error across independent selected whole sequences. Numerical Bayes precision is checked separately on each selected subset.',
        comparators=dict(uniform='Exactly 1/64; also the full-prior uniform-base prediction for a row with no observed outgoing transition.',
            kt_unigram='Counts-only Krichevsky-Trofimov rule (N_b+1/2)/(t+32), computed from all observed tokens. A descriptive baseline with a different prior.',
            one_token_bayes='Existing full-family one-token Bayes prediction mapped to the final observed label. This is a memory-reset reference, not an optimal predictor conditioned on time t and the unseen-row event.'),
        interpretation=['Positive gain shows useful prediction despite no observations from the queried transition row.',
            'Gain over uniform includes token-frequency information and does not establish an internal circuit or richer shared-base computation.',
            'The count baseline is competitive in most displayed conditions; it is not a matched learned-structure ablation.',
            'At L=8,t=16 the gain is not clearly separated from zero.',
            'Selection yields different source subsets at t=4 and t=16. The time points must not be treated as paired longitudinal changes.',
            'L=0 is iid; it is a descriptive comparison, not evidence of nontrivial row-specific transfer.'],
        model_checkpoint_sha256=model_hashes,groups=groups,
        input_sha256=tracked.files, script_sha256=sha(__file__),
        raw_chains_recomputed=False,
        validation_scope='Rechecked the hash-bound completed t16 raw-audit records, input/model identities, deterministic t4 mapping, all losses, and selected-subset numerical gates. No partially changing t64/256/1024 result is used.')
    return result, rows


def draw_panel(ax, summary, time=16, *, include_counts=True, legend=True):
    """Reusable compact panel; gains are relative to exactly uniform prediction."""
    chosen = sorted((g for g in summary['groups'] if g['time'] == time), key=lambda x:x['depth'])
    require(len(chosen) == 4 and all(g['subset_numerical_validation']['passed'] for g in chosen), 'Plot requires all four validated subsets')
    styles = [('bayes','#222222','D'),('transformer_seed0','#2468A0','o'),
              ('transformer_seed1','#BF611C','s')]
    if include_counts:
        styles.append(('kt_unigram','#777777','x'))
    offsets = np.linspace(-.24,.24,len(styles))
    for offset,(name,color,marker) in zip(offsets,styles):
        values = [g['predictors'][name]['gain_over_uniform_bits'] for g in chosen]
        ax.errorbar(np.arange(4)+offset,[v['mean'] for v in values],
                    yerr=[v['standard_error'] for v in values],label=LABELS[name],
                    color=color,marker=marker,markerfacecolor='none' if name == 'bayes' else color,
                    linestyle='none',markersize=3.4,elinewidth=.8,capsize=1.8)
    ax.axhline(0,color='#888888',linewidth=.65)
    ax.set_xticks(range(4),[f"L={g['depth']}\n n={g['n_selected']}" for g in chosen])
    ax.set_ylabel('Gain over uniform (bits)')
    ax.set_title(f'Unseen row; {time} observed tokens')
    ax.set_xlim(-.6,3.6)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    if legend:
        ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper right',columnspacing=.8,handletextpad=.3)


def figures(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42}):
        fig,ax = plt.subplots(figsize=(2.9,2.8))
        draw_panel(ax,summary,16)
        ax.set_ylim(-.12,.83)
        fig.text(.5,.015,'Bars: one SE across selected sequences',ha='center',fontsize=7)
        fig.tight_layout(rect=(0,.035,1,1))
        fig.savefig(out/'unseen_row_t16.pdf')
        fig.savefig(out/'unseen_row_t16.png',dpi=220)
        plt.close(fig)
        # Render at the manuscript's final width so all text remains >=7 pt.
        fig,axes = plt.subplots(1,2,figsize=(5.5,2.9),sharey=True)
        for ax,t in zip(axes,TIMES):
            draw_panel(ax,summary,t,legend=False)
            ax.set_ylim(-.12,.83)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,['Bayes','TF seed 0','TF seed 1','Token counts'],ncol=4,loc='upper center',frameon=False,fontsize=7)
        fig.text(.5,.015,'Bars: one SE; n is selected independently at each time and depth.',ha='center',fontsize=7)
        fig.tight_layout(rect=(0,.04,1,.91))
        fig.savefig(out/'unseen_row_t4_t16.pdf')
        fig.savefig(out/'unseen_row_t4_t16.png',dpi=220)
        plt.close(fig)


def self_test():
    assert unseen_row([0]) and unseen_row([0,0,1]) and unseen_row([0,1,0,2])
    assert not unseen_row([0,1,0]) and not unseen_row([0,1,1])
    p=np.ones(64)/64
    assert loss(p,p)==6.
    assert moments([1.,3.])==dict(n=2,mean=2.,standard_error=1.)
    try:
        loss(np.ones(64),p)
    except ValueError:
        pass
    else:
        raise AssertionError('Unnormalized prediction accepted')


def verify_export(out):
    """Independent arithmetic replay from compact per-sequence losses."""
    out = Path(out)
    summary = json.loads((out/'summary.json').read_bytes())
    require(summary['script_sha256'] == sha(__file__), 'Exporter changed after this result')
    require(summary['records_sha256'] == sha(out/'records.csv'), 'Per-sequence losses changed')
    with (out/'records.csv').open() as stream:
        records = list(csv.DictReader(stream))
    comparisons = 0
    for group in summary['groups']:
        rows = [r for r in records if int(r['depth']) == group['depth'] and int(r['time']) == group['time']]
        require([int(r['index']) for r in rows] == group['selected_indices'], 'Exported selection changed')
        def compare(values, saved):
            nonlocal comparisons
            require(saved['n'] == len(rows) == group['n_selected'], 'Exported sample count differs')
            mean = statistics.mean(values)
            se = statistics.stdev(values)/math.sqrt(len(values))
            require(math.isclose(mean,saved['mean'],abs_tol=2e-14,rel_tol=1e-12)
                    and math.isclose(se,saved['standard_error'],abs_tol=2e-14,rel_tol=1e-12), 'Independent loss/statistic replay differs')
            comparisons += 1
        for name in PREDICTORS:
            values = [float(r[name+'_loss_bits']) for r in rows]
            p = group['predictors'][name]
            compare(values,p['loss_bits'])
            compare([v-float(r['entropy_bits']) for v,r in zip(values,rows)],p['regret_bits'])
            compare([6-v for v in values],p['gain_over_uniform_bits'])
            compare([float(r['one_token_bayes_loss_bits'])-v for v,r in zip(values,rows)],p['gain_over_one_token_bayes_bits'])
        for name in ('transformer_seed0','transformer_seed1','kt_unigram'):
            compare([float(r[name+'_loss_bits'])-float(r['bayes_loss_bits']) for r in rows],group['paired_loss_bits'][name+'_minus_bayes'])
    return dict(passed=True,statistic_pairs_recomputed=comparisons,records=len(records),
                method='Python statistics.mean/stdev over compact whole-sequence loss records; independent of NumPy aggregation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    self_test()
    if args.self_test:
        print('Selection, scoring and uncertainty checks passed.')
        return
    if args.verify_only:
        print(json.dumps(verify_export(args.out)))
        return
    summary,records=build()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=['depth','time','index','source_id','query','entropy_bits']+[n+'_loss_bits' for n in PREDICTORS])
        writer.writeheader()
        for row in records:
            writer.writerow({key:row[key] for key in ('depth','time','index','source_id','query','entropy_bits')} |
                            {name+'_loss_bits':row['loss_bits'][name] for name in PREDICTORS})
    summary['records_sha256']=sha(args.out/'records.csv')
    (args.out/'summary.json').write_text(json.dumps(native(summary),indent=2,allow_nan=False)+'\n')
    figures(summary,args.out)
    receipt=dict(passed=True,scientific_scope='exploratory conditional subgroups',
                 groups=len(summary['groups']),selected_records=len(records),source_sequences_declared=1024,
                 exact_selection_replayed=True,all_subset_numerical_gates_passed=True,script_sha256=sha(__file__),
                 independent_csv_replay=verify_export(args.out),
                 files={name:sha(args.out/name) for name in ('summary.json','records.csv','unseen_row_t16.pdf',
                         'unseen_row_t16.png','unseen_row_t4_t16.pdf','unseen_row_t4_t16.png')})
    (args.out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),groups=8,records=len(records),subset_gates_passed=True)))


if __name__=='__main__':
    main()
