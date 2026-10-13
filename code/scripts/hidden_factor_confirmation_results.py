"""Independently audit and present saved hidden-encoding confirmation scores.

No model inference, Bayesian inference or source generation is performed.
References and all six models use indices256:4096 of the frozen cohort; the
256 development sequences are excluded. Every training run stays separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json'
SOURCE = ROOT/'output/hidden_factor_confirmation_2026-09-18'
COHORT = ROOT/'output/hidden_factor_cluster_pilot_2026-09-16'
REFERENCES = ROOT/'output/hidden_factor_full_references_2026-09-18'
OUTPUT = ROOT/'results/transformer/hidden_factor_confirmation_2026-09-18'
PILOT = ROOT/'results/transformer/hidden_factor_scaling_2026-09-17'
TIMES = np.array([1, 4, 16, 64, 256, 1024])
METRICS = ('next_kl_bits', 'next_loss_bits', 'cumulative_loss_bits')
ARMS = ('baseline', 'longsmall', 'large')
REFERENCE_NAMES = ('bayes_hidden', 'bayes_encoding_given', 'flat_dirichlet', 'oracle')
COLORS = {'baseline': '#2468A0', 'longsmall': '#BF611C', 'large': '#288167'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


def moments(values):
    return dict(n_seq=len(values), mean=values.mean(0).tolist(),
                se=(values.std(0, ddof=1)/np.sqrt(len(values))).tolist())


def compare_moments(values, saved, name):
    rebuilt = moments(values)
    require(saved['n_seq'] == rebuilt['n_seq'], name+': sequence count changed')
    for key in ('mean', 'se'):
        require(np.allclose(saved[key], rebuilt[key], rtol=0, atol=2e-10), name+': '+key+' mismatch')


def oracle_rows(data):
    """Reconstruct selected transition rows directly from the two source matrices."""
    inverse = np.argsort(data['encoding'], axis=1)
    rows = []
    for i in range(len(inverse)):
        a, b = inverse[i]//8, inverse[i]%8
        labels = data['tokens'][i, TIMES-1]
        p = data['first_matrix'][i][a[labels, None], a[None, :]]
        p *= data['second_matrix'][i][b[labels, None], b[None, :]]
        rows.append(p)
    return np.stack(rows)


def check_predictions(p, oracle, tokens, saved, label):
    require(p.shape == oracle.shape and np.isfinite(p).all() and (p > 0).all()
            and np.allclose(p.sum(-1), 1, rtol=0, atol=4e-6), label+': invalid probabilities')
    kl = np.sum(oracle*np.log2(oracle/p), axis=-1)
    require(kl.min() >= -2e-6, label+': negative KL beyond numerical allowance')
    require(np.allclose(np.maximum(0, kl), saved['next_kl_bits'], rtol=0, atol=1e-11), label+': KL replay mismatch')
    realized = -np.log2(np.take_along_axis(p, tokens[:, TIMES, None], axis=-1)[..., 0])
    require(np.allclose(realized, saved['next_loss_bits'], rtol=0, atol=1e-11), label+': realized loss replay mismatch')
    for metric in METRICS:
        require(saved[metric].shape == p.shape[:2] and np.isfinite(saved[metric]).all(), label+': invalid '+metric)
    require(np.allclose(saved['cumulative_loss_bits'][:, 0], realized[:, 0], rtol=0, atol=1e-11),
            label+': first-token scoring convention changed')
    require((np.diff(saved['cumulative_loss_bits'], axis=1) >= -1e-10).all(), label+': cumulative loss decreases')


def validate(source=SOURCE, cohort_root=COHORT, references=REFERENCES, models_only=False):
    source, cohort_root, references = map(Path, (source, cohort_root, references))
    summary, status, plan = read(source/'summary.json'), read(source/'status.json'), read(PLAN)
    require(summary['schema'] == 'hidden-factor-confirmation-v1' and summary['complete'] is True
            and summary['scientific_results'] is True and summary['test_role'] == 'untouched_confirmation',
            'Incomplete or wrong confirmation summary')
    names = {f'{arm}_seed{seed}' for arm in ARMS for seed in (0, 1)}
    require(status['complete'] is True and status['status'] == 'completed'
            and set(status['completed_models']) == names and status['evaluated_n_sequences'] == 3840,
            'Incomplete confirmation status')
    require(summary['plan_sha256'] == sha(PLAN) and summary['sequence_start'] == 256
            and summary['sequence_stop'] == 4096 and summary['evaluated_n_sequences'] == 3840
            and summary['development_n_sequences'] == 256 and summary['planned_n_sequences'] == 4096
            and summary['times'] == TIMES.tolist() and summary['dimensions'] == [2, 8]
            and summary['sequence_length'] == 1025 and summary['first_token_scored'] is False
            and summary['new_training'] is False and summary['new_bayesian_computation'] is False
            and summary['primary'] == plan['primary'], 'Confirmation scope or plan changed')
    cohort = read(cohort_root/'cohort.json')
    require(sha(cohort_root/'cohort.npz') == summary['cohort_data_sha256'] == cohort['data_sha256'] == plan['test']['data_sha256']
            and fingerprint(cohort['identity']) == cohort['cohort_id'] == summary['cohort_id'] == plan['test']['cohort_id'],
            'Frozen cohort identity changed')
    data = arrays(cohort_root/'cohort.npz')
    require(data['tokens'].shape == (4096, 1025), 'Incomplete frozen cohort')
    oracle = oracle_rows(data)
    indices = np.arange(256, 4096)
    pilot, pilot_audit = read(PILOT/'comparison.json'), read(PILOT/'validation.json')
    require(pilot_audit['passed'] is True and pilot_audit['summary_sha256'] == sha(PILOT/'comparison.json'),
            'Unvalidated original model identities')
    predictors, hashes = {}, {}
    for name in sorted(names):
        path = source/'models'/f'{name}.npz'
        receipt = read(path.with_suffix('.json')); ident = receipt['identity']
        require(receipt == summary['model_receipts'][name] and receipt['complete'] is True
                and receipt['data_sha256'] == sha(path) and receipt['n_sequences'] == 3840,
                name+': score receipt or checksum mismatch')
        require(ident['schema'] == summary['schema'] and ident['name'] == name
                and ident['test_role'] == summary['test_role'] and ident['plan_sha256'] == sha(PLAN)
                and ident['sequence_start'] == 256 and ident['sequence_stop'] == 4096
                and ident['times'] == TIMES.tolist() and ident['cohort_id'] == summary['cohort_id']
                and ident['cohort_data_sha256'] == summary['cohort_data_sha256']
                and ident['reference_records_sha256'] == summary['reference_records_sha256']
                and ident['first_token_scored'] is False, name+': source/model scope mismatch')
        for key, file in [('evaluator_sha256', 'hidden_factor_confirmation_eval.py'),
                          ('scaling_evaluator_sha256', 'hidden_factor_scaling_eval.py'),
                          ('original_evaluator_sha256', 'eval_hidden_factor.py')]:
            require(ident[key] == sha(ROOT/'code/transformer'/file), name+': evaluator changed')
        seed = name[-1]
        expected_checkpoint = (plan['baseline']['checkpoint_sha256'][seed] if name.startswith('baseline')
                               else pilot['model_provenance'][name]['checkpoint_sha256'])
        expected_records = (plan['baseline']['model_records_sha256']['seed'+seed] if name.startswith('baseline')
                            else pilot['model_records_sha256'][name])
        require(ident['pilot']['checkpoint_sha256'] == expected_checkpoint
                and ident['pilot']['records_sha256'] == expected_records, name+': differs from the planned final pilot model')
        require(receipt['causality_check']['passed'] is True
                and receipt['causality_check']['max_absolute_logit_difference'] <= 2e-5, name+': causal check failed')
        scored = arrays(path)
        require(np.array_equal(scored['sequence_indices'], indices)
                and np.array_equal(scored['times'], TIMES), name+': pairing changed')
        check_predictions(scored['predictive'], oracle[indices], data['tokens'][indices], scored, name)
        predictors[name] = {m: scored[m] for m in METRICS}
        hashes[name] = dict(data_sha256=sha(path), receipt_sha256=sha(path.with_suffix('.json')))
    reference_manifest_hash = None
    if not models_only:
        manifest = read(references/'manifest.json'); identity = manifest['identity']
        require(manifest['complete'] is True and manifest['evaluated_n_sequences'] == 4096
                and manifest['planned_n_sequences'] == 4096 and identity['cohort_id'] == summary['cohort_id']
                and identity['kind'] == 'reference', 'Full reference is incomplete or mismatched')
        for key, file in [('source_sha256', 'hidden_factor.py'), ('evaluator_sha256', 'eval_hidden_factor.py'),
                          ('reference_sha256', 'hidden_factor_gpu.py')]:
            require(identity[key] == sha(ROOT/'code/transformer'/file), 'Reference implementation changed')
        reference_rows, digests = [], []
        for i in range(4096):
            path = references/f'seq_{i:06d}.npz'; record = read(path.with_suffix('.json'))
            digest = sha(path)
            require(record['identity'] == dict(identity, sequence_index=i) and record['data_sha256'] == digest,
                    f'Reference{i}: identity/checksum mismatch')
            digests.append(digest)
            a = arrays(path)
            require(np.array_equal(a['times'], TIMES) and np.allclose(a['oracle_probs'], oracle[i], rtol=0, atol=1e-13),
                    f'Reference{i}: source/time mismatch')
            for name in REFERENCE_NAMES:
                p = a[name+'_predictive'][None]
                scored = {m: a[name+'_'+m][None] for m in METRICS}
                check_predictions(p, oracle[i:i+1], data['tokens'][i:i+1], scored, f'Reference{i}/{name}')
            if i >= 256:
                reference_rows.append(a)
        require(fingerprint(digests) == summary['reference_records_sha256'], 'Full reference record digest differs')
        for name in REFERENCE_NAMES:
            predictors[name] = {m: np.stack([a[name+'_'+m] for a in reference_rows]) for m in METRICS}
        reference_manifest_hash = sha(references/'manifest.json')
    strata = data['stratum'][indices]
    groups = {g['id']: g for g in summary['groups']}
    require(set(groups) == {'full_prior', 'entropy_0', 'entropy_1', 'entropy_2', 'entropy_3'}, 'Missing or extra summary group')
    comparisons = 0
    for group_name, group in groups.items():
        mask = np.ones(len(indices), dtype=bool) if group_name == 'full_prior' else strata == int(group_name[-1])
        require(group['status'] == 'complete' and group['n_seq'] == int(mask.sum()), 'Summary group membership changed')
        if group_name != 'full_prior':
            require(group['entropy_label'] == cohort['stratum_labels'][int(group_name[-1])], 'Entropy label changed')
        require(set(group['predictors']) == names | set(REFERENCE_NAMES), 'Predictor omitted or selected')
        for name, values in predictors.items():
            for m in METRICS:
                compare_moments(values[m][mask], group['predictors'][name][m], group_name+'/'+name+'/'+m)
                comparisons += 1
        expected_pairs = {'bayes_hidden_minus_bayes_encoding_given', 'flat_dirichlet_minus_bayes_hidden'}
        expected_pairs |= {name+'_minus_bayes_hidden' for name in names}
        expected_pairs |= {name+'_minus_baseline_seed'+name[-1] for name in names if not name.startswith('baseline')}
        require(set(group['paired']) == expected_pairs, 'Paired summary omitted or selected')
        for key, saved in group['paired'].items():
            left, right = key.split('_minus_')
            if left not in predictors or right not in predictors:
                continue
            for m in METRICS:
                compare_moments((predictors[left][m]-predictors[right][m])[mask], saved[m], group_name+'/'+key+'/'+m)
                comparisons += 1
    audit = dict(schema='hidden-factor-confirmation-independent-audit-v1', passed=not models_only,
                 model_checks_passed=True, full_reference_checked=not models_only,
                 summary_sha256=sha(source/'summary.json'), plan_sha256=sha(PLAN),
                 consumer_sha256=sha(__file__), cohort_data_sha256=sha(cohort_root/'cohort.npz'),
                 reference_records_sha256=summary['reference_records_sha256'],
                 reference_manifest_sha256=reference_manifest_hash, models=hashes,
                 evaluated_sequences=3840, excluded_development_sequences=256,
                 groups_checked=5, metric_summaries_checked=comparisons,
                 scientific_scope='Final preselected models on the untouched suffix; each training run separate.',
                 limitations=['Saved cumulative model losses are aggregated, not recomputed by tokenwise model inference.',
                              'Weight binaries are not transferred; recorded identities match the independently audited pilot.',
                              'Bayesian probability arrays and record hashes are checked; exact encoding enumeration is not rerun.',
                              'Sequence standard errors do not estimate variability over training runs.'])
    return summary, audit


def draw(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    group = next(g for g in summary['groups'] if g['id'] == 'full_prior')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8., 'axes.titlesize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
        fig.subplots_adjust(left=.11, right=.975, bottom=.22, top=.70, wspace=.17)
        for seed, ax in enumerate(axes):
            for name, color, marker, style in [(f'{arm}_seed{seed}', COLORS[arm], None, '-') for arm in ARMS]+[('bayes_hidden', '#222222', 'o', '--')]:
                value = group['predictors'][name]['next_kl_bits']
                mean, se = np.asarray(value['mean']), np.asarray(value['se'])
                ax.fill_between(TIMES, np.maximum(0, mean-se), mean+se, color=color, alpha=.1, linewidth=0)
                ax.plot(TIMES, mean, color=color, linestyle=style, linewidth=1.1,
                        marker=marker, markersize=3.2, markerfacecolor='white', markeredgewidth=.75)
            ax.set_xscale('log', base=2); ax.set_xlim(.85, 1250)
            ax.set_xticks(TIMES, [str(t) for t in TIMES]); ax.minorticks_off()
            ax.set_ylim(0, .9); ax.set_yticks([0, .3, .6, .9]); ax.tick_params(labelsize=7., length=2.5, pad=2)
            ax.grid(axis='y', color='#D8DEE5', linewidth=.5); ax.set_axisbelow(True)
            ax.set_title(f'({chr(97+seed)}) Training run {seed+1}', loc='left', pad=5)
        handles = [Line2D([], [], color=COLORS[arm], linewidth=1.4) for arm in ARMS]
        handles += [Line2D([], [], color='#222222', linestyle='--', linewidth=1.2, marker='o', markerfacecolor='white', markersize=3.2)]
        fig.legend(handles, ['Original', 'More training', 'Larger model', 'Informed Bayes'],
                   loc='upper center', bbox_to_anchor=(.54, 1.01), ncol=2, frameon=False, fontsize=7.5,
                   handlelength=2., columnspacing=1.5, labelspacing=.45)
        fig.supylabel('Expected next-token regret (bits)', x=.008, fontsize=8.)
        fig.supxlabel('Observed tokens, $t$', y=.095, fontsize=8.)
        fig.text(.54, .018, 'Same 3,840 confirmation sequences; shading shows ±1 standard error.', ha='center', fontsize=7.)
        for ext in ('pdf', 'png'):
            options = {'metadata': {'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {'dpi': 260}
            fig.savefig(out/f'next_token_regret.{ext}', **options)
        plt.close(fig)


def draw_conditional(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {
        'baseline_seed0': ('#2468A0', '-', 'Transformer, training run 1'),
        'baseline_seed1': ('#288167', '-', 'Transformer, training run 2'),
        'bayes_hidden': ('#BF611C', '--', 'Bayes, encoding hidden'),
        'bayes_encoding_given': ('#8A4E98', ':', 'Bayes, encoding given'),
    }
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8., 'axes.titlesize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0))
        fig.subplots_adjust(left=.105, right=.972, bottom=.14, top=.76, wspace=.28, hspace=.48)
        for ax, group in zip(axes.flat, [g for g in summary['groups'] if g['id'] != 'full_prior']):
            require(group['status'] == 'complete', 'Incomplete entropy group')
            for name, (color, style, label) in styles.items():
                values = group['predictors'][name]['next_kl_bits']
                mean, se = np.asarray(values['mean']), np.asarray(values['se'])
                ax.plot(TIMES, mean, color=color, linestyle=style, linewidth=1.3, label=label)
                ax.fill_between(TIMES, np.maximum(0, mean-se), mean+se, color=color, alpha=.12, linewidth=0)
            ax.set_title(f"{group['entropy_label']} entropy ($N={group['n_seq']}$)", loc='left')
            ax.set_xscale('log', base=2); ax.set_xlim(.85, TIMES[-1]*1.2)
            ax.set_xticks([1, 16, 1024], ['1', '16', '1024']); ax.minorticks_off()
            ax.grid(axis='y', color='#D8DEE5', linewidth=.5); ax.tick_params(labelsize=8, length=3)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=7.5)
        fig.text(.54, .845, 'Original model; coordinate 1 / coordinate 2 mean row entropy', ha='center', fontsize=8)
        fig.supylabel('Expected next-token regret (bits)', x=.008, fontsize=8.5)
        fig.supxlabel('Observed tokens, $t$', y=.052, fontsize=8.5)
        fig.text(.54, .008, '3,840 confirmation sequences; shading shows ±1 standard error.', ha='center', fontsize=7.5)
        for ext in ('pdf', 'png'):
            options = {'metadata': {'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {'dpi': 260}
            fig.savefig(out/f'conditional_curves.{ext}', **options)
        plt.close(fig)


def write_results(summary, audit, out):
    out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    audit['summary_sha256'] = sha(out/'comparison.json')
    figure_dir = out/'figures'; figure_dir.mkdir(exist_ok=True)
    draw(summary, figure_dir)
    draw_conditional(summary, figure_dir)
    audit['artifacts'] = {str(p.relative_to(out)): sha(p) for stem in ('next_token_regret', 'conditional_curves')
                          for p in sorted(figure_dir.glob(stem+'.*'))}
    (out/'validation.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    group = next(g for g in summary['groups'] if g['id'] == 'full_prior')
    table = ['| Predictor | Run 1 | Run 2 |', '|---|---:|---:|']
    for arm, label in [('baseline', 'Original'), ('longsmall', 'More training'), ('large', 'Larger model'), ('bayes_hidden', 'Informed Bayes')]:
        values = []
        for seed in (0, 1):
            p = group['predictors'][arm if arm == 'bayes_hidden' else f'{arm}_seed{seed}']['next_kl_bits']
            values.append(f"{p['mean'][-1]:.5f} ± {p['se'][-1]:.5f}")
        table.append('| '+label+' | '+' | '.join(values)+' |')
    text = '# Hidden-encoding source: independent confirmation\n\n'
    text += 'All six final models use the same 3,840 unused sequences, indices 256 through 4095 of the frozen cohort. '
    text += 'The first 256 sequences were used for exploratory model comparisons and are excluded here. '
    text += 'No model is selected using this confirmation; both training runs remain separate.\n\n'
    text += 'Expected next-token regret after 1,024 observed tokens, in bits; uncertainty is one standard error across sequences:\n\n'
    text += '\n'.join(table)+'\n\n'
    text += 'The original model has two layers and 0.53 million parameters, trained on 6.4 million sequences. '
    text += 'More training continues that model to 25.6 million sequences. The larger model has four layers and 3.43 million parameters at the original 6.4-million-sequence budget.\n\n'
    text += 'Both changes improve prediction in both runs; a clear Bayesian gap remains. The larger model with the longer training budget has not been tested. '
    text += 'These results do not establish convergence or a general scaling law.\n\n'
    text += 'The independent consumer checks the six compact model arrays, all 4,096 saved Bayesian records, model and source identities, true transition rows, next-token KL and realized losses, '
    text += 'all five source-only groups, and all paired means and standard errors. It aggregates saved cumulative losses without rerunning model inference. '
    text += 'The exact Bayesian enumeration is not repeated. Recorded final checkpoint identities match the earlier validated pilot; weight binaries are not transferred.\n\n'
    text += 'Figures retain the earlier display convention: solid blue/orange/green transformer curves, dashed black Bayes with open circles, one panel per training run, one-SE bands. '
    text += 'Run 1 is the preselected main-text comparison.\n\n'
    text += '`figures/conditional_curves.pdf` separately shows the four source-only entropy groups for the original models. '
    text += 'It retains the earlier conditional-figure style: training runs 1 and 2 in solid blue and green, hidden-encoding Bayes dashed orange, encoding-given Bayes dotted purple. '
    text += 'All group counts exclude the development sequences; the original-model gap persists in every group.\n\n'
    text += 'Reproduce from saved outputs and references:\n\n```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/hidden_factor_confirmation_results.py\n```\n'
    (out/'README.md').write_text(text)


def load_confirmation(source=OUTPUT):
    """Return audited saved data and the same panel contract as the pilot figure."""
    source = Path(source)
    paths = [source/'comparison.json', source/'validation.json', PLAN]
    data, audit, plan = [read(path) for path in paths]
    require(audit['schema'] == 'hidden-factor-confirmation-independent-audit-v1'
            and audit['passed'] is True and audit['model_checks_passed'] is True
            and audit['full_reference_checked'] is True
            and audit['summary_sha256'] == sha(paths[0])
            and audit['consumer_sha256'] == sha(__file__), 'Confirmation lacks an unchanged passing independent audit')
    require(data['plan_sha256'] == audit['plan_sha256'] == sha(PLAN)
            and data['complete'] is True and data['scientific_results'] is True
            and data['test_role'] == 'untouched_confirmation'
            and data['evaluated_n_sequences'] == audit['evaluated_sequences'] == 3840
            and data['sequence_start'] == 256 and data['sequence_stop'] == 4096
            and data['planned_n_sequences'] == 4096 and data['dimensions'] == [2, 8]
            and data['times'] == TIMES.tolist(), 'Confirmation source or scope changed')
    groups = [g for g in data['groups'] if g['id'] == 'full_prior']
    require(len(groups) == 1 and groups[0]['status'] == 'complete' and groups[0]['n_seq'] == 3840,
            'Incomplete full-prior confirmation')
    predictors = groups[0]['predictors']

    def curve(values):
        mean, se = np.asarray(values['mean']), np.asarray(values['se'])
        require(values['n_seq'] == 3840 and mean.shape == se.shape == (6,)
                and np.isfinite(mean).all() and np.isfinite(se).all()
                and (mean >= 0).all() and (se >= 0).all(), 'Invalid confirmation curve')
        return dict(times=TIMES.tolist(), mean=mean.tolist(), se=se.tolist())

    cases = []
    for arm in ARMS:
        for seed in (0, 1):
            name = f'{arm}_seed{seed}'
            spec = plan['baseline'] if arm == 'baseline' else plan['arms'][arm]
            steps = plan['baseline']['final_step'] if arm == 'baseline' else plan['primary']['steps'][arm]
            cases.append(dict(label={'baseline': 'Baseline', 'longsmall': 'More training', 'large': 'Larger'}[arm],
                              arm=arm, color=COLORS[arm], condition=dict(dimensions=[2, 8], arm=arm, training_seed=seed),
                              checkpoint_sha256=data['model_receipts'][name]['identity']['pilot']['checkpoint_sha256'],
                              checkpoint_step=steps, n_params=spec['n_params'], training_sequences=steps*64,
                              n_sequences=3840, n_tokens=1025,
                              transformer=curve(predictors[name]['next_kl_bits'])))
    require(len({c['checkpoint_sha256'] for c in cases}) == 6, 'Confirmation reuses a checkpoint')
    panel = dict(id='c', title='Product source: hidden encoding', cases=cases,
                 bayes=curve(predictors['bayes_hidden']['next_kl_bits']), bayes_style='dashed black',
                 cohort_id=data['cohort_id'], pilot=False, test_role='untouched_confirmation',
                 evaluated_sequences=3840, planned_sequences=4096, excluded_development_sequences=256,
                 quantity='Full-prior expected next-token regret; same unknown encoding for Bayes and all six transformers.',
                 seed_rule='Training runs1/2 correspond to saved seeds0/1, shown separately. Main figure shows fixed run1. No pooling or selection by performance.',
                 checkpoint_scope='Final planned checkpoints; recorded identities match the validated pilot; raw weights not transferred.')
    return paths, data, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--cohort', type=Path, default=COHORT)
    parser.add_argument('--references', type=Path, default=REFERENCES)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--check-models-only', action='store_true', help='Verify model scores only; do not write paper results without reference validation.')
    parser.add_argument('--check-only', action='store_true', help='Audit the raw data without regenerating outputs.')
    args = parser.parse_args()
    summary, audit = validate(args.input, args.cohort, args.references, args.check_models_only)
    if not (args.check_models_only or args.check_only):
        write_results(summary, audit, args.out)
    print(json.dumps({k: audit[k] for k in ('passed', 'model_checks_passed', 'full_reference_checked', 'evaluated_sequences', 'metric_summaries_checked')}))


if __name__ == '__main__':
    main()
