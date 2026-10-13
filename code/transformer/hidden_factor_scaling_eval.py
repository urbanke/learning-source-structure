"""Paired evaluation of planned hidden-factor scaling arms on the frozen pilot.

No source generation, Bayesian reference computation, or training is performed.
The original pilot remains read-only. Every arm/seed has a separate output and
only the predeclared FINAL checkpoint can enter the comparison. The reused
pilot is development data, not untouched confirmation.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
from io import BytesIO
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from eval_hidden_factor import (file_hash, fingerprint, entropy_strata, read_complete,
                               checked_existing, metrics, moments, save_json, write_result)
from eval_mixture import check_causality
from train_hidden_factor import validate_config, source_contract, draw_validation
from model import GPT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE/'configs/hidden_factor_scaling_plan_2026-09-16.json'
SCHEMA = 'hidden-factor-scaling-evaluation-v1'
METRICS = ('next_kl_bits', 'next_loss_bits', 'cumulative_loss_bits')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_plan(path):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    require(plan.get('schema') == 'hidden-factor-scaling-plan-v1', 'Wrong scaling plan schema')
    require(plan['seeds'] == [0, 1] and set(plan['arms']) == {'longsmall', 'large'}, 'Both planned arms and seeds required')
    require(plan['test']['role'] == 'development', 'Reused pilot must be marked development data')
    require(2 <= plan['test']['n_evaluate'] < plan['test']['n_planned'], 'Expected a frozen pilot prefix')
    require(plan['test']['times'][-1] == plan['test']['length']-1, 'Context/scoring mismatch')
    require(set(plan['implementation_sha256']) == {'hidden_factor.py', 'model.py', 'train_hidden_factor.py', 'train_mixture.py'}, 'Missing original implementation bindings')
    for name, expected in plan['implementation_sha256'].items():
        require(file_hash(HERE/name) == expected, f'Original implementation changed: {name}')
    for arm in plan['arms']:
        require(plan['primary']['steps'][arm] == plan['arms'][arm]['config']['steps'], 'Primary checkpoint differs from final planned step')
    return plan, hashlib.sha256(raw).hexdigest()


def planned_config(plan, arm, seed):
    require(arm in plan['arms'] and seed in plan['seeds'], 'Unplanned arm/seed')
    return validate_config(dict(plan['arms'][arm]['config'], seed=seed, source_dimensions=plan['source_dimensions']))


@lru_cache(maxsize=16)
def monitoring_hashes(dimensions, n_ctx, n_sequences, seed):
    stream = draw_validation(dimensions, n_ctx, n_sequences, seed)
    return {name: hashlib.sha256(tokens.numpy().tobytes()).hexdigest() for name, tokens in stream.items()}


def validate_checkpoint(checkpoint, plan, plan_sha256, arm):
    """Independent plan-specific check; does not relax the original evaluator."""
    spec = plan['arms'][arm]
    schemas = ({'hidden-factor-scaling-weights-v1', 'hidden-factor-scaling-checkpoint-v1'}
               if spec['mode'] == 'continuation' else {'hidden-factor-weights-v1', 'hidden-factor-checkpoint-v1'})
    require(checkpoint.get('schema') in schemas, 'Wrong checkpoint schema for this arm')
    receipt = checkpoint['cfg']
    actual = receipt['config']
    seed = actual['seed']
    declared = planned_config(plan, arm, seed)
    provenance = actual.get('scaling_provenance')
    require(isinstance(provenance, dict), 'Missing scaling provenance')
    require({key: value for key, value in actual.items() if key != 'scaling_provenance'} == declared, 'Checkpoint configuration differs from the frozen arm')
    digest = fingerprint(actual)
    require(checkpoint.get('config_hash') == receipt.get('config_hash') == digest, 'Checkpoint configuration hash mismatch')
    require(provenance.get('plan_sha256') == plan_sha256 and provenance.get('arm') == arm
            and provenance.get('seed') == seed, 'Checkpoint belongs to another plan/arm/seed')
    require(provenance.get('trainer_sha256') == file_hash(HERE/'train_hidden_factor_scaling.py'), 'Scaling trainer changed')
    if 'scaling_provenance' in receipt:
        require(receipt['scaling_provenance'] == provenance, 'Conflicting scaling provenance')
    contract = source_contract(plan['source_dimensions'])
    require(receipt.get('source_contract') == contract and receipt.get('source_contract_hash') == fingerprint(contract), 'Source contract differs')
    require(checkpoint.get('implementation_hashes') == receipt.get('implementation_hashes') == plan['implementation_sha256'], 'Source/model/helper implementation drift')
    model = dict(vocab=math.prod(plan['source_dimensions']), n_ctx=declared['n_ctx'],
                 n_layer=declared['n_layer'], n_head=declared['n_head'], d_model=declared['d_model'], mlp=True, dropout=0.)
    require(receipt.get('model') == model and receipt.get('n_params') == spec['n_params'], 'Checkpoint architecture differs from plan')
    require(checkpoint.get('step') == declared['steps']
            and checkpoint.get('completed_sequences') == declared['steps']*declared['batch'], 'Only complete final checkpoints may be evaluated')
    if 'completed_batches' in checkpoint:
        require(checkpoint['completed_batches'] == declared['steps'], 'Incomplete training batches')
    require(declared['n_ctx'] == plan['test']['length']-1, 'Checkpoint context differs from frozen test')
    require(receipt.get('precision') == 'float32', 'Unexpected training precision')
    require(isinstance(receipt.get('validation_hashes'), dict) and bool(receipt['validation_hashes']), 'Missing monitoring stream identity')
    require(plan['test']['data_sha256'] not in receipt['validation_hashes'].values(), 'Test data was supplied as monitoring data')
    require(receipt['validation_hashes'] == monitoring_hashes(tuple(plan['source_dimensions']), declared['n_ctx'],
            declared['val_n'], declared['validation_seed']), 'Monitoring data differs from the original separate source stream')
    parent = provenance.get('parent')
    if spec['mode'] == 'continuation':
        require(isinstance(parent, dict), 'Continuation ancestry is missing')
        require(parent.get('schema') == 'hidden-factor-checkpoint-v1', 'Continuation did not use original full checkpoint')
        base_config = validate_config(dict(spec['parent_config'], seed=seed, source_dimensions=plan['source_dimensions']))
        require(parent.get('config_hash') == fingerprint(base_config)
                and parent.get('step') == spec['parent_step']
                and parent.get('completed_sequences') == spec['parent_completed_sequences'], 'Wrong continuation starting point')
        require(parent.get('weights_checkpoint_sha256') == plan['baseline']['checkpoint_sha256'][str(seed)], 'Continuation did not start from the scored baseline weights')
        require(isinstance(parent.get('checkpoint_sha256'), str) and re.fullmatch('[0-9a-f]{64}', parent['checkpoint_sha256']) is not None, 'Missing full parent checkpoint hash')
        require(parent.get('implementation_hashes') == plan['implementation_sha256'], 'Parent source/model drift')
        require(parent.get('validation_hashes') == receipt['validation_hashes'], 'Continuation changed monitoring stream')
        require(parent.get('runtime') == receipt.get('runtime'), 'Continuation changed numerical runtime')
        require(receipt.get('learning_rate_schedule') == dict(kind='constant continuation',
            value=spec['continuation_lr'], starts_after_step=spec['parent_step']), 'Continuation learning-rate receipt differs from plan')
    else:
        require(spec['mode'] == 'fresh' and parent is None, 'Fresh architecture arm has unexpected ancestry')
    return declared, model, provenance


def load_frozen_baseline(directory, plan):
    """Verify original files and all cached references/scores, without writing."""
    directory = Path(directory)
    cohort = json.loads((directory/'cohort.json').read_bytes())
    raw_data = (directory/'cohort.npz').read_bytes()
    require(cohort.get('schema') == 'hidden-factor-evaluation-v1'
            and fingerprint(cohort['identity']) == cohort['cohort_id']
            and hashlib.sha256(raw_data).hexdigest() == cohort['data_sha256'], 'Frozen cohort checksum/identity mismatch')
    with np.load(BytesIO(raw_data), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    test, spec = plan['test'], cohort['identity']
    require(cohort['cohort_id'] == test['cohort_id'] and cohort['data_sha256'] == test['data_sha256'], 'Wrong frozen test cohort')
    require(spec['dimensions'] == plan['source_dimensions'] and spec['n_seq'] == test['n_planned']
            and spec['length'] == test['length'] and spec['times'] == test['times']
            and spec['first_token_scored'] is False, 'Source/test/scoring mismatch')
    require(spec['source_sha256'] == plan['implementation_sha256']['hidden_factor.py'], 'Test source changed')
    require(data['tokens'].shape == (test['n_planned'], test['length'])
            and np.array_equal(data['times'], test['times']), 'Frozen cohort shape/times mismatch')
    entropy, medians, strata = entropy_strata((data['first_matrix'], data['second_matrix']))
    # The archived arrays remain byte-for-byte hash bound. Recomputing log2
    # across CPU platforms can differ by one ulp; only this float audit has
    # tolerance. Stratum membership and sequence order remain exact.
    require(np.allclose(entropy, data['normalized_row_entropy'], rtol=0, atol=1e-12)
            and np.allclose(medians, cohort['entropy_medians'], rtol=0, atol=1e-12)
            and np.array_equal(strata, data['stratum'])
            and np.bincount(strata, minlength=4).tolist() == cohort['stratum_counts'],
            'Frozen source-only entropy strata changed')
    summary_path = directory/f"summary_prefix_{test['n_evaluate']:06d}.json"
    require(file_hash(summary_path) == plan['baseline']['summary_sha256'], 'Original pilot summary changed')
    reference, ref_hash = read_complete(directory, 'references', test['n_evaluate'], test['cohort_id'], test['n_planned'])
    require(ref_hash == plan['baseline']['reference_records_sha256'], 'Frozen Bayesian references changed')
    baseline, manifests = {}, {}
    for seed in plan['seeds']:
        name = f'seed{seed}'
        arrays, digest = read_complete(directory, 'models/'+name, test['n_evaluate'], test['cohort_id'], test['n_planned'])
        require(digest == plan['baseline']['model_records_sha256'][name], 'Original transformer scores changed')
        manifest = json.loads((directory/'models'/name/'manifest.json').read_bytes())
        ident = manifest['identity']
        require(ident['name'] == name and ident['checkpoint_step'] == plan['baseline']['final_step']
                and ident['checkpoint_sha256'] == plan['baseline']['checkpoint_sha256'][str(seed)], 'Baseline checkpoint mismatch')
        require(manifest['causality_check']['passed'] is True, 'Baseline causal check missing')
        baseline[name], manifests[name] = arrays, manifest
    return dict(cohort=cohort, data=data, references=reference, reference_hash=ref_hash,
                baseline=baseline, baseline_manifests=manifests)


def stage_inputs(baseline, target):
    """Only create isolated output directories; all frozen input stays read-only."""
    baseline, target = Path(baseline).resolve(), Path(target).resolve()
    require(target != baseline and baseline not in target.parents and target not in baseline.parents, 'Scaling output must be separate from baseline evaluation')
    target.mkdir(parents=True, exist_ok=True)


def evaluate(plan_path, baseline_directory, arm, checkpoint_path, output, device='cpu', batch_size=16):
    started = time.monotonic()
    plan, plan_sha = read_plan(plan_path)
    require(device in ('cpu', 'cuda') and batch_size > 0, 'Invalid device or batch size')
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable')
    require(arm in plan['arms'], 'Unknown arm')
    raw_checkpoint = Path(checkpoint_path).read_bytes()
    checkpoint_sha = hashlib.sha256(raw_checkpoint).hexdigest()
    checkpoint = torch.load(BytesIO(raw_checkpoint), map_location='cpu', weights_only=False)
    config, model_config, provenance = validate_checkpoint(checkpoint, plan, plan_sha, arm)
    frozen = load_frozen_baseline(baseline_directory, plan)
    target = Path(output)/arm
    stage_inputs(baseline_directory, target)
    name, n = f"seed{config['seed']}", plan['test']['n_evaluate']
    identity = dict(schema=SCHEMA, plan_sha256=plan_sha, arm=arm, name=name,
        cohort_id=plan['test']['cohort_id'], test_role='development', reference_records_sha256=frozen['reference_hash'],
        checkpoint_sha256=checkpoint_sha, checkpoint_step=checkpoint['step'],
        checkpoint_schema=checkpoint['schema'], checkpoint_receipt=checkpoint['cfg'],
        completed_sequences=checkpoint['completed_sequences'], config_hash=checkpoint['config_hash'],
        model=model_config, n_params=plan['arms'][arm]['n_params'], scaling_provenance=provenance,
        original_implementation_sha256=plan['implementation_sha256'], evaluator_sha256=file_hash(__file__),
        device=device, torch_version=str(torch.__version__), batch_size=batch_size, inference_precision='float32; log-softmax float64')
    model_dir = target/'models'/name
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir/'.evaluation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another evaluator holds this arm/seed') from error
        manifest_path = model_dir/'manifest.json'
        if manifest_path.exists():
            require(json.loads(manifest_path.read_bytes())['identity'] == identity, 'Existing evaluation belongs to different checkpoint/settings')
        model = GPT(**model_config)
        require(model.n_params() == plan['arms'][arm]['n_params'], 'Loaded model size differs')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(device).eval()
        times = np.asarray(plan['test']['times'])
        tokens = frozen['data']['tokens'][:n]
        causal = check_causality(model, torch.as_tensor(tokens[:1], dtype=torch.long, device=device), times)
        def write_manifest(completed):
            save_json(manifest_path, dict(schema=SCHEMA, identity=identity, n_seq=completed,
                evaluated_n_sequences=completed, planned_n_sequences=plan['test']['n_planned'],
                requested_n_sequences=n, complete=False, partial_evaluation=True,
                complete_for_requested_pilot=(completed == n), causality_check=causal,
                elapsed_evaluation_seconds=time.monotonic()-started))
        write_manifest(0)
        for start in range(0, n, batch_size):
            stop = min(start+batch_size, n)
            missing = [i for i in range(start, stop) if checked_existing(model_dir/f'seq_{i:06d}.npz', dict(identity, sequence_index=i), recover_orphan=True) is None]
            if missing:
                require(file_hash(__file__) == identity['evaluator_sha256'], 'Evaluator changed during inference')
                selected = tokens[missing]
                with torch.inference_mode():
                    logp = torch.log_softmax(model(torch.as_tensor(selected[:, :-1], dtype=torch.long, device=device)).double(), dim=-1).cpu().numpy()/math.log(2)
                realized = -np.take_along_axis(logp, selected[:, 1:, None], axis=-1).squeeze(-1)
                cumulative = np.cumsum(realized, axis=-1)[:, times-1]
                for j, i in enumerate(missing):
                    probabilities = np.exp2(logp[j, times-1])
                    values = metrics(probabilities, frozen['references']['oracle_probs'][i], tokens[i], times, cumulative[j])
                    values.update(predictive=probabilities, times=times)
                    write_result(model_dir/f'seq_{i:06d}.npz', dict(identity, sequence_index=i), values)
            write_manifest(stop)
            print(json.dumps(dict(arm=arm, seed=config['seed'], evaluated=stop, requested=n)), flush=True)
    return json.loads(manifest_path.read_bytes())


def summarize(plan_path, baseline_directory, evaluations, output):
    plan, plan_sha = read_plan(plan_path)
    frozen = load_frozen_baseline(baseline_directory, plan)
    require(Path(baseline_directory).resolve() not in Path(output).resolve().parents, 'Summary must not write into original baseline')
    test = plan['test']
    n = test['n_evaluate']
    predictors = {'bayes_hidden': {metric: frozen['references']['bayes_hidden_'+metric] for metric in METRICS}}
    predictors.update({'baseline_'+name: {metric: arrays[metric] for metric in METRICS} for name, arrays in frozen['baseline'].items()})
    provenance, hashes = {}, {}
    pairs = [(f'baseline_seed{seed}', 'bayes_hidden') for seed in plan['seeds']]
    for arm in plan['arms']:
        for seed in plan['seeds']:
            name = f'seed{seed}'
            target = Path(evaluations)/arm
            path = target/'models'/name/'manifest.json'
            require(path.is_file(), f'Missing final inference: {arm}/{name}')
            manifest = json.loads(path.read_bytes())
            ident = manifest['identity']
            require(manifest.get('complete_for_requested_pilot') is True
                    and manifest['evaluated_n_sequences'] == n and manifest['requested_n_sequences'] == n, 'Incomplete final inference')
            require(ident.get('schema') == SCHEMA and ident.get('plan_sha256') == plan_sha
                    and ident.get('arm') == arm and ident.get('name') == name and ident.get('test_role') == 'development'
                    and ident.get('checkpoint_step') == plan['arms'][arm]['config']['steps']
                    and ident.get('reference_records_sha256') == frozen['reference_hash']
                    and ident.get('evaluator_sha256') == file_hash(__file__), 'Inference identity mismatch')
            metadata_checkpoint = dict(schema=ident['checkpoint_schema'], cfg=ident['checkpoint_receipt'],
                config_hash=ident['config_hash'], step=ident['checkpoint_step'], completed_sequences=ident['completed_sequences'],
                implementation_hashes=ident['original_implementation_sha256'])
            config, expected_model, expected_provenance = validate_checkpoint(metadata_checkpoint, plan, plan_sha, arm)
            require(config['seed'] == seed and ident['model'] == expected_model
                    and ident['n_params'] == plan['arms'][arm]['n_params']
                    and ident['scaling_provenance'] == expected_provenance, 'Saved model metadata drift')
            require(manifest['causality_check']['passed'] is True
                    and manifest['causality_check']['max_absolute_logit_difference'] <= 2e-5, 'Missing or failed causality check')
            arrays, digest = read_complete(target, 'models/'+name, n, test['cohort_id'], test['n_planned'])
            require(np.array_equal(arrays['times'], np.tile(test['times'], (n, 1))), 'Saved inference uses different times')
            for i in range(n):
                replay = metrics(arrays['predictive'][i], frozen['references']['oracle_probs'][i],
                                 frozen['data']['tokens'][i], np.asarray(test['times']), arrays['cumulative_loss_bits'][i])
                for metric in ('next_kl_bits', 'next_loss_bits'):
                    require(np.allclose(replay[metric], arrays[metric][i], atol=1e-11, rtol=0), 'Saved inference scores differ from predictions')
            label = f'{arm}_{name}'
            predictors[label] = {metric: arrays[metric] for metric in METRICS}
            pairs += [(label, 'bayes_hidden'), (label, f'baseline_{name}')]
            provenance[label], hashes[label] = ident, digest
    groups = []
    masks = [('full_prior', np.ones(n, dtype=bool), None)]
    masks += [(f'entropy_{i}', frozen['data']['stratum'][:n] == i, label) for i, label in enumerate(frozen['cohort']['stratum_labels'])]
    for group_id, mask, label in masks:
        group = dict(id=group_id, entropy_label=label, n_seq=int(mask.sum()))
        if mask.sum() < 2:
            groups.append(dict(group, status='insufficient_sample'))
            continue
        group.update(status='complete', predictors={}, paired={})
        for name, values in predictors.items():
            group['predictors'][name] = {metric: moments(values[metric][mask]) for metric in METRICS}
        for left, right in pairs:
            group['paired'][left+'_minus_'+right] = {metric: moments((predictors[left][metric]-predictors[right][metric])[mask]) for metric in METRICS}
        groups.append(group)
    result = dict(schema='hidden-factor-scaling-comparison-v1', plan_sha256=plan_sha,
        test_role='development', scientific_scope='Paired follow-up on the same pilot that motivated these arms; not untouched confirmation.',
        cohort_id=test['cohort_id'], cohort_data_sha256=test['data_sha256'], dimensions=plan['source_dimensions'],
        evaluated_n_sequences=n, planned_n_sequences=test['n_planned'], sequence_length=test['length'], times=test['times'],
        complete_for_requested_comparison=True, full_cohort_complete=False, baseline=plan['baseline'],
        primary=plan['primary'], groups=groups, model_records_sha256=hashes, model_provenance=provenance,
        reference_records_sha256=frozen['reference_hash'], new_bayesian_computation=False,
        uncertainty='One standard error across independent whole sequences; paired differences; training seeds reported separately.',
        improvement_sign='Negative arm_minus_baseline means improvement; negative arm_minus_Bayes may occur from finite-sample variation.',
        interpretation={arm: spec['interpretation'] for arm, spec in plan['arms'].items()})
    save_json(output, result)
    return result


def plot_comparison(summary, directory):
    """Render only the validated complete comparison; no predictor is evaluated."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    require(summary.get('complete_for_requested_comparison') is True, 'Cannot plot an incomplete comparison')
    group = next(group for group in summary['groups'] if group['id'] == 'full_prior')
    times = np.asarray(summary['times'])
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    colors = ('#2468A0', '#288167')
    def line(ax, values, label, color, style='-'):
        mean, se = np.asarray(values['mean']), np.asarray(values['se'])
        ax.plot(times, mean, label=label, color=color, linestyle=style, linewidth=1.4)
        ax.fill_between(times, mean-se, mean+se, color=color, alpha=.12, linewidth=0)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(3, 2, figsize=(8, 7), sharex=True)
        for col, (arm, title) in enumerate((('longsmall', 'More training, same architecture'), ('large', 'Larger architecture, same sample count'))):
            line(axes[0, col], group['predictors']['bayes_hidden']['next_kl_bits'], 'Informed Bayes', '#BF611C', '--')
            for seed, color in enumerate(colors):
                name, baseline = f'{arm}_seed{seed}', f'baseline_seed{seed}'
                line(axes[0, col], group['predictors'][name]['next_kl_bits'], f'Arm seed {seed}', color)
                line(axes[0, col], group['predictors'][baseline]['next_kl_bits'], f'Original seed {seed}', '#777777', '--' if seed == 0 else ':')
                line(axes[1, col], group['paired'][name+'_minus_bayes_hidden']['cumulative_loss_bits'], f'Seed {seed}', color)
                line(axes[2, col], group['paired'][name+'_minus_'+baseline]['cumulative_loss_bits'], f'Seed {seed}', color)
            axes[0, col].set_title(title)
            for row in (1, 2):
                axes[row, col].axhline(0, color='#777777', linewidth=.6)
            for row in range(3):
                ax = axes[row, col]
                ax.set_xscale('log', base=2)
                ax.set_xticks(times, [str(t) for t in times])
                ax.minorticks_off()
                ax.grid(axis='y', color='#D8DEE5', linewidth=.5)
            axes[2, col].set_xlabel('Observed tokens, t')
        axes[0, 0].set_ylabel('Next-token regret (bits)')
        axes[1, 0].set_ylabel('Cumulative arm minus Bayes (bits)')
        axes[2, 0].set_ylabel('Cumulative arm minus original (bits)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, fontsize=7)
        fig.text(.5, .012, f"Development pilot: same {summary['evaluated_n_sequences']} sequences; shaded bands are one paired SE in difference panels.", ha='center', fontsize=7)
        fig.tight_layout(rect=(0, .038, 1, .955))
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'comparison.{extension}', dpi=200)
        plt.close(fig)
    save_json(out/'receipt.json', dict(schema='hidden-factor-scaling-figure-v1', summary_sha256=fingerprint(summary),
        evaluator_sha256=file_hash(__file__), test_role='development',
        outputs={name: file_hash(out/name) for name in ('comparison.png', 'comparison.pdf')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('model', 'summary'):
        command = commands.add_parser(name)
        command.add_argument('--plan', type=Path, default=PLAN)
        command.add_argument('--baseline-eval', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
    model = commands.choices['model']
    model.add_argument('--arm', required=True, choices=('longsmall', 'large'))
    model.add_argument('--checkpoint', type=Path, required=True)
    model.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    model.add_argument('--batch-size', type=int, default=16)
    commands.choices['summary'].add_argument('--evaluations', type=Path, required=True)
    commands.choices['summary'].add_argument('--figure-out', type=Path)
    args = parser.parse_args()
    if args.command == 'model':
        result = evaluate(args.plan, args.baseline_eval, args.arm, args.checkpoint, args.out, args.device, args.batch_size)
        print(json.dumps(dict(complete=result['complete_for_requested_pilot'], records=result['evaluated_n_sequences'])))
    else:
        result = summarize(args.plan, args.baseline_eval, args.evaluations, args.out)
        if args.figure_out:
            plot_comparison(result, args.figure_out)
        print(json.dumps(dict(complete=result['complete_for_requested_comparison'], groups=len(result['groups']))))


if __name__ == '__main__':
    main()
