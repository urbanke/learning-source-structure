"""Score the six fixed hidden-encoding models on the untouched cohort suffix.

Consumes the completed exact references and the original scaling pilot. No
training, new evaluation sequences, or Bayesian inference is performed. Original
inputs stay read-only; completed compact model files can be reused on restart.
Run on a compute node: verifying the saved reference records entails bulk I/O.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import eval_hidden_factor as original
import hidden_factor_scaling_eval as scaling
from eval_mixture import check_causality
from model import GPT

SCHEMA = 'hidden-factor-confirmation-v1'
METRICS = scaling.METRICS
REFERENCES = ('bayes_hidden', 'bayes_encoding_given', 'flat_dirichlet', 'oracle')
require = scaling.require


def _paths(baseline_root, scaling_root, plan):
    paths = {}
    for seed in plan['seeds']:
        paths[f'baseline_seed{seed}'] = Path(baseline_root)/f'train/seed{seed}'/f"model_{plan['baseline']['final_step']}.pt"
        for arm in plan['arms']:
            paths[f'{arm}_seed{seed}'] = Path(scaling_root)/f'train/{arm}/seed{seed}'/f"model_{plan['primary']['steps'][arm]}.pt"
    return paths


def load_inputs(plan_path, baseline_root, scaling_root):
    """Validate the frozen pilot and the full references without changing either."""
    plan, plan_sha = scaling.read_plan(plan_path)
    baseline_eval = Path(baseline_root)/'eval'
    frozen = scaling.load_frozen_baseline(baseline_eval, plan)
    test = plan['test']
    first, stop = test['n_evaluate'], test['n_planned']
    require(stop-first >= 2, 'At least two untouched confirmation sequences required')
    reference, reference_hash = original.read_complete(baseline_eval, 'references', stop, test['cohort_id'])
    manifest = json.loads((baseline_eval/'references/manifest.json').read_text())
    ident = manifest['identity']
    require(ident['kind'] == 'reference'
            and ident['source_sha256'] == plan['implementation_sha256']['hidden_factor.py']
            and ident['evaluator_sha256'] == original.file_hash(original.__file__)
            and ident['reference_sha256'] == original.file_hash(scaling.HERE/'hidden_factor_gpu.py'),
            'Full reference implementation differs from the frozen source/evaluator')
    require(np.array_equal(reference['times'], np.tile(test['times'], (stop, 1))), 'Reference times differ')
    pilots = {}
    for seed in plan['seeds']:
        name = f'baseline_seed{seed}'
        pilot_manifest = frozen['baseline_manifests'][f'seed{seed}']
        pilots[name] = dict(checkpoint_sha256=plan['baseline']['checkpoint_sha256'][str(seed)],
                            records_sha256=plan['baseline']['model_records_sha256'][f'seed{seed}'],
                            manifest_sha256=original.fingerprint(pilot_manifest))
        for arm in plan['arms']:
            path = Path(scaling_root)/f'eval/{arm}/models/seed{seed}/manifest.json'
            pilot = json.loads(path.read_text())
            identity = pilot['identity']
            require(pilot.get('complete_for_requested_pilot') is True
                    and pilot['evaluated_n_sequences'] == first and pilot['requested_n_sequences'] == first
                    and identity['schema'] == scaling.SCHEMA and identity['test_role'] == 'development'
                    and identity['plan_sha256'] == plan_sha and identity['arm'] == arm
                    and identity['name'] == f'seed{seed}'
                    and identity['checkpoint_step'] == plan['primary']['steps'][arm]
                    and identity['reference_records_sha256'] == frozen['reference_hash']
                    and identity['evaluator_sha256'] == original.file_hash(scaling.__file__),
                    'Scaling pilot identity differs from the frozen comparison')
            metadata = dict(schema=identity['checkpoint_schema'], cfg=identity['checkpoint_receipt'],
                            config_hash=identity['config_hash'], step=identity['checkpoint_step'],
                            completed_sequences=identity['completed_sequences'],
                            implementation_hashes=identity['original_implementation_sha256'])
            config, model, provenance = scaling.validate_checkpoint(metadata, plan, plan_sha, arm)
            require(config['seed'] == seed and identity['model'] == model
                    and identity['scaling_provenance'] == provenance, 'Scaling pilot model metadata changed')
            require(pilot['causality_check']['passed'] is True
                    and pilot['causality_check']['max_absolute_logit_difference'] <= 2e-5,
                    'Scaling pilot causal check failed')
            arrays, digest = original.read_complete(Path(scaling_root)/f'eval/{arm}', f'models/seed{seed}', first,
                                                    test['cohort_id'], stop)
            require(np.array_equal(arrays['times'], np.tile(test['times'], (first, 1))), 'Pilot times differ')
            for i in range(first):
                replay = original.metrics(arrays['predictive'][i], reference['oracle_probs'][i],
                                          frozen['data']['tokens'][i], test['times'], arrays['cumulative_loss_bits'][i])
                for metric in ('next_kl_bits', 'next_loss_bits'):
                    require(np.allclose(replay[metric], arrays[metric][i], rtol=0, atol=1e-11), 'Pilot metric replay failed')
            pilots[f'{arm}_seed{seed}'] = dict(checkpoint_sha256=identity['checkpoint_sha256'],
                                               records_sha256=digest, manifest_sha256=original.file_hash(path))
    return dict(plan=plan, plan_sha256=plan_sha, frozen=frozen, references=reference,
                reference_records_sha256=reference_hash, pilots=pilots,
                checkpoints=_paths(baseline_root, scaling_root, plan),
                indices=np.arange(first, stop, dtype=np.int64), times=np.asarray(test['times'], dtype=np.int64))


def score_log_probabilities(logp_bits, tokens, oracle, times):
    """Observed prefix t predicts token t+1; cumulative loss excludes token1."""
    require(logp_bits.shape[:2] == (len(tokens), tokens.shape[1]-1), 'Inference/token shape mismatch')
    require(np.isfinite(logp_bits).all(), 'Nonfinite model output')
    realized = -np.take_along_axis(logp_bits, tokens[:, 1:, None], axis=-1).squeeze(-1)
    cumulative = np.cumsum(realized, axis=1)[:, times-1]
    predictive = np.exp2(logp_bits[:, times-1])
    rows = [original.metrics(predictive[i], oracle[i], tokens[i], times, cumulative[i]) for i in range(len(tokens))]
    result = {metric: np.stack([row[metric] for row in rows]) for metric in METRICS}
    result['predictive'] = predictive
    return result


def model_identity(inputs, name):
    test = inputs['plan']['test']
    return dict(schema=SCHEMA, name=name, test_role='untouched_confirmation',
                plan_sha256=inputs['plan_sha256'], cohort_id=test['cohort_id'],
                cohort_data_sha256=test['data_sha256'], sequence_start=int(inputs['indices'][0]),
                sequence_stop=int(inputs['indices'][-1]+1), times=test['times'], first_token_scored=False,
                reference_records_sha256=inputs['reference_records_sha256'], pilot=inputs['pilots'][name],
                evaluator_sha256=original.file_hash(__file__),
                scaling_evaluator_sha256=original.file_hash(scaling.__file__),
                original_evaluator_sha256=original.file_hash(original.__file__))


def _load_model_result(inputs, name, out):
    path = Path(out)/'models'/f'{name}.npz'
    meta_path = path.with_suffix('.json')
    require(path.exists() and meta_path.exists(), f'Missing confirmation scores: {name}')
    receipt = json.loads(meta_path.read_text())
    require(receipt['identity'] == model_identity(inputs, name) and receipt.get('complete') is True,
            'Confirmation identity differs from the fixed inputs')
    require(receipt['data_sha256'] == original.file_hash(path), 'Confirmation score checksum mismatch')
    require(receipt['causality_check']['passed'] is True
            and receipt['causality_check']['max_absolute_logit_difference'] <= 2e-5, 'Confirmation causal check failed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(np.array_equal(arrays['sequence_indices'], inputs['indices'])
            and np.array_equal(arrays['times'], inputs['times']), 'Confirmation sequence pairing/times changed')
    n, nt = len(inputs['indices']), len(inputs['times'])
    for metric in METRICS:
        require(arrays[metric].shape == (n, nt) and np.isfinite(arrays[metric]).all(), 'Invalid confirmation metric array')
    require(arrays['predictive'].shape == (n, nt, math.prod(inputs['plan']['source_dimensions'])), 'Invalid prediction shape')
    for j, i in enumerate(inputs['indices']):
        replay = original.metrics(arrays['predictive'][j], inputs['references']['oracle_probs'][i],
                                  inputs['frozen']['data']['tokens'][i], inputs['times'], arrays['cumulative_loss_bits'][j])
        for metric in ('next_kl_bits', 'next_loss_bits'):
            require(np.allclose(replay[metric], arrays[metric][j], atol=1e-11, rtol=0), 'Confirmation metric replay failed')
    return arrays, receipt


def evaluate_model(inputs, name, out, device='cpu', batch_size=16):
    require(name in inputs['checkpoints'], 'Unknown confirmation model')
    require(device in ('cpu', 'cuda') and batch_size > 0, 'Invalid inference device or batch size')
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    checkpoint_path = inputs['checkpoints'][name]
    raw = checkpoint_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == inputs['pilots'][name]['checkpoint_sha256'],
            'Checkpoint differs from the model evaluated in the pilot')
    checkpoint = torch.load(BytesIO(raw), map_location='cpu', weights_only=False)
    arm, seed_string = name.rsplit('_seed', 1)
    if arm == 'baseline':
        config, model_config = original.validate_checkpoint(checkpoint, inputs['frozen']['cohort']['identity'])
    else:
        config, model_config, _ = scaling.validate_checkpoint(checkpoint, inputs['plan'], inputs['plan_sha256'], arm)
    require(config['seed'] == int(seed_string), 'Checkpoint training seed differs')
    path = Path(out)/'models'/f'{name}.npz'
    # A complete model is reused only after validating its source/model bindings.
    if path.exists() and path.with_suffix('.json').exists():
        _, receipt = _load_model_result(inputs, name, out)
        return receipt
    started = time.monotonic()
    model = GPT(**model_config)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    tokens = inputs['frozen']['data']['tokens'][inputs['indices']]
    oracle = inputs['references']['oracle_probs'][inputs['indices']]
    causal = check_causality(model, torch.as_tensor(tokens[:1], dtype=torch.long, device=device), inputs['times'])
    require(causal['passed'] is True and causal['max_absolute_logit_difference'] <= 2e-5, 'Confirmation causal check failed')
    chunks = []
    for start in range(0, len(tokens), batch_size):
        stop = min(start+batch_size, len(tokens))
        with torch.inference_mode():
            tensor = torch.as_tensor(tokens[start:stop, :-1], dtype=torch.long, device=device)
            logp = (torch.log_softmax(model(tensor).double(), dim=-1)/math.log(2)).cpu().numpy()
        chunks.append(score_log_probabilities(logp, tokens[start:stop], oracle[start:stop], inputs['times']))
        if stop == len(tokens) or stop % (batch_size*16) == 0:
            print(json.dumps(dict(stage='model', name=name, completed=stop, total=len(tokens))), flush=True)
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    arrays.update(sequence_indices=inputs['indices'], times=inputs['times'])
    original.save_npz(path, arrays)
    receipt = dict(identity=model_identity(inputs, name), complete=True, n_sequences=len(tokens),
                   data_sha256=original.file_hash(path), causality_check=causal,
                   inference=dict(device=device, batch_size=batch_size, torch_version=str(torch.__version__),
                                  precision='float32; log-softmax float64'), elapsed_seconds=time.monotonic()-started)
    original.save_json(path.with_suffix('.json'), receipt)
    return receipt


def summarize(inputs, out):
    indices = inputs['indices']
    predictors = {name: {metric: inputs['references'][f'{name}_{metric}'][indices] for metric in METRICS}
                  for name in REFERENCES}
    receipts = {}
    for name in inputs['checkpoints']:
        arrays, receipt = _load_model_result(inputs, name, out)
        predictors[name] = {metric: arrays[metric] for metric in METRICS}
        receipts[name] = receipt
    pairs = [('bayes_hidden', 'bayes_encoding_given'), ('flat_dirichlet', 'bayes_hidden')]
    for name in receipts:
        pairs.append((name, 'bayes_hidden'))
        if not name.startswith('baseline_'):
            pairs.append((name, 'baseline_seed'+name.rsplit('_seed', 1)[1]))
    cohort = inputs['frozen']['cohort']
    strata = inputs['frozen']['data']['stratum'][indices]
    masks = [('full_prior', np.ones(len(indices), dtype=bool), None)]
    masks += [(f'entropy_{i}', strata == i, label) for i, label in enumerate(cohort['stratum_labels'])]
    groups = []
    for name, mask, label in masks:
        group = dict(id=name, entropy_label=label, n_seq=int(mask.sum()), status='complete', predictors={}, paired={})
        if mask.sum() < 2:
            group['status'] = 'insufficient_sample'
        else:
            group['predictors'] = {name: {metric: original.moments(values[mask]) for metric, values in metrics.items()}
                                   for name, metrics in predictors.items()}
            group['paired'] = {f'{left}_minus_{right}': {metric: original.moments((predictors[left][metric]-predictors[right][metric])[mask])
                                                       for metric in METRICS} for left, right in pairs}
        groups.append(group)
    result = dict(schema=SCHEMA, complete=True, scientific_results=True, test_role='untouched_confirmation',
                  scientific_scope='Fixed final models on the unused suffix of the original frozen cohort; the development prefix is excluded.',
                  plan_sha256=inputs['plan_sha256'], cohort_id=cohort['cohort_id'], cohort_data_sha256=cohort['data_sha256'],
                  sequence_start=int(indices[0]), sequence_stop=int(indices[-1]+1), evaluated_n_sequences=len(indices),
                  planned_n_sequences=inputs['plan']['test']['n_planned'], development_n_sequences=int(indices[0]),
                  sequence_length=inputs['plan']['test']['length'], dimensions=inputs['plan']['source_dimensions'],
                  times=inputs['times'].tolist(), first_token_scored=False,
                  primary=inputs['plan']['primary'], groups=groups, model_receipts=receipts,
                  reference_records_sha256=inputs['reference_records_sha256'],
                  new_training=False, new_bayesian_computation=False,
                  uncertainty='One standard error across independent confirmation sequences; differences are paired; training seeds remain separate.')
    original.save_json(Path(out)/'summary.json', result)
    return result


def run(plan_path, baseline_root, scaling_root, out, stage='all', device='cpu', batch_size=16):
    require(stage in ('all', 'models', 'summary'), 'Unknown stage')
    out = Path(out).resolve()
    for source in (baseline_root, scaling_root):
        source = Path(source).resolve()
        require(out != source and out not in source.parents and source not in out.parents,
                'Confirmation output must be separate from both original experiment roots')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.confirmation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another confirmation process holds this output') from error
        original.save_json(out/'status.json', dict(schema=SCHEMA, status='checking_inputs', complete=False))
        try:
            inputs = load_inputs(plan_path, baseline_root, scaling_root)
            completed = []
            if stage in ('all', 'models'):
                for name in inputs['checkpoints']:
                    original.save_json(out/'status.json', dict(schema=SCHEMA, status='scoring', complete=False,
                                       current_model=name, completed_models=completed))
                    evaluate_model(inputs, name, out, device, batch_size)
                    completed.append(name)
            result = summarize(inputs, out) if stage in ('all', 'summary') else None
            status = dict(schema=SCHEMA, status='completed' if result else 'models_completed', complete=bool(result),
                          evaluated_n_sequences=len(inputs['indices']), completed_models=list(inputs['checkpoints']))
            original.save_json(out/'status.json', status)
            return result or status
        except Exception as error:
            original.save_json(out/'status.json', dict(schema=SCHEMA, status='failed', complete=False, error=str(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'models', 'summary'), default='all')
    parser.add_argument('--plan', type=Path, default=scaling.PLAN)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--scaling-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    torch.set_num_threads(1)
    result = run(args.plan, args.baseline_root, args.scaling_root, args.out, args.stage, args.device, args.batch_size)
    print(json.dumps(dict(complete=result.get('complete'), evaluated_n_sequences=result['evaluated_n_sequences'])), flush=True)


if __name__ == '__main__':
    main()
