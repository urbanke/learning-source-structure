"""Compare explicitly selected campaign checkpoints on the reused 3840 cases.

All reference/weight reads, hashes, inference and aggregation belong on a compute
node. No training, Bayesian inference or fresh confirmation is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

import hidden_factor_campaign_eval as campaign_eval
import hidden_factor_confirmation_eval as previous
import eval_hidden_factor as original
from eval_mixture import check_causality
from hidden_factor_campaign_train import PLAN, ROOT, read_plan, require, sha
from model import GPT

SCHEMA = 'hidden-factor-campaign-comparison-v1'
METRICS = previous.METRICS
SELECTIONS = {'continue_large': (400000,), 'xlarge': (200000, 400000)}


def selection(arm, step):
    require(arm in SELECTIONS and step in SELECTIONS[arm],
            'Select continue_large at 400000 or xlarge at 200000/400000 explicitly')


def protect_output(out, sources):
    out = Path(out).resolve()
    protected = [Path(p).resolve() for p in sources]
    protected += [ROOT/name for name in ('code', 'results', 'notes', 'figures', '.git')]
    require(out != ROOT and out not in ROOT.parents
            and not any(out == p or out in p.parents or p in out.parents for p in protected),
            'Comparison output must be separate from original inputs and source files')


@contextmanager
def output_claim(out, arm):
    """Atomic directory ownership on Slurm, local flock only for CPU tests.

Reuse the tested claim/terminal-accounting primitives without changing the
training launchers. A killed Slurm owner requires confirmed terminal accounting
before a later job can acquire a new generation. Unknown owners fail closed.
    """
    out.mkdir(parents=True, exist_ok=True)
    if not os.environ.get('SLURM_JOB_ID'):
        with (out/'.evaluation.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        return
    sys.path.insert(0, str(ROOT/'code/scripts'))
    import campaign_job_guard as guard
    # One key protects the whole output, even for direct CLI callers that
    # accidentally request different arms in the same directory.
    claim = guard.reserve(out, 'hidden_factor', ('eval', 'xlarge', '0'), uuid.uuid4().hex)
    job = os.environ['SLURM_JOB_ID']
    guard.write(claim/'allocation.json', dict(job_id=job))
    (claim/'active').mkdir()
    guard.write(claim/'started.json', dict(job_id=job, pid=os.getpid(), started_unix=time.time()))
    code = 1
    try:
        yield
        code = 0
    finally:
        guard.write(claim/'finished.json', dict(exit_code=code, finished_unix=time.time()))


def code_hashes():
    paths = [Path(__file__), Path(campaign_eval.__file__), Path(previous.__file__),
             ROOT/'code/transformer/eval_mixture.py',
             ROOT/'code/scripts/campaign_job_guard.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def load_inputs(plan_path, baseline_root, scaling_root, comparison_root):
    plan, digest = read_plan(plan_path)
    old = previous.load_inputs(ROOT/plan['parent']['plan_path'], baseline_root, scaling_root)
    test = old['plan']['test']
    require(np.array_equal(old['indices'], np.arange(256, 4096))
            and plan['evaluation']['development_prefix'] == 256
            and plan['evaluation']['planned_cohort'] == 4096,
            'Comparison must use exactly the existing indices 256:4096')
    require(test['cohort_id'] == plan['evaluation']['cohort_id']
            and test['data_sha256'] == plan['evaluation']['cohort_data_sha256']
            and test['times'] == plan['evaluation']['times']
            and old['plan']['source_dimensions'] == plan['source_dimensions'],
            'Campaign and original frozen comparison disagree')
    parents, receipts = {}, {}
    for seed in (0, 1):
        name = f'large_seed{seed}'
        parents[name], receipts[name] = previous._load_model_result(old, name, comparison_root)
        require(old['pilots'][name]['checkpoint_sha256'] == plan['parent']['checkpoint_sha256'][str(seed)],
                'Original 100000-update comparison uses another parent')
    return dict(plan=plan, plan_sha256=digest, old=old, parents=parents, parent_receipts=receipts,
                implementation_sha256=code_hashes())


def checkpoint(inputs, campaign_root, arm, seed, step, device):
    path = Path(campaign_root)/f'train/{arm}/seed{seed}/model_{step}.pt'
    pilot = Path(campaign_root)/f'eval/{arm}/seed{seed}/step{step}'
    receipt = json.loads((pilot/'summary.json').read_text())
    identity = receipt['identity']
    require(receipt.get('complete') is True
            and identity['plan_sha256'] == inputs['plan_sha256']
            and identity['arm'] == arm and identity['seed'] == seed and identity['step'] == step
            and identity['test_role'] == 'existing_development_prefix'
            and identity['n_sequences'] == 256
            and identity['cohort_id'] == inputs['old']['plan']['test']['cohort_id']
            and identity['times'] == inputs['old']['times'].tolist()
            and identity['evaluator_sha256'] == sha(campaign_eval.__file__)
            and identity['reference_records_sha256'] == inputs['old']['frozen']['reference_hash'],
            'Campaign development evaluation identity differs')
    require(receipt['data_sha256'] == sha(pilot/'scores.npz'), 'Development score checksum differs')
    require(receipt['causality_check']['passed'] is True
            and receipt['causality_check']['max_absolute_logit_difference'] <= 2e-5,
            'Campaign development causality check failed')
    raw = path.read_bytes()
    weight_hash = hashlib.sha256(raw).hexdigest()
    require(weight_hash == identity['checkpoint_sha256'], 'Weights changed since development evaluation')
    saved = torch.load(BytesIO(raw), map_location='cpu', weights_only=False)
    require(campaign_eval.validate_checkpoint(saved, inputs['plan'], inputs['plan_sha256'], arm, seed, device) == step,
            'Requested checkpoint step differs')
    return saved, dict(checkpoint_sha256=weight_hash, development_summary_sha256=sha(pilot/'summary.json'),
                      development_scores_sha256=receipt['data_sha256'])


def model_identity(inputs, arm, seed, step, provenance):
    old = inputs['old']
    return dict(schema=SCHEMA, test_role='reused_comparison', fresh_confirmation=False,
                arm=arm, seed=seed, step=step, final_checkpoint=step == inputs['plan']['primary']['final_steps'][arm],
                plan_sha256=inputs['plan_sha256'], original_plan_sha256=old['plan_sha256'],
                cohort_id=old['plan']['test']['cohort_id'], cohort_data_sha256=old['plan']['test']['data_sha256'],
                sequence_start=256, sequence_stop=4096, times=old['times'].tolist(), first_token_scored=False,
                reference_records_sha256=old['reference_records_sha256'], checkpoint=provenance,
                parent_score_identity=inputs['parent_receipts'][f'large_seed{seed}']['identity'],
                parent_scores_sha256=inputs['parent_receipts'][f'large_seed{seed}']['data_sha256'],
                implementation_sha256=inputs['implementation_sha256'])


def load_scores(path, identity, old):
    receipt = json.loads(path.with_suffix('.json').read_text())
    require(receipt.get('identity') == identity and receipt.get('complete') is True,
            'Saved comparison identity differs')
    require(receipt['data_sha256'] == sha(path), 'Comparison score checksum differs')
    require(receipt['causality_check']['passed'] is True
            and receipt['causality_check']['max_absolute_logit_difference'] <= 2e-5,
            'Saved comparison causality check failed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    indices, times = old['indices'], old['times']
    require(np.array_equal(arrays['sequence_indices'], indices) and np.array_equal(arrays['times'], times),
            'Saved comparison sequence pairing or times differ')
    n, nt = len(indices), len(times)
    for metric in METRICS:
        require(arrays[metric].shape == (n, nt) and np.isfinite(arrays[metric]).all(), 'Invalid comparison metric')
    p = arrays['predictive']
    require(p.shape == (n, nt, math.prod(old['plan']['source_dimensions']))
            and np.isfinite(p).all() and (p > 0).all() and np.max(abs(p.sum(-1)-1)) < 1e-11,
            'Invalid comparison predictive distribution')
    for j, i in enumerate(indices):
        replay = original.metrics(p[j], old['references']['oracle_probs'][i],
                                  old['frozen']['data']['tokens'][i], times, arrays['cumulative_loss_bits'][j])
        for metric in ('next_kl_bits', 'next_loss_bits'):
            require(np.allclose(replay[metric], arrays[metric][j], rtol=0, atol=1e-11),
                    'Comparison metric replay failed')
    return arrays, receipt


def score_model(inputs, campaign_root, out, arm, seed, step, device, batch_size):
    saved, provenance = checkpoint(inputs, campaign_root, arm, seed, step, device)
    identity = model_identity(inputs, arm, seed, step, provenance)
    path = Path(out)/'models'/f'{arm}_seed{seed}.npz'
    if path.exists() and path.with_suffix('.json').exists():
        return load_scores(path, identity, inputs['old'])
    model = GPT(**saved['cfg']['model']).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    require(model.n_params() == inputs['plan']['arms'][arm]['n_params'], 'Model parameter count changed')
    old = inputs['old']; indices = old['indices']; times = old['times']
    tokens = old['frozen']['data']['tokens'][indices]
    oracle = old['references']['oracle_probs'][indices]
    causal = check_causality(model, torch.as_tensor(tokens[:1], dtype=torch.long, device=device), times)
    require(causal['passed'] is True and causal['max_absolute_logit_difference'] <= 2e-5, 'Causal prediction check failed')
    started = time.monotonic(); chunks = []
    for start in range(0, len(indices), batch_size):
        stop = min(start+batch_size, len(indices))
        with torch.inference_mode():
            x = torch.as_tensor(tokens[start:stop, :-1], dtype=torch.long, device=device)
            logp = (torch.log_softmax(model(x).double(), -1)/math.log(2)).cpu().numpy()
        chunks.append(previous.score_log_probabilities(logp, tokens[start:stop], oracle[start:stop], times))
        if stop == len(indices) or stop % (16*batch_size) == 0:
            print(json.dumps(dict(stage='model', arm=arm, seed=seed, step=step, completed=stop, total=len(indices))), flush=True)
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    arrays.update(sequence_indices=indices, times=times)
    original.save_npz(path, arrays)
    receipt = dict(identity=identity, complete=True, n_sequences=len(indices), data_sha256=sha(path),
                   causality_check=causal, elapsed_seconds=time.monotonic()-started,
                   inference=dict(device=device, batch_size=batch_size, torch_version=str(torch.__version__),
                                  precision='float32; log-softmax float64'))
    original.save_json(path.with_suffix('.json'), receipt)
    return load_scores(path, identity, old)


def summarize(inputs, results, arm, step):
    require(set(results) == {f'{arm}_seed0', f'{arm}_seed1'}, 'Both prescribed training seeds are required')
    old = inputs['old']; indices = old['indices']
    predictors = {name: {metric: old['references'][f'{name}_{metric}'][indices] for metric in METRICS}
                  for name in previous.REFERENCES}
    predictors.update({name: {m: arrays[m] for m in METRICS} for name, arrays in inputs['parents'].items()})
    predictors.update({name: {m: result[0][m] for m in METRICS} for name, result in results.items()})
    strata = old['frozen']['data']['stratum'][indices]
    masks = [('full_prior', np.ones(len(indices), dtype=bool), None)]
    masks += [(f'entropy_{i}', strata == i, label) for i, label in enumerate(old['frozen']['cohort']['stratum_labels'])]
    pairs = [('bayes_hidden', 'bayes_encoding_given')]
    for seed in (0, 1):
        pairs += [(f'{arm}_seed{seed}', 'bayes_hidden'), (f'{arm}_seed{seed}', f'large_seed{seed}'),
                  (f'large_seed{seed}', 'bayes_hidden')]
    groups = []
    for name, mask, label in masks:
        require(mask.sum() >= 2, 'Insufficient cases for a prescribed entropy group')
        groups.append(dict(id=name, entropy_label=label, n_seq=int(mask.sum()), status='complete',
            predictors={name: {m: original.moments(values[mask]) for m, values in metrics.items()}
                        for name, metrics in predictors.items()},
            paired={f'{left}_minus_{right}': {m: original.moments((predictors[left][m]-predictors[right][m])[mask])
                                            for m in METRICS} for left, right in pairs}))
    return dict(schema=SCHEMA, complete=True, scientific_results=True, test_role='reused_comparison',
                fresh_confirmation=False, new_training=False, new_bayesian_computation=False,
                arm=arm, step=step, final_checkpoint=step == inputs['plan']['primary']['final_steps'][arm],
                plan_sha256=inputs['plan_sha256'], cohort_id=old['plan']['test']['cohort_id'],
                cohort_data_sha256=old['plan']['test']['data_sha256'], reference_records_sha256=old['reference_records_sha256'],
                sequence_start=256, sequence_stop=4096, evaluated_n_sequences=3840, times=old['times'].tolist(),
                source_dimensions=inputs['plan']['source_dimensions'], first_token_scored=False,
                scientific_scope='Previously examined comparison cases; this campaign supplies no fresh independent confirmation.',
                uncertainty='One paired standard error across sequences; the two training seeds are reported separately.',
                interpretation=inputs['plan']['interpretation'], groups=groups,
                model_receipts={name: result[1] for name, result in results.items()},
                parent_receipts=inputs['parent_receipts'], implementation_sha256=inputs['implementation_sha256'])


def run(plan_path, baseline_root, scaling_root, comparison_root, campaign_root, out, arm, step,
        device='cuda', batch_size=8):
    selection(arm, step)
    require(device in ('cpu', 'cuda') and type(batch_size) is int and batch_size > 0, 'Invalid inference settings')
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    protect_output(out, (baseline_root, scaling_root, comparison_root, campaign_root))
    torch.set_num_threads(1)
    out = Path(out)
    with output_claim(out, arm):
        original.save_json(out/'status.json', dict(schema=SCHEMA, status='checking_inputs', complete=False, arm=arm, step=step))
        try:
            inputs = load_inputs(plan_path, baseline_root, scaling_root, comparison_root)
            # Binding the selection before scoring prevents one output directory
            # from being repurposed for a different arm or intermediate step.
            binding = dict(schema=SCHEMA, arm=arm, step=step, plan_sha256=inputs['plan_sha256'],
                           test_role='reused_comparison', implementation_sha256=inputs['implementation_sha256'])
            binding_path = out/'selection.json'
            if binding_path.exists():
                require(json.loads(binding_path.read_text()) == binding, 'Output selection or implementation changed')
            else:
                original.save_json(binding_path, binding)
            results = {}
            for seed in (0, 1):
                original.save_json(out/'status.json', dict(schema=SCHEMA, status='scoring', complete=False,
                                                         arm=arm, step=step, seed=seed))
                results[f'{arm}_seed{seed}'] = score_model(inputs, campaign_root, out, arm, seed, step, device, batch_size)
            summary = summarize(inputs, results, arm, step)
            original.save_json(out/'summary.json', summary)
            original.save_json(out/'status.json', dict(schema=SCHEMA, status='completed', complete=True,
                arm=arm, step=step, evaluated_n_sequences=3840, fresh_confirmation=False,
                summary_sha256=sha(out/'summary.json')))
            return summary
        except BaseException as error:
            original.save_json(out/'status.json', dict(schema=SCHEMA, status='failed', complete=False,
                                                     arm=arm, step=step, error=str(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    for name in ('baseline-root', 'scaling-root', 'comparison-root', 'campaign-root', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=SELECTIONS, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    a = parser.parse_args()
    result = run(a.plan, a.baseline_root, a.scaling_root, a.comparison_root, a.campaign_root, a.out,
                 a.arm, a.step, a.device, a.batch_size)
    print(json.dumps({k: result[k] for k in ('complete', 'arm', 'step', 'evaluated_n_sequences', 'final_checkpoint', 'fresh_confirmation')}))


if __name__ == '__main__':
    main()
