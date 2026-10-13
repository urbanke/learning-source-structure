"""Score a predeclared campaign milestone on the existing 256 exploratory cases.

This command cannot score the previously inspected 3840-sequence comparison.
It performs no Bayesian inference and does not change the frozen inputs.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

import hidden_factor_scaling_eval as old_eval
import eval_hidden_factor as original
from eval_mixture import check_causality
from hidden_factor_campaign_train import PLAN, ROOT, read_plan, require, sha
from model import GPT
import train_hidden_factor as training
from train_mixture import digest_json, runtime_contract


def validate_checkpoint(saved, plan, digest, arm, seed, device):
    step = saved['step']; receipt = saved['cfg']; cfg = receipt['config']; spec = plan['arms'][arm]
    require(step in spec['milestones'], 'Checkpoint is not a declared campaign milestone')
    expected = dict(spec['config'], source_dimensions=plan['source_dimensions'], seed=seed)
    provenance_key = 'scaling_provenance' if arm == 'continue_large' else 'campaign_provenance'
    require({k:v for k,v in cfg.items() if k != provenance_key} == expected, 'Campaign checkpoint configuration changed')
    provenance = cfg[provenance_key]
    require(provenance['plan_sha256'] == digest and provenance['arm'] == arm and provenance['seed'] == seed
            and provenance['trainer_sha256'] == sha(ROOT/'code/transformer/hidden_factor_campaign_train.py'),
            'Campaign checkpoint provenance changed')
    require(saved['completed_sequences'] == step*cfg['batch'] and saved['implementation_hashes'] == plan['implementation_sha256'],
            'Campaign source exposure or numerical implementation changed')
    if arm == 'continue_large':
        require(provenance['parent']['weights_checkpoint_sha256'] == plan['parent']['checkpoint_sha256'][str(seed)],
                'Continuation inherited another model')
    require(saved['schema'] == ('hidden-factor-scaling-weights-v1' if arm == 'continue_large' else 'hidden-factor-weights-v1'),
            'Unexpected weight checkpoint schema')
    expected_model = dict(vocab=math.prod(plan['source_dimensions']), n_ctx=cfg['n_ctx'],
                          n_layer=cfg['n_layer'], n_head=cfg['n_head'], d_model=cfg['d_model'],
                          mlp=True, dropout=0.0)
    require(receipt['model'] == expected_model and receipt['n_params'] == spec['n_params'],
            'Checkpoint architecture differs from the declared model')
    require(saved['config_hash'] == receipt['config_hash'] == digest_json(cfg), 'Checkpoint configuration hash changed')
    require(receipt['implementation_hashes'] == plan['implementation_sha256'], 'Receipt implementation changed')
    contract = training.source_contract(plan['source_dimensions'])
    require(receipt['source_contract'] == contract and receipt['source_contract_hash'] == digest_json(contract),
            'Checkpoint source contract changed')
    require(receipt['precision'] == 'float32' and receipt['runtime'] == runtime_contract(torch.device(device)),
            'Checkpoint numerical runtime differs')
    validation = training.draw_validation(plan['source_dimensions'], cfg['n_ctx'], cfg['val_n'], cfg['validation_seed'])
    hashes = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in validation.items()}
    require(receipt['validation_hashes'] == hashes, 'Checkpoint training-validation cohort changed')
    if arm == 'continue_large':
        require(provenance['parent']['validation_hashes'] == hashes, 'Continuation validation differs from parent')
    return step


def evaluate(plan_path, baseline_eval, arm, seed, checkpoint, out, device='cuda', batch_size=8):
    plan, digest = read_plan(plan_path)
    require(arm in plan['arms'] and seed in plan['seeds'] and batch_size > 0, 'Unplanned evaluation')
    require(plan['evaluation']['development_prefix'] == 256, 'Evaluation must use the declared 256-case prefix')
    parent_plan = json.loads((ROOT/plan['parent']['plan_path']).read_text())
    frozen = old_eval.load_frozen_baseline(baseline_eval, parent_plan)
    checkpoint = Path(checkpoint); saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    step = validate_checkpoint(saved, plan, digest, arm, seed, device)
    cfg = saved['cfg']['config']; spec = plan['arms'][arm]
    identity = dict(schema='hidden-factor-campaign-development-v1', plan_sha256=digest, arm=arm, seed=seed, step=step,
                    checkpoint_sha256=sha(checkpoint), evaluator_sha256=sha(__file__), test_role='existing_development_prefix',
                    n_sequences=256, cohort_id=plan['evaluation']['cohort_id'],
                    reference_records_sha256=frozen['reference_hash'], times=plan['evaluation']['times'])
    out = Path(out)
    require(out.resolve() != checkpoint.parent.resolve(), 'Evaluation cannot overwrite training directory')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.eval.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'summary.json').exists():
            prior = json.loads((out/'summary.json').read_text())
            require(prior['identity'] == identity and prior['complete'] is True
                    and prior['data_sha256'] == sha(out/'scores.npz'), 'Existing milestone evaluation differs')
            return prior
        require(device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
        model = GPT(**saved['cfg']['model']).to(device).eval()
        require(model.n_params() == spec['n_params'], 'Model parameter count changed')
        model.load_state_dict(saved['model'], strict=True)
        tokens = frozen['data']['tokens'][:256]; times = np.asarray(plan['evaluation']['times'])
        causal = check_causality(model, torch.as_tensor(tokens[:1], dtype=torch.long, device=device), times)
        require(causal['passed'] and causal['max_absolute_logit_difference'] <= 2e-5, 'Causal prediction check failed')
        rows = []
        for start in range(0, 256, batch_size):
            xs = tokens[start:start+batch_size]
            with torch.inference_mode():
                logp = (torch.log_softmax(model(torch.as_tensor(xs[:, :-1], dtype=torch.long, device=device)).double(), -1)/math.log(2)).cpu().numpy()
            cumulative = (-np.take_along_axis(logp, xs[:, 1:, None], -1)[..., 0]).cumsum(1)[:, times-1]
            for j in range(len(xs)):
                p = np.exp2(logp[j, times-1])
                record = original.metrics(p, frozen['references']['oracle_probs'][start+j], xs[j], times, cumulative[j])
                rows.append(dict(record, predictive=p))
        values = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        values.update(times=times, sequence_indices=np.arange(256))
        original.save_npz(out/'scores.npz', values)
        result = dict(identity=identity, complete=True, causality_check=causal, data_sha256=sha(out/'scores.npz'),
                      model={m: original.moments(values[m]) for m in old_eval.METRICS},
                      model_minus_bayes={m: original.moments(values[m]-frozen['references']['bayes_hidden_'+m]) for m in old_eval.METRICS},
                      uncertainty='One paired standard error over the existing 256 exploratory sequences; no training-seed pooling.',
                      new_bayesian_computation=False, fresh_confirmation=False)
        original.save_json(out/'summary.json', result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', type=Path, default=PLAN)
    p.add_argument('--baseline-eval', type=Path, required=True); p.add_argument('--arm', choices=('continue_large','xlarge'), required=True)
    p.add_argument('--seed', type=int, choices=(0,1), required=True); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--batch-size', type=int, default=8); a=p.parse_args(); torch.set_num_threads(1)
    r=evaluate(a.plan,a.baseline_eval,a.arm,a.seed,a.checkpoint,a.out,a.device,a.batch_size)
    print(json.dumps(dict(complete=True,arm=a.arm,seed=a.seed,step=r['identity']['step'],
                         regret_bits=r['model']['next_kl_bits']['mean'][-1],
                         excess_over_bayes_bits=r['model_minus_bayes']['next_kl_bits']['mean'][-1])),flush=True)


if __name__=='__main__': main()
