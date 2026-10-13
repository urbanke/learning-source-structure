"""Paired ordered-test evaluation of ordered and independently tail-shuffled training.

The source is the GENERAL shared-base LSA family. No Bayesian comparator is
substituted: the measured advantage is control loss minus ordered-training loss.
Every loss uses the same original prefix and original oracle transition row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model import GPT
from source import Source, seed_of

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json'
DEFAULT_TRAINING_PLAN = ROOT / 'code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json'
SCHEMA = 'shared-base-shuffle-evaluation-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order='C')).hexdigest()


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen output differs: {path}')
        return
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def source_arrays(depth, indices, master_seed, d=64, n_tokens=1025):
    """Replay the existing conditional benchmark's per-example Source draws."""
    tokens, base, transition = [], [], []
    source = Source(d, n_tokens, alpha=1., ls=depth, permute=True)
    for index in indices:
        seed = seed_of('conditional-shared-base-benchmark-v1', master_seed, '1', depth, index)
        rng = np.random.default_rng(seed)
        alpha, depths = source.draw_params(1, rng)
        q, matrix = source.matrices(alpha, depths, rng)
        raw = source.sequences(q, matrix, rng)[0]
        perm = rng.permutation(d)
        inverse = np.argsort(perm)
        x, initial, kernel = perm[raw], q[0, inverse], matrix[0][inverse][:, inverse]
        replay, _ = source.batch(1, np.random.default_rng(seed))
        if not np.array_equal(x, replay[0]):
            raise ValueError('Original Source.batch replay failed')
        tokens.append(x); base.append(initial); transition.append(kernel)
    return dict(tokens=np.asarray(tokens, dtype=np.int64), base=np.asarray(base),
                transition=np.asarray(transition))


def checksums(arrays):
    return {key: array_sha(arrays[key], '<i8' if key == 'tokens' else '<f8')
            for key in ('tokens', 'base', 'transition')}


def freeze_existing(out):
    """Freeze hashes against all existing records; no cohort selection by results."""
    conditions = []
    for depth in (0, 2, 3):
        folder = (ROOT / 'output/shared_base_conditional_curves_2026-09-15/a1_L0'
                  if depth == 0 else ROOT / f'output/general_lsa_copying_2026-09-17/natural/sources/a1_L{depth}')
        protocol = json.loads((folder / 'protocol.json').read_text())
        declaration = json.loads((folder / 'cohort.json').read_text())
        if protocol['master_seed'] != 20260930 or declaration['n_sequences'] != 256:
            raise ValueError('Unexpected diagnostic cohort')
        records = []
        for index, entry in enumerate(declaration['records']):
            path = folder / 'data' / f'seq_{index:06d}.json'
            if sha(path) != entry['file_sha256']:
                raise ValueError('Frozen source record changed')
            records.append(json.loads(path.read_text()))
        stored = dict(tokens=np.asarray([r['tokens'] for r in records]),
                      base=np.asarray([r['oracle']['initial_probs'] for r in records]),
                      transition=np.asarray([r['oracle']['transition_probs'] for r in records]))
        regenerated = source_arrays(depth, range(256), 20260930)
        if checksums(stored) != checksums(regenerated):
            raise ValueError('Regeneration differs from original frozen arrays')
        conditions.append(dict(id=f'a1_L{depth}', alpha=1., depth=depth, indices=list(range(256)),
                               original_folder=str(folder.relative_to(ROOT)),
                               original_cohort_sha256=sha(folder / 'cohort.json'),
                               arrays_sha256=checksums(stored)))
    manifest = dict(schema=SCHEMA, role='diagnostic reuse; not independent confirmation',
                    scientific_results=True, d=64, n_tokens=1025, master_seed=20260930,
                    source_code_sha256=sha(ROOT / 'code/transformer/source.py'),
                    conditions=conditions, primary_seed=0, confirmation='Freeze fresh sources before a final timing claim.',
                    scoring='At time t, use original prefix X1,...,Xt and true row T[Xt,:] to predict X(t+1). First token is supplied, not scored.')
    write_json(out, manifest, immutable=True)
    print(json.dumps(dict(status='frozen', manifest=str(out), sequences=768)), flush=True)


def prepare(manifest_path, out):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['schema'] != SCHEMA or sha(ROOT / 'code/transformer/source.py') != manifest['source_code_sha256']:
        raise ValueError('Source implementation differs from the frozen manifest')
    out = Path(out)
    for condition in manifest['conditions']:
        path = out / (condition['id'] + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                arrays = {key: old[key] for key in ('tokens', 'base', 'transition')}
                if str(old['manifest_sha256']) != sha(manifest_path):
                    raise ValueError('Existing cohort belongs to another manifest')
        else:
            arrays = source_arrays(condition['depth'], condition['indices'], manifest['master_seed'],
                                   manifest['d'], manifest['n_tokens'])
        if checksums(arrays) != condition['arrays_sha256']:
            raise ValueError(f"Regenerated arrays differ: {condition['id']}")
        if not path.exists():
            save_npz(path, **arrays, indices=np.asarray(condition['indices']), manifest_sha256=sha(manifest_path))
        print(json.dumps(dict(condition=condition['id'], status='verified', sequences=len(arrays['tokens']))), flush=True)
    return manifest


def expected_losses(logp, tokens, transition):
    """Independent oracle-row integration, with explicit t/target indexing."""
    n, times, d = logp.shape
    if tokens.shape != (n, times + 1) or transition.shape != (n, d, d):
        raise ValueError('Prediction/source dimensions differ')
    if not np.allclose(np.exp(logp).sum(-1), 1., rtol=0, atol=2e-6):
        raise ValueError('Predictions are not normalized')
    rows = transition[np.arange(n)[:, None], tokens[:, :-1]]
    expected = -np.sum(rows * logp, axis=-1) / np.log(2.)
    entropy = -np.sum(rows * np.log2(np.maximum(rows, np.finfo(float).tiny)), axis=-1)
    realized = -np.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0] / np.log(2.)
    return dict(expected_loss_bits=expected, regret_bits=expected-entropy, realized_loss_bits=realized)


def validate_checkpoint(saved, cfg, seed, role, smoke=False, checkpoint=None, plan_path=DEFAULT_TRAINING_PLAN):
    if saved['cfg'] != cfg:
        raise ValueError('Checkpoint and configuration disagree')
    if smoke:
        return
    plan = json.loads(Path(plan_path).read_text())
    expected = dict(schema='shared-base-scaling-training-v1' if role=='ordered' else 'shared-base-shuffle-training-v1',
                    **plan['config'], **plan['source'], seed=seed)
    if cfg['config'] != expected:
        raise ValueError(f'Unmatched {role} training configuration or optimizer schedule')
    expected_model=dict(vocab=64,n_ctx=1024,n_layer=4,n_head=8,d_model=256,mlp=True,dropout=0.)
    if cfg['model'] != expected_model or cfg['n_params'] != 3454464:
        raise ValueError('Checkpoint architecture differs from the matched larger model')
    if saved['step'] != 100000:
        raise ValueError('Primary evaluation requires the final planned checkpoint')
    folder = Path(checkpoint).parent
    hashes = plan['ordered_runs'][str(seed)]['files_sha256']
    if role == 'ordered':
        for filename in ('model_100000.pt','config.json','status.json','scaling_runtime.json'):
            if sha(folder/filename) != hashes[filename]:
                raise ValueError(f'Ordered checkpoint provenance differs: {filename}')
        if cfg['plan_sha256'] != plan['ordered_plan']['sha256']:
            raise ValueError('Wrong ordered training plan')
    else:
        if cfg.get('control') != 'independent-tail-position-shuffle' or cfg.get('source_contract') != plan['shuffle_contract']:
            raise ValueError('Control checkpoint does not declare the chosen shuffle law')
        if cfg['plan_sha256'] != sha(plan_path) or cfg['implementation_hashes'] != plan['implementation_sha256']:
            raise ValueError('Control checkpoint differs from the frozen training plan/code')
        if cfg['ordered_training']['original_files_sha256'] != hashes:
            raise ValueError('Control was paired with another ordered run')
    status = json.loads((folder/'status.json').read_text())
    if status['complete'] is not True or status['step'] != 100000 or status['completed_sequences'] != 6400000:
        raise ValueError('Training did not complete the matched sample budget')


def model_config(cfg):
    if 'model' in cfg:
        return cfg['model']
    inner = cfg.get('config', cfg)
    return dict(vocab=inner['d'], n_ctx=inner['n_ctx'], n_layer=inner['n_layer'],
                n_head=inner['n_head'], d_model=inner['d_model'], mlp=True, dropout=0.)


def shuffle_diagnostic(model, tokens, times, seed, device, count):
    """Same counts/first token, two fresh tail rearrangements, final prediction TV."""
    rng = np.random.default_rng(seed)
    selected = tokens[:count]
    rows = []
    with torch.inference_mode():
        for t in times:
            original = selected[:, :t].copy()
            p = model(torch.from_numpy(original).to(device))[:, -1].double().softmax(-1).cpu().numpy()
            distances = []
            for _ in range(2):
                rearranged = original.copy()
                for row in rearranged:
                    row[1:] = rng.permutation(row[1:])
                if not np.array_equal(original[:, 0], rearranged[:, 0]) or not np.array_equal(np.sort(original, axis=1), np.sort(rearranged, axis=1)):
                    raise ValueError('Diagnostic changed counts or first token')
                q = model(torch.from_numpy(rearranged).to(device))[:, -1].double().softmax(-1).cpu().numpy()
                distances.append(.5*np.abs(p-q).sum(-1))
            values = np.mean(distances, axis=0)
            rows.append(dict(time=t, n_sequences=len(values), mean_tv=float(values.mean()),
                             se_tv=float(values.std(ddof=1)/np.sqrt(len(values)))))
    return rows


def evaluate(manifest_path, cohort_dir, checkpoint, role, seed, out, device='cpu',
             batch=4, threads=4, dtype='float64', diagnostic_n=0, smoke=False,
             plan_path=DEFAULT_TRAINING_PLAN):
    manifest = json.loads(Path(manifest_path).read_text())
    if bool(manifest['scientific_results']) == bool(smoke):
        raise ValueError('Smoke and scientific inputs cannot be mixed')
    torch.set_num_threads(threads)
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = json.loads((checkpoint.parent / 'config.json').read_text())
    validate_checkpoint(saved, cfg, seed, role, smoke, checkpoint, plan_path)
    model = GPT(**model_config(cfg)).to(device=device, dtype=getattr(torch, dtype)).eval()
    model.load_state_dict(saved['model'])
    receipt = dict(schema=SCHEMA, role=role, seed=seed, scientific_results=not smoke,
                   manifest_sha256=sha(manifest_path), checkpoint_sha256=sha(checkpoint),
                   checkpoint_step=saved['step'], config_sha256=sha(checkpoint.parent/'config.json'),
                   model=model.cfg, dtype=dtype, device=device,
                   model_code_sha256=sha(ROOT/'code/transformer/model.py'),
                   evaluation_code_sha256=sha(__file__), shuffle_diagnostic_sequences=diagnostic_n)
    if not smoke:
        receipt['training_plan_sha256']=sha(plan_path)
        receipt['paired_ordered_files_sha256']=json.loads(Path(plan_path).read_text())['ordered_runs'][str(seed)]['files_sha256']
    out = Path(out)
    write_json(out/'protocol.json', receipt, immutable=True)
    start = time.monotonic()
    for condition in manifest['conditions']:
        ident = condition['id']
        destination = out / (ident+'.npz')
        if destination.exists():
            with np.load(destination, allow_pickle=False) as previous:
                if str(previous['protocol_sha256']) != sha(out/'protocol.json'):
                    raise ValueError('Existing evaluation provenance differs')
            continue
        with np.load(Path(cohort_dir)/(ident+'.npz'), allow_pickle=False) as arrays:
            tokens, transition = arrays['tokens'], arrays['transition']
            if str(arrays['manifest_sha256']) != sha(manifest_path) or checksums(arrays) != condition['arrays_sha256']:
                raise ValueError('Frozen evaluation cohort differs')
        metrics = {key: [] for key in ('expected_loss_bits','regret_bits','realized_loss_bits')}
        causal_error = 0.
        for left in range(0, len(tokens), batch):
            right = min(left+batch, len(tokens))
            x = torch.from_numpy(tokens[left:right, :-1]).to(device)
            with torch.inference_mode():
                logits = model(x)
                if left == 0:
                    t = min(32, x.shape[1])
                    shorter = model(x[:, :t])[:, -1]
                    causal_error = float((shorter-logits[:, t-1]).abs().max())
                logp = logits.double().log_softmax(-1).cpu().numpy()
            scored = expected_losses(logp, tokens[left:right], transition[left:right])
            for key in metrics:
                metrics[key].append(scored[key])
            if right % 64 == 0 or right == len(tokens):
                print(json.dumps(dict(role=role, seed=seed, condition=ident, completed=right,
                                      total=len(tokens), elapsed_seconds=time.monotonic()-start)), flush=True)
        tolerance = 1e-9 if dtype == 'float64' else 2e-5
        if causal_error > tolerance:
            raise ValueError(f'Prefix causality check failed: {causal_error}')
        merged = {key: np.concatenate(value) for key, value in metrics.items()}
        if not all(np.isfinite(value).all() for value in merged.values()) or merged['regret_bits'].min() < -tolerance:
            raise ValueError('Non-finite loss or negative oracle regret')
        diag = []
        if diagnostic_n:
            diag = shuffle_diagnostic(model, tokens, [t for t in (1,16,64,256,1024) if t < tokens.shape[1]],
                                      seed_of('shared-base-shuffle-diagnostic', condition['depth']), device, min(diagnostic_n,len(tokens)))
        save_npz(destination, **merged, protocol_sha256=sha(out/'protocol.json'),
                 source_tokens_sha256=condition['arrays_sha256']['tokens'],
                 times=np.arange(1,tokens.shape[1]), causal_error=causal_error)
        write_json(out/(ident+'_diagnostic.json'), dict(rows=diag,
                   interpretation='Tail-order sensitivity at fixed first token and counts; a diagnostic, not a proof of a Bayesian unigram.'))
    write_json(out/'status.json', dict(complete=True, scientific_results=not smoke,
                                      elapsed_seconds=time.monotonic()-start, conditions=len(manifest['conditions'])))


def crossing_description(curve):
    curve = np.asarray(curve)
    upward = (np.flatnonzero((curve[:-1] <= 0) & (curve[1:] > 0))+2).tolist()
    downward = (np.flatnonzero((curve[:-1] >= 0) & (curve[1:] < 0))+2).tolist()
    # A first-time positive mean is not evidence of a delayed transition.
    if curve[0] > 0:
        status = 'positive_from_first_scored_token'
    elif upward:
        status = 'observed_mean_upcrossing'
    else:
        status = 'no_positive_upcrossing_in_observed_range'
    if curve[-1] > 0:
        nonpositive=np.flatnonzero(curve<=0)
        persistent_start=int(nonpositive[-1]+2) if len(nonpositive) else 1
    else:
        persistent_start=None
    return dict(status=status, first_upcrossing=upward[0] if upward else None,
                all_upcrossings=upward, all_downcrossings=downward,
                positive_through_final_time=bool(curve[-1]>0),
                last_positive_run_start=persistent_start)


def paired_summary(values, counts):
    values = np.asarray(values)
    mean = values.mean(0)
    draws = counts @ values / len(values)
    lower, upper = np.quantile(draws, [.025,.975], axis=0)
    crossings = [crossing_description(row) for row in draws]
    finite = [row['first_upcrossing'] for row in crossings if row['status']=='observed_mean_upcrossing']
    crossing = crossing_description(mean)
    crossing.update(bootstrap_fraction_with_delayed_upcrossing=len(finite)/len(draws),
                    bootstrap_status_counts={status:sum(row['status']==status for row in crossings)
                                             for status in ('observed_mean_upcrossing','positive_from_first_scored_token','no_positive_upcrossing_in_observed_range')},
                    bootstrap_positive_through_final_time=sum(row['positive_through_final_time'] for row in crossings),
                    conditional_bootstrap_interval=np.quantile(finite,[.025,.975]).tolist() if finite else None,
                    interval_warning='Crossing interval conditions on a crossing; it is not a confirmed transition when bootstrap mass is absent or already positive.')
    return dict(mean=mean.tolist(), se=(values.std(0,ddof=1)/np.sqrt(len(values))).tolist(),
                lower95=lower.tolist(), upper95=upper.tolist(), crossing=crossing)


def summarize(manifest_path, evaluations, out, replicates=1000, seeds=(0,1)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = json.loads(Path(manifest_path).read_text())
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    summary = dict(schema=SCHEMA, scientific_results=manifest['scientific_results'],
                   manifest_sha256=sha(manifest_path), primary_seed=0,
                   sign='Positive means ordered training improves prediction on original ordered test sequences.',
                   uncertainty='95% pointwise paired whole-sequence bootstrap intervals; each training seed reported separately. Not simultaneous bands or uncertainty over training randomness.',
                   first_token='Supplied, not scored; time t predicts token t+1 after t observations.',
                   groups=[], inputs={})
    for seed in seeds:
        fig, axes = plt.subplots(2,len(manifest['conditions']),figsize=(11.4,6.0),squeeze=False,sharex='col')
        lossfig, lossaxes = plt.subplots(1,len(manifest['conditions']),figsize=(11.4,3.3),squeeze=False)
        for column, condition in enumerate(manifest['conditions']):
            loaded = {}
            receipts = {}
            for role in ('ordered','shuffled'):
                folder = Path(evaluations)/f'{role}_seed{seed}'
                status = json.loads((folder/'status.json').read_text())
                if not status['complete'] or status['scientific_results'] != manifest['scientific_results']:
                    raise ValueError('Incomplete or mixed scientific/smoke evaluations')
                receipt = json.loads((folder/'protocol.json').read_text())
                if receipt['manifest_sha256'] != sha(manifest_path) or receipt['role'] != role or receipt['seed'] != seed:
                    raise ValueError('Evaluation inputs are not paired')
                receipts[role]=receipt
                path = folder/(condition['id']+'.npz')
                with np.load(path,allow_pickle=False) as a:
                    if str(a['source_tokens_sha256']) != condition['arrays_sha256']['tokens'] or str(a['protocol_sha256']) != sha(folder/'protocol.json'):
                        raise ValueError('Source or evaluation receipt changed')
                    loaded[role] = {key:a[key] for key in ('expected_loss_bits','regret_bits','realized_loss_bits','times')}
                expected_shape=(len(condition['indices']),manifest['n_tokens']-1)
                if not np.array_equal(loaded[role]['times'],np.arange(1,manifest['n_tokens'])):
                    raise ValueError('Evaluation time grid differs')
                for key in ('expected_loss_bits','regret_bits','realized_loss_bits'):
                    value=loaded[role][key]
                    if value.shape!=expected_shape or not np.isfinite(value).all() or value.min() < -1e-8:
                        raise ValueError(f'Invalid saved {key}')
                summary['inputs'][f'{role}_seed{seed}/{condition["id"]}'] = sha(path)
            if manifest['scientific_results']:
                ordered=receipts['ordered'];shuffled=receipts['shuffled']
                if ordered['training_plan_sha256']!=shuffled['training_plan_sha256'] or ordered['model']!=shuffled['model']:
                    raise ValueError('Models are not a matched training pair')
                pair=shuffled['paired_ordered_files_sha256']
                if ordered['paired_ordered_files_sha256']!=pair or ordered['checkpoint_sha256']!=pair['model_100000.pt'] or ordered['config_sha256']!=pair['config.json']:
                    raise ValueError('Evaluated ordered checkpoint differs from the shuffled training pair')
            gap = loaded['shuffled']['expected_loss_bits']-loaded['ordered']['expected_loss_bits']
            n = len(gap)
            rng=np.random.default_rng(seed_of('shared-base-shuffle-bootstrap',seed,condition['depth']))
            counts=rng.multinomial(n,np.full(n,1/n),size=replicates).astype(float)
            instantaneous=paired_summary(gap,counts)
            cumulative=paired_summary(gap.cumsum(1),counts)
            times=loaded['ordered']['times']
            group=dict(seed=seed,condition=condition['id'],depth=condition['depth'],n_sequences=n,
                       instantaneous=instantaneous,cumulative=cumulative,
                       absolute_regret={role:dict(mean=loaded[role]['regret_bits'].mean(0).tolist(),
                                                 se=(loaded[role]['regret_bits'].std(0,ddof=1)/np.sqrt(n)).tolist())
                                        for role in loaded})
            summary['groups'].append(group)
            for row,stats in enumerate((instantaneous,cumulative)):
                ax=axes[row,column]
                ax.axhline(0,color='.35',lw=.8)
                ax.fill_between(times,stats['lower95'],stats['upper95'],color='#126a9c',alpha=.2,lw=0)
                ax.plot(times,stats['mean'],color='#126a9c',lw=1.4)
                ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
                if row==0: ax.set_title(f"Shared-base LSA: $L={condition['depth']}$")
                else: ax.set_xlabel('Observed tokens $t$')
                if column==0: ax.set_ylabel('Instantaneous advantage (bits)' if row==0 else 'Cumulative advantage (bits)')
            ax=lossaxes[0,column]
            for role,color,style in (('ordered','#126a9c','-'),('shuffled','#c46b23','--')):
                values=loaded[role]['regret_bits'];mean=values.mean(0);se=values.std(0,ddof=1)/np.sqrt(n)
                ax.plot(times,mean,label='Ordered training' if role=='ordered' else 'Shuffled training',color=color,ls=style,lw=1.5)
                ax.fill_between(times,mean-1.96*se,mean+1.96*se,color=color,alpha=.15,lw=0)
            ax.set_title(f"Shared-base LSA: $L={condition['depth']}$");ax.set_xlabel('Observed tokens $t$');ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            if column==0:ax.set_ylabel('Expected next-token regret (bits)');ax.legend(frameon=False)
        prefix='' if manifest['scientific_results'] else 'SMOKE ONLY — UNTRAINED MODELS — '
        fig.suptitle(prefix+f'Ordered-training advantage over shuffled-training control — training run {seed+1}')
        lossfig.suptitle(prefix+f'Both predictors on identical original sequences — training run {seed+1}')
        fig.tight_layout();lossfig.tight_layout()
        fig.savefig(out/f'gap_seed{seed}.png',dpi=180)
        lossfig.savefig(out/f'regret_seed{seed}.png',dpi=180)
        plt.close(fig);plt.close(lossfig)
    write_json(out/'summary.json',summary)
    print(json.dumps(dict(complete=True,scientific_results=manifest['scientific_results'],groups=len(summary['groups']))),flush=True)


def smoke(out):
    """Exercise the real pipeline using distinct tiny random checkpoints only."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    arrays=source_arrays(2,range(4),20260930,d=4,n_tokens=17)
    manifest=dict(schema=SCHEMA,scientific_results=False,d=4,n_tokens=17,master_seed=20260930,
                  source_code_sha256=sha(ROOT/'code/transformer/source.py'),
                  conditions=[dict(id='smoke_L2',alpha=1.,depth=2,indices=list(range(4)),arrays_sha256=checksums(arrays))])
    write_json(out/'manifest.json',manifest,immutable=True)
    prepare(out/'manifest.json',out/'cohorts')
    for role,random_seed in (('ordered',120),('shuffled',121)):
        torch.manual_seed(random_seed)
        model=GPT(vocab=4,n_ctx=16,n_layer=1,n_head=2,d_model=8)
        cfg=dict(model=model.cfg,smoke_only=True)
        folder=out/'checkpoints'/role;folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'config.json',cfg)
        torch.save(dict(model=model.state_dict(),cfg=cfg,step=0),folder/'model.pt')
        evaluate(out/'manifest.json',out/'cohorts',folder/'model.pt',role,0,
                 out/'eval'/f'{role}_seed0',batch=2,diagnostic_n=4,smoke=True)
    summarize(out/'manifest.json',out/'eval',out/'summary',replicates=100,seeds=(0,))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--out',type=Path,default=DEFAULT_MANIFEST)
    p=sub.add_parser('prepare');p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--plan',type=Path,default=DEFAULT_TRAINING_PLAN)
    p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--role',choices=['ordered','shuffled'],required=True);p.add_argument('--seed',type=int,choices=[0,1],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dtype',choices=['float32','float64'],default='float64');p.add_argument('--shuffle-diagnostic',type=int,default=0)
    p=sub.add_parser('summary');p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--evaluations',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    p=sub.add_parser('smoke');p.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='freeze':freeze_existing(args.out)
    elif args.command=='prepare':prepare(args.manifest,args.out)
    elif args.command=='evaluate':
        if args.batch<1 or args.threads<1 or args.shuffle_diagnostic not in (0,*range(2,257)):
            ap.error('Positive batch/threads and 0 or 2..256 diagnostic sequences required')
        evaluate(args.manifest,args.cohorts,args.checkpoint,args.role,args.seed,args.out,args.device,
                 args.batch,args.threads,args.dtype,args.shuffle_diagnostic,plan_path=args.plan)
    elif args.command=='summary':
        if args.bootstrap<100:ap.error('At least 100 bootstrap replicates required')
        summarize(args.manifest,args.evaluations,args.out,args.bootstrap)
    else:smoke(args.out)


if __name__=='__main__':
    main()
