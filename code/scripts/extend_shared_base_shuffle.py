"""Frozen paired evaluation extension; existing data and figures are read-only."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
import numpy as np
import shared_base_shuffle_eval as old

PLAN = ROOT / 'code/transformer/configs/shared_base_shuffle_extension_2026-09-22.json'
COHORT = ROOT / 'code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json'
TRAINING = ROOT / 'code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json'
METRICS = ('expected_loss_bits', 'regret_bits', 'realized_loss_bits')


def announce(event, **fields):
    print(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **fields)), flush=True)


def check_retained(arrays, original, retained):
    prefix = {k: v[:retained] for k, v in arrays.items()}
    if old.checksums(prefix) != original['arrays_sha256']:
        raise ValueError('First 256 source arrays do not match frozen original cohort')
    return old.checksums(arrays)


def check_scores(new, previous, retained, tolerance):
    differences = {}
    for key in METRICS:
        a, b = new[key][:retained], previous[key]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Retained evaluation shape or finiteness differs')
        differences[key] = float(np.max(np.abs(a-b)))
        if differences[key] > tolerance:
            raise ValueError(f'Retained {key} differs by {differences[key]}')
    return differences


def draw_chunk(args):
    depth, left, right, seed, d, tokens = args
    return old.source_arrays(depth, range(left, right), seed, d, tokens)


def validate_plan(path):
    plan = json.loads(path.read_text())
    if (plan['schema'] != 'shared-base-shuffle-extension-plan-v1'
            or plan['n_sequences'] != 1024 or plan['retained_sequences'] != 256
            or plan['depths'] != [0, 2, 3] or plan['seeds'] != [0, 1]
            or plan['master_seed'] != 20260930 or plan['dtype'] != 'float64'):
        raise ValueError('Unexpected production declaration')
    for relative, digest in plan['dependencies_sha256'].items():
        if old.sha(ROOT / relative) != digest:
            raise ValueError(f'Frozen dependency changed: {relative}')
    return plan


def prepare(plan, plan_path, out, parent_root):
    original = json.loads(COHORT.read_text())
    manifest = {key: original[key] for key in
                ('schema', 'scientific_results', 'd', 'n_tokens', 'master_seed', 'source_code_sha256', 'scoring')}
    manifest.update(role='256 retained + 768 predeclared new histories per depth',
                    extension_plan_sha256=old.sha(plan_path), original_manifest_sha256=old.sha(COHORT),
                    primary_seed=0, conditions=[], independent_confirmation=False)
    for depth in plan['depths']:
        ident = f'a1_L{depth}'
        parent = next(c for c in original['conditions'] if c['id'] == ident)
        original_path = parent_root/'cohorts'/(ident+'.npz')
        with np.load(original_path, allow_pickle=False) as saved:
            retained = {key: saved[key] for key in ('tokens', 'base', 'transition')}
            if str(saved['manifest_sha256']) != old.sha(COHORT) or old.checksums(retained) != parent['arrays_sha256']:
                raise ValueError('Frozen original cohort has changed')
        chunks = [(depth, left, min(left+64, plan['n_sequences']), plan['master_seed'],
                   plan['d'], plan['n_tokens']) for left in range(plan['retained_sequences'], plan['n_sequences'], 64)]
        parts = [retained]
        with ProcessPoolExecutor(max_workers=plan['source_workers']) as executor:
            for i, arrays in enumerate(executor.map(draw_chunk, chunks)):
                parts.append(arrays)
                announce('source_chunk_complete', condition=ident, completed=min(plan['retained_sequences']+(i+1)*64, plan['n_sequences']))
        arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        hashes = check_retained(arrays, parent, plan['retained_sequences'])
        condition = dict(id=ident, alpha=1., depth=depth, indices=list(range(plan['n_sequences'])),
                         arrays_sha256=hashes, retained_arrays_sha256=parent['arrays_sha256'],
                         retained_cohort_file_sha256=old.sha(original_path))
        manifest['conditions'].append(condition)
        old.save_npz(out/'cohorts'/(ident+'.npz'), **arrays, indices=np.arange(plan['n_sequences']))
        announce('cohort_frozen', condition=ident, sequences=plan['n_sequences'], retained_identity=True)
    old.write_json(out/'manifest.json', manifest, immutable=True)
    digest = old.sha(out/'manifest.json')
    for condition in manifest['conditions']:
        path = out/'cohorts'/(condition['id']+'.npz')
        with np.load(path, allow_pickle=False) as a:
            arrays = {key: a[key] for key in a.files}
        old.save_npz(path, **arrays, manifest_sha256=digest)
    announce('prepared', manifest_sha256=digest, sequences=3*plan['n_sequences'])
    return manifest


def summarize(plan, manifest, out, parent):
    summary = dict(schema='shared-base-shuffle-extension-summary-v1', complete=False,
                   manifest_sha256=old.sha(out/'manifest.json'), groups=[], input_sha256={},
                   no_retraining=True, no_new_bayesian_references=True,
                   uncertainty='Paired whole-source bootstrap; pointwise 95% intervals, no training uncertainty.',
                   scope='All1024, retained256 and new768 are reported separately. Existing Bayesian comparison stays on256.')
    parity = {}
    for seed in plan['seeds']:
        for condition in manifest['conditions']:
            ident = condition['id']
            loaded = {}
            for role in ('ordered', 'shuffled'):
                folder = out/'eval'/f'{role}_seed{seed}'
                protocol = json.loads((folder/'protocol.json').read_text())
                status = json.loads((folder/'status.json').read_text())
                if not status['complete'] or protocol['manifest_sha256'] != summary['manifest_sha256']:
                    raise ValueError('Incomplete or wrong-manifest evaluation')
                path = folder/(ident+'.npz')
                with np.load(path, allow_pickle=False) as a:
                    loaded[role] = {k: a[k] for k in METRICS}
                    if (str(a['protocol_sha256']) != old.sha(folder/'protocol.json')
                            or str(a['source_tokens_sha256']) != condition['arrays_sha256']['tokens']
                            or not np.array_equal(a['times'], np.arange(1, plan['n_tokens']))):
                        raise ValueError('Evaluation provenance or time grid differs')
                if any(v.shape != (plan['n_sequences'], plan['n_tokens']-1)
                       or not np.isfinite(v).all() for v in loaded[role].values()):
                    raise ValueError('Missing/non-finite evaluation rows')
                key = f'{role}_seed{seed}/{ident}'
                summary['input_sha256'][key] = old.sha(path)
                previous_path = parent/'eval'/f'{role}_seed{seed}'/(ident+'.npz')
                previous_protocol = json.loads((previous_path.parent/'protocol.json').read_text())
                if previous_protocol['checkpoint_sha256'] != protocol['checkpoint_sha256']:
                    raise ValueError('Old/new comparison uses different checkpoints')
                with np.load(previous_path, allow_pickle=False) as a:
                    previous = {k: a[k] for k in METRICS}
                parity[key] = check_scores(loaded[role], previous, plan['retained_sequences'],
                                           plan['old_score_absolute_tolerance'])
            gaps = loaded['shuffled']['expected_loss_bits'] - loaded['ordered']['expected_loss_bits']
            for cohort, start, stop in [('all1024', 0, 1024), ('retained256', 0, 256), ('new768', 256, 1024)]:
                values = gaps[start:stop]
                rng = np.random.default_rng(old.seed_of('shuffle-extension-bootstrap-v1', seed, condition['depth'], cohort))
                counts = rng.multinomial(len(values), np.full(len(values), 1/len(values)), size=plan['bootstrap']).astype(float)
                summary['groups'].append(dict(seed=seed, condition=ident, depth=condition['depth'], cohort=cohort,
                    n_sequences=len(values), instantaneous=old.paired_summary(values, counts),
                    cumulative=old.paired_summary(values.cumsum(1), counts)))
            announce('paired_condition_summarized', seed=seed, condition=ident)
    summary.update(complete=True, scientific_reference_validated=True, retained_score_parity=parity)
    old.write_json(out/'summary.json', summary, immutable=True)
    announce('complete', groups=len(summary['groups']), max_retained_difference=max(max(d.values()) for d in parity.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ordered-root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    announce('starting', job=os.environ.get('SLURM_JOB_ID'), scope='1024 histories/depth; transformer evaluation only')
    plan = validate_plan(args.plan)
    args.out = args.out.resolve()
    if args.out == args.parent.resolve() or args.parent.resolve() in args.out.parents:
        raise ValueError('New output must be separate from original evaluation')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'worker.claim').open('x') as stream:
        stream.write(json.dumps(dict(job=os.environ.get('SLURM_JOB_ID'), pid=os.getpid())))
    start = time.monotonic()
    # Check all four checkpoints and old-score provenance before generating sources.
    for seed in plan['seeds']:
        for role in ('ordered', 'shuffled'):
            folder = args.ordered_root/f'seed{seed}' if role == 'ordered' else args.parent/'train'/f'seed{seed}'
            if old.sha(folder/'model_100000.pt') != plan['checkpoint_sha256'][f'{role}_seed{seed}']:
                raise ValueError('Checkpoint hash differs from frozen evaluation pair')
            for depth in plan['depths']:
                if not (args.parent/'eval'/f'{role}_seed{seed}'/f'a1_L{depth}.npz').is_file():
                    raise ValueError('Original score array missing')
    manifest = prepare(plan, args.plan, args.out, args.parent)
    for seed in plan['seeds']:
        for role in ('ordered', 'shuffled'):
            folder = args.ordered_root/f'seed{seed}' if role == 'ordered' else args.parent/'train'/f'seed{seed}'
            announce('model_evaluation_started', seed=seed, role=role)
            old.evaluate(args.out/'manifest.json', args.out/'cohorts', folder/'model_100000.pt', role, seed,
                         args.out/'eval'/f'{role}_seed{seed}', device='cuda', batch=plan['batch'],
                         threads=plan['threads'], dtype=plan['dtype'], plan_path=TRAINING)
            announce('model_evaluation_complete', seed=seed, role=role)
    summarize(plan, manifest, args.out, args.parent)
    old.write_json(args.out/'status.json', dict(complete=True, elapsed_seconds=time.monotonic()-start,
                   job=os.environ.get('SLURM_JOB_ID'), summary_sha256=old.sha(args.out/'summary.json')))


if __name__ == '__main__':
    main()
