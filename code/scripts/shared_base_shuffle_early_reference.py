"""Reuse the validated full shared-base catalog and existing LSA unigram.

No source generation or posterior sampling: score the identical frozen
shuffled-control cohorts after 1..4 observed tokens. The existing unigram
mixes coding depths 1..20 and observes the supplied first token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
from lsa_seq import SeqLSA
from shared_base_short_evidence import SharedBaseShortEvidence
from eval_conditional_crossing import configure_backend


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a, dtype):
    return hashlib.sha256(np.asarray(a, dtype=dtype).tobytes(order='C')).hexdigest()


def statistics(a):
    return dict(mean=a.mean(0).tolist(), se=(a.std(0, ddof=1) / np.sqrt(len(a))).tolist())


def validate_cohort_arrays(data, condition):
    if data['tokens'].shape != (256, 1025):
        raise ValueError('All 256 frozen sequences are required')
    for name, value in data.items():
        if array_sha(value, '<i8' if name == 'tokens' else '<f8') != condition['arrays_sha256'][name]:
            raise ValueError('Frozen source checksum mismatch: ' + name)


def publish(report, raw_folder, destination):
    """Publish only summary statistics and a provenance receipt, never arrays."""
    if not report['complete'] or len(report['groups']) != 3:
        raise ValueError('Only all three complete conditions may be published')
    destination.mkdir(parents=True, exist_ok=True)
    summary = destination/'summary.json'
    summary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    receipt = dict(schema='shared-base-shuffle-early-reference-receipt-v1',
        complete=True, report_sha256=sha(summary), script_sha256=sha(__file__),
        manifest_sha256=report['manifest_sha256'],
        retained_raw_outputs={str(raw_folder/(g['id']+'.npz')):sha(raw_folder/(g['id']+'.npz')) for g in report['groups']},
        checks=dict(frozen_source_arrays_match=True, model_history_protocol_and_time_match=True,
                    validated_full_prior_catalog=True, pooled_unigram_depths=list(range(1,21)),
                    normalization_tolerance=1e-7, maximum_normalization_error=max(
                        max(g['maximum_normalization_errors'].values()) for g in report['groups']),
                    maximum_sequential_unigram_loss_error=max(g['maximum_unigram_realized_loss_error'] for g in report['groups']),
                    clipped=False, renormalized=False, new_sampling=False),
        scope='768 frozen histories, 256 at each L=0,2,3, times1..4; diagnostic reuse, not independent confirmation; no copying onset inferred.')
    (destination/'validation.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


def score_sequence(tokens, transition, catalog, unigram_kernel):
    """Time t predicts X[t] from X[:t]; its oracle is T[X[t-1],:]."""
    table = unigram_kernel.new_table()
    table.observe(int(tokens[0]))
    bp, up, realized_check = [], [], []
    for t in range(1, 5):
        b = catalog.next_probs(tokens[:t])
        u = table.predictive()
        realized_check.append(abs(table.observe(int(tokens[t])) + np.log2(u[tokens[t]])))
        bp.append(b); up.append(u)
    bp, up = np.asarray(bp), np.asarray(up)
    oracle = transition[np.asarray(tokens[:4], dtype=int)]
    for name, values in (('Bayes', bp), ('unigram', up), ('oracle', oracle)):
        if (not np.isfinite(values).all() or np.min(values) <= 0
                or np.max(np.abs(values.sum(-1)-1)) > 1e-7):
            raise ValueError(f'{name} has invalid raw probabilities')
    if max(realized_check) > 1e-10:
        raise ValueError('Unigram prediction and sequential code loss disagree')
    bl = -np.sum(oracle * np.log2(bp), axis=-1)
    ul = -np.sum(oracle * np.log2(up), axis=-1)
    entropy = -np.sum(oracle * np.log2(oracle), axis=-1)
    return dict(bayes_probs=bp, unigram_probs=up, oracle_probs=oracle,
                bayes_expected_loss_bits=bl, unigram_expected_loss_bits=ul,
                oracle_entropy_bits=entropy, gap_bits=ul-bl,
                cumulative_gap_bits=np.cumsum(ul-bl),
                unigram_realized_loss_error=max(realized_check))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohorts', type=Path, default=ROOT/'output/shared_base_shuffle_download_2026-09-18/cohorts')
    p.add_argument('--manifest', type=Path, default=ROOT/'code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json')
    p.add_argument('--evaluations', type=Path, default=ROOT/'output/shared_base_shuffle_download_2026-09-18/eval')
    p.add_argument('--out', type=Path, default=ROOT/'output/shared_base_shuffle_early_reference_2026-09-18')
    p.add_argument('--publish-out', type=Path, help='Also publish a compact summary and validation receipt; raw arrays remain in --out.')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if (manifest['d'], manifest['n_tokens'], manifest['master_seed']) != (64, 1025, 20260930):
        raise ValueError('Unexpected frozen cohort')
    catalog = SharedBaseShortEvidence()
    store, _ = configure_backend(os.environ.get('LSA_STORE'), os.environ.get('PMM_SRC'))
    kernel = SeqLSA(64, Lmax=20, depths=range(1, 21), rmax=6, u_lo=-30., u_hi=30.)
    first_table = kernel.new_table()
    first_table.observe(0)
    repeat = dict(bayes=float(catalog.next_probs([0])[0]), unigram=float(first_table.predictive()[0]))
    a.out.mkdir(parents=True, exist_ok=True)
    groups = []
    for condition in manifest['conditions']:
        path = a.cohorts / (condition['id'] + '.npz')
        with np.load(path, allow_pickle=False) as z:
            data = {name: z[name].copy() for name in ('tokens', 'base', 'transition')}
        validate_cohort_arrays(data, condition)
        rows = [score_sequence(x, t, catalog, kernel) for x, t in zip(data['tokens'], data['transition'])]
        arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        paired_models = []
        for seed in (0, 1):
            metrics_for_seed, model_files = {}, {}
            for role, reference in (('shuffled','unigram_expected_loss_bits'), ('ordered','bayes_expected_loss_bits')):
                folder = a.evaluations/f'{role}_seed{seed}'
                model_path = folder/(condition['id']+'.npz')
                protocol = json.loads((folder/'protocol.json').read_text())
                status = json.loads((folder/'status.json').read_text())
                if (status['complete'] is not True or status['scientific_results'] is not True
                        or protocol['manifest_sha256'] != sha(a.manifest)
                        or protocol['role'] != role or protocol['seed'] != seed
                        or protocol['evaluation_code_sha256'] != sha(ROOT/'code/transformer/shared_base_shuffle_eval.py')
                        or protocol['model_code_sha256'] != sha(ROOT/'code/transformer/model.py')):
                    raise ValueError('Model evaluation identity or completion mismatch')
                with np.load(model_path, allow_pickle=False) as z:
                    if (str(z['protocol_sha256']) != sha(folder/'protocol.json')
                            or str(z['source_tokens_sha256']) != condition['arrays_sha256']['tokens']
                            or not np.array_equal(z['times'], np.arange(1, 1025))
                            or z['expected_loss_bits'].shape != (256,1024)):
                        raise ValueError('Model/source pairing mismatch')
                    expected = z['expected_loss_bits'][:,:4].copy()
                if not np.isfinite(expected).all():
                    raise ValueError('Nonfinite transformer loss')
                name = ('unigram' if role == 'shuffled' else 'bayes') + f'_minus_{role}_seed{seed}'
                arrays[name] = arrays[reference]-expected
                metrics_for_seed[name] = statistics(arrays[name])
                metrics_for_seed[role+'_expected_loss_bits'] = statistics(expected)
                model_files[role] = dict(path=str(model_path), sha256=sha(model_path), protocol_sha256=sha(folder/'protocol.json'))
            paired_models.append(dict(seed=seed, metrics=metrics_for_seed, model_files=model_files))
        np.savez_compressed(a.out/(condition['id']+'.npz'), times=np.arange(1, 5), **arrays)
        metrics = {name: statistics(arrays[name]) for name in ('bayes_expected_loss_bits',
                   'unigram_expected_loss_bits', 'gap_bits', 'cumulative_gap_bits')}
        metrics['bayes_regret_bits'] = statistics(arrays['bayes_expected_loss_bits']-arrays['oracle_entropy_bits'])
        metrics['unigram_regret_bits'] = statistics(arrays['unigram_expected_loss_bits']-arrays['oracle_entropy_bits'])
        group = dict(id=condition['id'], depth=condition['depth'], n_sequences=len(rows), metrics=metrics,
                     paired_transformer_comparisons=paired_models,
                     source_sha256=sha(path), arrays_sha256=condition['arrays_sha256'],
                     maximum_normalization_errors={name: float(np.max(np.abs(arrays[name].sum(-1)-1)))
                            for name in ('bayes_probs', 'unigram_probs', 'oracle_probs')},
                     maximum_unigram_realized_loss_error=float(arrays['unigram_realized_loss_error'].max()))
        groups.append(group)
        print(json.dumps(dict(condition=group['id'], metrics=metrics)), flush=True)
    report = dict(schema='shared-base-shuffle-early-reference-v1', complete=True,
        times=[1, 2, 3, 4], manifest_sha256=sha(a.manifest), script_sha256=sha(__file__),
        bayes=dict(prior=catalog.validation['prior'], method='Unchanged validated full-prior catalog lookup; no posterior sampling.',
                   validation=catalog.validation),
        first_prediction_repeat_probabilities=repeat,
        unigram=dict(prior='Normalized iid product-exponential weights; uniform coding-depth mixture 1..20.',
                     method='Existing SeqLSA pooled count table.', depths=list(range(1,21)),
                     u_bounds=[-30.,30.], u_step=kernel.h,
                     sealed_store_receipts_sha256={name:sha(store/name) for name in ('manifest.json','anchors.json')},
                     code_sha256={name:sha(ROOT/'code/transformer'/name) for name in ('lsa_seq.py','phi_source.py','eval_conditional_crossing.py')}),
        scoring='At t=1..4, X1..Xt have been observed; score expected loss for X(t+1) using T[Xt,:]. X1 is supplied and updates unigram.',
        sign='Positive gap is unigram expected loss minus full informed shared-base Bayes expected loss; positive favors informed Bayes.',
        uncertainty='Paired standard errors across the same 256 independent source sequences per condition; cumulative losses summed per sequence before standard error.',
        limitation='Diagnostic reuse; not independent confirmation. No Bayesian values beyond t=4. No conversion of archived row-copier crossings.',
        groups=groups)
    (a.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if a.publish_out:
        publish(report, a.out, a.publish_out)


if __name__ == '__main__':
    main()
