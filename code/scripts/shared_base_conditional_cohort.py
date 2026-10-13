"""Fixed-condition shared-base sources and frozen transformer predictions.

Source settings describe evaluation only. Both transformers and any reference
predictor retain the original training prior. Source identities reuse the
conditional benchmark exactly and do not depend on cohort size or time grid.
Bayesian reference sampling is deliberately a separate computation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'code/transformer'
sys.path.insert(0, str(HERE))
from conditional_shared_base_benchmark import draw, identity as benchmark_identity
from eval_shared_base_models import (SOURCE_SCHEMA, atomic_json, atomic_npz,
    fingerprint, models, sha_array, sha_file, validate_source_record)
from source import Source

SCHEMA = 'shared-base-conditional-cohort-v1'
DEFAULT_RAW = ROOT / 'output/shared_base_conditional_curves_2026-09-15'
DEFAULT_RESULTS = ROOT / 'results/transformer/shared_base_conditional_curves'
TIMES = [1, 4, 16, 64, 256, 1024]
N_TOKENS = 1025


def setting_folder(out, alpha, depth):
    return Path(out) / f'a{alpha:g}_L{depth}'


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen record differs: {path}')
    else:
        atomic_json(path, value)


def protocol(alpha, depth, seed, times):
    return dict(schema=SCHEMA, alpha=float(alpha), depth=int(depth), master_seed=int(seed),
        d=64, n_tokens=N_TOKENS, times=list(times),
        source_identity_rule='conditional_shared_base_benchmark.identity(alpha, depth, index, master_seed)',
        source_template=benchmark_identity(alpha, depth, 0, seed),
        reference_prior=dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
        conditions_hidden_from_predictors=True,
        sequence_sampling='Independent source matrices, paths and relabelings for every declared index; retain every index.',
        scoring='After t observed tokens, compare the true transition row with prediction at model position t-1; target token has zero-based index t.')


def prepare_setting(out, alpha, depth, seed, times, n_seq):
    folder = setting_folder(out, alpha, depth)
    specification = protocol(alpha, depth, seed, times)
    immutable_json(folder / 'protocol.json', specification)
    entries = []
    replays = []
    old_folder = ROOT / 'output/conditional_shared_base_benchmark_2026-09-15' / f'a{alpha:g}_L{depth}'
    for index in range(n_seq):
        ident = benchmark_identity(alpha, depth, index, seed)
        path = folder / 'data' / f'seq_{index:06d}.json'
        old_path = old_folder / 'data' / path.name
        if path.exists():
            record = validate_source_record(json.loads(path.read_text()), ident)
        elif old_path.exists():
            record = validate_source_record(json.loads(old_path.read_text()), ident)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_path, path)
        else:
            tokens, q, transition, permutation = draw(alpha, depth, ident['sequence_seed'])
            record = dict(schema=SOURCE_SCHEMA, identity=ident, source_id=fingerprint(ident),
                tokens=tokens.tolist(), tokens_sha256=sha_array(tokens, '<i2'),
                generated_source=dict(parameters=dict(alpha=alpha, ls=depth, perm=permutation.tolist())),
                oracle=dict(initial_probs=q.tolist(), transition_probs=transition.tolist(),
                    initial_probs_sha256=sha_array(q, '<f8'), transition_probs_sha256=sha_array(transition, '<f8')))
            validate_source_record(record, ident)
            atomic_json(path, record)
        # Replay the original public source generator, rather than only the
        # helper that constructed the stored source and oracle.
        replay, params = Source(64, N_TOKENS, alpha=alpha, ls=depth, permute=True).batch(
            1, np.random.default_rng(ident['sequence_seed']))
        if not np.array_equal(replay[0], record['tokens']):
            raise ValueError('original Source.batch replay failed')
        if not np.array_equal(params['perm'][0], record['generated_source']['parameters']['perm']):
            raise ValueError('source relabeling replay failed')
        if old_path.exists() and sha_file(path) != sha_file(old_path):
            raise ValueError('old benchmark source record was not preserved byte for byte')
        replays.append(index)
        entries.append(dict(index=index, source_id=record['source_id'], file_sha256=sha_file(path),
                            path=str(path.resolve())))
    declaration = dict(schema=SCHEMA, protocol_sha256=sha_file(folder / 'protocol.json'),
        n_sequences=n_seq, prefixes=list(times), n_tokens_per_sequence=N_TOKENS,
        alpha=float(alpha), depth=int(depth), master_seed=seed, records=entries,
        source_replay=dict(passed=True, n_sequences=len(replays), method='Original Source.batch with each immutable sequence seed'))
    immutable_json(folder / f'cohort_n{n_seq:06d}.json', declaration)
    live = folder / 'cohort.json'
    if live.exists():
        previous = json.loads(live.read_text())
        if previous['protocol_sha256'] != declaration['protocol_sha256']:
            raise ValueError('cohort protocol changed')
        common = min(previous['n_sequences'], n_seq)
        if previous['records'][:common] != entries[:common]:
            raise ValueError('cohort changed existing source records')
        if previous['n_sequences'] > n_seq:
            raise ValueError('do not shrink a declared cohort')
    atomic_json(live, declaration)
    print(json.dumps(dict(stage='prepare', folder=str(folder), n_sequences=n_seq,
                          source_replays_passed=len(replays), times=times)), flush=True)
    return folder


def load_source(folder, index):
    """Return a validated source; independent of frozen-model availability."""
    folder = Path(folder)
    specification = json.loads((folder / 'protocol.json').read_text())
    cohort = json.loads((folder / 'cohort.json').read_text())
    if specification['schema'] != SCHEMA or cohort['schema'] != SCHEMA:
        raise ValueError('unexpected cohort schema')
    if cohort['protocol_sha256'] != sha_file(folder / 'protocol.json') or index not in range(cohort['n_sequences']):
        raise ValueError('invalid cohort protocol or sequence index')
    entry = cohort['records'][index]
    path = folder / 'data' / f'seq_{index:06d}.json'
    if entry['index'] != index or sha_file(path) != entry['file_sha256']:
        raise ValueError('immutable source record changed')
    ident = benchmark_identity(specification['alpha'], specification['depth'], index, specification['master_seed'])
    record = validate_source_record(json.loads(path.read_text()), ident)
    if record['source_id'] != entry['source_id']:
        raise ValueError('source identity differs from declaration')
    return record


def inputs(folder, index, prefix):
    """Benchmark-compatible tuple: tokens, oracle, model predictions, identity.

    The returned identity omits cohort size. Adding independent later source
    indices therefore cannot invalidate an existing reference. Observed tokens
    alone are to be passed to reference inference; oracle/model outputs are
    returned for later scoring and precision checks.
    """
    folder = Path(folder)
    specification = json.loads((folder / 'protocol.json').read_text())
    if prefix not in specification['times']:
        raise ValueError('undeclared prediction time')
    source = load_source(folder, index)
    tokens = np.asarray(source['tokens'], dtype=int)[:prefix]
    oracle = np.asarray(source['oracle']['transition_probs'])[tokens[-1]]
    predictions = {}
    for name in ['d64', 'd64_seed1']:
        path = folder / 'models' / name / f'seq_{index:06d}.json'
        report = json.loads(path.read_text())
        if report['cache_id'] != fingerprint(report['identity']) or report['source_id'] != source['source_id']:
            raise ValueError('model identity failed')
        if report['identity']['source_file_sha256'] != sha_file(folder / 'data' / path.name):
            raise ValueError('model source record mismatch')
        if report['prefix_tokens_sha256'][str(prefix)] != sha_array(tokens, '<i2'):
            raise ValueError('model prefix mismatch')
        j = report['prefixes'].index(prefix)
        if report['targets'][j] != source['tokens'][prefix] or report['target_indices'][j] != prefix:
            raise ValueError('model target mismatch')
        array_path = path.with_suffix('.npz')
        if sha_file(array_path) != report['arrays_sha256']:
            raise ValueError('model array changed')
        with np.load(array_path, allow_pickle=False) as arrays:
            prediction = arrays['predictive_probs'][j].copy()
            logp = arrays['logp'][j]
            if not np.array_equal(prediction, report['predictive_probs'][j]) or not np.array_equal(oracle, arrays['oracle_probs'][j]):
                raise ValueError('model/oracle readback failed')
            if np.max(np.abs(np.exp(logp) - prediction)) > 1e-15:
                raise ValueError('stored probabilities and log probabilities differ')
            if abs(-logp[source['tokens'][prefix]] / np.log(2) - arrays['selected_loss_bits'][j]) > 1e-12:
                raise ValueError('target loss indexing failed')
        if not np.isfinite(prediction).all() or np.any(prediction <= 0) or abs(prediction.sum() - 1) > 1e-12:
            raise ValueError('invalid transformer probability distribution')
        if not report['checks']['alignment']['passed']:
            raise ValueError('model causality/alignment check did not pass')
        predictions[name] = dict(prediction=prediction, path=str(path.resolve()), sha256=sha_file(path))
    input_identity = dict(protocol_sha256=sha_file(folder / 'protocol.json'), source_id=source['source_id'],
        source_file_sha256=sha_file(folder / 'data' / f'seq_{index:06d}.json'),
        tokens_sha256=source['tokens_sha256'], prefix_tokens_sha256=sha_array(tokens, '<i2'),
        index=index, prefix=prefix,
        model_files={k: dict(path=v['path'], sha256=v['sha256']) for k, v in predictions.items()})
    return tokens, oracle, predictions, input_identity


def evaluate_setting(folder, batch):
    cohort = json.loads((folder / 'cohort.json').read_text())
    records = []
    for index, entry in enumerate(cohort['records']):
        record = load_source(folder, index)
        records.append(dict(path=Path(entry['path']), record=record, file_sha256=entry['file_sha256']))
    return models(folder, records, cohort['prefixes'],
        [ROOT / 'results/transformer/runs/d64', ROOT / 'results/transformer/runs/d64_seed1'],
        device='cpu', batch=batch, torch_threads=1)


def moments(values):
    values = np.asarray(values)
    return dict(n=int(len(values)), mean=float(values.mean()),
                standard_error=float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)


def summarize(out, result_dir, alpha, depths):
    rows = []
    receipts = []
    first_path = ROOT / 'results/transformer/conditional_shared_base/first_step.json'
    first = json.loads(first_path.read_text())
    if not first['passed'] or first['time'] != 1 or first['prior'] != dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'):
        raise ValueError('first-step reference does not match full prior')
    first = first['reports'][-1]
    first_step_paired = []
    for depth in depths:
        folder = setting_folder(out, alpha, depth)
        cohort = json.loads((folder / 'cohort.json').read_text())
        times = cohort['prefixes']
        values = np.empty((cohort['n_sequences'], len(times), 2))
        bayes_first = np.empty(cohort['n_sequences'])
        receipt = dict(depth=depth, alpha=alpha, n_sequences=cohort['n_sequences'],
            cohort_sha256=sha_file(folder / 'cohort.json'), protocol_sha256=sha_file(folder / 'protocol.json'),
            source_and_model_inputs=[])
        for index in range(cohort['n_sequences']):
            for j, t in enumerate(times):
                tokens, oracle, predictions, ident = inputs(folder, index, t)
                for k, name in enumerate(['d64', 'd64_seed1']):
                    values[index, j, k] = np.sum(rel_entr(oracle, predictions[name]['prediction'])) / np.log(2)
                if t == 1:
                    bayes = np.full(64, first['unseen_probability'])
                    bayes[tokens[0]] = first['seen_probability']
                    bayes_first[index] = np.sum(rel_entr(oracle, bayes)) / np.log(2)
            receipt['source_and_model_inputs'].append(ident)
        if not np.isfinite(values).all() or np.min(values) < -1e-12:
            raise ValueError('invalid transformer regret')
        for j, t in enumerate(times):
            for k, name in enumerate(['d64', 'd64_seed1']):
                rows.append(dict(alpha=alpha, depth=depth, time=t, model=name, **moments(values[:, j, k])))
        for k, name in enumerate(['d64', 'd64_seed1']):
            first_step_paired.append(dict(alpha=alpha, depth=depth, time=1, model=name,
                bayes_regret=moments(bayes_first), transformer_regret=moments(values[:, times.index(1), k]),
                transformer_minus_bayes=moments(values[:, times.index(1), k] - bayes_first)))
        archive = folder / 'transformer_regrets.npz'
        atomic_npz(archive, times=np.asarray(times), regret_bits=values, bayes_regret_t1_bits=bayes_first,
                   models=np.asarray(['d64', 'd64_seed1']), source_indices=np.arange(cohort['n_sequences']))
        receipt['array_path'] = str(archive.resolve())
        receipt['array_sha256'] = sha_file(archive)
        receipt['input_digest'] = fingerprint(receipt.pop('source_and_model_inputs'))
        receipts.append(receipt)
    summary = dict(schema=SCHEMA, source='Original shared-base LSA', alpha=alpha, depths=depths,
        times=times, n_tokens_per_sequence=N_TOKENS,
        training='Two existing final d64 checkpoints, both trained on alpha Uniform[.5,1.5], depth Uniform{0,...,8}; no retraining.',
        evaluation='Condition on alpha and depth; average over independent matrices, sequences and label permutations. Neither predictor receives the selected alpha or depth.',
        statistic='Expected next-token regret in bits estimated from KL(true current transition row || predictive distribution); standard errors over independent sequences.',
        rows=rows, first_step_paired=first_step_paired,
        bayesian_reference_status='Only t=1 uses a validated full-family Bayesian reference. Other times are transformer-only and do not establish transformer-Bayes agreement.',
        checks=dict(source_replay=True, all_sources_retained=True, model_alignment_and_causality=True,
            model_and_source_identity=True, array_checksums=True, target_time_alignment=True,
            probability_normalization=True, first_step_prior_and_validation=True),
        provenance=dict(script_sha256=sha_file(__file__), first_step_reference_sha256=sha_file(first_path), cohorts=receipts))
    atomic_json(result_dir / 'summary.json', summary)
    lines = ['alpha\tdepth\ttime\tmodel\tn_sequences\tregret_bits\tstandard_error_bits']
    lines += [f"{r['alpha']:g}\t{r['depth']}\t{r['time']}\t{r['model']}\t{r['n']}\t{r['mean']:.12g}\t{r['standard_error']:.12g}" for r in rows]
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / 'transformer_regrets.tsv').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(stage='summary', validated_sequence_time_pairs=sum(r['n_sequences'] for r in receipts) * len(times),
                          summary=str((result_dir / 'summary.json').resolve()))), flush=True)


def validate(out, result_dir, alpha, depths):
    """Independent replay/reuse/extension checks in addition to summary checks."""
    checks = []
    old = ROOT / 'output/conditional_shared_base_benchmark_2026-09-15/a1_L4'
    if alpha == 1.0 and 4 in depths and (old / 'cohort.json').exists():
        folder = setting_folder(out, alpha, 4)
        old_cohort = json.loads((old / 'cohort.json').read_text())
        new_cohort = json.loads((folder / 'cohort.json').read_text())
        n = min(old_cohort['n_sequences'], new_cohort['n_sequences'])
        max_difference = 0.0
        for index in range(n):
            filename = f'seq_{index:06d}.json'
            if sha_file(old / 'data' / filename) != sha_file(folder / 'data' / filename):
                raise ValueError('old benchmark source bytes changed')
            for model in ['d64', 'd64_seed1']:
                previous = json.loads((old / 'models' / model / filename).read_text())
                new = json.loads((folder / 'models' / model / filename).read_text())
                for j, t in enumerate(previous['prefixes']):
                    if t not in new['prefixes']:
                        continue
                    k = new['prefixes'].index(t)
                    error = float(np.max(np.abs(np.asarray(previous['predictive_probs'][j]) - new['predictive_probs'][k])))
                    max_difference = max(max_difference, error)
                    if error > 1e-12:
                        raise ValueError('old and new predictions differ on the same sequence/time')
        checks.append(dict(check='Existing benchmark sources and overlapping model predictions',
            n_sources=n, n_models=2, max_probability_difference=max_difference, passed=True))
    with tempfile.TemporaryDirectory(prefix='conditional-cohort-') as temp:
        folder = prepare_setting(Path(temp), alpha, depths[0], 20260930, TIMES, 2)
        source_before = sha_file(folder / 'data/seq_000000.json')
        protocol_before = sha_file(folder / 'protocol.json')
        prepare_setting(Path(temp), alpha, depths[0], 20260930, TIMES, 3)
        if source_before != sha_file(folder / 'data/seq_000000.json') or protocol_before != sha_file(folder / 'protocol.json'):
            raise ValueError('cohort size changed a source or reference protocol identity')
        try:
            prepare_setting(Path(temp), alpha, depths[0], 20260930, [1, 2, 4, 16, 64, 256, 1024], 3)
        except ValueError:
            pass
        else:
            raise ValueError('changed frozen time grid was accepted')
    checks.append(dict(check='Cohort extension preserves identities; changed frozen time grid is rejected', passed=True))
    seeds = []
    expected_q = np.arange(1, 65, dtype=float) ** -alpha
    expected_q /= expected_q.sum()
    for depth in depths:
        folder = setting_folder(out, alpha, depth)
        cohort = json.loads((folder / 'cohort.json').read_text())
        for index in range(cohort['n_sequences']):
            record = load_source(folder, index)
            seeds.append(record['identity']['sequence_seed'])
            q = np.asarray(record['oracle']['initial_probs'])
            transition = np.asarray(record['oracle']['transition_probs'])
            if not np.allclose(np.sort(q), np.sort(expected_q), rtol=0, atol=1e-15):
                raise ValueError('ranked base law does not match declared alpha')
            if depth == 0 and not np.array_equal(transition, np.broadcast_to(q, transition.shape)):
                raise ValueError('L=0 rows do not equal the common base law')
    if len(set(seeds)) != len(seeds):
        raise ValueError('independent source seed collision')
    checks.append(dict(check='Distinct source seeds; ranked q matches alpha; L=0 rows equal q', n_sources=len(seeds), passed=True))
    atomic_json(result_dir / 'validation.json', dict(schema=SCHEMA, checks=checks, passed=True, script_sha256=sha_file(__file__)))
    print(json.dumps(dict(stage='validate', checks=checks)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare', 'models', 'summary', 'validate', 'all'], required=True)
    p.add_argument('--out', type=Path, default=DEFAULT_RAW)
    p.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--depths', nargs='+', type=int, default=[0, 1, 4, 8])
    p.add_argument('--seed', type=int, default=20260930)
    p.add_argument('--times', nargs='+', type=int, default=TIMES)
    p.add_argument('--n-seq', type=int, default=256)
    p.add_argument('--batch', type=int, default=4)
    args = p.parse_args()
    if not .5 <= args.alpha <= 1.5 or args.n_seq < 1 or args.batch < 1 or args.seed < 0:
        p.error('invalid alpha, count, batch or seed')
    if args.depths != sorted(set(args.depths)) or any(d not in range(9) for d in args.depths):
        p.error('depths must be distinct and increasing in 0,...,8')
    if args.times != sorted(set(args.times)) or not args.times or args.times[0] != 1 or args.times[-1] > 1024:
        p.error('times must start at1 and be distinct and increasing, at most1024')
    if args.stage in ['all', 'prepare']:
        for depth in args.depths:
            prepare_setting(args.out, args.alpha, depth, args.seed, args.times, args.n_seq)
    if args.stage in ['all', 'models']:
        for depth in args.depths:
            evaluate_setting(setting_folder(args.out, args.alpha, depth), args.batch)
    if args.stage in ['all', 'summary']:
        summarize(args.out, args.results, args.alpha, args.depths)
    if args.stage in ['all', 'validate']:
        validate(args.out, args.results, args.alpha, args.depths)


if __name__ == '__main__':
    main()
