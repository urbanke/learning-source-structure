"""Independent read-only audit of the saved 256-sequence hidden-factor pilot.

No evaluator, model, or hidden-encoding enumeration implementation is imported.
Oracle/known-encoding/flat predictions and all summary statistics are recomputed.
The hidden-mixture computation and transformer inference themselves are not rerun.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TIMES = np.array([1, 4, 16, 64, 256, 1024])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


class Audit:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.hashes = {}
        self.maximum_errors = {}

    def raw(self, relative):
        raw = (self.directory/relative).read_bytes()
        self.hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, relative):
        return json.loads(self.raw(relative))

    def npz(self, relative, expected_hash):
        from io import BytesIO
        raw = self.raw(relative)
        require(hashlib.sha256(raw).hexdigest() == expected_hash, f'Hash mismatch: {relative}')
        with np.load(BytesIO(raw), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def close(self, name, actual, expected, tolerance=2e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), f'{name}: invalid shape/value')
        error = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        self.maximum_errors[name] = max(error, self.maximum_errors.get(name, 0.))
        require(error <= tolerance, f'{name}: discrepancy {error} exceeds {tolerance}')


def independent_references(tokens, encoding, first_matrix, second_matrix):
    inverse = np.argsort(encoding)
    coordinates = (inverse//8, inverse % 8)
    transition = (first_matrix[coordinates[0][:, None], coordinates[0][None, :]]
                  * second_matrix[coordinates[1][:, None], coordinates[1][None, :]])
    oracle_p = transition[tokens[TIMES-1]]
    oracle_cum = np.cumsum(-np.log2(transition[tokens[:-1], tokens[1:]]))[TIMES-1]
    # Closed-form Dirichlet integrals, independent of the evaluator's
    # sequential update implementation.
    logfactorial = np.array([math.lgamma(j+1) for j in range(1050)])
    def integrated_loss(counts, k):
        return -(k*math.lgamma(k) - logfactorial[counts.sum(1)+k-1].sum()
                 + logfactorial[counts].sum())/math.log(2)
    known_p, known_cum, flat_p, flat_cum = [], [], [], []
    for t in TIMES:
        pred, loss = np.ones(16), 0.
        for labels, k in zip(coordinates, (2, 8)):
            x = labels[tokens]
            counts = np.bincount(x[:t-1]*k+x[1:t], minlength=k*k).reshape(k, k)
            current = x[t-1]
            pred *= (counts[current, labels]+1)/(counts[current].sum()+k)
            counts[current, x[t]] += 1
            loss += integrated_loss(counts, k)
        known_p.append(pred)
        known_cum.append(loss)
        counts = np.bincount(tokens[:t-1]*16+tokens[1:t], minlength=256).reshape(16, 16)
        current = tokens[t-1]
        flat_p.append((counts[current]+1)/(counts[current].sum()+16))
        counts[current, tokens[t]] += 1
        flat_cum.append(integrated_loss(counts, 16))
    return dict(oracle=(oracle_p, oracle_cum),
                bayes_encoding_given=(np.asarray(known_p), np.asarray(known_cum)),
                flat_dirichlet=(np.asarray(flat_p), np.asarray(flat_cum)))


def run(directory):
    started = time.monotonic()
    audit = Audit(directory)
    summary = audit.json('summary_prefix_000256.json')
    cohort = audit.json('cohort.json')
    spec = cohort['identity']
    require(cohort['cohort_id'] == fingerprint(spec) and summary['cohort'] == cohort, 'Cohort identity mismatch')
    require(spec['dimensions'] == [2, 8] and spec['n_seq'] == 4096 and spec['length'] == 1025, 'Wrong source/cohort size')
    require(spec['first_token_scored'] is False and spec['production_size'] is True, 'Wrong scoring or production scope')
    require(summary['production_size'] is True and summary['evaluated_n_sequences'] == 256
            and summary['planned_n_sequences'] == 4096 and summary['partial_evaluation'] is True
            and summary['evaluation_complete'] is False and summary['two_repeat_comparison_complete'] is False,
            'Incorrect pilot completion flags')
    require(summary['bayes_only'] is False and summary['present_training_seeds'] == [0, 1]
            and summary['missing_training_seeds'] == [], 'Both trained seeds are required')
    audit.close('times', spec['times'], TIMES, 0)
    audit.close('summary times', summary['times'], TIMES, 0)
    data = audit.npz('cohort.npz', cohort['data_sha256'])
    tokens, encodings = data['tokens'], data['encoding']
    require(tokens.shape == (4096, 1025) and tokens.dtype.kind in 'iu'
            and np.all((0 <= tokens) & (tokens < 16)), 'Invalid source tokens')
    require(encodings.shape == (4096, 16) and np.all(np.sort(encodings, axis=1) == np.arange(16)), 'Invalid hidden encodings')
    entropy = []
    for k, key in ((2, 'first_matrix'), (8, 'second_matrix')):
        matrix = data[key]
        require(matrix.shape == (4096, k, k) and np.all(matrix > 0), 'Invalid source matrices')
        audit.close('source normalization', matrix.sum(-1), np.ones((4096, k)), 2e-12)
        entropy.append(-(matrix*np.log2(matrix)).sum(-1).mean(-1)/math.log2(k))
    entropy = np.column_stack(entropy)
    medians = np.median(entropy, axis=0)
    strata = 2*(entropy[:, 0] >= medians[0])+(entropy[:, 1] >= medians[1])
    audit.close('entropy', data['normalized_row_entropy'], entropy, 2e-12)
    audit.close('entropy medians', cohort['entropy_medians'], medians, 2e-12)
    audit.close('strata', data['stratum'], strata, 0)
    audit.close('stratum counts', cohort['stratum_counts'], np.bincount(strata, minlength=4), 0)
    require(len({hashlib.sha256(row.tobytes()).hexdigest() for row in tokens[:256]}) == 256, 'Repeated test sequences')

    def code_hash(name):
        return hashlib.sha256((ROOT/'code/transformer'/name).read_bytes()).hexdigest()
    require(spec['source_sha256'] == code_hash('hidden_factor.py'), 'Source implementation changed')
    manifests, records, record_hashes = {}, {}, {}
    for folder in ('references', 'models/seed0', 'models/seed1'):
        manifest = audit.json(folder+'/manifest.json')
        ident = manifest['identity']
        require(ident['cohort_id'] == cohort['cohort_id'] and manifest['evaluated_n_sequences'] == 256
                and manifest['planned_n_sequences'] == 4096 and manifest['complete'] is False
                and manifest['partial_evaluation'] is True, 'Incorrect prediction coverage')
        require(ident['evaluator_sha256'] == code_hash('eval_hidden_factor.py'), 'Evaluator changed')
        if folder == 'references':
            require(ident['source_sha256'] == code_hash('hidden_factor.py')
                    and ident['reference_sha256'] == code_hash('hidden_factor_gpu.py'), 'Reference implementation changed')
        else:
            require(ident['name'] == folder.split('/')[-1] and ident['checkpoint_step'] == 100000
                    and ident['model_sha256'] == code_hash('model.py'), 'Wrong recorded checkpoint')
            causal = manifest['causality_check']
            require(causal['passed'] is True and causal['max_absolute_logit_difference'] <= causal['tolerance'] <= 2e-5,
                    'Causal alignment check failed')
        require(ident['device'] == 'cuda', 'Unexpected evaluation device')
        rows, hashes = [], []
        for i in range(256):
            relative = f'{folder}/seq_{i:06d}'
            metadata = audit.json(relative+'.json')
            require(metadata['identity'] == dict(ident, sequence_index=i), 'Per-sequence identity mismatch')
            rows.append(audit.npz(relative+'.npz', metadata['data_sha256']))
            hashes.append(metadata['data_sha256'])
        manifests[folder], records[folder], record_hashes[folder] = manifest, rows, fingerprint(hashes)
    require(record_hashes['references'] == summary['reference_records_sha256'], 'Reference summary input hash mismatch')
    require({name: record_hashes['models/'+name] for name in ('seed0', 'seed1')} == summary['model_records_sha256'], 'Model summary input hash mismatch')
    require(manifests['models/seed0']['identity']['checkpoint_sha256'] != manifests['models/seed1']['identity']['checkpoint_sha256'], 'Repeated model weights')

    metrics = ('next_kl_bits', 'next_loss_bits', 'cumulative_loss_bits')
    names = ('oracle', 'bayes_hidden', 'bayes_encoding_given', 'flat_dirichlet', 'seed0', 'seed1')
    all_values = {name: {metric: [] for metric in metrics} for name in names}
    bound = math.log2(math.factorial(16)//(math.factorial(2)*math.factorial(8)))
    for i in range(256):
        ref = records['references'][i]
        xs = tokens[i]
        independent = independent_references(xs, encodings[i], data['first_matrix'][i], data['second_matrix'][i])
        oracle = independent['oracle'][0]
        audit.close('oracle probabilities', ref['oracle_probs'], oracle)
        for name, (pred, loss) in independent.items():
            audit.close('independent '+name+' prediction', ref[name+'_predictive'], pred)
            audit.close('independent '+name+' cumulative loss', ref[name+'_cumulative_loss_bits'], loss, 2e-8)
        for name in names:
            row = records['models/'+name][i] if name.startswith('seed') else ref
            prefix = '' if name.startswith('seed') else name+'_'
            pred = row[prefix+'predictive']
            audit.close('record times', row['times'], TIMES, 0)
            require(pred.shape == (6, 16) and np.isfinite(pred).all() and np.all(pred > 0), 'Invalid predictive distribution')
            audit.close('prediction normalization', pred.sum(1), np.ones(6), 4e-6)
            kl = np.maximum(0, np.sum(oracle*(np.log2(oracle)-np.log2(pred)), axis=1))
            loss = -np.log2(pred[np.arange(6), xs[TIMES]])
            audit.close('recomputed KL', row[prefix+'next_kl_bits'], kl)
            audit.close('recomputed next loss', row[prefix+'next_loss_bits'], loss)
            cumulative = row[prefix+'cumulative_loss_bits']
            require(np.isfinite(cumulative).all() and np.all(np.diff(cumulative) >= 0), 'Invalid cumulative loss')
            audit.close('first prediction loss', cumulative[:1], loss[:1])
            for metric in metrics:
                all_values[name][metric].append(row[prefix+metric])
        evidence = dict(zip(ref['hidden_evidence_times'].tolist(), ref['hidden_log_evidence_bits'].tolist()))
        audit.close('hidden evidence next loss', [-evidence[int(t+1)]+evidence[int(t)] for t in TIMES], ref['bayes_hidden_next_loss_bits'], 2e-8)
        audit.close('hidden evidence cumulative loss', [-evidence[int(t+1)]-4 for t in TIMES], ref['bayes_hidden_cumulative_loss_bits'], 2e-8)
        cost = ref['bayes_hidden_cumulative_loss_bits']-ref['bayes_encoding_given_cumulative_loss_bits']
        audit.close('encoding cost', cost, ref['encoding_uncertainty_cost_bits'])
        require(np.all(cost <= bound+2e-8), 'Encoding-mixture likelihood bound failed')
    arrays = {name: {metric: np.asarray(values) for metric, values in group.items()} for name, group in all_values.items()}
    for group in summary['groups']:
        mask = np.ones(256, dtype=bool) if group['id'] == 'full_prior' else strata[:256] == int(group['id'].split('_')[1])
        require(group['n_seq'] == int(mask.sum()) and group['status'] == 'complete', 'Incorrect stratum count')
        for kind in ('predictors', 'paired'):
            for name, statistics in group[kind].items():
                left, right = name.split('_minus_') if kind == 'paired' else (name, None)
                for metric in metrics:
                    values = arrays[left][metric][mask]
                    if right is not None:
                        values = values-arrays[right][metric][mask]
                    saved = statistics[metric]
                    require(saved['n_seq'] == len(values), 'Statistic sample count mismatch')
                    audit.close('summary means', saved['mean'], values.mean(0), 2e-9)
                    audit.close('summary paired/ordinary SE', saved['se'], values.std(0, ddof=1)/math.sqrt(len(values)), 2e-9)
    figures = audit.json('pilot_figures/receipt.json')
    require(figures['summary_sha256'] == audit.hashes['summary_prefix_000256.json'], 'Figure summary hash mismatch')
    require(figures['script_sha256'] == hashlib.sha256((ROOT/'code/scripts/hidden_factor_figures.py').read_bytes()).hexdigest(), 'Figure script changed')
    for name, expected in figures['outputs'].items():
        audit.raw('pilot_figures/'+name)
        require(audit.hashes['pilot_figures/'+name] == expected, 'Figure artifact hash mismatch')
    full = next(g for g in summary['groups'] if g['id'] == 'full_prior')
    return dict(schema='hidden-factor-independent-pilot-audit-v1', passed=True,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy_version=np.__version__,
        evaluated_sequences=256, planned_sequences=4096, sequence_length=1025, dimensions=[2, 8],
        prediction_records_checked=768, hashed_files=len(audit.hashes),
        input_tree_sha256=fingerprint(audit.hashes), summary_sha256=audit.hashes['summary_prefix_000256.json'],
        checkpoint_records={name: manifests['models/'+name]['identity'] for name in ('seed0', 'seed1')},
        maximum_absolute_errors=audit.maximum_errors, encoding_count=math.factorial(16)//(math.factorial(2)*math.factorial(8)),
        encoding_cost_bound_bits=bound, pilot_stratum_counts=np.bincount(strata[:256], minlength=4).tolist(),
        full_prior=full, elapsed_audit_seconds=time.monotonic()-started,
        scope='Independent saved-array/analytic-baseline/statistical audit; no transformer inference or full hidden-encoding enumeration replay.',
        limitations=['Raw trained checkpoint files were not part of the eval transfer; recorded distinct hashes and final steps checked, not weight bytes.',
                     'Only the first256 of the frozen4096 test sequences were evaluated; this is preliminary evidence.',
                     'Runtime not stored per sequence; job elapsed2:12:16 comes from user-provided Slurm accounting, not these arrays.'])


def corruption_checks(directory):
    """Read-only fault injections; never modify transferred scientific files."""
    from unittest.mock import patch
    original = Audit.json
    cases = [('summary_stat', 'summary_prefix_000256.json'), ('count', 'summary_prefix_000256.json'),
             ('npz_hash', 'references/seq_000000.json'), ('seed_identity', 'models/seed1/manifest.json')]
    rejected = []
    for kind, target in cases:
        def corrupted(self, relative, kind=kind, target=target):
            value = original(self, relative)
            if relative == target:
                if kind == 'summary_stat':
                    value['groups'][0]['paired']['seed0_minus_bayes_hidden']['cumulative_loss_bits']['mean'][-1] += 1
                elif kind == 'count':
                    value['evaluated_n_sequences'] = 255
                elif kind == 'npz_hash':
                    value['data_sha256'] = '0'*64
                elif kind == 'seed_identity':
                    value['identity']['name'] = 'seed0'
            return value
        try:
            with patch.object(Audit, 'json', corrupted):
                run(directory)
        except ValueError:
            rejected.append(kind)
        else:
            raise ValueError(f'Corruption was incorrectly accepted: {kind}')
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'output/hidden_factor_cluster_pilot_2026-09-16')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true', help='Also reject four deliberately corrupted inputs in memory')
    args = parser.parse_args()
    result = run(args.input)
    if args.self_test:
        result['passed_corruption_checks'] = corruption_checks(args.input)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('passed', 'prediction_records_checked', 'hashed_files', 'maximum_absolute_errors', 'elapsed_audit_seconds')}, indent=2))


if __name__ == '__main__':
    main()
