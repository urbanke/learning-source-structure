"""Replay the accepted paired copying endpoint at t=32; no cumulative claim."""
from pathlib import Path
import gzip, hashlib, json
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def replay(summary, arrays):
    require(summary['schema'] == 'copying-endpoint32-complete-cohort-v1'
        and all(summary[k] is True for k in ('complete', 'raw_replay_passed',
        'aggregate_precision_passed', 'population_reference_approved'))
        and not any(summary[k] for k in ('missing', 'failed_audits', 'failed_paired_orders')),
        'Endpoint32 is not a complete accepted calculation')
    require(summary['times'] == [32] and summary['original_histories'] == 768
        and summary['banks'] == 256 and summary['posterior_tasks'] == 1536
        and summary['cumulative_claim'] is False and summary['payoff_forecast_approved'] is False
        and np.array_equal(arrays['times'], [32]), 'Endpoint32 scope changed')
    require({g['condition'] for g in summary['groups']} == {'a1_L0', 'a1_L2', 'a1_L3'}
        and len(summary['groups']) == 3, 'Endpoint32 conditions changed')
    result, checks = {}, {}
    for group in summary['groups']:
        condition = group['condition']; depth = int(condition.split('L')[-1])
        scores = {n: arrays[condition+'_'+n] for n in ('stationary_loss', 'bayes_loss_a32',
            'ordered_seed0', 'shuffled_seed0', 'ordered_seed1', 'shuffled_seed1')}
        require(all(v.shape == (256, 1) and np.isfinite(v).all() for v in scores.values()),
            'Endpoint32 must retain all256 paired histories')
        gains = {'bayes': scores['stationary_loss']-scores['bayes_loss_a32']}
        gains.update({f'neural_seed{s}': scores[f'shuffled_seed{s}']-scores[f'ordered_seed{s}'] for s in (0, 1)})
        record = {}
        for name, gain in gains.items():
            record[name+'_mean'] = float(gain.mean())
            record[name+'_source_se'] = float(gain.std(ddof=1)/16)
        gate = group['instantaneous']
        mean, se, allowance, budget = (np.asarray(gate[k]) for k in
            ('mean_bits', 'source_se_bits', 'numerical_allowance_bits', 'numerical_budget_bits'))
        require(gate['passed'] is True and all(v.shape == (1,) and np.isfinite(v).all()
            for v in (mean, se, allowance, budget)), 'Endpoint32 acceptance is incomplete')
        error = max(abs(record['bayes_mean']-mean[0]), abs(record['bayes_source_se']-se[0]))
        require(error < 2e-12 and 0 <= allowance[0] <= budget[0]
            and abs(budget[0]-min(.01/3, .25*record['bayes_source_se'])) < 2e-14,
            'Endpoint32 scores or unchanged numerical budget differ')
        result[depth] = record; checks[f'copying_endpoint32_L{depth}_max_error'] = float(error)
    return result, checks

def load(source):
    source = Path(source); paths = [source/'paired_scores.npz', source/'provenance.json.gz']
    with gzip.open(paths[1], 'rt') as handle:
        p = json.load(handle)
    review = p['independent_replay']
    require(p['schema'] == 'copying-endpoint32-figure-input-v1' and p['passed'] is True
        and p['paired_scores_sha256'] == sha(paths[0]) and p['adapter_sha256'] == sha(__file__)
        and review['accepted'] is True and review['instantaneous_approved'] is True
        and review['cumulative_approved'] is False and all(review[k] is True for k in
        ('source_mapping_replayed', 'task_seed_inventory_replayed', 'all_neural_scores_bitwise_replayed',
        'aggregate_allowance_replayed')), 'Endpoint32 compact release or independent review changed')
    s = p['source_summary']; files = review['files']
    require(hashlib.sha256(p['source_summary_text'].encode()).hexdigest()
        == files['release32/scientific/summary.json'] and json.loads(p['source_summary_text']) == s
        and s['plan_id'] == review['scientific_plan_id']
        and s['arrays_sha256'] == files['release32/scientific/arrays.npz']
        and s['input_bindings_sha256'] == files['release32/scientific/input_bindings.json'],
        'Endpoint32 source binding changed')
    with np.load(paths[0], allow_pickle=False) as arrays:
        result, checks = replay(s, arrays)
    return result, checks, paths
