"""Calibrate a source-derived stationary-frequency comparator, at times 1..4.

The latent prior is the original shared-base prior. Given (q,T), the comparator
uses X1~q and independent subsequent tokens from the stationary law nu of T.
It is NOT exact Bayes for finite tail shuffling. All comparisons reuse the
same 768 frozen ordered histories; no neural scores tune the reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/transformer'))
from shared_base_short_evidence import SharedBaseShortEvidence, _canonical, _patterns
from shared_base_frequency_reference import stationary_iid_joint, count_only_original_probs
from source import Source

BASE = ROOT/'output/shared_base_shuffle_download_2026-09-18'
MANIFEST = ROOT/'code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json'
PRIOR = ROOT/'output/multifamily_results_2026-09-16/heldout/shared_base_prior.npz'
STAGES = {'saved': (8, None, None), 'stratified': (8, 8, 16), 'refined': (16, 16, 16)}
WORDS = [p for n in range(1, 6) for p in _patterns(n)]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def stationary(transition):
    t = np.asarray(transition, dtype=np.float64)
    d = t.shape[-1]
    require(t.ndim == 3 and t.shape[-2] == d and np.isfinite(t).all()
            and np.min(t) > 0 and np.max(abs(t.sum(-1)-1)) < 1e-12, 'Invalid source matrices')
    a = t.transpose(0, 2, 1)-np.eye(d)
    a[:, -1] = 1
    rhs = np.zeros((len(t), d, 1)); rhs[:, -1, 0] = 1
    nu = np.linalg.solve(a, rhs)[..., 0]
    residual = np.sum(abs(np.einsum('bi,bij->bj', nu, t)-nu), axis=-1)
    require(np.min(nu) > 0 and np.isfinite(nu).all()
            and np.max(abs(nu.sum(-1)-1)) < 1e-12 and residual.max() < 1e-11,
            'Stationary solve failed; never clip or normalize a failed solve')
    return nu, float(residual.max())


def joint_vector(q, nu):
    return np.asarray([np.mean(stationary_iid_joint(word, q, nu)) for word in WORDS])


def bank(stage, out):
    banks, order, repetitions = STAGES[stage]
    path = out/'bank.npz'
    require(not path.exists(), 'Use a new output directory; bank already exists')
    records = []; residual = 0.; draws = 0
    q_records = []; nu_records = []
    source = Source(64, 2, permute=False)
    if stage == 'saved':
        metadata = json.loads(PRIOR.with_suffix('.json').read_text())
        require(metadata['condition'] == dict(id='shared_base_prior', family='shared_base', kind='prior', n_seq=1024)
                and metadata['data_sha256'] == sha(PRIOR), 'Independent prior-bank identity differs')
        with np.load(PRIOR, allow_pickle=False) as z:
            q = z['initial_probs']; t = z['transition_probs']
        require(q.shape == (1024, 64) and t.shape == (1024, 64, 64), 'Prior bank shape differs')
        nu, residual = stationary(t)
        for i in range(banks):
            sl = slice(128*i, 128*(i+1)); records.append(joint_vector(q[sl], nu[sl]))
        q_records.append(q); nu_records.append(nu); draws = len(q)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        for i in range(banks):
            total = np.zeros(len(WORDS))
            for a, (node, weight) in enumerate(zip(nodes, weights)):
                alpha = 1 + node/2
                for depth in range(9):
                    rng = np.random.default_rng(np.random.SeedSequence([20260919, 0x434F5059, order, i, a, depth]))
                    n = 1 if depth == 0 else repetitions
                    q, t = source.matrices(np.full(n, alpha), np.full(n, depth), rng)
                    nu, r = stationary(t); residual = max(residual, r)
                    total += (weight/18) * joint_vector(q, nu)
                    draws += n; q_records.append(q); nu_records.append(nu)
            records.append(total)
            print(json.dumps(dict(stage=stage, bank=i+1, banks=banks, matrices=draws)), flush=True)
    records = np.asarray(records)
    require(np.isfinite(records).all() and records.min() > 0, 'Invalid unnormalized evidence')
    np.savez_compressed(path, joint_banks=records, q=np.concatenate(q_records),
                        nu=np.concatenate(nu_records), word_json=np.asarray(json.dumps(WORDS)))
    return records, dict(banks=banks, alpha_order=order, matrices=draws,
                        maximum_stationary_residual_l1=residual, bank_sha256=sha(path),
                        source_sampling='Previously saved independent full-prior matrices' if stage=='saved'
                        else 'Independent source matrices, stratified by depth and Gaussian alpha quadrature')


def lookup(values):
    return {word: float(v) for word, v in zip(WORDS, values)}


def prediction(prefix, joints):
    seen = list(dict.fromkeys(map(int, prefix))); pattern = _canonical(prefix)
    denominator = joints[pattern]; k = len(seen)
    p = np.full(64, joints[pattern+(k,)]/denominator)
    p[seen] = [joints[pattern+(i,)]/denominator for i in range(k)]
    require(np.isfinite(p).all() and p.min() > 0 and abs(p.sum()-1) < 1e-9,
            'Predictive normalization failure; no clipping or renormalization permitted')
    return p


def se(values):
    return np.std(values, axis=0, ddof=1)/math.sqrt(len(values))


def crossing(mean, tolerance=1e-12):
    a = np.asarray(mean)
    upward = (np.flatnonzero((a[:-1] <= tolerance) & (a[1:] > tolerance))+2).tolist()
    return dict(status='positive_initially' if a[0] > tolerance else 'delayed_mean_crossing'
                if upward else 'no_positive_mean_crossing_by_t4', upcrossings=upward,
                last_mean=float(a[-1]), horizon=4, equality_tolerance_bits=tolerance)


def summarize(values, prior_leave_one_out=None):
    mean = values.mean(0)
    result = dict(mean=mean.tolist(), paired_sequence_se=se(values).tolist(),
                  descriptive_crossing=crossing(mean))
    if prior_leave_one_out is not None:
        x = np.asarray(prior_leave_one_out); center = x.mean(0)
        error = np.sqrt((len(x)-1)/len(x)*np.sum((x-center)**2, axis=0))
        result['prior_bank_mcse'] = error.tolist()
    return result


def score(records, out):
    manifest = json.loads(MANIFEST.read_text()); catalog = SharedBaseShortEvidence()
    full = lookup(records.mean(0))
    jackknife = [lookup((records.sum(0)-row)/(len(records)-1)) for row in records]
    count_cache = {p:count_only_original_probs(p, catalog.log_joint)
                   for t in range(1,5) for p in _patterns(t)}
    groups = []; inputs = {str(MANIFEST.relative_to(ROOT)): sha(MANIFEST)}
    for condition in manifest['conditions']:
        name = condition['id']; source_path = BASE/'cohorts'/(name+'.npz')
        inputs[str(source_path.relative_to(ROOT))] = sha(source_path)
        with np.load(source_path, allow_pickle=False) as z:
            tokens = z['tokens']; transitions = z['transition']
            for field in ('tokens','base','transition'):
                digest = hashlib.sha256(np.asarray(z[field], dtype='<i8' if field=='tokens' else '<f8').tobytes()).hexdigest()
                require(digest == condition['arrays_sha256'][field], 'Frozen cohort differs: '+field)
        require(tokens.shape == (256, 1025), 'Incomplete scoring cohort')
        loss = {k: np.empty((256, 4)) for k in ('stationary','bayes','count_only')}
        loo = np.empty((len(records), 256, 4)); maximum_error = 0.
        for i, x in enumerate(tokens):
            for t in range(1, 5):
                oracle = transitions[i, x[t-1]]
                seen = list(dict.fromkeys(map(int,x[:t])))
                canonical_count = count_cache[_canonical(x[:t])]
                count_prediction = np.full(64,canonical_count[len(seen)])
                count_prediction[seen] = canonical_count[:len(seen)]
                predictors = dict(stationary=prediction(x[:t], full), bayes=catalog.next_probs(x[:t]),
                    count_only=count_prediction)
                for role, p in predictors.items():
                    maximum_error = max(maximum_error, abs(float(p.sum())-1))
                    loss[role][i,t-1] = -np.sum(oracle*np.log2(p))
                for j, table in enumerate(jackknife):
                    loo[j,i,t-1] = -np.sum(oracle*np.log2(prediction(x[:t], table)))
        arrays = dict(times=np.arange(1,5), **loss, stationary_leave_one_bank_out=loo)
        comparisons = {}
        for role in ('stationary','count_only'):
            gap = loss[role]-loss['bayes']
            uncertainty = (loo-loss['bayes']).mean(1) if role=='stationary' else None
            comparisons[role+'_minus_bayes'] = summarize(gap, uncertainty)
            comparisons[role+'_minus_bayes_cumulative'] = summarize(np.cumsum(gap,axis=1),
                np.cumsum(uncertainty,axis=1) if uncertainty is not None else None)
        for seed in (0,1):
            model = {}
            for role in ('ordered','shuffled'):
                folder = BASE/'eval'/f'{role}_seed{seed}'; path=folder/(name+'.npz')
                protocol_path=folder/'protocol.json'; status=json.loads((folder/'status.json').read_text())
                protocol=json.loads(protocol_path.read_text())
                require(status['complete'] is True and status['scientific_results'] is True
                    and protocol['manifest_sha256']==sha(MANIFEST) and protocol['role']==role
                    and protocol['seed']==seed
                    and protocol['evaluation_code_sha256']==sha(ROOT/'code/transformer/shared_base_shuffle_eval.py')
                    and protocol['model_code_sha256']==sha(ROOT/'code/transformer/model.py'), 'Model provenance mismatch')
                with np.load(path, allow_pickle=False) as z:
                    require(str(z['protocol_sha256'])==sha(protocol_path)
                        and str(z['source_tokens_sha256'])==condition['arrays_sha256']['tokens']
                        and np.array_equal(z['times'],np.arange(1,1025))
                        and z['expected_loss_bits'].shape==(256,1024),'Model/source pairing mismatch')
                    model[role]=z['expected_loss_bits'][:,:4].copy()
                require(np.isfinite(model[role]).all(), 'Nonfinite model loss')
                inputs[str(path.relative_to(ROOT))] = sha(path)
                arrays[f'{role}_seed{seed}']=model[role]
            for role in ('stationary','count_only','bayes'):
                comparisons[f'{role}_minus_shuffled_seed{seed}']=summarize(loss[role]-model['shuffled'],
                    (loo-model['shuffled']).mean(1) if role=='stationary' else None)
            gain=model['shuffled']-model['ordered']
            comparisons[f'neural_gain_seed{seed}']=summarize(gain)
            comparisons[f'neural_cumulative_gain_seed{seed}']=summarize(np.cumsum(gain,axis=1))
        np.savez_compressed(out/(name+'.npz'),**arrays)
        groups.append(dict(id=name,depth=condition['depth'],n_sequences=256,
                           maximum_predictive_normalization_error=maximum_error,comparisons=comparisons))
    return groups,inputs,catalog.validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,default='saved');p.add_argument('--out',type=Path,
        default=ROOT/'output/copying_frequency_calibration_2026-09-19')
    a=p.parse_args();out=a.out/a.stage;out.mkdir(parents=True,exist_ok=True)
    require(not (out/'plan.json').exists(),'Stage already started; use its saved result or a separate output')
    started=time.monotonic()
    plan=dict(schema='copying-frequency-calibration-plan-v1',stage=a.stage,settings=STAGES[a.stage],
        runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,platform=sys.platform),
        random_stream='SeedSequence[20260919,0x434F5059,alpha_order,bank,alpha_node,depth]',
        times=[1,2,3,4],conditions=['a1_L0','a1_L2','a1_L3'],n_per_condition=256,
        prior=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'),
        code_sha256={str(x.relative_to(ROOT)):sha(x) for x in (Path(__file__),ROOT/'code/transformer/shared_base_frequency_reference.py',ROOT/'code/transformer/source.py')},
        cohort_manifest_sha256=sha(MANIFEST),prior_bank_sha256=sha(PRIOR) if a.stage=='saved' else None,
        new_model_training=False,neural_fit=False,finite_shuffle_exact=False,
        empirical_bayes=False,scientific_timing_validated=False)
    write(out/'plan.json',plan)
    records,diagnostic=bank(a.stage,out)
    groups,inputs,catalog=score(records,out)
    repeat=prediction([0],lookup(records.mean(0)))[0]
    result=dict(schema='copying-frequency-calibration-v1',complete=True,stage=a.stage,
        plan_sha256=sha(out/'plan.json'),times=[1,2,3,4],reference=diagnostic,
        first_repeat_probability=float(repeat),exact_bayes_validation=catalog,
        scoring_inputs_sha256=inputs,groups=groups,elapsed_seconds=time.monotonic()-started,
        interpretation='Stationary-iid comparator with original q first-token law, not exact finite shuffled-training Bayes. Count-only companion conditions the original ordered source on first token and remaining prefix counts.',
        uncertainty='Paired whole-sequence SE and separate leave-one-prior-bank-out Monte Carlo SE. Alpha quadrature needs an independent refinement. Mean crossings are descriptive only; dense Bayes after t4 and confirmation remain missing.',
        scientific_timing_validated=False,new_paper_claim=False)
    write(out/'summary.json',result)
    print(json.dumps(dict(complete=True,stage=a.stage,first_repeat_probability=float(repeat),
        elapsed_seconds=result['elapsed_seconds'],scientific_timing_validated=False)),flush=True)


if __name__=='__main__':
    main()
