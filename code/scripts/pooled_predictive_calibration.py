"""Live full-prior SMC calibration against the existing exact short catalog.

Only [0] and the 15 length-four equality patterns are used. No new exact
integrals, transformer scores, long-context references or population results
are computed. Run and merge commands belong on compute nodes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
import uuid

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC
from shared_base_predictive_smc import estimate, pool_repetitions, checked_probs
from shared_base_short_evidence import SharedBaseShortEvidence

PLAN = ROOT / 'code/transformer/configs/pooled_predictive_calibration_plan_2026-09-18.json'
SCHEMA = 'pooled-predictive-exact-short-plan-v1'
ATTEMPTS = {
    'initial': dict(particles=2048, temperatures=32, moves=2, replicates=32),
    'stronger': dict(particles=8192, temperatures=64, moves=4, replicates=64),
}
CRITERIA = dict(tolerance_bits=.01, mcse_multiplier=3., minimum_replicates=32,
                minimum_effective_replica_count=8., maximum_likelihood_share=.2,
                normalization_tolerance=1e-7)
NUMERICS = dict(d=64, alpha_bounds=[.5, 1.5], alpha_order=16,
                depths=list(range(9)), depth_prior='uniform', ranks='uniform permutation',
                row_bounds=[-32., 32.], row_step=1/32, resample_fraction=.75)
CODE = ('code/scripts/pooled_predictive_calibration.py',
        'code/transformer/shared_base_predictive_smc.py',
        'code/transformer/shared_base_evidence_smc.py',
        'code/transformer/shared_base_mcmc.py', 'code/transformer/ranked_lsa.py',
        'code/transformer/weighted_lsa.py', 'code/transformer/shared_depth.py',
        'code/transformer/lsa_seq.py', 'code/transformer/phi_source.py',
        'code/transformer/shared_base_short_evidence.py',
        'code/scripts/shortprefix_shared_base_exact.py',
        'code/scripts/shortprefix_shared_base_summary.py',
        'code/scripts/conditional_shared_base_first_step.py')
REFERENCES = ('results/transformer/shared_base_conditional_curves/shortprefix_reference.json',
              'results/transformer/shared_base_conditional_curves/time4_summary.json',
              'results/transformer/conditional_shared_base/first_step.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cases():
    patterns = [()]
    for _ in range(4):
        patterns = [p + (a,) for p in patterns for a in range(max(p, default=-1)+2)]
    return [dict(id='n1_0', tokens=[0])] + [dict(id='n4_' + ''.join(map(str, p)), tokens=list(p)) for p in patterns]


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    require(plan['schema'] == SCHEMA and plan['cases'] == cases(), 'Unplanned short-prefix cases')
    require(plan['attempts'] == ATTEMPTS and plan['criteria'] == CRITERIA
            and plan['numerics'] == NUMERICS, 'Predeclared numerical settings changed')
    require(isinstance(plan['seed'], int) and not isinstance(plan['seed'], bool) and plan['seed'] >= 0,
            'Invalid master seed')
    require(plan['replicas_per_task'] == 4 and plan['seconds_per_replica'] == 1800
            and plan['concurrency'] == 64 and plan['allocation_minutes'] == 120,
            'Predeclared allocation bounds changed')
    require(set(plan['code_sha256']) == set(CODE) and set(plan['reference_sha256']) == set(REFERENCES),
            'Incomplete source/reference provenance')
    for name, expected in {**plan['code_sha256'], **plan['reference_sha256']}.items():
        require(sha(ROOT/name) == expected, 'Changed input: ' + name)
    return plan, sha(path)


def task_spec(plan, attempt, index):
    require(attempt in ATTEMPTS, 'Unknown calibration attempt')
    repetitions = plan['attempts'][attempt]['replicates']
    blocks = repetitions // plan['replicas_per_task']
    require(isinstance(index, int) and 0 <= index < len(plan['cases'])*blocks, 'Task index outside plan')
    case = plan['cases'][index // blocks]
    start = (index % blocks)*plan['replicas_per_task']
    return dict(index=index, case=case, replica_start=start, replica_stop=start+plan['replicas_per_task'])


def replica_seed(plan, attempt, case_id, replica):
    return int(digest(['pooled-predictive-exact-short-replicate-v1', plan['seed'],
                       attempt, case_id, replica])[:16], 16)


def dependencies(plan, store, src):
    item = plan['dependencies']
    manifest_path = ROOT/item['manifest']
    require(sha(manifest_path) == item['sha256'], 'Dependency manifest changed')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema'] == 'shared-base-dependency-manifest-v1', 'Unexpected dependency manifest')
    found = {}
    for name, record in manifest['files'].items():
        if name.startswith('src/'):
            path = src / Path(name).relative_to('src')
        else:
            require(name.startswith('tables/anchors_prod/'), 'Unexpected dependency member')
            path = store / Path(name).relative_to('tables/anchors_prod')
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == record['size_bytes']
                and sha(path) == record['sha256'], 'Dependency differs: ' + name)
        found[name] = record['sha256']
    return found


def numerical_environment(store, src):
    for name in ('PMM_PHI_BIAS', 'PMM_PHI_WAVE'):
        value = float(os.environ.get(name, '0') or 0)
        require(np.isfinite(value) and value == 0, 'Numerical perturbation forbidden: ' + name)
    # Use the same production kernel settings as the validated catalog/sampler.
    os.environ.update(LSA_STORE=str(store), PMM_SRC=str(src), PMM_UNIVERSAL_TABLES=str(store),
                      PMM_PHI_LADDER_EVERY='1', PMM_PHI_LADDER_DEGREE='11', PMM_PHI_SADDLE_MIN_L='54')
    names = ('PMM_PHI_BIAS', 'PMM_PHI_WAVE', 'PMM_PHI_WAVE_SCALE', 'PMM_PHI_LADDER',
             'PMM_PHI_LADDER_EVERY', 'PMM_PHI_LADDER_DEGREE', 'PMM_PHI_SADDLE_MIN_L',
             'PMM_SERIES_TAIL', 'PMM_KERNEL', 'PMM_INTERP_KERNEL', 'PMM_BUILD_EXACT')
    return {name: os.environ.get(name) for name in names}


def exact_reference(catalog, tokens):
    log_z = catalog.log_joint(tokens)
    p = checked_probs(catalog.next_probs(tokens), 64)
    joints = np.array([catalog.log_joint(tokens+[b]) for b in range(64)])
    require(np.max(np.abs(joints-log_z-np.log(p))) <= 1e-9, 'Exact catalog successor inconsistency')
    return dict(log_evidence=log_z, predictive_probs=p.tolist(), log_joint_next=joints.tolist())


def identity(plan, plan_hash, attempt, dependency_hashes, environment):
    return dict(schema='pooled-predictive-exact-short-identity-v1', plan_sha256=plan_hash,
                attempt=attempt, settings=plan['attempts'][attempt], numerics=plan['numerics'],
                code_sha256=plan['code_sha256'], reference_sha256=plan['reference_sha256'],
                dependency_sha256=dependency_hashes, numerical_environment=environment,
                runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                scientific_reference_validated=False)


def validate_record(record, plan, attempt, case, wanted_replica):
    require(record['replica'] == wanted_replica and record['case_id'] == case['id'], 'Replica identity changed')
    value = record['estimate']
    require(record['estimate_sha256'] == digest(value), 'Saved replica changed')
    require(value['schema'] == 'shared-base-predictive-smc-repetition-v1'
            and value['n_observed'] == len(case['tokens'])
            and value['seed'] == replica_seed(plan, attempt, case['id'], wanted_replica), 'Sampler identity changed')
    settings = plan['attempts'][attempt]
    require(all(value[k] == settings[k] for k in ('particles', 'temperatures', 'moves'))
            and value['resample_fraction'] == NUMERICS['resample_fraction']
            and value['scientific_reference_validated'] is False, 'Sampler settings changed')
    checked_probs(value['predictive_probs'], 64, CRITERIA['normalization_tolerance'])
    log_z = value['log_evidence']
    require(np.isfinite(log_z) and log_z <= 1e-12, 'Invalid absolute evidence')
    require(np.isfinite(value['maximum_conditional_normalization_error'])
            and 0 <= value['maximum_conditional_normalization_error'] <= CRITERIA['normalization_tolerance'],
            'Conditional prediction did not normalize')
    require(np.allclose(value['log_joint_next'], log_z+np.log(value['predictive_probs']), rtol=0, atol=1e-12),
            'Saved joint evidence differs from Z times p')
    for name in ('sampler_seconds', 'prediction_seconds', 'elapsed_seconds'):
        require(np.isfinite(value[name]) and value[name] >= 0, 'Invalid timing')
    require(len(value['effective_sample_sizes']) == settings['temperatures']
            and np.isfinite(value['effective_sample_sizes']).all()
            and np.min(value['effective_sample_sizes']) > 0, 'Invalid particle diagnostics')


def validate_report(report, expected_identity, spec, plan, attempt):
    require(report['schema'] == 'pooled-predictive-exact-short-shard-v1'
            and report['identity'] == expected_identity and report['task'] == spec,
            'Saved shard provenance differs')
    records = report['records']
    expected = list(range(spec['replica_start'], spec['replica_start']+len(records)))
    require(len(records) <= spec['replica_stop']-spec['replica_start'], 'Too many repetitions in shard')
    for record, replica in zip(records, expected):
        validate_record(record, plan, attempt, spec['case'], replica)
    require(report['complete'] == (len(records) == spec['replica_stop']-spec['replica_start']),
            'Shard coverage flag differs')
    return records


@contextmanager
def writer_claim(path):
    # Shared-filesystem mkdir, not advisory flock, excludes duplicate writers.
    claim = path.with_suffix('.writer')
    try:
        claim.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f'Writer claim exists: {claim}; inspect its job before retrying, never remove an active claim') from error
    atomic_json(claim/'owner.json', dict(host=socket.gethostname(), pid=os.getpid(),
                job_id=os.environ.get('SLURM_JOB_ID'), array_id=os.environ.get('SLURM_ARRAY_TASK_ID')))
    try:
        yield
    finally:
        (claim/'owner.json').unlink(missing_ok=True)
        claim.rmdir()


def run(plan_path, attempt, index, store, src, out):
    plan, plan_hash = read_plan(plan_path)
    spec = task_spec(plan, attempt, index)
    store, src, out = Path(store).resolve(), Path(src).resolve(), Path(out).resolve()
    environment = numerical_environment(store, src)
    provenance = identity(plan, plan_hash, attempt, dependencies(plan, store, src), environment)
    catalog = SharedBaseShortEvidence()
    reference = exact_reference(catalog, spec['case']['tokens'])
    path = out/attempt/'shards'/f'task-{index:04d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    old_handlers = {}
    def stop(signum, _frame):
        raise InterruptedError('Signal ' + signal.Signals(signum).name)
    with writer_claim(path):
        if path.exists():
            report = json.loads(path.read_text())
            validate_report(report, provenance, spec, plan, attempt)
            require(report['exact_reference'] == reference, 'Saved exact reference changed')
        else:
            report = dict(schema='pooled-predictive-exact-short-shard-v1', identity=provenance, task=spec,
                          exact_reference=reference, records=[], complete=False, status='running',
                          scientific_reference_validated=False)
        if report['complete']:
            print(json.dumps(dict(task=index, status='already_complete')), flush=True)
            return 0
        nodes, weights = leggauss(NUMERICS['alpha_order'])
        kernel = RankedLSARowEvidence(64, 1+nodes/2, depths=NUMERICS['depths'], store=store, src=src,
                                      u_lo=-32., u_hi=32., step=1/32)
        sampler = SharedBaseMCMC(64, kernel, weights/2)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            old_handlers[sig] = signal.signal(sig, stop)
        try:
            settings = plan['attempts'][attempt]
            for replica in range(spec['replica_start']+len(report['records']), spec['replica_stop']):
                print(json.dumps(dict(event='replica_started', case=spec['case']['id'], replica=replica)), flush=True)
                value = estimate(sampler, spec['case']['tokens'],
                    **{k:settings[k] for k in ('particles', 'temperatures', 'moves')},
                    seed=replica_seed(plan, attempt, spec['case']['id'], replica),
                    resample_fraction=NUMERICS['resample_fraction'], seconds_limit=plan['seconds_per_replica'])
                record = dict(case_id=spec['case']['id'], replica=replica, estimate=value, estimate_sha256=digest(value))
                validate_record(record, plan, attempt, spec['case'], replica)
                report['records'].append(record)
                report['complete'] = replica+1 == spec['replica_stop']
                report['status'] = 'completed' if report['complete'] else 'running'
                atomic_json(path, report)
                print(json.dumps(dict(event='replica_saved', case=spec['case']['id'], replica=replica,
                                      elapsed_seconds=value['elapsed_seconds'])), flush=True)
        except (TimeoutError, InterruptedError) as error:
            complete = len(report['records']) == spec['replica_stop']-spec['replica_start']
            report.update(status='completed' if complete else 'interrupted', error=str(error), complete=complete)
            atomic_json(path, report)
            return 0 if complete else 75
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return 0


def summarize(records, reference, expected_repetitions, criteria=CRITERIA):
    """Log-ratio uncertainty retains the covariance of Z and Z*p_b."""
    if not records:
        return dict(repetitions=0, passed=False, checks={'complete_repetitions': False})
    pooled = pool_repetitions(records)
    p = np.array([r['predictive_probs'] for r in records])
    logs = np.array([r['log_evidence'] for r in records])
    z = np.exp(logs-logs.max()); joint = z[:, None]*p
    mean_z, mean_joint = z.mean(), joint.mean(axis=0)
    log_z = float(logs.max()+np.log(mean_z))
    log_p = np.log(mean_joint/mean_z)
    exact_p = np.asarray(reference['predictive_probs'])
    evidence_error = float(-(log_z-reference['log_evidence'])/np.log(2))
    predictive_error = (log_p-np.log(exact_p))/np.log(2)
    joint_error = (log_z+log_p-np.asarray(reference['log_joint_next']))/np.log(2)
    n = len(records)
    if n > 1:
        evidence_se = float(z.std(ddof=1)/(math.sqrt(n)*mean_z*np.log(2)))
        log_ratio_influence = (joint/mean_joint-z[:, None]/mean_z)/np.log(2)
        predictive_se = log_ratio_influence.std(axis=0, ddof=1)/math.sqrt(n)
        joint_se = joint.std(axis=0, ddof=1)/(math.sqrt(n)*mean_joint*np.log(2))
        evidence_allowance = abs(evidence_error)+criteria['mcse_multiplier']*evidence_se
        predictive_allowance = np.abs(predictive_error)+criteria['mcse_multiplier']*predictive_se
    else:
        evidence_se = evidence_allowance = None
        predictive_se = predictive_allowance = joint_se = None
    checks = dict(complete_repetitions=n == expected_repetitions,
                  minimum_repetitions=n >= criteria['minimum_replicates'],
                  effective_replicas=pooled['effective_replica_count'] >= criteria['minimum_effective_replica_count'],
                  maximum_likelihood_share=pooled['maximum_single_replica_likelihood_share'] <= criteria['maximum_likelihood_share'],
                  evidence_precision=evidence_allowance is not None and evidence_allowance <= criteria['tolerance_bits'],
                  every_predictive_log_probability=(predictive_allowance is not None
                      and bool(np.max(predictive_allowance) <= criteria['tolerance_bits'])))
    return dict(repetitions=n, passed=all(checks.values()), checks=checks, pooled=pooled,
        evidence_loss_error_bits=evidence_error, evidence_loss_mcse_bits=evidence_se,
        evidence_error_allowance_bits=evidence_allowance,
        predictive_log_error_bits=predictive_error.tolist(),
        predictive_log_mcse_bits=None if predictive_se is None else predictive_se.tolist(),
        predictive_error_allowance_bits=None if predictive_allowance is None else predictive_allowance.tolist(),
        joint_log_error_bits=joint_error.tolist(), joint_log_mcse_bits=None if joint_se is None else joint_se.tolist(),
        total_variation=float(np.abs(np.exp(log_p)-exact_p).sum()/2),
        exact_to_estimated_kl_bits=float(np.dot(exact_p, np.log(exact_p)-log_p)/np.log(2)),
        uncertainty='Delta-method independent-repetition MCSE; 3-SE gates are numerical diagnostics, not a simultaneous confidence theorem or missed-mode bound.')


def merge(plan_path, attempt, store, src, out):
    plan, plan_hash = read_plan(plan_path)
    store, src, out = Path(store).resolve(), Path(src).resolve(), Path(out).resolve()
    provenance = identity(plan, plan_hash, attempt, dependencies(plan, store, src), numerical_environment(store, src))
    catalog = SharedBaseShortEvidence()
    total = len(plan['cases'])*plan['attempts'][attempt]['replicates']//plan['replicas_per_task']
    grouped = {case['id']: [] for case in plan['cases']}
    missing, unfinished, receipts = [], [], {}
    for index in range(total):
        spec = task_spec(plan, attempt, index)
        path = out/attempt/'shards'/f'task-{index:04d}.json'
        if not path.exists():
            missing.append(index); continue
        payload = path.read_bytes()
        report = json.loads(payload)
        require(report['exact_reference'] == exact_reference(catalog, spec['case']['tokens']), 'Exact reference differs')
        records = validate_report(report, provenance, spec, plan, attempt)
        grouped[spec['case']['id']].extend(records)
        if not report['complete']:
            unfinished.append(index)
        receipts[str(path.relative_to(out/attempt))] = hashlib.sha256(payload).hexdigest()
    summaries = []
    for case in plan['cases']:
        records = sorted(grouped[case['id']], key=lambda r:r['replica'])
        require(len({r['replica'] for r in records}) == len(records), 'Repeated replica identity')
        summaries.append(dict(case=case, **summarize([r['estimate'] for r in records],
                         exact_reference(catalog, case['tokens']), plan['attempts'][attempt]['replicates'])))
    complete = not missing and not unfinished
    result = dict(schema='pooled-predictive-exact-short-summary-v1', identity=provenance,
        complete=complete, missing_tasks=missing, unfinished_tasks=unfinished, cases=summaries,
        exact_short_passed=complete and all(case['passed'] for case in summaries),
        scientific_reference_validated=False, population_reference_validated=False,
        criteria=plan['criteria'], shard_sha256=receipts,
        scope='Only full-prior d64 shared-base SMC evidence and predictions on 16 exact short histories. Mixture reconstruction on actual pooled histories, longer-context refinement and population averages remain unvalidated.')
    atomic_json(out/attempt/'summary.json', result)
    print(json.dumps({k:result[k] for k in ('complete','exact_short_passed','missing_tasks','unfinished_tasks','scientific_reference_validated')}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'merge'))
    p.add_argument('--plan', type=Path, default=PLAN)
    p.add_argument('--attempt', choices=tuple(ATTEMPTS), default='initial')
    p.add_argument('--task-index', type=int)
    p.add_argument('--store', type=Path, required=True); p.add_argument('--src', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'run':
        if a.task_index is None: p.error('run requires --task-index')
        return run(a.plan, a.attempt, a.task_index, a.store, a.src, a.out)
    if a.task_index is not None: p.error('merge cannot select one task')
    merge(a.plan, a.attempt, a.store, a.src, a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
