"""Read saved depth weights; compare legacy and scale-invariant split Rhat.

This diagnostic cannot change an audit decision, sample, or predictor. It
reads both integration orders for each of the two affected frozen cases.
All array processing runs in a user-submitted Jed allocation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

import recover_shared_base_posterior_audit as recovery
from prepare_shared_base_posterior_cluster import ROOT, require, sha, fingerprint, immutable_json

TASKS = (18, 19, 24, 25)
DEFAULT_OUT = ROOT/'output/shared_base_depth_diagnostics_2026-09-17'
OWN_FILES = (
    'code/scripts/inspect_shared_base_depth_diagnostics.py',
    'code/scripts/depth_rhat_statistics.py',
    'code/transformer/scitas/submit_shared_base_depth_diagnostics.sh',
    'code/transformer/scitas/shared_base_depth_diagnostics.sbatch',
)


def locate_launch(directory):
    candidates = []
    for path in (Path(directory)/'launches').glob('*.json'):
        value, _ = recovery.checked_json(path)
        if value.get('mode') == 'audit' and value.get('selected_tasks') == list(range(32)):
            candidates.append(path.resolve())
    require(len(candidates) == 1, 'Need the completed full-pilot recovery launch')
    return candidates[0]


def build_launch(args):
    source = locate_launch(args.recovery)
    recovered, old, _ = recovery.read_recovery(source)
    out = args.out.expanduser().resolve()
    for protected in (Path(recovered['context']['output']),
                      Path(old['campaign']['output']), Path(old['campaign']['inputs'])):
        require(out != protected and out not in protected.parents and protected not in out.parents,
                'Diagnostic output must be separate from original and recovery records')
    spec = dict(schema='shared-base-depth-diagnostic-launch-v1',
        recovery_launch=str(source), recovery_launch_sha256=sha(source),
        output=str(out), task_indices=list(TASKS), depths=list(range(9)),
        code_sha256={name:sha(ROOT/name) for name in OWN_FILES},
        no_sampling=True, changes_acceptance=False, venv=recovered['venv'])
    spec['launch_id'] = fingerprint(spec)
    manifest = out/'launch.json'
    command = ['sbatch', '--parsable', f'--chdir={ROOT}', '--job-name=sb-depth-diagnostic',
               f'--output={out}/logs/diagnostic-%j.out', str(ROOT/OWN_FILES[-1]),
               str(manifest), spec['venv']]
    return spec, manifest, command


def submit(args):
    spec, manifest, command = build_launch(args)
    print(shlex.join(command), flush=True)
    if not args.submit:
        print('Dry run only; no files written or jobs submitted.', flush=True)
        return
    require((Path(spec['venv'])/'bin/python').is_file(), 'Prepared Python environment is missing')
    immutable_json(manifest, spec)
    (Path(spec['output'])/'logs').mkdir(exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize_depths(checkpoint, estimate):
    """Validate and inspect depth arrays only; no posterior evaluation."""
    import numpy as np
    from depth_rhat_statistics import coordinate_summary
    from shared_base_chain_runner import _hash_array
    arrays = []
    diagnostics = estimate['diagnostics']
    require(diagnostics['chains'] == 4, 'Expected four saved chains')
    with np.load(checkpoint, allow_pickle=False) as stored:
        encoded = str(stored['metadata'].item())
        require(hashlib.sha256(encoded.encode()).hexdigest() == str(stored['metadata_sha256'].item()),
                'Checkpoint metadata checksum mismatch')
        meta = json.loads(encoded)
        require(meta['format'] == 'shared_base_chain_run_v1' and meta['config']['chains'] == 4,
                'Unexpected saved checkpoint format')
        for chain in range(4):
            name = f'depths_{chain}'
            values = stored[name]
            require(_hash_array(values) == meta['array_hashes'][name], 'Saved depth array checksum mismatch')
            require(values.dtype == np.dtype('float64') and values.shape == (diagnostics['draws'], 9),
                    'Saved depth dimensions differ from audited report')
            require(np.isfinite(values).all() and np.all(values >= 0) and np.all(values <= 1)
                    and np.max(np.abs(values.sum(axis=1)-1)) <= 1e-7, 'Invalid saved depth distributions')
            arrays.append(values)
    samples = np.stack(arrays)
    del arrays
    records = []
    for depth in range(9):
        result = coordinate_summary(samples[:, :, depth])
        require(math.isclose(result['mean'], estimate['posterior_depth'][depth], rel_tol=2e-12, abs_tol=0),
                'Recomputed depth mean differs from audited report')
        actual, saved = result['original']['rhat'], diagnostics['rhat_depth'][depth]
        require((actual is None and saved is None) or
                (actual is not None and saved is not None and math.isclose(actual, saved, rel_tol=2e-12, abs_tol=1e-14)),
                'Legacy Rhat does not reproduce the audited diagnostic')
        records.append(dict(depth=depth, recorded_mcse=estimate['mcse_depth'][depth], **result))
    return records


def run(path):
    spec, manifest_hash = recovery.checked_json(path)
    require(spec.get('schema') == 'shared-base-depth-diagnostic-launch-v1'
            and spec['launch_id'] == fingerprint({k:v for k,v in spec.items() if k != 'launch_id'}),
            'Diagnostic launch checksum/schema mismatch')
    require(spec['task_indices'] == list(TASKS) and spec['depths'] == list(range(9))
            and spec['no_sampling'] is True and spec['changes_acceptance'] is False,
            'Diagnostic scope changed')
    require(spec['code_sha256'] == {name:sha(ROOT/name) for name in OWN_FILES}, 'Diagnostic code changed')
    require(sha(spec['recovery_launch']) == spec['recovery_launch_sha256'], 'Recovery launch changed')
    launch, old, plan = recovery.read_recovery(spec['recovery_launch'])
    runtime = recovery.original.runtime_setup(old['campaign']['dependency_root'])
    results = []
    out = Path(spec['output'])
    # Use a separate diagnostic lock. Never open any original artifact for writing.
    with (out/'diagnostic.lock').open('a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out/f"diagnostics-{spec['launch_id'][:16]}.json"
        if result_path.exists():
            saved, _ = recovery.checked_json(result_path)
            require(saved['schema'] == 'shared-base-depth-diagnostic-v1'
                    and saved['launch_sha256'] == manifest_hash and saved['runtime'] == runtime
                    and saved['complete'] is True and saved['no_sampling'] is True
                    and saved['changes_acceptance'] is False,
                    'Existing diagnostic report identity differs')
            require([r['task'] for r in saved['tasks']] == [plan['tasks'][i] for i in TASKS],
                    'Existing diagnostic report has different tasks')
            for row in saved['tasks']:
                task = row['task']
                receipt_path = Path(launch['context']['output'])/'tasks'/f"{task['id']}.json"
                require(sha(receipt_path) == row['recovery_receipt_sha256'], 'Recovery receipt changed')
                require(row['original_file_sha256'] == recovery.original_snapshot(old, plan, task)['files'],
                        'Original artifacts changed after diagnostics')
            print(json.dumps(dict(status='diagnostics_complete', tasks=len(TASKS), reused=True,
                                  changes_acceptance=False)), flush=True)
            return
        for i in TASKS:
            task = plan['tasks'][i]
            receipt_path = Path(launch['context']['output'])/'tasks'/f"{task['id']}.json"
            receipt, receipt_hash = recovery.checked_json(receipt_path)
            paths = recovery.original.task_paths(old['campaign'], task)
            # The original worker/auditor already created this advisory lock.
            with paths['reference'].with_suffix('.lock').open('rb') as lock:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                recovery.validate_recovered(receipt, launch['context'], old, plan, task)
                require(runtime == receipt['runtime'], 'Use the original numerical runtime for diagnostics')
                started = time.monotonic()
                reference, _ = recovery.checked_json(paths['reference'])
                print(json.dumps(dict(task=task['id'], status='reading_saved_depth_weights', no_sampling=True)), flush=True)
                rows = summarize_depths(paths['checkpoint'], reference['estimate'])
                require(sha(receipt_path) == receipt_hash, 'Recovery receipt changed during inspection')
                require(receipt['original'] == recovery.original_snapshot(old, plan, task),
                        'Original artifacts changed during inspection')
            result = dict(task=task, depth_diagnostics=rows, recovery_receipt_sha256=receipt_hash,
                original_file_sha256=receipt['original']['files'], elapsed_seconds=time.monotonic()-started)
            results.append(result)
            for row in rows:
                if row['depth'] == 2 or row['old_cutoff_trigger']:
                    print(json.dumps(dict(task=task['id'], depth=row['depth'], mean=row['mean'],
                        maximum_saved_weight=row['max'], original=row['original'],
                        rescaled=row['rescaled'])), flush=True)
        result = dict(schema='shared-base-depth-diagnostic-v1', launch_sha256=manifest_hash,
            runtime=runtime, complete=True, tasks=results, no_sampling=True, changes_acceptance=False,
            interpretation='Read-only arithmetic diagnostic on saved samples. No audit flag is overridden; this does not certify posterior convergence.')
        immutable_json(result_path, result)
        print(json.dumps(dict(status='diagnostics_complete', tasks=len(results), changes_acceptance=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    launch = modes.add_parser('launch')
    launch.add_argument('--recovery', type=Path, default=recovery.DEFAULT_OUT)
    launch.add_argument('--out', type=Path, default=DEFAULT_OUT)
    launch.add_argument('--submit', action='store_true')
    worker = modes.add_parser('run')
    worker.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'launch':
        submit(args)
    else:
        run(args.launch)


if __name__ == '__main__':
    main()
