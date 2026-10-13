"""Package existing posterior-pilot inputs without generating sources or scores."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'code/transformer/configs/shared_base_posterior_plan_2026-09-16.json'
BUNDLE_ROOT = 'shared_base_posterior_inputs_2026-09-16'
DEPENDENCY_MANIFEST = 'code/transformer/configs/shared_base_dependency_manifest_2026-09-16.json'
DEPTHS, TIMES, ORDERS, INDICES = (0, 1, 4, 8), (256, 1024), (16, 32), (0, 1)
WORKER = dict(proposal='iid', steps=64, burn=16384, initial_draws=4096,
              max_draws=262144, seed=2026091670, chains=4, thin=1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_bytes()) == value, f'Refusing to change frozen file: {path}')
    else:
        atomic_json(path, value)


def task_list():
    return [dict(id=f'L{depth}_i{index}_t{time}_a{order}', depth=depth, index=index,
                 time=time, order=order, setting=f'a1_L{depth}')
            for depth in DEPTHS for index in INDICES for time in TIMES for order in ORDERS]


def input_names():
    names = []
    for depth in DEPTHS:
        setting = f'a1_L{depth}'
        names.extend(f'{setting}/{name}' for name in ('cohort.json', 'protocol.json'))
        for index in INDICES:
            names.append(f'{setting}/data/seq_{index:06d}.json')
            names.extend(f'{setting}/models/{model}/seq_{index:06d}.{suffix}'
                         for model in ('d64', 'd64_seed1') for suffix in ('json', 'npz'))
    return sorted(names)


def bound_code():
    from run_shared_base_curve_campaign import CODE
    names = set(CODE)
    names.update('code/scripts/'+name for name in (
        'shared_base_curve_audit.py', 'shared_base_full_context_summary.py',
        'shared_base_surrogate_summary.py', 'prepare_shared_base_cluster.py', 'calibrate_shared_base_evidence.py',
        'prepare_shared_base_posterior_cluster.py', 'shared_base_posterior_cluster.py'))
    names.update('code/transformer/'+name for name in (
        'eval_shared_base_surrogate.py', 'extend_shared_base_models.py', 'model.py',
        'shared_base_short_evidence.py', 'shared_base_evidence_smc.py'))
    names.update('code/transformer/scitas/'+name for name in (
        'submit_shared_base_posterior.sh', 'shared_base_posterior.sbatch', 'environment.sh'))
    names.add(DEPENDENCY_MANIFEST)
    names.update(f'results/transformer/runs/{model}/{name}'
                 for model in ('d64', 'd64_seed1') for name in ('config.json', 'model_100000.pt'))
    return sorted(names)


def validate_sources(source_root):
    """Use the existing independent loader on every selected source/time."""
    sys.path.insert(0, str(ROOT/'code/transformer'))
    from shared_base_conditional_cohort import inputs
    from shared_base_surrogate_summary import load_models
    from shared_base_full_context_summary import Audit
    import numpy as np
    source_root = Path(source_root)
    sources = []
    for depth in DEPTHS:
        folder = source_root/f'a1_L{depth}'
        cohort = json.loads((folder/'cohort.json').read_bytes())
        protocol = json.loads((folder/'protocol.json').read_bytes())
        require(cohort['n_sequences'] == 256 and cohort['depth'] == depth
                and cohort['alpha'] == 1. and cohort['n_tokens_per_sequence'] == 1025
                and protocol['conditions_hidden_from_predictors'] is True
                and protocol['reference_prior'] == dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
                'Unexpected original conditional source declaration')
        for index in INDICES:
            record = json.loads((folder/'data'/f'seq_{index:06d}.json').read_bytes())
            for time in TIMES:
                tokens, _, _, _ = inputs(folder, index, time)
                load_models(Audit(), folder, dict(record=record, tokens=np.asarray(record['tokens'])), index, time)
                require(len(tokens) == time, 'Input prefix length differs')
            sources.append(dict(setting=folder.name, index=index, source_id=record['source_id'],
                                tokens_sha256=record['tokens_sha256']))
    return sources


def pack_files(source_root, archive, files):
    """Deterministic regular-file archive; never overwrite a changed archive."""
    source_root, archive = Path(source_root), Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix='.posterior-', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with temporary.open('wb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as tar:
                for name in sorted(files):
                    path = source_root/name
                    require(path.is_file() and not path.is_symlink() and sha(path) == files[name]['sha256'],
                            f'Input changed during packing: {name}')
                    info = tar.gettarinfo(str(path), arcname=f'{BUNDLE_ROOT}/{name}')
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    with path.open('rb') as source:
                        tar.addfile(info, source)
        if archive.exists():
            require(sha(archive) == sha(temporary), 'Refusing to replace a different input archive')
            temporary.unlink()
        else:
            temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()


def export(source_root, archive, plan_path):
    source_root, archive = Path(source_root), Path(archive)
    sources = validate_sources(source_root)
    files = {name: dict(sha256=sha(source_root/name), size_bytes=(source_root/name).stat().st_size)
             for name in input_names()}
    pack_files(source_root, archive, files)
    plan = dict(schema='shared-base-posterior-cluster-plan-v1',
        scope='Fixed computational pilot: two source sequences per condition. No population learning curve or automatic expansion.',
        conditions=dict(alpha=1., depths=list(DEPTHS), indices=list(INDICES),
                        observed_tokens=list(TIMES), exponent_orders=list(ORDERS)),
        reference_prior=dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
        source_sequences=sources, tasks=task_list(), worker=WORKER,
        resources=dict(cpus=2, memory_gib=12, wall_seconds=28800, worker_seconds=27900, concurrency=32),
        code_sha256={name: sha(ROOT/name) for name in bound_code()},
        dependency_manifest=DEPENDENCY_MANIFEST, dependency_manifest_sha256=sha(ROOT/DEPENDENCY_MANIFEST),
        inputs=dict(root_directory=BUNDLE_ROOT, archive_filename=archive.name,
                    archive_sha256=sha(archive), archive_size_bytes=archive.stat().st_size, files=files),
        numerical_gates='Unchanged shared_base_curve_reference and independent shared_base_curve_audit checks. No failed case is discarded.',
        seed_policy='Fresh sampling master seed; same frozen source sequences and saved transformer probabilities. Every chain/order/case uses existing seed construction.',
        source_cohort_note='Original 256-sequence declarations are preserved byte for byte; this pilot selects indices 0 and 1 before any new run. No other indices are bundled.',
        automatic_successors=False, production_approved=False)
    immutable_json(plan_path, plan)
    return plan


def read_plan(path, verify_code=True):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    require(plan.get('schema') == 'shared-base-posterior-cluster-plan-v1', 'Wrong posterior plan schema')
    require(plan.get('tasks') == task_list() and plan.get('worker') == WORKER, 'Fixed pilot cases or sampling budget changed')
    require(plan.get('resources') == dict(cpus=2, memory_gib=12, wall_seconds=28800, worker_seconds=27900, concurrency=32),
            'Fixed resource bounds changed')
    require(plan['inputs']['root_directory'] == BUNDLE_ROOT
            and set(plan['inputs']['files']) == set(input_names()), 'Incomplete or changed frozen input inventory')
    require(plan.get('automatic_successors') is False and plan.get('production_approved') is False,
            'Pilot must not authorize production')
    require(plan['dependency_manifest'] == DEPENDENCY_MANIFEST
            and plan['dependency_manifest_sha256'] == sha(ROOT/DEPENDENCY_MANIFEST), 'Numerical dependency changed')
    if verify_code:
        require(set(plan['code_sha256']) == set(bound_code()), 'Incomplete bound source/checkpoint inventory')
        for name, expected in plan['code_sha256'].items():
            require(sha(ROOT/name) == expected, f'Bound code/checkpoint changed: {name}')
    return plan, hashlib.sha256(raw).hexdigest()


def verify_inputs(root, plan):
    root = Path(root)
    for name, expected in plan['inputs']['files'].items():
        path = root/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['size_bytes']
                and sha(path) == expected['sha256'], f'Frozen input mismatch: {name}')
    return len(plan['inputs']['files'])


def unpack(archive, destination, plan):
    archive, destination = Path(archive), Path(destination)
    require(sha(archive) == plan['inputs']['archive_sha256'], 'Input archive checksum mismatch')
    target = destination/BUNDLE_ROOT
    require(not target.exists(), f'Input directory exists; verify instead: {target}')
    destination.mkdir(parents=True, exist_ok=True)
    expected = {f'{BUNDLE_ROOT}/{name}': meta for name, meta in plan['inputs']['files'].items()}
    with tempfile.TemporaryDirectory(dir=destination, prefix='.posterior-extract-') as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, 'r:gz') as tar:
            entries = tar.getmembers()
            require(len(entries) == len(expected) and {e.name for e in entries} == set(expected),
                    'Unexpected, missing or repeated input archive members')
            require(all(e.isfile() and not Path(e.name).is_absolute() and '..' not in Path(e.name).parts
                        and e.size == expected[e.name]['size_bytes'] for e in entries), 'Unsafe input archive member')
            for entry in entries:
                path = staging/entry.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as source, path.open('xb') as out:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        out.write(block)
        verify_inputs(staging/BUNDLE_ROOT, plan)
        (staging/BUNDLE_ROOT).rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    commands = parser.add_subparsers(dest='command', required=True)
    exp = commands.add_parser('export')
    exp.add_argument('--source-root', type=Path, default=ROOT/'output/shared_base_conditional_curves_2026-09-15')
    exp.add_argument('--archive', type=Path, default=ROOT/'output/shared_base_posterior_cluster_2026-09-16/inputs.tar.gz')
    unp = commands.add_parser('unpack')
    unp.add_argument('--archive', type=Path, required=True)
    unp.add_argument('--destination', type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        plan = export(args.source_root, args.archive, args.plan)
        print(json.dumps(dict(plan=str(args.plan), archive=str(args.archive), bytes=plan['inputs']['archive_size_bytes'],
                              sha256=plan['inputs']['archive_sha256'], tasks=len(plan['tasks']))))
    else:
        plan, _ = read_plan(args.plan)
        root = unpack(args.archive, args.destination, plan) if args.command == 'unpack' else args.root
        count = verify_inputs(root, plan)
        sources = validate_sources(root)
        print(json.dumps(dict(root=str(root), verified_files=count, verified_sources=len(sources), sampling_started=False)))


if __name__ == '__main__':
    main()
