"""Freeze the remaining Section 3 shared-base posterior inputs.

Reuse the existing 256-sequence cohorts, excluding the two completed pilot
sequences at every depth. This packages saved inputs only: no new source
generation, transformer inference or posterior sampling.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

import prepare_shared_base_posterior_cluster as pilot
from prepare_shared_base_posterior_cluster import (
    ROOT, DEPENDENCY_MANIFEST, require, sha, fingerprint, atomic_json, immutable_json)

PLAN = ROOT / 'code/transformer/configs/shared_base_posterior_full_plan_2026-09-17.json'
BUNDLE_ROOT = 'shared_base_posterior_full_inputs_2026-09-17'
DEFAULT_ARCHIVE = ROOT / 'output/shared_base_posterior_full_2026-09-17/inputs.tar.gz'
DEPTHS, TIMES, INDICES = (0, 1, 4, 8), (256, 1024), tuple(range(2, 256))
WORKER = dict(proposal='iid', steps=64, burn=16384, initial_draws=4096,
              max_draws=524288, seed=2026091780, chains=4, thin=1)
RESOURCES = dict(cpus=4, memory_gib=24, wall_seconds=28800,
                 worker_seconds=26400, audit_seconds=1800, concurrency=64,
                 max_worker_generations=32, max_attempts_per_task=4)
PILOT_RELEASE = 'results/transformer/shared_base_long_context_2026-09-17/summary.json'
PILOT_FILES = {
    'pilot/base.json': ROOT / 'output/shared_base_refinement_base_2026-09-17.json',
    'pilot/refinement.json': ROOT / 'output/shared_base_refinement_completed_2026-09-17.json',
}
SCHEMA = 'shared-base-posterior-full-plan-v1'


def orders_for(depth, time):
    return (32, 64) if (depth, time) == (0, 1024) else (16, 32)


def task_list():
    return [dict(id=f'L{depth}_i{index}_t{time}_a{order}', depth=depth, index=index,
                 time=time, order=order, setting=f'a1_L{depth}')
            for depth in DEPTHS for index in INDICES for time in TIMES
            for order in orders_for(depth, time)]


def task_input_names(task):
    setting, index = task['setting'], task['index']
    return [f'{setting}/protocol.json', f'{setting}/cohort.json',
            f'{setting}/data/seq_{index:06d}.json'] + [
        f'{setting}/models/{model}/seq_{index:06d}.{suffix}'
        for model in ('d64', 'd64_seed1') for suffix in ('json', 'npz')]


def input_names():
    return sorted({name for task in task_list() for name in task_input_names(task)} | set(PILOT_FILES))


def bound_code():
    # The old pilot remains immutable. Its worker and all established numerical
    # helpers are reused, with new launch/preparation code separately pinned.
    names = set(pilot.bound_code())
    names.update('code/scripts/' + name for name in (
        'prepare_shared_base_posterior_full.py', 'shared_base_posterior_full.py',
        'shared_base_posterior_full_worker.py',
        'shared_base_long_context_results.py', 'refine_shared_base_posterior.py',
        'recover_shared_base_posterior_audit.py', 'revalidate_shared_base_depth.py',
        'inspect_shared_base_depth_diagnostics.py', 'depth_rhat_statistics.py'))
    names.update('code/transformer/scitas/' + name for name in (
        'submit_shared_base_posterior_full.sh', 'shared_base_posterior_full.sbatch'))
    names.update(('code/transformer/configs/shared_base_posterior_plan_2026-09-16.json',
                  'code/transformer/configs/shared_base_posterior_refinement_plan_2026-09-17.json'))
    return sorted(names)


def verify_files(root, plan, names):
    root = Path(root)
    for name in names:
        expected = plan['inputs']['files'][name]
        path = root / name
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == expected['size_bytes']
                and sha(path) == expected['sha256'], f'Frozen full-cohort input mismatch: {name}')
    return len(names)


def verify_inputs(root, plan):
    return verify_files(root, plan, input_names())


def verify_task_inputs(root, plan, task):
    require(task in plan['tasks'], 'Task not declared in the frozen full-cohort plan')
    return verify_files(root, plan, task_input_names(task))


def validate_sources(source_root):
    """Reuse independent source/target/probability readback; never regenerate."""
    sys.path.insert(0, str(ROOT / 'code/transformer'))
    from shared_base_conditional_cohort import inputs, load_source
    source_root = Path(source_root)
    sources = []
    for depth in DEPTHS:
        folder = source_root / f'a1_L{depth}'
        cohort = json.loads((folder / 'cohort.json').read_bytes())
        protocol = json.loads((folder / 'protocol.json').read_bytes())
        require(cohort['n_sequences'] == 256 and cohort['depth'] == depth
                and cohort['alpha'] == 1. and cohort['n_tokens_per_sequence'] == 1025
                and protocol['conditions_hidden_from_predictors'] is True
                and protocol['reference_prior'] == dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
                'Existing source population/prior changed')
        for index in INDICES:
            record = load_source(folder, index)
            for time in TIMES:
                tokens, _, models, identity = inputs(folder, index, time)
                require(len(tokens) == time and set(models) == {'d64', 'd64_seed1'}
                        and identity['source_id'] == record['source_id'], 'Frozen prefix/model alignment changed')
            sources.append(dict(setting=folder.name, index=index, source_id=record['source_id'],
                                tokens_sha256=record['tokens_sha256']))
        print(json.dumps(dict(stage='verified_existing_inputs', depth=depth, sequences=len(INDICES))), flush=True)
    require(len(sources) == 1016 and len({r['source_id'] for r in sources}) == 1016,
            'Source identities must be independent and complete')
    return sources


def pack_files(source_root, archive, files, source_overrides=None):
    """Deterministic regular-file archive in a distinct input directory."""
    source_root, archive = Path(source_root), Path(archive)
    source_overrides = source_overrides or {}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix='.full-posterior-', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with temporary.open('wb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as tar:
                for name in sorted(files):
                    path = source_overrides.get(name, source_root / name)
                    require(path.is_file() and not path.is_symlink() and sha(path) == files[name]['sha256'],
                            f'Input changed during archive construction: {name}')
                    info = tar.gettarinfo(str(path), arcname=f'{BUNDLE_ROOT}/{name}')
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    with path.open('rb') as source:
                        tar.addfile(info, source)
        if archive.exists():
            require(sha(archive) == sha(temporary), 'Refusing to replace different frozen inputs')
            temporary.unlink()
        else:
            temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()


def export(source_root, archive, plan_path):
    sources = validate_sources(source_root)
    source_root, archive = Path(source_root), Path(archive)
    files = {}
    for name in input_names():
        path = PILOT_FILES.get(name, source_root / name)
        files[name] = dict(sha256=sha(path), size_bytes=path.stat().st_size)
    release = json.loads((ROOT / PILOT_RELEASE).read_bytes())
    require(release['schema'] == 'shared-base-long-context-results-v1'
            and release['all_numerical_checks_passed'] is True
            and release['n_sequences'] == 8 and release['indices'] == [0, 1],
            'Require the completed pilot release before full-cohort work')
    refined = json.loads(PILOT_FILES['pilot/refinement.json'].read_bytes())
    require(refined['base']['summary_sha256'] == files['pilot/base.json']['sha256']
            and refined['complete'] is True and refined['all_numerical_checks_passed'] is True
            and all(release['input_sha256'][str(path.relative_to(ROOT))] == files[name]['sha256']
                    for name, path in PILOT_FILES.items()), 'Completed pilot raw-summary binding changed')
    pack_files(source_root, archive, files, PILOT_FILES)
    plan = dict(schema=SCHEMA,
        scope='All remaining 254 existing test sequences per depth. No new source draws; completed pilot indices 0 and 1 remain untouched.',
        conditions=dict(alpha=1., depths=list(DEPTHS), indices=list(INDICES), observed_tokens=list(TIMES),
                        exponent_orders_by_condition={f'L{d}_t{t}': list(orders_for(d, t)) for d in DEPTHS for t in TIMES}),
        reference_prior=dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
        source_sequences=sources, tasks=task_list(), worker=WORKER, resources=RESOURCES,
        completed_pilot=dict(path=PILOT_RELEASE, sha256=sha(ROOT / PILOT_RELEASE), indices=[0, 1],
            base_summary='pilot/base.json', refinement_summary='pilot/refinement.json',
            role='Reuse its separately validated references in the eventual 256-sequence comparison; do not recompute or overwrite.'),
        code_sha256={name: sha(ROOT / name) for name in bound_code()},
        dependency_manifest=DEPENDENCY_MANIFEST, dependency_manifest_sha256=sha(ROOT / DEPENDENCY_MANIFEST),
        inputs=dict(root_directory=BUNDLE_ROOT, archive_filename=archive.name,
                    archive_sha256=sha(archive), archive_size_bytes=archive.stat().st_size, files=files),
        numerical_gates='Existing independent raw audit, rounding-only canonical model readback and scale-invariant depth Rhat correction. Every declared source retained; incomplete means blocked.',
        source_cohort_note='Original 256-sequence declarations preserved. Package only existing indices 2..255. Shared initial samples are not new independent sequences.',
        seed_policy='Fresh master seed 2026091780; existing per-case/time/order/chain seed derivation.',
        automatic_successors=True, production_approved=True)
    immutable_json(plan_path, plan)
    return plan


def read_plan(path, verify_code=True):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    require(plan.get('schema') == SCHEMA and plan.get('tasks') == task_list()
            and plan.get('worker') == WORKER and plan.get('resources') == RESOURCES,
            'Full-cohort cases, integration orders, sampling or resources changed')
    require(plan['conditions'] == dict(alpha=1., depths=list(DEPTHS), indices=list(INDICES), observed_tokens=list(TIMES),
        exponent_orders_by_condition={f'L{d}_t{t}': list(orders_for(d, t)) for d in DEPTHS for t in TIMES}),
        'Full-cohort conditions changed')
    require(plan['inputs']['root_directory'] == BUNDLE_ROOT and set(plan['inputs']['files']) == set(input_names()),
            'Full-cohort input inventory changed')
    require(plan['completed_pilot']['path'] == PILOT_RELEASE and plan['completed_pilot']['indices'] == [0, 1]
            and plan['completed_pilot']['sha256'] == sha(ROOT / PILOT_RELEASE), 'Completed pilot release changed')
    require(plan['automatic_successors'] is True and plan['production_approved'] is True,
            'Full-cohort authorization changed')
    require(plan['dependency_manifest'] == DEPENDENCY_MANIFEST
            and plan['dependency_manifest_sha256'] == sha(ROOT / DEPENDENCY_MANIFEST), 'Numerical dependencies changed')
    require(len(plan['source_sequences']) == 1016
            and {(r['setting'], r['index']) for r in plan['source_sequences']} ==
            {(f'a1_L{d}', i) for d in DEPTHS for i in INDICES}
            and len({r['source_id'] for r in plan['source_sequences']}) == 1016,
            'Full-cohort source identity list is incomplete')
    if verify_code:
        require(set(plan['code_sha256']) == set(bound_code()), 'Incomplete full-cohort code inventory')
        for name, expected in plan['code_sha256'].items():
            require(sha(ROOT / name) == expected, f'Bound implementation changed: {name}')
    return plan, hashlib.sha256(raw).hexdigest()


def unpack(archive, destination, plan):
    archive, destination = Path(archive), Path(destination)
    require(sha(archive) == plan['inputs']['archive_sha256'], 'Archive checksum mismatch')
    target = destination / BUNDLE_ROOT
    require(not target.exists(), f'Input directory already exists; use verify: {target}')
    destination.mkdir(parents=True, exist_ok=True)
    expected = {f'{BUNDLE_ROOT}/{name}': meta for name, meta in plan['inputs']['files'].items()}
    with tempfile.TemporaryDirectory(dir=destination, prefix='.full-posterior-extract-') as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, 'r:gz') as tar:
            entries = tar.getmembers()
            require(len(entries) == len(expected) and {e.name for e in entries} == set(expected),
                    'Unexpected, missing or repeated input archive members')
            require(all(e.isfile() and not Path(e.name).is_absolute() and '..' not in Path(e.name).parts
                        and e.size == expected[e.name]['size_bytes'] for e in entries), 'Unsafe input archive member')
            for entry in entries:
                path = staging / entry.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as source, path.open('xb') as out:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        out.write(block)
        verify_files(staging / BUNDLE_ROOT, plan, list(plan['inputs']['files']))
        (staging / BUNDLE_ROOT).rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    commands = parser.add_subparsers(dest='command', required=True)
    exp = commands.add_parser('export')
    exp.add_argument('--source-root', type=Path, default=ROOT / 'output/shared_base_conditional_curves_2026-09-15')
    exp.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    unp = commands.add_parser('unpack')
    unp.add_argument('--archive', type=Path, required=True)
    unp.add_argument('--destination', type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        plan = export(args.source_root, args.archive, args.plan)
        print(json.dumps(dict(plan=str(args.plan), archive=str(args.archive),
            bytes=plan['inputs']['archive_size_bytes'], files=len(plan['inputs']['files']),
            sha256=plan['inputs']['archive_sha256'], tasks=len(plan['tasks']), sequences=len(plan['source_sequences']))))
    else:
        plan, _ = read_plan(args.plan)
        root = unpack(args.archive, args.destination, plan) if args.command == 'unpack' else args.root
        count = verify_inputs(root, plan)
        print(json.dumps(dict(root=str(root), verified_files=count, verified_sources=len(plan['source_sequences']),
                              sampling_started=False)))


if __name__ == '__main__':
    main()
