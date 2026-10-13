"""Freeze the two withheld unseen-context precision continuations; never sample.

The compact checked-in plan binds a deterministic ignored manifest containing
all 123 sources and 246 original references. Raw hashes are inherited here and
must be rechecked immediately before each isolated copy. Runtime and absolute
model paths remain part of the original numerical identity: moving to another
platform or rewriting that identity is deliberately unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'code/transformer/configs/unseen_context_refinement_plan_2026-09-19.json'
READINESS = ROOT / 'output/shared_base_unseen_refinement_readiness_2026-09-19.json'
MANIFEST = ROOT / 'output/unseen_context_refinement_2026-09-19/frozen_manifest.json'
ORIGINAL_PLAN = ROOT / 'output/shared_base_curve_campaign_2026-09-15/plan.json'
RUNTIME_PYTHON = ROOT / 'output/unseen_context_refinement_runtime_2026-09-19/bin/python'
MULTIPLIER = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def immutable_json(path, value):
    path = Path(path)
    data = encoded(value)
    if path.exists():
        require(path.read_bytes() == data, f'Frozen file differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def relative(path, root=ROOT):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def safe_path(root, name):
    root = Path(root).resolve()
    path = root / name
    require(not Path(name).is_absolute() and '..' not in Path(name).parts,
            'Manifest paths must be within the repository')
    require(path.resolve().is_relative_to(root), 'Manifest path escapes repository')
    return path


def slot_bytes(draws, order):
    """Same conservative measured-summary allowance as the previous local run."""
    require(order in (16, 32), 'Unexpected integration order')
    return 3 * 1024**3 // 2 + 6 * 4 * draws * 64 * 8


def file_binding(path, expected=None, *, raw=False):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Missing or linked input: {path}')
    digest = expected if raw else sha(path)
    require(expected is None or digest == expected, f'Input checksum changed: {path}')
    require(isinstance(digest, str) and len(digest) == 64, 'Missing frozen digest')
    return dict(path=relative(path), sha256=digest, size_bytes=path.stat().st_size)


def validate_tasks(plan):
    groups = plan['groups']
    require([(g['depth'], g['time'], len(g['selected_indices'])) for g in groups]
            == [(4, 64, 66), (8, 64, 57)], 'Frozen group counts changed')
    require(all(g['selected_indices'] == sorted(set(g['selected_indices'])) for g in groups),
            'Duplicate or unsorted selected histories')
    expected = {(g['depth'], i, 64, order) for g in groups
                for i in g['selected_indices'] for order in (16, 32)}
    tasks = plan['tasks']
    actual = [(s['task']['depth'], s['task']['index'], s['task']['time'], s['task']['order'])
              for s in tasks]
    require(len(actual) == 246 and set(actual) == expected and len(set(actual)) == 246,
            'Need exactly all 123 frozen histories and both integration orders')
    require(actual == sorted(actual), 'Task order changed')
    for spec in tasks:
        task = spec['task']
        require(task['setting'] == f"a1_L{task['depth']}" and task['id'] ==
                f"L{task['depth']}_i{task['index']}_t64_a{task['order']}", 'Task label mismatch')
        require(spec['baseline_draws'] >= 4096 and spec['baseline_draws'] <= 262144
                and spec['target_draws'] == MULTIPLIER * spec['baseline_draws'],
                'Every saved draw count must receive the same fixed fourfold target')
        identity = spec['parent_identity']
        require(spec['parent_reference_id'] == fingerprint(identity), 'Reference identity mismatch')
        for name, field in [('source', 'inputs'), ('target', 'target'), ('proposal', 'proposal')]:
            require(spec[name + '_identity_sha256'] == fingerprint(identity[field]),
                    f'Frozen {name} identity mismatch')
        require(spec['runtime_identity_sha256'] == fingerprint(identity['target']['runtime']),
                'Frozen runtime identity mismatch')
        require(identity['proposal'] == plan['proposal'] and identity['target']['runtime'] == plan['runtime'],
                'Heterogeneous original runtime/proposal is unsupported')
        require(identity['master_sampling_seed'] == plan['worker']['seed']
                and identity['burn'] == plan['worker']['burn'], 'Original seed or burn changed')
        require(spec['estimated_peak_bytes'] == slot_bytes(spec['target_draws'], task['order']),
                'Memory admission estimate changed')


def verify_task_inputs(plan, spec, root=ROOT, verify_raw=False):
    """Validate immutable parent inputs; raw verification streams, never loads arrays."""
    paths = {}
    bindings = dict(spec['source_files'], **spec['parent_files'])
    for key, item in bindings.items():
        path = safe_path(root, item['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['size_bytes'],
                f'Original input missing or size changed: {path}')
        if verify_raw or key not in ('checkpoint', 'moments'):
            require(sha(path) == item['sha256'], f'Original checksum changed: {path}')
        paths[key] = path
    parent = read(paths['reference'])
    require(parent['identity'] == spec['parent_identity'] and
            parent['reference_id'] == spec['parent_reference_id'], 'Original numerical identity changed')
    require(parent['checkpoint_sha256'] == spec['parent_files']['checkpoint']['sha256'] and
            parent['moments_sha256'] == spec['parent_files']['moments']['sha256'], 'Parent raw bindings differ')
    require(parent['record']['precision']['draws_per_chain'] == spec['baseline_draws'],
            'Parent draw budget changed')
    source = read(paths['source'])
    task = spec['task']
    require(source['source_id'] == spec['source_id'] and source['tokens'][63] not in source['tokens'][:63]
            and source['identity']['source_depth'] == task['depth']
            and source['identity']['source_index'] == task['index'], 'Frozen unseen-context selection changed')
    # The old worker fingerprints absolute model paths. Preserve them exactly.
    for model, binding in parent['identity']['inputs']['model_files'].items():
        require(paths['model_' + model].resolve() == Path(binding['path']).resolve(),
                'Original model paths must be preserved for an exact continuation')
    return paths


def build_manifest(readiness_path=READINESS):
    ready = read(readiness_path)
    require(ready['schema'] == 'shared-base-unseen-refinement-readiness-v1' and
            ready['ready_to_prepare_isolated_continuation'] and not ready['missing_inputs']
            and not ready['readiness_check_failures'], 'Readiness audit has unresolved inputs')
    for path, binding in ready['provenance'].items():
        require(sha(path) == binding['sha256'], f'Readiness provenance changed: {path}')
    sources = {(s['depth'], s['index']): s for s in ready['sources']}
    tasks = []
    all_uncompressed = 0
    maximum_uncompressed = 0
    numerical_hashes = {}
    for record in sorted(ready['references'], key=lambda r: (r['depth'], r['index'], r['time'], r['order'])):
        parent = read(record['reference_path'])
        identity = parent['identity']
        source = sources[record['depth'], record['index']]
        folder = Path(record['source_folder'])
        source_files = dict(source=file_binding(source['source_path'], source['source_sha256']),
                            protocol=file_binding(folder / 'protocol.json', identity['inputs']['protocol_sha256']),
                            cohort=file_binding(folder / 'cohort.json'))
        for model, item in identity['inputs']['model_files'].items():
            source_files['model_' + model] = file_binding(item['path'], item['sha256'])
        parent_files = dict(reference=file_binding(record['reference_path'], record['reference_sha256']),
                            case_receipt=file_binding(record['case_receipt_path'], record['case_receipt_sha256']))
        for name in ('checkpoint', 'moments'):
            parent_files[name] = file_binding(record[name]['path'], record[name]['recorded_sha256'], raw=True)
        with zipfile.ZipFile(record['checkpoint']['path']) as archive:
            uncompressed = sum(info.file_size for info in archive.infolist())
        all_uncompressed += uncompressed
        maximum_uncompressed = max(maximum_uncompressed, uncompressed)
        task = dict(id=f"L{record['depth']}_i{record['index']}_t64_a{record['order']}",
                    depth=record['depth'], index=record['index'], time=64, order=record['order'],
                    setting=f"a1_L{record['depth']}")
        target = MULTIPLIER * record['draws_per_chain']
        spec = dict(task=task, key=record['key'], source_folder=relative(folder),
                    source_file=source_files['source']['path'], source_id=source['source_id'],
                    source_files=source_files, parent_files=parent_files,
                    parent_reference_id=parent['reference_id'], parent_identity=identity,
                    source_identity_sha256=fingerprint(identity['inputs']),
                    target_identity_sha256=fingerprint(identity['target']),
                    proposal_identity_sha256=fingerprint(identity['proposal']),
                    runtime_identity_sha256=fingerprint(identity['target']['runtime']),
                    baseline_draws=record['draws_per_chain'], target_draws=target,
                    estimated_peak_bytes=slot_bytes(target, record['order']),
                    baseline_elapsed_seconds=record['reported_elapsed_seconds'],
                    checkpoint_uncompressed_bytes=uncompressed)
        tasks.append(spec)
        numerical_hashes.update({'code/transformer/' + k: v for k, v in identity['proposal']['code_sha256'].items()})
        numerical_hashes.update({'code/scripts/' + k: v for k, v in identity['scoring_code'].items()})
        numerical_hashes.update({'code/transformer/' + k: v for k, v in identity['target']['numerical_code_hashes'].items()
                                 if '/' not in k})
    original = read(ORIGINAL_PLAN)
    worker = {key: original['worker'][key] for key in ('proposal', 'steps', 'burn', 'seed')}
    worker['python'] = str(RUNTIME_PYTHON)
    environment = {key: value for key, value in original['worker']['environment'].items() if value is not None}
    environment.update({key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    groups = [dict(depth=g['depth'], time=64, selected_indices=g['selected_indices'],
                   n_histories=g['n_histories']) for g in ready['groups']]
    manifest = dict(schema='unseen-context-refinement-manifest-v1', groups=groups, tasks=tasks,
                    readiness=file_binding(readiness_path), original_plan=file_binding(ORIGINAL_PLAN),
                    worker=worker, environment=environment, proposal=tasks[0]['parent_identity']['proposal'],
                    runtime=tasks[0]['parent_identity']['target']['runtime'], numerical_code_sha256=numerical_hashes,
                    provenance={relative(path): value['sha256'] for path, value in ready['provenance'].items()},
                    chain_storage=dict(compressed_bytes=sum(t['parent_files']['checkpoint']['size_bytes'] for t in tasks),
                                       uncompressed_bytes=all_uncompressed,
                                       largest_uncompressed_bytes=maximum_uncompressed))
    validate_tasks(manifest)
    for spec in tasks:
        verify_task_inputs(manifest, spec)
    for path, digest in numerical_hashes.items():
        require(sha(ROOT / path) == digest, f'Original numerical code changed: {path}')
    return manifest


def compact_plan(manifest, manifest_path=MANIFEST):
    elapsed = sum(t['baseline_elapsed_seconds'] for t in manifest['tasks'])
    inherited = read(ROOT/'output/shared_base_local_finish_2026-09-17/plan.json')['code_sha256']
    for name, digest in inherited.items():
        require(sha(safe_path(ROOT, name)) == digest, 'Inherited audit implementation changed: ' + name)
    implementations = dict(inherited)
    for name in ('code/scripts/prepare_unseen_context_refinement.py',
                 'code/scripts/unseen_context_refinement.py',
                 'code/scripts/shared_base_posterior_full_worker.py',
                 'code/transformer/start_unseen_context_refinement.sh'):
        implementations[name] = sha(ROOT/name)
    return dict(schema='unseen-context-refinement-plan-v1',
        scope='Precision-only continuation of all fixed L4/L8 unseen-context histories at time64; both orders.',
        manifest=dict(path=relative(manifest_path), sha256=hashlib.sha256(encoded(manifest)).hexdigest()),
        readiness=manifest['readiness'], groups=manifest['groups'], distinct_histories=123, reference_records=246,
        selection_rule='X_64 is absent from X_1,...,X_63; exact historical IDs retained without score-based selection.',
        draw_policy=dict(multiplier=MULTIPLIER, minimum_target=16384, maximum_target=1048576,
                         burn_repeated=False, automatic_budget_extension=False,
                         rationale='Uniform fourfold totals give idealized twofold MCSE reduction; all cases and both orders retained. Improvement is not guaranteed.'),
        tolerances_changed=False, training=False, model_inference=False, scientific_results_released=False,
        resources=dict(max_workers=4, memory_budget_bytes=18 * 1024**3, disk_reserve_bytes=50_000_000_000,
                       max_attempts=2, task_wall_seconds=7200, sampling_seconds=6000, audit_seconds=1100,
                       checkpoint_seconds=300, max_dispatch_seconds=43200,
                       maximum_worker_allocation_hours=984, total_worker_seconds=128 * 3600),
        feasibility=dict(baseline_reported_hours=elapsed / 3600,
                         conservative_linear_extra_hours=3 * elapsed / 3600,
                         slowest_linear_extra_seconds=3 * max(t['baseline_elapsed_seconds'] for t in manifest['tasks']),
                         largest_task_reserved_bytes=max(t['estimated_peak_bytes'] for t in manifest['tasks']),
                         input_chain_bytes=manifest['chain_storage']['compressed_bytes'],
                         largest_target_array_bytes=4 * manifest['chain_storage']['largest_uncompressed_bytes'],
                         estimated_final_compressed_chain_bytes=4 * manifest['chain_storage']['compressed_bytes'],
                         caveats=['Recorded elapsed includes burn-in; linear scaling is only a planning estimate.',
                                  'Report/checkpoint time and cache memory can grow nonlinearly; admission includes margin.',
                                  'Same saved runtime and absolute model paths required; native Jed longdouble64 cannot resume longdouble53 identity.']),
        runtime=manifest['runtime'], accumulator_mantissa_bits=53,
        execution_location='Original local host with exactly matching runtime; no portability override.',
        code_sha256=implementations)


def read_plan(path=PLAN):
    path = Path(path)
    compact = read(path)
    require(compact['schema'] == 'unseen-context-refinement-plan-v1', 'Unexpected plan schema')
    manifest_path = safe_path(ROOT, compact['manifest']['path'])
    require(sha(manifest_path) == compact['manifest']['sha256'], 'Frozen manifest checksum mismatch')
    manifest = read(manifest_path)
    require(manifest['schema'] == 'unseen-context-refinement-manifest-v1' and
            manifest['groups'] == compact['groups'] and manifest['readiness'] == compact['readiness'],
            'Manifest differs from committed cohort')
    require(compact == compact_plan(manifest, manifest_path),
            'Frozen cohort, draw budget, resource limits or implementation bindings changed')
    expanded = dict(compact, **{key: value for key, value in manifest.items() if key != 'schema'})
    validate_tasks(expanded)
    for name, digest in dict(expanded['numerical_code_sha256'], **compact['code_sha256']).items():
        require(sha(safe_path(ROOT, name)) == digest, f'Frozen code changed: {name}')
    return expanded, sha(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readiness', type=Path, default=READINESS)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--write', action='store_true', help='Write immutable manifest and compact plan; never run sampling.')
    args = parser.parse_args(argv)
    manifest = build_manifest(args.readiness)
    plan = compact_plan(manifest, args.manifest)
    if args.write:
        immutable_json(args.manifest, manifest)
        immutable_json(args.plan, plan)
    print(json.dumps(dict(written=args.write, histories=123, references=246,
                         manifest_sha256=plan['manifest']['sha256'], feasibility=plan['feasibility'],
                         raw_hashes_recomputed=False, sampling=False), indent=2))


if __name__ == '__main__':
    main()
