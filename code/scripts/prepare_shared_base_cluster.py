"""Export frozen test inputs and package the existing sealed LSA dependency.

No sources are generated or transformers retrained here. The pilot uses
saved full-prior sequences 1..4 in each family, disjoint from sequence 0
used for numerical calibration. The numerical dependency archive stays
outside git; its file hashes are committed with the portable plan.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
from shared_base_short_evidence import SharedBaseShortEvidence

CONFIG = ROOT / 'code/transformer/configs'
MANIFEST = CONFIG / 'shared_base_dependency_manifest_2026-09-16.json'
PLAN = CONFIG / 'shared_base_cluster_plan_2026-09-16.json'
GPU_CASES = CONFIG / 'shared_base_gpu_cases_2026-09-16.json'
FAMILIES = ('product', 'shared_base', 'uniform_base')
TIMES = [1, 4, 16, 64, 256, 1024]
DEP_NAME = 'shared_base_dependencies_2026-09-16'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != raw:
        raise ValueError(f'Refusing to replace a different frozen artifact: {path}')
    path.write_text(raw)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack_dependency(companion, archive, manifest_path):
    companion, archive = Path(companion).resolve(), Path(archive)
    names = subprocess.check_output(['git', '-C', str(companion), 'ls-files',
                                     'src/product_model_with_memory'], text=True).splitlines()
    names = [n for n in names if Path(n).suffix in ('.py', '.c')]
    names += ['tables/anchors_prod/' + n for n in ['manifest.json', 'anchors.json'] +
              [f'level_{level:02d}.{suffix}' for level in range(2, 9) for suffix in ('bin', 'index.json')]]
    require(bool(names), 'No numerical dependency files found')
    members = {}
    for name in sorted(names):
        path = companion / name
        require(path.is_file() and not path.is_symlink(), f'Missing or nonregular dependency {name}')
        members[name] = dict(sha256=sha(path), size_bytes=path.stat().st_size)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + '.tmp')
    with temporary.open('wb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for name in members:
                info = tar.gettarinfo(str(companion/name), arcname=f'{DEP_NAME}/{name}')
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with (companion/name).open('rb') as source:
                    tar.addfile(info, source)
    digest = sha(temporary)
    if archive.exists() and sha(archive) != digest:
        temporary.unlink()
        raise ValueError('Refusing to replace a different dependency archive')
    temporary.replace(archive)
    manifest = dict(schema='shared-base-dependency-manifest-v1', root_directory=DEP_NAME,
        source_repository='https://github.com/urbanke/product_model_with_memory',
        source_commit=subprocess.check_output(['git', '-C', str(companion), 'rev-parse', 'HEAD'], text=True).strip(),
        archive_filename=archive.name, archive_sha256=digest, archive_size_bytes=archive.stat().st_size,
        files=members, store_levels=list(range(2, 9)),
        scope='Unmodified existing numerical package plus required immutable sealed-store files; no compiled host binaries or private data.')
    write_json(manifest_path, manifest)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest, files=len(members))))


def unpack_dependency(archive, destination, manifest_path):
    archive, destination = Path(archive), Path(destination)
    manifest = json.loads(Path(manifest_path).read_text())
    require(sha(archive) == manifest['archive_sha256'], 'Dependency archive hash mismatch')
    root = destination / manifest['root_directory']
    require(not root.exists(), f'Dependency directory already exists: {root}; verify it instead of overwriting')
    expected = {f"{manifest['root_directory']}/{name}" for name in manifest['files']}
    with tarfile.open(archive, 'r:gz') as tar:
        entries = tar.getmembers()
        require(len(entries) == len(expected) and {x.name for x in entries} == expected,
                'Unexpected or repeated dependency archive members')
        require(all(x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts
                    for x in entries), 'Unsafe dependency archive member')
        destination.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            path = destination / entry.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(entry) as source, path.open('xb') as target:
                for block in iter(lambda: source.read(1024*1024), b''):
                    target.write(block)
    verify_dependency(root, manifest_path)


def verify_dependency(root, manifest_path):
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    for name, expected in manifest['files'].items():
        path = root/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['size_bytes']
                and sha(path) == expected['sha256'], f'Dependency file mismatch: {name}')
    print(json.dumps(dict(status='verified', root=str(root), files=len(manifest['files']),
                         scientific_reference_validated=False)))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def export_plan(raw, family_root, plan_path, gpu_path, manifest_path):
    raw, family_root = Path(raw), Path(family_root)
    exact = SharedBaseShortEvidence()
    numerical = json.loads((family_root/'validation.json').read_text())
    require(numerical.get('passed') is True, 'Family references require passed numerical validation')
    source_manifest = json.loads((raw/'heldout/manifest.json').read_text())
    source_plan_id = source_manifest['plan_id']
    cases, comparison = [], {}
    verified = {str(p.relative_to(ROOT)): sha(p) for p in
                (family_root/'validation.json', raw/'heldout/manifest.json', manifest_path)}
    for family in FAMILIES:
        cohort = f'{family}_prior'
        path = raw/'heldout'/f'{cohort}.npz'
        record = json.loads(path.with_suffix('.json').read_text())
        require(record['data_sha256'] == sha(path) and record['plan_id'] == source_plan_id,
                'Source cohort identity mismatch')
        source = load_npz(path)
        require(source['tokens'].shape == (1024,1025) and source['prefixes'].tolist() == TIMES,
                'Unexpected cohort dimensions/time grid')
        verified[str(path.relative_to(ROOT))] = sha(path)
        refs, models = {}, {}
        for ref in ('product','uniform_base'):
            file = family_root/f'{ref}_on_{cohort}.npz'
            meta = json.loads(file.with_suffix('.json').read_text())
            require(meta['complete'] is True and meta['data_sha256'] == sha(file)
                    and meta['source_data_sha256'] == record['data_sha256']
                    and meta['validation_sha256'] == sha(family_root/'validation.json')
                    and meta['reference_family'] == ref and meta['evidence_kind'] == 'normalized_sequence_marginal',
                    'Reference identity mismatch')
            refs[ref] = load_npz(file)
            require(np.array_equal(refs[ref]['sequence_seed'],source['sequence_seed'])
                    and refs[ref]['prefixes'].tolist() == TIMES, 'Reference alignment mismatch')
            verified[str(file.relative_to(ROOT))] = sha(file)
            verified[str(file.with_suffix('.json').relative_to(ROOT))] = sha(file.with_suffix('.json'))
        for seed in (0,1):
            model = f'pooled_seed{seed}'
            file = raw/'eval'/model/f'{cohort}.npz'
            meta = json.loads(file.with_suffix('.json').read_text())
            manifest = json.loads((file.parent/'manifest.json').read_text())
            require(meta['data_sha256'] == sha(file) and meta['source_data_sha256'] == record['data_sha256']
                    and meta['run_id'] == manifest['run_id'] and meta['identity'] == manifest['identity'],
                    'Transformer identity mismatch')
            models[model] = load_npz(file)
            require(np.array_equal(models[model]['sequence_seed'],source['sequence_seed'])
                    and models[model]['prefixes'].tolist() == TIMES,'Transformer alignment mismatch')
            verified[str(file.relative_to(ROOT))] = sha(file)
            verified[str(file.with_suffix('.json').relative_to(ROOT))] = sha(file.with_suffix('.json'))
        for index in range(5):
            short = index == 0
            tokens = source['tokens'][index,:5 if short else 1025].astype(int).tolist()
            case = dict(id=f"{'cal' if short else 'pilot'}_{family}_{index}",
                        name=f'{family}_prior_{index}', kind='frozen_heldout_prefix', family=family,
                        tokens=tokens, source_identity=dict(source_plan_id=source_plan_id,
                        cohort=cohort, cohort_sha256=record['data_sha256'],sequence_index=index,
                        sequence_seed=source['sequence_seed'][index].tolist()))
            if short:
                case['known_log_evidence'] = exact.log_joint(tokens)
            else:
                comparison[case['id']] = dict(
                    product_log_marginals=refs['product']['log_marginals'][index].tolist(),
                    uniform_base_log_marginals=refs['uniform_base']['log_marginals'][index].tolist(),
                    shared_base_short_log_marginals=[exact.log_joint(tokens[:t+1]) for t in TIMES[:2]],
                    transformer_cumulative_loss_bits={name: values['cumulative_loss_bits'][index].tolist()
                                                       for name,values in models.items()})
            cases.append(case)
    for name,tokens in [('aaaaa',[0,0,0,0,0]),('ababa',[0,1,0,1,0]),('abaca',[0,1,0,2,0])]:
        cases.append(dict(id=f'cal_{name}',name=name,kind='exact_stress_control',tokens=tokens,
                          known_log_evidence=exact.log_joint(tokens)))
    cal = [c['id'] for c in cases if c['id'].startswith('cal_')]
    pilot = [c['id'] for c in cases if c['id'].startswith('pilot_')]
    def stage(purpose, requires, ids, length, particles, repetitions, seed, order=16, bounds=(-32.,32.), step=1/32):
        return dict(purpose=purpose,requires=requires,case_ids=ids,n_tokens=length,
                    particles=particles,temperatures=64,moves=4,replicates=repetitions,
                    alpha_order=order,seed=seed,grid_bounds=list(bounds),grid_step=step,sampler='original')
    stages = {
        'preflight': stage('preflight',[],cal,5,256,2,2026091620),
        'calibrate': stage('exact_short',['preflight'],cal,5,8192,64,2026091621),
        'confirm': stage('confirmation',['calibrate'],cal,5,16384,128,2026091622),
        'alpha_refine': stage('quadrature_refinement',['confirm'],cal,5,16384,128,2026091623,order=32),
        'row_refine': stage('quadrature_refinement',['confirm'],cal,5,16384,128,2026091624,bounds=(-36.,36.),step=1/64),
    }
    previous = ['calibrate','confirm','alpha_refine','row_refine']
    for j,t in enumerate(TIMES[2:]):
        name = f'pilot{t}'
        stages[name] = stage('repeatability',previous,pilot,t+1,256,16,2026091630+j)
        previous = [name]
    gpu = dict(schema='shared-base-gpu-cases-v1',cases=[c for c in cases if c['id'] in cal],
               prior=dict(alphabet=64,alpha_uniform=[.5,1.5],depth_uniform=list(range(9))))
    write_json(gpu_path,gpu)
    plan = dict(schema='shared-base-cluster-plan-v1',date='2026-09-16',
        purpose='Calibrate normalized shared-base evidence, then compare pooled transformers with the informed three-family mixture on 12 fixed sequences.',
        production_reference_approved=False,
        prior=dict(alphabet=64,alpha_uniform=[.5,1.5],depth_uniform=list(range(9)),permutation='uniform'),
        cases=cases,stages=stages,
        gpu_cases=str(gpu_path.relative_to(ROOT)),gpu_cases_sha256=sha(gpu_path),
        dependency_manifest=str(manifest_path.relative_to(ROOT)),dependency_manifest_sha256=sha(manifest_path),
        acceptance=dict(exact_short=dict(tolerance_bits=.01,minimum_completed_replicates=16,
                        minimum_effective_replica_count=8,maximum_single_replica_likelihood_share=.2),
                        long_precision_bits=.01,production_reference_approved=False),
        selection='Pilot indices1..4 in each existing full-prior cohort; these held-out sequences were scored by both saved transformers but have not been used to tune the Bayesian sampler. Calibration uses index0. No generating parameters are exported.',
        comparison=dict(prefixes=TIMES,family_order=list(FAMILIES),family_prior=[1/3]*3,
                        loss_convention='First token supplied/unscored. At time t, predict tokens[1:t+1].',cases=comparison),
        provenance=dict(input_sha256=verified,short_catalog_validation=exact.validation,
                        generator_sha256=sha(__file__)),
        limitations=['Larger budgets and GPU speedups are proposed, not measured.',
                     'Long stages are diagnostic; agreement and small Monte Carlo intervals do not alone rule out missed modes.',
                     'No full-cohort run or manuscript update is automatic.'])
    write_json(plan_path,plan)
    print(json.dumps(dict(plan=str(plan_path),sha256=sha(plan_path),short_cases=len(cal),pilot_cases=len(pilot))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    pack=sub.add_parser('bundle')
    pack.add_argument('--companion',type=Path,default=ROOT.parent/'product_model_with_memory')
    pack.add_argument('--archive',type=Path,default=ROOT/'output/scitas_shared_base_2026-09-16/dependencies.tar.gz')
    pack.add_argument('--manifest',type=Path,default=MANIFEST)
    extract=sub.add_parser('unpack')
    extract.add_argument('--archive',type=Path,required=True)
    extract.add_argument('--destination',type=Path,required=True)
    extract.add_argument('--manifest',type=Path,default=MANIFEST)
    verify=sub.add_parser('verify')
    verify.add_argument('--root',type=Path,required=True)
    verify.add_argument('--manifest',type=Path,default=MANIFEST)
    plan=sub.add_parser('plan')
    plan.add_argument('--raw',type=Path,default=ROOT/'output/multifamily_results_2026-09-16')
    plan.add_argument('--family-root',type=Path,default=ROOT/'output/multifamily_bayes_2026-09-16/family_evidence')
    plan.add_argument('--out',type=Path,default=PLAN)
    plan.add_argument('--gpu-cases',type=Path,default=GPU_CASES)
    plan.add_argument('--manifest',type=Path,default=MANIFEST)
    args=parser.parse_args()
    if args.action=='bundle':pack_dependency(args.companion,args.archive,args.manifest)
    elif args.action=='unpack':unpack_dependency(args.archive,args.destination,args.manifest)
    elif args.action=='verify':verify_dependency(args.root,args.manifest)
    else:export_plan(args.raw,args.family_root,args.out,args.gpu_cases,args.manifest)


if __name__=='__main__':
    main()
