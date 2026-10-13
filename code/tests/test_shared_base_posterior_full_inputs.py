"""Independent checks of full-cohort selection and portable input safety."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_shared_base_posterior_full as prep


def test_exact_missing_population_and_primary_integration_orders():
    tasks = prep.task_list()
    expected = {(d, i, t, o) for d in (0, 1, 4, 8) for i in range(2, 256)
                for t in (256, 1024) for o in ((32, 64) if (d, t) == (0, 1024) else (16, 32))}
    actual = {(r['depth'], r['index'], r['time'], r['order']) for r in tasks}
    assert len(tasks) == len(actual) == 4064 and actual == expected
    assert len({r['id'] for r in tasks}) == 4064
    assert len(prep.input_names()) == 5090
    assert all('/seq_000000.' not in p and '/seq_000001.' not in p for p in prep.input_names())
    assert {'pilot/base.json', 'pilot/refinement.json'} <= set(prep.input_names())
    assert len(prep.task_input_names(tasks[0])) == 7


@pytest.fixture
def package(tmp_path):
    source = tmp_path / 'source'
    names = ['a1_L0/protocol.json', 'a1_L0/data/seq_000002.json', 'pilot/base.json']
    files = {}
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = dict(sha256=prep.sha(path), size_bytes=path.stat().st_size)
    archive = tmp_path / 'inputs.tar.gz'
    prep.pack_files(source, archive, files)
    plan = dict(inputs=dict(root_directory=prep.BUNDLE_ROOT, files=files, archive_sha256=prep.sha(archive)))
    return source, archive, plan


def test_deterministic_archive_roundtrip_and_corruption(package, tmp_path):
    source, archive, plan = package
    second = tmp_path / 'second.tar.gz'
    prep.pack_files(source, second, plan['inputs']['files'])
    assert prep.sha(archive) == prep.sha(second)
    root = prep.unpack(archive, tmp_path / 'destination', plan)
    assert prep.verify_files(root, plan, list(plan['inputs']['files'])) == 3
    with pytest.raises(ValueError, match='already exists'):
        prep.unpack(archive, tmp_path / 'destination', plan)
    (root / 'pilot/base.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Frozen full-cohort input mismatch'):
        prep.verify_files(root, plan, ['pilot/base.json'])


@pytest.mark.parametrize('kind', ['symlink', 'duplicate', 'traversal'])
def test_unsafe_archive_rejected_without_installation(package, tmp_path, kind):
    _, _, plan = package
    bad = tmp_path / 'bad.tar.gz'
    with tarfile.open(bad, 'w:gz') as tar:
        for index, (name, meta) in enumerate(plan['inputs']['files'].items()):
            entry = tarfile.TarInfo(f'{prep.BUNDLE_ROOT}/{name}')
            entry.size = meta['size_bytes']
            if index == 0 and kind == 'symlink':
                entry.type = tarfile.SYMTYPE
                entry.linkname = '/tmp/escape'
                tar.addfile(entry)
            else:
                if index == 0 and kind == 'traversal':
                    entry.name = '../escape'
                tar.addfile(entry, io.BytesIO(name.encode()))
                if index == 0 and kind == 'duplicate':
                    tar.addfile(entry, io.BytesIO(name.encode()))
    plan = deepcopy(plan)
    plan['inputs']['archive_sha256'] = prep.sha(bad)
    destination = tmp_path / 'bad-destination'
    with pytest.raises(ValueError):
        prep.unpack(bad, destination, plan)
    assert not (destination / prep.BUNDLE_ROOT).exists()


def test_task_scoped_verification_does_not_scan_other_sequences(tmp_path):
    task, other = prep.task_list()[0], prep.task_list()[4]
    names = prep.task_input_names(task)
    files = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = dict(sha256=prep.sha(path), size_bytes=path.stat().st_size)
    plan = dict(tasks=[task, other], inputs=dict(files=files))
    assert prep.verify_task_inputs(tmp_path, plan, task) == 7
    with pytest.raises(ValueError, match='not declared'):
        prep.verify_task_inputs(tmp_path, plan, dict(task, index=999))
