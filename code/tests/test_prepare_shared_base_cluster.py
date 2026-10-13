"""Portable inputs retain the frozen cases; dependency unpacking is checked."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_cluster_inputs', ROOT/'code/scripts/prepare_shared_base_cluster.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def archive_fixture(tmp_path, member='dependency/src/numerics.py', payload=b'fixed numerical source\n'):
    archive = tmp_path/'dependency.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(root_directory='dependency', archive_sha256=prepare.sha(archive),
        files={member.removeprefix('dependency/'):dict(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})))
    return archive, manifest


def test_dependency_roundtrip_and_content_corruption(tmp_path):
    archive, manifest = archive_fixture(tmp_path)
    destination = tmp_path/'extracted'
    prepare.unpack_dependency(archive, destination, manifest)
    prepare.verify_dependency(destination/'dependency', manifest)
    with pytest.raises(ValueError, match='already exists'):
        prepare.unpack_dependency(archive, destination, manifest)
    (destination/'dependency/src/numerics.py').write_bytes(b'changed numerical data\n')
    with pytest.raises(ValueError, match='mismatch'):
        prepare.verify_dependency(destination/'dependency', manifest)


def test_archive_mismatch_prevents_any_extraction(tmp_path):
    archive, manifest = archive_fixture(tmp_path)
    archive.write_bytes(archive.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='archive hash mismatch'):
        prepare.unpack_dependency(archive, tmp_path/'destination', manifest)
    assert not (tmp_path/'destination').exists()


def test_unsafe_member_rejected_even_with_matching_archive_hash(tmp_path):
    archive, manifest = archive_fixture(tmp_path, 'dependency/../escaped.py')
    with pytest.raises(ValueError, match='Unsafe'):
        prepare.unpack_dependency(archive, tmp_path/'destination', manifest)
    assert not (tmp_path/'escaped.py').exists()


def test_frozen_artifact_cannot_be_silently_replaced(tmp_path):
    path = tmp_path/'frozen.json'
    prepare.write_json(path, {'version':1})
    prepare.write_json(path, {'version':1})
    with pytest.raises(ValueError, match='Refusing to replace'):
        prepare.write_json(path, {'version':2})
    assert json.loads(path.read_text()) == {'version':1}


def test_committed_plan_balances_pilot_without_exporting_hidden_parameters():
    plan = json.loads(prepare.PLAN.read_text())
    calibration = [c for c in plan['cases'] if c['id'].startswith('cal_')]
    pilot = [c for c in plan['cases'] if c['id'].startswith('pilot_')]
    assert len(calibration) == 6 and len(pilot) == 12
    allowed = {'id','name','kind','family','tokens','source_identity','known_log_evidence'}
    assert all(set(case) <= allowed for case in plan['cases'])
    for family in prepare.FAMILIES:
        selected = [c for c in pilot if c['family']==family]
        assert {c['source_identity']['sequence_index'] for c in selected} == {1,2,3,4}
        assert all(len(c['tokens'])==1025 and 'known_log_evidence' not in c for c in selected)
    assert set(plan['comparison']['cases']) == {c['id'] for c in pilot}
    assert plan['gpu_cases_sha256'] == prepare.sha(ROOT/plan['gpu_cases'])
    assert plan['dependency_manifest_sha256'] == prepare.sha(ROOT/plan['dependency_manifest'])
    assert plan['provenance']['generator_sha256'] == prepare.sha(prepare.__file__)
