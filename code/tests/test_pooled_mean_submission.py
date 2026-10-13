"""Scheduler-contract checks with a fake scheduler, never real submissions."""
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'repo'
    scripts = root / 'code/transformer/scitas'
    scripts.mkdir(parents=True)
    for name in ('submit_pooled_mean_population.sh', 'pooled_mean_population.sbatch'):
        shutil.copy2(ROOT / 'code/transformer/scitas' / name, scripts / name)
    (root / 'code/scripts').mkdir()
    (root / 'code/scripts/pooled_mean_population.py').write_text('# test worker\n')
    plan = root / 'plan.json'
    plan.write_text(json.dumps(dict(schema='pooled-mean-population-plan-v1', observed=16,
        scheduler=dict(concurrency=64, maximum_array_size=768, prepare_minutes=60,
            merge_minutes=60, cpu_hour_rate_chf=.0055,
            calibration=dict(tasks=52, allocation_minutes=30),
            population=dict(tasks=768, allocation_minutes=240)))))
    raw, dependencies = tmp_path / 'raw', tmp_path / 'dependencies'
    (raw / 'heldout').mkdir(parents=True)
    (raw / 'heldout/manifest.json').write_text('{}')
    (dependencies / 'tables/anchors_prod').mkdir(parents=True)
    (dependencies / 'tables/anchors_prod/anchors.json').write_text('{}')
    binary = tmp_path / 'bin'
    binary.mkdir()
    calls = tmp_path / 'calls.jsonl'
    scheduler = binary / 'sbatch'
    scheduler.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
p=pathlib.Path(os.environ['FAKE_SCHEDULER_CALLS'])
n=len(p.read_text().splitlines()) if p.exists() else 0
with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAKE_FAIL_CALL')==str(n+1): sys.exit(1)
print(1000+n)
''')
    scheduler.chmod(0o755)
    (binary / 'squeue').write_text('#!/bin/sh\necho "1000|pmean16-prepare|PENDING|0:00"\n')
    (binary / 'squeue').chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       FAKE_SCHEDULER_CALLS=str(calls))
    out = root / 'output/test'
    base = ['bash', str(scripts / 'submit_pooled_mean_population.sh'), '--plan', str(plan),
            '--raw', str(raw), '--dependencies', str(dependencies), '--out', str(out)]
    return dict(root=root, plan=plan, out=out, calls=calls, env=environment, base=base)


def run(campaign, *options):
    return subprocess.run(campaign['base'] + list(options), text=True,
                          capture_output=True, env=campaign['env'])


CAPS = ('--max-cpu-hours', '3101', '--max-cost-chf', '30')


def test_dry_run_never_writes_or_contacts_scheduler(campaign):
    result = run(campaign, '--dry-run')
    assert result.returncode == 0, result.stderr
    assert not campaign['out'].exists()
    assert not campaign['calls'].exists()
    assert '"maximum_cpu_hours": 3101.0' in result.stdout
    assert '"maximum_cost_chf": 17.0555' in result.stdout
    assert '--dependency=afterok:DRY_calibration' in result.stdout


@pytest.mark.parametrize('options', [(), ('--max-cpu-hours', '3100', '--max-cost-chf', '30'),
                                    ('--max-cpu-hours', '3101', '--max-cost-chf', '17')])
def test_missing_or_exceeded_caps_reject_before_output(campaign, options):
    result = run(campaign, '--submit', *options)
    assert result.returncode != 0
    assert not campaign['out'].exists()
    assert not campaign['calls'].exists()


def test_dependency_graph_receipt_and_duplicate_guard(campaign):
    result = run(campaign, '--submit', *CAPS)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in campaign['calls'].read_text().splitlines()]
    assert len(calls) == 5
    assert '--array=0-51%64' in calls[1]
    assert '--dependency=afterok:1000' in calls[1]
    assert '--dependency=afterany:1001' in calls[2]
    assert '--array=0-767%64' in calls[3]
    assert '--dependency=afterok:1002' in calls[3]
    assert '--dependency=afterany:1003' in calls[4]
    assert all('--kill-on-invalid-dep=yes' in call for call in calls[1:])
    receipt = json.loads((campaign['out'] / 'submission.json').read_text())
    assert receipt['status'] == 'submitted'
    assert receipt['population_submitted'] is True
    assert receipt['population_waits_for_calibration'] is True
    assert receipt['identity']['maximum_cpu_hours'] == 3101
    assert receipt['identity']['maximum_cost_chf'] == pytest.approx(17.0555)
    assert receipt['identity']['approved_cost_cap_chf'] == 30
    assert '"sampling_start_verified": false' in result.stdout
    duplicate = run(campaign, '--submit', *CAPS)
    assert duplicate.returncode == 0
    assert 'already_submitted' in duplicate.stdout
    assert len(campaign['calls'].read_text().splitlines()) == 5


def test_uncertain_partial_launch_cannot_be_repeated(campaign):
    campaign['env']['FAKE_FAIL_CALL'] = '2'
    failed = run(campaign, '--submit', *CAPS)
    assert failed.returncode != 0
    receipt = json.loads((campaign['out'] / 'submission.json').read_text())
    assert receipt['population_submitted'] is False
    assert receipt['submitting']['stage'] == 'calibration'
    assert [job['job_id'] for job in receipt['jobs']] == ['1000']
    campaign['env'].pop('FAKE_FAIL_CALL')
    duplicate = run(campaign, '--submit', *CAPS)
    assert duplicate.returncode != 0
    assert 'uncertain' in duplicate.stderr
    assert len(campaign['calls'].read_text().splitlines()) == 2


def test_chunks_share_concurrency_and_remain_disjoint(campaign):
    plan = json.loads(campaign['plan'].read_text())
    plan['scheduler']['maximum_array_size'] = 384
    campaign['plan'].write_text(json.dumps(plan))
    result = run(campaign, '--submit', *CAPS)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in campaign['calls'].read_text().splitlines()]
    assert '--array=0-383%32' in calls[3]
    assert '--array=0-383%32' in calls[4]
    assert calls[3][-1] == '0'
    assert calls[4][-1] == '384'
    assert '--dependency=afterany:1003:1004' in calls[5]


def test_only_t16_and_bounded_workers(campaign):
    plan = json.loads(campaign['plan'].read_text())
    plan['observed'] = 64
    campaign['plan'].write_text(json.dumps(plan))
    assert run(campaign, '--dry-run').returncode != 0
    plan['observed'] = 16
    plan['scheduler']['concurrency'] = 65
    campaign['plan'].write_text(json.dumps(plan))
    assert run(campaign, '--dry-run').returncode != 0


@pytest.mark.parametrize('before,after', [('--mem=6G', '--mem=8G'),
                                       ('--cpus-per-task=1', '--cpus-per-task=2')])
def test_partition_memory_and_cost_cpu_contract_before_submission(campaign, before, after):
    script = campaign['root'] / 'code/transformer/scitas/pooled_mean_population.sbatch'
    original = script.read_text()
    assert before in original
    script.write_text(original.replace(before, after))
    result = run(campaign, '--submit', *CAPS)
    assert result.returncode != 0
    assert not campaign['calls'].exists()
    assert not campaign['out'].exists()
