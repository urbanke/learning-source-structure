"""Preview or explicitly submit the bounded overnight shared-base controller."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/scripts'))
import shared_base_overnight as workflow


def main(argv=None):
    dependency = Path(os.environ.get('CLUSTER_DEP_ROOT', str(Path.home() / 'Projects/shared_base_dependencies_2026-09-16')))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-job', help='Completed or running ABACA array ID (all 32 tasks must succeed)')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/scitas_shared_base_overnight_2026-09-16')
    parser.add_argument('--original-receipt', type=Path, default=ROOT / 'output/scitas_shared_base_2026-09-16/merged/calibrate.json')
    parser.add_argument('--followup-shards', type=Path, default=ROOT / 'output/scitas_shared_base_abaca_2026-09-16/shards/abaca_check')
    parser.add_argument('--store', type=Path, default=Path(os.environ.get('SHARED_BASE_STORE', str(dependency / 'tables/anchors_prod'))))
    parser.add_argument('--src', type=Path, default=Path(os.environ.get('SHARED_BASE_SRC', str(dependency / 'src'))))
    parser.add_argument('--venv', type=Path, default=Path(os.environ.get('SHARED_BASE_VENV', str(Path.home() / 'venv-tf'))))
    parser.add_argument('--account', default=os.environ.get('SHARED_BASE_ACCOUNT'))
    parser.add_argument('--resume', action='store_true', help='Attach to saved child job IDs after a stopped coordinator; never retry failed experiments')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--submit', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        out = args.out.expanduser().resolve()
        launch_path = out / 'launch.json'
        if args.resume:
            workflow.require(launch_path.is_file(), 'No saved overnight launch to resume')
            launch = json.loads(launch_path.read_bytes())
            protocol = workflow.validate_launch(launch)
            workflow.require(launch['out'] == str(out), 'Output identity changed')
            workflow.require(args.after_job is None or args.after_job == launch['after_job'], 'ABACA dependency cannot change on resume')
        else:
            workflow.require(args.after_job is not None and re.fullmatch(r'[0-9]+', args.after_job), '--after-job must be an ABACA array ID')
            protocol = workflow.read_protocol(workflow.DEFAULT_PROTOCOL)
            launch = dict(schema=workflow.SCHEMA, repo_root=str(ROOT), protocol=str(workflow.DEFAULT_PROTOCOL),
                protocol_sha256=workflow.sha(workflow.DEFAULT_PROTOCOL), code_sha256=workflow.frozen_code(),
                after_job=args.after_job, out=str(out), original_receipt=str(args.original_receipt.expanduser().resolve()),
                followup_shards=str(args.followup_shards.expanduser().resolve()), store=str(args.store.expanduser().resolve()),
                src=str(args.src.expanduser().resolve()), venv=str(args.venv.expanduser().resolve()), account=args.account,
                poll_seconds=60, controller_seconds=(protocol['coordinator_wall_minutes'] - 5) * 60)
            launch['launch_id'] = workflow.fingerprint(launch)
        original = Path(launch['original_receipt']).parent.parent
        followup = Path(launch['followup_shards']).parent.parent
        workflow.require(not any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in (original, followup)),
                         'Use a separate output root; original experiments must remain unchanged')
        command = ['sbatch', '--parsable', f'--chdir={ROOT}',
                   f"--job-name=sbo-controller-{launch['launch_id'][:12]}",
                   '--time=1-00:00:00', f'--output={out}/logs/controller-%j.out']
        if not args.resume:
            command += [f"--dependency=afterany:{launch['after_job']}", '--kill-on-invalid-dep=yes']
        if launch['account']:
            command.append(f"--account={launch['account']}")
        command += [str(ROOT / 'code/transformer/scitas/shared_base_overnight.sbatch'), str(launch_path)]
        if args.submit and not args.resume:
            # A completed array may have aged out of Slurm's dependency table.
            # Accounting still verifies every task; the controller repeats it.
            dependency_state, _ = workflow.Slurm(launch['venv']).array_state(launch['after_job'], protocol['followup_tasks'])
            workflow.require(dependency_state in ('active', 'success'),
                             'ABACA failed or has incomplete scheduler accounting; no overnight job submitted')
            if dependency_state == 'success':
                command = [part for part in command if not part.startswith('--dependency=')
                           and part != '--kill-on-invalid-dep=yes']
        print(json.dumps(dict(action='submit' if args.submit else 'dry run', stage_groups=protocol['stage_groups'],
            stop_after='pilot16', pilot_particles=16384, pilot_replicates=32,
            full_cohort_queued=False, original_failure_preserved=True)), flush=True)
        print(shlex.join(command), flush=True)
        if not args.submit:
            return 0
        workflow.require(Path(launch['original_receipt']).is_file(), 'Original complete calibration receipt is missing')
        workflow.require(Path(launch['followup_shards']).is_dir(), 'ABACA shard directory is missing')
        workflow.require((Path(launch['venv']) / 'bin/python').is_file(), 'Prepared Python environment is missing')
        workflow.require((Path(launch['store']) / 'anchors.json').is_file()
                         and (Path(launch['store']) / 'manifest.json').is_file(), 'Sealed numerical store is missing')
        workflow.require((Path(launch['src']) / 'product_model_with_memory/universal_tables.py').is_file(), 'Numerical source is missing')
        # Only this tiny submission operation runs on the login node.
        out.mkdir(parents=True, exist_ok=True)
        with workflow.lock(out / 'submission.lock'):
            saved = out / 'submission.json'
            slurm = workflow.Slurm(launch['venv'])
            if args.resume:
                previous = json.loads(saved.read_bytes())
                workflow.require('job_id' in previous, 'Previous submission outcome is uncertain; inspect scheduler before resuming')
                status, _ = slurm.array_state(previous['job_id'], 1)
                workflow.require(status in ('success', 'failed'), 'Previous coordinator is still active or its scheduler records are missing')
                state_path = out / 'status.json'
                if state_path.exists():
                    state = json.loads(state_path.read_bytes())
                    workflow.require(state.get('status') not in ('complete', 'stopped'), 'Completed or failed experiments are not automatically rerun')
            else:
                workflow.require(not launch_path.exists() and not saved.exists(), 'Output already has a launch; use --resume to attach after coordinator interruption')
            workflow.immutable_json(launch_path, launch)
            (out / 'logs').mkdir(exist_ok=True)
            previous_submissions = []
            if saved.exists():
                previous = json.loads(saved.read_bytes())
                previous_submissions = previous.get('previous_submissions', []) + [{k: v for k, v in previous.items() if k != 'previous_submissions'}]
            receipt = dict(launch_id=launch['launch_id'], phase='submitting', command=command,
                           previous_submissions=previous_submissions)
            workflow.atomic_json(saved, receipt)
            receipt['job_id'] = slurm.submit(command)
            receipt['phase'] = 'submitted'
            workflow.atomic_json(saved, receipt)
            print(receipt['job_id'], flush=True)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
