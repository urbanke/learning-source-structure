"""Bounded one-job submission; login-node work is hashes and file presence only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'output/shared_base_shuffle_extension_2026-09-22')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != ROOT/'output':
        raise ValueError('Use a fresh direct child of this checkout/output')
    runner = ROOT/'code/scripts/extend_shared_base_shuffle.py'
    plan = ROOT/'code/transformer/configs/shared_base_shuffle_extension_2026-09-22.json'
    batch = ROOT/'code/transformer/kuma/shared_base_shuffle_extension.sbatch'
    command = ['sbatch', '--parsable', '--chdir='+str(ROOT), '--job-name=lsa-shuffle-ext',
               '--output='+str(out/'logs/job-%j.out'), str(batch), str(out), sha(runner), sha(plan)]
    receipt = dict(schema='shared-base-shuffle-extension-submission-v1', command=command,
                   gpus=1, cpus=8, memory_gib=32, maximum_hours=2, maximum_gpu_hours=2,
                   maximum_cpu_hours=16, automatic_retries=False, code_sha256=sha(runner),
                   plan_sha256=sha(plan), batch_sha256=sha(batch))
    print(json.dumps(receipt), flush=True)
    if not args.submit:
        return
    if out.exists():
        raise ValueError('Output already exists: inspect receipt/job before any new submission')
    for seed in (0, 1):
        for training in ('shared_base_scaling_2026-09-17', 'shared_base_shuffle_2026-09-17'):
            if not (ROOT/'output'/training/'train'/f'seed{seed}'/'model_100000.pt').is_file():
                raise ValueError('Required checkpoint is missing')
    (out/'logs').mkdir(parents=True)
    (out/'frozen').mkdir()
    for path in (runner, plan, batch):
        shutil.copy2(path, out/'frozen'/path.name)
    receipt['status'] = 'submitting'
    (out/'submission.json').write_text(json.dumps(receipt, indent=2)+'\n')
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, check=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Uncertain scheduler response; inspect receipt before any action')
    receipt.update(status='submitted', job_id=job)
    (out/'submission.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(job_id=job, output=str(out))), flush=True)


if __name__ == '__main__':
    main()
