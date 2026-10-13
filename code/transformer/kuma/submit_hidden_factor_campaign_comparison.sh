#!/bin/bash
# Login side: standard-library argument/path checks and sbatch only.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 - "$repo_root" "$@" <<'PY'
import argparse, os, shlex, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
p=argparse.ArgumentParser(description='Evaluate both campaign seeds on the reused 3840 cases; print only unless --submit.')
p.add_argument('--arm',choices=['continue_large','xlarge'],required=True)
p.add_argument('--step',type=int,required=True)
p.add_argument('--out',type=Path,default=root/'output/hidden_factor_campaign_comparison_2026-09-19')
p.add_argument('--campaign-root',type=Path,default=root/'output/hidden_factor_campaign_2026-09-18')
p.add_argument('--baseline-root',type=Path,default=root/'output/hidden_factor_2026-09-16/2x8')
p.add_argument('--scaling-root',type=Path,default=root/'output/hidden_factor_scaling_2026-09-16')
p.add_argument('--comparison-root',type=Path,default=root/'output/hidden_factor_confirmation_2026-09-18')
p.add_argument('--plan',type=Path,default=root/'code/transformer/configs/hidden_factor_campaign_plan_2026-09-18.json')
p.add_argument('--after-job'); p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--submit',action='store_true'); a=p.parse_args(sys.argv[2:])
if a.step not in ({'continue_large':(400000,), 'xlarge':(200000,400000)}[a.arm]):
    p.error('select continue_large at 400000 or xlarge at 200000/400000')
if a.batch_size<1: p.error('batch size must be positive')
if a.after_job and not a.after_job.isdigit(): p.error('dependency must be a numeric job ID')
for name in ('out','campaign_root','baseline_root','scaling_root','comparison_root','plan'):
    value=getattr(a,name)
    if any(c in str(value) for c in '\n\r\t'): p.error('paths cannot contain control characters')
    setattr(a,name,value.resolve())
out=a.out/a.arm/f'step{a.step}'
protected=[a.campaign_root,a.baseline_root,a.scaling_root,a.comparison_root,*[root/n for n in ('code','results','notes','figures','.git')]]
if a.out==root or a.out in root.parents or any(a.out==q or a.out in q.parents or q in a.out.parents for q in protected):
    p.error('output must be separate from original inputs and source files')
cmd=['sbatch','--parsable','--chdir='+str(root),'--job-name=hf-campaign-compare',
     '--output='+str(out/'logs/comparison-%j.out')]
if os.environ.get('HIDDEN_FACTOR_ACCOUNT'): cmd+=['--account='+os.environ['HIDDEN_FACTOR_ACCOUNT']]
if a.after_job: cmd+=['--dependency=afterok:'+a.after_job,'--kill-on-invalid-dep=yes']
cmd += [str(root/'code/transformer/kuma/hidden_factor_campaign_comparison.sbatch'),
        str(out),str(a.plan),str(a.campaign_root),str(a.baseline_root),str(a.scaling_root),
        str(a.comparison_root),a.arm,str(a.step),str(a.batch_size)]
print(shlex.join(cmd),flush=True)
if a.submit:
    # Existence checks only. Immutable identities and checksums are checked in sbatch.
    for seed in (0,1):
        for f in (a.campaign_root/f'train/{a.arm}/seed{seed}/model_{a.step}.pt',
                  a.campaign_root/f'eval/{a.arm}/seed{seed}/step{a.step}/summary.json',
                  a.comparison_root/f'models/large_seed{seed}.npz',
                  a.comparison_root/f'models/large_seed{seed}.json'):
            if not a.after_job and not f.is_file(): p.error('missing saved comparison input: '+str(f))
    if not a.plan.is_file(): p.error('campaign plan is missing')
    (out/'logs').mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,check=True,cwd=root)
PY
