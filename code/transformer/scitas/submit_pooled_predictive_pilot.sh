#!/bin/bash
# Lightweight submission only; all input reads and numerical work use allocations.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 - "$repo_root" "$@" <<'PY'
import argparse, shlex, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
p=argparse.ArgumentParser(description='Three-family predictive-output smoke test; no population submission.')
p.add_argument('--raw',type=Path,default=Path.home()/'Projects/layered_architectures_multifamily/output/multifamily_2026-09-16')
p.add_argument('--dependencies',type=Path,default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
p.add_argument('--out',type=Path,default=root/'output/pooled_predictive_pilot_2026-09-18')
p.add_argument('--observed',type=int,choices=(16,64),default=16)
p.add_argument('--submit',action='store_true')
a=p.parse_args(sys.argv[2:])
raw,dep,out=(q.expanduser().resolve() for q in (a.raw,a.dependencies,a.out))
protected=[raw,dep,*[root/n for n in ('code','results','notes','figures','.git')]]
if out==root or out in root.parents or any(out==q or out in q.parents or q in out.parents for q in protected):
    p.error('output must be separate from frozen inputs and repository sources')
cmd=['sbatch','--parsable','--chdir='+str(root),'--job-name=pooled-predictive-smoke',
     '--array=0-2%3','--output='+str(out/'logs'/f't{a.observed}-%A_%a.out'),
     str(root/'code/transformer/scitas/pooled_predictive_pilot.sbatch'),
     str(raw),str(dep),str(out),str(a.observed)]
print(shlex.join(cmd),flush=True)
if a.submit:
    if not (raw/'heldout/manifest.json').is_file() or not (dep/'tables/anchors_prod/manifest.json').is_file():
        p.error('frozen pooled evaluation or sealed reference dependency is missing at the supplied path')
    (out/'logs').mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,check=True,cwd=root)
PY
