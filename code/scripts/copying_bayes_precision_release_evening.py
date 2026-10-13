"""Independent full-cohort release with the two bounded evening repair layers."""
from pathlib import Path
import argparse,json,os
import copying_bayes_precision_release_timeouts as prior
import copying_bayes_precision_evening as evening
release=prior.release;b=release.b;r=release.r
PRIOR_CODES=prior.codes
ORIGINAL_READER=release.read_reference

def codes():return {**PRIOR_CODES(),**evening.codes(),str(Path(__file__).relative_to(b.ROOT)):b.sha(__file__)}

def layers(folders,original,parent):
    result=[];indices=set()
    for folder in folders:
        layer,_=evening.load(folder);request=b.read(layer['request']['path'])
        b.require(Path(layer['parent']).resolve()==parent.resolve() and layer['parent_plan']['sha256']==b.sha(parent/'plan.json') and layer['runtime']==original['runtime'],'Evening layer has another parent')
        ids=set(request['original_indices']);b.require(not indices.intersection(ids),'Evening layers overlap');indices|=ids;result.append((folder,layer))
    b.require(len(result)==2 and sorted(len(b.read(p['request']['path'])['original_indices']) for _,p in result)==[19,67] and len(indices)==86,'Need exactly both frozen evening scopes')
    return result

def prepare(out,parent,recovered,banks,extra,timeouts,folders):
    b.require(not (out/'plan.json').exists(),'Release already prepared')
    for source in (parent,recovered,banks,extra,timeouts,*folders):
        a,z=out.resolve(),source.resolve();b.require(a!=z and a not in z.parents and z not in a.parents,'Release overlaps source')
    original,_=release.precision.load(parent);loaded=layers(folders,original,parent)
    previous=prior.codes;prior.codes=codes
    try:prior.prepare(out/'base_release_plan',parent,recovered,banks,extra,timeouts)
    finally:prior.codes=previous
    p=b.read(out/'base_release_plan/plan.json');p.pop('plan_id');p['evening_layers']=[str(f.resolve()) for f,_ in loaded]
    p['evening_reference_ids']=sorted(s['task']['id'] for _,layer in loaded for s in layer['tasks'])
    for i,(folder,_) in enumerate(loaded):p['input_plans'][f'evening{i}']=r.bind(folder/'plan.json')
    p['plan_id']=b.fp(p);b.write(out/'plan.json',p,immutable=True)
    print(json.dumps(dict(event='evening_release_prepared',references=len(p['expected_reference_ids']),evening_replacements=len(p['evening_reference_ids']))),flush=True)

def merge(out):
    previous=release.codes;release.codes=codes
    try:
        p,original,replacement,additional,bank_plan,data=release.load(out)
        parent=Path(p['parent']);timeout=prior.verify_layer(Path(p['timeout_layer']),original,parent)
        loaded=layers([Path(x) for x in p['evening_layers']],original,parent)
        b.require(sorted(s['task']['id'] for _,layer in loaded for s in layer['tasks'])==p['evening_reference_ids'],'Evening replacement inventory changed')
        selection={}
        for folder,layer in [(Path(p['timeout_layer']),timeout),*loaded]:
            for spec in layer['tasks']:
                key=spec['task']['id'];b.require(key not in selection,'Duplicate repair');selection[key]=(folder,layer,spec)
        seen=[]
        def reader(folder,spec,plan,bindings,retained=None):
            if spec and spec['task']['id'] in selection:
                key=spec['task']['id'];f,l,s=selection[key]
                b.require(Path(folder).resolve()==parent.resolve() and plan['plan_id']==original['plan_id'] and spec==original['tasks'][s['original_index']],'Unexpected repair parent')
                seen.append(key);return ORIGINAL_READER(f,s,l,bindings)
            return ORIGINAL_READER(folder,spec,plan,bindings,retained)
        release.read_reference=reader
        try:code=release.merge(out)
        finally:release.read_reference=ORIGINAL_READER
        if code==0 and not (len(seen)==len(selection) and set(seen)==set(selection)):
            summary=b.read(out/'summary.json');summary.update(population_reference_approved=False,aggregate_precision_passed=False,publication_status='evening_selection_failure');b.write(out/'summary.json',summary);raise ValueError('Incomplete selected-repair coverage')
        summary=b.read(out/'summary.json');summary['timeout_replacement_reference_count']=len(timeout['tasks']);summary['evening_replacement_reference_count']=len(p['evening_reference_ids'])
        if 'replacement_reference_count' in summary:summary['replacement_reference_count']+=len(selection)
        b.write(out/'summary.json',summary)
        b.write(out/'evening_selection.json',dict(schema='copying-evening-release-selection-v1',complete=code==0,expected_repair_ids=sorted(selection),selected_repair_ids=seen,evening_plans=[r.bind(f/'plan.json') for f,_ in loaded],timeout_plan=r.bind(Path(p['timeout_layer'])/'plan.json'),release_summary=r.bind(out/'summary.json'),sampling_performed=False),immutable=True)
        return code
    finally:release.codes=previous

def main():
    print('Copying independent full-cohort release with bounded evening repairs.',flush=True);b.require(os.environ.get('SLURM_JOB_ID'),'Run on a compute allocation')
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('prepare','merge'));a.add_argument('--out',type=Path,required=True)
    for n in ('parent','recovery','banks','extra','timeouts'):a.add_argument('--'+n,type=Path)
    a.add_argument('--evening',type=Path,nargs=2);v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.parent,v.recovery,v.banks,v.extra,v.timeouts,v.evening);return 0
    return merge(v.out)
if __name__=='__main__':raise SystemExit(main())
