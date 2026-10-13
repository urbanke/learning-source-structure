"""Preserve the independent release gate while adding a bound timeout layer."""
from pathlib import Path
import argparse,json,os
import copying_bayes_precision_release as release
import copying_bayes_precision_timeouts as timeout
b,r=release.b,release.r
ORIGINAL_CODES=release.codes
ORIGINAL_READER=release.read_reference


def codes():return {**ORIGINAL_CODES(),**timeout.codes(),str(Path(__file__).relative_to(b.ROOT)):b.sha(__file__)}


def verify_layer(folder,original,parent):
    layer,_=timeout.load(folder)
    b.require(Path(layer['parent']).resolve()==parent.resolve() and layer['parent_plan']['sha256']==b.sha(parent/'plan.json') and layer['runtime']==original['runtime'],'Timeout layer has another parent')
    candidates=b.read(layer['candidates']['path']);eligible={x['index'] for x in candidates}
    b.require(len(candidates)==len(eligible)==40,'Timeout scope differs')
    protected=set(original['shards'][322]['tasks'])|release.EXTRA_INDICES
    b.require(not eligible.intersection(protected),'Timeout layer overlaps early recovery')
    by_id={original['tasks'][i]['task']['id']:i for i in eligible}
    tasks={s['task']['id']:s for s in layer['tasks']}
    b.require(len(tasks)==len(layer['tasks']) and not set(tasks).intersection(layer['retained']) and set(tasks)|set(layer['retained'])==set(by_id),'Timeout coverage is incomplete or duplicate')
    for key,s in tasks.items():
        original_spec=original['tasks'][by_id[key]]
        b.require(s['original_index']==by_id[key] and s['task']==original_spec['task'] and s['target_draws']==original_spec['target_draws'],'Timeout recovery changed seed, source or target')
    b.require({s['index'] for s in layer['terminal_snapshots']}==eligible and all(s['terminal_attempt']==dict(index=s['index'],returncode=75) for s in layer['terminal_snapshots']),'Missing terminal-task evidence')
    for key,binding in layer['retained'].items():
        r.verify(binding);source=original['tasks'][by_id[key]];path=b.accuracy.paths(parent,source['task'])['report']
        b.require(Path(binding['path'])==path,'Retained another reference')
        report=b.read(path);b.require(report['task']==source and report['plan_id']==original['plan_id'] and report['complete'] and report['audit_passed'],'Retained failed reference')
    return layer


def prepare(out,parent,recovered,banks,extra,timeouts):
    b.require(not (out/'plan.json').exists(),'Release already prepared')
    for source in (parent,recovered,banks,extra,timeouts):
        a,z=out.resolve(),source.resolve();b.require(a!=z and a not in z.parents and z not in a.parents,'Release overlaps source')
    original,_=release.precision.load(parent);layer=verify_layer(timeouts,original,parent)
    previous=release.codes;release.codes=codes
    try:release.prepare(out/'base_release_plan',parent,recovered,banks,extra)
    finally:release.codes=previous
    p=b.read(out/'base_release_plan/plan.json');p.pop('plan_id')
    p.update(timeout_layer=str(timeouts.resolve()),timeout_reference_ids=sorted(s['task']['id'] for s in layer['tasks']))
    p['input_plans']['timeouts']=r.bind(timeouts/'plan.json');p['plan_id']=b.fp(p)
    b.write(out/'plan.json',p,immutable=True)
    print(json.dumps(dict(event='timeout_release_prepared',references=len(p['expected_reference_ids']),timeouts=len(p['timeout_reference_ids']))),flush=True)


def merge(out):
    previous_codes=release.codes;release.codes=codes
    try:
        p,original,replacement,additional,bank_plan,data=release.load(out)
        layer=verify_layer(Path(p['timeout_layer']),original,Path(p['parent']))
        specs={s['task']['id']:s for s in layer['tasks']}
        b.require(sorted(specs)==p['timeout_reference_ids'],'Timeout replacement inventory changed')
        seen=[]
        def reader(folder,spec,plan,bindings,retained=None):
            if spec and spec['task']['id'] in specs:
                key=spec['task']['id']
                b.require(Path(folder).resolve()==Path(p['parent']).resolve() and plan['plan_id']==original['plan_id'] and spec==original['tasks'][specs[key]['original_index']],'Timeout reference chosen from unexpected layer')
                seen.append(key)
                return ORIGINAL_READER(Path(p['timeout_layer']),specs[key],layer,bindings)
            return ORIGINAL_READER(folder,spec,plan,bindings,retained)
        release.read_reference=reader
        try:code=release.merge(out)
        finally:release.read_reference=ORIGINAL_READER
        if code==0 and not (len(seen)==len(specs) and set(seen)==set(specs)):
            summary=b.read(out/'summary.json');summary.update(population_reference_approved=False,aggregate_precision_passed=False,publication_status='timeout_selection_failure')
            b.write(out/'summary.json',summary);raise ValueError('Not all timeout references were validated')
        summary=b.read(out/'summary.json');summary['timeout_replacement_reference_count']=len(specs)
        if 'replacement_reference_count' in summary:summary['replacement_reference_count']+=len(specs)
        b.write(out/'summary.json',summary)
        receipt=dict(schema='copying-timeout-release-selection-receipt-v1',complete=code==0,
          expected_timeout_ids=sorted(specs),selected_timeout_ids=seen,timeout_plan=r.bind(Path(p['timeout_layer'])/'plan.json'),
          release_summary=r.bind(out/'summary.json'),sampling_performed=False)
        b.write(out/'timeout_selection.json',receipt,immutable=True)
        return code
    finally:release.codes=previous_codes


def main():
    print('Copying independent release with immutable terminal-timeout continuations.',flush=True)
    b.require(os.environ.get('SLURM_JOB_ID'),'Run release audit on a compute allocation')
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('prepare','merge'));a.add_argument('--out',type=Path,required=True)
    for n in ('parent','recovery','banks','extra','timeouts'):a.add_argument('--'+n,type=Path)
    v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.parent,v.recovery,v.banks,v.extra,v.timeouts);return 0
    return merge(v.out)
if __name__=='__main__':raise SystemExit(main())
