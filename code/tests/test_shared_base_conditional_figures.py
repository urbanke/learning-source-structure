"""Scientific publication barriers for the optional time16 figure points."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_base_conditional_figures import SOURCE,DEPTHS,read_panels,receipt_base


def fixture_summary():
    tf=json.loads((SOURCE/'summary.json').read_text())
    checks=[dict(name=f'{rule}:{name}',passed=True) for rule in ['coarse','refined']
        for name in ['all_basic_checks','all_scalar_mixing','all_jensen_local_checks','cohort_numerical_budget']]
    checks.extend(dict(name=name,passed=True) for name in ['all_case_refinement_checks',
        'aggregate_refinement_resolution','aggregate_refinement_stochastic_agreement'])
    summary=dict(schema='shared-base-curve-paired-summary-v1',time=16,
        prior=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation'),
        every_declared_sequence_retained=True,reference_validation_passed=True,
        rows=[],validations=[],provenance=dict(cohorts=[]))
    for depth in DEPTHS:
        original=next(c for c in tf['provenance']['cohorts'] if c['depth']==depth)
        summary['provenance']['cohorts'].append(dict(
            **{key:original[key] for key in ['alpha','depth','n_sequences','cohort_sha256','protocol_sha256']},
            audit_sha256='0'*64))
        # Synthetic reference values exercise the guards; they are never a
        # scientific estimate or written into the committed figure outputs.
        bayes_mean=.6+.25*depth
        validation=dict(schema='shared-base-cohort-precision-v1',complete=True,passed=True,
            n_sequences=256,expected_ids=list(range(256)),checks=deepcopy(checks),
            rules=[dict(rule=name,n_sequences=256,preliminary_passed=True,
                numerical_error_allowance_bits=.001,numerical_budget_bits=.002,
                candidate_mean_bayes_regret_bits=bayes_mean,observed_sequence_se_bits=.02)
                for name in ['coarse','refined']],
            refinement=dict(per_case=[dict(case_id=i,passed=True) for i in range(256)]),
            reported_mean_bayes_regret_bits=bayes_mean)
        summary['validations'].append(dict(depth=depth,orders=[16,32],cohort_validation=validation,
            status_counts=dict(missing=0,running=0,invalid=0,needs_more_draws=0,ready=512)))
        for model in ['d64','d64_seed1']:
            row=next(r for r in tf['rows'] if r['depth']==depth and r['time']==16 and r['model']==model)
            summary['rows'].append(dict(alpha=1.,depth=depth,time=16,model=model,n_sequences=256,
                bayes_regret=dict(n=256,mean=bayes_mean,standard_error=.02),
                transformer_regret={key:row[key] for key in ['n','mean','standard_error']},
                transformer_minus_bayes=dict(n=256,mean=row['mean']-bayes_mean,standard_error=.01)))
    return summary


def write_summary(tmp_path,summary):
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(summary))
    return path


def test_no_optional_summary_preserves_deterministic_values():
    files,panels=read_panels()
    assert len(files)==4
    receipt=receipt_base(files,panels)
    assert all(panel['bayes']['times']==[1,4] for panel in panels)
    assert receipt['display']['deterministic_bayes_times']==[1,4]
    assert receipt['display']['mcmc_bayes_times']==[]
    assert receipt['display']['bayes_pending_times']==[16,64,256,1024]


def test_validated_summary_adds_only_time16_and_keeps_earlier_values(tmp_path):
    _,before=read_panels()
    files,panels=read_panels(write_summary(tmp_path,fixture_summary()))
    for old,new in zip(before,panels):
        assert new['bayes']['times']==[1,4,16]
        assert new['bayes']['mean'][:2]==old['bayes']['mean']
        assert new['bayes']['se'][:2]==old['bayes']['se']
        assert new['transformers']==old['transformers']
    display=receipt_base(files,panels)['display']
    assert display['deterministic_bayes_times']==[1,4]
    assert display['mcmc_bayes_times']==[16]
    assert display['bayes_pending_times']==[64,256,1024]
    assert display['bayes_connecting_lines'] is False
    assert display['bayes_extrapolation'] is False


@pytest.mark.parametrize('failure',[
    'flag','depth_missing','cohort_changed','case_missing','failed_check',
    'missing_check','budget_exceeded','failed_refinement','one_order',
    'unresolved_case','audit_mean_changed','transformer_changed','wrong_prior',
])
def test_incomplete_or_inconsistent_summary_cannot_reach_figure(tmp_path,failure):
    summary=fixture_summary()
    validation=summary['validations'][0]
    aggregate=validation['cohort_validation']
    if failure=='flag':summary['reference_validation_passed']=False
    elif failure=='depth_missing':summary['rows']=summary['rows'][:-2]
    elif failure=='cohort_changed':summary['provenance']['cohorts'][0]['cohort_sha256']='f'*64
    elif failure=='case_missing':aggregate['expected_ids'].pop()
    elif failure=='failed_check':aggregate['checks'][0]['passed']=False
    elif failure=='missing_check':aggregate['checks'].pop()
    elif failure=='budget_exceeded':aggregate['rules'][1]['numerical_error_allowance_bits']=.01
    elif failure=='failed_refinement':aggregate['refinement']['per_case'][20]['passed']=False
    elif failure=='one_order':validation['orders']=[32]
    elif failure=='unresolved_case':validation['status_counts']['needs_more_draws']=1
    elif failure=='audit_mean_changed':aggregate['reported_mean_bayes_regret_bits']+=.1
    elif failure=='transformer_changed':summary['rows'][0]['transformer_regret']['mean']+=.1
    elif failure=='wrong_prior':summary['prior']['alpha']=[1,1]
    with pytest.raises(ValueError):read_panels(write_summary(tmp_path,summary))
