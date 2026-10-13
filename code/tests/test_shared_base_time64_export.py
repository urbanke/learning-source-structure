"""Publication requires every fixed time64 sequence and numerical check."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/scripts'))
sys.path.insert(0, str(ROOT/'code/transformer'))
from export_shared_base_time64 import checked_cohort_gate
from validate_shared_base_mcmc import json_safe


@pytest.fixture
def complete_gate():
    path=ROOT/'results/transformer/shared_base_time64_complete_2026-09-17/summary.json'
    summary=json.loads(path.read_text())
    return deepcopy(next(c for c in summary['cohorts'] if c['depth']==4)['numerical_audit'])


def test_real_complete_cohort_is_publishable(complete_gate):
    checked_cohort_gate(complete_gate)


@pytest.mark.parametrize('change', ['missing_expected_id','wrong_count','incomplete'])
def test_incomplete_sequence_population_is_rejected(complete_gate,change):
    if change=='missing_expected_id': complete_gate['expected_ids'].pop()
    elif change=='wrong_count': complete_gate['n_sequences']=255
    else: complete_gate['complete']=False
    with pytest.raises(ValueError,match='Incomplete or failed whole-cohort gate'):
        checked_cohort_gate(complete_gate)


@pytest.mark.parametrize('change', ['failed','missing','duplicated'])
def test_all_eleven_named_checks_are_required(complete_gate,change):
    if change=='failed': complete_gate['checks'][0]['passed']=False
    elif change=='missing': complete_gate['checks'].pop()
    else: complete_gate['checks'][-1]=deepcopy(complete_gate['checks'][0])
    with pytest.raises(ValueError,match='Missing or failed numerical check'):
        checked_cohort_gate(complete_gate)


@pytest.mark.parametrize('change', ['failed','missing','duplicate_id'])
def test_each_fixed_sequence_requires_passing_refinement(complete_gate,change):
    cases=complete_gate['refinement']['per_case']
    if change=='failed': cases[0]['passed']=False
    elif change=='missing': cases.pop()
    else: cases[-1]['case_id']=cases[0]['case_id']
    with pytest.raises(ValueError,match='Refinement coverage incomplete'):
        checked_cohort_gate(complete_gate)


@pytest.mark.parametrize('change', ['count','preliminary','budget','order'])
def test_both_integration_rules_must_meet_their_precision_budgets(complete_gate,change):
    rule=complete_gate['rules'][1]
    if change=='count': rule['n_sequences']=255
    elif change=='preliminary': rule['preliminary_passed']=False
    elif change=='budget': rule['numerical_error_allowance_bits']=rule['numerical_budget_bits']+1e-5
    else: complete_gate['rules'].reverse()
    with pytest.raises(ValueError,match='Numerical precision budget failed'):
        checked_cohort_gate(complete_gate)


def test_numpy_boolean_conversion_preserves_passing_and_failing_decisions(complete_gate):
    complete_gate['passed']=np.bool_(True)
    complete_gate['refinement']['per_case'][0]['passed']=np.bool_(True)
    converted=json_safe(complete_gate)
    assert converted['passed'] is True
    assert converted['refinement']['per_case'][0]['passed'] is True
    checked_cohort_gate(converted)
    complete_gate['refinement']['per_case'][0]['passed']=np.bool_(False)
    converted=json_safe(complete_gate)
    assert converted['refinement']['per_case'][0]['passed'] is False
    with pytest.raises(ValueError,match='Refinement coverage incomplete'):
        checked_cohort_gate(converted)
    complete_gate['passed']=np.bool_(False)
    converted=json_safe(complete_gate)
    assert converted['passed'] is False
    with pytest.raises(ValueError,match='Incomplete or failed whole-cohort gate'):
        checked_cohort_gate(converted)
