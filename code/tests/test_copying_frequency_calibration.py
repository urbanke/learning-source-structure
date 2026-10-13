"""Independent checks of stationary calibration and its uncertainty arithmetic."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import copying_frequency_calibration as calibration
from shared_base_frequency_reference import stationary_iid_joint


def test_stationary_two_state_closed_form():
    transitions = np.asarray([[[.8,.2],[.6,.4]], [[.9,.1],[.25,.75]]])
    nu, residual = calibration.stationary(transitions)
    assert np.allclose(nu, [[.75,.25], [.25/.35,.1/.35]], rtol=0, atol=2e-15)
    assert residual < 1e-14


def test_depth_zero_stationary_law_equals_original_q():
    q = np.arange(1, 65, dtype=float); q /= q.sum()
    nu, _ = calibration.stationary(np.broadcast_to(q, (2,64,64)).copy())
    assert np.allclose(nu, np.broadcast_to(q, (2,64)), rtol=0, atol=1e-15)


def test_stationary_solver_rejects_bad_rows_without_repair():
    with pytest.raises(ValueError, match='Invalid source matrices'):
        calibration.stationary(np.asarray([[[.8,.3],[.6,.4]]]))


def test_joint_banks_are_averaged_before_predictive_ratio():
    q = np.asarray([np.arange(1,65,dtype=float), np.arange(64,0,-1,dtype=float)])
    q /= q.sum(1,keepdims=True)
    nu = np.asarray([np.arange(1,65,dtype=float)**2, np.ones(64)])
    nu /= nu.sum(1,keepdims=True)
    table = calibration.lookup(calibration.joint_vector(q,nu))
    prefix = (7,7)
    p = calibration.prediction(prefix,table)
    den = stationary_iid_joint(prefix,q,nu)
    num = stationary_iid_joint(prefix+(7,),q,nu)
    assert abs(p[7]-num.mean()/den.mean()) < 1e-15
    assert abs(p[7]-np.mean(num/den)) > 1e-5
    assert abs(p.sum()-1) < 1e-12


def test_distinguished_first_repeat_uses_q_times_stationary():
    q = np.asarray([np.arange(1,65,dtype=float), np.ones(64)])
    q /= q.sum(1,keepdims=True)
    nu = q[:,::-1].copy()
    table = calibration.lookup(calibration.joint_vector(q,nu))
    p = calibration.prediction([11],table)
    assert abs(p[11]-np.sum(q*nu,axis=1).mean()) < 1e-15
    assert np.allclose(np.delete(p,11),(1-p[11])/63,rtol=0,atol=2e-16)


def test_leave_one_bank_out_mcse_matches_independent_linear_mean_formula():
    # For a linear quantity, delete-one jackknife SE equals ordinary bank-mean SE.
    bank_values = np.asarray([[1.,3.],[2.,5.],[4.,7.],[8.,9.]])
    loo = (bank_values.sum(0)-bank_values)/3
    scored_sequences = np.asarray([[1.,2.],[2.,4.],[3.,6.]])
    summary = calibration.summarize(scored_sequences,loo)
    assert np.allclose(summary['prior_bank_mcse'],bank_values.std(0,ddof=1)/2)
    assert np.allclose(summary['paired_sequence_se'],scored_sequences.std(0,ddof=1)/np.sqrt(3))


def test_cumulative_uncertainty_sums_inside_each_bank_first():
    bank_values = np.asarray([[1.,-1.,0.,0.],[-1.,1.,0.,0.],[2.,-2.,0.,0.],[-2.,2.,0.,0.]])
    loo = (bank_values.sum(0)-bank_values)/3
    result = calibration.summarize(np.zeros((3,4)),np.cumsum(loo,axis=1))
    assert result['prior_bank_mcse'][0] > 0
    assert result['prior_bank_mcse'][1:] == [0.,0.,0.]
