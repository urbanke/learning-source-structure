"""Independent band inversion, pairing, censoring and numerical edge checks."""
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import product_context_statistics as stats


def threshold(result, condition='one', index=0):
    return result['conditions'][condition]['thresholds'][index]


def test_first_crossing_is_not_last_crossing_or_monotone_envelope():
    times=[1,3,8,17]
    assert stats.first_tested_context([.4,.1,.3,.05],times,.2)==3
    assert stats.first_tested_context([.4,.3,.25,.21],times,.2) is None
    assert stats.first_tested_context([.2,.3,.1,.4],times,.2)==1


def test_band_inversion_retains_holes_and_stops_after_forced_crossing():
    result=stats.invert_first_crossing_band([.4,.1,.4,.1,.05],[.5,.3,.5,.15,.1],
                                           [1,4,8,16,32],.2)
    assert result['tested_contexts']==[4,16]
    assert result['unreached_possible'] is False
    assert result['minimum_tested_context']==4 and result['maximum_tested_context']==16


def test_inverted_set_matches_exhaustive_nonmonotone_curves():
    times=[1,4,9]
    # Endpoints and target represent every crossing/noncrossing possibility.
    for lower,upper in [([.1,.3,.1],[.3,.4,.15]),([.3,.1,.1],[.5,.4,.4]),
                        ([.3,.4,.5],[.4,.5,.6]),([.1,.05,.3],[.1,.2,.4])]:
        possible=set()
        choices=[[lo,hi]+([.2] if lo<=.2<=hi else []) for lo,hi in zip(lower,upper)]
        for curve in product(*choices):
            possible.add(stats.first_tested_context(curve,times,.2))
        actual=stats.invert_first_crossing_band(lower,upper,times,.2)
        represented=set(actual['tested_contexts'])
        if actual['unreached_possible']:represented.add(None)
        assert represented==possible


def test_nonattainment_prevents_finite_only_difference_or_ratio_intervals():
    b=stats.invert_first_crossing_band([.1,.05],[.3,.25],[1,4],.2)
    t=stats.invert_first_crossing_band([.1,.01],[.15,.1],[1,4],.2)
    result=stats.compatible_comparison(b,t)
    assert result['difference_interval'] is None and result['ratio_interval'] is None
    assert result['possible_attainment_outcomes']==['both_reached','bayes_unreached_only']
    never=stats.invert_first_crossing_band([.3,.4],[.4,.5],[1,4],.2)
    assert never['tested_contexts']==[] and never['unreached_possible']
    assert stats.compatible_comparison(never,never)['possible_attainment_outcomes']==['both_unreached']


def test_joint_finite_ranges_are_transformer_minus_and_divided_by_bayes():
    b=dict(tested_contexts=[2,8],unreached_possible=False)
    t=dict(tested_contexts=[4,16],unreached_possible=False)
    result=stats.compatible_comparison(b,t)
    assert result['difference_interval']==[-4,14]
    assert result['ratio_interval']==[.5,8.]


def test_exact_constant_coordinates_keep_exact_threshold_boundary():
    values=np.empty((13,2,2))
    values[:,0,0]=.1;values[:,1,0]=.05
    values[:,0,1]=np.nextafter(.1,np.inf);values[:,1,1]=.2
    result=stats.analyze_context_requirements({'one':values},[1,8],[.1],bootstrap_replicates=64)
    row=threshold(result)
    assert result['simultaneous_critical_value']==0
    assert row['bayes']['point_context']==1
    assert row['transformer']['point_unreached']
    assert row['transformer']['bootstrap']['unreached_fraction']==1
    assert row['comparison']['difference_interval'] is None
    assert all(all(r) for r in result['conditions']['one']['zero_standard_error'])
    json.dumps(result,allow_nan=False)


def test_pairing_keeps_identical_predictor_crossings_identical():
    rng=np.random.default_rng(29)
    risks=rng.uniform(.04,.4,size=(19,4))
    values=np.repeat(risks[:,:,None],2,axis=2)
    result=stats.analyze_context_requirements({'one':values},[1,4,16,64],[.18,.25],
                                             bootstrap_replicates=256,seed=771)
    for row in result['conditions']['one']['thresholds']:
        assert row['bayes']['bootstrap']==row['transformer']['bootstrap']
        joint=row['comparison']['bootstrap_attainment']
        assert joint['bayes_unreached_only']['count']==0
        assert joint['transformer_unreached_only']['count']==0
        assert sum(v['count'] for v in joint.values())==256


def test_global_critical_value_matches_independent_direct_bootstrap():
    values={
        'a':np.array([[[.7,.8],[.3,.4]],[[.2,.5],[.1,.2]],[[.4,.1],[.2,.3]]]),
        'b':np.array([[[.9,.4],[.1,.3]],[[.3,.7],[.2,.4]],[[.1,.2],[.3,.1]],[[.6,.8],[.4,.2]]])}
    repetitions=128;seed=81
    actual=stats.analyze_context_requirements(values,[1,8],[.2],
        bootstrap_replicates=repetitions,seed=seed,batch_size=17)
    maxima=np.zeros(repetitions)
    for name,stream in zip(sorted(values),np.random.SeedSequence(seed).spawn(2)):
        x=values[name];n=len(x)
        rng=np.random.Generator(np.random.PCG64(stream))
        mean=x.mean(0);se=x.std(0,ddof=1)/np.sqrt(n)
        for r in range(repetitions):
            count=rng.multinomial(n,np.full(n,1/n))
            direct=np.sum(x*count[:,None,None],axis=0)/n
            maxima[r]=max(maxima[r],np.max(np.abs((direct-mean)/se)))
    expected=np.quantile(maxima,.95,method='higher')
    assert actual['simultaneous_critical_value']==pytest.approx(expected,rel=1e-13,abs=1e-13)


def test_tiny_nonzero_variance_is_not_classified_as_deterministic():
    base=np.arange(1.,7.)[:,None,None]
    values=base*np.array([[[1.,2.],[.5,1.]]])*1e-200
    result=stats.analyze_context_requirements({'one':values},[1,8],[2e-200],
                                             bootstrap_replicates=64)
    case=result['conditions']['one']
    assert np.all(np.asarray(case['standard_error_bits'])>0)
    assert not np.asarray(case['zero_standard_error']).any()
    assert np.isfinite(result['simultaneous_critical_value'])
    rescaled=stats.analyze_context_requirements({'one':values/1e-200},[1,8],[2.],
                                               bootstrap_replicates=64)
    assert result['simultaneous_critical_value']==pytest.approx(rescaled['simultaneous_critical_value'])


def test_condition_order_reproducibility_variable_grids_and_no_input_mutation():
    rng=np.random.default_rng(31)
    a=rng.uniform(0,.5,size=(7,3,2));b=rng.uniform(0,.5,size=(9,2,2))
    before_a=a.copy();before_b=b.copy();a.flags.writeable=False;b.flags.writeable=False
    grids={'a':[1,4,16],'b':[2,32]}
    first=stats.analyze_context_requirements({'b':b,'a':a},grids,bootstrap_replicates=32,seed=12)
    second=stats.analyze_context_requirements({'a':a,'b':b},grids,bootstrap_replicates=32,seed=12)
    assert first==second
    np.testing.assert_array_equal(a,before_a);np.testing.assert_array_equal(b,before_b)
    json.dumps(first,allow_nan=False)


@pytest.mark.parametrize('times',[[0,1],[1,1],[4,1],[1.,2.],[],[True,False]])
def test_invalid_test_grids_rejected(times):
    with pytest.raises(ValueError):stats.first_tested_context(np.zeros(len(times)),times,.1)


@pytest.mark.parametrize('kind',['nan','negative','one_sequence','shape','target','unmatched_grid'])
def test_invalid_risks_or_settings_rejected(kind):
    x=np.ones((3,2,2))*.1;times=[1,4];targets=[.1]
    if kind=='nan':x[0,0,0]=np.nan
    if kind=='negative':x[0,0,0]=-.1
    if kind=='one_sequence':x=x[:1]
    if kind=='shape':x=x[:,:,:1]
    if kind=='target':targets=[.1,.1]
    if kind=='unmatched_grid':times={'wrong':[1,4]}
    with pytest.raises(ValueError):
        stats.analyze_context_requirements({'one':x},times,targets,bootstrap_replicates=32)


def test_small_negative_roundoff_preserved_but_never_widens_crossing_tolerance():
    x=np.full((4,2,2),.2);x[:,0,0]=-1e-14
    result=stats.analyze_context_requirements({'one':x},[1,4],[0.],bootstrap_replicates=32)
    assert result['conditions']['one']['mean_risk_bits'][0][0]==-1e-14
    assert threshold(result)['bayes']['point_context']==1
    assert threshold(result)['transformer']['point_unreached']
