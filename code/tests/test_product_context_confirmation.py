"""Checks for independent data and numerical settings in the confirmation run."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import product_context_confirmation as experiment


def test_generator_seeds_do_not_repeat_development_or_other_conditions():
    old=dict(master_seed=20260916,depth_pairs=experiment.PAIRS,n_sequences_per_pair=256)
    new=dict(old,master_seed=2026091704,n_sequences_per_pair=1024)
    audit=experiment.seed_audit(new,old)
    assert audit['confirmation_seeds']==4096
    assert audit['disjoint_from_development']
    with pytest.raises(ValueError,match='reuses a development'):
        experiment.seed_audit(dict(new,master_seed=old['master_seed']),old)
    with patch.object(experiment.previous,'seed_of',return_value=7):
        with pytest.raises(ValueError,match='Duplicate'):
            experiment.seed_audit(new,old)


def test_design_does_not_read_transformer_results():
    times=[2**j for j in range(11)]
    data=dict(cases=[dict(depth_pair=p,times=times,
                         informed_bayes=dict(mean=(4/np.sqrt(times)).tolist()))
                     for p in experiment.PAIRS])
    expected=experiment.dense_grid(data)
    for row in data['cases']:row['transformer']=dict(mean=['deliberately unavailable'])
    assert experiment.dense_grid(data)==expected


def test_perturbed_numerical_backend_is_rejected():
    with patch.dict(experiment.os.environ,{'PMM_PHI_BIAS':'0.01'}):
        with pytest.raises(ValueError,match='experimental'):
            experiment.backend_environment()


def test_invalid_probability_vectors_are_not_silently_renormalized():
    p=np.full((26,64),1/64)
    assert np.all(experiment.source_risk({'true_probs':p},p)==0)
    with pytest.raises(ValueError):experiment.source_risk({'true_probs':p},p*.99)
    q=p.copy();q[:,0]=0;q[:,1]+=1/64
    with pytest.raises(ValueError,match='support'):experiment.source_risk({'true_probs':p},q)
