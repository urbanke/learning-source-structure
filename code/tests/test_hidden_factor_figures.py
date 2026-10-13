"""Pilot figures keep the evaluated sample count and paired comparisons explicit."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import hidden_factor_figures as figures


def stats(values,n=256):
    return {'mean':values,'se':[.1]*len(values),'n_seq':n}


def summary(partial=True,model=True,production=True):
    n=256 if partial else 4096
    predictors={name:{'next_kl_bits':stats([1.,.5,.1],n)} for name in
                ['bayes_hidden','bayes_encoding_given','flat_dirichlet']}
    paired={'bayes_hidden_minus_bayes_encoding_given':{'cumulative_loss_bits':stats([0.,2.,4.],n)}}
    if model:
        predictors['seed0']={'next_kl_bits':stats([1.1,.7,.2],n)}
        paired.update(seed0_minus_bayes_encoding_given={'cumulative_loss_bits':stats([0.,3.,7.],n)},
                      seed0_minus_bayes_hidden={'cumulative_loss_bits':stats([0.,1.,3.],n)})
    full=dict(id='full_prior',n_seq=n,status='complete',predictors=predictors,paired=paired)
    groups=[full]+[dict(full,id=f'entropy_{i}',n_seq=n//4,entropy_label=label)
                   for i,label in enumerate(['low / low','low / high','high / low','high / high'])]
    return dict(schema='hidden-factor-evaluation-v1',times=[1,16,1024],groups=groups,
                cohort={'identity':{'n_seq':4096,'stratum_rule':'source entropy medians'}},
                partial_evaluation=partial,evaluated_n_sequences=n,planned_n_sequences=4096,
                evaluation_complete=not partial,production_size=production,
                two_repeat_comparison_complete=False,bayes_only=not model,uncertainty='paired sequence SE')


def test_partial_production_uses_actual_sample_count_and_paired_gap(tmp_path,monkeypatch):
    data=summary();captured=[]
    # Inspect the same figures that the renderer saves, including the distinct paired gap.
    monkeypatch.setattr(figures.plt,'close',lambda fig:captured.append(fig))
    try:
        figures.render(data,tmp_path)
        overview,conditional=captured
        for fig in captured:
            assert any('Preliminary: first 256 of 4,096 test sequences' in t.get_text() for t in fig.texts)
        np.testing.assert_array_equal(overview.axes[1].lines[0].get_ydata(),[0.,2.,4.])
        assert len(overview.axes[1].lines)==2  # Bayesian uncertainty cost plus zero line.
        np.testing.assert_array_equal(overview.axes[2].lines[0].get_ydata(),[0.,1.,3.])
        assert len(conditional.axes)==4
        for name in ['uncertainty_cost','conditional_curves']:
            for extension in ['pdf','png']:assert (tmp_path/f'{name}.{extension}').stat().st_size>1000
    finally:
        for fig in captured:fig.clear()


def test_bayes_only_pilot_does_not_invent_transformer_curve(tmp_path,monkeypatch):
    captured=[];monkeypatch.setattr(figures.plt,'close',lambda fig:captured.append(fig))
    try:
        figures.render(summary(model=False),tmp_path)
        ax=captured[0].axes[2]
        assert any(t.get_text()=='Awaiting trained models' for t in ax.texts)
        assert len(ax.lines)==1  # Zero reference line only.
    finally:
        for fig in captured:fig.clear()


@pytest.mark.parametrize('changes',[
    {'partial_evaluation':False}, {'evaluated_n_sequences':4096},
    {'planned_n_sequences':256}, {'evaluation_complete':True},
    {'two_repeat_comparison_complete':True},
])
def test_pilot_cannot_claim_full_completion(changes):
    data=summary();data.update(changes)
    with pytest.raises(ValueError,match='inconsistent'):figures.evaluation_scope(data)


def test_legacy_full_summary_remains_readable():
    data=summary(partial=False)
    for key in ['partial_evaluation','evaluated_n_sequences','planned_n_sequences','evaluation_complete']:
        del data[key]
    scope=figures.evaluation_scope(data)
    assert scope['evaluation_complete'] and not scope['partial_evaluation']
    assert scope['label']=='4,096 independent test sequences'


def test_partial_receipt_and_smoke_guard(tmp_path,monkeypatch):
    path=tmp_path/'summary_prefix_000256.json';out=tmp_path/'figures'
    path.write_text(json.dumps(summary()))
    argv=['figures','--summary',str(path),'--out',str(out)]
    monkeypatch.setattr(sys,'argv',argv);figures.main()
    receipt=json.loads((out/'receipt.json').read_text())
    assert receipt['partial_evaluation'] and receipt['evaluated_n_sequences']==256
    assert receipt['planned_n_sequences']==4096 and not receipt['evaluation_complete']
    monkeypatch.setattr(sys,'argv',argv+['--check-only']);figures.main()
    path.write_text(json.dumps(summary(production=False)))
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='--allow-smoke'):figures.main()
    monkeypatch.setattr(sys,'argv',argv+['--allow-smoke']);figures.main()
    assert 'not scientific evidence' in figures.evaluation_scope(summary(production=False))['label']
