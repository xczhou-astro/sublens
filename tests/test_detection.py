"""Fit-only decisions must not conflate stability, fit failure and nondetection."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from sublens import summarize_subhalo


def candidate(rank=1, chi=30., gain=10., **extra):
    return dict(rank=rank, position_arcsec=[.3,-.4], strength_arcsec=.05,
                delta_chi2=chi, delta_objective=gain, **extra)


def test_selects_passing_candidate_even_when_global_objective_best_fails_chi2():
    data=dict(lens_id='lens',candidates=[candidate(1,20.,100.),candidate(2,25.,50.)])
    previous=copy.deepcopy(data)
    result=summarize_subhalo(data)
    assert result['detected'] and result['candidate_rank']==2
    assert result['position_arcsec']==[.3,-.4] and result['strength_arcsec']==.05
    assert result['rule']['min_delta_chi2']==25.
    assert not result['rule']['stability_required'] and not result['rule']['significance_calibrated']
    assert data==previous


@pytest.mark.parametrize('rows',[[],[candidate(chi=24.99)],[candidate(gain=0.)],[candidate(gain=-1.)]])
def test_valid_nondetection_has_no_reported_halo_parameters(rows):
    result=summarize_subhalo(dict(candidates=rows))
    assert result['detected'] is False
    assert result['position_arcsec'] is None and result['strength_arcsec'] is None
    json.dumps(result,allow_nan=False)


def test_diagnostic_flags_do_not_veto_user_selected_fit_improvement_rule():
    row=candidate(converged=False,strength_at_cap=True,passes_all_descriptive_stability_checks=False)
    assert summarize_subhalo(dict(candidates=[row]))['detected'] is True
    assert summarize_subhalo(dict(candidates=[row]),min_delta_chi2=31.)['detected'] is False


@pytest.mark.parametrize('bad',[0.,-1.,np.nan,np.inf,True])
def test_invalid_threshold_is_an_error(bad):
    with pytest.raises(ValueError):summarize_subhalo(dict(candidates=[]),min_delta_chi2=bad)


@pytest.mark.parametrize('data',[
    dict(status='failed',candidates=[]),dict(status='missing_input',candidates=[]),{},
    dict(candidates=[{**candidate(),'strength_arcsec':np.nan}]),
    dict(candidates=[{**candidate(),'position_arcsec':[1.,np.inf]}]),
    dict(candidates=[{**candidate(),'strength_arcsec':-1.}]),
    dict(candidates=[{**candidate(),'strength_arcsec':0.}]),
])
def test_invalid_or_failed_fit_is_never_classified_as_absent(data):
    with pytest.raises(ValueError):summarize_subhalo(data)


def test_saved_metric_schemas_and_result_object_use_same_rule():
    canonical=dict(lens_id='lens',candidates=[candidate()])
    full=dict(identifier='lens',candidates=[dict(rank=1,position=[.3,-.4],strength=.05,
                                               delta_chi2=30.,penalized_gain=10.)])
    scan=dict(identifier='lens',status='completed',candidates=[dict(rank=1,position_xy_arcsec=[.3,-.4],
        strength_arcsec=.05,delta_chi2=30.,penalized_gain=10.)])
    obj=SimpleNamespace(to_dict=lambda:canonical)
    assert summarize_subhalo(canonical)==summarize_subhalo(full)==summarize_subhalo(scan)==summarize_subhalo(obj)


def test_compact_catalogue_uses_nominal_full_fit_and_preserves_pending(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sublens_testing'))
    import euclid_dr1_candidate_followup as followup
    monkeypatch.setattr(followup,'OUT',tmp_path)
    monkeypatch.setattr(followup,'ROOT',tmp_path)
    base=tmp_path/'lens/full_sie';base.mkdir(parents=True)
    data=dict(candidates=[candidate(chi=30.)])
    (base/'metrics.json').write_text(json.dumps(data))
    control=tmp_path/'lens/mesh800_seed33';control.mkdir()
    (control/'metrics.json').write_text(json.dumps(dict(candidates=[candidate(chi=0.)])))
    failed=tmp_path/'failed';failed.mkdir()
    (failed/'base_failure.json').write_text(json.dumps(dict(status='failed')))
    before=(base/'metrics.json').read_bytes()
    rows=followup.write_simple_results(['lens','pending','failed'])
    assert rows[0]['detected'] is True
    assert rows[1]['detected'] is None and rows[1]['status']=='pending_full_refinement'
    assert rows[2]['detected'] is None and rows[2]['status']=='failed_full_refinement'
    assert (base/'metrics.json').read_bytes()==before
    with (tmp_path/'fit_improvement/subhalo_results.csv').open() as handle:
        assert len(list(csv.DictReader(handle)))==3


def test_batch_catalogue_prefers_full_refinement_and_keeps_failures_unassessed(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sublens_testing'))
    import euclid_dr1_grade_a_scan as scan
    monkeypatch.setattr(scan,'OUT',tmp_path)
    folder=tmp_path/'candidate_followup/lens/full_sie';folder.mkdir(parents=True)
    (folder/'metrics.json').write_text(json.dumps(dict(candidates=[candidate(chi=30.)])))
    rows=scan.write_compact_results([
        dict(identifier='lens',status='completed',candidates=[candidate(chi=0.)]),
        dict(identifier='failed',status='failed'),
        dict(identifier='empty',status='completed',candidates=[]),
    ])
    assert rows[0]['detected'] is True and rows[0]['refinement_stage']=='full_sie'
    assert rows[1]['detected'] is None
    assert rows[2]['detected'] is False and rows[2]['refinement_stage']=='batch_joint'
