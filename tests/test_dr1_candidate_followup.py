"""Scientific contracts for the observational candidate follow-up experiment."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sublens_testing'))
from euclid_dr1_candidate_followup import FullMacroProblem,vector,macro_at,vary_observation
from sublens import LensingObservation,torch_backend
from sublens.inversion.source_plane import PixelSourceProblem


def observation():
    m=dict(theta_E=.7,q=.8,phi=.3,gamma=2.,center=[.02,-.01],gamma_ext=.03,phi_ext=.2)
    return LensingObservation.from_arrays(np.ones((16,18)),np.full((16,18),.1),np.array([[0.,.1,0.],[.1,.6,.1],[0.,.1,0.]]),
        pixel_scale=.15,oversample=2,macro_model=m)


def test_followup_has_exact_nested_null_and_correct_macro_mapping():
    obs=observation()
    p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=40)
    with torch_backend('cpu'):
        for slope in (False,True):
            model=FullMacroProblem(obs,p,slope)
            v=model.engine.tensor(vector(obs.macro_model,slope))
            a,b=model.rays(v)
            np.testing.assert_allclose(a,obs.bx,atol=1e-8)
            np.testing.assert_allclose(b,obs.by,atol=1e-8)
            hx,hy=model.rays(torch.cat((v,model.engine.tensor([.5,.3,0.]))),True)
            torch.testing.assert_close(a,hx,rtol=0,atol=0)
            torch.testing.assert_close(b,hy,rtol=0,atol=0)
            m=macro_at(vector(obs.macro_model,slope),slope)
            for k in ('theta_E','q','phi','gamma','center','gamma_ext','phi_ext'):
                np.testing.assert_allclose(m[k],obs.macro_model[k],atol=1e-12)


def test_free_epl_slope_gradient_matches_difference():
    obs=observation()
    p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=40)
    with torch_backend('cpu'):
        model=FullMacroProblem(obs,p,True)
        v=model.engine.tensor(vector(obs.macro_model,True)).requires_grad_(True)
        def loss(v):
            x,y=model.rays(v)
            return x.square().sum()+y.square().sum()
        g,=torch.autograd.grad(loss(v),v)
        d=1e-6;delta=torch.zeros_like(v);delta[7]=d
        expected=(loss(v.detach()+delta)-loss(v.detach()-delta))/(2*d)
        torch.testing.assert_close(g[7],expected,rtol=1e-6,atol=1e-7)


def test_foreground_perturbation_preserves_data_noise_mask():
    obs=observation();lens=np.full(obs.image.shape,2.)
    changed=vary_observation(obs,lens,foreground_scale=1.02)
    np.testing.assert_allclose(changed.image,obs.image-.04)
    np.testing.assert_array_equal(changed.rms,obs.rms)
    np.testing.assert_array_equal(changed.mask,obs.mask)
    p=vary_observation(obs,lens,psf_width=1.05)
    np.testing.assert_allclose(p.psf.sum(),1.,atol=1e-12)


@pytest.mark.parametrize('device',['cpu']+(['mps'] if torch.backends.mps.is_available() else []))
def test_null_rescue_accepts_device_resident_macro_start(device):
    obs=observation()
    p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=40,regularization=.02)
    with torch_backend(device):
        from euclid_dr1_candidate_followup import macro_bounds
        model=FullMacroProblem(obs,p)
        start=model.engine.tensor(vector(obs.macro_model))
        fitted=model.optimize(start,macro_bounds(obs.macro_model),False,maxiter=1)
        assert np.isfinite(float(fitted.source.objective))


def test_periodic_macro_angle_does_not_turn_into_a_bound_start():
    from euclid_dr1_candidate_followup import macro_bounds,bounded_start
    obs=observation();macro={**obs.macro_model,'phi':1.4}
    v=vector(macro);v[2]=1.7-np.pi
    b=macro_bounds(macro)
    prepared=bounded_start(v,b)
    assert abs(prepared[2]-1.7)<1e-12
    np.testing.assert_allclose(prepared[np.arange(7)!=2],v[np.arange(7)!=2])


def test_bound_limited_strong_peak_does_not_hide_viable_alternative():
    from euclid_dr1_candidate_followup import other_viable_ranks
    good=dict(rank=2,delta_chi2=50.,penalized_gain=49.,converged=True,
              macro_at_bound=False,position_at_bound=False,strength_at_cap=False)
    strongest={**good,'rank':1,'delta_chi2':100.,'strength_at_cap':True}
    record=dict(null_macro_at_bound=False,null=dict(converged=True),candidates=[strongest,good])
    assert other_viable_ranks(record,1)==[2]


@pytest.mark.parametrize('fails',[False,True])
def test_sequential_gate_never_skips_checks_for_passing_candidate(tmp_path,monkeypatch,fails):
    import json
    from types import SimpleNamespace
    import euclid_dr1_candidate_followup as module
    fit=SimpleNamespace(position=(0.,0.),strength=.1)
    base=SimpleNamespace(best_candidate=SimpleNamespace(fit=fit))
    names=['mesh600_seed11','mesh800_seed33','lambda_x10','free_epl_slope','psf_width_plus5pct']
    monkeypatch.setattr(module,'variant_options',lambda *args:[(name,{}) for name in names])
    monkeypatch.setattr(module,'current_execution',lambda:SimpleNamespace(device='cpu'))
    calls=[]
    def refit(*args,**kwargs):
        calls.append(1)
        candidate=dict(position=[0.,0.],strength=.2 if fails else .1,delta_chi2=30.,penalized_gain=30.,
            converged=True,null_converged=True,macro_at_bound=False,position_at_bound=False,strength_at_cap=False)
        fitted=SimpleNamespace(best_candidate=SimpleNamespace(fit=SimpleNamespace(delta_chi2=30.)))
        return fitted,dict(candidates=[candidate],null_macro_at_bound=False)
    def save(folder,fitted,record):
        folder.mkdir(parents=True)
        (folder/'metrics.json').write_text(json.dumps(record))
    monkeypatch.setattr(module,'refit',refit);monkeypatch.setattr(module,'save_fit',save)
    module.run_variants(base,tmp_path,None,'test',early_gate=True)
    assert len(calls)==(1 if fails else 5)
    assert (tmp_path/'stability_stopping.json').exists()==fails
