"""Forward-model physics, PSF conventions, and truth-free analysis handoff."""
import json

import numpy as np
import pytest
import torch
from astropy.io import fits
from scipy.integrate import quad
from scipy.signal import fftconvolve

from sublens import (EPL, SIE, Sersic, NFW, TruncatedHalo, simulate_lens,
                     AnalysisConfig, load_lens_data, analyze_lens)
from sublens.grids import make_image_rays
from sublens.profiles.subhalo import nfw_subhalo_deflection


DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []) + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_nfw_matches_projected_density_and_deflection_jacobian(device):
    dtype = torch.float32 if device == 'mps' else torch.float64
    radius = [.001, .03, .049, .051, .2, .999, 1., 1.001, 2., 10.]
    coordinates = torch.tensor([[r*.2, 0.] for r in radius], device=device, dtype=dtype, requires_grad=True)
    alpha, kappa = nfw_subhalo_deflection(coordinates, (0., 0.), .3, .2)
    gx, = torch.autograd.grad(alpha[:, 0].sum(), coordinates, retain_graph=True)
    gy, = torch.autograd.grad(alpha[:, 1].sum(), coordinates)
    recovered = .5*(gx[:, 0]+gy[:, 1])
    tolerance = 8e-4 if device == 'mps' else 2e-8
    np.testing.assert_allclose(recovered.detach().cpu(), kappa.detach().cpu(), rtol=tolerance, atol=tolerance)
    # Independent LOS integral of rho(r)=rho_s/[r/r_s*(1+r/r_s)^2].
    expected = np.array([.6*quad(lambda z: 1/(np.hypot(r,z)*(1+np.hypot(r,z))**2), 0, np.inf,
                                  epsabs=1e-11, epsrel=1e-11)[0] for r in radius])
    np.testing.assert_allclose(kappa.detach().cpu(), expected, rtol=tolerance, atol=tolerance)
    at_scale = radius.index(1.)
    assert float(kappa[at_scale].detach()) == pytest.approx(.2, rel=tolerance)
    assert float(alpha[at_scale, 0].detach()) == pytest.approx(4*.3*.2*(1-np.log(2)), rel=tolerance)


@pytest.mark.parametrize('device', DEVICES)
def test_nfw_centre_and_scale_have_finite_autograd(device):
    dtype = torch.float32 if device == 'mps' else torch.float64
    center = torch.tensor([0., 0.], device=device, dtype=dtype, requires_grad=True)
    strength = torch.tensor(.3, device=device, dtype=dtype, requires_grad=True)
    coordinates = torch.tensor([[0.,0.],[.2,0.],[.1999999,.0001],[.001,.003]], device=device, dtype=dtype, requires_grad=True)
    alpha, kappa = nfw_subhalo_deflection(coordinates, center, strength, .2)
    (alpha.sum()+kappa.sum()).backward()
    assert torch.equal(alpha[0], torch.zeros_like(alpha[0]))
    for value in (alpha,kappa,coordinates.grad,center.grad,strength.grad):
        assert torch.isfinite(value).all()


def test_model_parameter_conventions_and_sie_limit():
    points = torch.tensor([[.4,.3],[-.2,.6]], dtype=torch.float64)
    sie = SIE(theta_E=.7,q=.8,phi=.3,gamma_ext=.03,phi_ext=.2)
    epl = EPL(theta_E=.7,q=.8,phi=.3,gamma=2.,gamma_ext=.03,phi_ext=.2)
    torch.testing.assert_close(sie.deflection(points), epl.deflection(points))
    assert sie.to_dict()['model']=='SIE' and sie.to_dict()['gamma']==2.
    with pytest.raises(TypeError):SIE(gamma=2.1)
    for invalid in (lambda:EPL(q=0),lambda:Sersic(r_eff=0),lambda:NFW(r_s=0),
                    lambda:TruncatedHalo(core=.3,truncation=.2)):
        with pytest.raises(ValueError):invalid()


@pytest.mark.parametrize('device', DEVICES)
def test_analytic_simulation_preserves_asymmetric_psf_foreground_and_seed(device):
    psf = np.array([[0.,.1,0.],[0.,.6,.2],[0.,.1,0.]])
    foreground = Sersic(center=(.05,-.02),i_eff=.2,r_eff=.3,n_sersic=1.,q=.8)
    options = dict(lens_mass=SIE(theta_E=.65), source=Sersic(center=(.03,.02)),
        subhalos=[TruncatedHalo(.04,center=(.65,.1)), NFW(.1,.15,center=(-.6,-.1))],
        lens_light=foreground, image_size=(20,24),pixel_scale=.12,oversample=2,
        psf=psf,noise=np.linspace(.01,.03,480).reshape(20,24),seed=37,device=device)
    simulated = simulate_lens(**options)
    xr,yr = make_image_rays((20,24),.12,2)
    coords = torch.tensor(np.column_stack((xr.ravel(),yr.ravel())),dtype=torch.float64)
    native = foreground.brightness(coords).reshape(20,24,-1).mean(-1).numpy()
    np.testing.assert_allclose(simulated.lens_light, fftconvolve(native,psf,mode='same'),rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(simulated.clean_image,simulated.lensed_source_image+simulated.lens_light)
    expected_noise = simulated.rms*np.random.default_rng(37).standard_normal((20,24))
    np.testing.assert_allclose(simulated.image-simulated.clean_image,expected_noise,atol=1e-15)
    assert [h['model'] for h in simulated.truth['subhalos']]==['TruncatedHalo','NFW']
    np.testing.assert_array_equal(simulated.psf,psf)


def test_simulation_exports_observations_only_and_truth_is_explicit(tmp_path):
    simulated = simulate_lens(SIE(theta_E=.65),Sersic(),TruncatedHalo(.04,center=(.65,.1)),
        image_size=20,pixel_scale=.12,psf=np.ones((1,1)),lens_light=Sersic(),device='cpu')
    path = simulated.to_fits(tmp_path/'observed.fits')
    with fits.open(path) as h:
        assert [a.name for a in h]==['PRIMARY','RMS','PSF','MASK','LENS_LIGHT']
        assert not any(key.startswith(('MACRO_','SUB_','LENS_')) for key in h[0].header)
        assert not h[0].header['LNSUB']
        assert all(a.verify_checksum()==1 for a in h)
        np.testing.assert_array_equal(h[0].data,simulated.image)
    simulated.save_truth(tmp_path/'truth.json')
    assert json.loads((tmp_path/'truth.json').read_text())['lens_mass']['model']=='SIE'
    with pytest.raises(FileExistsError):simulated.save_truth(tmp_path/'truth.json')
    with pytest.raises(OSError):simulated.to_fits(path)
    # Deliberately contaminate diagnostics; loading sees only selected arrays.
    simulated.truth['lens_mass']={'theta_E':999.}
    simulated.truth['source']='forbidden'
    simulated.truth['subhalos']='forbidden'
    data=load_lens_data(image=simulated.image,noise=simulated.rms,psf=simulated.psf,
        mask=simulated.mask,pixel_scale=simulated.pixel_scale,lens_light=simulated.lens_light)
    np.testing.assert_array_equal(data.image,simulated.image)
    assert not hasattr(data,'truth') and not hasattr(data,'macro_model')
    assert not hasattr(simulated,'analyze')


@pytest.mark.parametrize('device', DEVICES)
def test_one_call_simulation_runs_real_image_only_analysis(device,tmp_path):
    simulated = simulate_lens(SIE(theta_E=.65,q=.83),Sersic(center=(.02,-.01)),
        subhalos=TruncatedHalo(.03,center=(.65,.1)),image_size=(20,24),
        pixel_scale=.12,psf=np.ones((1,1)),oversample=1,device=device)
    config = AnalysisConfig(source_nodes=(35,),regularizations=(.002,),oversample=1,
        initializer_oversample=1,initializer_steps=2,macro_steps=2,pinn_steps=2,
        proposal_steps=2,joint_steps=2,joint_starts=1,full_refine_steps=2,
        full_refine_starts=1,adaptive_source=False,top_k=1)
    data=load_lens_data(simulated.to_fits(tmp_path/'observation.fits'))
    result = analyze_lens(data,config=config,device=device)
    assert result.config['analysis']['initializer']['origin']=='image_only_four_starts'
    assert result.config['analysis']['external_macro_input_accepted'] is False
    assert result.config['analysis']['external_source_input_accepted'] is False
    assert np.isfinite(result.baseline.image_model).all()
    result.save_json(tmp_path/'fit.json')
    assert 'truth' not in json.loads((tmp_path/'fit.json').read_text())
