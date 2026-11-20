"""Normalization and PSF-integration checks for the image-only DR1 initializer."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.signal import fftconvolve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sublens_testing'))
from euclid_dr1_batch_macro import sie_alpha,ParametricInitializer
from sublens.profiles.macro import epl_deflection
from sublens.simulation.sources import sersic_profile
from sublens import LensingObservation


def test_fast_sie_preserves_epl_normalization_and_derivatives():
    coordinates=torch.tensor([[.3,.8],[-.5,.7],[.9,-.5]],dtype=torch.float64)
    for q in (.3,.7,.995):
        v=torch.tensor([1.3,q,.4,.02,-.03],dtype=torch.float64,requires_grad=True)
        fast=sie_alpha(coordinates,*v)
        expected=epl_deflection(coordinates,theta_E=v[0],q=v[1],phi=v[2],center=(v[3],v[4]))[0]
        np.testing.assert_allclose(fast.detach(),expected.detach(),rtol=1e-8,atol=1e-8)
        for i in range(2):
            a,=torch.autograd.grad(fast[:,i].sum(),v,retain_graph=True)
            b,=torch.autograd.grad(expected[:,i].sum(),v,retain_graph=True)
            np.testing.assert_allclose(a,b,rtol=2e-7,atol=1e-7)


def test_initializer_keeps_asymmetric_psf_orientation_and_pixel_integration():
    image=np.exp(-((np.arange(20)[:,None]-8)**2+(np.arange(24)[None,:]-13)**2)/30)
    psf=np.array([[.01,.02,.03],[.04,.65,.08],[.05,.07,.05]])
    obs=LensingObservation.from_arrays(image,np.full(image.shape,.08),psf,pixel_scale=.12,oversample=2,
        macro_model=dict(theta_E=.8,q=.75,phi=.2,gamma=2.,gamma_ext=.03,phi_ext=.4,center=[.01,-.02]))
    initializer=ParametricInitializer(obs)
    v=torch.tensor([np.log(.8),.75,.2,.05,-.1,.04,.03,.03,-.02,np.log(.2),.8,.3,1.1],dtype=torch.float64)
    model,amplitude=initializer.render(v)
    host=epl_deflection(initializer.coordinates,theta_E=v[0].exp(),q=v[1],phi=v[2],center=(.2*v[3],.2*v[4]))[0]
    x=initializer.coordinates[:,0]-.2*v[3];y=initializer.coordinates[:,1]-.2*v[4]
    beta=initializer.coordinates-host-torch.stack((.2*v[5]*x+.2*v[6]*y,.2*v[6]*x-.2*v[5]*y),dim=-1)
    source=sersic_profile(beta[:,0],beta[:,1],center=(v[7],v[8]),r_eff=v[9].exp(),q=v[10],phi=float(v[11]),n_sersic=v[12])
    native=source.reshape(initializer.shape).mean(-1).numpy()
    expected=float(amplitude)*fftconvolve(native,obs.psf,mode='same')
    np.testing.assert_allclose(model.detach(),expected,rtol=1e-8,atol=1e-10)


def test_comparison_handles_zero_sources_without_invalid_json():
    import json
    from euclid_dr1_fixed_source import source_scores
    y,x=np.indices((31,31));reference=np.exp(-((x-15)**2+(y-15)**2)/20.)
    empty=source_scores(reference,np.zeros_like(reference),.01)
    assert empty['cosine']==0. and empty['zero_reconstructed_source']
    unavailable=source_scores(np.zeros_like(reference),reference,.01)
    assert unavailable['cosine'] is None and not unavailable['comparison_defined']
    json.dumps([empty,unavailable],allow_nan=False)
