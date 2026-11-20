import numpy as np
import pytest
from scipy.signal import fftconvolve

from sublens import build_psf_kernel
from sublens.inversion.source_plane import PixelSourceProblem, _blur_operator


@pytest.mark.parametrize("profile",["gaussian","moffat","diffraction"])
def test_pixel_integrated_psf_core_flux_and_symmetry(profile):
    kernel,meta = build_psf_kernel(.15,.05,profile=profile,return_metadata=True)
    assert kernel.ndim == 2 and all(n%2 for n in kernel.shape)
    assert np.isfinite(kernel).all() and (kernel>=0).all()
    assert kernel.sum() == pytest.approx(1.,abs=1e-12)
    np.testing.assert_allclose(kernel,kernel[::-1,::-1],rtol=1e-7,atol=1e-11)
    assert .99 < meta["estimated_flux_before_stamp_renormalization"] < 1.01
    assert (meta["fwhm_x_arcsec"]+meta["fwhm_y_arcsec"])/2 == pytest.approx(.15,rel=.003)
    assert meta["encircled_energy_radii_arcsec"]["0.98"] > meta["encircled_energy_radii_arcsec"]["0.5"]


def test_fft_source_operator_preserves_asymmetric_wide_psf_and_out_of_frame_flux():
    axis = np.linspace(-1,1,13)
    x,y = np.meshgrid(axis,axis)
    yy,xx = np.indices((41,41))
    psf = np.exp(-((xx-22)**2/30+(yy-19)**2/80))
    psf /= psf.sum()
    image = .4+np.exp(-((x-.2)**2+(y+.1)**2)/.1)
    mask = np.ones(image.shape,dtype=bool)
    mask[::3,::3] = False
    problem = PixelSourceProblem(image,np.full_like(image,.03),psf,mask,x[...,None],y[...,None],
                                 method="cartesian",source_size=9,source_half_width=1.1)
    native = problem._native_mapping(x[...,None],y[...,None])
    explicit = _blur_operator(image.shape,mask,problem.psf)
    # Force the FFT path on this small fixture for a direct independent comparison.
    problem._blur = None
    problem._initialize_fft()
    fft_operator = problem._weighted_operator(native)
    expected = (explicit@native).toarray()*(problem.flux_scale*problem.weight)[:,None]
    np.testing.assert_allclose(fft_operator,expected,rtol=1e-11,atol=1e-13)
    assert problem.convolution_psf.sum() < 1.
    coefficients = np.ones(problem.size)
    full = fftconvolve((native@coefficients).reshape(image.shape),problem.psf,mode="same")
    np.testing.assert_allclose(problem.render(x[...,None],y[...,None],coefficients),full,atol=1e-13)
    fitted = problem.fit(x[...,None],y[...,None])
    torch_fit = problem.fit_torch(x[...,None],y[...,None],max_steps=3000,tolerance=1e-6)
    assert torch_fit.objective == pytest.approx(fitted.objective,rel=2e-4)
    reweighted=problem.with_regularization(.02)
    cached=reweighted.fit(x[...,None],y[...,None])
    fresh=PixelSourceProblem(image,np.full_like(image,.03),psf,mask,x[...,None],y[...,None],
                             method="cartesian",source_size=9,source_half_width=1.1,regularization=.02)
    fresh._blur=None
    fresh._initialize_fft()
    independent=fresh.fit(x[...,None],y[...,None])
    np.testing.assert_allclose(cached.image_model,independent.image_model,atol=1e-7)
    assert reweighted._operator_cache is problem._operator_cache


def test_wide_simulator_psf_preserves_correlation_orientation_and_image_gradient():
    import torch
    from sublens import LensingSimulator

    y,x=np.indices((41,41))
    psf=np.exp(-((x-22)**2/13+(y-18)**2/30))
    sim=LensingSimulator(N_img=13,psf_kernel=psf,device="cpu")
    image=torch.linspace(0,1,169).reshape(1,1,13,13).requires_grad_()
    expected=torch.nn.functional.conv2d(image,sim.psf_kernel,padding=20)
    actual=sim.convolve_psf(image)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-7)
    first,=torch.autograd.grad(actual.square().sum(),image,retain_graph=True)
    second,=torch.autograd.grad(expected.square().sum(),image)
    torch.testing.assert_close(first,second,rtol=2e-6,atol=2e-7)
