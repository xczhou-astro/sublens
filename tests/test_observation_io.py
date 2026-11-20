"""Generic array/FITS entry points preserve the inference geometry and noise."""

import numpy as np
import pytest
import torch
from astropy.io import fits

from sublens import LensingObservation, LensingSimulator, MacroLensModel, SubLensPipeline
from sublens.grids import make_image_rays


def test_simulation_fits_preserves_psf_rays_and_spatial_noise(tmp_path):
    psf = np.array([[0, 0.1, 0], [0, 0.6, 0.2], [0, 0.1, 0]], dtype=np.float32)
    noise = np.linspace(0.01, 0.03, 30 * 30).reshape(30, 30)
    sim = LensingSimulator(N_img=30, N_src=49, pixel_scale=0.05, L_src=0.8,
                           psf_kernel=psf, noise_map=noise)
    macro = dict(theta_E=0.5, q=0.85, phi=0.2, gamma=2.1,
                 gamma_ext=0.03, phi_ext=0.1, center=(0.0, 0.0))
    with torch.no_grad():
        alpha, _ = MacroLensModel(**macro)(sim.theta_grid)
    sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
    source = torch.exp(-(sx**2 + sy**2) / 0.04)[None, None]
    mask = np.ones((30, 30), dtype=bool)
    mask[:2] = False
    path = tmp_path / "simulation.fits"
    exported = sim.simulate_to_fits(
        path, source, alpha, alpha_macro=alpha, macro_model=macro, seed=8, mask=mask,
    )
    expected, clean = sim.simulate(source, alpha, seed=8)
    torch.manual_seed(8)
    expected_noise = sim.noise_map * torch.randn_like(clean)
    torch.testing.assert_close(expected - clean, expected_noise, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(exported.image, expected[0, 0].numpy())

    restored = LensingObservation.from_fits(path)
    np.testing.assert_array_equal(restored.image, exported.image)
    np.testing.assert_array_equal(restored.noise, exported.noise)
    np.testing.assert_array_equal(restored.mask, mask)
    np.testing.assert_array_equal(restored.bx, exported.bx)
    np.testing.assert_array_equal(restored.image_rays()[0], exported.image_rays()[0])
    np.testing.assert_allclose(restored.psf, psf[::-1, ::-1], rtol=1e-6)
    assert restored.macro_model == exported.macro_model
    assert restored.image.dtype.isnative and restored.bx.dtype.isnative
    with fits.open(path, checksum=True) as hdus:
        assert [hdu.name for hdu in hdus][:4] == ["SCI", "RMS", "PSF", "MASK"]
        assert "SUB_X" not in hdus[0].header
    with pytest.raises(OSError):
        exported.to_fits(path)


def test_arrays_and_fits_pipeline_use_identical_rectangular_data(tmp_path):
    shape = (28, 34)
    x, y = make_image_rays(shape, 0.05)
    image = np.exp(-((x[..., 0] - 0.12)**2 + (y[..., 0] + 0.08)**2) / 0.03)
    noise = np.full(shape, 0.02)
    psf = np.array([[0, 0, 0], [0.1, 0.7, 0.1], [0, 0.1, 0]])
    macro = dict(theta_E=0.45, q=0.8, phi=0.2, gamma=2.0,
                 gamma_ext=0.02, phi_ext=0.1, center=(0.0, 0.0))
    pipeline = SubLensPipeline(pixel_scale=0.05, device="cpu")
    obs = pipeline.observation(image, noise=noise, psf=psf, macro_model=macro)
    path = tmp_path / "input.fits"
    obs.to_fits(path)
    fits_pipeline = SubLensPipeline.from_fits(path, device="cpu")
    from_fits = fits_pipeline.observation()
    assert isinstance(from_fits, LensingObservation)
    np.testing.assert_array_equal(from_fits.bx, obs.bx)

    options = dict(seeds=(11,), source_nodes=100)
    array_fit = pipeline.reconstruct_delaunay(image, noise=noise, psf=psf, macro_model=macro, **options)
    fits_fit = fits_pipeline.reconstruct_delaunay(**options)
    np.testing.assert_allclose(array_fit.source.image_model, fits_fit.source.image_model, rtol=1e-6, atol=1e-8)
    search_options = dict(source_nodes=100, pinn_steps=3, refine_steps=2, top_k=1)
    array_search = pipeline.search_pinn(image, noise=noise, psf=psf, macro_model=macro, **search_options)
    fits_search = fits_pipeline.search_pinn(**search_options)
    np.testing.assert_allclose(array_search.baseline.image_model, fits_search.baseline.image_model, atol=1e-10)
    np.testing.assert_allclose(array_search.residual_potential.convergence,
                               fits_search.residual_potential.convergence, atol=1e-6)
    assert pipeline.fit(obs, proposal="pinn", num_pinn_steps=2).observation is obs


@pytest.mark.parametrize("extension", ["VAR", "WHT"])
def test_fits_variance_weight_conversion_and_separate_files(tmp_path, extension):
    shape = (20, 22)
    rms = 0.04
    noise = np.full(shape, rms**2 if extension == "VAR" else 1 / rms**2)
    header = fits.Header({"PIXSCALE": 0.08})
    science = tmp_path / "science.fits"
    fits.HDUList([fits.PrimaryHDU(np.ones(shape), header=header),
                  fits.ImageHDU(noise, name=extension)]).writeto(science)
    psf = tmp_path / "psf.fits"
    fits.PrimaryHDU(np.ones((3, 3))).writeto(psf)
    macro = {"theta_E": 0.5, "q": 0.9}
    obs = LensingObservation.from_fits(science, psf=psf, macro_model=macro)
    np.testing.assert_allclose(obs.noise, rms)
    rms_file = tmp_path / "rms.fits"
    fits.PrimaryHDU(np.full(shape, 0.07)).writeto(rms_file)
    pipeline = SubLensPipeline.from_fits(science, noise_path=rms_file, psf_path=psf, device="cpu")
    np.testing.assert_allclose(pipeline.observation(macro_model=macro).noise, 0.07)
