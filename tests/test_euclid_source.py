from pathlib import Path

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from sublens.inversion.source_plane import fit_pixel_source
from sublens.simulation.simulator import LensingSimulator
from sublens.survey import EuclidSourceObservation, centered_crop, load_euclid_source_observation


def test_simulator_delaunay_reconstruction_ranks_heldout_and_plots(tmp_path):
    psf = np.array([[0, 0, 0], [0, 0.7, 0.2], [0, 0.1, 0]], dtype=np.float32)
    sim = LensingSimulator(N_img=32, N_src=49, pixel_scale=0.05, L_src=1.0, psf_kernel=psf)
    y, x = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
    source = torch.exp(-((x - 0.11)**2 + (y + 0.08)**2) / 0.06)
    alpha = torch.zeros_like(sim.theta_grid)
    image, _ = sim.simulate(source[None, None], alpha, add_noise=False)
    obs = EuclidSourceObservation.from_simulation(sim, image, alpha)
    np.testing.assert_allclose(obs.psf, psf[::-1, ::-1])
    assert np.array_equal(obs.image_rays()[0], obs.bx)

    validation = np.zeros(obs.image.shape, dtype=bool)
    validation[::3, ::3] = True
    runs = obs.reconstruct_delaunay(validation_mask=validation, seeds=(11, 22), source_nodes=180)
    assert len(runs.runs) == 2
    assert runs.best is min(runs.runs, key=lambda run: run.validation_chi2_per_pixel)
    assert np.array_equal(runs.best.result.fit_mask, ~validation)
    assert (runs.source.coefficients >= 0).all()
    assert runs.source.optimality < 1e-5
    output = tmp_path / "source.png"
    figure = runs.plot(save_path=output)
    assert output.stat().st_size > 0
    plt.close(figure)


def test_centroidal_mesh_ignores_heldout_brightness_and_predicts_linear_source():
    from sublens.inversion.source_plane import PixelSourceProblem

    axis = np.linspace(-1, 1, 24)
    x, y = np.meshgrid(axis, axis)
    image = 1.0 + .2*x + .1*y
    mask = np.ones(image.shape, dtype=bool)
    mask[::3, ::3] = False
    options = dict(source_nodes=70, regularization=.002, mesh_strategy="centroidal", seed=11)
    first = PixelSourceProblem(image, np.full_like(image, .03), np.ones((1, 1)),
                               mask, x[..., None], y[..., None], **options)
    changed = image.copy()
    changed[~mask] += 10000
    second = PixelSourceProblem(changed, np.full_like(image, .03), np.ones((1, 1)),
                                mask, x[..., None], y[..., None], **options)
    np.testing.assert_array_equal(first.grid.points, second.grid.points)
    source = first.fit(x[..., None], y[..., None])
    assert source.mesh_strategy == "centroidal"
    np.testing.assert_allclose(source.image_model[~mask], image[~mask], atol=.005)


@pytest.mark.parametrize("method", ["cartesian", "delaunay"])
def test_source_solver_preserves_asymmetric_psf_and_predicts_heldout_pixels(method):
    axis = np.linspace(-1, 1, 33)
    x, y = np.meshgrid(axis, axis)
    bx = np.abs(x) - 0.2
    truth = np.exp(-((bx - 0.35)**2 + (y + 0.35)**2) / 0.05)
    psf = np.array([[0, 0, 0], [0, 0.7, 0.2], [0, 0.1, 0]])
    image = fftconvolve(truth, psf, mode="same")
    rms = np.full_like(image, 0.03)
    mask = np.ones_like(image, dtype=bool)
    # The negative-x counterimage constrains the source behind this masked patch.
    mask[9:13, 23:27] = False
    result = fit_pixel_source(
        image, rms, psf, mask, bx[..., None], y[..., None],
        method=method, source_nodes=450, source_size=33,
    )
    assert np.all(result.coefficients >= 0)
    assert np.sqrt(np.mean((result.image_model[~mask] - image[~mask])**2)) < 0.07
    assert result.sample(0.35, -0.35) > 0.9
    assert result.sample(0.35, 0.35) < 0.08


def test_q1_fixed_mass_reconstruction_matches_release_frame(tmp_path):
    lens_id = "102023476_NEG610025395461180513"
    data_dir = Path(__file__).resolve().parents[1] / "archive" / "data"
    release = data_dir / "lens" / lens_id / "result"
    if not (release / "source_reconstruction.fits").is_file():
        pytest.skip("Euclid Q1 release cutout is unavailable")

    obs = load_euclid_source_observation(lens_id, data_dir)
    result = obs.reconstruct()
    prior_source = fits.getdata(release / "source_reconstruction.fits", "VIS")
    prior_image = centered_crop(fits.getdata(release / "source_light.fits", "VIS"), 100)
    bright_arc = obs.mask & (prior_image >= 2 * obs.rms)
    assert np.mean((result.residual[bright_arc] / obs.rms[bright_arc])**2) < 0.65

    # Compare at the released coordinates, with no fitted shift or reflection.
    official = gaussian_filter(prior_source.astype(float), 2).ravel()
    reconstructed = gaussian_filter(result.source_image(), 2).ravel()
    cosine = official @ reconstructed / (np.linalg.norm(official) * np.linalg.norm(reconstructed))
    assert cosine > 0.93

    output = tmp_path / "source.fits"
    result.save_fits(output)
    with fits.open(output) as hdus:
        assert [hdu.name for hdu in hdus] == ["SOURCE", "MODEL", "RESIDUAL", "RMS", "MASK"]
        assert hdus["SOURCE"].header["CDELT2"] == pytest.approx(0.01)
        assert hdus["SOURCE"].header["CRPIX2"] == 101
        np.testing.assert_array_equal(hdus["MASK"].data, obs.mask)
