import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from dataclasses import replace
import json
from astropy.io import fits
from scipy.signal import fftconvolve

from sublens.survey import search_euclid_subhalos
from sublens.survey.euclid import _halo_deflection
from sublens.survey.euclid_source import EuclidSourceObservation, _q1_image_rays


@pytest.fixture
def compact_deflector():
    shape = (42, 42)
    pixel_scale = 0.1
    theta_e = 0.9
    x, y = _q1_image_rays(shape, pixel_scale, oversample=2)
    magnification_map = 1 - theta_e / np.hypot(x, y)
    bx, by = x * magnification_map, y * magnification_map
    truth = (1.07, 0.28, 0.075)
    ax, ay = _halo_deflection(torch.from_numpy(x), torch.from_numpy(y), *truth)
    gx, gy = bx - ax.numpy(), by - ay.numpy()
    source_light = (
        0.30 * np.exp(-0.5 * (((gx - 0.21) / 0.049)**2 + ((gy - 0.065) / 0.065)**2))
        + 0.19 * np.exp(-0.5 * (((gx - 0.14) / 0.099)**2 + ((gy - 0.12) / 0.045)**2))
    )
    px, py = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    psf = np.exp(-0.5 * ((px / 0.7)**2 + (py / 0.86)**2))
    psf /= psf.sum()
    rms = np.full(shape, 0.0025)
    image = fftconvolve(source_light.mean(axis=-1), psf, mode="same")
    image += np.random.default_rng(42).standard_normal(shape) * rms
    mask = np.hypot(x.mean(axis=-1), y.mean(axis=-1)) < 1.75
    observation = EuclidSourceObservation(
        "injection", image, rms, psf, mask, bx, by, pixel_scale, (0, 0), theta_e
    )
    return observation, truth


def test_profile_search_recovers_a_compact_deflector_with_one_source_basis(compact_deflector):
    observation, truth = compact_deflector
    result = search_euclid_subhalos(
        observation, grid_size=11, profile_grid_size=9, top_k=1, source_nodes=300,
        refinement_radius=0.35, refine_steps=30,
    )
    candidate = result.best_candidate
    assert candidate is not None
    assert candidate.refitted.source.grid is result.baseline.grid
    assert candidate.frozen.source is result.baseline
    assert candidate.refitted.delta_objective > candidate.frozen.delta_objective + 10
    assert np.linalg.norm(np.asarray(candidate.refitted.position) - truth[:2]) < 0.18
    assert abs(candidate.refitted.strength - truth[2]) < 0.03


def test_pinn_uses_cross_fitted_residuals_and_skips_the_grids(compact_deflector):
    observation, _ = compact_deflector
    result = search_euclid_subhalos(observation, proposal="pinn", source_nodes=300,
                                   top_k=2, pinn_steps=20, refine_steps=4)
    assert not result.profile_scores.size and not result.radar_scores.size
    potential = result.residual_potential
    assert 0 < potential.mask.sum() < observation.mask.sum()
    assert np.isfinite(potential.convergence).all()
    assert np.isnan(potential.baseline_residual[~potential.mask]).all()
    assert np.nanmean(potential.fitted_residual**2) < np.nanmean(potential.baseline_residual**2)
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.proposal_method == "pinn"
        assert candidate.refitted.source.grid is result.baseline.grid


def test_pinn_method_refines_and_plots_ranked_peaks(compact_deflector, tmp_path):
    observation, truth = compact_deflector
    result = observation.search_pinn(source_nodes=300, top_k=2, pinn_steps=20,
                                     refine_steps=4, separation=0.25)
    assert [candidate.seed for candidate in result.candidates] == [
        peak[:2] for peak in result.residual_potential.peaks
    ]
    assert result.best_candidate is not None
    assert result.best_candidate is max(result.candidates, key=lambda item: item.fit.delta_objective)
    assert result.best_candidate.fit.source.grid is result.baseline.grid

    kappa = result.plot(truth_position=truth[:2], save_path=tmp_path / "kappa.png")
    fit = result.plot_fit(save_path=tmp_path / "fit.png")
    np.testing.assert_allclose(result.residual_macro, observation.image - result.baseline.image_model)
    prediction = result.best_candidate.fit.image_model
    np.testing.assert_allclose(result.residual_with_subhalo, observation.image - prediction)
    residuals = result.plot_residuals(save_path=tmp_path / "residuals.png")
    macro_map, halo_map = (axis.images[0] for axis in residuals.axes[:2])
    assert macro_map.get_clim() == halo_map.get_clim() == (-4, 4)
    np.testing.assert_allclose(halo_map.get_array()[observation.mask],
                               result.residual_with_subhalo[observation.mask] / observation.rms[observation.mask])
    assert (tmp_path / "kappa.png").stat().st_size > 0
    assert (tmp_path / "fit.png").stat().st_size > 0
    assert (tmp_path / "residuals.png").stat().st_size > 0
    plt.close(kappa)
    plt.close(fit)
    plt.close(residuals)


def test_competing_kappa_peaks_survive_without_changing_the_gain_aperture():
    from sublens.inversion.potential import _potential_peaks

    axis = np.arange(-60, 61)*.02
    x, y = np.meshgrid(axis, axis)
    first = np.exp(-((x+.30)**2+(y-.20)**2)/(2*.04**2))
    second = np.exp(-((x+.30)**2+(y-.56)**2)/(2*.04**2))
    kappa = .2*first+second
    gain = 3*first+2*second
    options = dict(center=(0., 0.), radius=(0., 2.), pixel_scale=.02, top_k=5, gain_radius=.10)
    aggressive = _potential_peaks(kappa, gain, x, y, separation=.44, **options)
    competing = _potential_peaks(kappa, gain, x, y, separation=.04, **options)
    assert len(aggressive) == 1
    assert len(competing) == 2
    np.testing.assert_allclose(aggressive[0], competing[0])
    assert np.linalg.norm(np.array(competing[1][:2])-[-.30, .56]) < .021
    no_exclusion = _potential_peaks(kappa, gain, x, y, separation=0., **options)
    np.testing.assert_allclose(no_exclusion, competing)


def test_merge_candidates_uses_final_fits_and_preserves_best_and_provenance(compact_deflector, tmp_path):
    from sublens.inversion.halo_search import HaloFit, PinnHaloCandidate, PinnHaloSearchResult
    from sublens.inversion.potential import ResidualPotential

    observation, _ = compact_deflector
    source = observation.reconstruct(source_nodes=80).source
    potential = ResidualPotential(observation.mask, observation.image, observation.image,
                                  np.zeros_like(observation.image), [])
    def candidate(rank, seed, x, strength, score, converged=True):
        fit = HaloFit(x, .20, strength, score, score, converged, source, source.image_model)
        return PinnHaloCandidate(rank, seed, 1., fit)

    original = PinnHaloSearchResult(observation, source, potential, (
        candidate(1, (0., 0.), 1., .05, 10.),
        candidate(2, (-1., 0.), 1.02, .051, 20.),
        candidate(3, (.01, 0.), -1., .05, 5.),
        candidate(4, (0., 0.), 1.01, .10, 8.),
        candidate(5, (0., 0.), 1.02, 0., 0.),
        candidate(6, (0., 0.), 1.02, .05, 2., False),
    ), (0., 2.))
    merged = original.merge_candidates(radius=.05)
    assert len(original.candidates) == 6
    assert [c.rank for c in merged.candidates] == [2, 3, 4, 5, 6]
    assert merged.best_candidate is original.best_candidate
    group = next(g for g in merged.config["candidate_merge"]["groups"] if g["representative_rank"] == 2)
    assert [m["rank"] for m in group["members"]] == [2, 1]
    assert merged.merge_candidates(radius=.05).config["candidate_merge"] == merged.config["candidate_merge"]
    merged.save_fits(tmp_path / "merged.fits", resolution=11)
    with fits.open(tmp_path / "merged.fits") as hdus:
        assert len(hdus["CANDIDATES"].data) == 5
        assert hdus[0].header["BESTRANK"] == 2
        assert "FLUX_2" in hdus["SOURCE_NODES"].columns.names
    with pytest.raises(ValueError):
        original.merge_candidates(radius=0.)


def test_cross_fit_ablation_keeps_scoring_pixels_and_uses_full_source_residual(compact_deflector):
    from sublens.inversion.potential import _arc_folds, _arc_masks

    observation, _ = compact_deflector
    x, y = (a.mean(axis=-1) for a in observation.image_rays())
    expected = [covered for _, _, covered in _arc_masks(observation, x, y)]
    folds = list(_arc_folds(observation, x, y, 150, .002, cross_fit=False))
    full = observation.reconstruct(source_nodes=150, regularization=.002)
    assert len(folds) == len(expected) > 0
    for (covered, residual, gradient), mask in zip(folds, expected):
        np.testing.assert_array_equal(covered, mask)
        np.testing.assert_allclose(residual, full.residual)
        assert np.isfinite(gradient).all()
    result = observation.search_pinn(
        source_nodes=150, top_k=1, pinn_steps=2, refine_steps=1, cross_fit=False,
    )
    assert result.config["cross_fit"] is False
    assert result.config["separation_arcsec"] == 2*observation.pixel_scale
    assert result.config["gain_radius_arcsec"] == .22
    np.testing.assert_array_equal(result.residual_potential.mask, np.logical_or.reduce(expected))


def test_long_wing_psf_guard_cap_restores_source_overlap_without_changing_kernel(compact_deflector):
    from sublens.inversion.potential import _arc_masks, _psf_guard_radius
    observation,_ = compact_deflector
    kernel = np.ones((101,101))/101**2
    broad = replace(observation,psf=kernel)
    x,y = (a.mean(axis=-1) for a in broad.image_rays())
    strict = list(_arc_masks(broad,x,y))
    capped = list(_arc_masks(broad,x,y,psf_guard_max_radius=.2))
    assert not strict and capped
    np.testing.assert_array_equal(broad.psf,kernel)
    row,col = np.indices(kernel.shape)
    radii = np.hypot(row-50,col-50).ravel()
    order = np.argsort(radii)
    expected = radii[order[np.searchsorted(np.cumsum(kernel.ravel()[order]),.98)]]
    assert _psf_guard_radius(kernel) == expected


def test_joint_refinement_reprofiles_macro_null_and_halo_on_same_mesh(compact_deflector):
    observation, _ = compact_deflector
    original = observation.search_pinn(source_nodes=250, top_k=2, pinn_steps=20,
                                       refine_steps=5, separation=0.25)
    macro = dict(theta_E=0.9, q=1.0, phi=0.0, gamma=2.0,
                 gamma_ext=0.0, phi_ext=0.0, center=(0.0, 0.0))
    joint = original.refine_joint(macro, maxiter=15)
    assert joint.baseline.grid is original.baseline.grid
    assert joint.baseline.objective <= original.baseline.objective + 1e-7
    assert joint.macro_model is not None
    assert joint.macro_converged is not None
    count = int(observation.mask.sum())
    null_chi2 = np.sum((joint.residual_macro[observation.mask] / observation.rms[observation.mask])**2)
    for before, after in zip(original.candidates, joint.candidates):
        fit = after.fit
        assert fit.source.grid is original.baseline.grid
        assert fit.source.objective <= before.fit.source.objective + 1e-7
        assert fit.source.objective <= joint.baseline.objective + 1e-7
        assert 0.9 * 0.97 <= fit.macro_model["theta_E"] <= 0.9 * 1.03
        assert fit.macro_model["q"] == macro["q"]
        assert abs(fit.x - after.seed[0]) <= original.refinement_radius + 1e-7
        assert 0 <= fit.strength <= original.max_strength
        halo_chi2 = np.sum(((observation.image - fit.image_model)[observation.mask]
                           / observation.rms[observation.mask])**2)
        assert fit.delta_chi2 == pytest.approx(null_chi2 - halo_chi2)
        assert fit.delta_objective == pytest.approx(count * (joint.baseline.objective - fit.source.objective))
    with pytest.raises(ValueError, match="input EPL"):
        original.refine_joint({**macro, "theta_E": 1.0}, maxiter=1)


def test_adaptive_refinement_selects_predictive_source_and_shares_hypothesis_mesh(compact_deflector):
    observation, _ = compact_deflector
    original = observation.search_pinn(source_nodes=120, top_k=1, pinn_steps=20, refine_steps=2,
                                       mesh_strategy="centroidal")
    macro = dict(theta_E=.9, q=1., phi=0., gamma=2., gamma_ext=0., phi_ext=0., center=(0., 0.))
    validation = observation.mask.copy()
    validation[:] = False
    validation[::4, ::4] = observation.mask[::4, ::4]
    fitted = original.refine_adaptive(
        macro, source_nodes=(80, 120), regularizations=(.2, 2.), seeds=(11, 22),
        validation_mask=validation, mesh_rounds=1, maxiter=4, n_starts=2,
        retry_failed=False, expand_boundary=False,
    )
    history = fitted.config["adaptive_source"]["history"][0]
    assert len(history["source_trials"]) == 8
    assert history["selected"] == min(history["source_trials"], key=lambda r:r["validation_chi2_per_pixel"])
    assert fitted.baseline.mesh_strategy == "centroidal"
    assert fitted.baseline.regularization == history["selected"]["regularization"]
    for candidate in fitted.candidates:
        assert candidate.fit.source.grid is fitted.baseline.grid
        assert candidate.fit.source.regularization == fitted.baseline.regularization
        assert candidate.fit.optimizer_message is not None
    assert fitted.config["adaptive_source"]["validation_pixel_count"] == validation.sum()


def test_pinn_result_reports_and_exports_fixed_joint_and_empty_searches(compact_deflector, tmp_path, capsys):
    observation, _ = compact_deflector
    fixed = observation.search_pinn(
        source_nodes=np.int64(200), top_k=2, pinn_steps=20, refine_steps=3,
    )
    macro = dict(theta_E=0.9, q=1.0, phi=0.0, gamma=2.0,
                 gamma_ext=0.0, phi_ext=0.0, center=(0.0, 0.0))
    joint = fixed.refine_joint(macro, maxiter=3)
    empty = replace(observation, image=np.zeros_like(observation.image)).search_pinn(
        source_nodes=100, pinn_steps=2, refine_steps=1,
    )
    assert not empty.candidates
    assert "joint_refinement" not in fixed.config
    for label, result in (("fixed", fixed), ("joint", joint), ("empty", empty)):
        text = result.summary()
        assert capsys.readouterr().out.strip() == text
        assert result.summary(print_output=False) == text
        assert capsys.readouterr().out == ""
        metadata = result.to_dict()
        json_path, fits_path = tmp_path / f"{label}.json", tmp_path / f"{label}.fits"
        assert result.save_json(json_path) == json_path
        assert result.save_fits(fits_path, resolution=31) == fits_path
        assert json.loads(json_path.read_text()) == metadata
        assert metadata["config"]["seeds"]["pinn"] == 42
        assert metadata["fit_pixel_count"] == int(observation.mask.sum())
        assert result.detection() == metadata['detection']
        assert 'subhalo detected =' in text
        assert 'Macro-only chi2/pixel' in result.summary(print_output=False,detailed=True)

        with fits.open(fits_path, checksum=True) as hdus:
            assert all(hdu.verify_checksum() == 1 for hdu in hdus)
            np.testing.assert_allclose(hdus["RESID_MACRO"].data,
                                       hdus["SCI"].data - hdus["MODEL_MACRO"].data)
            np.testing.assert_allclose(hdus["RESIDUAL"].data, hdus["SCI"].data - hdus["MODEL"].data)
            assert len(hdus["CANDIDATES"].data) == len(result.candidates)
            assert json.loads(hdus["METADATA"].data["JSON"][0]) == metadata
            assert hdus[0].header['HALODET'] == metadata['detection']['detected']
            assert hdus[0].header['DCHIMIN'] == 25.
            assert hdus[0].header['DETCAL'] is False
            np.testing.assert_array_equal(hdus["SOURCE_TRI"].data, result.baseline.grid.simplices)
            best = result.best_candidate
            source = result.baseline if best is None else best.fit.source
            np.testing.assert_array_equal(hdus["SOURCE_NODES"].data["FLUX"], source.coefficients)
            if best is not None:
                assert hdus[0].header["BESTRANK"] == best.rank
                assert metadata["best_candidate_rank"] == best.rank
                np.testing.assert_allclose(hdus["MODEL"].data, best.fit.image_model)
            else:
                assert metadata["best_candidate_rank"] is None
                assert "No PINN candidates" in text
        original_bytes = json_path.read_bytes(), fits_path.read_bytes()
        with pytest.raises(FileExistsError):
            result.save_json(json_path)
        with pytest.raises(FileExistsError):
            result.save_fits(fits_path, resolution=31)
        assert original_bytes == (json_path.read_bytes(), fits_path.read_bytes())
        result.save_json(json_path, overwrite=True)
        result.save_fits(fits_path, resolution=31, overwrite=True)
