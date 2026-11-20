"""Check multi-halo injections, common source priors, forward fits and evaluation."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from scipy.spatial import Delaunay
import torch

from sublens import LensingObservation, MultiHaloProblem
from sublens.inversion.source_plane import PixelSourceProblem


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "multi_subhalo"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert summary["completed"] == summary["planned"] == 4
    assert summary["halo_count_not_selected"]
    source = fits.getdata(OUTPUT / "reference/source_truth.fits")
    np.testing.assert_array_equal(source, fits.getdata(ROOT / "one_subhalo/complex_source/reference/sersic_source.fits"))
    psf = fits.getdata(OUTPUT / "reference/psf.fits")
    np.testing.assert_allclose(psf.sum(), 1., rtol=0, atol=1e-12)
    axis = np.linspace(-.65, .65, 201); sx, sy = np.meshgrid(axis, axis)
    models_checked, conditional_matches, nonmonotonic = 0, [], []
    for record in summary["cases"]:
        folder = OUTPUT / record["case"]
        truth = json.loads((folder / "truth.json").read_text())["halos"]
        assert truth == record["truth"]
        assert record["inferred_halo_count"] is None
        assert [r["K"] for r in record["models"]] == [0, 1, 2, 3]
        obs = LensingObservation.from_fits(folder / "observation.fits")
        assert digest(folder / "observation.fits") == record["observation_sha256"]
        clean = fits.getdata(OUTPUT / "reference" / f"{record['case']}_clean.fits")
        native = fits.getdata(OUTPUT / "reference" / f"{record['case']}_native.fits")
        np.testing.assert_allclose(clean, fftconvolve(native, psf, mode="same"), rtol=1e-12, atol=1e-13)
        noise = np.random.default_rng(314159).standard_normal(clean.shape)
        np.testing.assert_allclose(obs.image, clean+.012*noise, rtol=0, atol=1e-12)
        np.testing.assert_allclose(obs.rms, .012, rtol=0, atol=1e-15)
        np.testing.assert_allclose(obs.psf, psf, rtol=1e-13, atol=1e-16)
        if record["case"] == "null":
            previous = ROOT / "one_subhalo/strength_scan/trials/strength_0.000000_seed_314159/observation.fits"
            np.testing.assert_array_equal(obs.image, fits.getdata(previous, "SCI"))
        with fits.open(folder / "model_0.fits") as null:
            null_image = null["MODEL"].data.copy()
            null_vertices = np.column_stack((null["SOURCE_NODES"].data["X"], null["SOURCE_NODES"].data["Y"]))
            null_triangles = null["SOURCE_TRI"].data.copy()
        for row in record["models"]:
            k = row["K"]
            meta = json.loads((folder / f"model_{k}.json").read_text())
            assert {key: value for key, value in row.items() if key != "evaluation"} == meta
            with fits.open(folder / f"model_{k}.fits") as result:
                assert json.loads(result["METADATA"].data["JSON"][0]) == meta
                np.testing.assert_array_equal(result["SCI"].data, obs.image)
                np.testing.assert_array_equal(result["MODEL_MACRO"].data, null_image)
                np.testing.assert_allclose(result["PSF"].data, psf, rtol=1e-13, atol=1e-16)
                for model, residual in (("MODEL_MACRO", "RESID_MACRO"), ("MODEL", "RESIDUAL")):
                    np.testing.assert_allclose(obs.image-result[model].data, result[residual].data, rtol=0, atol=1e-12)
                chi = np.mean((result["RESIDUAL"].data[obs.mask]/obs.rms[obs.mask])**2)
                np.testing.assert_allclose(chi, row["residual"]["chi2_per_pixel"], rtol=1e-12)
                vertices = np.column_stack((result["SOURCE_NODES"].data["X"], result["SOURCE_NODES"].data["Y"]))
                np.testing.assert_array_equal(vertices, null_vertices)
                np.testing.assert_array_equal(result["SOURCE_TRI"].data, null_triangles)
                assert row["regularization"] == record["models"][0]["regularization"]
                problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
                    source_grid=Delaunay(vertices), regularization=row["regularization"], mesh_strategy="centroidal")
                refiner = MultiHaloProblem(obs, problem=problem)
                rays = refiner.rays(torch.tensor(row["parameters"], dtype=torch.float64)).numpy()
                coefficients = result["SOURCE_NODES"].data["FLUX"]
                prediction = problem.render(rays[..., 0], rays[..., 1], coefficients)
                np.testing.assert_allclose(prediction, result["MODEL"].data, rtol=1e-10, atol=1e-11)
                normalized = coefficients/problem.flux_scale
                objective = chi+normalized@(problem._prior@normalized)+problem._ridge*(normalized@normalized)
                np.testing.assert_allclose(objective, row["source_objective"], rtol=1e-9, atol=1e-10)
                raster = LinearNDInterpolator(vertices, coefficients, fill_value=0.)(sx, sy)
                np.testing.assert_allclose(raster, result["SOURCE"].data, rtol=1e-10, atol=1e-11)
                assert result["SOURCE"].header["CDELT1"] == .0065
                halos = np.asarray(row["halos"]).reshape(-1, 3)
                expected_halos = np.asarray(row["parameters"])[3:].reshape(-1, 3).copy()
                expected_halos[:, 2] *= .12
                np.testing.assert_array_equal(halos, expected_halos)
                assert len(result["HALOS"].data) == k
                for j, name in enumerate(("X", "Y", "STRENGTH")):
                    np.testing.assert_array_equal(halos[:, j], result["HALOS"].data[name])
                # All local anchors must come from actual sequential PINN peaks.
                for j, anchor in enumerate(row["anchors"]):
                    proposals = json.loads((folder / f"proposal_{j+1}.json").read_text())["candidates"]
                    assert any(np.allclose(anchor, c["seed_position_arcsec"], atol=1e-12, rtol=0) for c in proposals)
                matches = []
                if truth and k:
                    truth_array = np.array(truth)
                    distances = np.linalg.norm(truth_array[:, None, :2]-halos[None, :, :2], axis=-1)
                    ii, jj = linear_sum_assignment(distances)
                    for i, j in zip(ii, jj):
                        position = float(1000*distances[i, j]); amplitude = float(halos[j, 2]/truth_array[i, 2]-1)
                        matches.append(dict(truth_index=int(i), component_index=int(j), position_error_mas=position,
                                            strength_error_fraction=amplitude,
                                            accurate=bool(row["converged"] and position <= 50 and abs(amplitude) <= .30)))
                assert matches == row["evaluation"]["matches"]
                if k:
                    expected = obs.mask.sum()*(record["models"][k-1]["source_objective"]-row["source_objective"])
                    np.testing.assert_allclose(expected, row["incremental_gain"], rtol=1e-12)
                    if expected < -1e-5: nonmonotonic.append(dict(case=record["case"], K=k, gain=float(expected)))
                models_checked += 1
        if truth:
            conditional_matches.append(dict(case=record["case"], assumed_K=len(truth),
                evaluation=record["models"][len(truth)]["evaluation"]))
    files = sorted(OUTPUT.glob("*/*.fits"))
    for path in files:
        with fits.open(path, checksum=True) as hdus:
            hdus.verify("exception")
            assert all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in hdus), path
    for name in ("strength_scan", "psf_scan", "complex_source"):
        previous = ROOT / "one_subhalo" / name
        hashes = json.loads((previous / "verification.json").read_text())["fits_sha256"]
        for relative, value in hashes.items(): assert digest(previous / relative) == value
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    for name, value in provenance["source_sha256"].items():
        if name.startswith("sublens/"): assert digest(ROOT.parent / name) == value
    report = dict(all_checks_passed=True, cases=4, models=models_checked, fits_files=len(files),
                  matched_psf_and_paired_noise=True, common_source_vertices_and_priors=True,
                  complete_lens_forward_predictions_verified=True, prior_results_preserved=True,
                  no_halo_count_decision=True, nonmonotonic_incremental_objectives=nonmonotonic,
                  conditional_recovery=conditional_matches,
                  fits_sha256={str(p.relative_to(OUTPUT)): digest(p) for p in files})
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("all_checks_passed", "cases", "models", "fits_files", "nonmonotonic_incremental_objectives")}, indent=2))


if __name__ == "__main__": main()
