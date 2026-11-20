"""Verify paired source injections, flux matching, FITS, rasters and recovery."""

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "complex_source"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    summary = json.loads((OUTPUT / "summary.json").read_text())
    cases = {c["id"]: c for c in manifest["cases"]}
    assert summary["completed"] == summary["planned"] == len(cases) == 8
    assert len({r["id"] for r in summary["trials"]}) == 8
    assert {r["id"] for r in summary["trials"]} == set(cases)
    kernel = fits.getdata(OUTPUT / "reference" / "psf.fits")
    np.testing.assert_allclose(kernel.sum(), 1., rtol=0, atol=1e-12)
    assert (kernel >= 0).all() and np.isfinite(kernel).all()
    morphologies = manifest["config"]["morphologies"]
    sources = {k: fits.getdata(OUTPUT / "reference" / f"{k}_source.fits") for k in morphologies}
    for kind, source in sources.items():
        assert source.shape == (257, 257) and (source >= 0).all() and np.isfinite(source).all()
        np.testing.assert_allclose(source.sum(), sources["sersic"].sum(), rtol=2e-7)
        header = fits.getheader(OUTPUT / "reference" / f"{kind}_source.fits")
        assert header["CDELT1"] == header["CDELT2"] == .0125
        assert header["CRPIX1"] == header["CRPIX2"] == 129.
    shared = ROOT / "strength_scan" / "trials" / "strength_0.050000_seed_314159" / "observation.fits"
    with fits.open(shared) as previous:
        geometry = {name: previous[name].data.copy()
                    for name in ("RMS", "PSF", "MASK", "BETA_X", "BETA_Y", "THETA_X", "THETA_Y")}
    axis = np.linspace(-.65, .65, 201)
    sx, sy = np.meshgrid(axis, axis)
    checks = []
    for record in summary["trials"]:
        case = cases[record["id"]]
        folder = OUTPUT / "trials" / record["id"]
        kind, strength = case["morphology"], case["injected_strength_arcsec"]
        truth = json.loads((folder / "truth.json").read_text())
        assert truth["injected_strength_arcsec"] == strength and truth["morphology"] == kind
        key = f"{kind}_strength_{strength:.6f}"
        native = fits.getdata(OUTPUT / "reference" / f"{key}_native.fits")
        clean = fits.getdata(OUTPUT / "reference" / f"{key}_clean.fits")
        null_clean = fits.getdata(OUTPUT / "reference" / f"{kind}_strength_0.000000_clean.fits")
        np.testing.assert_allclose(clean, fftconvolve(native, kernel, mode="same"), rtol=1e-12, atol=1e-13)
        z = np.random.default_rng(case["noise_seed"]).standard_normal(clean.shape)
        metadata = json.loads((folder / "result.json").read_text())
        with fits.open(folder / "observation.fits") as obs, fits.open(folder / "result.fits") as result:
            for name, array in geometry.items():
                np.testing.assert_array_equal(obs[name].data, array)
                np.testing.assert_allclose(result[name].data, array, rtol=1e-13, atol=1e-16)
            np.testing.assert_array_equal(obs["SCI"].data, result["SCI"].data)
            np.testing.assert_allclose(obs["SCI"].data, clean+.012*z, rtol=0, atol=1e-12)
            assert record["observation_sha256"] == digest(folder / "observation.fits")
            if kind == "sersic":
                old = ROOT / "strength_scan" / "trials" / f"strength_{strength:.6f}_seed_314159" / "observation.fits"
                np.testing.assert_array_equal(obs["SCI"].data, fits.getdata(old, "SCI"))
            assert metadata == json.loads(result["METADATA"].data["JSON"][0])
            config = metadata["config"]
            assert config["separation_arcsec"] == .1 and config["gain_radius_arcsec"] == .22
            assert config["seed_strength_arcsec"] == .015 and metadata["max_strength_arcsec"] == .12
            assert config["adaptive_source"]["mesh_rounds"] == 2
            assert config["adaptive_source"]["source_nodes"] == [200, 350, 500]
            assert config["adaptive_source"]["regularizations"] == [2., 20., 200.]
            mask = obs["MASK"].data.astype(bool)
            rho = np.linalg.norm((clean-null_clean)[mask])/.012
            np.testing.assert_allclose(rho, record["rho_image"], rtol=1e-12, atol=1e-12)
            best = next((c for c in metadata["candidates"] if c["is_best"]), None)
            for model, residual, field in (("MODEL_MACRO", "RESID_MACRO", "macro"),
                                           ("MODEL", "RESIDUAL", "with_subhalo")):
                np.testing.assert_allclose(obs["SCI"].data-result[model].data, result[residual].data, rtol=0, atol=1e-12)
                if field in record:
                    chi = np.mean((result[residual].data[mask]/obs["RMS"].data[mask])**2)
                    np.testing.assert_allclose(chi, record[field]["chi2_per_pixel"], rtol=1e-12)
            if best is not None:
                assert best["position_arcsec"] == record["position_arcsec"]
                assert best["strength_arcsec"] == record["strength_arcsec"]
                np.testing.assert_allclose(best["delta_objective"], record["delta_objective"], rtol=1e-12)
                if not case["is_null"]:
                    h = truth["common_parameters"]["halo"]
                    error = 1000*np.linalg.norm(np.array(best["position_arcsec"])-[h["x"], h["y"]])
                    amplitude_error = best["strength_arcsec"]/strength-1
                    np.testing.assert_allclose(error, record["position_error_mas"], rtol=1e-12)
                    np.testing.assert_allclose(amplitude_error, record["strength_error_fraction"], rtol=1e-12)
                    expected = bool(record["macro_converged"] and record["halo_converged"]
                                    and error <= 50 and abs(amplitude_error) <= .30)
                    assert expected == record["recovered"]
            elif not case["is_null"]:
                assert record["recovered"] is False
            if case["is_null"]:
                assert record["recovered"] is None and record["position_error_mas"] is None
            with fits.open(folder / "source_comparison.fits") as rasters:
                source_truth = map_coordinates(sources[kind], [(sy+1.6)/.0125, (sx+1.6)/.0125], order=1, mode="constant")
                np.testing.assert_allclose(rasters["TRUTH"].data, source_truth, rtol=1e-12, atol=1e-13)
                nodes = result["SOURCE_NODES"].data
                points = np.column_stack((nodes["X"], nodes["Y"]))
                for image, coefficients in (("MACRO_SOURCE", "BASE_FLUX"), ("HALO_SOURCE", "FLUX")):
                    evaluated = LinearNDInterpolator(points, nodes[coefficients], fill_value=0.)(sx, sy)
                    np.testing.assert_allclose(rasters[image].data, evaluated, rtol=1e-10, atol=1e-11)
                    assert rasters[image].header["CDELT1"] == .0065
        checks.append(record["id"])
    files = sorted(OUTPUT.glob("reference/*.fits"))+sorted(OUTPUT.glob("trials/*/*.fits"))
    for path in files:
        with fits.open(path, checksum=True) as hdus:
            hdus.verify("exception")
            assert all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in hdus), path
    for directory in ("strength_scan", "psf_scan"):
        report = json.loads((ROOT / directory / "verification.json").read_text())
        for name, value in report["fits_sha256"].items():
            assert digest(ROOT / directory / name) == value
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    for name, value in provenance["source_sha256"].items():
        if not name.endswith("complex_source_scan.py"):
            assert digest(ROOT.parents[1] / name) == value, name
    report = dict(all_checks_passed=True, cases=len(checks), fits_files=len(files), paired_noise=True,
                  fixed_geometry_and_psf=True, source_fluxes_matched=True, source_rasters_verified=True,
                  baseline_sersic_inputs_match_strength_study=True, previous_results_preserved=True,
                  nulls_not_classified_as_detections=True,
                  fits_sha256={str(p.relative_to(OUTPUT)): digest(p) for p in files},
                  analysis_sources={p.name: digest(p) for p in (ROOT / "complex_source_scan.py", Path(__file__))})
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("all_checks_passed", "cases", "fits_files")}, indent=2))


if __name__ == "__main__":
    main()
