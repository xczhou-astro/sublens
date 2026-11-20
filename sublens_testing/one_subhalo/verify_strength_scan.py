"""Audit paired injections, FITS products and truth-blind recovery bookkeeping."""

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "strength_scan"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    summary = json.loads((OUTPUT / "summary.json").read_text())
    cases = {c["id"]: c for c in manifest["cases"]}
    assert summary["completed"] == summary["planned"] == len(cases) == 10
    assert len({r["id"] for r in summary["trials"]}) == 10
    assert {r["id"] for r in summary["trials"]} == set(cases)
    kernel = fits.getdata(OUTPUT / "reference" / "psf.fits")
    np.testing.assert_allclose(kernel.sum(), 1, rtol=0, atol=1e-12)
    assert np.isfinite(kernel).all() and (kernel >= 0).all()
    common = ROOT / "psf_scan" / "trials" / "gaussian_fwhm_0.090_seed_314159" / "observation.fits"
    with fits.open(common) as previous:
        geometry = {name: previous[name].data.copy()
                    for name in ("RMS", "PSF", "MASK", "BETA_X", "BETA_Y", "THETA_X", "THETA_Y")}
        reference_image = previous["SCI"].data.copy()
    null_clean = fits.getdata(OUTPUT / "reference" / "strength_0.000000_clean.fits")
    checks = []
    for record in summary["trials"]:
        case = cases[record["id"]]
        folder = OUTPUT / "trials" / record["id"]
        strength = case["injected_strength_arcsec"]
        truth = json.loads((folder / "truth.json").read_text())
        assert truth["halo"]["strength"] == strength
        native = fits.getdata(OUTPUT / "reference" / f"strength_{strength:.6f}_native.fits")
        clean = fits.getdata(OUTPUT / "reference" / f"strength_{strength:.6f}_clean.fits")
        np.testing.assert_allclose(clean, fftconvolve(native, kernel, mode="same"), rtol=1e-12, atol=1e-13)
        noise = np.random.default_rng(case["noise_seed"]).standard_normal(clean.shape)
        metadata = json.loads((folder / "result.json").read_text())
        with fits.open(folder / "observation.fits") as obs, fits.open(folder / "result.fits") as result:
            for name, array in geometry.items():
                np.testing.assert_array_equal(obs[name].data, array)
                np.testing.assert_array_equal(result[name].data, array)
            np.testing.assert_array_equal(obs["SCI"].data, result["SCI"].data)
            np.testing.assert_allclose(obs["SCI"].data, clean+.012*noise, rtol=0, atol=1e-12)
            if strength == .05:
                np.testing.assert_array_equal(obs["SCI"].data, reference_image)
            assert record["observation_sha256"] == digest(folder / "observation.fits")
            assert metadata == json.loads(result["METADATA"].data["JSON"][0])
            config = metadata["config"]
            assert config["separation_arcsec"] == .1 and config["gain_radius_arcsec"] == .22
            assert config["seed_strength_arcsec"] == .015 and metadata["max_strength_arcsec"] == .12
            assert config["psf_guard_max_radius_arcsec"] == .135
            assert config["adaptive_source"]["mesh_rounds"] == 2
            assert metadata["macro_model_input"] == truth["macro"]
            mask = obs["MASK"].data.astype(bool)
            rho = np.linalg.norm((clean-null_clean)[mask])/.012
            np.testing.assert_allclose(record["rho_image"], rho, rtol=1e-12, atol=1e-12)
            for model, residual, key in (("MODEL_MACRO", "RESID_MACRO", "macro"),
                                         ("MODEL", "RESIDUAL", "with_subhalo")):
                np.testing.assert_allclose(obs["SCI"].data-result[model].data, result[residual].data, rtol=0, atol=1e-12)
                chi = np.mean((result[residual].data[mask]/obs["RMS"].data[mask])**2)
                np.testing.assert_allclose(chi, record[key]["chi2_per_pixel"], rtol=1e-12)
            best = next(c for c in metadata["candidates"] if c["is_best"])
            assert best["position_arcsec"] == record["position_arcsec"]
            assert best["strength_arcsec"] == record["strength_arcsec"]
            np.testing.assert_allclose(best["delta_chi2"], record["delta_chi2"], rtol=1e-12)
            np.testing.assert_allclose(best["delta_objective"], record["delta_objective"], rtol=1e-12)
            if case["is_null"]:
                assert record["recovered"] is None
                assert record["position_error_mas"] is None and record["strength_error_fraction"] is None
            else:
                h = truth["halo"]
                position_error = 1000*np.linalg.norm(np.array(best["position_arcsec"])-[h["x"], h["y"]])
                strength_error = best["strength_arcsec"]/strength-1
                np.testing.assert_allclose(position_error, record["position_error_mas"], rtol=1e-12)
                np.testing.assert_allclose(strength_error, record["strength_error_fraction"], rtol=1e-12)
                expected = bool(record["macro_converged"] and record["halo_converged"]
                                and position_error <= 50 and abs(strength_error) <= .30)
                assert record["recovered"] == expected
        checks.append(case["id"])
    files = sorted(OUTPUT.glob("reference/*.fits"))+sorted(OUTPUT.glob("trials/*/*.fits"))
    for path in files:
        with fits.open(path, checksum=True) as hdus:
            hdus.verify("exception")
            assert all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in hdus), path
    preserved = []
    for directory in ("psf_scan", "noise_scan", "candidate_refinement"):
        path = ROOT / directory
        report = json.loads((path / "verification.json").read_text())
        if directory == "candidate_refinement":
            hashes = {row["path"]: row["sha256"] for row in report["fits_products"]}
        else:
            hashes = report["fits_sha256"]
        for name, value in hashes.items():
            assert digest(path / name) == value
        preserved.append(directory)
    source = json.loads((OUTPUT / "provenance.json").read_text())
    # Generator/plotting edits are allowed during a run; inference modules must match.
    workspace = ROOT.parents[1]
    for name, value in source["source_sha256"].items():
        if name.startswith("sublens/") or name.endswith("/run.py"):
            assert digest(workspace / name) == value, name
    report = dict(all_checks_passed=True, cases=len(checks), fits_files=len(files),
                  paired_noise=True, matched_psf=True, fixed_geometry=True,
                  baseline_b005_input_identical_to_psf_study=True,
                  previous_results_preserved=preserved, null_not_scored_as_recovery=True,
                  fits_sha256={str(p.relative_to(OUTPUT)): digest(p) for p in files},
                  analysis_sources={p.name: digest(p) for p in (ROOT / "strength_scan.py", Path(__file__))})
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("all_checks_passed", "cases", "fits_files")}, indent=2))


if __name__ == "__main__":
    main()
