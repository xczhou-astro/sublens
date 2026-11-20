"""Audit controlled meshes, matched SNR and independent-noise diagnostics."""

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spiral_diagnosis"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((OUTPUT / "config.json").read_text())
    summary = json.loads((OUTPUT / "summary.json").read_text())
    datasets = json.loads((OUTPUT / "datasets.json").read_text())
    assert summary["completed"] == summary["planned"] == len(config["methods"]) == 17
    records = {r["id"]: r for r in summary["methods"]}
    assert set(records) == {r["id"] for r in config["methods"]}
    arrays = {}
    for name, info in datasets.items():
        folder = OUTPUT / "datasets" / name
        clean = fits.getdata(folder / "clean.fits")
        original = Path(info["original_folder"])
        source_clean = ROOT / "complex_source/reference" / f"{info['morphology']}_strength_{info['injected_strength_arcsec']:.6f}_clean.fits"
        np.testing.assert_array_equal(clean, fits.getdata(source_clean))
        with fits.open(original / "observation.fits") as old:
            for filename, seed in (("observation.fits", 314159), ("independent_observation.fits", 271828)):
                with fits.open(folder / filename) as obs:
                    z = np.random.default_rng(seed).standard_normal(clean.shape)
                    np.testing.assert_allclose(obs["SCI"].data, clean+info["rms"]*z, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(obs["RMS"].data, info["rms"], rtol=0, atol=1e-15)
                    for field in ("PSF", "MASK", "BETA_X", "BETA_Y", "THETA_X", "THETA_Y"):
                        np.testing.assert_array_equal(obs[field].data, old[field].data)
        null_clean = fits.getdata(ROOT / "complex_source/reference" / f"{info['morphology']}_strength_0.000000_clean.fits")
        mask = fits.getdata(folder / "observation.fits", "MASK").astype(bool)
        rho = np.linalg.norm((clean-null_clean)[mask])/info["rms"]
        np.testing.assert_allclose(rho, info["rho_image"], rtol=1e-12, atol=1e-12)
        arrays[name] = (clean, fits.getdata(folder / "independent_observation.fits", "SCI"))
    np.testing.assert_allclose(datasets["spiral"]["rho_image"], datasets["sersic_low_snr"]["rho_image"], rtol=1e-12)
    np.testing.assert_allclose(datasets["sersic"]["rho_image"], datasets["spiral_high_snr"]["rho_image"], rtol=1e-12)
    for method in config["methods"]:
        row = records[method["id"]]
        folder = OUTPUT / "methods" / row["id"]
        info = datasets[row["dataset"]]
        path = OUTPUT / "datasets" / row["dataset"] / "observation.fits"
        assert digest(path) == row["input_sha256"]
        metadata = json.loads((folder / "result.json").read_text())
        clean, independent = arrays[row["dataset"]]
        with fits.open(path) as obs, fits.open(folder / "result.fits") as result:
            assert metadata == json.loads(result["METADATA"].data["JSON"][0])
            for field in ("SCI", "RMS", "MASK", "BETA_X", "BETA_Y", "THETA_X", "THETA_Y"):
                np.testing.assert_array_equal(obs[field].data, result[field].data)
            np.testing.assert_allclose(obs["PSF"].data, result["PSF"].data, rtol=1e-13, atol=1e-16)
            mask = obs["MASK"].data.astype(bool)
            for model, residual, key in (("MODEL_MACRO", "RESID_MACRO", "macro"), ("MODEL", "RESIDUAL", "with_subhalo")):
                np.testing.assert_allclose(obs["SCI"].data-result[model].data, result[residual].data, rtol=0, atol=1e-12)
                chi = np.mean((result[residual].data[mask]/info["rms"])**2)
                np.testing.assert_allclose(chi, row[key]["chi2_per_pixel"], rtol=1e-12)
            prediction = result["MODEL"].data
            independent_error = np.mean(((independent-prediction)[mask]/info["rms"])**2)
            clean_error = np.mean(((clean-prediction)[mask]/info["rms"])**2)
            np.testing.assert_allclose(independent_error, row["independent_prediction_chi2_per_pixel"], rtol=1e-12)
            np.testing.assert_allclose(clean_error, row["clean_model_error_per_pixel"], rtol=1e-12)
            macro_independent = np.mean(((independent-result["MODEL_MACRO"].data)[mask]/info["rms"])**2)
            np.testing.assert_allclose(macro_independent, row["independent_macro_chi2_per_pixel"], rtol=1e-12)
            np.testing.assert_allclose(mask.sum()*(macro_independent-independent_error),
                                       row["independent_predictive_delta_chi2"], rtol=1e-12)
            best = next(c for c in metadata["candidates"] if c["is_best"])
            assert best["position_arcsec"] == row["position_arcsec"]
            assert best["strength_arcsec"] == row["strength_arcsec"]
            truth = json.loads((ROOT / "data/truth.json").read_text())["halo"]
            if info["injected_strength_arcsec"]:
                error = 1000*np.linalg.norm(np.array(best["position_arcsec"])-[truth["x"], truth["y"]])
                amplitude = best["strength_arcsec"]/info["injected_strength_arcsec"]-1
                np.testing.assert_allclose(error, row["position_error_mas"], rtol=1e-12)
                np.testing.assert_allclose(amplitude, row["strength_error_fraction"], rtol=1e-12)
                assert row["recovered"] == bool(row["macro_converged"] and row["halo_converged"]
                                                and error <= 50 and abs(amplitude) <= .30)
            else: assert row["recovered"] is None
            if row["mode"] in ("fixed", "updated"):
                assert metadata["baseline"]["regularization"] == row["regularization"] == method["regularization"]
            if row["mode"].startswith("reference"):
                old = Path(info["original_folder"])
                filename = "initial_joint.fits" if row["mode"] == "reference_initial" else "result.fits"
                with fits.open(old / filename) as previous:
                    for field in ("MODEL_MACRO", "MODEL", "RESIDUAL", "SOURCE"):
                        np.testing.assert_allclose(result[field].data, previous[field].data, rtol=1e-10, atol=1e-11)
            # Independently rasterize stored source coefficients on the comparison frame.
            nodes = result["SOURCE_NODES"].data
            points = np.column_stack((nodes["X"], nodes["Y"]))
            axis = np.linspace(-.65, .65, 201); sx, sy = np.meshgrid(axis, axis)
            with fits.open(folder / "source_comparison.fits") as source:
                for field, coefficients in (("MACRO_SOURCE", "BASE_FLUX"), ("HALO_SOURCE", "FLUX")):
                    sampled = LinearNDInterpolator(points, nodes[coefficients], fill_value=0.)(sx, sy)
                    np.testing.assert_allclose(sampled, source[field].data, rtol=1e-10, atol=1e-11)
        if row["mode"] in ("fixed", "repeat_initial"):
            with fits.open(folder / "round_1.fits") as first, fits.open(folder / "round_2.fits") as second:
                np.testing.assert_array_equal(first["SOURCE_NODES"].data["X"], second["SOURCE_NODES"].data["X"])
                np.testing.assert_array_equal(first["SOURCE_NODES"].data["Y"], second["SOURCE_NODES"].data["Y"])
    # The first-round basis/prediction must match in every fixed-vs-updated pair;
    # only the second-round vertex update separates these experiments.
    for n in (500, 1000, 2000):
        with fits.open(OUTPUT / "methods" / f"fixed_{n}_lambda2/round_1.fits") as fixed, \
             fits.open(OUTPUT / "methods" / f"updated_{n}_lambda2/round_1.fits") as updated:
            for field in ("X", "Y"):
                np.testing.assert_array_equal(fixed["SOURCE_NODES"].data[field], updated["SOURCE_NODES"].data[field])
            np.testing.assert_allclose(fixed["MODEL"].data, updated["MODEL"].data, rtol=1e-10, atol=1e-11)
    with fits.open(OUTPUT / "methods/fixed_1000_lambda2/round_1.fits") as base:
        for name in ("fixed_1000_lambda0p2", "fixed_1000_lambda20"):
            with fits.open(OUTPUT / "methods" / name / "round_1.fits") as changed:
                for field in ("X", "Y"):
                    np.testing.assert_array_equal(base["SOURCE_NODES"].data[field], changed["SOURCE_NODES"].data[field])
    files = sorted(OUTPUT.glob("datasets/*/*.fits"))+sorted(OUTPUT.glob("methods/*/*.fits"))
    for path in files:
        with fits.open(path, checksum=True) as hdus:
            hdus.verify("exception")
            assert all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in hdus), path
    for directory in ("complex_source", "strength_scan", "psf_scan"):
        previous = ROOT / directory
        hashes = json.loads((previous / "verification.json").read_text())["fits_sha256"]
        for name, value in hashes.items(): assert digest(previous / name) == value
    workspace = ROOT.parents[1]
    for name, value in json.loads((OUTPUT / "provenance.json").read_text())["source_sha256"].items():
        if name.startswith("sublens/"): assert digest(workspace / name) == value
    report = dict(all_checks_passed=True, methods=len(records), datasets=len(datasets), fits_files=len(files),
                  matched_rho_pairs_verified=True, independent_noise_metrics_verified=True,
                  paired_first_round_meshes_and_predictions_identical=True,
                  fixed_meshes_remain_fixed=True, cached_references_reproduced=True,
                  earlier_results_preserved=True,
                  fits_sha256={str(p.relative_to(OUTPUT)): digest(p) for p in files},
                  analysis_sources={p.name: digest(p) for p in (ROOT / "spiral_source_diagnosis.py", Path(__file__))})
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("all_checks_passed", "methods", "datasets", "fits_files")}, indent=2))


if __name__ == "__main__": main()
