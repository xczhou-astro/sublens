"""Check the noise-scan data, paired random fields, FITS and saved metrics."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from astropy.io import fits
import numpy as np


ROOT = Path(__file__).resolve().parent
SCAN = ROOT / "noise_scan"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SCAN / "manifest.json").read_text())
    extension = SCAN / "extension_manifest.json"
    if extension.exists():
        manifest["cases"] += json.loads(extension.read_text())["cases"]
    summary = json.loads((SCAN / "summary.json").read_text())
    config = manifest["config"]
    assert summary["completed"] == summary["planned"] == len(manifest["cases"])
    assert not any(g["inference_failed"] for g in summary["groups"])
    reference = SCAN / "reference"
    clean0 = fits.getdata(reference / "clean_macro.fits")
    clean1 = fits.getdata(reference / "clean_with_subhalo.fits")
    delta = fits.getdata(reference / "delta_image.fits")
    np.testing.assert_array_equal(clean1-clean0, delta)
    saved = json.loads((reference / "signal_metrics.json").read_text())
    normalized_fields, files, checks = {}, [], []
    for case in manifest["cases"]:
        folder = SCAN / "trials" / case["id"]
        inference = json.loads((folder / "inference.json").read_text())
        metadata = json.loads((folder / "result.json").read_text())
        evaluation = json.loads((folder / "evaluation.json").read_text())
        assert digest(folder / "observation.fits") == inference["observation_sha256"]
        with fits.open(folder / "observation.fits") as obs, fits.open(folder / "result.fits") as result:
            assert json.loads(result["METADATA"].data["JSON"][0]) == metadata
            for name in ("SCI", "RMS", "PSF", "MASK", "BETA_X", "BETA_Y", "THETA_X", "THETA_Y"):
                np.testing.assert_array_equal(obs[name].data, result[name].data)
            rms = obs["RMS"].data
            mask = obs["MASK"].data.astype(bool)
            np.testing.assert_allclose(rms, case["noise_rms"], rtol=1e-12)
            field = (obs["SCI"].data-clean1)/rms
            expected = np.random.default_rng(case["seed"]).standard_normal(field.shape)
            np.testing.assert_allclose(field, expected, rtol=0, atol=1e-12)
            if case["seed"] in normalized_fields:
                np.testing.assert_allclose(field, normalized_fields[case["seed"]], rtol=0, atol=1e-12)
            else:
                normalized_fields[case["seed"]] = field.copy()
            rho = float(np.linalg.norm(delta[mask]/rms[mask]))
            np.testing.assert_allclose(rho, case["rho_image"], rtol=1e-12)
            np.testing.assert_allclose(rho*case["noise_rms"], saved["signal_l2"], rtol=1e-12)
            for model, residual in (("MODEL", "RESIDUAL"), ("MODEL_MACRO", "RESID_MACRO")):
                np.testing.assert_allclose(obs["SCI"].data-result[model].data, result[residual].data,
                                           rtol=0, atol=1e-12)
            best = next((r for r in metadata["candidates"] if r["is_best"]), None)
            if best is not None:
                chi2 = float(np.mean((result["RESIDUAL"].data[mask]/rms[mask])**2))
                np.testing.assert_allclose(chi2, best["chi2_per_pixel"], rtol=1e-12)
                np.testing.assert_allclose(chi2, inference["with_subhalo"]["chi2_per_pixel"], rtol=1e-12)
                assert inference["best_rank"] == best["rank"]
                assert evaluation["position_arcsec"] == best["position_arcsec"]
            checks.append(dict(case_id=case["id"], rho_image=rho, observation_preserved=True,
                               residual_identity_ok=True, paired_noise_ok=True, metadata_consistent=True))
        files += [folder / "observation.fits", folder / "result.fits"]
    files += list(reference.glob("*.fits"))
    diagnostic = SCAN / "diagnostic_retry_low_noise_seed161803" / "result.fits"
    if diagnostic.exists():
        with fits.open(diagnostic) as result:
            metadata = json.loads((diagnostic.parent / "result.json").read_text())
            assert json.loads(result["METADATA"].data["JSON"][0]) == metadata
            np.testing.assert_allclose(result["SCI"].data-result["MODEL"].data,
                                       result["RESIDUAL"].data, rtol=0, atol=1e-12)
        files.append(diagnostic)
    for path in files:
        with fits.open(path, checksum=True) as hdus:
            hdus.verify("exception")
            assert all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in hdus), path
    fields = np.stack(list(normalized_fields.values()))
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch", "astropy", "matplotlib")})
    workspace = ROOT.parents[1]
    code_paths = [ROOT / p for p in ("noise_scan.py", "run.py", "verify_noise_scan.py",
                                    "extend_noise_scan.py", "plot_noise_diagnostics.py")] + [
        workspace / "sublens" / p for p in ("inversion/adaptive_refinement.py", "inversion/potential.py",
                                            "inversion/halo_search.py", "inversion/joint_refinement.py",
                                            "inversion/source_plane.py", "result_io.py")]
    report = dict(all_checks_passed=True, checked_trials=len(checks), checked_fits_files=len(files),
                  checks=checks, environment=environment,
                  source_sha256={str(p.relative_to(workspace)): digest(p) for p in code_paths},
                  fits_sha256={str(p.relative_to(SCAN)): digest(p) for p in files},
                  noise_field_correlation=np.corrcoef(fields.reshape(len(fields), -1)).tolist())
    (SCAN / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("all_checks_passed", "checked_trials", "checked_fits_files")}, indent=2))


if __name__ == "__main__":
    main()
