"""Verify saved one-subhalo inputs, predictions, residuals and metadata.

Run after run.py --case all. This checks artifact consistency and does not apply
a detection threshold or replace the library's regression tests.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from astropy.io import fits
import numpy as np


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    products = []
    for path in sorted((ROOT / "data").glob("*.fits")) + [
        ROOT / case / "result.fits" for case in ("improved", "repeat", "null")
    ]:
        with fits.open(path, checksum=True) as hdus:
            hdus.verify("exception")
            assert all(hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1 for hdu in hdus), path
            products.append(dict(path=str(path.relative_to(ROOT)), hdus=len(hdus), checksum_ok=True,
                                 sha256=digest(path)))

    cases = {}
    for case, data_name in (("improved", "observation"), ("repeat", "repeat_observation"),
                            ("null", "null_observation")):
        folder = ROOT / case
        metrics = json.loads((folder / "metrics.json").read_text())
        metadata = json.loads((folder / "result.json").read_text())
        with fits.open(folder / "result.fits") as result, fits.open(ROOT / "data" / f"{data_name}.fits") as obs:
            assert json.loads(result["METADATA"].data["JSON"][0]) == metadata
            for name in ("SCI", "RMS", "PSF", "MASK", "BETA_X", "BETA_Y", "THETA_X", "THETA_Y"):
                np.testing.assert_array_equal(result[name].data, obs[name].data)
            image, rms = result["SCI"].data, result["RMS"].data
            mask = result["MASK"].data.astype(bool)
            for model, residual in (("MODEL_MACRO", "RESID_MACRO"), ("MODEL", "RESIDUAL")):
                np.testing.assert_allclose(image-result[model].data, result[residual].data, rtol=0, atol=1e-12)
            best = next(row for row in metadata["candidates"] if row["is_best"])
            chi2_per_pixel = float(np.mean((result["RESIDUAL"].data[mask]/rms[mask])**2))
            np.testing.assert_allclose(chi2_per_pixel, best["chi2_per_pixel"], rtol=1e-12)
            np.testing.assert_allclose(chi2_per_pixel, metrics["with_subhalo"]["chi2_per_pixel"], rtol=1e-12)
            assert metadata["fit_pixel_count"] == int(mask.sum())
            nodes = result["SOURCE_NODES"].data
            for name in nodes.names:
                assert np.isfinite(nodes[name]).all()
                if "FLUX" in name:
                    assert (nodes[name] >= 0).all()
            assert len(nodes) == metadata["baseline"]["source_node_count"]
            assert metadata["baseline"]["mesh_strategy"] == "centroidal"
            cases[case] = dict(fitted_pixels=int(mask.sum()), chi2_per_pixel=chi2_per_pixel,
                               halo_converged=best["converged"], macro_converged=metadata["macro_converged"],
                               residual_identity_ok=True, observation_preserved=True,
                               metadata_consistent=True, shared_nonnegative_source_basis=True)

    old_observation = ROOT.parent / "pinn_subhalo_results" / "simulation_injected.fits"
    original_unchanged = None
    if old_observation.exists():
        original_unchanged = digest(old_observation) == digest(ROOT / "data" / "observation.fits")
        assert original_unchanged
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "astropy", "matplotlib")}
    environment = dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                       inference_device="cpu", torch_threads=1,
                       run_command="MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sublens_testing/one_subhalo/run.py --case all --force")
    (ROOT / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    workspace = ROOT.parents[1]
    code_paths = [ROOT / "run.py", ROOT / "verify.py"] + [
        workspace / "sublens" / path for path in (
            "inversion/adaptive_refinement.py", "inversion/source_plane.py", "inversion/potential.py",
            "inversion/halo_search.py", "inversion/joint_refinement.py", "result_io.py",
        )
    ]
    report = dict(all_artifact_checks_passed=True, original_observation_unchanged=original_unchanged,
                  cases=cases, fits_products=products,
                  source_sha256={str(path.relative_to(workspace)): digest(path) for path in code_paths})
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(all_artifact_checks_passed=True, cases=cases), indent=2))


if __name__ == "__main__":
    main()
