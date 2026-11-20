"""Verify paired inputs, physical grouping, FITS and preservation of the old scan."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from astropy.io import fits
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "candidate_refinement"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary_path = OUTPUT / "matrix_summary.json"
    if not summary_path.exists():
        summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    products = []
    checks = []
    for record in summary["cases"]:
        folder = OUTPUT / record["case"]
        original = ROOT / "noise_scan" / "trials" / record["case"]
        assert digest(folder / "observation.fits") == digest(original / "observation.fits") == record["input_sha256"]
        full = json.loads((folder / "unmerged_result.json").read_text())
        merged = json.loads((folder / "result.json").read_text())
        assert merged["best_candidate_rank"] == full["best_candidate_rank"]
        assert merged["baseline"] == full["baseline"]
        best_before = next(c for c in full["candidates"] if c["is_best"])
        best_after = next(c for c in merged["candidates"] if c["is_best"])
        assert best_before == best_after
        by_rank = {c["rank"]:c for c in full["candidates"]}
        groups = merged["config"]["candidate_merge"]
        for group in groups["groups"]:
            representative = by_rank[group["representative_rank"]]
            for member in group["members"]:
                raw = by_rank[member["rank"]]
                assert raw["position_arcsec"] == member["position_arcsec"]
                if len(group["members"]) > 1:
                    assert representative["converged"] and raw["converged"]
                    a, b = representative["strength_arcsec"], raw["strength_arcsec"]
                    assert a > 0 and b > 0
                    assert abs(a-b) <= groups["relative_strength_tolerance"]*max(a,b)
                    assert np.linalg.norm(np.array(raw["position_arcsec"])-representative["position_arcsec"]) <= groups["radius_arcsec"]
                    assert representative["delta_objective"] >= raw["delta_objective"]
        assert sorted(m["rank"] for g in groups["groups"] for m in g["members"]) == sorted(by_rank)
        with fits.open(original / "result.fits") as old, fits.open(folder / "result.fits") as new:
            np.testing.assert_array_equal(old["PINN_KAPPA"].data,new["PINN_KAPPA"].data)
            for name in ("SCI","RMS","MASK","PSF","BETA_X","BETA_Y","THETA_X","THETA_Y"):
                np.testing.assert_array_equal(old[name].data,new[name].data)
            assert json.loads(new["METADATA"].data["JSON"][0]) == merged
            for model,residual in (("MODEL","RESIDUAL"),("MODEL_MACRO","RESID_MACRO")):
                np.testing.assert_allclose(new["SCI"].data-new[model].data,new[residual].data,rtol=0,atol=1e-12)
            mask = new["MASK"].data.astype(bool)
            chi2 = np.mean((new["RESIDUAL"].data[mask]/new["RMS"].data[mask])**2)
            np.testing.assert_allclose(chi2,record["with_subhalo"]["chi2_per_pixel"],rtol=1e-12)
        for name in ("observation.fits","unmerged_result.fits","result.fits"):
            path = folder / name
            with fits.open(path,checksum=True) as hdus:
                assert all(h.verify_checksum()==1 and h.verify_datasum()==1 for h in hdus)
            products.append(dict(path=str(path.relative_to(OUTPUT)),sha256=digest(path)))
        checks.append(dict(case=record["case"],same_input=True,same_neural_field=True,
                           merge_preserves_best=True,merge_members_valid=True,residuals_consistent=True))
    previous = json.loads((ROOT / "noise_scan" / "verification.json").read_text())
    for name, expected in previous["fits_sha256"].items():
        assert digest(ROOT / "noise_scan" / name) == expected
    matrix_path = OUTPUT / "matrix_data.json"
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text())
        records = {r["case"]:r for r in summary["cases"]}
        assert len(matrix["cells"]) == len(records) == 32
        arrays = np.load(OUTPUT / "recovery_matrix.npz")
        for cell in matrix["cells"]:
            record = records[cell["case"]]
            iy = matrix["seeds"].index(cell["seed"])
            ix = matrix["noise_factors"].index(cell["noise_factor"])
            assert cell["position_error_mas"] == record["position_error_mas"] == arrays["position_error_mas"][iy,ix]
            assert cell["strength_error_fraction"] == record["strength_error_fraction"] == arrays["strength_error_fraction"][iy,ix]
            expected = bool(record["converged"] and record["position_error_mas"] <= 50
                            and abs(record["strength_error_fraction"]) <= .30)
            assert expected == cell["recovered"] == arrays["recovered"][iy,ix]
        assert (OUTPUT / "paired_recovery.png").stat().st_size > 10000
        assert (OUTPUT / "paired_recovery.pdf").read_bytes().startswith(b"%PDF")
    workspace = ROOT.parents[1]
    sources = [workspace / p for p in (
        "sublens/inversion/potential.py", "sublens/inversion/halo_search.py",
        "sublens/inversion/joint_refinement.py", "sublens/inversion/adaptive_refinement.py",
        "sublens_testing/one_subhalo/test_competing_candidates.py",
        "sublens_testing/one_subhalo/verify_candidate_refinement.py")]
    report = dict(all_checks_passed=True,checks=checks,fits_products=products,
                  original_noise_scan_fits_unchanged=True,
                  source_sha256={str(p.relative_to(workspace)):digest(p) for p in sources},
                  environment=dict(python=platform.python_version(),platform=platform.platform(),
                      packages={p:importlib.metadata.version(p) for p in ("numpy","scipy","torch","astropy","matplotlib")}))
    (OUTPUT / "verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(all_checks_passed=True,cases=len(checks),fits_files=len(products),
                         original_noise_scan_fits_unchanged=True),indent=2))


if __name__ == "__main__":
    main()
