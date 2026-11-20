"""Validate PSF normalization, matched forward data, paired noise and results."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve


ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/"psf_scan"


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial",action="store_true",help="Validate completed trials while the scan runs")
    args=parser.parse_args()
    manifest=json.loads((OUTPUT/"manifest.json").read_text())
    summary=json.loads((OUTPUT/"summary.json").read_text())
    assert len(manifest["cases"])==48
    assert summary["planned"]==12*len(summary["noise_seeds"])
    selected_ids=set(summary["selected_case_ids"])
    expected_ids={c["id"] for c in manifest["cases"] if c["noise_seed"] in summary["noise_seeds"]}
    assert selected_ids==expected_ids and len(selected_ids)==summary["planned"]
    actual_ids=[r["id"] for r in summary["trials"]]
    assert len(actual_ids)==len(set(actual_ids))==summary["completed"]
    assert set(actual_ids)<=selected_ids
    assert 0<summary["completed"]<=summary["planned"]
    if not args.allow_partial:assert summary["completed"]==summary["planned"]
    cases={c["id"]:c for c in manifest["cases"]}
    native0=fits.getdata(OUTPUT/"reference"/"native_macro.fits")
    native1=fits.getdata(OUTPUT/"reference"/"native_with_halo.fits")
    files=[];checks=[]
    for shape in manifest["profiles"]:
        key=f"{shape['profile']}_fwhm_{shape['fwhm_arcsec']:.3f}"
        path=OUTPUT/"psfs"/f"{key}.fits"
        kernel=fits.getdata(path)
        assert np.isfinite(kernel).all() and (kernel>=0).all()
        np.testing.assert_allclose(kernel.sum(),1.,rtol=0,atol=1e-12)
        np.testing.assert_allclose(kernel,kernel[::-1,::-1],rtol=1e-7,atol=1e-11)
        m=shape["psf_metadata"]
        assert .99 < m["estimated_flux_before_stamp_renormalization"] < 1.01
        np.testing.assert_allclose((m["fwhm_x_arcsec"]+m["fwhm_y_arcsec"])/2,shape["fwhm_arcsec"],rtol=.003)
        for suffix,native in (("macro",native0),("halo",native1)):
            clean=fits.getdata(OUTPUT/"reference"/f"{key}_clean_{suffix}.fits")
            np.testing.assert_allclose(clean,fftconvolve(native,kernel,mode="same"),rtol=1e-11,atol=1e-13)
    for record in summary["trials"]:
        case=cases[record["id"]]
        folder=OUTPUT/"trials"/record["id"]
        metadata=json.loads((folder/"result.json").read_text())
        with fits.open(folder/"observation.fits") as obs,fits.open(folder/"result.fits") as result:
            for name in ("SCI","RMS","MASK","BETA_X","BETA_Y","THETA_X","THETA_Y"):
                np.testing.assert_array_equal(obs[name].data,result[name].data)
            np.testing.assert_allclose(obs["PSF"].data,result["PSF"].data,rtol=1e-13,atol=1e-16)
            kernel=fits.getdata(OUTPUT/"psfs"/f"{case['psf_key']}.fits")
            np.testing.assert_allclose(kernel,obs["PSF"].data,rtol=1e-13,atol=1e-16)
            clean=fits.getdata(OUTPUT/"reference"/f"{case['psf_key']}_clean_halo.fits")
            z=np.random.default_rng(case["noise_seed"]).standard_normal(clean.shape)
            np.testing.assert_allclose(obs["RMS"].data,.012,rtol=1e-12)
            np.testing.assert_allclose(obs["SCI"].data,clean+.012*z,rtol=0,atol=1e-12)
            assert digest(folder/"observation.fits")==record["observation_sha256"]
            assert json.loads(result["METADATA"].data["JSON"][0])==metadata
            assert metadata["config"]["psf_guard_max_radius_arcsec"]==1.5*case["fwhm_arcsec"]
            assert int(result["PINN_MASK"].data.sum())==record["pinn_scoring_pixels"]==case["coverage_audit"]["capped_cross_fit"]["pixels"]
            mask=obs["MASK"].data.astype(bool)
            for model,residual in (("MODEL","RESIDUAL"),("MODEL_MACRO","RESID_MACRO")):
                np.testing.assert_allclose(obs["SCI"].data-result[model].data,result[residual].data,rtol=0,atol=1e-12)
            best=next((c for c in metadata["candidates"] if c["is_best"]),None)
            if best is not None:
                score=np.mean((result["RESIDUAL"].data[mask]/obs["RMS"].data[mask])**2)
                np.testing.assert_allclose(score,record["with_subhalo"]["chi2_per_pixel"],rtol=1e-12)
                assert best["position_arcsec"]==record["position_arcsec"]
                expected=bool(record["macro_converged"] and record["halo_converged"]
                              and record["position_error_mas"]<=50 and abs(record["strength_error_fraction"])<=.30)
                assert expected==record["recovered"]
            else:assert not record["recovered"]
        checks.append(dict(case=record["id"],paired_noise=True,matched_psf=True,residuals_consistent=True,
                           coverage_consistent=True,metadata_consistent=True))
        files.extend((folder/"observation.fits",folder/"result.fits"))
    files+=list((OUTPUT/"psfs").glob("*.fits"))+list((OUTPUT/"reference").glob("*.fits"))
    for path in files:
        with fits.open(path,checksum=True) as hdus:
            hdus.verify("exception")
            assert all(h.verify_checksum()==1 and h.verify_datasum()==1 for h in hdus),path
    old=json.loads((ROOT/"noise_scan"/"verification.json").read_text())
    for name,value in old["fits_sha256"].items():assert digest(ROOT/"noise_scan"/name)==value
    earlier=json.loads((ROOT/"candidate_refinement"/"verification.json").read_text())
    for item in earlier["fits_products"]:assert digest(ROOT/"candidate_refinement"/item["path"])==item["sha256"]
    workspace=ROOT.parents[1]
    sources=[workspace/p for p in ("sublens/simulation/psf.py","sublens/simulation/simulator.py",
        "sublens/inversion/source_plane.py","sublens/inversion/potential.py","sublens/inversion/halo_search.py",
        "sublens/inversion/joint_refinement.py","sublens/inversion/adaptive_refinement.py",
        "sublens_testing/one_subhalo/psf_scan.py","sublens_testing/one_subhalo/verify_psf_scan.py")]
    report=dict(all_checks_passed=True,cases=len(checks),planned_cases=summary["planned"],
                noise_seeds=summary["noise_seeds"],full_grid_cases=48,
                experiment_complete=len(checks)==summary["planned"],fits_files=len(files),checks=checks,
                original_noise_and_candidate_results_preserved=True,
                fits_sha256={str(p.relative_to(OUTPUT)):digest(p) for p in files},
                source_sha256={str(p.relative_to(workspace)):digest(p) for p in sources},
                environment=dict(python=platform.python_version(),platform=platform.platform(),
                    packages={p:importlib.metadata.version(p) for p in ("numpy","scipy","torch","astropy","matplotlib")}))
    filename="verification_partial.json" if len(checks)<summary["planned"] else "verification.json"
    (OUTPUT/filename).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("all_checks_passed","cases","fits_files")},indent=2))


if __name__=="__main__":main()
