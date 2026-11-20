"""Euclid Q1 Grade A real-data macro/source comparison and PINN screening."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "one_subhalo"))
from run import SEARCH, ADAPT, save_result
from strength_scan import write_json, digest
from sublens import LensingObservation
from sublens.profiles.macro import epl_deflection
from sublens.inversion.halo_search import _delaunay_native_image
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.survey.euclid import get_available_grade_a_lenses, centered_crop
from sublens.survey.euclid_source import load_euclid_source_observation


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "archive/data"
OUTPUT = ROOT / "sublens_testing/euclid_grade_a_10"


def macro_from_release(mass):
    """Convert the released PyAutoLens SIE normalization to the EPL convention."""
    e = np.hypot(mass["ell_comps_0_max_lh"], mass["ell_comps_1_max_lh"])
    q = (1-e)/(1+e)
    g1, g2 = mass["shear_gamma_1_max_lh"], mass["shear_gamma_2_max_lh"]
    return dict(theta_E=float(mass["einstein_radius_max_lh"]*2*np.sqrt(q)/(1+q)), q=float(q),
                phi=float(.5*np.arctan2(mass["ell_comps_0_max_lh"], mass["ell_comps_1_max_lh"])),
                gamma=2., gamma_ext=float(np.hypot(g1, g2)), phi_ext=float(.5*np.arctan2(g2, g1)),
                center=[mass["centre_1_max_lh"], mass["centre_0_max_lh"]])


def effective_sie_radius(macro, angles=16384):
    """Area-equivalent tangential critical radius for SIE+constant shear.

    Exact radial critical-curve expression for this scale-free isothermal model;
    compact-halo perturbations are not included. Angular integration is periodic.
    """
    a = np.arange(angles)*2*np.pi/angles
    q, phi = macro["q"], macro["phi"]
    g1 = macro["gamma_ext"]*np.cos(2*macro["phi_ext"])
    g2 = macro["gamma_ext"]*np.sin(2*macro["phi_ext"])
    r = macro["theta_E"]*(1-g1*np.cos(2*a)-g2*np.sin(2*a))/(
        (1-g1*g1-g2*g2)*np.sqrt(q*np.cos(a-phi)**2+np.sin(a-phi)**2/q))
    return float(np.sqrt(np.mean(r*r)))


def source_shift(macro):
    """Return Γ·centre, where β_centered = β_origin + Γ·centre."""
    g1 = macro["gamma_ext"]*np.cos(2*macro["phi_ext"])
    g2 = macro["gamma_ext"]*np.sin(2*macro["phi_ext"])
    return np.array([[g1, g2], [g2, -g1]])@macro["center"]


def source_raster(source, macro):
    axis = np.linspace(-1., 1., 201); x, y = np.meshgrid(axis, axis)
    shift = source_shift(macro)
    return source.sample(x+shift[0], y+shift[1])


def similarity(reference, reconstructed):
    a, b = [gaussian_filter(np.maximum(s.astype(float), 0), 2.) for s in (reference, reconstructed)]
    support = a >= .05*a.max()
    def cosine(x, y):
        norm = np.linalg.norm(x)*np.linalg.norm(y)
        return float(x@y/norm) if norm else None
    av, bv = a[support], b[support]
    return dict(cosine_full=cosine(a.ravel(), b.ravel()), cosine_support=cosine(av, bv),
                pearson_support=float(np.corrcoef(av, bv)[0, 1]) if bv.std() > 0 else None,
                support_pixels=int(support.sum()), smoothing_arcsec=.02,
                flux_ratio=float(reconstructed.sum()/reference.sum()),
                no_fitted_alignment_or_flux_rescaling=True)


@dataclass
class MacroFit:
    macro: dict
    source: object
    parameters: np.ndarray
    converged: bool
    attempts: list


class ProfiledMacroFit:
    """Broad seven-parameter SIE+shear optimization with source profiling."""

    def __init__(self, obs, nodes=300, regularization=20.):
        self.obs = obs
        self.problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
            source_nodes=nodes, regularization=regularization, seed=42, mesh_strategy="centroidal")
        x, y = obs.image_rays()
        self.coordinates = torch.tensor(np.column_stack((x.ravel(), y.ravel())), dtype=torch.float64)
        self.ray_shape = x.shape

    @staticmethod
    def vector(macro):
        return np.array([np.log(macro["theta_E"]), macro["q"], macro["phi"],
            macro["center"][0]/.2, macro["center"][1]/.2,
            macro["gamma_ext"]*np.cos(2*macro["phi_ext"])/.2,
            macro["gamma_ext"]*np.sin(2*macro["phi_ext"])/.2])

    @staticmethod
    def model(v):
        g1, g2 = .2*v[5], .2*v[6]
        return dict(theta_E=float(np.exp(v[0])), q=float(v[1]), phi=float((v[2]+np.pi/2)%np.pi-np.pi/2),
                    gamma=2., center=[float(.2*v[3]), float(.2*v[4])],
                    gamma_ext=float(np.hypot(g1, g2)), phi_ext=float(.5*np.arctan2(g2, g1)))

    def rays(self, v):
        center = (.2*v[3], .2*v[4])
        host, _ = epl_deflection(self.coordinates, theta_E=v[0].exp(), q=v[1], phi=v[2], gamma=2., center=center)
        dx, dy = self.coordinates[:, 0]-center[0], self.coordinates[:, 1]-center[1]
        g1, g2 = .2*v[5], .2*v[6]
        shear = torch.stack((g1*dx+g2*dy, g2*dx-g1*dy), dim=-1)
        return (self.coordinates-host-shear).reshape(*self.ray_shape, 2)

    def objective(self, values, warm=None):
        v = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        bx, by = self.rays(v).unbind(dim=-1)
        source = self.problem.fit(bx.detach().numpy(), by.detach().numpy(),
                                  initial=None if warm is None else warm.coefficients)
        residual = (source.image_model[self.obs.mask]-self.obs.image[self.obs.mask])*self.problem.weight
        image_gradient = np.zeros(self.obs.image.shape)
        image_gradient[self.obs.mask] = 2*residual*self.problem.weight
        adjoint = fftconvolve(image_gradient, self.problem.convolution_psf[::-1, ::-1], mode="same")
        native = _delaunay_native_image(bx, by, source.grid, source.coefficients)
        gradient, = torch.autograd.grad((native*torch.from_numpy(adjoint)).sum(), v)
        return source.objective, gradient.numpy(), source

    def fit(self, maxiter=120):
        reference = self.vector(self.obs.macro_model)
        bounds = [(np.log(.2), np.log(4.5)), (.25, .995), (-np.pi, np.pi),
                  (-2.5, 2.5), (-2.5, 2.5), (-1.5, 1.5), (-1.5, 1.5)]
        starts = [reference.copy()]
        for scale, q, dphi in ((.9, .9, .3), (1.1, .7, -.3), (1., .8, .7)):
            v = reference.copy(); v[0] += np.log(scale); v[1] = q; v[2] += dphi; v[5:] = 0.
            starts.append(np.clip(v, np.array(bounds)[:, 0], np.array(bounds)[:, 1]))
        initial = self.problem.fit(self.obs.bx, self.obs.by)
        results, logs = [], []
        for start in starts:
            best, warm = None, initial
            def fun(v):
                nonlocal best
                value, gradient, source = self.objective(v, warm)
                if best is None or value < best[0]: best = (value, v.copy(), source)
                return value, gradient
            outcome = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=bounds,
                options=dict(maxiter=maxiter, maxls=30, ftol=1e-9, gtol=1e-5))
            retried = False
            if not outcome.success:
                warm = best[2]
                outcome = minimize(fun, best[1], jac=True, method="L-BFGS-B", bounds=bounds,
                    options=dict(maxiter=2*maxiter, maxls=40, ftol=1e-10, gtol=1e-5))
                retried = True
            logs.append(dict(start=start.tolist(), objective=float(best[0]), converged=bool(outcome.success),
                             message=str(outcome.message), iterations=int(outcome.nit), retried=retried))
            results.append((*best, bool(outcome.success)))
        minimum = min(r[0] for r in results)
        chosen = min([r for r in results if r[0] <= minimum+1e-7], key=lambda r: (not r[3], r[0]))
        return MacroFit(self.model(chosen[1]), chosen[2], chosen[1], chosen[3], logs)


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    eligible = []
    for row in get_available_grade_a_lenses(DATA / "q1_discovery_engine_lens_catalog.csv", DATA):
        p = Path(row["dir_path"])
        needed = [p / "info.json", p / "mask_extra_galaxies.fits", p / "result/lens_light.fits",
                  p / "result/source_light.fits", p / "result/source_reconstruction.fits"]
        if all(f.exists() for f in needed): eligible.append(row)
    selection = eligible[:10]
    if len(selection) != 10: raise RuntimeError("Ten complete Grade A systems are required")
    write_json(OUTPUT / "selection.json", dict(rule="top 10 expert_score among Grade A with complete local Q1 model products",
               available_complete=len(eligible), lenses=selection, selection_independent_of_sublens_fit=True))
    write_json(OUTPUT / "protocol.json", dict(release="Euclid Q1", band="VIS", crop_size=100, oversample=2,
        mask="published info.json circular aperture plus mask_extra_galaxies.fits and finite/RMS validity",
        foreground="published lens_light.fits VIS subtracted and held fixed", search=SEARCH, adaptive=ADAPT,
        macro_model="SIE+shear", free_macro=["theta_E", "q", "phi", "center_x", "center_y", "shear_1", "shear_2"],
        macro_starts=4, starts_use_released_mass=True, broad_theta_bounds_arcsec=[.2, 4.5],
        later_halo_refinement_macro_free=["theta_E", "shear_1", "shear_2"],
        halo_hypotheses="competing single-halo alternatives, not simultaneous halo detections",
        halo_profile=dict(name="softened truncated deflection", core_arcsec=.035, truncation_arcsec=.22),
        psf_guard="default 98 percent energy; no cap", source_raster_centers_arcsec=[-1., 1.], source_raster_size=201,
        source_raster_origin="historical Q1 exporter; y increases with stored rows", smoothing_source_arcsec=.02,
        candidate_significance_calibrated=False,
        data_source="https://zenodo.org/records/15025832",
        psf_caveat="Publisher: OU-MER VIS PSF may be inadequate for detailed dark-matter substructure science"))
    for row in selection:
        lid = row["id_str"]; folder = OUTPUT / lid; folder.mkdir(exist_ok=True)
        p = Path(row["dir_path"])
        mass = json.loads((p / "result_lens_mass.json").read_text())
        raw = load_euclid_source_observation(lid, DATA, oversample=2)
        macro = macro_from_release(mass)
        obs = LensingObservation.from_arrays(raw.image, raw.rms, raw.psf, pixel_scale=raw.pixel_scale,
            macro_model=macro, mask=raw.mask, native_rays=raw.image_rays(), lens_id=lid)
        if not (folder / "observation.fits").exists(): obs.to_fits(folder / "observation.fits")
        official = centered_crop(fits.getdata(p / "result/source_light.fits", "VIS"), 100).astype(float)
        source = fits.getdata(p / "result/source_reconstruction.fits", "VIS").astype(float)
        hdus = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(official, name="MODEL"), fits.ImageHDU(source, name="SOURCE")])
        if not (folder / "official_reference.fits").exists(): hdus.writeto(folder / "official_reference.fits", checksum=True)
        write_json(folder / "released_mass.json", mass)
        inputs = [p / f"{lid}.fits", p / "info.json", p / "mask_extra_galaxies.fits", p / "result_lens_mass.json",
                  p / "result/lens_light.fits", p / "result/source_light.fits", p / "result/source_reconstruction.fits"]
        write_json(folder / "input_sha256.json", {str(f.relative_to(ROOT)): digest(f) for f in inputs})
    return [r["id_str"] for r in selection]


def run_lens(lid):
    torch.set_num_threads(1); matplotlib.use("Agg")
    folder = OUTPUT / lid
    if (folder / "metrics.json").exists(): return json.loads((folder / "metrics.json").read_text())
    started = perf_counter()
    obs = LensingObservation.from_fits(folder / "observation.fits")
    mass = json.loads((folder / "released_mass.json").read_text())
    if (folder / "macro_fit.json").exists():
        macro_record = json.loads((folder / "macro_fit.json").read_text())
        macro = macro_record["macro"]
    else:
        fit = ProfiledMacroFit(obs).fit()
        macro = fit.macro
        macro_record = dict(macro=macro, converged=fit.converged, attempts=fit.attempts,
                            parameters=fit.parameters.tolist(), objective=fit.source.objective)
        write_json(folder / "macro_fit.json", macro_record)
        print(f"{lid}: broad macro fit theta_circ={macro['theta_E']:.4f}, Reff={effective_sie_radius(macro):.4f}", flush=True)
    fitted = LensingObservation.from_arrays(obs.image, obs.rms, obs.psf, pixel_scale=obs.pixel_scale,
        macro_model=macro, mask=obs.mask, native_rays=obs.image_rays(), lens_id=lid)
    if not (folder / "fitted_macro_observation.fits").exists(): fitted.to_fits(folder / "fitted_macro_observation.fits")
    from run import load_result
    if (folder / "initial_joint.fits").exists(): result = load_result(folder / "initial_joint.fits")
    else:
        result = fitted.search_pinn(**SEARCH)
        result.save_json(folder / "initial_search.json", overwrite=True)
        result = result.refine_joint(macro, maxiter=120, n_starts=2, retry_failed=True)
        result.save_fits(folder / "initial_joint.fits", overwrite=True)
    if result.candidates: result = result.refine_adaptive(macro, **ADAPT)
    result = result.merge_candidates(radius=obs.pixel_scale)
    save_result(result, folder)
    best = result.best_candidate
    macro_null = result.macro_model or macro
    macro_halo = macro_null if best is None else best.fit.macro_model
    source_halo = result.baseline if best is None else best.fit.source
    reference_macro = macro_from_release(mass)
    common = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
        source_grid=result.baseline.grid, regularization=result.baseline.regularization, mesh_strategy="centroidal")
    published_source_refit = common.fit(obs.bx, obs.by)
    with fits.open(folder / "official_reference.fits") as h:
        official_source, official_model = h["SOURCE"].data.copy(), h["MODEL"].data.copy()
    sources = dict(OFFICIAL=official_source, PUBLISHED_MASS=source_raster(published_source_refit, reference_macro),
                   MACRO_ONLY=source_raster(result.baseline, macro_null), MACRO_HALO=source_raster(source_halo, macro_halo))
    source_header = fits.Header()
    for axis in (1, 2):
        source_header[f"CTYPE{axis}"] = "LINEAR"; source_header[f"CUNIT{axis}"] = "arcsec"
        source_header[f"CRPIX{axis}"] = 101.; source_header[f"CRVAL{axis}"] = 0.; source_header[f"CDELT{axis}"] = .01
    fits.HDUList([fits.PrimaryHDU()]+[fits.ImageHDU(a, header=source_header, name=k) for k, a in sources.items()]).writeto(
        folder / "source_comparison.fits", overwrite=True, checksum=True)
    bright = obs.mask & (official_model >= 2*obs.rms)
    def image_score(model):
        return dict(aperture_chi2_per_pixel=float(np.mean(((obs.image-model)[obs.mask]/obs.rms[obs.mask])**2)),
                    bright_arc_chi2_per_pixel=None if not bright.any() else float(np.mean(((obs.image-model)[bright]/obs.rms[bright])**2)))
    row = dict(lens_id=lid, seconds=perf_counter()-started, mask_pixels=int(obs.mask.sum()),
        bright_pixels=int(bright.sum()), macro_initial_converged=macro_record["converged"],
        macro_final_converged=result.macro_converged, macro_input=macro, macro_only=macro_null, macro_with_halo=macro_halo,
        theta_released_internal_arcsec=mass["einstein_radius_max_lh"],
        theta_released_effective_arcsec=mass["einstein_radius_effective_max_lh"],
        theta_released_effective_recomputed_arcsec=effective_sie_radius(reference_macro),
        theta_macro_effective_arcsec=effective_sie_radius(macro_null),
        theta_halo_host_effective_arcsec=effective_sie_radius(macro_halo),
        theta_macro_difference_mas=1000*(effective_sie_radius(macro_null)-mass["einstein_radius_effective_max_lh"]),
        theta_macro_difference_percent=100*(effective_sie_radius(macro_null)/mass["einstein_radius_effective_max_lh"]-1),
        source_similarity={key: similarity(official_source, a) for key, a in sources.items() if key != "OFFICIAL"},
        image_scores=dict(official=image_score(official_model), published_mass=image_score(published_source_refit.image_model),
                          macro=image_score(result.baseline.image_model), halo=image_score(source_halo.image_model)),
        candidate_count=len(result.candidates), pinn_scoring_pixels=int(result.residual_potential.mask.sum()),
        screening_status="no_proposals" if best is None else "uncalibrated_candidate",
        subhalo_presence_confirmed=None, best_candidate=None)
    if best is not None:
        fit = best.fit
        row["best_candidate"] = dict(position_arcsec=list(fit.position), strength_arcsec=fit.strength,
            delta_chi2=fit.delta_chi2, penalized_gain=fit.delta_objective, converged=fit.converged,
            strength_at_upper_bound=fit.strength >= .98*result.max_strength,
            position_at_local_bound=bool(np.max(np.abs(np.array(fit.position)-best.seed)) >= .97*result.refinement_radius))
    write_json(folder / "metrics.json", row)
    plot_lens(row)
    print(f"Finished {lid}: Reff={row['theta_macro_effective_arcsec']:.4f}; "
          f"source cosine={row['source_similarity']['MACRO_ONLY']['cosine_support']:.3f}; "
          f"candidates={row['candidate_count']}; {row['seconds']:.1f}s", flush=True)
    return row


def plot_lens(row):
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    folder = OUTPUT / row["lens_id"]
    with fits.open(folder / "source_comparison.fits") as sources, fits.open(folder / "result.fits") as h:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
        arrays = [sources[k].data for k in ("OFFICIAL", "PUBLISHED_MASS", "MACRO_ONLY", "MACRO_HALO")]
        maximum = max(np.percentile(a, 99.8) for a in arrays)
        for ax, a, title in zip(axes[0], arrays, ("Official source", "SubLens / published mass", "SubLens / fitted macro", "SubLens / macro + halo")):
            shown = ax.imshow(a, origin="lower", extent=(-1.005, 1.005, -1.005, 1.005), cmap="magma",
                              norm=AsinhNorm(linear_width=.03*maximum, vmin=0, vmax=maximum))
            ax.set(title=title, xlabel="βx [arcsec]", ylabel="βy [arcsec]")
        fig.colorbar(shown, ax=axes[0], shrink=.7, ticks=[0, .1*maximum, maximum], format="%.2g")
        mask = h["MASK"].data.astype(bool); image = h["SCI"].data; rms = h["RMS"].data
        with fits.open(folder / "official_reference.fits") as ref: official = ref["MODEL"].data.copy()
        for ax, a, title in zip(axes[1], (image, image-official, h["RESID_MACRO"].data, h["RESIDUAL"].data),
                                ("Foreground-subtracted data", "Official residual", "Macro residual", "Macro + halo residual")):
            if ax is axes[1, 0]:
                ax.imshow(np.where(mask, a, np.nan), origin="lower", extent=(-5, 5, -5, 5), cmap="magma")
            else:
                residual = ax.imshow(np.where(mask, a/rms, np.nan), origin="lower", extent=(-5, 5, -5, 5), cmap="RdBu_r", vmin=-4, vmax=4)
            ax.set(title=title, xlabel="θx [arcsec]", ylabel="θy [arcsec]")
        fig.colorbar(residual, ax=axes[1, 1:], shrink=.7, label="Observed − model / RMS")
        fig.suptitle(row["lens_id"]+f"; Reff official={row['theta_released_effective_arcsec']:.3f}″, SubLens={row['theta_macro_effective_arcsec']:.3f}″; candidates uncalibrated")
        fig.savefig(folder / "comparison.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def summarize(rows):
    import csv
    import matplotlib.pyplot as plt
    ids = [r["id_str"] for r in json.loads((OUTPUT / "selection.json").read_text())["lenses"]]
    rows = sorted(rows, key=lambda r: ids.index(r["lens_id"]))
    differences = np.array([abs(r['theta_macro_difference_percent']) for r in rows])
    cosines = np.array([r['source_similarity']['MACRO_ONLY']['cosine_support'] for r in rows])
    aggregate = {} if not rows else dict(
        median_absolute_theta_difference_percent=float(np.median(differences)),
        maximum_absolute_theta_difference_percent=float(differences.max()),
        theta_within_5_percent=int((differences <= 5).sum()),
        theta_within_10_percent=int((differences <= 10).sum()),
        median_source_cosine=float(np.median(cosines)),
        source_cosine_range=[float(cosines.min()), float(cosines.max())],
        best_halo_strength_upper_bound_count=sum(bool((r['best_candidate'] or {}).get('strength_at_upper_bound')) for r in rows))
    write_json(OUTPUT / "summary.json", dict(completed=len(rows), planned=10, lenses=rows,
               aggregate=aggregate,
               robust_subhalo_detection_claims=False, official_psf_unsuitable_for_definitive_substructure_claims=True))
    table = []
    for r in rows:
        b = r["best_candidate"] or {}
        table.append(dict(lens_id=r["lens_id"], theta_official=r["theta_released_effective_arcsec"],
            theta_macro=r["theta_macro_effective_arcsec"], theta_halo_host=r["theta_halo_host_effective_arcsec"],
            theta_difference_percent=r["theta_macro_difference_percent"],
            source_cosine_published_mass=r["source_similarity"]["PUBLISHED_MASS"]["cosine_support"],
            source_cosine_macro=r["source_similarity"]["MACRO_ONLY"]["cosine_support"],
            source_cosine_halo=r["source_similarity"]["MACRO_HALO"]["cosine_support"],
            chi2_official=r["image_scores"]["official"]["aperture_chi2_per_pixel"],
            chi2_published_mass=r["image_scores"]["published_mass"]["aperture_chi2_per_pixel"],
            chi2_macro=r["image_scores"]["macro"]["aperture_chi2_per_pixel"],
            chi2_halo=r["image_scores"]["halo"]["aperture_chi2_per_pixel"],
            bright_chi2_official=r["image_scores"]["official"]["bright_arc_chi2_per_pixel"],
            bright_chi2_macro=r["image_scores"]["macro"]["bright_arc_chi2_per_pixel"],
            bright_chi2_halo=r["image_scores"]["halo"]["bright_arc_chi2_per_pixel"],
            candidate_x=(b.get('position_arcsec') or [None, None])[0],
            candidate_y=(b.get('position_arcsec') or [None, None])[1],
            candidate_strength=b.get("strength_arcsec"), gain=b.get("penalized_gain"),
            delta_chi2=b.get('delta_chi2'), candidate_strength_at_bound=b.get('strength_at_upper_bound'),
            candidate_converged=b.get("converged"), scoring_pixels=r["pinn_scoring_pixels"]))
    if table:
        with (OUTPUT / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        official = [r["theta_official"] for r in table]; estimated = [r["theta_macro"] for r in table]
        axes[0].scatter(official, estimated); limits=[min(official+estimated)*.9, max(official+estimated)*1.1]
        axes[0].plot(limits, limits, "k:"); axes[0].set(xlabel="Released effective θE [arcsec]", ylabel="SubLens macro effective θE [arcsec]")
        for i, r in enumerate(table): axes[0].annotate(str(i+1), (official[i], estimated[i]))
        x=np.arange(len(table))
        axes[1].bar(x, [r["theta_difference_percent"] for r in table]); axes[1].set(xlabel="Selected lens rank", ylabel="Effective θE difference [%]", xticks=x, xticklabels=x+1)
        for key,label in (("source_cosine_published_mass","Published mass"),("source_cosine_macro","Fitted macro"),("source_cosine_halo","Macro + halo")):
            axes[2].plot(x, [r[key] for r in table], "o-", label=label)
        axes[2].set(xticks=x, xticklabels=x+1, xlabel="Selected lens rank", ylabel="Source cosine on reference support"); axes[2].legend(fontsize=8)
        for ax in axes: ax.grid(alpha=.2)
        fig.savefig(OUTPUT / "theta_source_comparison.png", dpi=175, bbox_inches="tight"); plt.close(fig)
    lines=["# Euclid Q1 Grade A: ten real lenses\n",f"Completed {len(rows)}/10 systems.\n",
        (f"Effective-radius agreement: {aggregate['theta_within_5_percent']}/{len(rows)} within 5%, "
         f"{aggregate['theta_within_10_percent']}/{len(rows)} within 10%; median absolute difference "
         f"{aggregate['median_absolute_theta_difference_percent']:.2f}%. Source cosine median "
         f"{aggregate['median_source_cosine']:.3f}. Best halo strength reaches the upper bound in "
         f"{aggregate['best_halo_strength_upper_bound_count']}/{len(rows)} cases. "
         "The present workflow does not consistently reproduce the released macro/source solutions. "
         "No confirmed subhalo detections or absence claims follow from this experiment.\n") if rows else "",
        "Selection is expert-score order among Grade A systems with complete VIS flux/RMS/PSF, official foreground model, mask and source products. "
        "The published 4-arcsec aperture and extra-galaxy mask are retained, with finite/positive-RMS validity only. "
        "No new annular science mask is substituted. Pixel scale 0.1 arcsec, 2×2 subpixel rays.\n",
        "The foreground MGE model is held fixed at the released solution. Broad seven-parameter SIE+shear macro fitting uses four starts "
        "including the released solution; theta bounds 0.2–4.5 arcsec, q=0.25–0.995, centre coordinates ±0.5 arcsec, shear components ±0.3. "
        "This is a warm-started conditional reanalysis, not a blind independent discovery or a lens-light re-fit. "
        "The latest PINN/candidate/joint/adaptive-source workflow then runs unchanged; its later macro freedom is theta_E and shear.\n",
        "SIE normalization conversion: theta_circ = theta_AutoLens × 2√q/(1+q). Main comparisons use the area-equivalent critical-curve radius, "
        "including macro shear. The halo-model radius describes its smooth macro+shear component, not the total perturbed critical curve. "
        "Released and analytically recomputed reference effective radii are both retained to expose contour discretization differences.\n",
        "Source rasters use the historical Q1 export frame: 201 centres from −1 to +1 arcsec. The known Γ·centre source translation "
        "converts our centered-shear coordinates to the released origin-shear convention. There is no fitted image registration, mirroring or flux rescaling. "
        "Cosine/Pearson metrics are evaluated after common 0.02-arcsec smoothing, including a fixed reference support above 5% of its peak. "
        "Similarity to another fitted model is agreement, not recovery of source truth.\n",
        "The publisher explicitly notes that OU-MER VIS PSFs may be inadequate for detailed dark-matter substructure studies. "
        "Candidates are competing single-halo fits, not simultaneous multiple-halo detections. The phenomenological "
        "halo template retains core=0.035 arcsec and truncation=0.22 arcsec; its angular strength is not a physical mass. "
        "Candidates/gains are uncalibrated; neither presence nor absence of real subhalos is established. "
        "Foreground subtraction, macro/source flexibility, correlated noise and PSF mismatch can mimic perturbations.\n",
        "| Rank | Lens | Official Reff | Macro Reff | Δ [%] | Source cosine | χ²/Npix official | χ²/Npix macro→halo | Candidate b | Penalized gain |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i,r in enumerate(table,1):
        strength="—" if r['candidate_strength'] is None else f"{r['candidate_strength']:.4f}"
        gain="—" if r['gain'] is None else f"{r['gain']:.1f}"
        lines.append(f"| {i} | {r['lens_id']} | {r['theta_official']:.4f} | {r['theta_macro']:.4f} | {r['theta_difference_percent']:+.2f} | {r['source_cosine_macro']:.3f} | {r['chi2_official']:.3f} | {r['chi2_macro']:.3f} → {r['chi2_halo']:.3f} | {strength} | {gain} |")
    lines += ["\nEvery system contains the exact input mask, PSF, RMS and rays in observation.fits; "
        "released inputs and SHA256; macro multi-start diagnostics; result FITS/JSON; source comparison FITS; PINN and residual figures.\n",
        "Sources: [Q1 release](https://zenodo.org/records/15025832), "
        "[publisher description and PSF caveat](https://researchportal.port.ac.uk/en/datasets/euclid-quick-data-release-q1-the-strong-lensing-discovery-engine-2/).\n",
        "Run `.venv/bin/python sublens_testing/euclid_grade_a_10.py --workers 3`; `--summarize-only` redraws.\n"]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    screening_report(rows)


def screening_report(rows):
    """Describe failures and provide a visual index without changing any fit."""
    import html
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    diagnostics = []
    for i, r in enumerate(rows, 1):
        candidate = r['best_candidate'] or {}
        flags = []
        if abs(r['theta_macro_difference_percent']) > 10:
            flags.append('effective_radius_disagreement_above_10_percent')
        if r['macro_only']['gamma_ext'] > .2: flags.append('macro_shear_above_0.2')
        if r['macro_only']['q'] < .26: flags.append('macro_axis_ratio_near_lower_bound_0.25')
        if not r['macro_initial_converged'] or not r['macro_final_converged']:
            flags.append('macro_optimizer_not_converged')
        if candidate.get('strength_at_upper_bound'): flags.append('halo_strength_upper_bound')
        if candidate.get('position_at_local_bound'): flags.append('halo_position_local_bound')
        if candidate and not candidate['converged']: flags.append('halo_optimizer_not_converged')
        diagnostics.append(dict(rank=i, lens_id=r['lens_id'], diagnostic_flags=flags,
            candidate_is_detection=False, hypotheses_are_competing_single_halos=True,
            delta_chi2=candidate.get('delta_chi2'), gain=candidate.get('penalized_gain'),
            strength_arcsec=candidate.get('strength_arcsec'), position_arcsec=candidate.get('position_arcsec')))
    write_json(OUTPUT/'screening_diagnostics.json', dict(
        flag_thresholds_are_descriptive_not_significance_tests=True, lenses=diagnostics,
        confirmed_subhalo_detections=0, subhalo_absence_not_established=True))
    if not rows: return
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.9*len(rows)), squeeze=False, layout='constrained')
    for i, r in enumerate(rows):
        with fits.open(OUTPUT/r['lens_id']/'source_comparison.fits') as h:
            arrays = [h[k].data.copy() for k in ('OFFICIAL', 'MACRO_ONLY', 'MACRO_HALO')]
        vmax = max(np.percentile(a, 99.8) for a in arrays)
        for j, a in enumerate(arrays):
            ax = axes[i, j]
            ax.imshow(a, origin='lower', extent=(-1.005, 1.005, -1.005, 1.005), cmap='magma',
                      norm=AsinhNorm(linear_width=.03*vmax, vmin=0, vmax=vmax))
            ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1])
            if i == 0: ax.set_title(('Official source', 'SubLens macro', 'SubLens macro + halo')[j])
            if j == 0: ax.set_ylabel(f"{i+1}. {r['lens_id'].split('_')[0]}\nβy [arcsec]")
            if i == len(rows)-1: ax.set_xlabel('βx [arcsec]')
    fig.savefig(OUTPUT/'source_gallery.png', dpi=140, bbox_inches='tight'); plt.close(fig)
    body = ['<!doctype html><meta charset="utf-8"><title>Euclid Grade A: 10 systems</title>',
        '<style>body{font:16px sans-serif;max-width:1300px;margin:35px auto}img{width:100%}section{margin:50px 0}code{font-size:14px}</style>',
        '<h1>Euclid Q1: ten Grade A lenses</h1>',
        '<p>Published masks and foreground subtraction; four-start SIE+shear fitting followed by PINN screening. '
        'Candidates are uncalibrated competing single-halo fits. No subhalo existence or absence is established.</p>',
        '<p><a href="README.md">Protocol and result table</a> · <a href="comparison.csv">CSV</a> · '
        '<a href="screening_diagnostics.json">Fit diagnostics</a> · <a href="verification.json">Verification</a></p>',
        '<img src="theta_source_comparison.png" alt="Effective Einstein radius and source similarity comparison">']
    for d in diagnostics:
        lid = html.escape(d['lens_id']); flags = ', '.join(d['diagnostic_flags']) or 'none of the listed diagnostic flags'
        body.append(f'<section><h2>{d["rank"]}. {lid}</h2><p>Diagnostics: <code>{flags}</code></p>'
            f'<p><a href="{lid}/metrics.json">Metrics</a> · <a href="{lid}/result.fits">Inference FITS</a> · '
            f'<a href="{lid}/source_comparison.fits">Source FITS</a></p>'
            f'<img src="{lid}/comparison.png" alt="Source and residual comparison">'
            f'<img src="{lid}/pinn_candidates.png" alt="PINN convergence and candidate alternatives"></section>')
    (OUTPUT/'index.html').write_text('\n'.join(body))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--summarize-only",action="store_true"); args=parser.parse_args()
    matplotlib.use("Agg"); torch.set_num_threads(1)
    if args.workers<1: parser.error("workers must be positive")
    if args.summarize_only:
        ids=[r['id_str'] for r in json.loads((OUTPUT/'selection.json').read_text())['lenses']]
        rows=[json.loads((OUTPUT/lid/'metrics.json').read_text()) for lid in ids if (OUTPUT/lid/'metrics.json').exists()]
    else:
        ids=prepare(); rows=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(run_lens,lid) for lid in ids]): rows.append(future.result())
    summarize(rows)


if __name__=="__main__": main()
