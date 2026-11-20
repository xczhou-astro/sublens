"""Latest PINN multi-halo experiment: fit K=0..3, without a count threshold."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
import numpy as np
import torch
from astropy.io import fits
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent / "one_subhalo"))
from run import ROOT as ONE, TRUTH, SEARCH, ADAPT, residual_metrics
from strength_scan import write_json, digest
from sublens import (LensingObservation, LensingSimulator, MacroLensModel, MultiHaloProblem,
                     build_psf_kernel, sersic_profile, truncated_subhalo_deflection)
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.adaptive_refinement import _select_source
from sublens.io import observation_hdus


OUTPUT = Path(__file__).resolve().parent / "multi_subhalo"
CASES = dict(null=[], opposite_arcs=[[-1.28, .61, .05], [1.10, -.75, .04]],
             same_arc=[[-1.28, .61, .05], [-.80, 1.05, .035]],
             three_halos=[[-1.28, .61, .05], [1.10, -.75, .04], [-.80, 1.05, .025]])
NOISE_SEED = 314159
MAX_HALOS = 3
FWHM = .09


def generate():
    OUTPUT.mkdir(exist_ok=True)
    reference = OUTPUT / "reference"; reference.mkdir(exist_ok=True)
    config = dict(cases=list(CASES), max_halos=MAX_HALOS, noise_seed=NOISE_SEED, rms=.012,
                  psf_fwhm_arcsec=FWHM, pixel_scale_arcsec=.05, source="original_sersic",
                  search=SEARCH, adaptive=ADAPT, source_selection_geometry="largest_K_model",
                  shared_source_basis_for_model_comparison=True, macro_free=["theta_E", "shear_1", "shear_2"],
                  macro_shape_slope_center_fixed_at_simulation_values=True,
                  candidate_min_component_separation_arcsec=.05, max_strength_arcsec=.12,
                  count_selection="none; all K=0..3 retained", truth_used_only_for_generation_and_evaluation=True)
    path = OUTPUT / "config.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Existing multi-halo experiment settings differ; preserve them")
    write_json(path, config)
    obs = LensingObservation.from_fits(ONE / "data" / "observation.fits")
    psf, metadata = build_psf_kernel(FWHM, .05, return_metadata=True)
    if not (reference / "psf.fits").exists():
        fits.PrimaryHDU(psf).writeto(reference / "psf.fits", checksum=True)
    write_json(reference / "psf.json", metadata)
    sim = LensingSimulator(N_img=100, pixel_scale=.05, N_src=257, L_src=1.6,
                          psf_kernel=np.ones((1, 1)), device="cpu")
    image_header = fits.getheader(ONE / "data" / "clean_image.fits")
    with torch.no_grad():
        sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
        source = sersic_profile(sx, sy, **TRUTH["source"])
        if not (reference / "source_truth.fits").exists():
            from complex_source_scan import source_header
            fits.PrimaryHDU(source.numpy(), header=source_header()).writeto(reference / "source_truth.fits", checksum=True)
        macro, _ = MacroLensModel(**TRUTH["macro"])(sim.theta_grid)
        tx, ty = sim.theta_grid.unbind(dim=1)
        for name, halos in CASES.items():
            rays = sim.theta_grid-macro
            for x, y, strength in halos:
                ax, ay = truncated_subhalo_deflection(tx, ty, x, y, strength)
                rays = rays-torch.stack((ax, ay), dim=-1)
            native = sim.render(source[None, None], rays).numpy().squeeze().astype(float)
            clean = fftconvolve(native, psf, mode="same")
            for suffix, array in (("native", native), ("clean", clean)):
                path = reference / f"{name}_{suffix}.fits"
                if not path.exists(): fits.PrimaryHDU(array, header=image_header).writeto(path, checksum=True)
            folder = OUTPUT / name; folder.mkdir(exist_ok=True)
            write_json(folder / "truth.json", dict(halos=halos, common_parameters=TRUTH))
            if not (folder / "observation.fits").exists():
                z = np.random.default_rng(NOISE_SEED).standard_normal(clean.shape)
                replace(obs, image=clean+.012*z, rms=np.full(clean.shape, .012), psf=psf,
                        lens_id=name).to_fits(folder / "observation.fits")
    return list(CASES)


def fit_models(obs, folder):
    """Discover candidate locations and fit every K, without halo truth/count input."""
    initial_problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
                                        source_nodes=300, regularization=20., mesh_strategy="centroidal")
    refiner = MultiHaloProblem(obs, problem=initial_problem)
    baseline = initial_problem.fit(obs.bx, obs.by)
    states = [refiner.fit(np.zeros(3), [], initial_source=baseline)]
    anchors, radii, history = [[]], [.35], []
    for number in range(1, MAX_HALOS+1):
        previous = states[-1]
        rays = refiner.rays(torch.from_numpy(previous.parameters)).numpy()
        conditional = replace(obs, bx=rays[..., 0], by=rays[..., 1], macro_model=None)
        proposal = conditional.search_pinn(**SEARCH, gain_radius=.22, psf_guard_max_radius=1.5*FWHM)
        proposal.save_json(folder / f"proposal_{number}.json", overwrite=True)
        fits.HDUList([fits.PrimaryHDU(proposal.residual_potential.convergence),
                      fits.ImageHDU(proposal.residual_potential.mask.astype(np.uint8), name="PINN_MASK")]).writeto(
                          folder / f"proposal_{number}.fits", overwrite=True, checksum=True)
        choices, attempts = [], []
        for candidate in proposal.candidates:
            fit = candidate.fit
            xy = fit.position if fit.strength > 0 else candidate.seed
            # Only avoid seeds already sitting on a fitted component. Competing
            # maxima survive to simultaneous physical refinement.
            if len(previous.halos) and np.linalg.norm(previous.halos[:, :2]-xy, axis=1).min() < .05:
                continue
            new_anchors = anchors[-1]+[list(candidate.seed)]
            initial = np.r_[previous.parameters, xy, max(fit.strength, .001)/.12]
            fitted = refiner.fit(initial, new_anchors, initial_source=previous.source, n_starts=3)
            xy_fit = fitted.halos[:, :2]
            distances = np.linalg.norm(xy_fit[:, None]-xy_fit[None, :], axis=-1)
            distances += np.eye(len(xy_fit))*1e6
            duplicate = bool(distances.min() < .05) if len(xy_fit) > 1 else False
            attempts.append(dict(rank=candidate.rank, seed=list(candidate.seed), objective=fitted.source.objective,
                                 converged=fitted.converged, duplicate_components=duplicate))
            if not duplicate: choices.append((fitted, new_anchors))
        if not choices:
            raise RuntimeError(f"No distinct new proposals for K={number}; retain partial files for diagnosis")
        state, new_anchors = min(choices, key=lambda item: (item[0].source.objective, not item[0].converged))
        states.append(state); anchors.append(new_anchors); radii.append(.35)
        history.append(dict(K=number, trials=attempts))
        write_json(folder / "growth_history.json", history)
        print(f"{obs.lens_id}: grew K={number}; halos={state.halos.tolist()}", flush=True)
    validation = np.zeros(obs.mask.size, bool)
    indices = np.flatnonzero(obs.mask)
    rng = np.random.default_rng(20261118)
    validation[rng.choice(indices, int(.2*len(indices)), replace=False)] = True
    validation = validation.reshape(obs.mask.shape)
    adaptive = []
    for iteration in range(ADAPT["mesh_rounds"]):
        geometry = states[-1]
        rays = refiner.rays(torch.from_numpy(geometry.parameters)).numpy()
        selected, trials = _select_source(obs, rays[..., 0], rays[..., 1], validation,
            ADAPT["source_nodes"], ADAPT["regularizations"], ADAPT["seeds"], "centroidal")
        problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, rays[..., 0], rays[..., 1],
                    source_nodes=selected["source_nodes"], regularization=selected["regularization"],
                    seed=selected["seed"], mesh_strategy="centroidal")
        refiner = MultiHaloProblem(obs, problem=problem)
        updated = []
        for k, state in enumerate(states):
            if k:
                offset = np.max(np.abs(state.halos[:, :2]-np.array(anchors[k])))
                if offset >= .97*radii[k]: radii[k] = min(.7, 1.6*radii[k])
            rays = refiner.rays(torch.from_numpy(state.parameters)).numpy()
            warm = problem.fit(rays[..., 0], rays[..., 1])
            updated.append(refiner.fit(state.parameters, anchors[k], initial_source=warm,
                                        refinement_radius=radii[k], n_starts=3))
        states = updated
        adaptive.append(dict(round=iteration+1, selected=selected, source_trials=trials,
                             refinement_radii=radii.copy()))
        write_json(folder / "adaptive_history.json", adaptive)
        print(f"{obs.lens_id}: adaptive round {iteration+1}; source={selected}", flush=True)
    return states, anchors, radii, adaptive


def evaluate(halos, truth, converged):
    """One-to-one evaluation only after every model has been fitted."""
    matches = []
    if len(halos) and len(truth):
        truth = np.asarray(truth)
        distances = np.linalg.norm(truth[:, None, :2]-halos[None, :, :2], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        for i, j in zip(rows, cols):
            error = float(1000*distances[i, j]); amplitude = float(halos[j, 2]/truth[i, 2]-1)
            matches.append(dict(truth_index=int(i), component_index=int(j), position_error_mas=error,
                                strength_error_fraction=amplitude,
                                accurate=bool(converged and error <= 50 and abs(amplitude) <= .30)))
    return dict(matches=matches, accurate_truth_count=sum(m["accurate"] for m in matches),
                unmatched_truth_count=len(truth)-sum(m["accurate"] for m in matches),
                extra_fitted_component_count=max(0, len(halos)-len(truth)))


def save_models(obs, states, anchors, radii, folder, adaptive):
    baseline = states[0].source.image_model
    count = int(obs.mask.sum()); axis = np.linspace(-.65, .65, 201)
    rows = []
    for k, state in enumerate(states):
        residual = obs.image-state.source.image_model
        score = residual_metrics(obs, state.source.image_model)
        row = dict(K=k, halos=state.halos.tolist(), macro_model=state.macro_model,
                   converged=state.converged, residual=score, source_objective=state.source.objective,
                   gain_from_macro=float(count*(states[0].source.objective-state.source.objective)),
                   incremental_gain=None if not k else float(count*(states[k-1].source.objective-state.source.objective)),
                   parameters=state.parameters.tolist(), anchors=anchors[k], refinement_radius_arcsec=radii[k],
                   regularization=state.source.regularization, source_nodes=len(state.source.grid.points),
                   optimizer_attempts=state.attempts, source_hyperparameters=adaptive[-1]["selected"],
                   count_is_hypothesis_not_detection=True)
        rows.append(row)
        hdus = observation_hdus(obs)
        for name, array in (("MODEL_MACRO", baseline), ("RESID_MACRO", obs.image-baseline),
                            ("MODEL", state.source.image_model), ("RESIDUAL", residual)):
            hdus.append(fits.ImageHDU(array, name=name))
        raster = fits.ImageHDU(state.source.sample(*np.meshgrid(axis, axis)), name="SOURCE")
        for n in (1, 2):
            raster.header[f"CTYPE{n}"] = "LINEAR"; raster.header[f"CUNIT{n}"] = "arcsec"
            raster.header[f"CRPIX{n}"] = 101.; raster.header[f"CRVAL{n}"] = 0.; raster.header[f"CDELT{n}"] = .0065
        hdus.append(raster)
        nodes = state.source.grid.points
        hdus.append(fits.BinTableHDU.from_columns([
            fits.Column(name="X", format="D", array=nodes[:, 0]), fits.Column(name="Y", format="D", array=nodes[:, 1]),
            fits.Column(name="FLUX", format="D", array=state.source.coefficients),
            fits.Column(name="NULL_FLUX", format="D", array=states[0].source.coefficients)], name="SOURCE_NODES"))
        hdus.append(fits.ImageHDU(state.source.grid.simplices, name="SOURCE_TRI"))
        hdus.append(fits.BinTableHDU.from_columns([fits.Column(name=name, format="D", unit="arcsec", array=state.halos[:, i])
                     for i, name in enumerate(("X", "Y", "STRENGTH"))], name="HALOS"))
        payload = json.dumps(row, allow_nan=False)
        hdus.append(fits.BinTableHDU.from_columns([fits.Column(name="JSON", format=f"{len(payload)}A", array=[payload])], name="METADATA"))
        hdus.writeto(folder / f"model_{k}.fits", overwrite=True, checksum=True)
        write_json(folder / f"model_{k}.json", row)
    return rows


def run_case(task):
    name, force = task
    torch.set_num_threads(1); matplotlib.use("Agg")
    folder = OUTPUT / name
    if (folder / "metrics.json").exists() and not force:
        return json.loads((folder / "metrics.json").read_text())
    started = perf_counter()
    obs = LensingObservation.from_fits(folder / "observation.fits")
    states, anchors, radii, adaptive = fit_models(obs, folder)
    rows = save_models(obs, states, anchors, radii, folder, adaptive)
    # The injected count/locations first enter here, after K=0..3 are all fitted.
    truth = json.loads((folder / "truth.json").read_text())["halos"]
    for row, state in zip(rows, states): row["evaluation"] = evaluate(state.halos, truth, state.converged)
    metrics = dict(case=name, truth=truth, models=rows, seconds=perf_counter()-started,
                   observation_sha256=digest(folder / "observation.fits"),
                   true_count_used_only_for_postfit_evaluation=True, inferred_halo_count=None)
    write_json(folder / "metrics.json", metrics)
    return metrics


def summarize(records):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, AsinhNorm
    from scipy.ndimage import map_coordinates

    records = sorted(records, key=lambda r: list(CASES).index(r["case"]))
    write_json(OUTPUT / "summary.json", dict(completed=len(records), planned=4, cases=records,
               halo_count_not_selected=True))
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    matrix = np.full((4, 4), np.nan)
    for record in records:
        iy = list(CASES).index(record["case"])
        for row in record["models"]:
            k = row["K"]; e = row["evaluation"]; n = len(record["truth"])
            matrix[iy, k] = 2 if not n else int(e["accurate_truth_count"] == n and row["converged"])
            label = f"χ²/N={row['residual']['chi2_per_pixel']:.3f}"
            label += f"\ngain={row['incremental_gain']:.1f}" if k else "\nMacro only"
            if n: label += f"\nAccurate {e['accurate_truth_count']}/{n}; extra slots {e['extra_fitted_component_count']}"
            ax.text(k, iy, label, ha="center", va="center", fontsize=10)
    ax.imshow(matrix, cmap=ListedColormap(["#f4b79c", "#a9d7c4", "#d0d0d0"]), vmin=0, vmax=2, aspect="auto")
    ax.set(xticks=range(4), xticklabels=[f"Fit K={k}" for k in range(4)], yticks=range(4), yticklabels=list(CASES),
           title="PINN sequential proposals + simultaneous halo/macro/source fits\nGreen: all injected halos accurate; K is a hypothesis, not an inferred count")
    fig.savefig(OUTPUT / "multi_recovery_matrix.png", dpi=175, bbox_inches="tight")
    fig.savefig(OUTPUT / "multi_recovery_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    for record in records:
        folder = OUTPUT / record["case"]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
        for k, row in enumerate(record["models"]):
            with fits.open(folder / f"model_{k}.fits") as hdus:
                image = hdus["SCI"].data; mask = hdus["MASK"].data.astype(bool)
                axes[0, k].imshow(image, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5), cmap="magma")
                for x, y, _ in record["truth"]: axes[0, k].scatter(x, y, marker="*", s=100, facecolors="none", edgecolors="cyan")
                for x, y, _ in row["halos"]: axes[0, k].scatter(x, y, marker="+", s=80, color="lime")
                residual = np.where(mask, hdus["RESIDUAL"].data/hdus["RMS"].data, np.nan)
                shown = axes[1, k].imshow(residual, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5), cmap="RdBu_r", vmin=-4, vmax=4)
                axes[0, k].set_title(f"K={k}: cyan truth / green fit")
                axes[1, k].set_title(f"Residual χ²/Npix={row['residual']['chi2_per_pixel']:.3f}")
                for ax in axes[:, k]: ax.set(xlabel="θx [arcsec]", ylabel="θy [arcsec]")
        fig.colorbar(shown, ax=axes[1], shrink=.75, label="Observed − model / RMS")
        fig.suptitle(record["case"]+": common final source basis and regularization")
        fig.savefig(folder / "model_comparison.png", dpi=175, bbox_inches="tight")
        plt.close(fig)
        axis = np.linspace(-.65, .65, 201); sx, sy = np.meshgrid(axis, axis)
        truth_source = fits.getdata(OUTPUT / "reference/source_truth.fits")
        truth_source = map_coordinates(truth_source, [(sy+1.6)/.0125, (sx+1.6)/.0125], order=1)
        arrays = [truth_source]+[fits.getdata(folder / f"model_{k}.fits", "SOURCE") for k in range(4)]
        maximum = max(np.max(a) for a in arrays)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), layout="constrained")
        for ax, array, title in zip(axes, arrays, ["Truth"]+[f"K={k} source" for k in range(4)]):
            shown = ax.imshow(array, origin="lower", extent=(-.65325, .65325, -.65325, .65325),
                              cmap="magma", norm=AsinhNorm(linear_width=.03*maximum, vmin=0, vmax=maximum))
            ax.set(title=title, xlabel="βx [arcsec]", ylabel="βy [arcsec]")
        fig.colorbar(shown, ax=axes, shrink=.7, ticks=[0, .1*maximum, maximum], format="%.2g", label="Surface brightness")
        fig.suptitle(record["case"]+": source truth and common-basis reconstructions")
        fig.savefig(folder / "source_comparison.png", dpi=175, bbox_inches="tight")
        plt.close(fig)
    text = ["# Simultaneous multi-halo test with latest source adaptation\n",
            "Sérsic source, Gaussian pixel-integrated FWHM=0.09 arcsec, RMS=0.012, pixel=0.05 arcsec, "
            "paired noise seed=314159. Cases: null, opposite-arcs pair, same-arc pair, three halos.\n",
            "All cases fit K=0,1,2,3 without injected locations or counts. PINN proposes a new component "
            "conditional on the currently fitted complete lens. All components and theta_E/shear are optimized "
            "simultaneously, with the source re-solved at every step. Two conditional source-validation rounds "
            "use 200/350/500 nodes, lambda=2/20/200, four mesh seeds. The largest-K geometry selects the common "
            "basis; all K are refitted on it, so final incremental objectives share vertices and regularization.\n",
            "No halo-count decision or existence threshold is applied. Truth-count-conditioned recovery is an "
            "evaluation of parameter recovery; it is not a demonstration of blind count selection. "
            "Incremental penalized gain is not sigma significance. Macro shape, slope, center are fixed at simulation values.\n",
            "| Case | True N | Fit K | Accurate matches | Extra parameter slots | χ²/Npix | Incremental gain |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        for row in record["models"]:
            e = row["evaluation"]; gain = "—" if row["incremental_gain"] is None else f"{row['incremental_gain']:.2f}"
            text.append(f"| {record['case']} | {len(record['truth'])} | {row['K']} | {e['accurate_truth_count']} | {e['extra_fitted_component_count']} | {row['residual']['chi2_per_pixel']:.4f} | {gain} |")
    text += ["\nEach case stores observation.fits; proposal_i JSON/FITS; growth/adaptive histories; "
             "model_0..3 FITS/JSON with sources, macro and multi-halo residuals; metrics; model_comparison.png. "
             "source_comparison.png shows source truth and all four reconstructions on common coordinates. "
             "Truth is separate in truth.json. SOURCE_NODES and SOURCE_TRI retain the source basis.\n",
             "Validation: tests_report.txt records the active tests/ suite. archive_test_collection.txt records "
             "repository-wide collection failures in archive/legacy (missing herculens_wrapper, numpyro, "
             "models and pipelines packages); those archived suites were not executed.\n",
             "Run: `.venv/bin/python sublens_testing/multi_subhalo_scan.py --workers 4`; "
             "`--summarize-only` redraws; `--force` refits.\n"]
    (OUTPUT / "README.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    matplotlib.use("Agg"); torch.set_num_threads(1)
    if args.summarize_only:
        records = [json.loads((OUTPUT / name / "metrics.json").read_text()) for name in CASES if (OUTPUT / name / "metrics.json").exists()]
    else:
        names = generate(); records = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(run_case, (name, args.force)) for name in names]):
                record = future.result(); records.append(record)
                print(f"Finished {record['case']} in {record['seconds']:.1f}s", flush=True)
    summarize(records)


if __name__ == "__main__": main()
