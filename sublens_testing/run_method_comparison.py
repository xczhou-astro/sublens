"""Paired PINN ablation and profiled-grid benchmark with held-out observations.

Truth lives in manifest.json and is consulted only by summarize(). All candidate
generators receive the same observation, search annulus and source basis; they
share physical and joint refinement. This is an internal algorithm comparison,
not a runtime comparison with an external lens-modelling package.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter, process_time

import matplotlib
import numpy as np
import torch

from sublens import LensingObservation, LensingSimulator, MacroLensModel, sersic_profile, truncated_subhalo_deflection
from sublens.inversion.halo_search import (
    PinnHaloCandidate, PinnHaloSearchResult, _prepare_search, _profile_grid, _grid_peaks, _refine,
)
from sublens.inversion.potential import ResidualPotential, propose_residual_peaks
from run_pinn_subhalo import MACRO_PARAMS


METHODS = ("pinn_cross_fit", "pinn_in_sample", "profile_grid")
LABELS = ("PINN cross-fit", "PINN in-sample", "Profile grid")
SETTINGS = dict(
    schema_version=1, image_size=100, pixel_scale=.05, source_size=257, source_half_width=1.6,
    psf_fwhm=.09, noise_rms=.012, source_nodes=350, regularization=.002,
    pinn_steps=90, top_k=3, separation=.44, refinement_radius=.35,
    seed_strength=.015, max_strength=.12, refine_steps=55, joint_maxiter=80,
    grid_size=9, grid_strengths=[.015, .04, .08], fit_annulus=[.5, 1.8],
    search_radius_fractions=[.70, 1.30], macro_radius_fraction=.03, macro_shear_offset=.03,
    source_mesh_seed=42, pinn_seed=42, cpu_threads_per_process=1,
    macro_input_theta_e_fraction=1.01, macro_input_shear_offsets=[.004, -.003],
    positions=[[-1.28, .61], [1.10, -.75], [.15, 1.35]], strengths=[.02, .05],
    null_calibration_count=20, null_test_count=10,
    accurate_position_mas=50., accurate_strength_fraction=.30,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_digest():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *(root / "sublens" / "inversion" / name for name in
              ("potential.py", "halo_search.py", "joint_refinement.py", "source_plane.py"))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_manifest():
    cases = []
    for group, count, first_seed in (("calibration", 20, 61000), ("null_test", 10, 71000)):
        for index in range(count):
            cases.append(dict(id=f"{group}_{index:02d}", group=group, source_kind=index % 2,
                              noise_seed=first_seed + index, truth=None))
    for position in SETTINGS["positions"]:
        for strength in SETTINGS["strengths"]:
            for kind in (0, 1):
                index = sum(c["group"] == "injected_test" for c in cases)
                cases.append(dict(id=f"injected_test_{index:02d}", group="injected_test", source_kind=kind,
                                  noise_seed=51000 + index, truth=[*position, strength]))
    # Pilot observations are disjoint from calibration and final evaluation.
    cases += [dict(id="pilot_null", group="pilot", source_kind=1, noise_seed=81000, truth=None),
              dict(id="pilot_injected", group="pilot", source_kind=0, noise_seed=81001,
                   truth=[-1.28, .61, .05])]
    return cases


def generate_observations(output, manifest):
    sim = LensingSimulator(N_img=SETTINGS["image_size"], pixel_scale=SETTINGS["pixel_scale"],
                          N_src=SETTINGS["source_size"], L_src=SETTINGS["source_half_width"],
                          psf_fwhm=SETTINGS["psf_fwhm"], noise_std=SETTINGS["noise_rms"], device="cpu")
    input_macro = dict(MACRO_PARAMS)
    input_macro["theta_E"] *= SETTINGS["macro_input_theta_e_fraction"]
    angle = 2 * input_macro["phi_ext"]
    g1 = input_macro["gamma_ext"] * np.cos(angle) + SETTINGS["macro_input_shear_offsets"][0]
    g2 = input_macro["gamma_ext"] * np.sin(angle) + SETTINGS["macro_input_shear_offsets"][1]
    input_macro.update(gamma_ext=float(np.hypot(g1, g2)), phi_ext=float(np.arctan2(g2, g1) / 2))
    with torch.no_grad():
        true_alpha, _ = MacroLensModel(**MACRO_PARAMS)(sim.theta_grid)
        input_alpha, _ = MacroLensModel(**input_macro)(sim.theta_grid)
        sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
        smooth = sersic_profile(sx, sy, center=(.04, -.03), i_eff=.30, r_eff=.22,
                                n_sersic=1.25, q=.78, phi=.30)
        clumpy = smooth + .15 * torch.exp(-.5 * (((sx + .06) / .04)**2 + ((sy - .09) / .04)**2))
        clumpy = clumpy + .09 * torch.exp(-.5 * (((sx - .12) / .03)**2 + ((sy + .10) / .06)**2))
        x, y = sim.theta_grid.unbind(dim=1)
        mask = ((x*x + y*y).sqrt().reshape(sim.N_img, sim.N_img) >= .5)
        mask &= ((x*x + y*y).sqrt().reshape(sim.N_img, sim.N_img) <= 1.8)
        for case in manifest:
            path = output / "observations" / f"{case['id']}.fits"
            if path.exists():
                continue
            alpha = true_alpha.clone()
            if case["truth"] is not None:
                ax, ay = truncated_subhalo_deflection(x, y, *case["truth"])
                alpha += torch.stack((ax, ay), dim=-1)
            source = smooth if case["source_kind"] == 0 else clumpy
            image, _ = sim.simulate(source[None, None], alpha, seed=case["noise_seed"])
            observation = LensingObservation.from_simulation(
                sim, image, input_alpha, macro_model=input_macro, mask=mask.cpu().numpy(), lens_id=case["id"],
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            observation.to_fits(path)


def empty_potential(obs):
    missing = np.full(obs.image.shape, np.nan)
    return ResidualPotential(np.zeros(obs.image.shape, dtype=bool), missing, missing.copy(), missing.copy(), [])


def prepare_observation(path):
    """Fit the macro normalization/shear without using the manifest or halo truth."""
    start, cpu = perf_counter(), process_time()
    obs = LensingObservation.from_fits(path)
    _, source, _, _ = _prepare_search(obs, SETTINGS["source_nodes"], SETTINGS["regularization"])
    shell = PinnHaloSearchResult(obs, source, empty_potential(obs), (), (0., 1.))
    prefit = shell.refine_joint(obs.macro_model, maxiter=SETTINGS["joint_maxiter"])
    obs = LensingObservation.from_arrays(
        obs.image, obs.rms, obs.psf, pixel_scale=obs.pixel_scale, macro_model=prefit.macro_model,
        native_rays=obs.image_rays(), mask=obs.mask, lens_id=obs.lens_id,
    )
    problem, baseline, image_rays, rays = _prepare_search(
        obs, SETTINGS["source_nodes"], SETTINGS["regularization"],
    )
    return obs, problem, baseline, image_rays, rays, dict(
        seconds=perf_counter()-start, cpu_seconds=process_time()-cpu,
        input_macro=prefit.config["joint_refinement"]["macro_model_input"],
        fitted_macro=prefit.macro_model, macro_converged=prefit.macro_converged,
    )


def run_method(method, prepared, destination):
    """No injected parameters or source morphology labels are available here."""
    obs, problem, baseline, image_rays, rays, common = prepared
    start, cpu = perf_counter(), process_time()
    radius = tuple(f * obs.einstein_radius for f in SETTINGS["search_radius_fractions"])
    stage_start = perf_counter()
    grid_evaluations = 0
    if method == "profile_grid":
        offsets = np.linspace(-radius[1], radius[1], SETTINGS["grid_size"])
        xs, ys = (offsets + c for c in obs.lens_center)
        scores, _ = _profile_grid(problem, baseline, rays, xs, ys, obs.lens_center,
                                  radius, SETTINGS["grid_strengths"])
        peaks = [(float(xs[i]), float(ys[j]), float(scores[j, i])) for j, i in
                 _grid_peaks(scores, xs, ys, SETTINGS["top_k"], SETTINGS["separation"])]
        grid_evaluations = int(np.isfinite(scores).sum()) * len(SETTINGS["grid_strengths"])
        potential = replace(empty_potential(obs), peaks=peaks)
    else:
        potential = propose_residual_peaks(
            obs, *image_rays, radius=radius, source_nodes=SETTINGS["source_nodes"],
            regularization=SETTINGS["regularization"], separation=SETTINGS["separation"],
            top_k=SETTINGS["top_k"], steps=SETTINGS["pinn_steps"], cross_fit=method == "pinn_cross_fit",
        )
        peaks = potential.peaks
    proposal_seconds = perf_counter() - stage_start
    candidates = []
    stage_start = perf_counter()
    for rank, (x, y, score) in enumerate(peaks, 1):
        half = SETTINGS["refinement_radius"]
        bounds = [(x-half, x+half), (y-half, y+half), (0., 1.)]
        fit = _refine(problem, baseline, rays, [x, y, SETTINGS["seed_strength"] / SETTINGS["max_strength"]],
                      bounds, SETTINGS["max_strength"], True, SETTINGS["refine_steps"])
        candidates.append(PinnHaloCandidate(rank, (x, y), score, fit))
    physical_seconds = perf_counter() - stage_start
    result = PinnHaloSearchResult(
        obs, baseline, potential, tuple(candidates), radius, max_strength=SETTINGS["max_strength"],
        refinement_radius=SETTINGS["refinement_radius"],
        config=dict(benchmark_method=method, benchmark_settings=SETTINGS),
    )
    stage_start = perf_counter()
    joint = result.refine_joint(obs.macro_model, maxiter=SETTINGS["joint_maxiter"])
    joint_seconds = perf_counter() - stage_start
    best = joint.best_candidate
    fit = None if best is None else best.fit
    destination.mkdir(parents=True, exist_ok=True)
    joint.save_json(destination / "result.json", overwrite=True)
    joint.save_fits(destination / "result.fits", resolution=101, source_half_width=.65, overwrite=True)
    record = dict(
        method=method, case_id=obs.lens_id,
        score=0. if fit is None else float(fit.delta_objective),
        delta_chi2=0. if fit is None else float(fit.delta_chi2),
        position=None if fit is None else list(fit.position), strength=0. if fit is None else fit.strength,
        converged=bool(fit is not None and fit.converged and joint.macro_converged),
        halo_converged=None if fit is None else fit.converged, macro_converged=joint.macro_converged,
        candidate_count=len(candidates), seed_positions=[list(c.seed) for c in candidates],
        grid_source_evaluations=grid_evaluations,
        seconds=common["seconds"] + perf_counter()-start,
        cpu_seconds=common["cpu_seconds"] + process_time()-cpu,
        stage_seconds=dict(common=common["seconds"], proposal=proposal_seconds,
                           physical=physical_seconds, joint=joint_seconds),
        prefit_macro=common,
    )
    write_json(destination / "metrics.json", record)
    return record


def run_case(task):
    case_id, root, phase, repeat = task
    torch.set_num_threads(SETTINGS["cpu_threads_per_process"])
    output = Path(root)
    destination = output / phase / case_id
    if phase == "serial_timing":
        destination /= f"repeat_{repeat}"
    cached = []
    for method in METHODS:
        path = destination / method / "metrics.json"
        if path.exists():
            cached.append(json.loads(path.read_text()))
    if len(cached) == len(METHODS):
        return cached
    prepared = prepare_observation(output / "observations" / f"{case_id}.fits")
    offset = (sum(case_id.encode()) + repeat) % len(METHODS)
    order = METHODS[offset:] + METHODS[:offset]
    records = {r["method"]: r for r in cached}
    for method in order:
        if method not in records:
            records[method] = run_method(method, prepared, destination / method)
            r = records[method]
            print(f"{phase} {case_id} {method}: gain={r['score']:.1f}, "
                  f"converged={r['converged']}, seconds={r['seconds']:.1f}", flush=True)
    return [records[m] for m in METHODS]


def detection_score(record):
    return max(0., record["score"]) if record["converged"] else 0.


def error_metrics(record, truth):
    if truth is None or record["position"] is None:
        return None, None
    return float(1000 * np.linalg.norm(np.asarray(record["position"]) - truth[:2])), float(record["strength"] / truth[2] - 1)


def summarize(output, manifest):
    # Scoring is intentionally outside the fitting workers.
    cases = [c for c in manifest if c["group"] != "pilot"]
    records = []
    for case in cases:
        for method in METHODS:
            path = output / "accuracy" / case["id"] / method / "metrics.json"
            r = json.loads(path.read_text())
            position_error, strength_error = error_metrics(r, case["truth"])
            records.append(dict(**r, group=case["group"], source_kind=case["source_kind"],
                                truth=case["truth"], position_error_mas=position_error,
                                strength_error_fraction=strength_error))
    summaries = []
    for method in METHODS:
        rs = [r for r in records if r["method"] == method]
        calibration = [r for r in rs if r["group"] == "calibration"]
        nulls = [r for r in rs if r["group"] == "null_test"]
        injected = [r for r in rs if r["group"] == "injected_test"]
        threshold = max(detection_score(r) for r in calibration)

        def passed(r, cut):
            return detection_score(r) > cut

        def accurate(r, cut):
            return (passed(r, cut) and r["position_error_mas"] <= SETTINGS["accurate_position_mas"]
                    and abs(r["strength_error_fraction"]) <= SETTINGS["accurate_strength_fraction"])

        timing = [json.loads(p.read_text()) for p in
                  (output / "serial_timing").glob(f"*/*/{method}/metrics.json")]
        summaries.append(dict(
            method=method, threshold=threshold, calibration_count=len(calibration),
            null_count=len(nulls), injected_count=len(injected),
            null_accepted=sum(passed(r, threshold) for r in nulls),
            injected_accepted=sum(passed(r, threshold) for r in injected),
            injected_accurate=sum(accurate(r, threshold) for r in injected),
            injected_converged=sum(r["converged"] for r in injected),
            null_accepted_at_25=sum(passed(r, 25.) for r in nulls),
            injected_accurate_at_25=sum(accurate(r, 25.) for r in injected),
            geometric_accurate_ignoring_score_and_convergence=sum(
                r["position_error_mas"] is not None and r["position_error_mas"] <= 50
                and abs(r["strength_error_fraction"]) <= .30 for r in injected),
            source_groups=[dict(source_kind=kind, count=sum(r["source_kind"] == kind for r in injected),
                                accurate=sum(accurate(r, threshold) for r in injected if r["source_kind"] == kind))
                           for kind in (0, 1)],
            serial_timing_count=len(timing),
            median_serial_seconds=float(np.median([r["seconds"] for r in timing])) if timing else None,
            median_serial_stages={k: float(np.median([r["stage_seconds"][k] for r in timing]))
                                  for k in ("common", "proposal", "physical", "joint")} if timing else {},
        ))
    payload = dict(settings=SETTINGS, summaries=summaries, trials=records,
                   limitations=["Small held-out test sample, not a population completeness measurement",
                                "Host shape/slope/center fixed at truth; only normalization and shear fitted",
                                "20 calibration nulls give a coarse threshold reference",
                                "Grid is an internal profiled-search baseline, not an external package benchmark",
                                "No hyperparameter tuning or optimizer repair after viewing test outcomes",
                                "Common macro/source preparation is charged to each method",
                                "Timing excludes simulation, includes source solves and all candidate refinements"])
    write_json(output / "summary.json", payload)
    rows = ["| Method | Null-derived gain threshold | False detections /10 | Accurate recovery /12 | Converged /12 | Serial median [s] |",
            "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        timing_text = "pending" if s["median_serial_seconds"] is None else f"{s['median_serial_seconds']:.1f}"
        rows.append(f"| {s['method']} | {s['threshold']:.2f} | {s['null_accepted']} | "
                    f"{s['injected_accurate']} | {s['injected_converged']} | {timing_text} |")
    report = (
        "# Paired method comparison\n\n"
        "20 calibration nulls, 10 independent test nulls and 12 injected test observations. "
        "Two source morphologies, three halo positions and strengths 0.02/0.05 arcsec. "
        "Initial theta_E is +1% from truth and shear offsets are +0.004/-0.003; "
        "all methods first fit a macro-only normalization/shear, then share physical/joint refinement.\n\n"
        "Acceptance requires final macro and best-halo convergence and a score strictly above the "
        "maximum accepted-score statistic of that method's 20 calibration nulls. "
        "Accurate recovery also requires <=50 mas position error and <=30% strength error. "
        "Truth is used only in generation and final scoring.\n\n"
        + "\n".join(rows) + "\n\n"
        "Accuracy jobs run in parallel; their wall times are not used for efficiency claims. "
        "Runtime summaries use the separate serial reruns. Common preparation is measured once "
        "per observation and charged to each method. No trained networks or ground-truth halo "
        "positions initialize the searches. The profile-grid baseline explicitly refits the source "
        "at every trial; it does not use the frozen-template radar.\n\n"
        + "\n".join("- " + note for note in payload["limitations"]) + "\n"
    )
    (output / "REPORT.md").write_text(report)
    plot_comparison(payload, output)
    print(report, flush=True)


def plot_comparison(payload, output):
    import matplotlib.pyplot as plt

    summaries, records = payload["summaries"], payload["trials"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    colors = ("#2374ab", "#e08025", "#368453")
    for color, method, label, summary in zip(colors, METHODS, LABELS, summaries):
        rs = [r for r in records if r["method"] == method]
        nulls = [r for r in rs if r["group"] == "null_test"]
        injected = [r for r in rs if r["group"] == "injected_test"]
        thresholds = sorted(set([0., *[detection_score(r) for r in nulls + injected]]))
        fpr, recovery = [], []
        for cut in thresholds:
            fpr.append(sum(detection_score(r) > cut for r in nulls) / len(nulls))
            recovery.append(sum(detection_score(r) > cut and r["position_error_mas"] <= 50
                                and abs(r["strength_error_fraction"]) <= .30 for r in injected) / len(injected))
        axes[0].plot(fpr, recovery, ".-", color=color, label=label)
        axes[0].scatter(summary["null_accepted"]/len(nulls), summary["injected_accurate"]/len(injected),
                           s=110, marker="*", color=color, edgecolors="black", zorder=4)
        index = METHODS.index(method)
        axes[1].bar(index, summary["injected_accurate"]/len(injected), color=color)
        if summary["median_serial_seconds"] is not None:
            axes[2].bar(index, summary["median_serial_seconds"], color=color)
    axes[0].set(xlabel="Independent null false-positive fraction", ylabel="Accurate recovery fraction",
                title="Threshold sweep; stars: calibration cuts", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Accurate recoveries /12", ylim=(0, 1), title="At separate null-derived thresholds")
    axes[2].set(ylabel="Median total seconds", title="Separate serial timing reruns")
    for ax in axes[1:]:
        ax.set_xticks(range(3), LABELS, rotation=20, ha="right", fontsize=8)
    fig.savefig(output / "method_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "accuracy", "timing", "report"), default="pilot")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "method_comparison_results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    matplotlib.use("Agg")
    torch.set_num_threads(SETTINGS["cpu_threads_per_process"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(json.dumps(dict(settings=SETTINGS, code=source_digest())))
    path = output / "frozen_config.json"
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError("Configuration/code changed; use a separate output directory")
    write_json(path, configuration)
    manifest = make_manifest()
    write_json(output / "manifest.json", manifest)
    if args.stage == "report":
        summarize(output, manifest)
        return
    generate_observations(output, manifest)
    if args.stage == "pilot":
        selected = [c["id"] for c in manifest if c["group"] == "pilot"]
        tasks = [(case_id, str(output), "pilot", 0) for case_id in selected]
    elif args.stage == "accuracy":
        # Interleave groups to expose progress while keeping the predeclared cases.
        selected = [c["id"] for c in manifest if c["group"] != "pilot"]
        np.random.default_rng(20261118).shuffle(selected)
        tasks = [(case_id, str(output), "accuracy", 0) for case_id in selected]
    else:
        # Three predeclared observations, two serial repeats, balanced method order.
        selected = ["null_test_00", "injected_test_02", "injected_test_05"]
        tasks = [(case_id, str(output), "serial_timing", repeat) for repeat in range(2) for case_id in selected]
    if args.stage == "timing" or args.workers == 1:
        for task in tasks:
            run_case(task)
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_case, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"Completed observations {index}/{len(tasks)}", flush=True)
    if args.stage in ("accuracy", "timing"):
        summarize(output, manifest)


if __name__ == "__main__":
    main()
