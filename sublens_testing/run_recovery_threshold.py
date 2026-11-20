"""Measure blind PINN recovery versus injected angular strength and noise seed.

The injected position is used only by the simulator and by scoring after fitting.
Null calibration and validation use disjoint noise seeds. All trials use the same
public search_pinn().refine_joint() path, source mesh settings and candidate budget.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import torch

from sublens import LensingObservation, MacroLensModel, truncated_subhalo_deflection
from run_pinn_subhalo import HALO_POSITION, MACRO_PARAMS, simulate_observations


SCHEMA_VERSION = 1
SIMULATION = dict(N_img=100, pixel_scale=0.05, N_src=257, L_src=1.6,
                  psf_fwhm=0.09, noise_std=0.012)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@lru_cache(maxsize=1)
def setup(device):
    torch.set_num_threads(2)
    sim, source, _, _ = simulate_observations(device)
    with torch.no_grad():
        macro, _ = MacroLensModel(**MACRO_PARAMS).to(sim.device)(sim.theta_grid)
    return sim, source, macro


def score_result(result, injected_strength):
    """Select by the fit objective before consulting truth for evaluation."""
    best = result.best_candidate
    if best is None:
        return dict(score=0.0, delta_chi2=0.0, position_arcsec=None, strength_arcsec=0.0,
                    position_error_mas=None, strength_error_percent=None, converged=False,
                    pinn_seed_error_mas=None)
    fit = best.fit
    position_error = 1000 * np.linalg.norm(np.asarray(fit.position) - HALO_POSITION)
    seed_error = min(1000 * np.linalg.norm(np.asarray(c.seed) - HALO_POSITION)
                     for c in result.candidates)
    return dict(
        score=float(fit.delta_objective), delta_chi2=float(fit.delta_chi2),
        position_arcsec=list(fit.position), strength_arcsec=float(fit.strength),
        position_error_mas=float(position_error) if injected_strength else None,
        strength_error_percent=float(100 * (fit.strength / injected_strength - 1))
        if injected_strength else None,
        converged=bool(fit.converged and result.macro_converged is not False),
        pinn_seed_error_mas=float(seed_error) if injected_strength else None,
    )


def run_trial(task):
    group, strength, seed, config, output_string = task
    sim, source, alpha_macro = setup(config["device"])
    output = Path(output_string) / group / f"strength_{strength:.6f}_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with torch.no_grad():
        x, y = sim.theta_grid.unbind(dim=1)
        ax, ay = truncated_subhalo_deflection(x, y, *HALO_POSITION, strength)
        alpha = alpha_macro + torch.stack((ax, ay), dim=-1)
        image, _ = sim.simulate(source[None, None], alpha, seed=seed)
    obs = LensingObservation.from_simulation(
        sim, image, alpha_macro, macro_model=MACRO_PARAMS,
        lens_id=f"{group}_{strength:.6f}_{seed}",
    )
    x, y = obs.image_rays()
    radius = np.hypot(x[..., 0], y[..., 0])
    obs = replace(obs, mask=(radius >= 0.50) & (radius <= 1.80))
    obs.to_fits(output / "observation.fits", overwrite=True)
    obs = LensingObservation.from_fits(output / "observation.fits")
    fixed = obs.search_pinn()
    joint = fixed.refine_joint(obs.macro_model, maxiter=config["joint_maxiter"])
    fixed.save_json(output / "fixed_macro.json", overwrite=True)
    joint.save_json(output / "joint.json", overwrite=True)
    joint.save_fits(output / "result.fits", overwrite=True)
    record = dict(
        schema_version=SCHEMA_VERSION, config=config, group=group,
        injected_strength_arcsec=strength, noise_seed=seed,
        actual_simulation_device=str(sim.device),
        fixed_macro=score_result(fixed, strength), joint=score_result(joint, strength),
        seconds=perf_counter() - started,
    )
    write_json(output / "metrics.json", record)
    return record


def accepted(record, threshold, mode="joint"):
    fit = record[mode]
    return fit["converged"] and fit["score"] > threshold


def recovery_rates(records, threshold, mode="joint"):
    selected = [r for r in records if accepted(r, threshold, mode)]
    localized = [r for r in selected if r[mode]["position_error_mas"] <= 50.0]
    accurate = [r for r in localized if abs(r[mode]["strength_error_percent"]) <= 30.0]
    position_errors = [r[mode]["position_error_mas"] for r in records
                       if r[mode]["position_error_mas"] is not None]
    strength_errors = [r[mode]["strength_error_percent"] for r in records
                       if r[mode]["strength_error_percent"] is not None]
    return dict(
        trials=len(records), accepted=len(selected), localized=len(localized), accurate=len(accurate),
        converged=sum(r[mode]["converged"] for r in records),
        acceptance_rate=len(selected) / len(records),
        localization_rate=len(localized) / len(records),
        accurate_recovery_rate=len(accurate) / len(records),
        median_position_error_mas=float(np.median(position_errors)) if position_errors else None,
        median_strength_error_percent=float(np.median(strength_errors)) if strength_errors else None,
    )


def summarize(records, output, config):
    calibration = [r for r in records if r["group"] == "null_calibration"]
    validation = [r for r in records if r["group"] == "null_validation"]
    injected = [r for r in records if r["group"] == "injected"]
    # Use the maximum across the full blind search, including unconverged fits.
    # This is a conservative reference, not a measured 5% false-alarm guarantee.
    null_reference = max(r["joint"]["score"] for r in calibration)
    thresholds = sorted(set([10.0, 25.0, 50.0, 100.0, float(null_reference)]))
    groups = {s: [r for r in injected if r["injected_strength_arcsec"] == s]
              for s in config["strengths_arcsec"]}
    grid = []
    for threshold in thresholds:
        grid.append(dict(
            threshold=threshold,
            null_calibration_accepted=sum(accepted(r, threshold) for r in calibration),
            null_validation_accepted=sum(accepted(r, threshold) for r in validation),
            injected=[dict(strength_arcsec=s, **recovery_rates(rs, threshold))
                      for s, rs in groups.items()],
        ))
    summary = dict(
        config=config, completed_trials=len(records),
        detection_statistic="N_fit_pixels * (macro_source_objective - halo_source_objective)",
        detection_rule="best blind candidate converged, macro-only fit converged, score > threshold",
        accurate_recovery_rule="detected AND position error <= 50 mas AND abs(strength error) <= 30%",
        null_max_reference_threshold=null_reference,
        null_calibration_trials=len(calibration), null_validation_trials=len(validation),
        null_reference_note="Calibration maximum; limited sample, not a validated 5% false-alarm level.",
        fixed_macro_at_25=[dict(strength_arcsec=s, **recovery_rates(rs, 25.0, "fixed_macro"))
                           for s, rs in groups.items()],
        threshold_scan=grid,
        limitations=["One fixed halo position, source, noise level, PSF and profile",
                     "Macro starts at truth; only theta_E and two shear components are free",
                     "Source brightness is reprofiled on a fixed Delaunay mesh",
                     "Noise seeds vary; PINN and source mesh seeds stay at 42",
                     "Angular strength in arcsec; no physical mass conversion",
                     "Small numbers of trials; no population completeness claim"],
        trials=records,
    )
    write_json(output / "summary.json", summary)
    plot_results(records, groups, null_reference, output)
    rows = ["| Strength [arcsec] | Accepted @25 | Accurate @25 | Accurate @null-max | Position median [mas] | Strength error median |",
            "|---:|---:|---:|---:|---:|---:|"]
    for s, rs in groups.items():
        original, calibrated = recovery_rates(rs, 25.0), recovery_rates(rs, null_reference)
        n = len(rs)
        position_text = "—" if original['median_position_error_mas'] is None else f"{original['median_position_error_mas']:.1f}"
        strength_text = "—" if original['median_strength_error_percent'] is None else f"{original['median_strength_error_percent']:+.1f}%"
        rows.append(f"| {s:g} | {original['accepted']}/{n} | {original['accurate']}/{n} | "
                    f"{calibrated['accurate']}/{n} | {position_text} | {strength_text} |")
    report = (
        "# PINN angular-strength recovery scan\n\n"
        f"Position: {HALO_POSITION} arcsec. Noise RMS 0.012, pixel scale 0.05 arcsec, PSF FWHM 0.09 arcsec.\n\n"
        "Accurate recovery requires convergence, gain above threshold, position error <=50 mas "
        "and strength error <=30%. Best candidate is chosen without truth. "
        "Gain includes source regularization and is not a sigma significance.\n\n"
        + "\n".join(rows) + "\n\n"
        f"Null-max reference: {null_reference:.3f}, fitted from {len(calibration)} null trials. "
        f"At gain >25: {sum(accepted(r, 25) for r in calibration)}/{len(calibration)} "
        f"calibration false detections; {sum(accepted(r, 25) for r in validation)}/{len(validation)} "
        "independent validation false detections. "
        f"At gain >null-max: {sum(accepted(r, null_reference) for r in validation)}/{len(validation)} "
        "independent validation false detections.\n\n"
        "Macro starts at truth; only theta_E and shear are free. Source brightness is "
        "refitted on a fixed mesh. Noise seeds vary, algorithm seeds remain fixed. "
        "This is a local, small-sample sensitivity experiment, not a multi-halo or survey mass threshold.\n"
    )
    (output / "REPORT.md").write_text(report)
    print(report, flush=True)


def plot_results(records, groups, null_reference, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    strengths = np.array(list(groups))
    colors = ("#276fbf", "#d66b18")
    for color, threshold, label in zip(colors, (25.0, null_reference), ("Gain >25", "Gain >null-max")):
        rates = [recovery_rates(groups[s], threshold) for s in strengths]
        axes[0, 0].plot(strengths, [r["accurate_recovery_rate"] for r in rates], "o-", color=color,
                        label=f"{label}: accurate")
        axes[0, 0].plot(strengths, [r["acceptance_rate"] for r in rates], "--", color=color,
                        alpha=.7, label=f"{label}: accepted")
    axes[0, 0].set(ylabel="Fraction of noise realizations", ylim=(-.05, 1.05),
                    title="Accurate: position <=50 mas, strength error <=30%")
    axes[0, 0].legend(fontsize=8)
    for status, marker, label in ((True, "o", "Converged"), (False, "x", "Unconverged")):
        items = [r for r in records if r["group"] == "injected" and r["joint"]["converged"] == status
                 and r["joint"]["position_error_mas"] is not None]
        xs = [r["injected_strength_arcsec"] for r in items]
        axes[0, 1].scatter(xs, [r["joint"]["position_error_mas"] for r in items],
                           marker=marker, color=colors[0], label=label)
        axes[1, 0].scatter(xs, [r["joint"]["strength_error_percent"] for r in items],
                           marker=marker, color=colors[0])
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].axhline(50, color="gray", linestyle="--")
    axes[0, 1].set(ylabel="Position error [mas]", yscale="log", title="Best candidate in every injected trial")
    axes[1, 0].axhspan(-30, 30, alpha=.12, color="green")
    axes[1, 0].axhline(0, color="gray", linewidth=.8)
    axes[1, 0].set(ylabel="Strength error [%]", title="Best candidate, including rejected / unconverged fits")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set(xscale="log", xlabel="Injected angular strength [arcsec]", xticks=strengths)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{s:g}" for s in strengths])
        ax.grid(alpha=.2)
    for group, marker, label in (("null_calibration", "o", "Calibration null"),
                                  ("null_validation", "s", "Independent null")):
        items = [r for r in records if r["group"] == group]
        axes[1, 1].scatter(range(len(items)), [r["joint"]["score"] for r in items], marker=marker, label=label)
    axes[1, 1].axhline(25, color=colors[0], linestyle="--", label="Current threshold 25")
    axes[1, 1].axhline(null_reference, color=colors[1], linestyle=":", label=f"Null maximum {null_reference:.1f}")
    axes[1, 1].set(xlabel="Noise realization index within each null group", ylabel="Best blind-search gain",
                    title="Null distribution (all optimizer statuses)")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(output / "recovery_threshold.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strengths", nargs="+", type=float, default=[.005, .010, .015, .020, .030, .050])
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--null-calibration", type=int, default=20)
    parser.add_argument("--null-validation", type=int, default=5)
    parser.add_argument("--joint-maxiter", type=int, default=160)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "recovery_threshold_results")
    args = parser.parse_args()
    if (min(args.null_calibration, args.null_validation, args.workers, args.joint_maxiter) < 1
            or not np.isfinite(args.strengths).all() or min(args.strengths) <= 0
            or max(args.strengths) > .12 or len(set(args.strengths)) != len(args.strengths)
            or len(set(args.noise_seeds)) != len(args.noise_seeds)
            or any(s < 0 or s >= 2**32 for s in args.noise_seeds)):
        parser.error("Use positive counts, distinct seeds in [0, 2**32), and distinct strengths in (0, .12]")
    calibration_seeds = list(range(1000, 1000 + args.null_calibration))
    validation_seeds = list(range(2000, 2000 + args.null_validation))
    pools = [set(calibration_seeds), set(validation_seeds), set(args.noise_seeds)]
    if any(pools[i] & pools[j] for i in range(3) for j in range(i)):
        parser.error("Calibration, validation and injection noise seeds must be disjoint")
    matplotlib.use("Agg")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        schema_version=SCHEMA_VERSION, strengths_arcsec=sorted(args.strengths),
        injected_noise_seeds=args.noise_seeds, calibration_noise_seeds=calibration_seeds,
        validation_noise_seeds=validation_seeds, device=args.device, joint_maxiter=args.joint_maxiter,
        simulation=SIMULATION, injected_position_arcsec=list(HALO_POSITION), macro_input=MACRO_PARAMS,
        halo_core_arcsec=.035, halo_truncation_arcsec=.22, fit_annulus_arcsec=[.5, 1.8],
        source_nodes=400, regularization=.002, top_k=3, pinn_steps=90, refine_steps=55,
        source_mesh_seed=42, pinn_seed=42,
    )
    # Normalize tuples before comparing cached configuration to JSON-loaded data.
    config = json.loads(json.dumps(config))
    tasks = [("injected", s, seed, config, str(output))
             for s in reversed(config["strengths_arcsec"]) for seed in args.noise_seeds]
    tasks += [("null_calibration", 0., seed, config, str(output)) for seed in calibration_seeds]
    tasks += [("null_validation", 0., seed, config, str(output)) for seed in validation_seeds]
    records, pending = [], []
    for task in tasks:
        group, strength, seed = task[:3]
        path = output / group / f"strength_{strength:.6f}_seed_{seed}" / "metrics.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["config"] != config:
                raise ValueError(f"Existing configuration differs: {path}; use a separate output directory")
            records.append(cached)
        else:
            pending.append(task)
    write_json(output / "config.json", config)
    print(f"{len(tasks)} trials, {len(records)} cached, {args.workers} workers", flush=True)

    def completed(record):
        records.append(record)
        fit = record["joint"]
        print(f"[{len(records)}/{len(tasks)}] {record['group']} s={record['injected_strength_arcsec']:.3f} "
              f"seed={record['noise_seed']} gain={fit['score']:.2f} "
              f"position_mas={fit['position_error_mas']} strength_error={fit['strength_error_percent']} "
              f"converged={fit['converged']} ({record['seconds']:.1f}s)", flush=True)

    if args.workers == 1:
        for task in pending:
            completed(run_trial(task))
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_trial, task) for task in pending]
            for future in as_completed(futures):
                completed(future.result())
    records.sort(key=lambda r: (r["group"], r["injected_strength_arcsec"], r["noise_seed"]))
    summarize(records, output, config)


if __name__ == "__main__":
    main()
