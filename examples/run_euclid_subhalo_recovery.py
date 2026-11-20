"""Compare subhalo proposal routes on mock, Euclid VIS, or J0946 HST images."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import fftconvolve

from sublens.survey import load_euclid_source_observation, search_euclid_subhalos
from sublens.survey.euclid import _halo_deflection
from sublens.survey.euclid_source import EuclidSourceObservation, _q1_image_rays, _q1_source_rays


INJECTIONS = {
    "null": None,
    "bright_arc": (-1.45, 0.55, 0.065),
    "counterimage": (0.46, 0.75, 0.065),
    "faint_counterimage": (0.75, -2.05, 0.065),
    "weaker_bright_arc": (-1.45, 0.55, 0.035),
}


def simulate_observation(label, halo, noise_seed=42):
    """Generate analytic source knots independently of the inverse source basis."""
    macro = {
        "centre_1_max_lh": -0.137, "centre_0_max_lh": -0.187,
        "ell_comps_1_max_lh": -0.00962, "ell_comps_0_max_lh": -0.05022,
        "einstein_radius_max_lh": 1.427,
        "shear_gamma_1_max_lh": -0.0804, "shear_gamma_2_max_lh": -0.1943,
    }
    shape = (76, 76)
    scale = 0.10
    x, y, bx, by = _q1_source_rays(shape, scale, macro, 2)
    _, _, true_x, true_y = _q1_source_rays(shape, scale, macro, 4)
    if halo is not None:
        rays = map(torch.from_numpy, _q1_image_rays(shape, scale, 4))
        ax, ay = _halo_deflection(*rays, *halo)
        true_x, true_y = true_x - ax.numpy(), true_y - ay.numpy()

    def gaussian(x, y, sx, sy):
        return np.exp(-0.5 * ((x / sx)**2 + (y / sy)**2))

    u, v = true_x + 0.18, true_y + 0.435
    brightness = (0.22 * gaussian(u, v, 0.055, 0.042)
                  + 0.13 * gaussian(u - 0.084, v - 0.067, 0.047, 0.055)
                  + 0.10 * gaussian(u + 0.028, v - 0.015, 0.128, 0.071))
    psf_x, psf_y = np.meshgrid(np.arange(-6, 7) * scale, np.arange(-6, 7) * scale)
    psf = (0.92 * gaussian(psf_x, psf_y, 0.067, 0.074)
           + 0.08 * gaussian(psf_x - 0.049, psf_y + 0.027, 0.105, 0.11))
    psf /= psf.sum()
    model = fftconvolve(brightness.mean(axis=-1), psf, mode="same")
    radius = np.hypot(x, y)
    rms = 0.0035 + 0.0025 * np.exp(-radius / 0.5)
    noise = np.random.default_rng(noise_seed).standard_normal(shape) * rms
    return EuclidSourceObservation(
        label, model + noise, rms, psf, radius < 3.1, bx, by, scale,
        (-0.137, -0.187), 1.427,
    )


def save_diagnostic(obs, result, truth, output, reference_position=None):
    candidate = result.best_candidate
    smooth = result.baseline
    refitted = candidate.refitted if candidate else None
    model = refitted.image_model if refitted else smooth.image_model
    source = refitted.source if refitted else smooth
    height, width = obs.image.shape
    sky = np.array([-width, width, -height, height]) * obs.pixel_scale / 2
    order = np.linspace(-1, 1, 201)
    x, y = np.meshgrid(order, order)
    s0, s1 = smooth.sample(x, y), source.sample(x, y)
    vmax = max(float(s0.max()), float(s1.max()))
    bright = (np.maximum(s0, s1) > 0.12 * vmax) if vmax else np.ones_like(s0, dtype=bool)
    source_view = (max(-1, x[bright].min() - 0.10), min(1, x[bright].max() + 0.10),
                   max(-1, y[bright].min() - 0.10), min(1, y[bright].max() + 0.10))
    potential = result.residual_potential
    nrows = 3 if potential is not None else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(10.5, 3.5 * nrows), layout="constrained")
    flux = axes[0, 0].imshow(np.where(obs.mask, obs.image, np.nan), origin="lower",
                            extent=sky, cmap="magma", vmin=0,
                            vmax=np.percentile(obs.image[obs.mask], 99.5))
    fig.colorbar(flux, ax=axes[0, 0], shrink=0.7, label="Observed flux")
    axes[0, 0].set_title("Foreground-subtracted image")
    residual_panels = [(axes[0, 1], smooth.image_model, "Smooth-model residual")]
    source_panels = [(axes[1, 0], s0, "Smooth-model source")]
    if refitted is not None:
        residual_panels.append((axes[1, 1], model, "Candidate residual"))
        source_panels.append((axes[1, 2], s1, "Refitted source"))
    else:
        axes[1, 1].set_axis_off()
        axes[1, 1].text(0.5, 0.5, "0 candidates", ha="center", va="center",
                        transform=axes[1, 1].transAxes)
        axes[1, 2].set_axis_off()
    for ax, prediction, title in residual_panels:
        residual = np.full_like(obs.image, np.nan)
        residual[obs.mask] = (obs.image[obs.mask] - prediction[obs.mask]) / obs.rms[obs.mask]
        field = ax.imshow(residual, origin="lower", extent=sky, cmap="RdBu_r", vmin=-5, vmax=5)
        ax.set_title(title)
    fig.colorbar(field, ax=[panel[0] for panel in residual_panels], shrink=0.7, label="Residual / RMS")
    for ax, field, title in source_panels:
        src = ax.imshow(field, origin="lower", extent=(-1.005, 1.005, -1.005, 1.005),
                        cmap="magma", vmin=0, vmax=vmax)
        ax.axis(source_view)
        ax.set_title(title)
    fig.colorbar(src, ax=[panel[0] for panel in source_panels], shrink=0.7, label="Source brightness")
    ax = axes[0, 2]
    profile = result.profile_scores.size > 0
    if potential is not None:
        radar = ax.imshow(potential.convergence, origin="lower", extent=sky, cmap="RdBu_r")
        if potential.peaks:
            px, py, _ = np.asarray(potential.peaks).T
            ax.scatter(px, py, marker="x", color="black", label="PINN seeds")
        ax.set_title("PINN convergence perturbation")
    else:
        xs, ys = (result.profile_x, result.profile_y) if profile else (result.radar_x, result.radar_y)
        scores = result.profile_scores if profile else result.radar_scores
        radar = ax.pcolormesh(xs, ys, scores, shading="nearest", cmap="cividis")
        ax.set_title("Coarse source-refit gain" if profile else "Radar score")
    if truth is not None:
        ax.plot(*truth[:2], marker="+", ms=12, color="white", mew=2, label="Injected")
    if reference_position is not None:
        ax.plot(*reference_position, marker="+", ms=12, color="gold", mew=2, label="Published reference")
    if refitted and refitted.strength > 0:
        ax.plot(refitted.x, refitted.y, marker="o", mfc="none", mec="tab:red",
                mew=1.8, ls="none", label="Refitted")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.colorbar(radar, ax=ax, shrink=0.7)
    if potential is not None:
        before = potential.baseline_residual
        after = potential.fitted_residual
        for panel, field, label, limit in zip(
            axes[2], (before, after, before**2 - after**2),
            ("Cross-fitted residual / RMS", "Residual after PINN / RMS", "Squared residual improvement"),
            (5, 5, 15),
        ):
            artist = panel.imshow(field, origin="lower", extent=sky, cmap="RdBu_r",
                                  vmin=-limit, vmax=limit)
            panel.set_title(label)
            fig.colorbar(artist, ax=panel, shrink=0.7)
    for panel in axes.ravel():
        panel.set_xlabel("x [arcsec]")
    for panel in axes[:, 0]:
        panel.set_ylabel("y [arcsec]")
    fig.suptitle(obs.lens_id)
    fig.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(fig)


def run_case(obs, truth, outfile, reference_position=None, **search_options):
    start = perf_counter()
    result = search_euclid_subhalos(obs, **search_options)
    elapsed = perf_counter() - start
    print(f"{obs.lens_id}  {len(result.candidates)} candidates in {elapsed:.1f}s",
          flush=True)
    save_diagnostic(obs, result, truth, outfile, reference_position)
    reference = {
        "case": obs.lens_id,
        "injected_x": truth[0] if truth else None,
        "injected_y": truth[1] if truth else None,
        "injected_strength": truth[2] if truth else None,
        "baseline_mean_r2": np.mean(
            ((obs.image[obs.mask] - result.baseline.image_model[obs.mask]) / obs.rms[obs.mask])**2
        ),
        "max_coarse_gain": np.nanmax(result.profile_scores) if result.profile_scores.size else None,
        "proposal_route": search_options.get("proposal", "grid"),
        "seconds": elapsed,
    }
    potential = result.residual_potential
    if potential is not None and potential.mask.any():
        reference.update(
            pinn_coverage=int(potential.mask.sum()),
            pinn_baseline_mean_r2=float(np.nanmean(potential.baseline_residual**2)),
            pinn_fitted_mean_r2=float(np.nanmean(potential.fitted_residual**2)),
        )
    rows = []
    for rank, candidate in enumerate(result.candidates, 1):
        for method, fit in [("frozen", candidate.frozen), ("refitted", candidate.refitted)]:
            rows.append({
                **reference, "rank_by_refitted_objective": rank, "method": method,
                "seed_x": candidate.seed[0], "seed_y": candidate.seed[1],
                "proposal_method": candidate.proposal_method,
                "proposal_score": candidate.proposal_score, "seed_strength": candidate.seed_strength,
                "x": fit.x, "y": fit.y,
                "strength": fit.strength, "delta_chi2": fit.delta_chi2,
                "delta_objective": fit.delta_objective, "converged": fit.converged,
                "position_error_arcsec": np.hypot(fit.x - truth[0], fit.y - truth[1]) if truth else None,
                "reference_separation_arcsec": (
                    np.hypot(fit.x - reference_position[0], fit.y - reference_position[1])
                    if reference_position is not None and fit.strength > 0 else None
                ),
            })
    if rows:
        leading = rows[1]
        print(f"  ({leading['x']:+.3f}, {leading['y']:+.3f}) arcsec  "
              f"strength {leading['strength']:.4f}  gain {leading['delta_objective']:.2f}", flush=True)
    if not rows:
        rows.append({**reference, "rank_by_refitted_objective": None, "method": "no_candidate",
                     "delta_chi2": 0, "delta_objective": 0})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=tuple(INJECTIONS))
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--profile-grid-size", type=int, default=15, help="Use 0 for Radar-only proposals")
    parser.add_argument("--real-lens-id")
    parser.add_argument("--j0946-bundle", type=Path, help="Public LensAgent HST F814W NPZ cutout")
    parser.add_argument("--reference-position", nargs=2, type=float, metavar=("X", "Y"),
                        help="Aligned published HST reference used for reporting only")
    parser.add_argument("--proposal", action="append", choices=("grid", "pinn"))
    parser.add_argument("--pinn-steps", type=int, default=90)
    parser.add_argument("--data-dir", type=Path, default=Path("archive/data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("euclid_q1_diagnostic_plots/subhalo_recovery"))
    args = parser.parse_args()
    if sum(bool(value) for value in (args.real_lens_id, args.case, args.j0946_bundle)) > 1:
        parser.error("Choose simulation cases, a Q1 lens, or the J0946 bundle")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.j0946_bundle:
        try:
            from .j0946_observation import load_j0946_observation
        except (ImportError, ModuleNotFoundError):
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from j0946_observation import load_j0946_observation

        obs = load_j0946_observation(args.j0946_bundle)
        cases = [(obs, None, obs.lens_id)]
        summary = f"{obs.lens_id}_candidates.csv"
    elif args.real_lens_id:
        obs = load_euclid_source_observation(args.real_lens_id, args.data_dir)
        cases = [(obs, None, obs.lens_id)]
        summary = f"{obs.lens_id}_candidates.csv"
    else:
        cases = [(simulate_observation(label, INJECTIONS[label], args.noise_seed),
                  INJECTIONS[label], f"{label}_{args.noise_seed}") for label in args.case or INJECTIONS]
        summary = f"simulation_{args.noise_seed}.csv"
    for obs, truth, name in cases:
        for proposal in args.proposal or ["grid"]:
            suffix = f"_{proposal}" if args.proposal else ""
            rows.extend(run_case(obs, truth, args.output_dir / f"{name}{suffix}.png",
                                 reference_position=args.reference_position,
                                 profile_grid_size=args.profile_grid_size, proposal=proposal,
                                 pinn_steps=args.pinn_steps))
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output_dir / summary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
