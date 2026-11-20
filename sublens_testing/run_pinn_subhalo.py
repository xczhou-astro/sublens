"""Run Delaunay reconstruction and PINN discovery on simulations or a FITS observation."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch


HALO_POSITION = (-1.28, 0.61)  # arcsec
HALO_STRENGTH = 0.050  # angular strength in arcsec
NOISE_SEED = 42
MACRO_PARAMS = dict(
    theta_E=1.35, q=0.80, phi=0.35, gamma=2.05,
    gamma_ext=0.045, phi_ext=0.20, center=(0.0, 0.0),
)


@torch.no_grad()
def simulate_observations(device):
    from sublens import (
        LensingObservation, LensingSimulator, MacroLensModel,
        sersic_profile, truncated_subhalo_deflection,
    )

    sim = LensingSimulator(
        N_img=100, pixel_scale=0.05, N_src=257, L_src=1.6,
        psf_fwhm=0.09, noise_std=0.012, device=device,
    )
    macro_params = MACRO_PARAMS
    macro = MacroLensModel(**macro_params).to(sim.device)
    alpha_macro, _ = macro(sim.theta_grid)

    sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
    source = sersic_profile(
        sx, sy, center=(0.04, -0.03), i_eff=0.30, r_eff=0.22,
        n_sersic=1.25, q=0.78, phi=0.30,
    )
    theta_x, theta_y = sim.theta_grid.unbind(dim=1)
    ax, ay = truncated_subhalo_deflection(theta_x, theta_y, *HALO_POSITION, HALO_STRENGTH)
    alpha_halo = torch.stack((ax, ay), dim=-1)

    # Match the noise realization in the single-halo and zero-halo observations.
    control_image, _ = sim.simulate(source[None, None], alpha_macro, seed=NOISE_SEED)
    injected_image, _ = sim.simulate(source[None, None], alpha_macro + alpha_halo, seed=NOISE_SEED)
    geometry = dict(macro_model=macro_params)
    control = LensingObservation.from_simulation(
        sim, control_image, alpha_macro, lens_id="null", **geometry,
    )
    injected = LensingObservation.from_simulation(
        sim, injected_image, alpha_macro, lens_id="injected", **geometry,
    )
    return sim, source, control, injected


def report_search(result):
    print()
    result.summary()
    for candidate in result.candidates:
        fit = candidate.fit
        print(
            f"  Peak {candidate.rank}: "
            f"seed=({candidate.seed[0]:+.4f}, {candidate.seed[1]:+.4f}), "
            f"fit=({fit.x:+.4f}, {fit.y:+.4f}), strength={fit.strength:.5f} arcsec, "
            f"Δχ²={fit.delta_chi2:.2f}, converged={fit.converged}"
        )

    best = result.best_candidate
    if best is not None and best.fit.macro_model is not None:
        macro = best.fit.macro_model
        print(f"  Refined theta_E={macro['theta_E']:.5f} arcsec, "
              f"shear={macro['gamma_ext']:.5f}, angle={macro['phi_ext']:.5f} rad")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto",
                        help="PyTorch device for simulation and Delaunay source fitting")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "pinn_subhalo_results",
                        help="Directory for the PNG figures and residual arrays")
    parser.add_argument("--show", action="store_true", help="Also display the diagnostic figures")
    parser.add_argument("--input-fits", type=Path,
                        help="Analyze a FITS observation with macro parameters or saved source rays")
    parser.add_argument("--noise-fits", type=Path, help="Separate RMS FITS file for --input-fits")
    parser.add_argument("--psf-fits", type=Path, help="Separate PSF FITS file for --input-fits")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sublens import SubLensPipeline

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = sim = source_true = None
    if args.input_fits is None:
        sim, source_true, control, injected = simulate_observations(args.device)
        x, y = injected.image_rays()
        radius = np.hypot(x[..., 0], y[..., 0])
        injected = replace(injected, mask=(radius >= 0.50) & (radius <= 1.80))
        control.to_fits(output / "simulation_null.fits", overwrite=True)
        injected.to_fits(output / "simulation_injected.fits", overwrite=True)
        input_fits = output / "simulation_injected.fits"
        print(f"Injected halo: {HALO_POSITION} arcsec, strength {HALO_STRENGTH:.5f} arcsec")
    else:
        input_fits = args.input_fits
    pipeline = SubLensPipeline.from_fits(
        input_fits, noise_path=args.noise_fits, psf_path=args.psf_fits, device=args.device,
    )
    injected = pipeline.observation()
    reconstruction_data = injected if control is None else control
    print(f"Simulation and Delaunay device: {pipeline.device}")

    validation = np.zeros(reconstruction_data.image.size, dtype=bool)
    rng = np.random.default_rng(2026)
    valid_pixels = np.flatnonzero(reconstruction_data.mask)
    if valid_pixels.size < 2:
        raise ValueError("Source reconstruction needs at least two valid pixels for an 80/20 split")
    validation[rng.choice(valid_pixels, max(1, valid_pixels.size // 5), replace=False)] = True
    reconstruction = pipeline.reconstruct_delaunay(
        reconstruction_data,
        validation_mask=validation.reshape(reconstruction_data.image.shape),
    )
    print(f"\nDelaunay source reconstruction on {reconstruction_data.lens_id}")
    for run in reconstruction.runs:
        print(f"  Seed {run.seed}: fit χ²/pixel={run.train_chi2_per_pixel:.4f}, "
              f"held-out χ²/pixel={run.validation_chi2_per_pixel:.4f}")
    print(f"  Best seed: {reconstruction.best.seed}")

    if source_true is not None:
        source_axis = sim.src_coords_1d.cpu().numpy()
        sx, sy = np.meshgrid(source_axis, source_axis)
        roi = (np.abs(sx) <= 0.65) & (np.abs(sy) <= 0.65)
        truth = source_true.cpu().numpy()
        source_fit = reconstruction.source.sample(sx, sy)
        source_error = np.linalg.norm((source_fit - truth)[roi]) / np.linalg.norm(truth[roi])
        print(f"  Source relative L2 error within ±0.65 arcsec: {source_error:.2%}")

    x, y = injected.image_rays()
    cx, cy = injected.lens_center
    radius = np.hypot(x[..., 0] - cx, y[..., 0] - cy)
    annulus = injected.mask
    if args.input_fits is None:
        annulus = annulus & (radius >= 0.50) & (radius <= 1.80)
    fixed_search = pipeline.search_pinn(injected, fit_mask=annulus)
    report_search(fixed_search)
    search = fixed_search
    if injected.macro_model is not None:
        search = fixed_search.refine_joint(injected.macro_model)
        report_search(search)
    if args.input_fits is None and search.best_candidate is not None:
        fit = search.best_candidate.fit
        offset_mas = 1000 * np.linalg.norm(np.asarray(fit.position) - HALO_POSITION)
        strength_error = fit.strength / HALO_STRENGTH - 1
        print(f"  Position error: {offset_mas:.1f} mas, strength error: {strength_error:+.1%}")

    if control is not None:
        null_search = pipeline.search_pinn(control, fit_mask=annulus)
        report_search(null_search)
        report_search(null_search.refine_joint(control.macro_model))

    truth_position = HALO_POSITION if args.input_fits is None else None
    figures = [
        reconstruction.plot(save_path=output / "delaunay_source.png"),
        search.plot(truth_position=truth_position, save_path=output / "pinn_convergence.png"),
        search.plot_fit(save_path=output / "subhalo_fit.png"),
        search.plot_residuals(save_path=output / "residual_comparison.png"),
        fixed_search.plot_residuals(save_path=output / "residuals_fixed_macro.png"),
    ]
    arrays = dict(
        observed=injected.image, rms=injected.rms, fit_mask=annulus,
        macro_model=search.baseline.image_model, residual_macro=search.residual_macro,
        residual_macro_fixed=fixed_search.residual_macro,
    )
    if fixed_search.best_candidate is not None:
        arrays["residual_with_subhalo_fixed"] = fixed_search.residual_with_subhalo
    if search.best_candidate is not None:
        arrays.update(
            macro_subhalo_model=search.best_candidate.fit.image_model,
            residual_with_subhalo=search.residual_with_subhalo,
        )
    np.savez_compressed(output / "residuals.npz", **arrays)
    search.save_json(output / "pinn_result.json", overwrite=True)
    search.save_fits(output / "pinn_result.fits", overwrite=True)
    print(f"\nDiagnostics saved to {output}")

    if args.show:
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
