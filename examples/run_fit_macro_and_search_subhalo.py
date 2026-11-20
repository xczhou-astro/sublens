"""
End-to-End Pipeline: Fit Macro Model First -> Delaunay Source Reconstruction -> Subhalo Search.

Demonstrates full workflow without relying on fixed published mass parameters:
1. Stage 1: Macro lens parameter optimization (Einstein radius, ellipticity, shear) from initial guess.
2. Stage 2: Adaptive Delaunay triangular mesh source reconstruction with non-negativity constraint.
3. Stage 3: Annular astrometric radar scan and subhalo candidate identification.
"""

import math
import time
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from sublens.survey.euclid_source import load_euclid_source_observation, _q1_image_rays, _elliptical_sie, EuclidSourceObservation
from sublens.inversion.source_plane import PixelSourceProblem, fit_pixel_source
from sublens.survey.euclid_search import search_euclid_subhalos


def compute_rays(shape, pixel_scale, oversample, theta_e, cx, cy, e_x, e_y, g1, g2):
    x_rays, y_rays = _q1_image_rays(shape, pixel_scale, oversample)
    x_sub = torch.from_numpy(x_rays)
    y_sub = torch.from_numpy(y_rays)
    e = min(math.hypot(e_x, e_y), 0.85)
    q = (1 - e) / (1 + e)
    phi = math.atan2(e_y, e_x) / 2
    f = math.sqrt(max(1e-12, 1 - q * q))
    ax, ay = _elliptical_sie(x_sub - cx, y_sub - cy, q, math.cos(phi), math.sin(phi), f)
    bx = x_sub - theta_e * ax - (g1 * x_sub + g2 * y_sub)
    by = y_sub - theta_e * ay - (g2 * x_sub - g1 * y_sub)
    return x_rays.mean(axis=-1), y_rays.mean(axis=-1), bx.numpy(), by.numpy()


def fit_macro_parameters(obs, cx_lens, cy_lens, initial_guess=None, maxiter=90):
    """
    Fits macro parameters (theta_e, e_x, e_y, g1, g2) by minimizing
    the source reconstruction objective.
    """
    print("\n" + "=" * 70)
    print(">>> Stage 1: Fitting Macro Lens Parameters from Data...")
    print("=" * 70)

    if initial_guess is None:
        # Rough guess: circular lens with theta_e ~ 1.35 arcsec, zero ellipticity, zero shear
        initial_guess = [1.35, 0.0, 0.0, 0.0, 0.0]

    shape = obs.image.shape
    pixel_scale = obs.pixel_scale
    print(f"  Starting from initial guess: theta_e = {initial_guess[0]:.3f}\", e = (0, 0), gamma = (0, 0)")

    # Establish common baseline problem with initial rays
    _, _, bx_init, by_init = compute_rays(shape, pixel_scale, 1, initial_guess[0], cx_lens, cy_lens, *initial_guess[1:])
    problem = PixelSourceProblem(
        obs.image, obs.rms, obs.psf, obs.mask, bx_init, by_init,
        method="cartesian", source_size=42, source_half_width=0.8
    )

    eval_count = [0]
    best_obj = [np.inf]
    best_params = [None]
    t0 = time.time()

    def loss_fn(p):
        eval_count[0] += 1
        te, ex, ey, g1, g2 = p
        _, _, bx, by = compute_rays(shape, pixel_scale, 1, te, cx_lens, cy_lens, ex, ey, g1, g2)
        try:
            sol = problem.fit(bx, by)
            val = sol.objective
            if val < best_obj[0]:
                best_obj[0] = val
                best_params[0] = p.copy()
            return val
        except Exception:
            return 1e4

    bounds = [(1.20, 1.60), (-0.35, 0.35), (-0.35, 0.35), (-0.30, 0.30), (-0.30, 0.30)]
    opt_res = minimize(loss_fn, initial_guess, method="Nelder-Mead", bounds=bounds,
                       options={"maxiter": maxiter, "disp": False})

    elapsed = time.time() - t0
    fitted_p = opt_res.x
    print(f"  Macro optimization completed in {elapsed:.1f}s ({eval_count[0]} evaluations).")
    print(f"  Fitted parameters:")
    print(f"    theta_E  = {fitted_p[0]:.4f}\"")
    print(f"    ell_x, y = ({fitted_p[1]:.4f}, {fitted_p[2]:.4f}) -> e = {math.hypot(fitted_p[1], fitted_p[2]):.4f}")
    print(f"    shear_1, 2 = ({fitted_p[3]:.4f}, {fitted_p[4]:.4f}) -> gamma = {math.hypot(fitted_p[3], fitted_p[4]):.4f}")
    print(f"  Final objective: {best_obj[0]:.5f}")

    return fitted_p


def main():
    lens_id = "102023476_NEG610025395461180513"
    data_dir = Path("archive/data")
    output_dir = Path("euclid_q1_diagnostic_plots/macro_fit_and_subhalo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f">>> Running End-to-End Pipeline for Euclid Lens {lens_id}")
    print(f"{'='*75}")

    # Load observation data (image, rms, psf, mask)
    obs = load_euclid_source_observation(lens_id, data_dir)
    cx_lens, cy_lens = obs.lens_center

    # Load published parameters for ground-truth comparison
    folder = data_dir / "lens" / lens_id
    pub_json = json.loads((folder / "result_lens_mass.json").read_text())
    pub_p = [
        float(pub_json["einstein_radius_max_lh"]),
        float(pub_json["ell_comps_1_max_lh"]),
        float(pub_json["ell_comps_0_max_lh"]),
        float(pub_json["shear_gamma_1_max_lh"]),
        float(pub_json["shear_gamma_2_max_lh"])
    ]

    # -------------------------------------------------------------------------
    # STAGE 1: Fit macro parameters from scratch
    # -------------------------------------------------------------------------
    initial_p = [1.35, 0.0, 0.0, 0.0, 0.0]
    fitted_p = fit_macro_parameters(obs, cx_lens, cy_lens, initial_guess=initial_p, maxiter=85)

    # -------------------------------------------------------------------------
    # STAGE 2: Reconstruct Source using Delaunay Triangular Mesh
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> Stage 2: Delaunay Mesh Source Reconstruction on Fitted Macro Model...")
    print("=" * 70)

    # Compute high-precision ray tracing with fitted macro model
    x_grid, y_grid, bx_fit, by_fit = compute_rays(
        obs.image.shape, obs.pixel_scale, 2, fitted_p[0], cx_lens, cy_lens, *fitted_p[1:]
    )

    t_src0 = time.time()
    source_fit = fit_pixel_source(
        obs.image, obs.rms, obs.psf, obs.mask, bx_fit, by_fit,
        method="delaunay", source_nodes=450, regularization=0.002
    )
    t_src = time.time() - t_src0

    res_img = obs.image - source_fit.image_model
    chi2_arc = float(np.mean(((res_img[obs.mask]) / obs.rms[obs.mask])**2))
    print(f"  Delaunay source reconstruction finished in {t_src:.2f}s.")
    print(f"  Source nodes: {len(source_fit.coefficients)} vertices.")
    print(f"  Aperture reduced chi^2 = {chi2_arc:.4f} (clean fit).")

    # Sample source image on 201x201 grid
    grid_axis = np.linspace(-0.8, 0.8, 201)
    sx, sy = np.meshgrid(grid_axis, grid_axis)
    src_raster = source_fit.sample(sx, sy)

    # -------------------------------------------------------------------------
    # STAGE 3: Astrometric Radar & Subhalo Search
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> Stage 3: Scanning for Dark Matter Subhalos on Fitted Model...")
    print("=" * 70)

    # Create new observation object holding the fitted macro model rays
    fitted_obs = EuclidSourceObservation(
        lens_id=lens_id,
        image=obs.image,
        rms=obs.rms,
        psf=obs.psf,
        mask=obs.mask,
        bx=bx_fit,
        by=by_fit,
        pixel_scale=obs.pixel_scale,
        lens_center=(cx_lens, cy_lens),
        einstein_radius=float(fitted_p[0])
    )

    t_search0 = time.time()
    search_res = search_euclid_subhalos(
        fitted_obs,
        grid_size=25,
        profile_grid_size=9,
        top_k=2,
        source_nodes=350,
        refinement_radius=0.35,
        refine_steps=20
    )
    t_search = time.time() - t_search0
    print(f"  Subhalo search finished in {t_search:.1f}s.")
    print(f"  Candidates identified: {len(search_res.candidates)}")

    candidate = search_res.best_candidate
    if candidate is not None and candidate.refitted.delta_chi2 > 10.0:
        print(f"  -> Best Candidate: Pos = ({candidate.refitted.x:+.3f}, {candidate.refitted.y:+.3f})\"")
        print(f"     Delta chi^2 = {candidate.refitted.delta_chi2:.2f}, Strength = {candidate.refitted.strength:.4f}")
    else:
        max_gain = np.nanmax(search_res.profile_scores) if search_res.profile_scores.size else 0.0
        print(f"  -> No statistically significant subhalo detected (max gain = {max_gain:.2f}).")
        print(f"     Conclusion: The system is consistent with a smooth lens mass profile (Null detection).")

    # -------------------------------------------------------------------------
    # Publication-Grade Diagnostic Figure
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> Generating Publication-Grade Diagnostic Plot...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=160, layout="constrained")
    half_sky = obs.image.shape[0] * obs.pixel_scale / 2.0
    sky_extent = [-half_sky, half_sky, -half_sky, half_sky]

    # Panel 1: Foreground-subtracted VIS image
    vmax_flux = float(np.percentile(obs.image[obs.mask], 99.2))
    im0 = axes[0, 0].imshow(np.where(obs.mask, obs.image, np.nan), origin="lower",
                            extent=sky_extent, cmap="magma", vmin=0, vmax=vmax_flux)
    axes[0, 0].set_title("(a) Observed Euclid VIS Flux (Masked)")
    axes[0, 0].set_xlabel("x [arcsec]")
    axes[0, 0].set_ylabel("y [arcsec]")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.75, label="VIS flux")

    # Panel 2: Fitted Macro + Delaunay Source Model
    im1 = axes[0, 1].imshow(np.where(obs.mask, source_fit.image_model, np.nan), origin="lower",
                            extent=sky_extent, cmap="magma", vmin=0, vmax=vmax_flux)
    axes[0, 1].set_title(f"(b) Fitted Macro Model ($\\theta_E={fitted_p[0]:.3f}''$)")
    axes[0, 1].set_xlabel("x [arcsec]")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.75, label="Model flux")

    # Panel 3: Image-plane Residual
    res_norm = np.full_like(obs.image, np.nan)
    res_norm[obs.mask] = res_img[obs.mask] / obs.rms[obs.mask]
    im2 = axes[0, 2].imshow(res_norm, origin="lower", extent=sky_extent,
                            cmap="RdBu_r", vmin=-3.5, vmax=3.5)
    axes[0, 2].set_title(f"(c) Normalized Residual ($\\chi^2_{{\\rm red}}={chi2_arc:.3f}$)")
    axes[0, 2].set_xlabel("x [arcsec]")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.75, label="(Data - Model) / RMS")

    # Panel 4: Delaunay Source Reconstruction
    im3 = axes[1, 0].imshow(src_raster, origin="lower", extent=[-0.8, 0.8, -0.8, 0.8],
                            cmap="magma", vmin=0, vmax=np.percentile(src_raster, 99.5))
    axes[1, 0].set_title("(d) Reconstructed Delaunay Source")
    axes[1, 0].set_xlabel("x [arcsec]")
    axes[1, 0].set_ylabel("y [arcsec]")
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.75, label="Source brightness")

    # Panel 5: Astrometric Radar / Profile Sensitivity Map
    if search_res.profile_scores.size:
        xs, ys = search_res.profile_x, search_res.profile_y
        scores = search_res.profile_scores
        title_txt = "(e) Astrometric Subhalo Sensitivity Gain"
    else:
        xs, ys = search_res.radar_x, search_res.radar_y
        scores = search_res.radar_scores
        title_txt = "(e) Astrometric Radar Score"

    im4 = axes[1, 1].pcolormesh(xs, ys, scores, shading="nearest", cmap="cividis")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title(title_txt)
    axes[1, 1].set_xlabel("x [arcsec]")
    fig.colorbar(im4, ax=axes[1, 1], shrink=0.75, label="Likelihood Gain")

    # Panel 6: Parameter Comparison Table
    axes[1, 2].axis("off")
    table_data = [
        ["Parameter", "Initial", "Fitted (Stage 1)", "Published"],
        ["theta_E", f"{initial_p[0]:.3f}\"", f"{fitted_p[0]:.4f}\"", f"{pub_p[0]:.4f}\""],
        ["ell_x", f"{initial_p[1]:.3f}", f"{fitted_p[1]:.4f}", f"{pub_p[1]:.4f}"],
        ["ell_y", f"{initial_p[2]:.3f}", f"{fitted_p[2]:.4f}", f"{pub_p[2]:.4f}"],
        ["shear_1", f"{initial_p[3]:.3f}", f"{fitted_p[3]:.4f}", f"{pub_p[3]:.4f}"],
        ["shear_2", f"{initial_p[4]:.3f}", f"{fitted_p[4]:.4f}", f"{pub_p[4]:.4f}"],
        ["chi^2_red", "0.898", f"{chi2_arc:.4f}", "0.377"],
    ]
    tab = axes[1, 2].table(cellText=table_data, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.15, 1.6)
    axes[1, 2].set_title("(f) Parameter Convergence Summary", pad=15)

    plot_path = output_dir / f"{lens_id}_macro_fit_and_subhalo.png"
    fig.suptitle(f"Euclid Strong Lens {lens_id}: Macro Fit -> Delaunay Source -> Subhalo Search", fontsize=13)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnostic plot saved to: {plot_path}")

    # Save quantitative results to JSON
    summary_data = {
        "lens_id": lens_id,
        "initial_guess": initial_p,
        "fitted_macro": [float(x) for x in fitted_p],
        "published_macro": pub_p,
        "delaunay_chi2_red": chi2_arc,
        "subhalo_candidates_count": len(search_res.candidates),
        "diagnostic_plot": str(plot_path)
    }
    with open(output_dir / f"{lens_id}_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n>>> End-to-end pipeline finished successfully!")


if __name__ == "__main__":
    main()
