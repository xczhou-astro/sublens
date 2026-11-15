"""
Example: End-to-End Inversion in the Redshift-Agnostic Regime (Unknown Redshifts).

Demonstrates:
- Operating in scale-free / dimensionless angular units (arcseconds, kappa, theta_E).
- Successfully discovering the subhalo position and angular deflection strength
  WITHOUT needing any redshift or distance information.
- Deriving the relative mass fraction f_sub = (theta_E,sub / theta_E,host)^2.
- Characterizing the continuous physical mass degeneracy curve M(z_l, z_s).
- Generating publication diagnostic plot.
"""

import time
import torch
import sublens as sl


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[SubLens Demo] Redshift-Agnostic / Scale-Free Inversion on {device}")
    print("=" * 80)

    # 1. Create scale-free benchmark (no redshifts provided!)
    data = sl.create_scale_free_benchmark(device=device)
    sim = data["simulator"]
    cosmo = data["cosmology"]
    obs_img = data["obs_image"]
    a_macro = data["alpha_macro"]
    true_pos = data["sub_true_pos"]
    theta_E_host = data["theta_E_host"]
    theta_E_sub = data["theta_E_sub"]

    print(f"Target System: {data['name']}")
    print(f"Cosmology State: Redshifts are UNKNOWN (z_l = None, z_s = None)")
    print(f"Host Einstein Radius: theta_E,host = {theta_E_host:.2f}\"")
    print(f"True Subhalo: theta_E,sub = {theta_E_sub:.3f}\" at {true_pos} arcsec")
    print(f"Relative Mass Fraction: f_sub = (theta_E,sub / theta_E,host)^2 = {(theta_E_sub / theta_E_host)**2:.4e}")

    # 2. Stage 1: Smooth Macro Baseline Fit (Dimensionless)
    print("\n>>> Stage 1: Fitting smooth macro baseline (TV pixelated source)...")
    fitter = sl.MacroFitter(sim, tv_weight=3e-4, lr=0.08, num_steps=120)
    macro_res = fitter.fit(obs_img, a_macro, verbose=True)

    # 3. Stage 2: Astrometric Radar Blind Scan (Angular Units)
    print("\n>>> Stage 2: Scanning Einstein ring with Astrometric Radar (Scale-Free)...")
    radar = sl.AstrometricRadar(sim, grid_resolution=45, r_min=0.4, r_max=1.8)
    radar_res = radar.scan(
        obs_img,
        pred_macro=macro_res["pred_macro"],
        res_macro=macro_res["res_macro"],
        source_macro=macro_res["source_reconstructed"],
        alpha_macro=a_macro,
        subhalo_type="SIS",
        cosmology=cosmo,
        verbose=True
    )

    # 4. Stage 3: PINN Continuous Autodiff Refinement
    print("\n>>> Stage 3: Running continuous autodiff PINN joint refinement (Scale-Free)...")
    pinn = sl.PINNOptimizer(sim, lr_pos=0.010, lr_mass=0.015, lr_src=0.04, num_steps=120)
    pinn_res = pinn.refine(
        obs_img,
        init_pos=radar_res["best_candidate"],
        init_src_param=macro_res["source_param"],
        alpha_macro=a_macro,
        vignette_mask=macro_res["vignette"],
        subhalo_type="SIS",
        init_kappa=0.035,
        verbose=True
    )

    # 5. Evaluate Metrics
    pinn_pos = pinn_res["refined_pos"]
    astrometrics = sl.compute_astrometric_error(pinn_pos, true_pos, pixel_scale=sim.pixel_scale)
    img_metrics = sl.compute_image_metrics(pinn_res["final_source"], data["source_true"])
    recovered_theta_E = pinn_res["final_kappa_s"]
    recovered_f_sub = (recovered_theta_E / theta_E_host)**2

    print("\n" + "=" * 80)
    print(">>> FINAL INVERSION REPORT (REDSHIFT-AGNOSTIC REGIME):")
    print(f"  Radar Blind Candidate: ({radar_res['best_candidate'][0]:+.3f}\", {radar_res['best_candidate'][1]:+.3f}\") [S = {radar_res['max_score']:.2f} sigma]")
    print(f"  PINN Converged Position: ({pinn_pos[0]:+.4f}\", {pinn_pos[1]:+.4f}\")")
    print(f"  Ground Truth Position: ({true_pos[0]:+.4f}\", {true_pos[1]:+.4f}\")")
    print(f"  Astrometric Location Error: {astrometrics['error_mas']:.2f} mas ({astrometrics['error_pixels']:.3f} pixels)")
    print(f"  True theta_E,sub: {theta_E_sub:.4f}\" | Recovered theta_E,sub: {recovered_theta_E:.4f}\"")
    print(f"  True Mass Fraction: {(theta_E_sub / theta_E_host)**2:.4e} | Recovered: {recovered_f_sub:.4e}")
    print(f"  Chi-Square Drop: chi^2_red {macro_res['chi2_red']:.2f} -> {pinn_res['chi2_final']:.2f}")
    print(f"  Source Reconstruction: PSNR = {img_metrics['psnr_db']:.2f} dB, SSIM = {img_metrics['ssim']:.4f}")
    print("\n  PHYSICAL MASS DEGENERACY THEOREM:")
    print("  Because redshifts are absent, the physical mass M_sun is not a single number,")
    print("  but a family defined along the line of sight:")
    print("  M_sub(z_l, z_s) = (c^2 / 4G) * (D_s / (D_l * D_ls)) * D_l^2 * pi * theta_E^2")
    print("  Any future spectroscopic measurement of z_l and z_s directly breaks this degeneracy.")
    print("=" * 80)

    # 6. Plot Diagnostic Figure
    plot_dict = {
        "simulator": sim,
        "obs_image": obs_img,
        "res_macro": macro_res["res_macro"],
        "res_final": pinn_res["res_final"],
        "pred_final": pinn_res["pred_final"],
        "source_true": data["source_true"],
        "source_final": pinn_res["final_source"],
        "radar_scores": radar_res["score_map"],
        "radar_extent": radar_res["extent"],
        "sub_true_pos": true_pos,
        "radar_pos": radar_res["best_candidate"],
        "pinn_pos": pinn_pos,
        "radar_score": radar_res["max_score"],
        "chi2_macro": macro_res["chi2_red"],
        "chi2_final": pinn_res["chi2_final"],
        "error_mas": astrometrics["error_mas"],
        "psnr_db": img_metrics["psnr_db"],
        "ssim": img_metrics["ssim"],
        "title": "Scale-Free Unknown-Redshift Benchmark"
    }
    sl.plot_diagnostic_figure(plot_dict, save_path="demo_without_redshift_results.png")
    print(">>> Diagnostic figure successfully saved to: demo_without_redshift_results.png\n")


if __name__ == "__main__":
    main()
