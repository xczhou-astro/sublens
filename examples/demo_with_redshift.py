"""
Example: End-to-End Inversion in the Redshift-Aware Regime (Known z_l, z_s).

Demonstrates:
- Setting up physical cosmology (z_l = 0.222, z_s = 0.609)
- Physical mass conversions (M_sun <-> NFW deflection)
- Three-stage inversion:
  * Stage 1: Macro baseline fit (EPL + Shear + TV source)
  * Stage 2: Astrometric Radar blind scan (annular matched filter)
  * Stage 3: PINN continuous autograd refinement
- Computing physical subhalo mass from converged parameters
- Generating publication diagnostic plot
"""

import time
import torch
import sublens as sl


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[SubLens Demo] Redshift-Aware Benchmark (SDSS J0946+5106) on {device}")
    print("=" * 80)

    # 1. Create landmark benchmark system
    data = sl.create_j0946_benchmark(device=device)
    sim = data["simulator"]
    cosmo = data["cosmology"]
    obs_img = data["obs_image"]
    a_macro = data["alpha_macro"]
    true_pos = data["sub_true_pos"]
    true_mass = data["sub_true_mass"]

    print(f"Target System: {data['name']}")
    print(f"Cosmology: z_l = {cosmo.z_l}, z_s = {cosmo.z_s} | Sigma_cr = {cosmo.sigma_cr:.2e}")
    print(f"Ground Truth Subhalo: M_sub = {true_mass:.2e} M_sun at {true_pos} arcsec")

    # 2. Stage 1: Smooth Macro Baseline Fit
    print("\n>>> Stage 1: Fitting smooth macro baseline (TV pixelated source)...")
    fitter = sl.MacroFitter(sim, tv_weight=4e-4, lr=0.08, num_steps=120)
    macro_res = fitter.fit(obs_img, a_macro, verbose=True)

    # 3. Stage 2: Astrometric Radar Matched Filter Blind Scan
    print("\n>>> Stage 2: Scanning Einstein ring with Astrometric Radar...")
    radar = sl.AstrometricRadar(sim, grid_resolution=45, r_min=0.8, r_max=2.2)
    radar_res = radar.scan(
        obs_img,
        pred_macro=macro_res["pred_macro"],
        res_macro=macro_res["res_macro"],
        source_macro=macro_res["source_reconstructed"],
        alpha_macro=a_macro,
        test_subhalo_mass=true_mass,
        subhalo_type="NFW",
        cosmology=cosmo,
        verbose=True
    )

    # 4. Stage 3: PINN Continuous Autodiff Refinement
    print("\n>>> Stage 3: Running continuous autodiff PINN joint refinement...")
    pinn = sl.PINNOptimizer(sim, lr_pos=0.015, lr_mass=0.02, lr_src=0.05, num_steps=120)
    pinn_res = pinn.refine(
        obs_img,
        init_pos=radar_res["best_candidate"],
        init_src_param=macro_res["source_param"],
        alpha_macro=a_macro,
        vignette_mask=macro_res["vignette"],
        subhalo_type="NFW",
        r_s=data["sub_theta_s"],
        init_kappa=data["sub_kappa_s"],
        verbose=True
    )

    # 5. Evaluate Metrics
    pinn_pos = pinn_res["refined_pos"]
    astrometrics = sl.compute_astrometric_error(pinn_pos, true_pos, pixel_scale=sim.pixel_scale)
    img_metrics = sl.compute_image_metrics(pinn_res["final_source"], data["source_true"])

    # Physical mass recovery
    recovered_mass = cosmo.nfw_to_physical_mass(
        pinn_res["final_kappa_s"], data["sub_theta_s"], c_200=18.0, concentration_boost=25.0
    )

    print("\n" + "=" * 80)
    print(">>> FINAL INVERSION REPORT (REDSHIFT-AWARE REGIME):")
    print(f"  Radar Blind Candidate: ({radar_res['best_candidate'][0]:+.3f}\", {radar_res['best_candidate'][1]:+.3f}\") [S = {radar_res['max_score']:.2f} sigma]")
    print(f"  PINN Converged Position: ({pinn_pos[0]:+.4f}\", {pinn_pos[1]:+.4f}\")")
    print(f"  Nature 2010 Ground Truth: ({true_pos[0]:+.4f}\", {true_pos[1]:+.4f}\")")
    print(f"  Astrometric Location Error: {astrometrics['error_mas']:.2f} mas ({astrometrics['error_pixels']:.3f} pixels)")
    print(f"  True Subhalo Mass: {true_mass:.2e} M_sun | Recovered Mass: {recovered_mass:.2e} M_sun")
    print(f"  Chi-Square Drop: chi^2_red {macro_res['chi2_red']:.2f} -> {pinn_res['chi2_final']:.2f}")
    print(f"  Source Reconstruction: PSNR = {img_metrics['psnr_db']:.2f} dB, SSIM = {img_metrics['ssim']:.4f}")
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
        "title": "SDSS J0946+5106 (Known z_l=0.222, z_s=0.609)"
    }
    sl.plot_diagnostic_figure(plot_dict, save_path="demo_with_redshift_results.png")
    print(">>> Diagnostic figure successfully saved to: demo_with_redshift_results.png\n")


if __name__ == "__main__":
    main()
