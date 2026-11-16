"""
Publication-Quality Scientific Visualization Toolkit.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_diagnostic_figure(
    results: Dict[str, Any],
    save_path: str = "lens_inversion_results.png",
    dpi: int = 300
):
    """
    Renders an 8-panel publication-grade diagnostic figure:
    1. Observation Image (with ground truth subhalo)
    2. Smooth Macro Residual (exposing astrometric defect)
    3. Radar S-Map (blind detection response)
    4. PINN Autodiff Sub-mas Refinement
    5. Final Inverted Model Image
    6. Final Annihilated Residual
    7. Ground Truth Source
    8. Reconstructed Source (TV Regularized)
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    cmap_img = 'inferno'
    cmap_res = 'RdBu_r'

    sim = results["simulator"]
    obs = results["obs_image"].squeeze().cpu().numpy()
    res_m = results["res_macro"].squeeze().cpu().numpy()
    res_f = results["res_final"].squeeze().cpu().numpy()
    pred_f = results["pred_final"].squeeze().cpu().numpy()
    src_true = results["source_true"].squeeze().cpu().numpy()
    src_recon = results["source_final"].squeeze().cpu().numpy()
    score_map = results["radar_scores"].cpu().numpy()

    true_pos = results.get("sub_true_pos", (0.0, 0.0))
    radar_pos = results.get("radar_pos", (0.0, 0.0))
    pinn_pos = results.get("pinn_pos", (0.0, 0.0))

    # (1) Observation
    ax = axes[0, 0]
    im = ax.imshow(obs, origin='lower', extent=sim.extent_img, cmap=cmap_img)
    ax.scatter([true_pos[0]], [true_pos[1]], facecolors='none', edgecolors='cyan', s=160, linewidth=2, label='True Subhalo')
    ax.set_title(f"1. Simulated Observation\n({results.get('title', 'Lensing System')})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    ax.legend(loc='lower left', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2) Macro Residual
    ax = axes[0, 1]
    vmax_r = max(abs(res_m.min()), abs(res_m.max()))
    im = ax.imshow(res_m, origin='lower', extent=sim.extent_img, cmap=cmap_res, vmin=-vmax_r, vmax=vmax_r)
    ax.scatter([true_pos[0]], [true_pos[1]], marker='+', color='black', s=120, linewidth=2, label='True Subhalo')
    ax.set_title(f"2. Smooth Macro Residual (No Subhalo)\nchi_red^2 = {results.get('chi2_macro', 0.0):.2f} (Astrometric Defect)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    ax.legend(loc='lower left', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (3) Radar Score Map
    ax = axes[0, 2]
    extent_r = results.get("radar_extent", sim.extent_img)
    im = ax.imshow(score_map, origin='lower', extent=extent_r, cmap='viridis')
    ax.scatter([radar_pos[0]], [radar_pos[1]], color='yellow', marker='*', s=220, edgecolors='black', label=f"Radar Peak ({results.get('radar_score', 0.0):.1f} sigma)")
    ax.scatter([true_pos[0]], [true_pos[1]], facecolors='none', edgecolors='red', s=160, linewidth=2, linestyle='--', label='True Pos')
    ax.set_title(f"3. Astrometric Radar S-Map\nPeak S = {results.get('radar_score', 0.0):.1f} sigma (Blind Found)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    ax.legend(loc='lower left', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (4) PINN Refinement
    ax = axes[0, 3]
    d_x = np.linspace(-0.20, 0.20, 100)
    d_y = np.linspace(-0.20, 0.20, 100)
    xx, yy = np.meshgrid(d_x, d_y)
    dist_field = np.sqrt((xx - (pinn_pos[0] - radar_pos[0]))**2 + (yy - (pinn_pos[1] - radar_pos[1]))**2)
    im = ax.imshow(dist_field * 1000.0, origin='lower', extent=[-200, 200, -200, 200], cmap='plasma')
    ax.scatter([0], [0], color='yellow', marker='*', s=180, edgecolors='black', label='Radar Seed')
    ax.scatter([(pinn_pos[0] - radar_pos[0]) * 1000], [(pinn_pos[1] - radar_pos[1]) * 1000], color='lime', marker='o', s=140, edgecolors='black', label='PINN Refined')
    ax.scatter([(true_pos[0] - radar_pos[0]) * 1000], [(true_pos[1] - radar_pos[1]) * 1000], color='cyan', marker='x', s=140, linewidth=2, label='Ground Truth')
    err_mas = results.get("error_mas", 0.0)
    ax.set_title(f"4. PINN Autodiff Sub-mas Refinement\nAstrometric Error: {err_mas:.2f} mas", fontsize=11, fontweight='bold')
    ax.set_xlabel("Delta x (mas)")
    ax.set_ylabel("Delta y (mas)")
    ax.legend(loc='lower left', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (5) Final Model
    ax = axes[1, 0]
    im = ax.imshow(pred_f, origin='lower', extent=sim.extent_img, cmap=cmap_img)
    ax.scatter([pinn_pos[0]], [pinn_pos[1]], facecolors='none', edgecolors='lime', s=160, linewidth=2, label='Inverted Subhalo')
    ax.set_title("5. Final Inverted Model Image\nJoint Macro + Subhalo", fontsize=11, fontweight='bold')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    ax.legend(loc='lower left', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (6) Final Residual
    ax = axes[1, 1]
    im = ax.imshow(res_f, origin='lower', extent=sim.extent_img, cmap=cmap_res, vmin=-vmax_r, vmax=vmax_r)
    ax.set_title(f"6. Final Inversion Residual\nchi_red^2 = {results.get('chi2_final', 1.0):.2f} (Noise Limit)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (7) True Source
    ax = axes[1, 2]
    im = ax.imshow(src_true, origin='lower', extent=sim.extent_src, cmap='magma')
    ax.set_title("7. Ground Truth Source", fontsize=11, fontweight='bold')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (8) Reconstructed Source
    ax = axes[1, 3]
    im = ax.imshow(src_recon, origin='lower', extent=sim.extent_src, cmap='magma')
    psnr = results.get("psnr_db", 0.0)
    ssim = results.get("ssim", 0.0)
    ax.set_title(f"8. Reconstructed Source (TV Regularized)\nPSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Arcsec")
    ax.set_ylabel("Arcsec")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_real_observation_figure(
    results: Dict[str, Any],
    save_path: str = "real_lens_results.png",
    dpi: int = 300
):
    """
    Renders an 8-panel publication-grade astronomical figure for real space telescope lenses:
    (a) Real Lensed Arc (observed science image, cyan circle at detected subhalo)
    (b) Pixel Noise RMS Map
    (c) Astrometric Radar Response (plasma colormap, yellow cross at peak)
    (d) 1D Radar Cut across Candidate (cross-section through peak with 4-sigma threshold)
    (e) Macro Model Arc Prediction (smooth lens model)
    (f) Macro Model Normalized Residual (chi^2_red of unmodeled perturber)
    (g) Joint Inversion Final Residual (subhalo included, reduced to noise limit)
    (h) Reconstructed Background Source (high-z source galaxy morphology)
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=dpi)
    plt.subplots_adjust(wspace=0.28, hspace=0.26, left=0.04, right=0.96, top=0.92, bottom=0.06)

    obs_img = results["obs_image"].squeeze().cpu().numpy()
    noise_map = results.get("noise_map")
    if noise_map is not None:
        noise_img = noise_map.squeeze().cpu().numpy()
    else:
        noise_img = np.ones_like(obs_img) * 0.01

    pred_macro = results.get("pred_macro", results["pred_final"]).squeeze().cpu().numpy()
    pred_final = results["pred_final"].squeeze().cpu().numpy()
    res_macro = results.get("res_macro", obs_img - pred_macro).squeeze().cpu().numpy()
    res_final = results["res_final"].squeeze().cpu().numpy()
    source_np = results["source_final"].squeeze().cpu().numpy()
    score_map = results["radar_scores"].squeeze().cpu().numpy()
    scan_coords = results.get("radar_coords", np.linspace(-1.5, 1.5, score_map.shape[0]))

    extent_img = results.get("extent_img", [-5.0, 5.0, -5.0, 5.0])
    radar_extent = results.get("radar_extent", extent_img)
    src_extent = results.get("source_extent", [-1.0, 1.0, -1.0, 1.0])

    radar_pos = results.get("radar_pos", (0.0, 0.0))
    pinn_pos = results.get("pinn_pos", radar_pos)
    radar_score = results.get("radar_score", 0.0)
    chi2_base = results.get("chi2_macro", 1.0)
    chi2_final = results.get("chi2_final", 1.0)
    lens_id = results.get("lens_id", "Euclid Q1 Lens")
    theta_e = results.get("theta_e", 1.0)

    # (a) Real Lensed Arc
    v_max = float(np.percentile(obs_img, 99.5))
    v_min = float(np.percentile(obs_img, 5))
    im1 = axes[0, 0].imshow(obs_img, origin='lower', extent=extent_img, cmap='magma', vmin=v_min, vmax=max(v_max, v_min + 1e-4))
    axes[0, 0].scatter([pinn_pos[0]], [pinn_pos[1]], color='cyan', marker='o', facecolors='none', s=130, linewidth=2, label="Detected Subhalo")
    axes[0, 0].set_title(rf"(a) Real Euclid VIS Lensed Arc" + f"\n[{str(lens_id)[:18]}...]", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel(r"$\theta_x\ (\mathrm{arcsec})$", fontsize=10)
    axes[0, 0].set_ylabel(r"$\theta_y\ (\mathrm{arcsec})$", fontsize=10)
    axes[0, 0].legend(loc='upper left', fontsize=8.5, framealpha=0.85)
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label=r"Flux $[\mathrm{counts/s}]$")

    # (b) Noise Map
    im2 = axes[0, 1].imshow(noise_img, origin='lower', extent=extent_img, cmap='viridis')
    axes[0, 1].set_title(rf"(b) Real Euclid VIS Noise Map ($\sigma_{{\mathrm{{RMS}}}}$)" + f"\n[Median: {float(np.median(noise_img)):.4f}]", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel(r"$\theta_x\ (\mathrm{arcsec})$", fontsize=10)
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label=r"Noise RMS")

    # (c) Radar S-Map
    im3 = axes[0, 2].imshow(score_map, origin='lower', extent=radar_extent, cmap='plasma')
    axes[0, 2].scatter([radar_pos[0]], [radar_pos[1]], color='yellow', marker='x', s=100, linewidth=2, label=rf"Radar Peak ($S={radar_score:.2f}\sigma$)")
    axes[0, 2].set_title(rf"(c) Astrometric Radar Response" + "\n" + rf"[Significance: $S={radar_score:.2f}\sigma$]", fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel(r"$\theta_x\ (\mathrm{arcsec})$", fontsize=10)
    axes[0, 2].legend(loc='upper left', fontsize=8.5, framealpha=0.85)
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04, label=r"Significance $[S/\sigma]$")

    # (d) 1D Radar Cut across Candidate
    if score_map.ndim == 2:
        y_idx = int(np.clip(int((radar_pos[1] - radar_extent[2]) / (radar_extent[3] - radar_extent[2]) * score_map.shape[0]), 0, score_map.shape[0]-1))
        profile_1d = score_map[y_idx, :]
    else:
        profile_1d = np.zeros(len(scan_coords))
    axes[0, 3].plot(scan_coords, profile_1d, color='crimson', lw=2.2, label=rf"Radar Cut ($\theta_y={radar_pos[1]:+.2f}''$)")
    axes[0, 3].axvline(radar_pos[0], color='navy', linestyle='--', lw=1.5, label=rf"Peak ($\theta_x={radar_pos[0]:+.2f}''$)")
    axes[0, 3].axhline(4.0, color='gray', linestyle=':', label=r"$4\sigma$ Threshold")
    axes[0, 3].set_title(r"(d) 1D Radar Cut across Candidate" + "\n" + rf"[{'Detection Peak' if radar_score >= 4.0 else 'Noise Background'}]", fontsize=11, fontweight='bold')
    axes[0, 3].set_xlabel(r"$\theta_x\ (\mathrm{arcsec})$", fontsize=10)
    axes[0, 3].set_ylabel(r"Significance $[S/\sigma]$", fontsize=10)
    axes[0, 3].legend(loc='upper left', fontsize=8.5, framealpha=0.85)
    axes[0, 3].grid(True, linestyle='--', alpha=0.5)

    # (e) Macro Model Arc Prediction
    im5 = axes[1, 0].imshow(pred_macro, origin='lower', extent=extent_img, cmap='magma', vmin=v_min, vmax=max(v_max, v_min + 1e-4))
    axes[1, 0].set_title(rf"(e) Macro Model Arc Prediction" + f"\n[Smooth Lens, " + r"$\theta_E=" + f"{theta_e:.3f}''$]", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel(r"$\theta_x\ (\mathrm{arcsec})$", fontsize=10)
    axes[1, 0].set_ylabel(r"$\theta_y\ (\mathrm{arcsec})$", fontsize=10)
    fig.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04, label=r"Flux $[\mathrm{counts/s}]$")

    # (f) Macro Model Normalized Residual
    fit_mask = results.get("fit_mask")
    norm_res_macro = res_macro / (noise_img + 1e-8)
    if fit_mask is not None:
        mask_np = fit_mask.squeeze().cpu().numpy().astype(bool) if isinstance(fit_mask, torch.Tensor) else np.array(fit_mask, dtype=bool)
        norm_res_macro = np.where(mask_np, norm_res_macro, np.nan)
    im6 = axes[1, 1].imshow(norm_res_macro, origin='lower', extent=extent_img, cmap='bwr', vmin=-3, vmax=3)
    axes[1, 1].scatter([radar_pos[0]], [radar_pos[1]], color='black', marker='x', s=90, linewidth=2, label="Subhalo Candidate")
    axes[1, 1].set_title(rf"(f) Macro Model Normalized Residual" + f"\n[Unmodeled Perturber, " + r"$\chi^2_{\mathrm{red}}=" + f"{chi2_base:.3f}$]", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel(r"$\theta_x\ (\mathrm{arcsec})$", fontsize=10)
    axes[1, 1].legend(loc='upper left', fontsize=8.5, framealpha=0.85)
    fig.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04, label=r"Residual $[\sigma]$")

    # (g) Final Joint Residual
    norm_res_final = res_final / (noise_img + 1e-8)
    if fit_mask is not None:
        norm_res_final = np.where(mask_np, norm_res_final, np.nan)
    im7 = axes[1, 2].imshow(norm_res_final, origin='lower', extent=extent_img, cmap='bwr', vmin=-3, vmax=3)
    axes[1, 2].set_title(rf"(g) Joint Inversion Final Residual" + f"\n[Subhalo Included, " + r"$\chi^2_{\mathrm{red}}=" + f"{chi2_final:.3f}$]", fontsize=11, fontweight='bold')
    axes[1, 2].set_xlabel(r"$\theta_x\ (\mathrm{arcsec})$", fontsize=10)
    fig.colorbar(im7, ax=axes[1, 2], fraction=0.046, pad=0.04, label=r"Residual $[\sigma]$")

    # (h) Reconstructed Source
    im8 = axes[1, 3].imshow(source_np, origin='lower', extent=src_extent, cmap='magma')
    axes[1, 3].set_title(r"(h) Reconstructed Background Source" + f"\n[High-z Source Galaxy Morphology]", fontsize=11, fontweight='bold')
    axes[1, 3].set_xlabel(r"$\beta_x\ (\mathrm{arcsec})$", fontsize=10)
    fig.colorbar(im8, ax=axes[1, 3], fraction=0.046, pad=0.04, label=r"Surface Brightness")

    fig.suptitle(rf"Space Telescope Observation: Euclid Q1 Strong Lens Inversion ({lens_id})",
                 fontsize=15, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

