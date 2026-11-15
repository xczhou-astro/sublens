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
