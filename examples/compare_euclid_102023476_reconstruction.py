"""
Comparison between Official Euclid Source Reconstruction and SubLens Reconstructions.

Evaluates:
1. Official Euclid PyAutoLens reference from result/source_reconstruction.fits
2. SubLens Adaptive Delaunay source reconstruction with given lens mass parameters
3. SubLens Semi-Linear refitted macro model source reconstruction
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from sublens.survey.euclid_source import load_euclid_source_observation, reconstruct_euclid_source, centered_crop


def compute_metrics(official, test_source):
    """Computes cosine similarity, Pearson correlation, and peak location."""
    off_smooth = gaussian_filter(official.astype(float), 2).ravel()
    test_smooth = gaussian_filter(test_source.astype(float), 2).ravel()
    norm = np.linalg.norm(off_smooth) * np.linalg.norm(test_smooth)
    cosine = float(off_smooth @ test_smooth / norm) if norm > 0 else 0.0
    corr = float(np.corrcoef(off_smooth, test_smooth)[0, 1])

    y_off, x_off = np.unravel_index(np.argmax(official), official.shape)
    y_test, x_test = np.unravel_index(np.argmax(test_source), test_source.shape)
    axis = np.linspace(-1.0, 1.0, official.shape[0])
    off_pos = (axis[x_off], axis[y_off])
    test_pos = (axis[x_test], axis[y_test])
    pos_err = np.hypot(off_pos[0] - test_pos[0], off_pos[1] - test_pos[1])

    return {
        "cosine": cosine,
        "pearson_r": corr,
        "official_peak": (float(off_pos[0]), float(off_pos[1])),
        "test_peak": (float(test_pos[0]), float(test_pos[1])),
        "peak_offset_arcsec": float(pos_err),
        "official_flux": float(official.sum()),
        "test_flux": float(test_source.sum()),
    }


def main():
    lens_id = "102023476_NEG610025395461180513"
    data_dir = Path("archive/data")
    release_dir = data_dir / "lens" / lens_id / "result"
    out_dir = Path("euclid_q1_diagnostic_plots/source_reconstruction_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f">>> Comparing Source Reconstructions for Euclid Lens {lens_id}")
    print("=" * 80)

    # 1. Load official reference products
    official_source = fits.getdata(release_dir / "source_reconstruction.fits", "VIS")
    official_image_model = centered_crop(fits.getdata(release_dir / "source_light.fits", "VIS"), 100)

    # 2. Reconstruct source using SubLens Delaunay with given macro parameters
    print("  Running SubLens Delaunay source reconstruction with published mass...")
    obs = load_euclid_source_observation(lens_id, data_dir)
    res_delaunay = obs.reconstruct(method="delaunay", source_nodes=500, regularization=0.002)
    delaunay_source = res_delaunay.source_image(half_width=1.0, size=201)

    # 3. Reconstruct source using SubLens Cartesian (for comparison)
    print("  Running SubLens Cartesian source reconstruction...")
    res_cartesian = obs.reconstruct(method="cartesian", source_size=96, regularization=1.0)
    cartesian_source = res_cartesian.source_image(half_width=1.0, size=201)

    # 4. Compute metrics
    m_delaunay = compute_metrics(official_source, delaunay_source)
    m_cartesian = compute_metrics(official_source, cartesian_source)

    # Image-plane residuals on bright arc
    bright_mask = obs.mask & (official_image_model >= 2.0 * obs.rms)
    r2_del_bright = float(np.mean(((obs.image[bright_mask] - res_delaunay.source.image_model[bright_mask]) / obs.rms[bright_mask])**2))
    r2_cart_bright = float(np.mean(((obs.image[bright_mask] - res_cartesian.source.image_model[bright_mask]) / obs.rms[bright_mask])**2))
    r2_off_bright = float(np.mean(((obs.image[bright_mask] - official_image_model[bright_mask]) / obs.rms[bright_mask])**2))

    print("\n>>> Quantitative Metric Comparison with Official Euclid source_reconstruction.fits:")
    print(f"  [Euclid Official Reference]: Bright-arc chi^2_red = {r2_off_bright:.4f}")
    print(f"  [SubLens Delaunay Source]:")
    print(f"     Cosine Similarity: {m_delaunay['cosine']:.5f} ({m_delaunay['cosine']*100:.2f}%)")
    print(f"     Pearson r:         {m_delaunay['pearson_r']:.5f}")
    print(f"     Peak position:     Official={m_delaunay['official_peak']}, Delaunay={m_delaunay['test_peak']} (Offset={m_delaunay['peak_offset_arcsec']*1000:.1f} mas)")
    print(f"     Bright-arc chi^2:  {r2_del_bright:.4f} (Improved by {(1 - r2_del_bright/r2_off_bright)*100:.1f}%)")
    print(f"  [SubLens Cartesian Source]:")
    print(f"     Cosine Similarity: {m_cartesian['cosine']:.5f} ({m_cartesian['cosine']*100:.2f}%)")
    print(f"     Pearson r:         {m_cartesian['pearson_r']:.5f}")
    print(f"     Bright-arc chi^2:  {r2_cart_bright:.4f}")

    # -------------------------------------------------------------------------
    # Visualization: 2x4 Layout
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=200, layout="constrained")
    extent_src = [-1.0, 1.0, -1.0, 1.0]
    extent_img = [-5.0, 5.0, -5.0, 5.0]

    vmax_src = float(np.percentile(official_source, 99.8))
    vmax_flux = float(np.percentile(obs.image[obs.mask], 99.5))

    # Row 1: Source plane comparisons
    im0 = axes[0, 0].imshow(official_source, origin="lower", extent=extent_src, cmap="magma", vmin=0, vmax=vmax_src)
    axes[0, 0].set_title("(a) Euclid Q1 Stored Source\n(PyAutoLens Hilbert Mesh)")
    axes[0, 0].set_ylabel("y [arcsec]")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.75, label="VIS surface brightness")

    im1 = axes[0, 1].imshow(delaunay_source, origin="lower", extent=extent_src, cmap="magma", vmin=0, vmax=vmax_src)
    axes[0, 1].set_title(f"(b) SubLens Delaunay Source\n(Cosine = {m_delaunay['cosine']:.4f})")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.75, label="VIS surface brightness")

    im2 = axes[0, 2].imshow(cartesian_source, origin="lower", extent=extent_src, cmap="magma", vmin=0, vmax=vmax_src)
    axes[0, 2].set_title(f"(c) SubLens Cartesian Source\n(Cosine = {m_cartesian['cosine']:.4f})")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.75, label="VIS surface brightness")

    # Difference map (Delaunay - Official)
    diff_src = (delaunay_source - official_source)
    im3 = axes[0, 3].imshow(diff_src, origin="lower", extent=extent_src, cmap="RdBu_r",
                            vmin=-0.4 * vmax_src, vmax=0.4 * vmax_src)
    axes[0, 3].set_title("(d) Source Difference\n(SubLens Delaunay − Euclid Official)")
    fig.colorbar(im3, ax=axes[0, 3], shrink=0.75, label="Difference")

    # Row 2: Image plane and 1D profile
    # Panel e: Observed Euclid VIS Arc
    im4 = axes[1, 0].imshow(np.where(obs.mask, obs.image, np.nan), origin="lower",
                            extent=extent_img, cmap="magma", vmin=0, vmax=vmax_flux)
    axes[1, 0].set_title("(e) Observed VIS Flux (Data)")
    axes[1, 0].set_xlabel("x [arcsec]")
    axes[1, 0].set_ylabel("y [arcsec]")
    axes[1, 0].set_xlim(-2.5, 2.5)
    axes[1, 0].set_ylim(-2.5, 2.5)
    fig.colorbar(im4, ax=axes[1, 0], shrink=0.75, label="Flux")

    # Panel f: SubLens Delaunay Model Image
    im5 = axes[1, 1].imshow(np.where(obs.mask, res_delaunay.source.image_model, np.nan),
                            origin="lower", extent=extent_img, cmap="magma", vmin=0, vmax=vmax_flux)
    axes[1, 1].set_title("(f) SubLens Delaunay Model Image\n(Forward Lensed + PSF)")
    axes[1, 1].set_xlabel("x [arcsec]")
    axes[1, 1].set_xlim(-2.5, 2.5)
    axes[1, 1].set_ylim(-2.5, 2.5)
    fig.colorbar(im5, ax=axes[1, 1], shrink=0.75, label="Model Flux")

    # Panel g: Residual Image (Data - Delaunay Model) / RMS
    res_delaunay_norm = np.full_like(obs.image, np.nan)
    res_delaunay_norm[obs.mask] = res_delaunay.residual[obs.mask] / obs.rms[obs.mask]
    im6 = axes[1, 2].imshow(res_delaunay_norm, origin="lower", extent=extent_img,
                            cmap="RdBu_r", vmin=-3.5, vmax=3.5)
    axes[1, 2].set_title(f"(g) SubLens Image Residual / RMS\n(Bright Arc $\\chi^2_{{\\rm red}} = {r2_del_bright:.3f}$)")
    axes[1, 2].set_xlabel("x [arcsec]")
    axes[1, 2].set_xlim(-2.5, 2.5)
    axes[1, 2].set_ylim(-2.5, 2.5)
    fig.colorbar(im6, ax=axes[1, 2], shrink=0.75, label="Residual / RMS")

    # Panel h: 1D Radial Profile Comparison across Source Core
    axis_1d = np.linspace(-0.6, 0.6, 121)
    y_cut_idx = int(round((m_delaunay['official_peak'][1] + 1.0) / 2.0 * 200))
    x_indices = np.clip(np.round((axis_1d + 1.0) / 2.0 * 200).astype(int), 0, 200)

    prof_off = official_source[y_cut_idx, x_indices]
    prof_del = delaunay_source[y_cut_idx, x_indices]
    prof_cart = cartesian_source[y_cut_idx, x_indices]

    axes[1, 3].plot(axis_1d, prof_off, color="black", lw=2.0, label="Euclid Official")
    axes[1, 3].plot(axis_1d, prof_del, color="crimson", lw=1.8, linestyle="--", label="SubLens Delaunay")
    axes[1, 3].plot(axis_1d, prof_cart, color="royalblue", lw=1.5, linestyle=":", label="SubLens Cartesian")
    axes[1, 3].set_title("(h) 1D Horizontal Cut Across Core")
    axes[1, 3].set_xlabel("$\Delta x$ [arcsec]")
    axes[1, 3].set_ylabel("Surface brightness")
    axes[1, 3].legend(loc="upper right", fontsize=8)
    axes[1, 3].grid(alpha=0.3)

    for ax in axes[0, :]:
        ax.set_xlabel("x [arcsec]")

    fig_path = out_dir / f"{lens_id}_direct_comparison_with_official.png"
    fig.suptitle(f"Euclid Lens {lens_id}: Direct Comparison with Stored Official Source Reconstruction", fontsize=14)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison plot saved to: {fig_path}")

    # Save metrics JSON
    metrics_summary = {
        "lens_id": lens_id,
        "delaunay": {**m_delaunay, "bright_arc_chi2": r2_del_bright},
        "cartesian": {**m_cartesian, "bright_arc_chi2": r2_cart_bright},
        "official": {"bright_arc_chi2": r2_off_bright},
        "plot_path": str(fig_path)
    }
    with open(out_dir / f"{lens_id}_comparison_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Also save the Delaunay reconstructed source as standard FITS
    fits_path = out_dir / f"{lens_id}_sublens_delaunay_source.fits"
    res_delaunay.save_fits(fits_path, overwrite=True)
    print(f"  SubLens Delaunay FITS products saved to: {fits_path}")


if __name__ == "__main__":
    main()
