"""Detailed review figures for the saved noise scan; no fitting or truth starts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parent / "noise_scan"


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    config = summary["config"]
    factors, seeds = config["factors"], config["noise_seeds"]
    rows = {(r["factor"], r["seed"]): r for r in summary["trials"]}
    matrix = np.full((len(seeds), len(factors)), np.nan)
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    for iy, seed in enumerate(seeds):
        for ix, factor in enumerate(factors):
            row = rows[(factor, seed)]
            if not row["completed"]:
                continue
            matrix[iy, ix] = int(row["recovered"])
            label = "error" if "position_error_mas" not in row else (
                f"{row['position_error_mas']:.0f} mas\n{100*row['strength_error_fraction']:+.0f}%")
            ax.text(ix, iy, label, ha="center", va="center", fontsize=10)
    ax.imshow(matrix, cmap=ListedColormap(["#f4b79c", "#a9d7c4"]), vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(len(factors)), xticklabels=[f"{f:g}x\nRMS={config['base_rms']*f:g}" for f in factors],
           yticks=range(len(seeds)), yticklabels=seeds, xlabel="Noise level (same field scaled within each row)",
           ylabel="Independent noise seed", title="Green: converged, position <=50 mas and strength error <=30%")
    fig.savefig(ROOT / "paired_recovery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # A fixed, preselected seed shows every noise level; it is not selected by fit quality.
    seed = seeds[0]
    clean = fits.getdata(ROOT / "reference" / "clean_with_subhalo.fits")
    n = len(factors)
    fig, axes = plt.subplots(3, n, figsize=(3*n, 8.5), layout="constrained")
    extent = (-2.5, 2.5, -2.5, 2.5)
    for col, factor in enumerate(factors):
        row = rows[(factor, seed)]
        path = ROOT / "trials" / row["id"] / "result.fits"
        if not path.exists():
            for ax in axes[:, col]:
                ax.axis("off")
            continue
        with fits.open(path) as hdus:
            image = hdus["SCI"].data
            rms, mask = hdus["RMS"].data, hdus["MASK"].data.astype(bool)
            axes[0, col].imshow(image, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=clean.max())
            axes[1, col].imshow(np.where(mask, hdus["RESID_MACRO"].data/rms, np.nan),
                                origin="lower", extent=extent, cmap="RdBu_r", vmin=-4, vmax=4)
            axes[2, col].imshow(np.where(mask, hdus["RESIDUAL"].data/rms, np.nan),
                                origin="lower", extent=extent, cmap="RdBu_r", vmin=-4, vmax=4)
        axes[0, col].set_title(f"{factor:g}x noise; RMS={row['noise_rms']:g}\nρ_image={row['rho_image']:.1f}")
        axes[2, col].set_xlabel(f"{row.get('position_error_mas', float('nan')):.1f} mas; "
                               f"{100*row.get('strength_error_fraction', float('nan')):+.1f}%")
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("Observation\ncommon flux scale")
    axes[1, 0].set_ylabel("Macro residual / RMS\ncommon ±4 scale")
    axes[2, 0].set_ylabel("Macro + halo residual / RMS\ncommon ±4 scale")
    fig.suptitle(f"Paired noise fields, fixed seed {seed}; selected-candidate errors shown below")
    fig.savefig(ROOT / "noise_gallery.png", dpi=155, bbox_inches="tight")
    plt.close(fig)

    signal = json.loads((ROOT / "reference" / "signal_metrics.json").read_text())
    strengths = signal["strengths"]
    b = np.array([s["strength_arcsec"] for s in strengths])
    rho = np.array([s["rho_at_baseline"] for s in strengths])
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    ax.loglog(b, rho, "o-", label="Exact forward-model image difference")
    ax.loglog(b, rho[0]*b/b[0], "--", label="Linear scaling from weakest injection")
    ax.set(xlabel="Angular strength [arcsec]", ylabel="ρ_image at RMS=0.012",
           title="A strength/noise scaling is only an approximation")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(ROOT / "strength_snr_relation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # The two-dimensional table is a forward-model signal diagnostic, not a
    # second strength-by-noise recovery benchmark.
    grid = np.array([[s["signal_l2"]/(config["base_rms"]*f) for f in factors] for s in strengths])
    records = [(s["strength_arcsec"], config["base_rms"]*f, grid[i, j])
               for i, s in enumerate(strengths) for j, f in enumerate(factors)]
    np.savetxt(ROOT / "strength_noise_snr.csv", records, delimiter=",", comments="",
               header="strength_arcsec,noise_rms_image_units,rho_image")

    ready = [g for g in summary["groups"] if g["completed"]]
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    for g in ready:
        rate = g["recovery_rate"]
        low, high = g["recovery_wilson95"]
        ax.errorbar(g["rho_image"], rate, yerr=[[max(0., rate-low)], [max(0., high-rate)]],
                    fmt="o", color="#2374ab", capsize=4)
        ax.annotate(f"{g['recovered']}/{g['completed']}", (g["rho_image"], rate),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.plot([g["rho_image"] for g in ready], [g["recovery_rate"] for g in ready], alpha=.5, color="#2374ab")
    ax.set(xscale="log", xlabel="Injected image perturbation SNR ρ_image", ylabel="Accurate recovery fraction",
           title="Fixed 0.05 arcsec halo; noise varied; Wilson 95% intervals", ylim=(-.04, 1.1),
           xticks=[g["rho_image"] for g in ready], xticklabels=[f"{g['rho_image']:.1f}" for g in ready])
    ax.invert_xaxis()
    ax.minorticks_off()
    ax.grid(alpha=.2)
    fig.savefig(ROOT / "recovery_vs_snr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
