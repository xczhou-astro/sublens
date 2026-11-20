"""Benchmark source-only fits against the Euclid Q1 VIS model products."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from sublens.survey import centered_crop, get_available_grade_a_lenses, load_euclid_source_observation


METHODS = {
    "Cartesian": dict(method="cartesian"),
    "Delaunay": dict(method="delaunay"),
}


def mean_square_residual(data, model, rms, mask):
    return float(np.mean(((data[mask] - model[mask]) / rms[mask])**2))


def source_similarity(left, right):
    """Cosine similarity at common sky coordinates after 0.02 arcsec Gaussian smoothing."""
    left = gaussian_filter(left.astype(float), 2).ravel()
    right = gaussian_filter(right.astype(float), 2).ravel()
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / norm) if norm else 0.0


def validation_blocks(mask):
    row, col = np.indices(mask.shape)
    for phase in (0, 5):
        yield mask & ((row + phase) % 10 < 4) & ((col + phase) % 10 < 4)


def plot_comparison(obs, models, output_path):
    fig, axes = plt.subplots(len(models), 3, figsize=(9.7, 9.3), layout="constrained")
    reference = models[0][1]
    r, c = np.where(reference >= 0.08 * reference.max())
    # The historical Q1 exporter used a 201-by-201 grid with centres from -1 to +1 arcsec.
    source_limits = ((max(-1, (c.min() - 12) / 100 - 1), min(1, (c.max() + 12) / 100 - 1)),
                     (max(-1, (r.min() - 12) / 100 - 1), min(1, (r.max() + 12) / 100 - 1)))
    half = obs.image.shape[0] * obs.pixel_scale / 2
    width = max(1, (np.max(np.abs(np.argwhere(obs.mask) - (obs.image.shape[0]-1)/2)) + 1) * obs.pixel_scale)
    flux_max = float(np.percentile(models[0][2][obs.mask], 99.5))
    for ax, (name, source, model) in zip(axes, models):
        source_plot = ax[0].imshow(source, origin="lower", extent=(-1.005, 1.005, -1.005, 1.005),
                                  vmin=0, vmax=0.8 * reference.max(), cmap="magma")
        ax[0].set(xlim=source_limits[0], ylim=source_limits[1], ylabel=f"{name}\ny [arcsec]")
        residual = np.full(obs.image.shape, np.nan)
        residual[obs.mask] = (obs.image[obs.mask] - model[obs.mask]) / obs.rms[obs.mask]
        image_plot = ax[1].imshow(np.where(obs.mask, model, np.nan), origin="lower",
                                 extent=(-half, half, -half, half), vmin=0, vmax=flux_max, cmap="magma")
        residual_plot = ax[2].imshow(residual, origin="lower", extent=(-half, half, -half, half),
                                     vmin=-4, vmax=4, cmap="RdBu_r")
        for image_axis in ax[1:]:
            image_axis.set(xlim=(-width, width), ylim=(-width, width))
    for index, (title, plot, unit) in enumerate([
        ("Source plane", source_plot, "VIS surface brightness"),
        ("Lensed source model", image_plot, "VIS flux"),
        ("Image residual", residual_plot, "(Data − model) / RMS"),
    ]):
        axes[0, index].set_title(title)
        axes[-1, index].set_xlabel("x [arcsec]")
        fig.colorbar(plot, ax=axes[:, index], shrink=0.65, pad=0.03, label=unit)
    fig.suptitle(obs.lens_id)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compare_lens(lens_id, data_dir, output_dir):
    obs = load_euclid_source_observation(lens_id, data_dir)
    reference_dir = data_dir / "lens" / lens_id / "result"
    source = fits.getdata(reference_dir / "source_reconstruction.fits", "VIS")
    model = centered_crop(fits.getdata(reference_dir / "source_light.fits", "VIS"), obs.image.shape[0])
    bright = obs.mask & (model >= 2 * obs.rms)
    if not bright.any():
        raise ValueError(f"{lens_id} has no VIS arc pixels above the reference threshold")

    def metrics(label, src, prediction, test_score):
        return {
            "lens_id": lens_id,
            "method": label,
            "mean_r2_aperture": mean_square_residual(obs.image, prediction, obs.rms, obs.mask),
            "mean_r2_bright_arc": mean_square_residual(obs.image, prediction, obs.rms, bright),
            "source_cosine": source_similarity(source, src),
            "mean_r2_heldout_bright_arc": test_score,
        }

    results = [metrics("Euclid Q1", source, model, None)]
    models = [("Euclid Q1", source, model)]
    for label, options in METHODS.items():
        result = obs.reconstruct(**options)
        reconstructed = result.source_image()
        heldout = []
        for block in validation_blocks(obs.mask):
            if (bright & block).any():
                fit = obs.reconstruct(fit_mask=obs.mask & ~block, **options)
                heldout.extend((fit.residual[bright & block] / obs.rms[bright & block])**2)
        results.append(metrics(label, reconstructed, result.source.image_model,
                               float(np.mean(heldout)) if heldout else None))
        models.append((label, reconstructed, result.source.image_model))
        result.save_fits(output_dir / f"{lens_id}_{result.source.method}.fits", overwrite=True)
    plot_comparison(obs, models, output_dir / f"{lens_id}_sources.png")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("euclid_q1_diagnostic_plots/source_only"))
    parser.add_argument("--lens-id", action="append", help="Repeat to specify multiple systems")
    parser.add_argument("--max-systems", type=int, default=10)
    args = parser.parse_args()
    if not args.lens_id and args.max_systems < 1:
        parser.error("--max-systems must be positive")
    lens_ids = args.lens_id
    if lens_ids is None:
        catalog = get_available_grade_a_lenses(
            args.data_dir / "q1_discovery_engine_lens_catalog.csv", args.data_dir
        )
        lens_ids = [row["id_str"] for row in catalog[:args.max_systems]]
    if not lens_ids:
        parser.error("No Q1 systems with the required data were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lens_id in lens_ids:
        rows.extend(compare_lens(lens_id, args.data_dir, args.output_dir))
        print(lens_id, flush=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
