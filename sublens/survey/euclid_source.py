"""Euclid Q1 VIS data adapters and published mass-model coordinates."""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from astropy.io import fits

from ..grids import make_image_rays as _q1_image_rays
from ..observation import (
    LensingObservation as EuclidSourceObservation,
    SourceReconstructionResult as EuclidSourceResult,
    DelaunayRun,
    DelaunayReconstruction,
)
from .euclid import _elliptical_sie, centered_crop

def _q1_source_rays(shape, pixel_scale, mass, oversample):
    x_rays, y_rays = _q1_image_rays(shape, pixel_scale, oversample)
    x_sub, y_sub = torch.from_numpy(x_rays), torch.from_numpy(y_rays)

    def value(name):
        return float(mass[f"{name}_max_lh"])

    e_y, e_x = value("ell_comps_0"), value("ell_comps_1")
    e = math.hypot(e_x, e_y)
    theta_e = value("einstein_radius")
    if not 0 <= e < 1 or not math.isfinite(theta_e) or theta_e <= 0:
        raise ValueError("Invalid released ellipticity or SIE Einstein radius")
    q = (1 - e) / (1 + e)
    phi = math.atan2(e_y, e_x) / 2
    ax, ay = _elliptical_sie(
        x_sub - value("centre_1"), y_sub - value("centre_0"),
        q, math.cos(phi), math.sin(phi), math.sqrt(max(1e-12, 1 - q*q)),
    )
    g1, g2 = value("shear_gamma_1"), value("shear_gamma_2")
    bx = x_sub - theta_e * ax - g1 * x_sub - g2 * y_sub
    by = y_sub - theta_e * ay - g2 * x_sub + g1 * y_sub
    return x_rays.mean(axis=-1), y_rays.mean(axis=-1), bx.numpy(), by.numpy()


def load_euclid_source_observation(
    lens_id: str, data_dir="archive/data", *, crop_size: int = 100,
    oversample: int = 1, mask_radius: Optional[float] = None,
) -> EuclidSourceObservation:
    """Load an official Q1 release directory and its foreground-subtracted VIS data."""
    if crop_size < 2 or crop_size % 2 or oversample < 1:
        raise ValueError("Q1 source reconstruction needs an even crop and positive oversampling")
    folder = Path(data_dir) / "lens" / lens_id
    info = json.loads((folder / "info.json").read_text())
    mass = json.loads((folder / "result_lens_mass.json").read_text())

    def crop(data):
        if any((length - crop_size) % 2 for length in data.shape):
            raise ValueError("Centered Q1 crops must preserve the published pixel centres")
        return centered_crop(data, crop_size)

    with fits.open(folder / f"{lens_id}.fits") as hdus:
        vis = hdus["VIS_FLUX"]
        image = crop(vis.data)
        rms = crop(hdus["VIS_RMS"].data)
        psf = hdus["VIS_PSF"].data.astype(float)
        pixel_scale = abs(float(vis.header["CD2_2"])) * 3600
    if not math.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("Q1 VIS pixel scale must be positive and finite")
    lens = fits.getdata(folder / "result" / "lens_light.fits", "VIS")
    image -= crop(lens)
    x, y, bx, by = _q1_source_rays(image.shape, pixel_scale, mass, oversample)
    radius = float(info["mask_radius"]) if mask_radius is None else float(mask_radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Q1 fit aperture radius must be positive and finite")
    cy, cx = info.get("mask_centre", (0, 0))
    mask = np.hypot(x - cx, y - cy) <= radius
    contaminants = folder / "mask_extra_galaxies.fits"
    if contaminants.exists():
        mask &= crop(fits.getdata(contaminants)) == 0
    mask &= np.isfinite(image) & np.isfinite(rms) & (rms > 0)
    lens_center = (float(mass["centre_1_max_lh"]), float(mass["centre_0_max_lh"]))
    return EuclidSourceObservation(
        lens_id, image, rms, psf, mask, bx, by, pixel_scale, lens_center,
        float(mass["einstein_radius_max_lh"]),
    )


def reconstruct_euclid_source(
    lens_id: str, data_dir="archive/data", *, crop_size: int = 100,
    oversample: int = 1, mask_radius: Optional[float] = None, **source_options,
) -> EuclidSourceResult:
    """Reconstruct the VIS source of one Q1 system with its published macro model."""
    obs = load_euclid_source_observation(
        lens_id, data_dir, crop_size=crop_size, oversample=oversample, mask_radius=mask_radius
    )
    return obs.reconstruct(**source_options)
