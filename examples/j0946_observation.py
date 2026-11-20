"""Prepare the inner arc of the public LensAgent J0946 HST/ACS F814W cutout."""

import json

import numpy as np
from lenstronomy.LensModel.Profiles.epl import EPL
from scipy.optimize import nnls
from scipy.signal import fftconvolve

from sublens.survey.euclid_source import EuclidSourceObservation, _q1_image_rays


J0946_EPL_SLOPE = 2.119
J0946_SHEAR = (0.059, -0.069)


def load_j0946_observation(path):
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(bundle["metadata"].item())
        if meta["system_id"] != "094656.68+100652.8" or meta["preparation_metadata"]["filter"] != "F814W":
            raise ValueError("Expected the public LensAgent HST/ACS F814W bundle for J0946")
        # The central 100 pixels include both rings and the foreground fitting aperture.
        original_height, original_width = bundle["image_data"].shape
        crop = tuple(slice((size - 100) // 2, (size + 100) // 2) for size in (original_height, original_width))
        data = bundle["image_data"][crop]
        rms = bundle["noise_map"][crop]
        valid = bundle["likelihood_mask"][crop].astype(bool)
        scale = float(meta["pixel_scale"])
        psf = bundle["psf_kernel"]
        # In this bundle the omitted outer PSF contains less than 1e-12 of its flux.
        py, px = (size // 2 for size in psf.shape)
        psf = psf[py - 15:py + 16, px - 15:px + 16]
    psf = psf / psf.sum()
    valid &= np.isfinite(data) & np.isfinite(rms) & (rms > 0)
    x, y = _q1_image_rays(data.shape, scale, oversample=2)
    center_pixel = np.asarray(meta["preparation_metadata"]["geometry"]["lens_pixel"])
    original_size_xy = np.array([original_width, original_height])
    light_center = (center_pixel - (original_size_xy - 1) / 2) * scale
    u, v = x - light_center[0], y - light_center[1]
    radius = np.hypot(u.mean(axis=-1), v.mean(axis=-1))

    # The Nightingale data panel aligns to the LensAgent stamp by (x, y) ->
    # (y, -x), verified by matching both observed images. Rotate all spin-2
    # components by the corresponding 90 degrees. Refit light amplitudes here.
    components = []
    for e1, e2, re, index in [(0.022, -0.071, 0.449, 1.630), (-0.070, 0.116, 2.030, 0.506)]:
        theta = np.arctan2(e2, e1) / 2
        ellipticity = np.hypot(e1, e2)
        axis_ratio = (1 - ellipticity) / (1 + ellipticity)
        major = np.cos(theta) * u + np.sin(theta) * v
        minor = -np.sin(theta) * u + np.cos(theta) * v
        elliptical = np.hypot(major, minor / axis_ratio) * np.sqrt(axis_ratio)
        profile = np.exp(-(1.9992 * index - 0.3271) * ((elliptical / re)**(1 / index) - 1))
        components.append(fftconvolve(profile.mean(axis=-1), psf, mode="same"))
    design = np.stack(components + [np.ones_like(data)], axis=-1)
    light_mask = valid & ((radius < 0.70) | ((radius > 2.30) & (radius < 2.45)))
    # Separate signed sky from nonnegative galaxy components.
    light_design = np.column_stack((design[light_mask], -np.ones(light_mask.sum())))
    weights = 1 / rms[light_mask]
    weights *= np.where(radius[light_mask] < 0.16, 0.10, 1.0)
    coefficients, _ = nnls(light_design * weights[:, None], data[light_mask] * weights)
    foreground = design @ coefficients[:3] - coefficients[3]

    # Nightingale et al. no-subhalo EPL + shear, aligned on the foreground center.
    center = (float(light_center[0] - 0.020), float(light_center[1]))
    theta_e = 1.409
    ax, ay = EPL().derivatives(x.ravel(), y.ravel(), theta_E=theta_e, gamma=J0946_EPL_SLOPE,
                               e1=0.019, e2=-0.046, center_x=center[0], center_y=center[1])
    sx, sy = x - center[0], y - center[1]
    g1, g2 = J0946_SHEAR
    bx = x - ax.reshape(x.shape) - (g1 * sx + g2 * sy)
    by = y - ay.reshape(y.shape) - (g2 * sx - g1 * sy)
    mask = valid & (radius > 0.80) & (radius < 1.78)
    return EuclidSourceObservation("J0946+1006_HST_F814W", data - foreground, rms, psf,
                                   mask, bx, by, scale, center, theta_e)
