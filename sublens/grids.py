"""Angular image coordinates, increasing along image columns and rows."""

import numpy as np


def make_image_rays(shape, pixel_scale, oversample=1):
    """Return centered angular coordinates with evenly spaced subpixel samples."""
    if (len(shape) != 2 or min(shape) < 1 or not np.isfinite(pixel_scale)
            or pixel_scale <= 0 or not isinstance(oversample, (int, np.integer)) or oversample < 1):
        raise ValueError("Use a positive image shape, pixel scale and integer oversampling")
    row, col = np.indices(shape, dtype=float)
    x = pixel_scale * (col - (shape[1] - 1) / 2)
    y = pixel_scale * (row - (shape[0] - 1) / 2)
    offsets = pixel_scale * ((np.arange(oversample) + 0.5) / oversample - 0.5)
    dx, dy = np.meshgrid(offsets, offsets)
    return x[..., None] + dx.ravel(), y[..., None] + dy.ravel()
