import pytest
import numpy as np
import torch
from sublens.inversion.voronoi import VoronoiSourceInverter


def test_voronoi_source_inverter_basic():
    # Synthetic test data
    N = 40
    img = np.zeros((N, N), dtype=np.float32)
    # Bright ring / arc
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.hypot(x - 20, y - 20)
    mask = (r >= 8) & (r <= 16)
    img[mask] = 1.0 + np.exp(-0.5 * ((r[mask] - 12) / 2.0)**2)

    psf = np.zeros((11, 11), dtype=np.float32)
    psf[5, 5] = 1.0 # delta-like
    noise = np.ones((N, N), dtype=np.float32) * 0.05

    # Mock linear mapping to source plane
    beta_x = (x - 20) * 0.05
    beta_y = (y - 20) * 0.05

    inverter = VoronoiSourceInverter(num_cells=60, reg_lambda=50.0, seed=42)
    res = inverter.fit(
        image_arr=img,
        psf_arr=psf,
        noise_arr=noise,
        fit_mask=mask,
        beta_x=beta_x,
        beta_y=beta_y
    )

    assert "centroids" in res
    assert "source_vector" in res
    assert "peak_brightness" in res
    assert res["peak_brightness"] > 0.0
    assert res["chi2_red"] >= 0.0

    # Test rasterization
    raster = inverter.rasterize(extent=(-0.5, 0.5, -0.5, 0.5), resolution=50)
    assert raster.shape == (50, 50)
    assert raster.max() > 0.0
