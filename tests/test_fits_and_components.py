"""
Unit tests for FITS inputs, individual component plotting, and data extraction.
"""

import os
import numpy as np
import pytest
from astropy.io import fits
import sublens as sl


def test_fits_io_and_component_controls(tmp_path):
    fits_file = str(tmp_path / "test_lens_package.fits")
    out_fits = str(tmp_path / "inversion_output.fits")
    component_png = str(tmp_path / "component_source.png")

    # 1. Create a dummy multi-extension FITS file
    sci = np.random.randn(30, 30).astype(np.float32)
    psf = np.ones((7, 7), dtype=np.float32) / 49.0
    rms = np.full((30, 30), 0.012, dtype=np.float32)

    hdr = fits.Header()
    hdr['PIXSCALE'] = 0.05
    hdr['Z_LENS'] = 0.222
    hdr['Z_SOURCE'] = 0.609

    hdul = fits.HDUList([
        fits.PrimaryHDU(data=sci, header=hdr),
        fits.ImageHDU(data=rms, name='RMS'),
        fits.ImageHDU(data=psf, name='PSF')
    ])
    hdul[0].header['EXTNAME'] = 'SCI'
    hdul.writeto(fits_file, overwrite=True)

    # 2. Test Pipeline.from_fits() factory
    pipeline = sl.SubLensPipeline.from_fits(fits_file, N_src=20, L_src=0.8, device="cpu")
    assert pipeline.z_l == 0.222
    assert pipeline.z_s == 0.609
    assert pipeline.simulator.pixel_scale == 0.05

    # 3. Fit directly using FITS file path as input
    macro_params = {"theta_E": 0.8, "q": 0.8}
    result = pipeline.fit(
        image=fits_file,
        macro_model=macro_params,
        radar_resolution=10,
        num_macro_steps=3,
        num_pinn_steps=3,
        verbose=False
    )

    # 4. Test Data Extraction Methods
    src_np = result.get_source(format="numpy")
    assert isinstance(src_np, np.ndarray)
    assert src_np.shape == (20, 20)

    mod_np = result.get_model_image(format="numpy")
    assert mod_np.shape == (30, 30)

    res_np = result.get_residual_image(format="numpy")
    assert res_np.shape == (30, 30)

    radar_np, extent = result.get_radar_map(format="numpy")
    assert radar_np.shape == (10, 10)
    assert len(extent) == 4

    kappa_sub = result.get_subhalo_convergence(format="numpy")
    assert kappa_sub.shape == (30, 30)

    # 5. Test Performance Metrics Dictionary
    m = result.metrics
    assert "subhalo_x_arcsec" in m
    assert "detection_significance_sigma" in m
    assert "chi2_reduced_final" in m
    assert "elapsed_seconds" in m

    # 6. Test Individual Component Plotting
    fig = result.plot_component("source", save_path=component_png)
    assert os.path.exists(component_png)

    # 7. Test Exporting Full Inversion to FITS
    result.save_fits(out_fits, overwrite=True)
    assert os.path.exists(out_fits)

    with fits.open(out_fits) as out_hdul:
        exts = [h.name.upper() for h in out_hdul]
        assert 'MODEL' in exts
        assert 'RESIDUAL' in exts
        assert 'SOURCE' in exts
        assert 'RADAR_SMAP' in exts
        assert 'SUBHALO_KAPPA' in exts
        assert 'SUB_X' in out_hdul['MODEL'].header
