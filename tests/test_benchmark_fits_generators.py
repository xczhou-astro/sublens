"""
Tests for Astronomical Benchmark FITS Generators (SDSS J0946 & JVAS B1938).
Verifies:
1. Single multi-extension FITS generation containing all modeling HDUs and metadata.
2. Three separate FITS generation (SCI, PSF, RMS).
3. Automatic ingestion by SubLensPipeline.from_fits and blind fitting.
"""

import os
import shutil
import pytest
import numpy as np
from astropy.io import fits

import sublens as sl


@pytest.fixture
def tmp_fits_dir(tmp_path):
    d = tmp_path / "fits_test_benchmarks"
    d.mkdir()
    return str(d)


def test_j0946_single_fits_generation_and_modeling(tmp_fits_dir):
    """Tests generating SDSS J0946 single multi-extension FITS and auto-modeling."""
    bench = sl.generate_benchmark_fits(
        system="j0946",
        mode="single",
        output_dir=tmp_fits_dir,
        prefix="test_j0946_single",
        device="cpu"
    )

    fits_path = bench["fits_path"]
    assert os.path.exists(fits_path)
    assert bench["system"] == "j0946"
    assert bench["mode"] == "single"

    # Inspect FITS HDU content
    with fits.open(fits_path) as hdul:
        ext_names = [h.name.upper() for h in hdul]
        assert "SCI" in ext_names or hdul[0].name.upper() == "SCI"
        assert "RMS" in ext_names
        assert "WHT" in ext_names
        assert "PSF" in ext_names

        hdr = hdul['SCI'].header if 'SCI' in ext_names else hdul[0].header
        assert hdr['TELESCOP'] == 'HST'
        assert hdr['FILTER'] == 'F814W'
        assert abs(hdr['Z_LENS'] - 0.222) < 1e-4
        assert abs(hdr['Z_SOURCE'] - 0.609) < 1e-4
        assert abs(hdr['MACRO_TE'] - 1.43) < 1e-4
        assert abs(hdr['SUB_X'] - (-0.651)) < 1e-4
        assert abs(hdr['SUB_Y'] - 1.040) < 1e-4

    # Direct automated ingestion and inversion
    pipeline = sl.SubLensPipeline.from_fits(fits_path, device="cpu")
    assert abs(pipeline.z_l - 0.222) < 1e-4
    assert abs(pipeline.z_s - 0.609) < 1e-4
    assert hasattr(pipeline, "macro_model")

    # Fast fit with auto-extracted macro model
    res = pipeline.fit(
        num_macro_steps=30,
        num_pinn_steps=30,
        radar_resolution=25,
        verbose=False
    )
    assert res.subhalo_pos is not None
    assert res.detection_significance > 5.0
    assert "subhalo_pos_arcsec" in res.metrics
    assert "chi2_red_initial" in res.metrics
    assert "chi2_red_final" in res.metrics


def test_b1938_separate_fits_generation_and_modeling(tmp_fits_dir):
    """Tests generating JVAS B1938 three separate FITS files and auto-modeling."""
    bench = sl.SubLensPipeline.generate_benchmark_fits(
        system="b1938",
        mode="separate",
        output_dir=tmp_fits_dir,
        prefix="test_b1938_sep",
        device="cpu"
    )

    sci_path = bench["sci_path"]
    psf_path = bench["psf_path"]
    rms_path = bench["rms_path"]

    assert os.path.exists(sci_path)
    assert os.path.exists(psf_path)
    assert os.path.exists(rms_path)
    assert bench["system"] == "b1938"
    assert bench["mode"] == "separate"

    # Verify separate FITS headers and shapes
    with fits.open(sci_path) as hdul_sci:
        hdr = hdul_sci[0].header
        assert hdr['TELESCOP'] == 'HST'
        assert hdr['FILTER'] == 'F160W'
        assert abs(hdr['Z_LENS'] - 0.881) < 1e-4
        assert abs(hdr['Z_SOURCE'] - 2.059) < 1e-4
        assert abs(hdr['MACRO_TE'] - 0.445) < 1e-4
        assert abs(hdr['SUB_X'] - 0.036) < 1e-4
        assert abs(hdr['SUB_Y'] - 0.576) < 1e-4

    with fits.open(psf_path) as hdul_psf:
        psf_arr = hdul_psf[0].data
        assert psf_arr.shape == (31, 31)
        assert abs(psf_arr.sum() - 1.0) < 1e-3

    with fits.open(rms_path) as hdul_rms:
        rms_arr = hdul_rms[0].data
        assert rms_arr.shape == (120, 120)
        assert np.all(rms_arr > 0)

    # Ingestion with separate files
    pipeline = sl.SubLensPipeline.from_fits(
        sci_path,
        psf_path=psf_path,
        noise_path=rms_path,
        device="cpu"
    )
    assert abs(pipeline.z_l - 0.881) < 1e-4
    assert abs(pipeline.z_s - 2.059) < 1e-4
    assert abs(pipeline.pixel_scale - 0.020) < 1e-4

    # Run inversion
    res = pipeline.fit(
        num_macro_steps=30,
        num_pinn_steps=30,
        radar_resolution=25,
        verbose=False
    )
    assert res.subhalo_pos is not None
    assert res.detection_significance > 5.0
    assert res.metrics["delta_chi2"] > 0.0


def test_invalid_system_raises_error():
    """Tests that requesting an unrecognized system raises ValueError."""
    with pytest.raises(ValueError, match="Unknown benchmark system"):
        sl.generate_benchmark_fits(system="nonexistent_galaxy")
