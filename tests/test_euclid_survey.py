"""
Unit and integration tests for Euclid Q1 Grade A survey data ingestion and inversion.
"""

import os
from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
from astropy.table import Table

from sublens import SubLensPipeline
from sublens.survey.euclid import (
    prepare_euclid_grade_a_fits,
    get_available_grade_a_lenses
)


def test_get_available_grade_a_lenses():
    """Tests catalog filtering for confirmed Grade A lenses."""
    catalog = get_available_grade_a_lenses(
        catalog_path="archive/data/q1_discovery_engine_lens_catalog.csv",
        data_dir="archive/data"
    )
    assert len(catalog) > 0
    top = catalog[0]
    assert top["grade"] == "A"
    assert top["expert_score"] > 3.0
    assert "102158584" in top["id_str"]


def test_prepare_euclid_grade_a_fits(tmp_path):
    """Tests standard multi-extension FITS file creation for Euclid lenses."""
    out_fits = tmp_path / "test_euclid.fits"
    fits_path = prepare_euclid_grade_a_fits(
        lens_id="102158584_2687900321650859511",
        data_dir="archive/data",
        output_path=str(out_fits),
        crop_size=100
    )
    assert os.path.exists(fits_path)

    with fits.open(fits_path) as hdul:
        exts = [h.name for h in hdul]
        assert "SCI" in exts
        assert "PSF" in exts
        assert "RMS" in exts
        assert "MASK" in exts

        hdr = hdul["SCI"].header
        assert hdr["TELESCOP"] == "Euclid"
        assert hdr["INSTRUME"] == "VIS"
        assert hdr["GRADE"] == "A"
        assert abs(hdr["PIXSCALE"] - 0.10) < 1e-3
        assert "MACRO_TE" in hdr
        assert hdul["SCI"].data.shape == (100, 100)
        assert hdul["MASK"].data.shape == (100, 100)


def test_euclid_pipeline_standard_api(tmp_path):
    """Tests end-to-end standard SubLens API execution on Euclid Grade A FITS."""
    fits_file = tmp_path / "euclid_test.fits"
    prepare_euclid_grade_a_fits(
        lens_id="102158584_2687900321650859511",
        data_dir="archive/data",
        output_path=str(fits_file),
        crop_size=100
    )

    pipeline = SubLensPipeline.from_fits(str(fits_file))
    assert hasattr(pipeline, "fit_mask")
    assert pipeline.fit_mask.shape == (100, 100)
    assert pipeline.macro_model["theta_E"] > 0.70

    # Execute fit with fast settings for unit testing
    res = pipeline.fit(radar_resolution=25, num_macro_steps=30, num_pinn_steps=20, verbose=False)
    m = res.metrics

    assert "detection_significance_s" in m
    assert m["detection_significance_s"] > 0.0
    assert "chi2_red_initial" in m
    assert "chi2_red_final" in m
    assert "delta_chi2" in m
    assert m["chi2_red_initial"] < 2.0

    # Test plot output
    plot_file = tmp_path / "diagnostic.png"
    res.plot(save_path=str(plot_file))
    assert plot_file.exists()
    assert plot_file.stat().st_size > 10000

    # Test FITS export
    prod_file = tmp_path / "products.fits"
    res.save_fits(str(prod_file))
    assert prod_file.exists()
