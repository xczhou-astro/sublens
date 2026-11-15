"""Unit tests for the High-Level API."""
import torch
import numpy as np
import sublens as sl


def test_high_level_pipeline():
    # Small test simulator
    pipeline = sl.SubLensPipeline(
        pixel_scale=0.05,
        N_img=40,
        N_src=32,
        L_src=1.0,
        device="cpu"
    )

    # Synthetic image
    dummy_img = torch.randn(1, 1, 40, 40)
    macro_params = {"theta_E": 1.0, "q": 0.8}

    # Run pipeline with few steps
    result = pipeline.fit(
        image=dummy_img,
        macro_model=macro_params,
        radar_resolution=15,
        num_macro_steps=5,
        num_pinn_steps=5,
        verbose=False
    )

    assert isinstance(result, sl.SubLensResult)
    assert len(result.subhalo_pos) == 2
    assert result.detection_significance is not None
    assert result.elapsed_time > 0.0

    # Test dictionary export
    d = result.to_dict()
    assert "subhalo_pos_arcsec" in d
    assert "detection_significance_sigma" in d


def test_custom_psf_and_noise_map():
    # Test real observational inputs: 2D pixelized PSF and 2D RMS noise map
    custom_psf = np.ones((7, 7)) / 49.0
    custom_noise_map = np.full((30, 30), 0.015)
    dummy_img = np.random.randn(30, 30)

    pipeline = sl.SubLensPipeline(
        pixel_scale=0.05,
        N_img=30,
        N_src=20,
        L_src=0.8,
        psf_kernel=custom_psf,
        noise_map=custom_noise_map,
        device="cpu"
    )

    macro_params = {"theta_E": 0.8, "q": 0.85}
    result = pipeline.fit(
        image=dummy_img,
        macro_model=macro_params,
        radar_resolution=10,
        num_macro_steps=3,
        num_pinn_steps=3,
        verbose=False
    )
    assert isinstance(result, sl.SubLensResult)
    assert result.reduced_chi2_final > 0.0
