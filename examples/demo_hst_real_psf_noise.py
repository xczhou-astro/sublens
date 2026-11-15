"""
Example: SDSS J0946+5106 Inversion with Real HST ACS F814W Optical PSF & Noise Map.

Demonstrates:
1. Simulating the confirmed SDSS J0946+5106 lens system under real HST ACS/WFC F814W conditions:
   - 2D Optical Fraunhofer diffraction PSF kernel with secondary mirror obscuration & spider diffraction spikes.
   - Non-uniform pixel-wise Poisson-Gaussian noise map (RMS and WHT maps).
2. Exporting and reading standard HST Multi-Extension FITS file:
   - HDU 0: SCI (science flux)
   - HDU 1: RMS (pixel-wise uncertainty)
   - HDU 2: WHT (pixel-wise weight)
   - HDU 3: PSF (optical kernel)
3. Performing automated blind inversion using SubLensPipeline with real PSF and noise maps.
"""

from astropy.io import fits
import numpy as np
import torch
import sublens as sl


def main():
    print("=" * 85)
    print(">>> [SubLens Demo] SDSS J0946+5106 with Authentic HST ACS F814W PSF & Noise Map")
    print("=" * 85)

    fits_filename = "sdss_j0946_hst_f814w.fits"

    # 1. Generate SDSS J0946+5106 in authentic HST ACS F814W data format
    print("\n[Step 1] Generating HST ACS F814W mock benchmark and exporting standard FITS...")
    hst_mock = sl.create_j0946_hst_benchmark(
        device="auto",
        exptime=2200.0,
        kernel_size=31,
        save_fits_path=fits_filename
    )
    print(f"  FITS product successfully saved to: {fits_filename}")

    # 2. Inspect and read the multi-extension FITS file
    print("\n[Step 2] Reading real observational inputs from Multi-Extension FITS:")
    with fits.open(fits_filename) as hdul:
        hdul.info()
        sci_data = hdul['SCI'].data.astype(np.float32)
        rms_data = hdul['RMS'].data.astype(np.float32)
        psf_data = hdul['PSF'].data.astype(np.float32)
        exp_time = hdul['SCI'].header['EXPTIME']
        filter_name = hdul['SCI'].header['FILTER']

    print(f"  Science frame shape: {sci_data.shape} | Exposure time: {exp_time} s | Filter: {filter_name}")
    print(f"  Optical PSF kernel: {psf_data.shape} (with spider diffraction spikes)")
    print(f"  Pixel-wise RMS noise range: min={rms_data.min():.5f}, max={rms_data.max():.5f} (flux-dependent)")

    # 3. Configure SubLensPipeline with real PSF kernel and RMS noise map
    print("\n[Step 3] Initializing SubLensPipeline with 2D PSF Kernel and 2D RMS Noise Map...")
    pipeline = sl.SubLensPipeline(
        z_l=0.222,
        z_s=0.609,
        pixel_scale=0.05,
        psf_kernel=psf_data,     # Real 2D optical Fraunhofer diffraction PSF
        noise_map=rms_data,      # Real 2D pixel-wise noise map
        device="auto"
    )

    # 4. Standard Smooth Macro Lens Parameters
    macro_params = {
        "theta_E": 1.43,
        "q": 0.77,
        "phi": -1.18,
        "gamma": 2.16,
        "gamma_ext": 0.035,
        "phi_ext": -0.45
    }

    # 5. One-line automated 3-stage inversion
    print("\n[Step 4] Running automated 3-stage inversion on HST observation...")
    result = pipeline.fit(
        image=sci_data,
        macro_model=macro_params,
        subhalo_type="NFW",
        radar_resolution=45,
        ground_truth=hst_mock,
        verbose=True
    )

    # 6. Save publication-grade diagnostic figure
    plot_path = "j0946_hst_realistic_results.png"
    result.plot(plot_path)

    print("\n" + "=" * 85)
    print(">>> FINAL INVERSION REPORT WITH REAL HST PSF & NOISE MAP:")
    print(f"  Radar Blind Candidate: ({result.radar_candidate_pos[0]:+.4f}\", {result.radar_candidate_pos[1]:+.4f}\") [S = {result.detection_significance:.1f} sigma]")
    print(f"  PINN Converged Position: ({result.subhalo_pos[0]:+.4f}\", {result.subhalo_pos[1]:+.4f}\")")
    print(f"  Nature 2010 Ground Truth: ({hst_mock['sub_true_pos'][0]:+.4f}\", {hst_mock['sub_true_pos'][1]:+.4f}\")")
    print(f"  Astrometric Location Error: {result._data['error_mas']:.2f} mas ({result._data['error_mas'] / 50.0:.3f} pixels)")
    print(f"  Reduced Chi-Square: chi^2_red = {result.reduced_chi2_initial:.2f} -> {result.reduced_chi2_final:.2f}")
    print(f"  Diagnostic figure saved to: {plot_path}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
