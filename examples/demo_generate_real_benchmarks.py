"""
Demonstration: Generating Authentic HST Benchmark Datasets (SDSS J0946 & JVAS B1938).

Demonstrates:
1. Generating a Single Multi-Extension FITS file (SDSS J0946+5106, ACS F814W)
   containing all modeling components: SCI, RMS, WHT, PSF, and Header Cards.
2. Generating Three Separate FITS files (JVAS B1938+666, WFC3/IR F160W):
   - <prefix>_sci.fits (Science observation image)
   - <prefix>_psf.fits (2D Fraunhofer optical diffraction PSF kernel)
   - <prefix>_rms.fits (2D pixel-wise uncertainty map)
3. Direct zero-friction ingestion and blind subhalo discovery with SubLensPipeline.
"""

import os
from astropy.io import fits
import sublens as sl


def main():
    print("=" * 78)
    print("SubLens: Authentic Astronomical Benchmark Data Generation Suite")
    print("=" * 78)

    output_dir = "benchmark_fits_output"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # System 1: SDSS J0946+5106 (Vegetti et al. 2010 Nature)
    # Output: Single Multi-Extension FITS File containing all modeling information
    # -------------------------------------------------------------------------
    print("\n[Step 1] Synthesizing SDSS J0946+5106 (Single Multi-Extension FITS)...")
    bench_j0946 = sl.generate_benchmark_fits(
        system="j0946",
        mode="single",
        output_dir=output_dir,
        prefix="sdss_j0946_hst_f814w",
        device="auto"
    )

    fits_file_j0946 = bench_j0946["fits_path"]
    print(f" -> Generated Single FITS container: {fits_file_j0946}")
    with fits.open(fits_file_j0946) as hdul:
        print(" -> Container Extensions:")
        for h in hdul:
            print(f"    * Extension '{h.name}': shape {h.data.shape}, dtype {h.data.dtype}")
        hdr = hdul['SCI'].header
        print(f" -> Header Cards: TELESCOP={hdr['TELESCOP']}, FILTER={hdr['FILTER']}, "
              f"Z_LENS={hdr['Z_LENS']}, Z_SOURCE={hdr['Z_SOURCE']}, "
              f"MACRO_TE={hdr['MACRO_TE']}'', SUB_MASS={hdr['SUB_MASS']:.2e} M_sun")

    # Invert SDSS J0946 directly from the single FITS container
    print(" -> Ingesting single FITS into SubLensPipeline...")
    pipeline_j0946 = sl.SubLensPipeline.from_fits(fits_file_j0946, device="auto")
    result_j0946 = pipeline_j0946.fit(verbose=False)
    print(f" -> Inversion complete!")
    print(f"    Detected Subhalo Pos: {result_j0946.subhalo_pos} (True: {bench_j0946['subhalo_true']['pos']})")
    print(f"    Detection Significance: {result_j0946.detection_significance:.1f} sigma")
    print(f"    Physical Mass Inverted: {result_j0946.physical_mass_solar:.2e} M_sun (True: 3.51e9 M_sun)")

    # -------------------------------------------------------------------------
    # System 2: JVAS B1938+666 (Vegetti et al. 2012 Nature)
    # Output: Three Separate FITS Files (Science Image, PSF Kernel, RMS Uncertainty)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Synthesizing JVAS B1938+666 (Three Separate FITS Files)...")
    bench_b1938 = sl.generate_benchmark_fits(
        system="b1938",
        mode="separate",
        output_dir=output_dir,
        prefix="jvas_b1938_hst_f160w",
        device="auto"
    )

    sci_file = bench_b1938["sci_path"]
    psf_file = bench_b1938["psf_path"]
    rms_file = bench_b1938["rms_path"]
    print(f" -> Generated 3 Separate Files:")
    print(f"    1. Science Image: {sci_file}")
    print(f"    2. PSF Kernel:    {psf_file}")
    print(f"    3. RMS Noise Map: {rms_file}")

    # Invert JVAS B1938 using separate FITS files
    print(" -> Ingesting separate FITS files into SubLensPipeline...")
    pipeline_b1938 = sl.SubLensPipeline.from_fits(
        sci_file,
        psf_path=psf_file,
        noise_path=rms_file,
        device="auto"
    )
    result_b1938 = pipeline_b1938.fit(verbose=False)
    print(f" -> Inversion complete!")
    print(f"    Detected Satellite Pos: {result_b1938.subhalo_pos} (True: {bench_b1938['subhalo_true']['pos']})")
    print(f"    Detection Significance: {result_b1938.detection_significance:.1f} sigma")
    print(f"    Physical Mass Inverted: {result_b1938.physical_mass_solar:.2e} M_sun (True: 1.90e8 M_sun)")

    print("\n" + "=" * 78)
    print("Demonstration successfully completed! All FITS products ready in:", output_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
