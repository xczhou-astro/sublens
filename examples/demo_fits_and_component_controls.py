"""
Example: Full User Control over FITS I/O, Component Plotting, and Data Extraction.

Demonstrates:
1. Initializing directly from FITS (single multi-extension or separate FITS files)
2. Extracting raw 2D NumPy arrays:
   - result.get_source()
   - result.get_model_image()
   - result.get_residual_image()
   - result.get_macro_residual()
   - result.get_radar_map()
   - result.get_subhalo_convergence()
3. Accessing performance metrics dictionary (result.metrics)
4. Plotting individual components separately:
   - result.plot_macro_residual("macro_residual.png")
   - result.plot_radar_map("radar_smap.png")
   - result.plot_final_model("final_model.png")
   - result.plot_final_residual("final_residual.png")
   - result.plot_reconstructed_source("reconstructed_source.png")
   - result.plot_subhalo_convergence("subhalo_kappa.png")
5. Exporting full results as Multi-Extension FITS file (result.save_fits())
"""

import os
from astropy.io import fits
import sublens as sl


def main():
    print("=" * 85)
    print(">>> [SubLens Demo] Granular Control: FITS I/O, Individual Plots & Data Extraction")
    print("=" * 85)

    fits_input = "sdss_j0946_hst_f814w.fits"

    # Ensure FITS file exists; generate if missing
    if not os.path.exists(fits_input):
        print(f"Creating sample HST FITS file: {fits_input}...")
        sl.create_j0946_hst_benchmark(save_fits_path=fits_input)

    # 1. Initialize Pipeline directly from FITS file
    # Automatically reads PIXSCALE, Z_LENS, Z_SOURCE, PSF kernel, and RMS noise map
    print(f"\n[1] Initializing SubLensPipeline directly from FITS: {fits_input}")
    pipeline = sl.SubLensPipeline.from_fits(fits_input, device="auto")

    macro_params = {
        "theta_E": 1.43,
        "q": 0.77,
        "phi": -1.18,
        "gamma": 2.16,
        "gamma_ext": 0.035,
        "phi_ext": -0.45
    }

    # 2. Run automated inversion directly on FITS file
    print("\n[2] Executing automated 3-stage inversion...")
    result = pipeline.fit(
        image=fits_input,
        macro_model=macro_params,
        subhalo_type="NFW",
        radar_resolution=45,
        verbose=True
    )

    # 3. Extract Raw Data Arrays (NumPy 2D arrays for downstream analysis)
    print("\n[3] Extracting Raw 2D Data Arrays:")
    source_plane = result.get_source(format="numpy")
    model_image  = result.get_model_image(format="numpy")
    final_res    = result.get_residual_image(format="numpy")
    macro_res    = result.get_macro_residual(format="numpy")
    radar_map, extent = result.get_radar_map(format="numpy")
    subhalo_kappa = result.get_subhalo_convergence(format="numpy")

    print(f"  * Reconstructed Source shape: {source_plane.shape} | Peak flux: {source_plane.max():.4f}")
    print(f"  * Modeled Lens Image shape:   {model_image.shape} | Mean: {model_image.mean():.5f}")
    print(f"  * Final Residual shape:       {final_res.shape} | RMS: {final_res.std():.5f}")
    print(f"  * Macro Defect Residual shape:{macro_res.shape} | Peak defect: {macro_res.max():.5f}")
    print(f"  * Radar S-Map shape:          {radar_map.shape} | Peak: {radar_map.max():.2f} sigma")
    print(f"  * Subhalo Convergence shape:  {subhalo_kappa.shape} | Max kappa: {subhalo_kappa.max():.4f}")

    # 4. Access Performance Metrics Dictionary
    print("\n[4] Quantitative Performance Metrics Dictionary:")
    metrics = result.metrics
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.5f}")
        else:
            print(f"  {k:<30}: {v}")

    # 5. Plot Individual Components Separately
    print("\n[5] Generating and Saving Individual Component Figures:")
    os.makedirs("component_plots", exist_ok=True)
    result.plot_macro_residual("component_plots/1_macro_defect_residual.png")
    result.plot_radar_map("component_plots/2_radar_score_map.png")
    result.plot_final_model("component_plots/3_final_model_image.png")
    result.plot_final_residual("component_plots/4_final_residual_annihilated.png")
    result.plot_reconstructed_source("component_plots/5_reconstructed_source.png")
    result.plot_subhalo_convergence("component_plots/6_subhalo_convergence_field.png")

    # 6. Export Full Inversion Products to Multi-Extension FITS File
    out_fits = "sdss_j0946_inversion_products.fits"
    print(f"\n[6] Exporting all inversion results to Multi-Extension FITS: {out_fits}")
    result.save_fits(out_fits, overwrite=True)

    with fits.open(out_fits) as hdul:
        print("\nExported FITS Structure:")
        hdul.info()

    print("\n" + "=" * 85)
    print(">>> Granular Control & FITS I/O Demo Completed Successfully!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
