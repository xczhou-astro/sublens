"""
========================================================================================
Euclid Q1 Grade A Strong Gravitational Lensing Survey & Dark Matter Subhalo Discovery
========================================================================================

Demonstrates the standard production API workflow:
1. Standard FITS ingestion for Euclid Q1 Grade A confirmed lenses (SCI, PSF, RMS, MASK)
2. Automated 3-stage inversion with SubLensPipeline:
   - Sersic + TV pixelated source reconstruction
   - Astrometric Radar annular matched-filter blind scan
   - PINN continuous autograd subhalo refinement
3. Generation of publication-grade 8-panel diagnostic figures
4. Compilation of structured survey catalogs (CSV & FITS tables)
"""

import os
from pathlib import Path
from sublens.survey.euclid import run_euclid_grade_a_survey

def main():
    print("=" * 80)
    print(">>> SubLens Euclid Q1 Grade A Strong Lensing Survey Pipeline")
    print("=" * 80)

    # Run survey on Euclid Q1 Grade A lenses
    catalog_table, results = run_euclid_grade_a_survey(
        catalog_path="archive/data/q1_discovery_engine_lens_catalog.csv",
        data_dir="archive/data",
        output_dir="euclid_q1_diagnostic_plots",
        table_output_csv="euclid_q1_grade_a_catalog.csv",
        table_output_fits="euclid_q1_grade_a_catalog.fits",
        max_systems=10,
        radar_resolution=45,
        crop_size=100,
        save_fits_products=True
    )

    print("\n>>> Survey Summary Table:")
    summary_cols = ["LENS_ID", "EXPERT_SCORE", "THETA_E", "RADAR_SIGNIFICANCE_SIGMA", "DELTA_CHI2", "STATUS"]
    catalog_table[summary_cols].pprint_all()

if __name__ == "__main__":
    main()
