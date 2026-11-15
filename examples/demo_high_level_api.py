"""
Example: Using the SubLens High-Level API.

Demonstrates how a user can detect dark matter subhalos in just 3 lines of code:
```python
import sublens as sl

# 1. Initialize Pipeline
pipeline = sl.SubLensPipeline(z_l=0.222, z_s=0.609, pixel_scale=0.05)

# 2. Fit image
result = pipeline.fit(image, macro_model={"theta_E": 1.43, "q": 0.77, "phi": -1.18, "gamma": 2.16})

# 3. Print report and save plots
result.summary()
result.plot("sublens_high_level_api_results.png")
```
"""

import sublens as sl


def main():
    print("=" * 80)
    print(">>> Demonstrating SubLens High-Level API (SDSS J0946 Benchmark)")
    print("=" * 80)

    # 1. Load an observation benchmark
    benchmark = sl.create_j0946_benchmark(device="auto")
    obs_image = benchmark["obs_image"]

    # 2. Use the High-Level Pipeline
    # The macro model can be passed either as a deflection tensor or a parametric dictionary
    macro_params = {
        "theta_E": 1.43,
        "q": 0.77,
        "phi": -1.18,
        "gamma": 2.16,
        "gamma_ext": 0.035,
        "phi_ext": -0.45
    }

    pipeline = sl.SubLensPipeline(
        z_l=0.222,
        z_s=0.609,
        pixel_scale=0.05,
        psf_fwhm=0.09,
        noise_std=0.012,
        device="auto"
    )

    # 3. One-line automated inversion
    result = pipeline.fit(
        image=obs_image,
        macro_model=macro_params,
        subhalo_type="NFW",
        radar_resolution=45,
        ground_truth=benchmark,
        verbose=True
    )

    # 4. Save diagnostic figure
    result.plot("sublens_high_level_api_results.png")

    # 5. Access structured attributes
    print(f"\nAPI Access Check:")
    print(f"  Converged Coordinates: {result.subhalo_pos}")
    print(f"  Radar Significance: {result.detection_significance:.1f} sigma")
    print(f"  Final Reduced Chi^2: {result.reduced_chi2_final:.2f}")
    print(f"  Elapsed Latency: {result.elapsed_time:.2f} s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
