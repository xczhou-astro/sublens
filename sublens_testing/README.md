# Research runners and observational validation

This directory versions experiment **source code**, not downloaded observations,
generated FITS/images, chains, fitted parameters or local result snapshots.
Run scripts from a source checkout using the project Python environment. Their
data paths and output folders are documented in the individual runners; inputs
must be obtained/prepared locally. Previously generated results are preserved
locally and are not bundled in the Python wheel.

The current general-purpose production entry points are:

```python
from sublens import load_lens_data, analyze_lens

data = load_lens_data("observation.fits")
result = analyze_lens(data, device="auto")
```

See the [data loader](../docs/data_loading.md), [analysis API](../docs/analyze_lens.md),
[simulation API](../docs/simulation_api.md), and
[observational validation protocol](OBSERVATIONAL_VALIDATION.md).

The older runners include conditional source/macro experiments and development
benchmarks. Their supplied macro models, official source comparisons or injected
truth must not be promoted to production inputs or calibrated detection evidence.
The production API always initializes from observed image data. Source CV and
PINN cross-fit residuals are internal diagnostics, not independent confirmation.

Useful source families:

- `one_subhalo/*.py`: controlled injection/noise/PSF/source studies and checks;
  several earlier experiments condition on an input macro, as labelled there.
- `euclid_dr1_*.py`, `euclid_grade_a_10.py`: historical and current catalogue
  runners/diagnostics; require released data/catalogues prepared locally.
- `multi_subhalo_scan.py`, `run_multi_subhalo.py`: multiple-halo experiments.
- `tests/test_joint_lambda_sampling.py`: additional research-sampler tests,
  opt in with an explicit pytest path.

`python -m pytest` runs the active `tests/` suite. Legacy Q1 real-data integration
tests report skips when their local `archive/data` catalogue/FITS are absent.
This does not validate those integrations; all their assertions run when the
inputs are available. CUDA tests likewise require CUDA hardware. Data-free
simulation, loading, physics and fitting tests run without survey downloads.
