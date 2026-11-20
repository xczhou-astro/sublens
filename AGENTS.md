# Project research guidance

The user's standing priority (2026-11-19) is observation-first inference. Methods must work from observed images, noise information, PSFs and masks without access to source-plane truth or injected halo parameters.

- Production inference (clarified 2026-11-20) must initialize macro parameters from observed image data, not supplied/published lens mass parameters. Published mass parameters, source maps, source meshes and model-dependent source rays are not production inputs, initializers or strong priors. They may be used only in explicitly separated conditional diagnostics or after-fit comparisons. Supplied lens light remains an explicitly declared foreground dependency. Agreement with published reconstructions is not an inference objective or evidence that a source is true.
- Keep simulation, measured-data loading, and inference as independent stages. Use instrument-independent `load_lens_data(...) -> LensData`, then `analyze_lens(data)` for both real and simulated observations. Simulation objects generate/export arrays and separate truth; do not attach analysis methods or pass their model/truth objects through the loader.
- On each evaluated lens, do not use injected truth or a published source to choose candidates, regularization, stopping criteria or detection thresholds. Simulations can develop/calibrate a method, with separate held-out evaluation and a documented selection function.
- Distinguish inner pixel/source cross-validation from independent validation of the complete discovery procedure. PSF mixing, noise correlations, tuning and candidate-search data reuse matter.
- Report source/macro/regularization degeneracies and observationally constrained resolution. Conditional recovery is not a calibrated detection or proof of a unique source.
- Preserve previous experiments. Document new validation designs as proposals until implemented and run; do not silently reinterpret old diagnostics as blind observational tests.

See [the observational validation protocol](sublens_testing/OBSERVATIONAL_VALIDATION.md). The inherited graphify instructions also apply.
