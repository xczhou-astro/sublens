# Complete PINN analysis API

```python
import sublens as sl

data = sl.load_lens_data(
    image=image, noise=rms, psf=psf, lens_light=lens_light,
    pixel_scale=0.1, mask=mask,
)
result = sl.analyze_lens(data, device="mps")
result.summary()
print(result.detected, result.position, result.strength)
decision = result.detection()
result.save_json("result.json")
result.save_fits("result.fits")
result.plot_residuals(save_path="residuals.png")
result.plot_fit(save_path="source_fit.png")
```

`image` is a 2-D observed cutout; `noise` is its positive RMS map or scalar, in
the same units. `psf` is an aligned odd-size convolution kernel, normalized once
without clipping small negative lobes. `lens_light` is a same-shape foreground
image, subtracted once and held fixed. The data and foreground arrays are not
modified. `mask=True` means a valid fitted pixel. Missing measured noise or PSF
is an error; no simulation default is silently supplied.

The independent [data-loading module](data_loading.md) reads either real or
simulated observed arrays/files into `LensData`, without fitting any model.
`analyze_lens(data)` is the recommended analysis boundary. Existing direct
array/FITS calls remain supported and delegate to this same loader internally.

The scalar `pixel_scale` is in arcseconds. Default coordinates are centred on
the image, with x along increasing columns and y along increasing rows. Output
position and strength are in arcseconds in that frame. A custom `native_rays`
pair can specify the angular subpixel coordinates explicitly. Output `b` is the
softened/truncated profile amplitude, not a mass or an uncertainty interval.

`detected` follows the user-selected rule: delta chi2 >=25 and positive penalized
gain, choosing the greatest penalized gain among passing single-halo alternatives.
A negative result has `position=None` and `strength=None`. Numerical/input
failures raise rather than being classified as absent. Stability, convergence
and bound diagnostics are retained, but do not veto this rule. This is a
model-based screening decision, not calibrated physical detection significance.

## FITS input and foreground handling

```python
data = sl.load_lens_data("lens.fits")
result = sl.analyze_lens(data, device="auto")

# Separate files / overrides are also supported:
data = sl.load_lens_data(
    "science.fits", noise="rms.fits", psf="psf.fits",
    lens_light="foreground.fits", pixel_scale=0.1, mask=mask,
)
result = sl.analyze_lens(data, device="cpu")
```

A raw bundle uses primary image data or `SCI`, `RMS` (or `VAR`/`WHT`), `PSF`,
and optionally `MASK` and `LENS_LIGHT`. Separate files can have their data in a
primary HDU. Pixel scale comes from `PIXSCALE` or a complete isotropic angular
WCS unless supplied explicitly. Anisotropic/sheared pixel grids need explicit
angular geometry. Rotated isotropic WCS is analysed in the image coordinate
frame; output x/y is not implicitly a celestial RA/Dec pair.

The production API always initializes macro parameters from the observed image.
It does not accept external `macro_model`, source maps or source meshes, even as
an initialization. FITS `MACRO_*`, `THETA_E` and `LENS_*` model parameter cards
are ignored; source-brightness/mesh products and `BETA_*` are not loaded.
Saved native `THETA_*` arrays describe angular image geometry and may be reused.
This distinction matters: `THETA_E` is a lens parameter, whereas `THETA_X/Y`
are measured image coordinates. Lower-level observation/result interfaces
still support model round trips and separately labelled conditional diagnostics.

Result/observation FITS stores lens-subtracted `SCI`, `LENS_LIGHT`, `RAW_IMAGE`
and `LNSUB=True`. Reloading does not subtract the foreground again. Replacing
the foreground on such a bundle uses raw pixels, not the already-subtracted
SCI. If supplying already-subtracted arrays with an associated foreground,
set `image_is_lens_subtracted=True`. With no foreground supplied, the image is
used as given; foreground light is not automatically estimated.

The same foreground support is available in
`LensingObservation.from_arrays(..., lens_light=...)` and `.from_fits(...)`.
`SubLensPipeline.analyze_lens(...)` delegates to this complete flow. Existing
lower-level `search_pinn`, `reconstruct`, and legacy `fit` behaviour is retained.

## Implemented workflow and declared defaults

1. Four image-only SIE + shear + Sersic fits initialize the macro. The source
   Sersic is initialization only; final source light is nonnegative Delaunay.
2. Internal 20% pixel CV chooses 300/500 sampled nodes and lambda from
   `(2e-6, 2e-4, .02, 2, 20)`. This is conditional tuning, not independent
   validation of discovery.
3. Seven macro parameters are refined with the selected pixel source: radius,
   axis ratio, angle, centre x/y and two shear components. Slope is fixed at 2
   at this stage; no supplied mass initialization is used.
4. Four-sector source cross-fitting with PSF guards/coverage checks feeds one
   PINN. Up to five proposals receive physical and initial joint refinement.
5. If the initial improvement rule is met, one source-basis-only adaptive round
   chooses a common null/halo grid and lambda, then recomputes source brightness
   and matched gains with lens/halo parameters held fixed. It does not run another
   local nonlinear fit or boundary-expansion fit. Every retained proposal then
   goes directly to full macro + halo + source refinement, with multiple starts
   and null rescue from halo macro solutions. An intermediate drop in gain does
   not discard a proposal before full refinement. The exact zero-strength branch
   is allowed. This removes the former repeated local refinement, while retaining
   initial physical/local joint screening and its threshold unchanged.
6. Candidates are merged only after physical refinement. The returned decision
   uses the final matched null. The eight sensitivity experiments are optional
   research diagnostics, not automatically required for a binary result.

Defaults assume a galaxy-scale cutout centred approximately on the main lens.
The initializer allows theta_E 0.2–5 arcsec, q 0.3–0.995, centre offsets +/-0.2
arcsec with a weak sigma=0.15 arcsec centring prior, and shear components within
approximately +/-0.3. The source Sersic Re starts within 0.015–1.5 arcsec and
n within 0.5–5. Pixel macro bounds are local to the image-fitted initialization.
Full refinement uses radius factors 0.7–1.3, q 0.25–0.995, angle +/-0.7 rad,
centre +/-0.15 arcsec and shear-component offsets +/-0.12. Halo position spans
+/-0.7 arcsec around each original proposal and b spans 0–0.30 arcsec. The
profile core/truncation scales remain 0.035/0.22 arcsec. These are declared model
assumptions, not priors derived from a source truth or measured halo position.

For off-centre systems, centre the cutout/image coordinate frame using observed
foreground positions before analysis. Atypical systems outside these declared
priors need a separately configured fitting method; production `analyze_lens`
does not accept a published mass solution as a substitute. This single
host / competing single-halo model does not automatically model multiple lens
galaxies or establish halo count.

## Devices and options

```python
config = sl.AnalysisConfig(
    source_nodes=(300, 500), min_delta_chi2=25.,
    full_refine_starts=3, free_slope=False,
)
result = sl.analyze_lens(
    image, rms, psf, lens_light, pixel_scale=0.1, mask=mask,
    device="cpu", config=config, verbose=True,
)
```

CPU/CUDA default to float64; MPS uses float32. `gpu` means CUDA; `auto` chooses
CUDA, then MPS, then CPU. An unavailable explicitly requested device raises.
FITS I/O and mesh setup are CPU work; iterative fitting uses the selected Torch
device. Optional `dtype` is accepted subject to device support. CUDA validation
requires CUDA hardware; CPU/MPS checks on this Mac do not establish CUDA timing.

`free_slope=True` lets full candidate refinement fit EPL slope within 1.6–2.4.
`adaptive_source=False` skips the adaptive source round, not initial source CV.
Configuration controls budgets, integration, source grids and halo ranges.
Analysis timings, initialization provenance, source CV, execution, optimization
and fixed-foreground assumptions are saved in `result.config['analysis']` and
portable metadata. A custom decision threshold is used consistently by the live
result, summary and JSON/FITS export.

`result.config['analysis']['workflow_version']` identifies the streamlined
`image_only_single_full_refinement_v2` flow. The initial screening threshold can
still exclude systems that might improve with a broader full fit; its discovery
completeness is not established by this refactor. The new standalone
`result.adapt_source_basis(...)` updates grid/lambda and source brightness only;
follow it with `result.refine_full(...)` for final lens/halo optimization.
Legacy `result.refine_adaptive(...)` retains its combined source/local nonlinear
behaviour for previous experiments and is not called by the production flow.

Source/macro/lambda degeneracy and foreground/PSF uncertainty can mimic a compact
perturber. The given foreground shape and PSF are fixed, and RMS uses a diagonal
likelihood; correlated noise and their uncertainties are not marginalized.
Internal pixel CV and PINN cross-fit residuals reuse this observation. They do
not calibrate a false-positive rate or turn point estimates into a posterior.
