# Independent observation loading

The interface is `data = load_lens_data(...)`, followed by
`result = analyze_lens(data, ...)`. Loading performs validation and data-format
conversion only. It does not initialize a lens, reconstruct a source, fit a
model or import simulation truth. The same module serves real and simulated
observations. Simulation objects have no `.analyze()` method.

## Variables

```python
from sublens import load_lens_data, analyze_lens

data = load_lens_data(
    image=image, noise=rms, psf=psf,
    mask=mask, pixel_scale=0.1,
    lens_light=lens_light,       # optional, already PSF blurred
)
result = analyze_lens(data, device="auto", verbose=True)
result.summary()
```

Inputs may be NumPy arrays or Torch tensors, including device tensors. Tensors
are detached and copied to CPU arrays for the measured-data container. This does
not choose the inference device: CPU/MPS/CUDA are chosen by `analyze_lens`.
`noise` is an RMS scalar/map in the image units; `pixel_scale` is arcsec/pixel.
`mask=True` means valid. Arrays must be aligned; the loader does not register or
resample images. Missing measured noise/PSF or invalid unmasked noise is an error.

## Bundled or separate files

```python
# Standard bundle: SCI or primary image, RMS/VAR/WHT, PSF, optional MASK/LENS_LIGHT.
data = load_lens_data("observation.fits")
result = analyze_lens(data, device="auto")

# Independent telescope files; no simulation object is involved.
data = load_lens_data(
    image="science.fits",
    noise="rms.fits",
    psf="psf.fits",
    mask="valid_pixels.fits",
    lens_light="foreground.fits",
    pixel_scale=0.1,
)
result = analyze_lens(data, device="auto")
```

Separate files default to their standard named plane, then primary image data.
Separate `.npy` arrays are supported with `allow_pickle=False`; array/.npy images
need an explicit pixel scale. FITS scale is inferred from `PIXSCALE` or isotropic
angular WCS if not supplied. Rotated WCS is represented in image coordinates,
not as implicit RA/Dec output; an anisotropic grid needs explicit scale/geometry.

For telescope-specific layouts, choose the actual observed HDUs explicitly:

```python
data = load_lens_data(
    image="dataset.fits", image_hdu="VIS_BGSUB",
    noise_hdu="VIS_VARIANCE", noise_kind="variance",
    psf_hdu="VIS_PSF",
    mask="bad_pixels.fits", mask_is_valid=False,
    lens_light="foreground_model.fits", lens_light_hdu="VIS_GALAXY_0",
    pixel_scale=0.1,
)
```

This is a layout example, not a claim that every catalogue has these exact
extensions. Selectors accept an HDU name, integer index or `(name, version)`.
They select from the image bundle when the corresponding file is omitted, or
from its separate file when supplied. Never substitute published source or mass
model planes for observed images. The loader rejects explicitly selected known
model products (`SOURCE*`, `BETA_*`, `MODEL*`, etc.) and does not load them by
default. Mass parameter cards such as `MACRO_*`, `THETA_E` and `LENS_*` are ignored.

`noise_kind` accepts `"rms"`, `"variance"`/`"var"`, or
`"inverse_variance"`/`"ivar"`/`"wht"`. Standard `VAR`/`VARIANCE` and `WHT`/`IVAR`
extension names imply the corresponding conversion; custom or primary planes
default to RMS. WHT is assumed to be inverse variance, not exposure time.
An explicit kind overrides inference and applies to array inputs too. Nonpositive
variance/weight becomes invalid RMS; provide a mask excluding those pixels.
The loader never silently broadens a supplied mask or classifies load errors as
nondetections. It currently represents diagonal RMS, not a full noise covariance.

`mask_is_valid=False` converts a True/nonzero-is-bad mask to the internal
True-is-valid convention once. For bit-specific flags, construct the desired
Boolean mask before loading; the loader does not guess bit definitions.

## Foreground accounting and saved data

`data.image` retains the supplied raw/subtracted pixels. `data.lens_light` is a
fixed, aligned foreground image. Loading itself does not subtract it;
`data.lens_subtracted_image` provides the derived image used by analysis.
For already subtracted variables, use `image_is_lens_subtracted=True` to associate
a foreground without subtracting it again. FITS defaults to its `LNSUB` card;
an explicit Boolean declaration overrides that state.

```python
data = load_lens_data(
    image=already_subtracted, noise=rms, psf=psf,
    lens_light=lens_light, pixel_scale=0.1,
    image_is_lens_subtracted=True,
)
data.to_fits("observed_bundle.fits")
reloaded = load_lens_data("observed_bundle.fits")
```

Exports contain observed image/RMS/PSF/mask, foreground and its subtraction state,
plus RAW_IMAGE for subtracted datasets when available. Replacing foreground on a
stored subtracted bundle restores RAW_IMAGE (or SCI + the old foreground) first.
If neither exists, replacement is an error. Input `THETA_X/Y` or explicit
`native_rays` may supply angular image geometry; they never represent `BETA`
source rays. Data exports contain no lens mass, source reconstruction, source mesh,
halo or simulation-truth products. `overwrite=False` preserves existing files.

## Container and provenance

`LensData` contains image, noise (alias rms), psf, mask, pixel_scale, optional
foreground/raw pixels/angular image geometry, ID and loading provenance. It has
no macro/source/halo fields or fitting methods. `.provenance` records file paths,
HDU selections, noise conversion and mask/foreground conventions, not model priors.
Analysis saves this record in `result.config['analysis']['input_data']`.

Passing an already loaded `LensData` to `load_lens_data` returns it unchanged;
overrides are rejected to prevent hidden data replacement. To change a dataset,
load its explicitly chosen observed arrays again. `SubLensPipeline.analyze_lens`
also accepts `LensData` without substituting constructor simulation defaults.
Legacy direct `analyze_lens(image, noise, psf, ...)` calls still work through the
same loading path. Old lower-level model-aware `LensingObservation` interfaces
remain available for separately labelled diagnostics; `LensData` does not use
them for reading mass/source products.

## Simulation uses the same boundary

```python
sim = simulate_lens(lens_mass=lens, source=source, subhalos=halos)
sim.to_fits("observation.fits")
data = load_lens_data("observation.fits")
result = analyze_lens(data, device="auto")

# Or choose just the observed variables:
data = load_lens_data(image=sim.image, noise=sim.rms, psf=sim.psf,
    mask=sim.mask, pixel_scale=sim.pixel_scale, lens_light=sim.lens_light)
```

`sim.truth` and clean images stay outside the loading/analysis path. Supplying a
foreground, exact simulation PSF or a diagonal noise approximation remains a
modelling assumption; this architectural separation does not calibrate discovery
significance or remove source/macro/regularization degeneracies.
