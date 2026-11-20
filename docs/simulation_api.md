# Model-object simulation API

```python
from sublens import SIE, EPL, Sersic, TruncatedHalo, NFW, simulate_lens, load_lens_data, analyze_lens

sim = simulate_lens(
    lens_mass=SIE(theta_E=1.35, q=.8, phi=.35, gamma_ext=.045, phi_ext=.2),
    source=Sersic(center=(.04, -.03), i_eff=.3, r_eff=.22, n_sersic=1.25),
    subhalos=TruncatedHalo(center=(-1.28, .61), strength=.05),
    lens_light=Sersic(i_eff=.18, r_eff=.45, n_sersic=3., q=.85, phi=.35),
    image_size=100, pixel_scale=.05, psf_fwhm=.09, noise=.012,
    seed=314159, device="auto",
)

sim.to_fits("observation.fits")
sim.save_truth("truth.json")          # explicit, separate evaluation file

# The same independent loader is used for real telescope observations.
data = load_lens_data("observation.fits")
result = analyze_lens(data, device="auto", verbose=True)
result.summary()

# Or load observed variables without saving first:
# data = load_lens_data(image=sim.image, noise=sim.rms, psf=sim.psf,
#     mask=sim.mask, pixel_scale=sim.pixel_scale, lens_light=sim.lens_light)
```

`sim.image` includes foreground and noise. `sim.rms` (alias `sim.noise`) is the
RMS map. `sim.psf`, `sim.mask`, `sim.pixel_scale`, and `sim.lens_light` are the
other observed inputs. `sim.clean_image`, `sim.lensed_source_image` and
`sim.truth` are simulation diagnostics, excluded from the observation FITS.
Simulation does not offer an analysis method: data loading and analysis are
independent stages, shared with real observations. The clean image includes the foreground; `lensed_source_image`
is a blurred image-plane background source, not a source-plane truth map.

`to_fits()` writes only raw SCI, RMS, PSF, MASK and optional LENS_LIGHT. There
are no mass, source mesh, BETA ray, halo parameter or clean-image products in
this observation bundle. It records `LNSUB=False`; the analysis subtracts the
foreground once. Simulation configurations are not accepted as production
inference inputs. Files are not overwritten unless `overwrite=True` is explicit.

## Model choices and units

| Object | Parameters | Meaning |
|---|---|---|
| `EPL(...)` | theta_E, q, phi, gamma, center, gamma_ext, phi_ext | Elliptical power-law host plus optional shear |
| `SIE(...)` | Same except gamma | Isothermal host; gamma fixed at 2 |
| `Sersic(...)` | center, i_eff, r_eff, n_sersic, q, phi | Analytic background source or unlensed foreground |
| `TruncatedHalo(...)` | strength, center, core, truncation | Existing softened/truncated subhalo, matching the current physical refiner |
| `NFW(...)` | kappa_s, r_s, center | Spherical untruncated NFW halo |

Coordinates, theta_E, r_eff, r_s, core, truncation and truncated-halo strength
are in arcseconds. Position angles are radians. EPL gamma is the **3-D density
slope**; gamma_ext is the dimensionless external shear amplitude. Sersic i_eff
is the image-brightness normalization at its effective radius, not total flux.
The configurations are immutable numerical specifications, not trainable modules.
Existing `MacroLensModel` and `Subhalo` Torch modules remain available.

For a non-isothermal host, replace the host specification, e.g.:

```python
lens = EPL(theta_E=1.35, q=.8, phi=.35, gamma=2.05,
           gamma_ext=.045, phi_ext=.2)
```

For genuine NFW simulation, replace the subhalo specification, e.g.:

```python
halo = NFW(center=(-1.28, .61), kappa_s=.2, r_s=.2)
```

NFW `kappa_s = rho_s * r_s_physical / Sigma_critical` is dimensionless; `r_s`
here is its **angular** scale radius. These are reduced lensing parameters,
not M200 or solar masses. No redshift/cosmology conversion is silently assumed.
All deflectors are on the same lens plane. NFW is untruncated, unlike
`TruncatedHalo`. There is no automatic conversion of its kappa_s to the latter's
angular strength.

The current production `analyze_lens` still initializes SIE and refines the
softened/truncated halo. Generating EPL or NFW does **not** tell the fitter those
truth models or change its family/priors. `AnalysisConfig(free_slope=True)` can
permit EPL slope variation during final refinement, but initial screening is
still based on the SIE initialization. A production NFW physical refiner is not
implemented by this simulation API. NFW injections therefore test model mismatch
if analysed with the current defaults; their recovered strength is not kappa_s.

## Images, PSFs and noise

`image_size` accepts an integer or `(rows, columns)`. Pixel coordinates follow
the observation convention: centred on the cutout, x increasing with columns,
y with rows. A missing mask means all pixels; no mask or candidate is selected
from source/halo truth. Source brightness is evaluated analytically at evenly
spaced subpixels (`oversample=4` means 4x4), averaged per pixel, then convolved
with the normalized PSF. No source-raster interpolation is required.

Use `psf=array` for an explicit odd-sized convolution kernel, or `psf_fwhm`
and `psf_profile="gaussian"`, `"moffat"` or `"diffraction"` for the existing PSF
builder. The default is a Gaussian-like PSF with FWHM .09 arcsec at .05 arcsec
pixels. Generated FWHM must exceed the pixel scale. A supplied kernel overrides
these generated-PSF settings. Asymmetric kernels use convolution, not correlation.

`noise` accepts a positive scalar RMS or spatial RMS map. It generates independent
Gaussian noise with local `np.random.default_rng(seed)`; it does not implement
Poisson or correlated noise. The noise field is reproducible across devices,
while clean-image rounding need not be bitwise identical. CPU/CUDA use float64
by default, MPS float32; explicit unavailable devices raise. `dtype` accepts
float32/float64 subject to device support.

`source` can be a sequence of Sersic components. `subhalos` accepts one halo, a
sequence (including mixed NFW/truncated profiles), `None` or an empty sequence.
`lens_light` accepts unlensed Sersic component(s), a **PSF-blurred** foreground
array, or None. The supplied foreground is fixed during analysis; its uncertainty
is not simulated or propagated by this basic example. Simulation/recovery tests
with perfect PSFs/foregrounds do not calibrate real-data discovery accuracy.

## NFW numerical correction

The existing NFW evaluator now uses analytic limits/series at small radius and
the scale radius, with finite inactive branches for autograd. In particular,
kappa(r_s)=2*kappa_s/3, rather than the old near-unity zero-density branch.
Exact-centre deflection is zero; the divergent central convergence is evaluated
at dimensionless radius 1e-10 (float64) or 1e-6 (float32). This is a numerical
centre convention, not a physical core. Deflection/convergence were checked
against a density line-of-sight integral and the deflection Jacobian on CPU/MPS.
This changes regenerated NFW examples near the affected regimes; prior stored
experiments and the production truncated-halo refiner were not rewritten.
Formula reference: [Bartelmann (1996), equations 2.8–2.10](https://arxiv.org/abs/astro-ph/9602053).

A runnable example is `examples/one_call_simulation.py`; `--simulate-only` skips
inference, and `--device cpu/mps/cuda/auto` selects the backend. Default output
directories are unique to preserve earlier experiments.
