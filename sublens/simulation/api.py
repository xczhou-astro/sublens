"""One-call analytic lens simulation with an observation-only analysis boundary."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from scipy.fft import next_fast_len

from ..execution import torch_backend
from ..grids import make_image_rays
from ..profiles.models import EPL, Sersic, NFW, TruncatedHalo
from .psf import build_psf_kernel


def _components(value, types, name, *, optional=False):
    if value is None:
        if optional:
            return ()
        raise TypeError(f'{name} needs a model or a sequence of models')
    values = (value,) if isinstance(value, types) else tuple(value)
    if not values and not optional:
        raise ValueError(f'{name} must not be empty')
    if any(not isinstance(v, types) for v in values):
        raise TypeError(f'{name} contains an unsupported model')
    return values


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


@dataclass(frozen=True)
class SimulatedLens:
    """Measured arrays plus separate simulation diagnostics/truth.

    image includes the foreground and noise. lens_light is already PSF blurred.
    clean_image includes foreground but no noise; lensed_source_image is the
    PSF-blurred background image, not a source-plane reconstruction.
    """
    image: np.ndarray
    rms: np.ndarray
    psf: np.ndarray
    mask: np.ndarray
    pixel_scale: float
    lens_light: object
    clean_image: np.ndarray
    lensed_source_image: np.ndarray
    truth: dict
    lens_id: str = 'simulation'

    @property
    def noise(self):
        return self.rms

    def to_fits(self, path, *, overwrite=False):
        """Write a raw observed bundle, never mass/source/halo truth products."""
        from ..data import load_lens_data
        data=load_lens_data(self.image,noise=self.rms,psf=self.psf,mask=self.mask,
            pixel_scale=self.pixel_scale,lens_light=self.lens_light,lens_id=self.lens_id)
        return data.to_fits(path,overwrite=overwrite)

    def save_truth(self, path, *, overwrite=False):
        """Explicit, separate diagnostics file; never loaded by the data loader."""
        payload = json.dumps(self.truth, indent=2, allow_nan=False)+'\n'
        with Path(path).open('w' if overwrite else 'x') as stream:
            stream.write(payload)
        return Path(path)


def simulate_lens(lens_mass, source, subhalos=(), *, lens_light=None,
                  image_size=100, pixel_scale=.05, oversample=4,
                  psf=None, psf_fwhm=.09, psf_profile='gaussian',
                  noise=.012, mask=None, seed=314159, device='auto', dtype=None,
                  lens_id='simulation'):
    """Simulate EPL/SIE + shear, analytic source(s), and zero/multiple subhalos.

    lens_mass: EPL(...) or SIE(...); source: Sersic(...) or a sequence.
    subhalos: NFW(...), TruncatedHalo(...), a sequence, or None.
    lens_light: unlensed Sersic component(s), an already blurred image, or None.
    image_size: integer or (rows, columns). noise: positive RMS scalar/map.
    psf: supplied odd-sized convolution kernel; otherwise build psf_profile
    with psf_fwhm in arcsec. Uses subpixel quadrature, exact analytic source
    evaluation, linear PSF convolution and a seeded Gaussian noise realization.

    Returns SimulatedLens containing generated arrays, .to_fits() and
    .save_truth(). Load observed arrays/files with load_lens_data(), then call
    analyze_lens(data); simulation and loading do not invoke analysis.
    """
    if not isinstance(lens_mass, EPL):
        raise TypeError('lens_mass must be EPL(...) or SIE(...)')
    sources = _components(source, Sersic, 'source')
    halos = _components(subhalos, (NFW, TruncatedHalo), 'subhalos', optional=True)
    if isinstance(image_size, (int, np.integer)) and not isinstance(image_size, bool):
        shape = (int(image_size), int(image_size))
    else:
        shape = tuple(image_size)
    if (len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2 for n in shape)):
        raise ValueError('image_size must be an integer >=2 or a (rows, columns) pair')
    if not isinstance(oversample, (int, np.integer)) or isinstance(oversample, bool) or oversample < 1:
        raise ValueError('oversample must be a positive integer')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer or None')
    xr, yr = make_image_rays(shape, pixel_scale, oversample)
    rms = np.broadcast_to(_numpy(noise), shape).astype(float, copy=True)
    fitted = np.ones(shape, dtype=bool) if mask is None else _numpy(mask).astype(bool, copy=True)
    if not np.isfinite(rms).all() or (rms <= 0).any():
        raise ValueError('noise must be a positive finite RMS scalar or map')
    if fitted.shape != shape or not fitted.any():
        raise ValueError('mask must match the image and contain valid pixels')
    kernel = (build_psf_kernel(psf_fwhm, pixel_scale, profile=psf_profile)
              if psf is None else _numpy(psf))
    kernel = np.asarray(kernel, dtype=float)
    if (kernel.ndim != 2 or any(n % 2 == 0 for n in kernel.shape)
            or not np.isfinite(kernel).all() or kernel.sum() <= 0):
        raise ValueError('psf must be a finite odd-sized kernel with positive sum')
    kernel = kernel/kernel.sum()
    foreground_models = ()
    foreground_array = None
    if lens_light is not None:
        if isinstance(lens_light, Sersic) or (isinstance(lens_light, (list, tuple))
                                             and all(isinstance(m, Sersic) for m in lens_light)):
            foreground_models = _components(lens_light, Sersic, 'lens_light')
        else:
            foreground_array = _numpy(lens_light).astype(float, copy=True)
            if foreground_array.shape != shape or not np.isfinite(foreground_array).all():
                raise ValueError('Array lens_light must be a finite, already blurred image of matching shape')

    with torch_backend(device, dtype=dtype) as execution, torch.no_grad():
        points = np.column_stack((xr.ravel(), yr.ravel()))
        source_values, foreground_values = [], []
        for start in range(0, len(points), 65536):
            coordinates = torch.as_tensor(points[start:start+65536],
                device=execution.device, dtype=execution.torch_dtype)
            beta = coordinates-lens_mass.deflection(coordinates)
            for halo in halos:
                beta = beta-halo.deflection(coordinates)
            source_values.append(sum(s.brightness(beta) for s in sources))
            if foreground_models:
                foreground_values.append(sum(s.brightness(coordinates) for s in foreground_models))
        native = torch.cat(source_values).reshape(*shape, -1).mean(-1)
        # Crop only PSF offsets unable to connect two in-frame pixels; retain
        # full-stamp normalization. This also bounds long-wing FFT memory.
        from ..inversion.source_plane import _image_psf_support
        active_kernel = _image_psf_support(kernel, shape)
        transform_shape = tuple(next_fast_len(n+k-1) for n, k in zip(shape, active_kernel.shape))
        tensor_kernel = torch.as_tensor(active_kernel.copy(), device=execution.device, dtype=execution.torch_dtype)
        frequency = torch.fft.rfft2(tensor_kernel, s=transform_shape)
        sy, sx = (k//2 for k in active_kernel.shape)

        def convolve(image):
            full = torch.fft.irfft2(torch.fft.rfft2(image, s=transform_shape)*frequency, s=transform_shape)
            return full[sy:sy+shape[0], sx:sx+shape[1]].cpu().numpy()

        lensed = convolve(native)
        foreground = foreground_array
        if foreground_models:
            foreground = convolve(torch.cat(foreground_values).reshape(*shape, -1).mean(-1))
        clean = lensed.copy() if foreground is None else lensed+foreground
        if not np.isfinite(clean).all():
            raise RuntimeError('Simulation produced a nonfinite image')
        settings = execution.to_dict()
    # Local NumPy generator gives one reproducible noise field across devices
    # and does not alter global NumPy or Torch RNG state.
    image = clean+rms*np.random.default_rng(seed).standard_normal(shape)
    truth = dict(lens_mass=lens_mass.to_dict(), source=[s.to_dict() for s in sources],
        subhalos=[h.to_dict() for h in halos],
        lens_light=[s.to_dict() for s in foreground_models] if foreground_models else
            ('supplied blurred array' if foreground_array is not None else None),
        image_shape=list(shape), pixel_scale=float(pixel_scale), oversample=int(oversample),
        psf_profile=psf_profile if psf is None else 'supplied kernel',
        psf_fwhm_arcsec=float(psf_fwhm) if psf is None else None,
        noise_model='independent Gaussian RMS', seed=None if seed is None else int(seed),
        execution=settings, truth_used_by_analysis=False,
        production_halo_profile='TruncatedHalo (not automatically changed from simulation truth)')
    return SimulatedLens(image, rms, kernel, fitted, float(pixel_scale), foreground,
                         clean, lensed, truth, lens_id)
