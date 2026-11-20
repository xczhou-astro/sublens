"""Instrument-independent loading of measured lens images, without lens models.

This layer does no fitting, source reconstruction or ray tracing. It accepts
observational arrays/FITS planes and never loads mass/source model products.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from .io import pixel_scale_from_header


def _array(value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if array.ndim == 4 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    if np.iscomplexobj(array):
        raise ValueError('Observed image/noise/PSF planes must be real-valued')
    return np.array(array, dtype=float, copy=True)


def _pick(hdus, selector, preferred=(), *, required=False, primary=False):
    selected = selector
    if selected is None:
        selected = next((name for name in preferred if name in hdus), None)
        if selected is None and primary and hdus[0].data is not None:
            selected = 0
    if selected is None:
        if required:
            raise ValueError(f'Missing observed FITS plane; supply one of {preferred} or an explicit HDU')
        return None
    hdu = hdus[selected]
    name = hdu.name.upper()
    if name.startswith(('BETA_', 'ALPHA_', 'SOURCE', 'MACRO_', 'MODEL', 'RESID', 'PINN_')) or name == 'VIS_SOURCE':
        raise ValueError(f'{name} is a model/source product, not an observed input plane')
    if hdu.data is None:
        raise ValueError(f'FITS plane {selected!r} contains no image data')
    return np.array(hdu.data), hdu.header.copy(), name


def _external(value, selector, preferred):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == '.npy':
            if selector is not None:
                raise ValueError('HDU selectors apply to FITS files, not .npy arrays')
            return np.load(path, allow_pickle=False), '', dict(path=str(path.resolve()), format='npy')
        with fits.open(path) as hdus:
            array, _, name = _pick(hdus, selector, preferred, required=True, primary=True)
        return array, name, dict(path=str(path.resolve()), hdu=name)
    if selector is not None:
        raise ValueError('An HDU selector requires a FITS file for that input')
    return value, '', dict(origin='variable')


def _rms(values, kind, extension):
    aliases = {'rms':'rms', 'variance':'variance', 'var':'variance',
               'inverse_variance':'inverse_variance', 'ivar':'inverse_variance', 'wht':'inverse_variance'}
    if kind is None:
        kind = ('variance' if extension in ('VAR', 'VARIANCE') else
                'inverse_variance' if extension in ('WHT', 'IVAR') else 'rms')
    if kind not in aliases:
        raise ValueError('noise_kind must be rms, variance or inverse_variance')
    kind = aliases[kind]
    values = _array(values)
    if kind == 'variance':
        values = np.sqrt(np.where(values > 0, values, np.nan))
    elif kind == 'inverse_variance':
        positive = np.isfinite(values) & (values > 0)
        rms = np.full(values.shape, np.nan)
        np.divide(1., np.sqrt(np.where(positive, values, 1.)), out=rms, where=positive)
        values = rms
    return values, kind


@dataclass(frozen=True)
class LensData:
    """Measured data only: image, RMS, PSF, valid mask and angular geometry.

    image retains its supplied raw/lens-subtracted state. Loading does not
    subtract foreground; lens_subtracted_image exposes that derived view.
    No macro, source grid, halo, clean simulation image or truth fields exist.
    """
    image: np.ndarray
    noise: np.ndarray
    psf: np.ndarray
    pixel_scale: float
    mask: object = None
    lens_light: object = None
    image_is_lens_subtracted: bool = False
    lens_id: str = 'observation'
    native_rays: object = None
    raw_image: object = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        image = _array(self.image)
        if image.ndim != 2:
            raise ValueError('image must be a 2-D observed array')
        noise = np.broadcast_to(_array(self.noise), image.shape).copy()
        mask_values = np.ones(image.shape) if self.mask is None else _array(self.mask)
        if mask_values.shape != image.shape or not np.isfinite(mask_values).all():
            raise ValueError('mask must be finite and match the image')
        mask = mask_values.astype(bool)
        if (not mask.any() or not np.isfinite(image[mask]).all()
                or not np.isfinite(noise[mask]).all() or (noise[mask] <= 0).any()):
            raise ValueError('Valid pixels need finite image values and positive finite noise RMS')
        psf = _array(self.psf)
        if (psf.ndim != 2 or any(n % 2 == 0 for n in psf.shape)
                or not np.isfinite(psf).all() or psf.sum() <= 0):
            raise ValueError('PSF must have odd dimensions, finite values and positive sum')
        scale = float(self.pixel_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('pixel_scale must be positive in arcsec')
        if not isinstance(self.image_is_lens_subtracted, (bool, np.bool_)):
            raise ValueError('image_is_lens_subtracted must be Boolean')
        foreground = None if self.lens_light is None else _array(self.lens_light)
        if foreground is not None and (foreground.shape != image.shape or not np.isfinite(foreground[mask]).all()):
            raise ValueError('lens_light must match the image and be finite on valid pixels')
        raw = None if self.raw_image is None else _array(self.raw_image)
        if raw is None:
            if not self.image_is_lens_subtracted:
                raw = image.copy()
            elif foreground is not None:
                raw = image+foreground
        if raw is not None and (raw.shape != image.shape or not np.isfinite(raw[mask]).all()):
            raise ValueError('raw_image must match the image and be finite on valid pixels')
        rays = self.native_rays
        if rays is not None:
            if len(rays) != 2:
                raise ValueError('native_rays needs angular x/y image coordinates')
            x, y = (_array(a) for a in rays)
            if x.ndim == 2:x = x[..., None]
            if y.ndim == 2:y = y[..., None]
            if (x.ndim != 3 or x.shape[:2] != image.shape or x.shape[-1] < 1 or y.shape != x.shape
                    or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError('Angular image rays must have matching finite (rows, columns, samples) shapes')
            rays = (x, y)
        for name, value in dict(image=image, noise=noise, psf=psf/psf.sum(), mask=mask,
                pixel_scale=scale, lens_light=foreground, raw_image=raw, native_rays=rays,
                provenance=dict(self.provenance)).items():
            object.__setattr__(self, name, value)

    @property
    def rms(self):
        return self.noise

    @property
    def lens_subtracted_image(self):
        if self.image_is_lens_subtracted or self.lens_light is None:
            return self.image.copy()
        return self.image-self.lens_light

    def to_fits(self, path, *, overwrite=False):
        """Save measured data/state only, with no mass/source model products."""
        header = fits.Header(dict(OBJECT=self.lens_id, PIXSCALE=self.pixel_scale,
                                  LNSUB=bool(self.image_is_lens_subtracted)))
        hdus = [fits.PrimaryHDU(self.image, header), fits.ImageHDU(self.noise, name='RMS'),
                fits.ImageHDU(self.psf, name='PSF'), fits.ImageHDU(self.mask.astype(np.uint8), name='MASK')]
        if self.lens_light is not None:
            hdus.append(fits.ImageHDU(self.lens_light, name='LENS_LIGHT'))
        if self.image_is_lens_subtracted and self.raw_image is not None:
            hdus.append(fits.ImageHDU(self.raw_image, name='RAW_IMAGE'))
        if self.native_rays is not None:
            hdus.extend(fits.ImageHDU(a, name=name) for a, name in zip(self.native_rays, ('THETA_X','THETA_Y')))
        fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)
        return Path(path)


def load_lens_data(image, noise=None, psf=None, lens_light=None, *, pixel_scale=None,
                   mask=None, lens_id=None, native_rays=None, image_is_lens_subtracted=None,
                   image_hdu=None, noise_hdu=None, psf_hdu=None, mask_hdu=None,
                   lens_light_hdu=None, noise_kind=None, mask_is_valid=True):
    """Load observed arrays or bundled/separate FITS planes into LensData.

    No fit or model initialization occurs. HDU selectors may be names, indices
    or (name, version) tuples. Scalar/array noise is RMS by default; VAR/WHT
    extensions infer variance/inverse variance unless noise_kind overrides it.
    mask=True means valid; mask_is_valid=False inverts an input bad-pixel mask.
    Separate .npy arrays are also supported (no pickle or hidden model fields).
    """
    if not isinstance(mask_is_valid, (bool, np.bool_)):
        raise ValueError('mask_is_valid must be Boolean')
    if image_is_lens_subtracted is not None and not isinstance(image_is_lens_subtracted, (bool, np.bool_)):
        raise ValueError('image_is_lens_subtracted must be Boolean or None')
    if isinstance(image, LensData):
        overrides = (noise, psf, lens_light, pixel_scale, mask, lens_id, native_rays,
            image_is_lens_subtracted, image_hdu, noise_hdu, psf_hdu, mask_hdu, lens_light_hdu, noise_kind)
        if any(v is not None for v in overrides) or not mask_is_valid:
            raise ValueError('LensData is already loaded; supply observed arrays to load a modified dataset')
        return image
    provenance = dict(loader='observed_data_v1', planes={}, model_products_used=False)
    planes = provenance['planes']
    header = fits.Header()
    stored_foreground = raw = None
    bundle_noise_name = ''
    if isinstance(image, (str, Path)) and Path(image).suffix.lower() != '.npy':
        path = Path(image)
        with fits.open(path) as hdus:
            pixels, science_header, science_name = _pick(hdus, image_hdu, ('SCI',), required=True, primary=True)
            header = hdus[0].header.copy();header.update(science_header)
            planes['image'] = dict(path=str(path.resolve()), hdu=science_name)
            for label, value, selector, preferred, required in (
                ('noise', noise, noise_hdu, ('RMS','VAR','VARIANCE','WHT','IVAR'), True),
                ('psf', psf, psf_hdu, ('PSF',), True),
                ('mask', mask, mask_hdu, ('MASK',), False),
            ):
                if value is None:
                    selected = _pick(hdus, selector, preferred, required=required)
                    if selected is not None:
                        array, _, name = selected
                        planes[label] = dict(path=str(path.resolve()), hdu=name)
                        if label == 'noise':noise, bundle_noise_name = array, name
                        elif label == 'psf':psf = array
                        else:mask = array
            selected = _pick(hdus, None, ('LENS_LIGHT',))
            if selected is not None:
                stored_foreground = selected[0]
            selected = _pick(hdus, None, ('RAW_IMAGE',))
            if selected is not None:raw = selected[0]
            if native_rays is None:
                geometry = [_pick(hdus, None, (name,)) for name in ('THETA_X','THETA_Y')]
                if (geometry[0] is None) != (geometry[1] is None):
                    raise ValueError('FITS angular image geometry needs both THETA_X and THETA_Y')
                if geometry[0] is not None:
                    native_rays = tuple(a[0] for a in geometry)
                    planes['native_rays'] = dict(path=str(path.resolve()), hdus=['THETA_X','THETA_Y'])
            # A custom foreground HDU in the same image bundle is explicit.
            if lens_light is None and lens_light_hdu is not None:
                selected = _pick(hdus, lens_light_hdu, required=True)
                lens_light = selected[0]
                planes['lens_light'] = dict(path=str(path.resolve()), hdu=selected[2])
                lens_light_hdu = None
        image = pixels
        pixel_scale = pixel_scale_from_header(header, pixel_scale)
        lens_id = lens_id or header.get('OBJECT', path.stem)
    else:
        image, _, planes['image'] = _external(image, image_hdu, ('SCI',))
        if pixel_scale is None:
            raise ValueError('Array/.npy images require pixel_scale in arcsec')
        if mask is None and mask_hdu is not None:
            raise ValueError('mask_hdu requires a mask file or image FITS bundle')
    if noise is None or psf is None:
        raise ValueError('Supply measured noise RMS and PSF, or include them in a FITS bundle')
    for label, value, selector, preferred in (
        ('noise', noise, noise_hdu, ('RMS','VAR','VARIANCE','WHT','IVAR')),
        ('psf', psf, psf_hdu, ('PSF',)), ('mask', mask, mask_hdu, ('MASK',)),
    ):
        if value is not None and label not in planes:
            array, name, origin = _external(value, selector, preferred)
            planes[label] = origin
            if label == 'noise':noise, bundle_noise_name = array, name
            elif label == 'psf':psf = array
            else:mask = array
    noise, noise_kind = _rms(noise, noise_kind, bundle_noise_name)
    supplied_foreground = lens_light is not None
    if supplied_foreground and 'lens_light' not in planes:
        lens_light, _, planes['lens_light'] = _external(lens_light, lens_light_hdu, ('LENS_LIGHT',))
    elif not supplied_foreground and stored_foreground is not None:
        lens_light = stored_foreground
        planes['lens_light'] = dict(path=planes['image']['path'], hdu='LENS_LIGHT')
    elif lens_light_hdu is not None:
        raise ValueError('lens_light_hdu requires a foreground file or image FITS bundle')
    stored_subtracted = bool(header.get('LNSUB', False))
    subtracted = stored_subtracted if image_is_lens_subtracted is None else image_is_lens_subtracted
    if supplied_foreground and stored_subtracted and not bool(image_is_lens_subtracted):
        # Replace foreground on the raw data, rather than subtracting twice.
        if raw is not None:image = raw
        elif stored_foreground is not None:image = _array(image)+_array(stored_foreground)
        else:raise ValueError('Replacing foreground on subtracted SCI needs RAW_IMAGE or stored LENS_LIGHT')
        subtracted = False
    if not subtracted:
        raw = _array(image)
    if mask is not None:
        values = _array(mask)
        if not np.isfinite(values).all():raise ValueError('mask must be finite')
        mask = values.astype(bool) if mask_is_valid else ~values.astype(bool)
    elif not mask_is_valid:
        raise ValueError('mask_is_valid=False requires an explicit or bundled mask')
    provenance.update(noise_kind_input=noise_kind, mask_input_true_is_valid=bool(mask_is_valid),
        image_is_lens_subtracted=bool(subtracted),
        foreground='supplied fixed observed foreground' if lens_light is not None else 'none')
    return LensData(image, noise, psf, pixel_scale, mask=mask, lens_light=lens_light,
        image_is_lens_subtracted=subtracted, lens_id=lens_id or 'observation',
        native_rays=native_rays, raw_image=raw, provenance=provenance)
