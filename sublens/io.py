"""FITS input and output for instrument-independent lensing observations."""

from pathlib import Path

import numpy as np
from astropy.io import fits


MACRO_CARDS = {
    "theta_E": "MACRO_TE", "q": "MACRO_Q", "phi": "MACRO_PA", "gamma": "MACRO_G",
    "gamma_ext": "SHEAR_G", "phi_ext": "SHEAR_PA",
}


def _image_hdu(hdus, extension, fallback=False):
    if extension in hdus:
        return hdus[extension]
    if fallback and hdus[0].data is not None:
        return hdus[0]
    raise ValueError(f"FITS data needs a {extension} extension")


def _separate_array(value, extension):
    if isinstance(value, (str, Path)):
        with fits.open(value) as hdus:
            return np.array(_image_hdu(hdus, extension, fallback=True).data)
    return value


def read_lensing_fits(path, *, noise=None, psf=None, include_source_rays=True):
    """Read measured planes and, optionally, saved model-dependent source rays.

    Production image-only analysis sets include_source_rays=False: BETA_* is
    never loaded. THETA_* is angular image geometry, not a lens/source model.
    The default preserves lower-level observation/result round trips.
    """
    with fits.open(path) as hdus:
        science = _image_hdu(hdus, "SCI", fallback=True)
        header = hdus[0].header.copy()
        header.update(science.header)
        data = {"image": np.array(science.data)}
        if noise is None:
            if "RMS" in hdus:
                noise = np.array(hdus["RMS"].data)
            elif "VAR" in hdus:
                variance = np.asarray(hdus["VAR"].data, dtype=float)
                noise = np.sqrt(np.where(variance > 0, variance, np.nan))
            elif "WHT" in hdus:
                weight = np.asarray(hdus["WHT"].data, dtype=float)
                noise = np.full(weight.shape, np.nan)
                np.divide(1.0, np.sqrt(np.maximum(weight, 0)), out=noise, where=weight > 0)
            else:
                raise ValueError("Supply noise RMS or include an RMS, VAR or WHT extension")
        if psf is None:
            psf = np.array(_image_hdu(hdus, "PSF").data)
        if "MASK" in hdus:
            data["mask"] = np.asarray(hdus["MASK"].data, dtype=bool)
        for name in ("LENS_LIGHT", "RAW_IMAGE"):
            if name in hdus:
                data[name.lower()] = np.array(hdus[name].data)
        ray_names = ("THETA_X", "THETA_Y")
        if include_source_rays:
            ray_names += ("BETA_X", "BETA_Y")
        for name in ray_names:
            if name in hdus:
                data[name.lower()] = np.array(hdus[name].data)

    data["noise"] = _separate_array(noise, "RMS")
    data["psf"] = _separate_array(psf, "PSF")
    data["header"] = header
    return data


def pixel_scale_from_header(header, pixel_scale=None):
    """Read an angular scalar pixel scale; coordinates remain image relative."""
    if pixel_scale is not None:
        return float(pixel_scale)
    if "PIXSCALE" in header:
        return float(header['PIXSCALE'])
    from astropy import units
    from astropy.wcs import WCS
    wcs=WCS(header)
    matrix=np.asarray(wcs.pixel_scale_matrix,dtype=float)
    if not any(k in header for k in ('CD2_2','CDELT2')):
        raise ValueError('Supply pixel_scale in arcsec or a PIXSCALE / angular WCS header')
    factors=np.array([units.Unit(str(wcs.wcs.cunit[i-1]) or header.get(f'CUNIT{i}','deg')).to(units.arcsec) for i in (1,2)])
    matrix=factors[:,None]*matrix
    gram=matrix.T@matrix
    if not np.allclose(gram,np.eye(2)*gram[1,1],rtol=1e-5,atol=1e-12):
        raise ValueError('Non-isotropic WCS needs explicit angular image rays and pixel scale')
    return float(np.sqrt(gram[1,1]))


def foreground_input(data, lens_light=None):
    """Resolve a foreground override using raw pixels, never subtract twice."""
    image=data['image'];stored=data.get('lens_light')
    subtracted=bool(data['header'].get('LNSUB',False))
    if lens_light is not None:
        if subtracted:
            if 'raw_image' in data:image=data['raw_image']
            elif stored is not None:image=image+stored
            else:raise ValueError('Replacing lens light on subtracted SCI needs RAW_IMAGE or stored LENS_LIGHT')
        return image,_separate_array(lens_light,'LENS_LIGHT'),False
    return image,stored,subtracted


def observation_hdus(observation):
    """Build a fresh FITS bundle with pixels, noise, PSF, mask and ray coordinates."""
    obs = observation
    header = fits.Header()
    header["OBJECT"] = obs.lens_id
    header["PIXSCALE"] = (obs.pixel_scale, "Angular pixel scale in arcsec")
    header["OVERSAMP"] = int(np.sqrt(obs.bx.shape[-1]))
    header["LENS_X"] = obs.lens_center[0]
    header["LENS_Y"] = obs.lens_center[1]
    header["PSFCONV"] = (True, "PSF uses the convolution convention")
    if obs.lens_light is not None:
        header['LNSUB']=(True,'SCI has supplied lens light subtracted once')
    if obs.einstein_radius is not None:
        header["THETA_E"] = obs.einstein_radius
    if obs.macro_model is not None:
        for key, card in MACRO_CARDS.items():
            header[card] = obs.macro_model[key]
        cx, cy = obs.macro_model["center"]
        header["MACRO_XC"], header["MACRO_YC"] = cx, cy
    for axis, length in enumerate(reversed(obs.image.shape), 1):
        header[f"CTYPE{axis}"] = "LINEAR"
        header[f"CUNIT{axis}"] = "arcsec"
        header[f"CRPIX{axis}"] = (length + 1) / 2
        header[f"CRVAL{axis}"] = 0.0
        header[f"CDELT{axis}"] = obs.pixel_scale

    x, y = obs.image_rays()
    arrays = [
        ("RMS", obs.rms), ("PSF", obs.psf), ("MASK", obs.mask.astype(np.uint8)),
        ("BETA_X", obs.bx), ("BETA_Y", obs.by), ("THETA_X", x), ("THETA_Y", y),
    ]
    if obs.lens_light is not None:
        arrays.extend([('LENS_LIGHT',obs.lens_light),('RAW_IMAGE',obs.raw_image)])
    science = fits.PrimaryHDU(obs.image, header=header)
    science.header["EXTNAME"] = "SCI"
    hdus = [science] + [fits.ImageHDU(array, name=name) for name, array in arrays]
    return fits.HDUList(hdus)


def write_lensing_fits(observation, path, *, overwrite=False):
    """Write science pixels, RMS, convolution PSF, valid mask and exact ray coordinates."""
    observation_hdus(observation).writeto(path, overwrite=overwrite, checksum=True)
