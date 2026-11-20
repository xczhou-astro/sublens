"""Controlled, pixel-integrated PSF families for resolution experiments.

The diffraction model is a monochromatic annular pupil with four spider vanes,
not a calibrated PSF of a specific instrument. Use measured PSF arrays/FITS for
instrument data. FWHM refers to the central detector-response profile.
"""

from functools import lru_cache

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq


@lru_cache(maxsize=4)
def _pupil_template(obscuration, spider_width, angle):
    axis = (np.arange(256)-127.5)/96.
    x,y = np.meshgrid(axis,axis)
    r = np.hypot(x,y)
    u,v = x*np.cos(angle)+y*np.sin(angle), -x*np.sin(angle)+y*np.cos(angle)
    pupil = ((r<=1)&(r>=obscuration)&(np.abs(u)>=spider_width)&(np.abs(v)>=spider_width)).astype(np.float32)
    field = fftshift(fft2(pupil,s=(2048,2048)))
    intensity = (field.real.astype(float)**2+field.imag.astype(float)**2)
    intensity /= intensity.sum()
    center = 1024
    profile = intensity[center,center:]
    index = np.flatnonzero(profile <= .5*profile[0])[0]
    half = index-1+(profile[index-1]-.5*profile[0])/(profile[index-1]-profile[index])
    yy,xx = np.indices(intensity.shape)
    bins = np.hypot(xx-center,yy-center).astype(int)
    energy = np.cumsum(np.bincount(bins.ravel(),weights=intensity.ravel()))
    return intensity,2*half,energy


def build_psf_kernel(fwhm, pixel_scale, *, profile="gaussian", beta=3.5,
                     obscuration=.30, spider_width=.015, spider_angle=.35,
                     stamp_fraction=.995, oversample=7, return_metadata=False):
    """Return a normalized convolution kernel with calibrated core FWHM.

    Pixel integration uses subpixel quadrature. The optical width is adjusted
    so the mean x/y central cut after pixel response reaches half maximum at
    fwhm/2. Finite stamps enclose at least approximately stamp_fraction of flux;
    their retained flux, encircled-energy radii and sampled widths are reported.
    All angular scales are arcsec; spider_angle is in radians.
    """
    if (not np.isfinite([fwhm,pixel_scale,beta,obscuration,spider_width,spider_angle,stamp_fraction]).all()
            or fwhm <= pixel_scale or pixel_scale <= 0 or beta <= 1
            or not 0 <= obscuration < 1 or not 0 < spider_width < .2
            or not .9 < stamp_fraction < 1 or not isinstance(oversample,int) or oversample < 3):
        raise ValueError("Use FWHM > pixel scale, beta > 1 and valid pupil, stamp and quadrature parameters")
    if profile not in ("gaussian","moffat","diffraction"):
        raise ValueError("PSF profile must be gaussian, moffat or diffraction")
    template = None
    if profile == "diffraction":
        template,template_fwhm,template_energy = _pupil_template(obscuration,spider_width,spider_angle)
    offsets = ((np.arange(oversample)+.5)/oversample-.5)*pixel_scale

    def density(width,x,y):
        if profile == "gaussian":
            sigma = width/(2*np.sqrt(2*np.log(2)))
            return np.exp(-(x*x+y*y)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
        if profile == "moffat":
            alpha = width/(2*np.sqrt(2**(1/beta)-1))
            return (beta-1)/(np.pi*alpha*alpha)*(1+(x*x+y*y)/(alpha*alpha))**(-beta)
        scale = width/template_fwhm
        x,y = np.broadcast_arrays(x,y)
        points = np.array([1024+y.ravel()/scale,1024+x.ravel()/scale])
        return map_coordinates(template,points,order=1,mode="constant",cval=0.,prefilter=False).reshape(x.shape)/scale**2

    def integrated(width,x,y):
        value = np.zeros(np.broadcast_shapes(np.shape(x),np.shape(y)))
        for dx in offsets:
            for dy in offsets:
                value += density(width,x+dx,y+dy)
        return value/(oversample**2)

    def core_ratio(width):
        return float((integrated(width,fwhm/2,0.)+integrated(width,0.,fwhm/2))
                     /(2*integrated(width,0.,0.))-.5)

    optical_width = brentq(core_ratio,.15*fwhm,2*fwhm,xtol=1e-10)
    if profile == "gaussian":
        radius = optical_width/(2*np.sqrt(2*np.log(2)))*np.sqrt(-2*np.log(1-stamp_fraction))
    elif profile == "moffat":
        alpha = optical_width/(2*np.sqrt(2**(1/beta)-1))
        radius = alpha*np.sqrt((1-stamp_fraction)**(1/(1-beta))-1)
    else:
        radius = (np.searchsorted(template_energy,stamp_fraction)+1)*optical_width/template_fwhm
    half_size = max(7,int(np.ceil((radius+pixel_scale)/pixel_scale)))
    axis = np.arange(-half_size,half_size+1)*pixel_scale
    x,y = np.meshgrid(axis,axis)
    kernel = integrated(optical_width,x,y)*pixel_scale**2
    retained = float(kernel.sum())
    kernel /= retained
    distances = np.hypot(x,y).ravel()
    order = np.argsort(distances)
    energy = np.cumsum(kernel.ravel()[order])
    def measured_width(axis_index):
        def value(r):
            return float(integrated(optical_width,r if axis_index==0 else 0.,0. if axis_index==0 else r)
                         /integrated(optical_width,0.,0.)-.5)
        return 2*brentq(value,.02*fwhm,1.5*fwhm,xtol=1e-10)
    metadata = dict(profile=profile,fwhm_target_arcsec=float(fwhm),pixel_scale_arcsec=float(pixel_scale),
                    fwhm_x_arcsec=measured_width(0),fwhm_y_arcsec=measured_width(1),
                    optical_width_arcsec=float(optical_width),pixel_quadrature=oversample,
                    kernel_shape=list(kernel.shape),stamp_fraction_requested=stamp_fraction,
                    estimated_flux_before_stamp_renormalization=retained,
                    encircled_energy_radii_arcsec={str(q):float(distances[order[np.searchsorted(energy,q)]]) for q in (.5,.8,.98)},
                    beta=beta if profile=="moffat" else None,
                    pupil=None if template is None else dict(obscuration=obscuration,spider_width=spider_width,spider_angle_rad=spider_angle),
                    calibrated_instrument_psf=False)
    return (kernel,metadata) if return_metadata else kernel
