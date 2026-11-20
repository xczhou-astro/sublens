"""Read ECLIPSE DR1 products into the generic lensing observation API.

Data/model FITS have +y north and +x west. COOLEST uses that sky frame,
with position angles measured from the y axis; source-node CSVs retain the
pipeline frame (+y south). No PyAutoLens or COOLEST dependency is required.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..observation import LensingObservation


def dr1_macro_from_coolest(template):
    """Read the released maximum-likelihood SIE and shear in our sky frame."""
    profiles = {}
    for entity in template['lensing_entities']:
        for model in entity.get('mass_model', []):
            kind = model['type']
            if kind not in ('SIE', 'ExternalShear') or kind in profiles:
                raise ValueError('Expected one SIE and one external shear')
            profiles[kind] = {k: float(p['point_estimate']['value'])
                              for k, p in model['parameters'].items()}
    if set(profiles) != {'SIE', 'ExternalShear'}:
        raise ValueError('Expected one SIE and one external shear')
    s, g = profiles['SIE'], profiles['ExternalShear']
    return dict(theta_E=s['theta_E'], q=s['q'], phi=float(np.deg2rad(s['phi']-90)),
        gamma=2., center=[s['center_x'], s['center_y']], gamma_ext=g['gamma_ext'],
        phi_ext=float(np.deg2rad(g['phi_ext']-90)))


@dataclass
class DR1LensData:
    observation: LensingObservation
    raw_image: np.ndarray
    lens_light: np.ndarray
    official_lensed_source: np.ndarray
    source_points: np.ndarray
    source_values: np.ndarray
    conjugate_points: np.ndarray
    header: fits.Header
    info: dict


def load_euclid_dr1_lens(directory, *, oversample=4):
    """Load native VIS arrays, official masks and best-fit sky-frame macro.

The official foreground model is subtracted and held fixed. Returned native
image rays follow the unrotated FITS axes; no image, PSF or residual is flipped.
Source-node y coordinates alone are mirrored into the sky frame. The CSV source
coordinates use shear about the origin, whereas LensingObservation uses shear
about the lens centre; source plotting must account for the known translation.
"""
    directory = Path(directory)
    info = json.loads((directory/'dataset/info.json').read_text())
    macro = dr1_macro_from_coolest(json.loads((directory/'coolest.json').read_text()))
    with fits.open(directory/'dataset'/f'{directory.name}.fits') as h:
        image = np.array(h['VIS_BGSUB'].data, dtype=float)
        rms = np.array(h['VIS_RMS'].data, dtype=float)
        psf = np.array(h['VIS_PSF'].data, dtype=float)
        header = h['VIS_BGSUB'].header.copy()
    # Rotated or differently sampled cutouts need explicit coordinate handling.
    if any(abs(header.get(key, 0.)) > 1e-12 for key in ('CD1_2', 'CD2_1', 'PC1_2', 'PC2_1')):
        raise ValueError('Rotated DR1 cutout WCS is not supported')
    if header.get('CD1_1', header.get('PC1_1', -1.)) >= 0:
        raise ValueError('Expected east-left DR1 cutout')
    if header.get('CD2_2', header.get('PC2_2', 1.)) <= 0:
        raise ValueError('Expected north-up DR1 cutout')
    with fits.open(directory/'model.fits') as h:
        lens = np.array(h['VIS_GALAXY_0'].data, dtype=float)
        source = np.array(h['VIS_GALAXY_1'].data, dtype=float)
    from ..grids import make_image_rays
    pixel_scale = float(info['pixel_scale'])
    x, y = make_image_rays(image.shape, pixel_scale, oversample)
    cy, cx = info['mask_centre']
    mask = np.hypot(x.mean(axis=-1)-cx, y.mean(axis=-1)+cy) <= info['mask_radius']
    artefact = directory/'dataset/segmentation/artefact_binary.fits'
    if artefact.exists(): mask &= fits.getdata(artefact) == 0
    mask &= np.isfinite(image) & np.isfinite(lens) & np.isfinite(rms) & (rms > 0) & (rms < 1e7)
    obs = LensingObservation.from_arrays(image-lens, rms, psf, pixel_scale=pixel_scale,
        macro_model=macro, mask=mask, native_rays=(x, y), lens_id=directory.name)
    nodes = np.genfromtxt(directory/'vis_pix_source_plane_reconstruction_0.csv', names=True, delimiter=',')
    points = np.column_stack((nodes['x'], -nodes['y']))
    positions = directory/'dataset/positions.json'
    conjugate = np.empty((0, 2))
    if positions.exists():
        values = json.loads(positions.read_text())['arguments']['values']['array']
        conjugate = np.asarray(values, dtype=float)[:, ::-1].copy()  # already sky (y,x) -> (x,y)
    return DR1LensData(obs, image, lens, source, points, nodes['reconstruction'], conjugate, header, info)
