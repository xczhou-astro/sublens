"""The DR1 adapter must preserve FITS arrays and convert frames exactly once."""

import json

import numpy as np
import pytest
import torch
from astropy.io import fits

from sublens.survey.euclid_dr1 import dr1_macro_from_coolest, load_euclid_dr1_lens
from sublens.profiles.macro import epl_deflection, external_shear_deflection


def template():
    profiles = [('SIE', dict(theta_E=.9, q=.71, phi=68., center_x=.03, center_y=.08)),
                ('ExternalShear', dict(gamma_ext=.06, phi_ext=73.))]
    return dict(lensing_entities=[dict(mass_model=[dict(type=kind,
        parameters={k: dict(point_estimate=dict(value=v)) for k, v in values.items()})])
        for kind, values in profiles])


def test_coolest_sky_angles_preserve_mirrored_pipeline_deflection():
    macro = dr1_macro_from_coolest(template())
    sky = torch.tensor([[-.8, .4], [.7, .9], [.2, -.6]], dtype=torch.float64)
    def alpha(v, model):
        a, _ = epl_deflection(v, **{k:model[k] for k in ('theta_E', 'q', 'phi', 'gamma', 'center')})
        return a+external_shear_deflection(v, gamma_ext=model['gamma_ext'], phi_ext=model['phi_ext'], center=model['center'])
    native = dict(macro, center=[.03, -.08], phi=np.deg2rad(22), phi_ext=np.deg2rad(17))
    mirror = torch.tensor([1., -1.], dtype=torch.float64)
    np.testing.assert_allclose(alpha(sky, macro).numpy(), (alpha(sky*mirror, native)*mirror).numpy(), atol=1e-14)


def test_loader_preserves_psf_and_translates_mask_and_source_node_frames(tmp_path):
    p = tmp_path/'lens'; (p/'dataset/segmentation').mkdir(parents=True)
    shape=(20, 20); image=np.full(shape, .5); rms=np.full(shape, .04)
    psf=np.array([[0., .03, 0.], [.02, .7, .07], [0., .08, 0.]])
    h=fits.Header(dict(CD1_1=-.1/3600, CD2_2=.1/3600))
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(image, h, name='VIS_BGSUB'),
                  fits.ImageHDU(rms,name='VIS_RMS'),fits.ImageHDU(psf,name='VIS_PSF')]).writeto(p/'dataset/lens.fits')
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(np.full(shape,.2),name='VIS_GALAXY_0'),
                  fits.ImageHDU(np.full(shape,.3),name='VIS_GALAXY_1')]).writeto(p/'model.fits')
    artefact=np.zeros(shape); artefact[12,11]=1
    fits.writeto(p/'dataset/segmentation/artefact_binary.fits',artefact)
    (p/'dataset/info.json').write_text(json.dumps(dict(pixel_scale=.1,mask_radius=.55,mask_centre=[-.2,.1])))
    (p/'coolest.json').write_text(json.dumps(template()))
    (p/'vis_pix_source_plane_reconstruction_0.csv').write_text('y,x,reconstruction,noise_map\n.2,.3,1,.1\n-.4,-.5,2,.1\n')
    (p/'dataset/positions.json').write_text(json.dumps(dict(arguments=dict(values=dict(array=[[.4,-.3],[-.2,.6]])))))
    data=load_euclid_dr1_lens(p,oversample=2); obs=data.observation
    x,y=[a.mean(axis=-1) for a in obs.image_rays()]
    expected=(np.hypot(x-.1,y-.2)<=.55)&(artefact==0)
    np.testing.assert_array_equal(obs.mask,expected)
    np.testing.assert_allclose(obs.image,.3)
    np.testing.assert_allclose(obs.psf,psf/psf.sum(),atol=1e-15)
    np.testing.assert_array_equal(data.source_points,[[.3,-.2],[-.5,.4]])
    np.testing.assert_array_equal(data.conjugate_points,[[-.3,.4],[.6,-.2]])
    assert data.conjugate_points.flags.c_contiguous  # downstream torch conversion


def test_reject_unsupported_mass_components():
    t=template();t['lensing_entities'].append(dict(mass_model=[dict(type='NFW')]))
    with pytest.raises(ValueError,match='one SIE'): dr1_macro_from_coolest(t)
