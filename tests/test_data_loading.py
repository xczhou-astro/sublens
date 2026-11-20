"""Observed-array/FITS loading is independent of simulation and inference."""
from dataclasses import fields

import numpy as np
import pytest
import torch
from astropy.io import fits

from sublens import LensData, load_lens_data, analyze_lens, AnalysisConfig, SubLensPipeline


DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else [])


def arrays():
    y,x=np.indices((20,24))
    light=.2*np.exp(-((x-11.5)**2+(y-9.5)**2)/25.)
    return dict(image=light.copy(),noise=np.full(light.shape,.03),
        psf=np.array([[0.,.1,0.],[0.,.6,.2],[0.,.1,0.]]),
        pixel_scale=.12,lens_light=light,mask=np.ones(light.shape,dtype=bool))


def config():
    return AnalysisConfig(source_nodes=(35,),regularizations=(.002,),oversample=1,
        initializer_oversample=1,initializer_steps=2,macro_steps=2,pinn_steps=2,
        proposal_steps=2,joint_steps=2,joint_starts=1,full_refine_steps=2,
        full_refine_starts=1,adaptive_source=False,top_k=1)


def test_loading_does_not_run_fits_or_need_any_mass_model(monkeypatch):
    from sublens.inversion.initialization import ParametricInitializer
    from sublens import LensingObservation
    def forbidden(*args,**kwargs):raise AssertionError('Loading invoked inference')
    monkeypatch.setattr(ParametricInitializer,'fit',forbidden)
    monkeypatch.setattr(LensingObservation,'reconstruct',forbidden)
    data=load_lens_data(**arrays())
    assert isinstance(data,LensData)
    assert load_lens_data(data) is data
    assert not {'macro_model','source','source_grid','truth','bx','by','halo'} & {f.name for f in fields(data)}
    assert data.provenance['model_products_used'] is False
    np.testing.assert_array_equal(data.lens_subtracted_image,np.zeros(data.image.shape))
    with pytest.raises(ValueError,match='already loaded'):load_lens_data(data,pixel_scale=.5)


@pytest.mark.parametrize('device',DEVICES)
def test_variable_and_fits_data_run_identical_analysis(device,tmp_path):
    data=load_lens_data(**arrays())
    saved=load_lens_data(data.to_fits(tmp_path/'observed.fits'))
    for name in ('image','noise','psf','mask','lens_light','raw_image'):
        np.testing.assert_array_equal(getattr(data,name),getattr(saved,name))
    first=analyze_lens(data,device=device,config=config())
    second=analyze_lens(saved,device=device,config=config())
    assert first.detection()==second.detection()
    assert first.observation.macro_model==second.observation.macro_model
    np.testing.assert_array_equal(first.baseline.image_model,second.baseline.image_model)
    assert second.config['analysis']['input_data']['planes']['image']['path']==str((tmp_path/'observed.fits').resolve())
    # A loaded observation owns its scale/noise/PSF, not simulator constructor defaults.
    pipeline=SubLensPipeline(pixel_scale=.05,N_img=20,N_src=16,device=device)
    third=pipeline.analyze_lens(saved,config=config())
    assert third.observation.pixel_scale==.12 and third.detected is False


@pytest.mark.parametrize('device',DEVICES)
def test_tensor_variables_detach_and_do_not_mutate_inputs(device):
    values=arrays()
    for name in ('image','noise','psf','lens_light'):
        values[name]=torch.tensor(values[name],dtype=torch.float32,device=device,requires_grad=True)
    values['mask']=torch.tensor(values['mask'],device=device)
    before=values['image'].detach().cpu().numpy().copy()
    data=load_lens_data(**values)
    assert isinstance(data.image,np.ndarray) and data.image.dtype==np.float64
    np.testing.assert_array_equal(data.image,before)
    np.testing.assert_array_equal(values['image'].detach().cpu(),before)


def test_custom_bundle_hdus_variance_and_bad_pixel_mask(tmp_path):
    a=arrays();bad=np.zeros(a['image'].shape,dtype=np.uint8);bad[0]=1
    variance=a['noise']**2;variance[0]=0
    path=tmp_path/'telescope.fits'
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(a['image'],fits.Header(dict(PIXSCALE=.12)),name='EXPOSURE'),
        fits.ImageHDU(variance,name='UNCERTAINTY'),fits.ImageHDU(a['psf'],name='KERNEL'),
        fits.ImageHDU(bad,name='BAD_PIXELS'),fits.ImageHDU(a['lens_light'],name='FOREGROUND'),
        fits.ImageHDU(np.full((3,3),999.),name='SOURCE')]).writeto(path)
    data=load_lens_data(path,image_hdu='EXPOSURE',noise_hdu='UNCERTAINTY',noise_kind='variance',
        psf_hdu='KERNEL',mask_hdu='BAD_PIXELS',mask_is_valid=False,lens_light_hdu='FOREGROUND')
    assert not data.mask[0].any() and data.mask[1:].all()
    np.testing.assert_allclose(data.noise[data.mask],.03)
    assert np.isnan(data.noise[0]).all()
    np.testing.assert_array_equal(data.lens_subtracted_image,np.zeros(data.image.shape))
    assert data.provenance['planes']['image']['hdu']=='EXPOSURE'


def test_separate_files_npy_and_inverse_variance(tmp_path):
    a=arrays();path=tmp_path/'image.fits'
    fits.PrimaryHDU(a['image'],fits.Header(dict(PIXSCALE=.12))).writeto(path)
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(1/a['noise']**2,name='IVAR')]).writeto(tmp_path/'noise.fits')
    np.save(tmp_path/'psf.npy',a['psf']);np.save(tmp_path/'mask.npy',a['mask'])
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(a['lens_light'],name='VIS_GALAXY_0')]).writeto(tmp_path/'foreground.fits')
    data=load_lens_data(path,noise=tmp_path/'noise.fits',psf=tmp_path/'psf.npy',
        mask=tmp_path/'mask.npy',lens_light=tmp_path/'foreground.fits',lens_light_hdu='VIS_GALAXY_0')
    np.testing.assert_allclose(data.rms,a['noise'])
    assert data.provenance['noise_kind_input']=='inverse_variance'
    np.testing.assert_array_equal(data.lens_subtracted_image,np.zeros(data.image.shape))
    np.save(tmp_path/'image.npy',a['image'])
    other=load_lens_data(tmp_path/'image.npy',noise=a['noise'],psf=a['psf'],pixel_scale=.12)
    np.testing.assert_array_equal(other.image,a['image'])


def test_foreground_roundtrip_preserves_state_and_replacement_uses_raw(tmp_path):
    a=arrays();raw=a['image']+.1
    data=load_lens_data(image=raw-a['lens_light'],noise=a['noise'],psf=a['psf'],pixel_scale=.12,
        lens_light=a['lens_light'],image_is_lens_subtracted=True)
    path=data.to_fits(tmp_path/'subtracted.fits')
    loaded=load_lens_data(path)
    np.testing.assert_array_equal(loaded.image,data.image)
    np.testing.assert_array_equal(loaded.raw_image,data.raw_image)
    assert loaded.image_is_lens_subtracted
    changed=load_lens_data(path,lens_light=2*a['lens_light'])
    assert not changed.image_is_lens_subtracted
    np.testing.assert_allclose(changed.image,raw,atol=1e-15)
    np.testing.assert_allclose(changed.lens_subtracted_image,raw-2*a['lens_light'],atol=1e-15)
    # Fitted observations preserve the stored exact raw provenance, rather than
    # constructing it repeatedly by subtracting/adding a foreground.
    from sublens.analysis import _input_observation
    obs=_input_observation(loaded,None,None,None,None,None,None,None,config(),None)
    np.testing.assert_array_equal(obs.image,loaded.image)
    np.testing.assert_array_equal(obs.raw_image,loaded.raw_image)


def test_model_products_are_not_loaded_or_accepted(tmp_path,monkeypatch):
    a=arrays();path=tmp_path/'mixed.fits'
    fits.HDUList([fits.PrimaryHDU(a['image'],fits.Header(dict(PIXSCALE=.12,MACRO_TE='invalid',THETA_E=999.,LENS_X=999.))),
        fits.ImageHDU(a['noise'],name='RMS'),fits.ImageHDU(a['psf'],name='PSF'),
        *[fits.ImageHDU(np.ones((3,3)),name=n) for n in ('BETA_X','BETA_Y','SOURCE','SOURCE_NODES','MODEL_MACRO')]]).writeto(path)
    original=fits.HDUList.__getitem__
    def measured_only(hdus,key):
        if isinstance(key,str) and key in {'BETA_X','BETA_Y','SOURCE','SOURCE_NODES','MODEL_MACRO'}:
            raise AssertionError('Loader accessed a source/mass model plane')
        return original(hdus,key)
    monkeypatch.setattr(fits.HDUList,'__getitem__',measured_only)
    data=load_lens_data(path)
    assert not hasattr(data,'macro_model') and data.pixel_scale==.12
    with pytest.raises(TypeError):load_lens_data(**a,macro_model={'theta_E':1.})
    with pytest.raises(TypeError):load_lens_data(**a,source_grid=np.zeros((4,2)))


def test_explicit_model_hdu_rejected_and_invalid_rms_requires_mask(tmp_path):
    a=arrays();path=tmp_path/'source.fits'
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(a['image'],name='SOURCE')]).writeto(path)
    with pytest.raises(ValueError,match='model/source product'):
        load_lens_data(path,image_hdu='SOURCE',noise=.03,psf=a['psf'],pixel_scale=.12)
    a['noise'][0]=0
    with pytest.raises(ValueError,match='positive finite noise'):load_lens_data(**a)
    a['mask'][0]=False
    assert not load_lens_data(**a).mask[0].any()


def test_unused_mask_selector_and_complex_images_are_not_silently_ignored():
    a=arrays();a.pop('mask')
    with pytest.raises(ValueError,match='mask_hdu requires'):
        load_lens_data(**a,mask_hdu='BAD_PIXELS')
    a['image']=a['image'].astype(complex)+1j
    with pytest.raises(ValueError,match='real-valued'):
        load_lens_data(**a)
