"""Array/FITS parity, foreground accounting and end-to-end PINN API contracts."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from astropy.io import fits
from scipy.signal import fftconvolve

from sublens import analyze_lens, AnalysisConfig, LensingObservation, torch_backend
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.halo_search import HaloFit,PinnHaloCandidate,PinnHaloSearchResult
from sublens.inversion.potential import ResidualPotential


DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else [])


def small_config(**options):
    defaults=dict(source_nodes=(40,),regularizations=(.002,.02),oversample=1,initializer_oversample=1,
        initializer_steps=2,macro_steps=2,pinn_steps=2,proposal_steps=2,joint_steps=2,joint_starts=1,
        full_refine_steps=2,full_refine_starts=1,adaptive_source=False,top_k=1)
    return AnalysisConfig(**{**defaults,**options})


def inputs():
    y,x=np.indices((22,24));light=.2*np.exp(-((x-11.5)**2+(y-10.5)**2)/25)
    psf=np.array([[.01,.05,.02],[.08,.65,.1],[.02,.04,.03]])
    return light.copy(),np.full(light.shape,.03),psf,light


@pytest.mark.parametrize('device',DEVICES)
def test_automatic_analysis_arrays_and_raw_fits_match_without_macro_truth(device,tmp_path):
    image,noise,psf,light=inputs();before=image.copy()
    options=dict(config=small_config(),device=device,pixel_scale=.12,lens_id='lens')
    a=analyze_lens(image,noise,psf,light,**options)
    assert not a.detected and a.position is None and a.strength is None
    assert not a.config['analysis']['full_refinement_triggered']
    assert a.config['analysis']['initializer']['origin']=='image_only_four_starts'
    np.testing.assert_array_equal(image,before)
    np.testing.assert_array_equal(a.observation.image,np.zeros_like(image))
    np.testing.assert_array_equal(a.observation.raw_image,image)
    path=tmp_path/'raw.fits'
    header=fits.Header(dict(PIXSCALE=.12,MACRO_TE='forbidden mass input',MACRO_Q=.01,OBJECT='lens'))
    fits.HDUList([fits.PrimaryHDU(image,header),fits.ImageHDU(noise,name='RMS'),fits.ImageHDU(psf,name='PSF'),
                  fits.ImageHDU(light,name='LENS_LIGHT'),
                  fits.ImageHDU(np.full((*image.shape,1),999.),name='BETA_X'),
                  fits.ImageHDU(np.full((*image.shape,1),999.),name='BETA_Y'),
                  fits.ImageHDU(np.full(image.shape,999.),name='SOURCE'),
                  fits.ImageHDU(np.full((9,2),999.),name='SOURCE_NODES')]).writeto(path)
    b=analyze_lens(path,config=small_config(),device=device)
    assert a.detection()==b.detection()
    assert a.observation.macro_model==b.observation.macro_model
    np.testing.assert_allclose(a.baseline.image_model,b.baseline.image_model,atol=1e-8)
    saved=tmp_path/'result.fits';a.save_fits(saved,resolution=21)
    restored=LensingObservation.from_fits(saved)
    np.testing.assert_array_equal(restored.image,a.observation.image)
    np.testing.assert_allclose(restored.raw_image,image,atol=1e-15)
    np.testing.assert_array_equal(restored.lens_light,light)
    a.save_json(tmp_path/'result.json')
    assert json.loads((tmp_path/'result.json').read_text())['detection']==a.detection()
    with fits.open(saved) as h:
        assert h[0].header['LNSUB']
        assert all(x.verify_checksum()==1 for x in h)


def test_foreground_roundtrip_and_override_use_raw_image_once(tmp_path):
    image,noise,psf,light=inputs()
    obs=LensingObservation.from_arrays(image+.1,noise,psf,pixel_scale=.12,macro_model={},lens_light=light)
    path=tmp_path/'bundle.fits';obs.to_fits(path)
    restored=LensingObservation.from_fits(path)
    changed=LensingObservation.from_fits(path,lens_light=2*light)
    np.testing.assert_allclose(restored.image,.1,atol=1e-15)
    np.testing.assert_allclose(changed.image,image+.1-2*light,atol=1e-15)
    np.testing.assert_allclose(changed.raw_image,image+.1,atol=1e-15)
    moved=restored.with_macro(dict(theta_E=.8))
    np.testing.assert_array_equal(moved.raw_image,restored.raw_image)
    np.testing.assert_array_equal(moved.lens_light,restored.lens_light)


@pytest.mark.parametrize('device',DEVICES)
def test_fixed_epl_full_refinement_uses_one_basis_and_common_rescued_null(device):
    macro=dict(theta_E=.65,q=.83,phi=.2,gamma=2.05,center=[.01,-.02],gamma_ext=.03,phi_ext=.4)
    blank=LensingObservation.from_arrays(np.ones((18,20)),.04,np.ones((3,3))/9,pixel_scale=.12,macro_model=macro)
    native=np.exp(-((blank.bx-.02)**2+(blank.by+.01)**2)/.035).mean(-1)
    obs=replace(blank,image=fftconvolve(native,blank.psf,mode='same'))
    with torch_backend(device):
        p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=40,regularization=.002)
        source=p.fit(obs.bx,obs.by)
        fit=HaloFit(.5,.1,.03,0.,0.,False,source,source.image_model,macro)
        potential=ResidualPotential(obs.mask,np.zeros(obs.image.shape),np.zeros(obs.image.shape),np.zeros(obs.image.shape),[])
        r=PinnHaloSearchResult(obs,source,potential,(PinnHaloCandidate(1,(.5,.1),1.,fit),),(.4,.9))
        out=r.refine_full(maxiter=2,n_starts=1,retry_failed=False)
    c=out.candidates[0].fit
    assert c.source.grid is out.baseline.grid and c.source.regularization==out.baseline.regularization
    assert c.macro_model['gamma']==macro['gamma'] and out.macro_model['gamma']==macro['gamma']
    expected=np.sum(((obs.image-out.baseline.image_model)[obs.mask]/obs.rms[obs.mask])**2)-np.sum(((obs.image-c.image_model)[obs.mask]/obs.rms[obs.mask])**2)
    assert c.delta_chi2==pytest.approx(expected)
    assert c.delta_objective==pytest.approx(obs.mask.sum()*(out.baseline.objective-c.source.objective))
    assert len(out.config['joint_refinement']['free_macro_parameters'])==7
    assert 'rescued_null' in out.config['full_refinement_record']['candidate_attempts'][0]


def test_analysis_selected_threshold_is_used_by_saved_and_live_results(tmp_path):
    from sublens.inversion.halo_search import PinnHaloSearchResult
    # Exercise metadata threshold propagation with a valid zero-data analysis.
    image,noise,psf,light=inputs()
    result=analyze_lens(image,noise,psf,light,pixel_scale=.12,device='cpu',config=small_config(min_delta_chi2=50.))
    assert result.detection()['rule']['min_delta_chi2']==50.
    result.save_json(tmp_path/'r.json')
    assert json.loads((tmp_path/'r.json').read_text())['detection']['rule']['min_delta_chi2']==50.


@pytest.mark.parametrize('adaptive',[False,True])
def test_provisional_candidate_triggers_full_fit_but_does_not_force_final_detection(monkeypatch,adaptive):
    calls=[]
    def proposed(obs,**options):
        p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=40,regularization=.002)
        baseline=p.fit(obs.bx,obs.by)
        # A deliberately false proposal on source-free data tests routing, not
        # recovery. The actual full fit must compare it against the exact null.
        fit=HaloFit(.4,.1,.03,30.,30.,True,baseline,baseline.image_model,obs.macro_model)
        pot=ResidualPotential(obs.mask,np.zeros(obs.image.shape),np.zeros(obs.image.shape),np.zeros(obs.image.shape),[])
        return PinnHaloSearchResult(obs,baseline,pot,(PinnHaloCandidate(1,(.4,.1),1.,fit),),(.2,1.))
    monkeypatch.setattr(LensingObservation,'search_pinn',proposed)
    def initial_joint(self,*args,**kwargs):
        calls.append('initial_joint')
        return self
    monkeypatch.setattr(PinnHaloSearchResult,'refine_joint',initial_joint)
    def intermediate(self,*args,**kwargs):
        calls.append('source_basis_only')
        c=self.candidates[0]
        return replace(self,candidates=(replace(c,fit=replace(c.fit,delta_chi2=0.,delta_objective=0.)),))
    monkeypatch.setattr(PinnHaloSearchResult,'adapt_source_basis',intermediate)
    def forbidden(*args,**kwargs):raise AssertionError('Legacy repeated local refinement was called')
    monkeypatch.setattr(PinnHaloSearchResult,'refine_adaptive',forbidden)
    image,noise,psf,light=inputs()
    result=analyze_lens(image,noise,psf,light,pixel_scale=.12,device='cpu',config=small_config(adaptive_source=adaptive))
    assert result.config['analysis']['full_refinement_triggered']
    assert 'full_refinement_record' in result.config
    assert calls==['initial_joint']+(['source_basis_only'] if adaptive else [])
    assert result.config['analysis']['external_macro_input_accepted'] is False
    assert result.detected is False
    np.testing.assert_array_equal(result.observation.lens_light,light)


@pytest.mark.parametrize('device',DEVICES)
def test_source_basis_update_preserves_candidates_and_has_no_nonlinear_refit(device,monkeypatch):
    # Algorithm contract fixture, not an observation-only recovery experiment.
    macro=dict(theta_E=.65,q=.83,phi=.2,center=[.01,-.02],gamma_ext=.03,phi_ext=.4)
    blank=LensingObservation.from_arrays(np.ones((18,20)),.04,np.ones((3,3))/9,pixel_scale=.12,macro_model=macro)
    native=np.exp(-((blank.bx-.02)**2+(blank.by+.01)**2)/.035).mean(-1)
    obs=replace(blank,image=fftconvolve(native,blank.psf,mode='same'))
    def forbidden(*args,**kwargs):raise AssertionError('Source-basis selection invoked nonlinear refinement')
    monkeypatch.setattr(PinnHaloSearchResult,'refine_joint',forbidden)
    monkeypatch.setattr(PinnHaloSearchResult,'refine_full',forbidden)
    with torch_backend(device):
        problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=40,regularization=.002)
        source=problem.fit(obs.bx,obs.by)
        # First position is at its old local search boundary. No extra boundary
        # expansion/refit should occur before the final wider full fit.
        fits_=[HaloFit(.55,.1,.03,30.,30.,True,source,source.image_model,obs.macro_model),
               HaloFit(-.5,.2,.02,29.,29.,True,source,source.image_model,obs.macro_model)]
        candidates=tuple(PinnHaloCandidate(i,(.2,.1) if i==1 else (-.5,.2),1.,f) for i,f in enumerate(fits_,1))
        potential=ResidualPotential(obs.mask,np.zeros(obs.image.shape),np.zeros(obs.image.shape),np.zeros(obs.image.shape),[])
        initial=PinnHaloSearchResult(obs,source,potential,candidates,(.4,.9),refinement_radius=.35,macro_model=obs.macro_model)
        updated=initial.adapt_source_basis(source_nodes=(35,),regularizations=(.002,.02),seeds=(22,))
    assert updated.observation is obs
    assert updated.refinement_radius==initial.refinement_radius
    assert len(updated.candidates)==len(initial.candidates)==2
    assert updated.config['adaptive_source']['nonlinear_refinement'] is False
    null_chi=np.sum(((obs.image-updated.baseline.image_model)[obs.mask]/obs.rms[obs.mask])**2)
    for before,after in zip(initial.candidates,updated.candidates):
        assert after.seed==before.seed and after.fit.position==before.fit.position and after.fit.strength==before.fit.strength
        assert after.fit.source.grid is updated.baseline.grid
        assert after.fit.source.regularization==updated.baseline.regularization
        chi=np.sum(((obs.image-after.fit.image_model)[obs.mask]/obs.rms[obs.mask])**2)
        assert after.fit.delta_chi2==pytest.approx(null_chi-chi)
        assert after.fit.delta_objective==pytest.approx(obs.mask.sum()*(updated.baseline.objective-after.fit.source.objective))
        assert after.fit.converged is False


def test_production_fits_never_loads_source_products(tmp_path,monkeypatch):
    image,noise,psf,light=inputs()
    path=tmp_path/'adversarial.fits'
    fits.HDUList([fits.PrimaryHDU(image,fits.Header(dict(PIXSCALE=.12,MACRO_TE='invalid'))),
        fits.ImageHDU(noise,name='RMS'),fits.ImageHDU(psf,name='PSF'),fits.ImageHDU(light,name='LENS_LIGHT'),
        *[fits.ImageHDU(np.ones((3,3)),name=name) for name in ('BETA_X','BETA_Y','SOURCE','SOURCE_NODES','SOURCE_TRI')]]).writeto(path)
    original=fits.HDUList.__getitem__
    def measured_only(hdus,key):
        if isinstance(key,str) and key.upper() in {'BETA_X','BETA_Y','SOURCE','SOURCE_NODES','SOURCE_TRI'}:
            raise AssertionError('Production analysis read a model/source product')
        return original(hdus,key)
    monkeypatch.setattr(fits.HDUList,'__getitem__',measured_only)
    r=analyze_lens(path,device='cpu',config=small_config())
    assert r.config['analysis']['initializer']['origin']=='image_only_four_starts'
    assert not r.detected


@pytest.mark.parametrize('model_input',[dict(macro_model={'theta_E':1.}),dict(source_grid=np.zeros((4,2)))])
def test_production_entry_rejects_external_mass_or_source_inputs(model_input):
    image,noise,psf,light=inputs()
    with pytest.raises(TypeError,match='unexpected keyword'):
        analyze_lens(image,noise,psf,light,pixel_scale=.12,config=small_config(),**model_input)


def test_pixel_scale_uses_normalized_wcs_units_and_rejects_anisotropic_grid():
    from sublens.io import pixel_scale_from_header
    header=fits.Header(dict(CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1='arcsec',CUNIT2='arcsec',
                           CDELT1=-.1,CDELT2=.1,CRVAL1=0.,CRVAL2=0.))
    assert pixel_scale_from_header(header)==pytest.approx(.1)
    header['CDELT2']=.2
    with pytest.raises(ValueError,match='Non-isotropic'):pixel_scale_from_header(header)


def test_pipeline_wrapper_uses_fits_scale_instead_of_simulation_default(tmp_path):
    from sublens import SubLensPipeline
    image,noise,psf,light=inputs()
    path=tmp_path/'raw.fits'
    fits.HDUList([fits.PrimaryHDU(image,fits.Header(dict(PIXSCALE=.12))),
                  fits.ImageHDU(noise,name='RMS'),fits.ImageHDU(psf,name='PSF'),
                  fits.ImageHDU(light,name='LENS_LIGHT')]).writeto(path)
    pipeline=SubLensPipeline(pixel_scale=.05,N_img=20,N_src=16,device='cpu')
    result=pipeline.analyze_lens(path,config=small_config())
    assert result.observation.pixel_scale==.12
    assert result.detected is False


@pytest.mark.parametrize('change',[dict(source_nodes=()),dict(regularizations=(0.,)),dict(pinn_steps=0),dict(min_delta_chi2=float('nan'))])
def test_analysis_rejects_invalid_configuration_before_fitting(change):
    image,noise,psf,light=inputs()
    with pytest.raises(ValueError):analyze_lens(image,noise,psf,light,pixel_scale=.12,config=small_config(**change))
