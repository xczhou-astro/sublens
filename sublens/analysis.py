"""Instrument-independent, image-to-subhalo PINN analysis."""
from dataclasses import asdict, dataclass, replace
from time import perf_counter
import numpy as np

from .execution import torch_backend
from .observation import LensingObservation
from .data import load_lens_data
from .grids import make_image_rays
from .inversion.initialization import ParametricInitializer
from .inversion.macro_parameters import vector
from .inversion.adaptive_refinement import _select_source


@dataclass(frozen=True)
class AnalysisConfig:
    """Budgets and declared angular priors; defaults match the tested workflow."""
    source_nodes: tuple = (300, 500)
    regularizations: tuple = (2e-6, 2e-4, .02, 2., 20.)
    validation_fraction: float = .2
    validation_seed: int = 20261120
    oversample: int = 4
    initializer_oversample: int = 2
    initializer_steps: int = 250
    macro_steps: int = 45
    pinn_steps: int = 90
    proposal_steps: int = 70
    joint_steps: int = 80
    joint_starts: int = 2
    top_k: int = 5
    adaptive_source: bool = True
    adaptive_seeds: tuple = (22, 42)
    full_refine_steps: int = 140
    full_refine_starts: int = 3
    initial_max_strength: float = .12
    max_strength: float = .30
    refinement_radius: float = .70
    free_slope: bool = False
    min_delta_chi2: float = 25.

    def validate(self):
        if not self.source_nodes or any(int(n)!=n or n<4 for n in self.source_nodes):
            raise ValueError('source_nodes must contain integer node counts >=4')
        if not self.regularizations or not np.isfinite(self.regularizations).all() or min(self.regularizations)<=0:
            raise ValueError('regularizations must contain positive finite values')
        for name in ('oversample','initializer_oversample','initializer_steps','macro_steps','pinn_steps',
                     'proposal_steps','joint_steps','joint_starts','top_k','full_refine_steps','full_refine_starts'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0<self.validation_fraction<1 or not self.adaptive_seeds:
            raise ValueError('Use a validation fraction in (0,1) and nonempty adaptive_seeds')
        for name in ('initial_max_strength','max_strength','refinement_radius','min_delta_chi2'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be positive and finite')
        if self.max_strength<self.initial_max_strength:
            raise ValueError('max_strength must not be smaller than initial_max_strength')


def _input_observation(image, noise, psf, lens_light, pixel_scale, mask, native_rays,
                       lens_id, config, image_is_lens_subtracted):
    """Adapt loaded measured data to the internal fitting representation."""
    data=load_lens_data(image,noise=noise,psf=psf,lens_light=lens_light,
        pixel_scale=pixel_scale,mask=mask,native_rays=native_rays,lens_id=lens_id,
        image_is_lens_subtracted=image_is_lens_subtracted)
    rays=make_image_rays(data.image.shape,data.pixel_scale,config.oversample) if data.native_rays is None else data.native_rays
    # Identity source rays are a construction placeholder, not a supplied macro.
    # They are replaced by image-fitted rays before CV or PINN is run.
    obs=LensingObservation.from_arrays(data.image,data.noise,data.psf,pixel_scale=data.pixel_scale,
        mask=data.mask,bx=rays[0],by=rays[1],native_rays=rays,lens_id=data.lens_id,
        lens_light=data.lens_light,image_is_lens_subtracted=data.image_is_lens_subtracted)
    return replace(obs,raw_image=data.raw_image)


def analyze_lens(image, noise=None, psf=None, lens_light=None, *, pixel_scale=None,
                 mask=None, device='auto', dtype=None, config=None,
                 native_rays=None, lens_id=None, image_is_lens_subtracted=None, verbose=False):
    """Analyze LensData (or measured arrays/FITS) with the PINN-only workflow.

    noise is a standard-deviation map (or positive scalar), in image units.
    Supplied lens_light is subtracted once and held fixed. No macro is required:
    four image-only SIE/shear/Sersic starts initialize the nonnegative Delaunay
    reconstruction. External macro/source model inputs are not accepted here.
    FITS mass parameters are ignored and model-dependent BETA_* is not loaded.

    Returns PinnHaloSearchResult: .detected, .position, .strength, .detection(),
    .summary(), .save_fits(), and existing source/residual plotting interfaces.
    A positive fit-improvement decision is not calibrated detection significance.
    Numerical failures raise, rather than returning a negative detection.
    """
    config=AnalysisConfig() if config is None else config
    if not isinstance(config,AnalysisConfig):raise TypeError('config must be AnalysisConfig')
    config.validate()
    started=perf_counter();timings={}
    data=load_lens_data(image,noise=noise,psf=psf,lens_light=lens_light,
        pixel_scale=pixel_scale,mask=mask,native_rays=native_rays,lens_id=lens_id,
        image_is_lens_subtracted=image_is_lens_subtracted)
    obs=_input_observation(data,None,None,None,None,None,None,None,config,None)
    def progress(message):
        if verbose:print(message,flush=True)
    with torch_backend(device,dtype=dtype) as execution:
        now=perf_counter()
        progress('Initializing macro from four image-only fits')
        x,y=make_image_rays(obs.image.shape,obs.pixel_scale,config.initializer_oversample)
        initial_obs=replace(obs,bx=x,by=y,native_rays=(x,y))
        initial=ParametricInitializer(initial_obs).fit(maxiter=config.initializer_steps)
        initial.pop('image_model')
        macro=initial['macro'];initial['origin']='image_only_four_starts'
        obs=obs.with_macro(macro)
        initial['macro']=obs.macro_model
        timings['initialization_seconds']=perf_counter()-now;now=perf_counter()
        progress('Selecting source complexity and regularization by internal pixel CV')
        indices=np.flatnonzero(obs.mask)
        if len(indices)<2:raise ValueError('Source CV needs at least two valid pixels')
        validation=np.zeros(obs.image.shape,dtype=bool)
        count=min(len(indices)-1,max(1,int(config.validation_fraction*len(indices))))
        validation.ravel()[np.random.default_rng(config.validation_seed).choice(indices,count,replace=False)]=True
        selected,trials=_select_source(obs,obs.bx,obs.by,validation,config.source_nodes,
                                        config.regularizations,(42,),'centroidal')
        timings['source_selection_seconds']=perf_counter()-now;now=perf_counter()
        progress('Fitting seven macro parameters with a profiled Delaunay source')
        v=vector(obs.macro_model)
        bounds=[(v[0]+np.log(.8),v[0]+np.log(1.2)),(.3,.995),(v[2]-.4,v[2]+.4),
                (v[3]-.25,v[3]+.25),(v[4]-.25,v[4]+.25),
                (max(-2.,v[5]-.25),min(2.,v[5]+.25)),(max(-2.,v[6]-.25),min(2.,v[6]+.25))]
        macro_fit=obs.fit_macro_torch(source_nodes=selected['source_nodes'],
                                    regularization=selected['regularization'],bounds=bounds,maxiter=config.macro_steps)
        macro_fit.pop('source')
        obs=obs.with_macro(macro_fit['macro_model'])
        timings['macro_seconds']=perf_counter()-now;now=perf_counter()
        progress('PINN proposals and initial matched physical refinement')
        result=obs.search_pinn(source_nodes=selected['source_nodes'],regularization=selected['regularization'],
            top_k=config.top_k,pinn_steps=config.pinn_steps,refine_steps=config.proposal_steps,
            max_strength=config.initial_max_strength,seed_strength=min(.015,.5*config.initial_max_strength),
            mesh_strategy='centroidal',cross_fit=True)
        if result.candidates:
            result=result.refine_joint(obs.macro_model,maxiter=config.joint_steps,n_starts=config.joint_starts,retry_failed=True)
        timings['pinn_and_initial_refinement_seconds']=perf_counter()-now;now=perf_counter()
        provisional=result.detection(min_delta_chi2=config.min_delta_chi2)['detected']
        if provisional and config.adaptive_source:
            progress('Adapting the common source basis for null and halo hypotheses')
            result=result.adapt_source_basis(source_nodes=config.source_nodes,
                regularizations=config.regularizations,seeds=config.adaptive_seeds)
        timings['adaptive_source_seconds']=perf_counter()-now;now=perf_counter()
        # Once a system passes initial screening, retain its competing proposals
        # through the full fit; an intermediate source update is not a veto.
        full=bool(provisional and result.candidates)
        if full:
            progress('Jointly refining every retained candidate, all macro parameters and source')
            result=result.refine_full(maxiter=config.full_refine_steps,n_starts=config.full_refine_starts,
                max_strength=config.max_strength,refinement_radius=config.refinement_radius,free_slope=config.free_slope)
        timings['full_refinement_seconds']=perf_counter()-now
        result=result.merge_candidates(radius=obs.pixel_scale)
        timings['total_seconds']=perf_counter()-started
        analysis=dict(config=asdict(config),timings=timings,initializer=initial,macro_fit=macro_fit,
            input_data=data.provenance,data_loading_version='observed_data_v1',
            workflow_version='image_only_single_full_refinement_v2',
            external_macro_input_accepted=False,external_source_input_accepted=False,
            adaptive_nonlinear_refinement=False,
            selected_source=selected,source_cv_trials=trials,source_cv_is_internal=True,
            source_cv_is_independent_detection_validation=False,
            full_refinement_triggered=bool(full),preliminary_detection=bool(provisional),
            foreground='supplied lens light subtracted once and fixed' if obs.lens_light is not None else 'no foreground supplied',
            psf='supplied normalized kernel, fixed',noise='supplied RMS, diagonal likelihood',
            coordinate_frame='input angular image rays; otherwise x increases with columns, y with rows, origin at image centre',
            actual_subpixel_count=int(obs.bx.shape[-1]),
            published_mass_or_source_used_implicitly=False,significance_calibrated=False)
        return replace(result,config={**result.config,'execution':execution.to_dict(),
                                      'detection_min_delta_chi2':config.min_delta_chi2,'analysis':analysis})
