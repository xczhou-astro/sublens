"""Conditional DR1 joint positive-source / SIE+shear / lambda inference.

This experiment keeps the published Sersic foreground subtraction and PSF.
Published mass/source/mesh are not inference inputs. The source grid is fixed
before sampling, from training pixels at an image-scale initial SIE model.

In normalized source t=s/flux_scale, R=G.T G + .01/K I, the target in
(v, t, u=log10(lambda)) is
 -chi2/2 - Nref*10**u*t.T R t/2 + K*ln(10)*u/2,
inside t>=0 and finite uniform v,u bounds. The positive-GMRF orthant constant
is independent of the scalar lambda, and fixed R is independent of macro v.
This includes the lambda prior normalization and integrates source uncertainty
by sampling, rather than exponentiating a profiled reduced-chi2 objective.

Macro updates mix fixed-source MH and exact affine Gaussian transport MH.
For transport, t'=mu'+L'**(-T) L.T (t-mu), L L.T=precision.
The Jacobian is det(L)/det(L'); any negative proposed flux is rejected.
Both kernels preserve the same truncated joint posterior. Adaptation stops
at burn-in. Coordinate Gibbs samples the positive Gaussian source block;
lambda|t is a bounded Gamma(K/2, rate=Nref*t.T R t/2).
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from numba import njit
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.ndimage import binary_dilation,gaussian_filter
from scipy.signal import fftconvolve
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from astropy.io import fits
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from euclid_grade_a_10 import ProfiledMacroFit,effective_sie_radius,source_shift
from euclid_dr1_fixed_source import source_scores
from sublens import LensingObservation
from sublens.grids import make_image_rays
from sublens.inversion.source_plane import PixelSourceProblem,_blur_operator

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'sublens_testing/euclid_dr1_one_lens'
OUT=PILOT/'joint_macro_lambda'
BOUNDS=np.array([(np.log(.6),np.log(2.5)),(.3,.995),(-np.pi,np.pi),(-2.5,2.5),(-2.5,2.5),(-1.5,1.5),(-1.5,1.5)])
UBOUNDS=(-10.,2.)


def save_json(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,default=convert,indent=2,allow_nan=False)+'\n')


@njit(cache=True)
def positive_gaussian_sweeps(precision,rhs,source,sweeps,seed):
    """Exact univariate truncated-normal updates; stable in the far tail."""
    np.random.seed(seed)
    source=source.copy()
    residual=rhs-precision@source
    for step in range(sweeps):
        # A random permutation gives a valid composition of Gibbs kernels.
        for i in np.random.permutation(len(source)):
            sigma=1/np.sqrt(precision[i,i])
            mean=source[i]+residual[i]/precision[i,i]
            lower=-mean/sigma
            if lower<=0:
                z=np.random.normal()
                while z<lower:z=np.random.normal()
            else:
                alpha=(lower+np.sqrt(lower*lower+4))/2
                z=lower+np.random.exponential()/alpha
                while np.random.random()>np.exp(-.5*(z-alpha)**2):
                    z=lower+np.random.exponential()/alpha
            value=max(0.,mean+sigma*z)
            delta=value-source[i];source[i]=value
            residual-=precision[:,i]*delta
    return source


def log_target(chi2,source,regularizer,nref,u):
    return -.5*chi2-.5*nref*10**u*(source@regularizer@source)+.5*len(source)*np.log(10)*u


def read_observation():
    """Only VIS, noise, PSF, mask and explicitly conditional foreground input."""
    row=json.loads((PILOT/'selection.json').read_text())['lens']
    folder=ROOT/'dr1_catalogue_v5_full'/row['tile_id']
    info=json.loads((folder/'dataset/info.json').read_text())
    with fits.open(folder/'dataset'/f'{folder.name}.fits') as h:
        raw=np.array(h['VIS_BGSUB'].data,dtype=float)
        rms=np.array(h['VIS_RMS'].data,dtype=float)
        psf=np.array(h['VIS_PSF'].data,dtype=float)
    foreground=np.array(fits.getdata(folder/'model.fits','VIS_GALAXY_0'),dtype=float)
    x,y=make_image_rays(raw.shape,float(info['pixel_scale']),4)
    cy,cx=info['mask_centre']
    mask=np.hypot(x.mean(-1)-cx,y.mean(-1)+cy)<=info['mask_radius']
    mask&=fits.getdata(folder/'dataset/segmentation/artefact_binary.fits')==0
    mask&=np.isfinite(raw)&np.isfinite(foreground)&np.isfinite(rms)&(rms>0)&(rms<1e7)
    # An image-scale start only; no published mass values are read here.
    macro=dict(theta_E=1.3,q=.85,phi=0.,gamma=2.,gamma_ext=.03,phi_ext=0.,center=[0.,0.])
    obs=LensingObservation.from_arrays(raw-foreground,rms,psf,pixel_scale=float(info['pixel_scale']),
        mask=mask,macro_model=macro,native_rays=(x,y),lens_id=row['unique_identifier'])
    xx,yy=x.mean(-1),y.mean(-1)
    heldout=np.zeros(mask.shape,dtype=bool)
    for cx,cy in ((1.,1.),(-1.,-1.)):
        heldout|=(abs(xx-cx)<.2)&(abs(yy-cy)<.2)
    heldout&=mask
    guard=binary_dilation(heldout,structure=np.ones(psf.shape,dtype=bool))&mask
    train=mask&~guard
    return obs,train,heldout,guard,folder


class JointProblem:
    def __init__(self,obs,train,points):
        self.obs=obs;self.train=train
        self.problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,train,obs.bx,obs.by,
            source_grid=Delaunay(points),regularization=.0002)
        # The exact released PSF, including tiny negative lobes, is preserved.
        self.problem._blur=_blur_operator(obs.image.shape,train,self.problem.convolution_psf)
        self.ray=ProfiledMacroFit.__new__(ProfiledMacroFit)
        self.ray.obs=replace(obs,mask=train);self.ray.problem=self.problem
        x,y=obs.image_rays();self.ray.ray_shape=x.shape
        self.ray.coordinates=torch.tensor(np.column_stack((x.ravel(),y.ravel())),dtype=torch.float64)
        self.regularizer=self.problem._gradient_gram.toarray()+np.eye(self.problem.size)*.01/self.problem.size
        self.nref=int(obs.mask.sum());self.ntrain=int(train.sum())
        self.target=obs.image[train]/obs.rms[train]

    def geometry(self,v):
        with torch.no_grad():rays=self.ray.rays(torch.tensor(v,dtype=torch.float64)).numpy()
        native=self.problem._native_mapping(rays[...,0],rays[...,1])
        operator=self.problem._weighted_operator(native)*np.sqrt(self.ntrain)
        return native,operator.tocsr()

    def state(self,v,u):
        native,operator=self.geometry(v)
        gram=(operator.T@operator).toarray()
        rhs=np.asarray(operator.T@self.target).ravel()
        state=dict(v=v.copy(),native=native,operator=operator,gram=gram,rhs=rhs)
        self.condition(state,u)
        return state

    def condition(self,state,u):
        state['precision']=state['gram']+self.nref*10**u*self.regularizer
        state['chol']=cholesky(state['precision'],lower=True)
        state['mean']=cho_solve((state['chol'],True),state['rhs'])

    def chi2(self,state,t):
        r=state['operator']@t-self.target
        return float(r@r)

    def profile(self,v,regularization=.0002):
        # Training optimizer convention uses chi2/Ntrain + lambda_train R.
        self.problem=self.problem.with_regularization(regularization*self.nref/self.ntrain)
        self.ray.problem=self.problem
        return self.ray.objective(v)


def prepare():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    obs,train,test,guard,folder=read_observation()
    grid=PixelSourceProblem(obs.image,obs.rms,obs.psf,train,obs.bx,obs.by,
        source_nodes=500,mesh_strategy='centroidal',seed=22,regularization=.0002).grid
    np.savez_compressed(OUT/'design.npz',points=grid.points,train=train,heldout=test,guard=guard)
    save_json(OUT/'protocol.json',dict(lens=obs.lens_id,macro_model='SIE+shear, slope fixed at 2',
        source='nonnegative Delaunay coefficients; one fixed own mesh, not published mesh',
        source_grid_initial_macro=obs.macro_model,source_nodes=len(grid.points),
        macro_parameterization=['ln_theta_E','q','phi','center_x/.2','center_y/.2','g1/.2','g2/.2'],
        macro_uniform_bounds=BOUNDS,log10_lambda_uniform_bounds=UBOUNDS,
        positive_source_prior='precision=Nref*lambda*(G.T@G+.01/K*I)',
        full_chi2_likelihood=True,lambda_prior_normalization_included=True,nref=int(obs.mask.sum()),
        train_pixels=int(train.sum()),test_pixels=int(test.sum()),excluded_guard_pixels=int((guard&~test).sum()),
        holdout='two predeclared 0.4 arcsec square blocks at (+1,+1),(-1,-1); full PSF support guard',
        conditional_foreground='published Sersic deblending VIS_GALAXY_0 subtracted and fixed; estimated from full image',
        psf='released VIS_PSF normalized, shape and tiny negative lobes fixed',
        noise='released RMS with diagonal Gaussian likelihood; coadd correlations not modeled',
        four_chains_from_four_image_based_profile_fits=True,adaptation='scalar proposal scales during burn-in only',
        published_mass_used_for_initialization=False,published_source_used_for_tuning=False,
        published_source_or_mesh_used_for_inference=False,independent_observations=False,
        validation='internal blocked conditional prediction; foreground derived from full data limits independence',
        no_subhalo_fitted=True))
    j=JointProblem(obs,train,grid.points)
    base=ProfiledMacroFit.vector(obs.macro_model)
    starts=[]
    for scale,q,phi in ((1.,.85,0.),(.9,.9,.6),(1.1,.7,-.6),(1.,.8,1.2)):
        v=base.copy();v[0]+=np.log(scale);v[1]=q;v[2]=phi;v[5:]=0.;starts.append(v)
    records=[]
    for index,start in enumerate(starts):
        cache={}
        def fun(v):
            value,gradient,source=j.profile(v)
            if not cache or value<cache['value']:
                cache.update(value=value,v=v.copy(),coefficients=source.coefficients.copy())
            return value,gradient
        result=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=BOUNDS,
            options=dict(maxiter=180,maxls=30,ftol=1e-10,gtol=1e-6))
        records.append(dict(start=start,value=cache['value'],v=cache['v'],converged=bool(result.success),message=str(result.message)))
        np.savez_compressed(OUT/f'initial_{index}.npz',v=cache['v'],coefficients=cache['coefficients'])
        save_json(OUT/'initializations.json',records)
        print(f'initial {index}: Q={cache["value"]:.6f}, success={result.success}, theta={np.exp(cache["v"][0]):.6f}',flush=True)
    best=int(np.argmin([r['value'] for r in records]))
    initial=np.load(OUT/f'initial_{best}.npz')
    # Local conditional Fisher preconditioner for the macro MH proposal.
    v=initial['v'];t=initial['coefficients']/j.problem.flux_scale
    state=j.state(v,-3.)
    jac=[]
    for k in range(7):
        step=1e-5;vp=v.copy();vm=v.copy();vp[k]+=step;vm[k]-=step
        _,ap=j.geometry(vp);_,am=j.geometry(vm)
        jac.append((ap@t-am@t)/(2*step))
    jac=np.array(jac).T
    covariance=np.linalg.inv(jac.T@jac+np.diag([1.,1.,.1,1.,1.,1.,1.]))
    np.savez_compressed(OUT/'initial.npz',v=v,coefficients=initial['coefficients'],proposal_chol=cholesky(covariance,lower=True))
    print('prepared',flush=True)


def run_chain(arguments):
    chain,steps,burn=arguments;torch.set_num_threads(1)
    obs,train,test,guard,folder=read_observation();design=np.load(OUT/'design.npz')
    assert np.array_equal(train,design['train'])
    j=JointProblem(obs,train,design['points']);initial=np.load(OUT/f'initial_{chain}.npz')
    rng=np.random.default_rng(2026111900+chain)
    v=initial['v'].copy();t=initial['coefficients']/j.problem.flux_scale
    u=-3.+.4*(chain-1.5);state=j.state(v,u)
    t=positive_gaussian_sweeps(state['precision'],state['rhs'],t,200,int(rng.integers(2**30)))
    jac=[]
    for k in range(7):
        vp=v.copy();vm=v.copy();vp[k]+=1e-5;vm[k]-=1e-5
        _,ap=j.geometry(vp);_,am=j.geometry(vm)
        jac.append((ap@t-am@t)/2e-5)
    jac=np.array(jac).T
    proposal_chol=cholesky(np.linalg.inv(jac.T@jac+np.eye(7)),lower=True)
    scales=np.array([1.,10.]);accepted=np.zeros(2,dtype=int);attempted=np.zeros(2,dtype=int)
    recent_acc=np.zeros(2);recent_n=np.zeros(2)
    params=np.zeros((steps,11));sources=np.zeros((steps,j.problem.size),dtype=np.float32)
    start=perf_counter();best=None
    for iteration in range(steps):
        t=positive_gaussian_sweeps(state['precision'],state['rhs'],t,10,int(rng.integers(2**30)))
        rate=.5*j.nref*(t@j.regularizer@t)
        lam=rng.gamma(j.problem.size/2,1/rate)
        while not 10**UBOUNDS[0]<=lam<=10**UBOUNDS[1]:lam=rng.gamma(j.problem.size/2,1/rate)
        u=float(np.log10(lam));j.condition(state,u)
        kind=int(rng.random()<.35)
        proposed_v=v+scales[kind]*(proposal_chol@rng.normal(size=7))
        attempted[kind]+=1;recent_n[kind]+=1
        chi2=j.chi2(state,t)
        current_log=log_target(chi2,t,j.regularizer,j.nref,u)
        if np.all(proposed_v>=BOUNDS[:,0]) and np.all(proposed_v<=BOUNDS[:,1]):
            if kind:
                proposed=j.state(proposed_v,u)
                z=state['chol'].T@(t-state['mean'])
                proposed_t=proposed['mean']+solve_triangular(proposed['chol'].T,z,lower=False)
                logjac=np.log(np.diag(state['chol'])).sum()-np.log(np.diag(proposed['chol'])).sum()
            else:
                native,operator=j.geometry(proposed_v)
                proposed=dict(v=proposed_v,native=native,operator=operator)
                proposed_t=t;logjac=0.
            if np.all(proposed_t>=0):
                proposed_chi=j.chi2(proposed,proposed_t)
                proposed_log=log_target(proposed_chi,proposed_t,j.regularizer,j.nref,u)
                if np.log(rng.random())<proposed_log-current_log+logjac:
                    if not kind:
                        proposed.update(gram=(operator.T@operator).toarray(),rhs=np.asarray(operator.T@j.target).ravel())
                        j.condition(proposed,u)
                    state=proposed;v=proposed_v;t=proposed_t;chi2=proposed_chi
                    current_log=proposed_log;accepted[kind]+=1;recent_acc[kind]+=1
        if iteration<burn and (iteration+1)%50==0:
            scales*=np.exp(np.clip(recent_acc/np.maximum(recent_n,1)-.25,-.15,.15))
            scales=np.clip(scales,.03,200.)
            recent_acc[:]=0;recent_n[:]=0
        macro=ProfiledMacroFit.model(v)
        params[iteration]=np.r_[v,u,chi2,effective_sie_radius(macro,angles=256),current_log]
        sources[iteration]=t*j.problem.flux_scale
        if iteration>=burn and (best is None or current_log>best[0]):best=(current_log,iteration)
        if (iteration+1)%100==0 or iteration==steps-1:
            np.savez_compressed(OUT/f'chain_{chain}.npz',parameters=params[:iteration+1],sources=sources[:iteration+1],
                accepted=accepted,attempted=attempted,scales=scales,burn=burn,chain=chain,elapsed=perf_counter()-start)
            print(f'chain {chain}: {iteration+1}/{steps} lambda={10**u:.5g} chi/N={chi2/j.ntrain:.4f} acc={np.round(accepted/attempted,3)} sec={perf_counter()-start:.1f}',flush=True)
    return chain


def diagnostics(samples):
    """Split-Rhat and autocorrelation ESS; both are Monte Carlo diagnostics."""
    from emcee.autocorr import integrated_time
    m,n,k=samples.shape;n2=n//2
    split=np.concatenate((samples[:,:n2],samples[:,-n2:]),axis=0)
    within=np.var(split,axis=1,ddof=1).mean(axis=0)
    between=n2*np.var(split.mean(axis=1),axis=0,ddof=1)
    rhat=np.sqrt(((n2-1)/n2*within+between/n2)/within)
    taus=[]
    for i in range(k):
        # Average chain autocorrelation; large Rhat separately flags slow between-mode exploration.
        tau=float(integrated_time(samples[:,:,i].T,quiet=True,tol=0)[0])
        taus.append(max(1.,tau))
    return rhat,np.array(taus),m*n/np.array(taus)


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    obs,train,test,guard,folder=read_observation();design=np.load(OUT/'design.npz')
    grid=Delaunay(design['points']);j=JointProblem(obs,train,grid.points)
    saved=[np.load(OUT/f'chain_{i}.npz') for i in range(4)]
    arrays=[];coeff=[]
    for record in saved:
        burn=int(record['burn']);arrays.append(record['parameters'][burn:]);coeff.append(record['sources'][burn:])
    length=min(map(len,arrays));arrays=np.array([a[-length:] for a in arrays]);coeff=np.array([a[-length:] for a in coeff])
    rhat,tau,ess=diagnostics(arrays[:,:,:8])
    source_rhat,source_tau,source_ess=diagnostics(coeff.astype(float))
    # Official source and mass are first loaded AFTER all chains are frozen.
    from sublens.survey.euclid_dr1 import load_euclid_dr1_lens
    reference_data=load_euclid_dr1_lens(folder,oversample=4)
    with fits.open(PILOT/'source_comparison.fits') as h:
        header=h[1].header.copy()
    axisx=(np.arange(301)+1-header['CRPIX1'])*header['CDELT1']+header['CRVAL1']
    axisy=(np.arange(301)+1-header['CRPIX2'])*header['CDELT2']+header['CRVAL2']
    xx,yy=np.meshgrid(axisx,axisy)
    official=LinearNDInterpolator(reference_data.source_points,reference_data.source_values,fill_value=0)(xx,yy)
    previous=fits.getdata(PILOT/'fixed_official_macro_source/our_centroidal_cv.fits','SOURCE')
    # A fixed deterministic thinning for product rendering, not a selection by official similarity.
    selected=np.linspace(0,length-1,min(75,length),dtype=int)
    source_maps=[];models=[];test_chis=[]
    for c in range(4):
        for i in selected:
            v=arrays[c,i,:7];macro=ProfiledMacroFit.model(v);shift=source_shift(macro)
            source_maps.append(LinearNDInterpolator(grid,coeff[c,i],fill_value=0)(xx+shift[0],yy+shift[1]))
            native,_=j.geometry(v)
            image=fftconvolve((native@coeff[c,i]).reshape(obs.image.shape),obs.psf,mode='same')
            models.append(image)
            test_chis.append(np.mean(((obs.image[test]-image[test])/obs.rms[test])**2))
    source_maps=np.array(source_maps);models=np.array(models)
    source_mean=source_maps.mean(0);source_sd=source_maps.std(0);model_mean=models.mean(0);model_sd=models.std(0)
    residual=(obs.image-model_mean)/obs.rms
    scores=source_scores(official,source_mean,header['CDELT1'])
    names=['ln_theta_E','q','phi','center_x/.2','center_y/.2','g1/.2','g2/.2','log10_lambda']
    chain_quantiles=np.quantile(arrays[:,:,:8].reshape(-1,8),[.16,.5,.84],axis=0)
    report=dict(lens=obs.lens_id,source_metrics=scores,
        posterior_status='convergence checks passed' if np.all(rhat<1.05) and np.all(ess>400) and np.all(source_rhat<1.05) and np.all(source_ess>400) else 'exploratory: chains not demonstrably converged',
        marginal_summaries={name:dict(q16=chain_quantiles[0,i],median=chain_quantiles[1,i],q84=chain_quantiles[2,i],
            split_rhat=rhat[i],autocorrelation_time=tau[i],ess=ess[i]) for i,name in enumerate(names)},
        source_chain_diagnostics=dict(max_split_rhat=float(source_rhat.max()),median_split_rhat=float(np.median(source_rhat)),minimum_ess=float(source_ess.min())),
        lambda_quantiles=np.quantile(10**arrays[:,:,7],[.16,.5,.84]),
        effective_einstein_radius_quantiles=np.quantile(arrays[:,:,9],[.16,.5,.84]),
        official_effective_einstein_radius=effective_sie_radius(reference_data.observation.macro_model),
        lambda_macro_correlations=dict(zip(names[:7],np.corrcoef(arrays[:,:,:8].reshape(-1,8).T)[7,:7])),
        mean_model_train_chi2_per_pixel=float(np.mean(residual[train]**2)),
        mean_model_heldout_chi2_per_pixel=float(np.mean(residual[test]**2)),
        sampled_model_heldout_chi2_quantiles=np.quantile(test_chis,[.16,.5,.84]),
        heldout_standardized_predictive_rms=float(np.sqrt(np.mean(((obs.image[test]-model_mean[test])/np.sqrt(obs.rms[test]**2+model_sd[test]**2))**2))),
        prior_and_mesh_conditional=True,published_foreground_fixed=True,psf_fixed=True,slope_fixed=2.,
        official_source_only_after_frozen_sampling=True,no_subhalo_fitted=True,
        accepted=[a['accepted'] for a in saved],attempted=[a['attempted'] for a in saved],
        kept_steps_per_chain=length,rendered_joint_draws=len(models),source_samples_include_macro_frame_translation=True)
    save_json(OUT/'summary.json',report)
    hdus=[fits.PrimaryHDU()]
    for name,value in [('DATA',obs.image),('RMS',obs.rms),('PSF',obs.psf),('TRAIN',train),('HELDOUT',test),('GUARD',guard),
                       ('MODEL_MEAN',model_mean),('MODEL_SD',model_sd),('RESIDUAL',obs.image-model_mean)]:
        hdus.append(fits.ImageHDU(np.asarray(value,dtype=float),name=name))
    for name,value in [('SOURCE_MEAN',source_mean),('SOURCE_SD',source_sd),('OFFICIAL_SOURCE',official)]:
        hdus.append(fits.ImageHDU(value,header=header.copy(),name=name))
    hdus.append(fits.BinTableHDU.from_columns([fits.Column(name='X',format='D',array=grid.points[:,0]),fits.Column(name='Y',format='D',array=grid.points[:,1])],name='SOURCE_NODES'))
    fits.HDUList(hdus).writeto(OUT/'joint_reconstruction.fits',overwrite=True,checksum=True)
    fig,ax=plt.subplots(2,4,figsize=(18,9),layout='constrained')
    extent=[axisx[0],axisx[-1],axisy[0],axisy[-1]]
    norm=AsinhNorm(linear_width=.03*official.max(),vmin=0,vmax=max(official.max(),source_mean.max()))
    for k,(image,title) in enumerate([(official,'Released source (comparison only)'),(previous,'Previous: fixed released macro / pixel-CV'),
                                    (source_mean,'Joint macro + source + lambda: sample mean'),(source_sd,'Joint source sample scatter')]):
        art=ax[0,k].imshow(image,origin='lower',extent=extent,cmap='magma',norm=norm if k<3 else None)
        ax[0,k].set(title=title,xlabel='βx west [arcsec]',ylabel='βy north [arcsec]');fig.colorbar(art,ax=ax[0,k],shrink=.7)
    ie=np.array([-5,5,-5,5])
    for k,(image,title,cmap,lims) in enumerate([(np.where(obs.mask,obs.image,np.nan),'Foreground-subtracted observation','magma',None),
        (np.where(obs.mask,model_mean,np.nan),'Joint predicted lensed image','magma',None),
        (np.where(obs.mask,residual,np.nan),'(Observed - mean model) / RMS','RdBu_r',(-5,5)),
        (train.astype(float)+2*test+0.4*(guard&~test),'Training / guarded holdout masks','viridis',None)]):
        kw={} if lims is None else dict(vmin=lims[0],vmax=lims[1])
        if k<2:kw['norm']=AsinhNorm(linear_width=.01,vmin=0,vmax=np.percentile(obs.image[obs.mask],99.9))
        art=ax[1,k].imshow(image,origin='lower',extent=ie,cmap=cmap,**kw)
        ax[1,k].set(title=title,xlim=(-2.5,2.5),ylim=(-2.5,2.5),xlabel='x west [arcsec]',ylabel='y north [arcsec]')
        fig.colorbar(art,ax=ax[1,k],shrink=.7)
    fig.suptitle(f'{obs.lens_id} · {report["posterior_status"]}\nFixed published foreground / PSF; SIE slope 2; own fixed source mesh; official source excluded from inference',fontsize=14)
    fig.savefig(OUT/'joint_source_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(13,10),layout='constrained')
    for k,axis in enumerate(axes.ravel()):
        for c in range(4):axis.plot(saved[c]['parameters'][:,k],alpha=.7,lw=.6)
        axis.axvline(int(saved[0]['burn']),color='k',ls='--');axis.set(title=f'{names[k]} · split Rhat={rhat[k]:.3f}',xlabel='Iteration')
    fig.savefig(OUT/'sampling_traces.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    flat=arrays.reshape(-1,11)[::5]
    for axis,k,label in zip(axes,[0,1,9],['theta_E [arcsec]','q','Effective Einstein radius [arcsec]']):
        vals=np.exp(flat[:,k]) if k==0 else flat[:,k]
        axis.hexbin(flat[:,7],vals,gridsize=35,mincnt=1,cmap='viridis');axis.set(xlabel='log10(lambda)',ylabel=label)
    fig.suptitle('Joint samples: correlations conditional on source prior, foreground, PSF and fixed slope')
    fig.savefig(OUT/'lambda_macro_degeneracy.png',dpi=160);plt.close(fig)
    # Verify products and preserve all prior FITS products.
    import hashlib
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    with fits.open(OUT/'joint_reconstruction.fits',checksum=True) as h:
        assert all(a.verify_checksum()==1 for a in h)
        np.testing.assert_allclose(h['DATA'].data-h['MODEL_MEAN'].data,h['RESIDUAL'].data,atol=1e-14)
        assert not np.any(h['TRAIN'].data.astype(bool)&h['GUARD'].data.astype(bool))
    previous_hashes=json.loads((PILOT/'verification.json').read_text())['fits_sha256']
    for name,digest in previous_hashes.items():assert sha(PILOT/name)==digest,name
    save_json(OUT/'verification.json',dict(fits_checksum_passed=True,train_guard_disjoint=True,
        old_pilot_fits_unchanged=len(previous_hashes),analysis_source_sha256=sha(Path(__file__)),
        reconstruction_sha256=sha(OUT/'joint_reconstruction.fits')))
    (OUT/'README.md').write_text('Conditional joint macro/source/lambda experiment on EUCL-DR1-L020080.\n\n'
        'See protocol.json for fixed foreground/PSF/noise assumptions and heldout design; summary.json for actual convergence. '
        'The published source and mass were opened only after chains finished. The source basis uses training pixels and an image-scale initial model. '
        'Traces, correlations, chain arrays and FITS are saved. Sample scatter is an uncertainty estimate only if chain diagnostics support convergence. '
        'Published Sersic foreground was itself fit using the full image, so blocked prediction remains conditional and is not independent exposure validation. '
        'Source similarity is evaluated after inference with a fixed analytic shear-origin coordinate conversion; no registration or intensity rescaling. '
        'The lambda prior is normalized; this is not source-profile chi-square sampling. No subhalo was fit.\n')
    print(json.dumps(report,default=lambda x:x.tolist(),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','sample','summarize']);p.add_argument('--steps',type=int,default=2400);p.add_argument('--burn',type=int,default=1200)
    args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='sample':
        with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(run_chain,[(i,args.steps,args.burn) for i in range(4)]))
    else:summarize()

if __name__=='__main__':main()
