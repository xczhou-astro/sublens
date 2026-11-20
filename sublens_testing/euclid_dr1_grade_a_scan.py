"""Resumable image-initialized point-estimate / PINN survey of DR1 part-1 Grade A.

Only Grade A local directories are eligible. All inspection outcomes remain in
the manifest. Foreground-subtracted images, fixed supplied PSFs/RMS/masks are
used for inference. Official mass/source products enter evaluation only.
Candidate flags are screening rules, not calibrated substructure detections.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from euclid_dr1_batch_macro import ParametricInitializer
from euclid_grade_a_10 import ProfiledMacroFit,effective_sie_radius,source_shift
import euclid_dr1_pilot as pilot
from sublens import LensingObservation,summarize_subhalo
from sublens.execution import current_execution,torch_backend
from sublens.grids import make_image_rays
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.adaptive_refinement import _select_source
from sublens.survey.euclid_dr1 import load_euclid_dr1_lens
from euclid_dr1_fixed_source import source_scores

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dr1_catalogue_v5_full'
OUT=ROOT/'sublens_testing/euclid_dr1_grade_a_part1'
REGS=(2e-6,2e-4,.02,2.,20.)
SEARCH=dict(top_k=5,pinn_steps=90,refine_steps=70,mesh_strategy='centroidal',cross_fit=True)
ADAPT=dict(source_nodes=(300,500),regularizations=REGS,seeds=(22,42),mesh_rounds=1,
    maxiter=80,n_starts=2,retry_failed=True,expand_boundary=True)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,default=convert,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest():
    OUT.mkdir(parents=True,exist_ok=True)
    catalog=ROOT/'downloads/euclid_dr1/dr1_catalogue_15072_master.csv'
    with catalog.open() as f:rows=[r for r in csv.DictReader(f) if r['grade']=='A' and (DATA/r['tile_id']).is_dir()]
    if len({r['tile_id'] for r in rows})!=len(rows):raise ValueError('Grade A tile aliases need explicit resolution')
    rows.sort(key=lambda r:r['unique_identifier'])
    manifest=[]
    for r in rows:
        directory=DATA/r['tile_id']
        required=[directory/'dataset'/f'{directory.name}.fits',directory/'dataset/info.json',directory/'model.fits']
        manifest.append(dict(identifier=r['unique_identifier'],tile_id=r['tile_id'],grade='A',
            inspection=r['inspection_consensus_75'],score=float(r['score']),
            missing_inputs=[str(p.relative_to(ROOT)) for p in required if not p.exists()],catalogue_row=r))
    write_json(OUT/'manifest.json',manifest)
    write_json(OUT/'protocol.json',dict(scope='all local part-1 Grade A directories; all inspection outcomes retained',
        total=len(rows),grade_filter='A only',foreground='published Sersic-stage VIS_GALAXY_0 subtracted and fixed',
        psf='normalized released VIS_PSF, no recentering or clipping',noise='released RMS; diagonal likelihood',
        initial_macro='image-only radial start + four SIE/shear/Sersic fits; published mass not used',
        source_initialization='Sersic for initialization only; final source is own nonnegative Delaunay mesh',
        macro_refinement='seven parameters, fixed slope 2; image-derived starts, bounded local source-profile optimization',
        later_macro_refinement='matched null/halo theta_E and shear, q/phi/centre held fixed',
        pixel_integration='2x2 for analytic initializer, 4x4 for Delaunay, PINN and physical refinement',
        source_nodes=[300,500],regularizations=REGS,source_mesh_seed=42,conditional_pixel_cv_fraction=.2,
        conditional_pixel_cv_seed=20261120,source_similarity_used_for_fitting_or_selection=False,
        official_source_used_for='post-fit histogram and visualization only; common coordinates, no fitted registration',
        radius_convention='area-equivalent tangential critical radius of fitted SIE+shear, not the raw circularized SIE normalization',
        official_radius='COOLEST maximum-likelihood in primary scatter; catalogue median in secondary scatter',
        search=SEARCH,adaptive=ADAPT,adaptive_trigger_delta_chi2=25.,
        candidate_review_rule='delta_chi2>=25 and positive penalized gain; not a significance threshold',
        candidate_quality_flags=['nonconvergence','strength cap','local position boundary','macro bounds','single-macro approximation'],
        calibrated_subhalo_detection=False,independent_holdout_validation=False,
        all_image_plane_panels='observed image minus same fixed lens-light model; model residuals use same original RMS',
        sources_and_macro_are_point_estimates=True,production_workflow_defaults_changed=False,
        catalog_sha256=sha(catalog)))
    resources=Path('/tmp/euclid_part1_resources.json')
    if resources.exists():shutil.copyfile(resources,OUT/'resources.json')
    else:write_json(OUT/'resources.json',dict(resource_detection='not captured in this session',
        cuda_available=torch.cuda.is_available(),mps_available=torch.backends.mps.is_available()))
    return manifest


def read_image(directory,oversample=4):
    """No official mass, source brightness or source mesh is opened here."""
    info=json.loads((directory/'dataset/info.json').read_text())
    with fits.open(directory/'dataset'/f'{directory.name}.fits') as h:
        raw=np.array(h['VIS_BGSUB'].data,dtype=float);rms=np.array(h['VIS_RMS'].data,dtype=float)
        psf=np.array(h['VIS_PSF'].data,dtype=float);header=h['VIS_BGSUB'].header.copy()
    lens=np.array(fits.getdata(directory/'model.fits','VIS_GALAXY_0'),dtype=float)
    if any(abs(header.get(k,0.))>1e-12 for k in ('CD1_2','CD2_1','PC1_2','PC2_1')):raise ValueError('Rotated WCS not supported')
    x,y=make_image_rays(raw.shape,float(info['pixel_scale']),oversample)
    cy,cx=info['mask_centre'];mask=np.hypot(x.mean(-1)-cx,y.mean(-1)+cy)<=info['mask_radius']
    extra=directory/'dataset/segmentation/artefact_binary.fits'
    if extra.exists():mask&=fits.getdata(extra)==0
    mask&=np.isfinite(raw)&np.isfinite(lens)&np.isfinite(rms)&(rms>0)&(rms<1e7)
    generic=dict(theta_E=1.3,q=.85,phi=0.,gamma=2.,gamma_ext=0.,phi_ext=0.,center=[0.,0.])
    obs=LensingObservation.from_arrays(raw-lens,rms,psf,pixel_scale=float(info['pixel_scale']),
        macro_model=generic,mask=mask,native_rays=(x,y),lens_id=directory.name)
    return obs,raw,lens,header


def with_macro(obs,macro):
    return LensingObservation.from_arrays(obs.image,obs.rms,obs.psf,pixel_scale=obs.pixel_scale,
        mask=obs.mask,macro_model=macro,native_rays=obs.image_rays(),lens_id=obs.lens_id)


def source_cv(obs):
    rng=np.random.default_rng(20261120);idx=np.flatnonzero(obs.mask)
    validation=np.zeros(obs.mask.shape,dtype=bool)
    validation.ravel()[rng.choice(idx,max(1,int(.2*len(idx))),replace=False)]=True
    selected,trials=_select_source(obs,obs.bx,obs.by,validation,(300,500),REGS,(42,),'centroidal')
    return selected,trials,validation


def refine_macro(obs,selected):
    initial=ProfiledMacroFit.vector(obs.macro_model)
    # Bounds depend only on the image-initialized fit, never the released mass.
    bounds=[(initial[0]+np.log(.8),initial[0]+np.log(1.2)),(.3,.995),(initial[2]-.4,initial[2]+.4),
        (max(-1.,initial[3]-.25),min(1.,initial[3]+.25)),(max(-1.,initial[4]-.25),min(1.,initial[4]+.25)),
        (max(-1.5,initial[5]-.25),min(1.5,initial[5]+.25)),(max(-1.5,initial[6]-.25),min(1.5,initial[6]+.25))]
    if current_execution() is not None:
        result=obs.fit_macro_torch(source_nodes=selected['source_nodes'],regularization=selected['regularization'],bounds=bounds,maxiter=45)
        vector=result['parameters']
        return dict(macro=result['macro_model'],converged=result['converged'],message=result['optimizer_message'],
            iterations=result['iterations'],objective=result['source'].objective,local_bounds=bounds,
            at_bound=bool(np.any(np.minimum(vector-np.array(bounds)[:,0],np.array(bounds)[:,1]-vector)<1e-5)),
            execution=result['execution'])
    fitter=ProfiledMacroFit(obs,nodes=selected['source_nodes'],regularization=selected['regularization'])
    fitter.problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,
        source_nodes=selected['source_nodes'],regularization=selected['regularization'],seed=42,mesh_strategy='centroidal')
    cache={}
    def fun(v):
        loss,gradient,source=fitter.objective(v)
        if not cache or loss<cache['loss']:cache.update(loss=loss,v=v.copy())
        return loss,gradient
    result=minimize(fun,initial,jac=True,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=45,maxls=25,ftol=1e-8,gtol=2e-5))
    return dict(macro=ProfiledMacroFit.model(cache['v']),converged=bool(result.success),
        message=str(result.message),iterations=int(result.nit),objective=cache['loss'],
        local_bounds=bounds,at_bound=bool(np.any(np.minimum(cache['v']-np.array(bounds)[:,0],np.array(bounds)[:,1]-cache['v'])<1e-5)))


def image_scores(obs,model,arc):
    residual=(obs.image-model)/obs.rms
    return dict(chi2_per_pixel=float(np.mean(residual[obs.mask]**2)),pixels=int(obs.mask.sum()),
        bright_arc_chi2_per_pixel=float(np.mean(residual[arc]**2)) if arc.any() else None,bright_pixels=int(arc.sum()))


def finish(row,result,initial,macro_record,seconds,timing):
    folder=OUT/row['identifier'];directory=DATA/row['tile_id']
    # Inference is frozen before these released mass/source products are read.
    data=load_euclid_dr1_lens(directory,oversample=4);obs=result.observation
    pilot.OUTPUT=folder
    psf=pilot.psf_check(data,directory);write_json(folder/'psf_diagnostics.json',psf)
    grid=Delaunay(data.source_points+source_shift(data.observation.macro_model))
    p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,data.observation.bx,data.observation.by,source_grid=grid,regularization=1.)
    reference=p.render(data.observation.bx,data.observation.by,data.source_values)
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(obs.image,name='LENS_SUB_DATA'),
        fits.ImageHDU(reference,name='REFERENCE_SOURCE'),fits.ImageHDU(obs.image-reference,name='REFERENCE_RESIDUAL')]).writeto(folder/'official_reference.fits',overwrite=True,checksum=True)
    arrays,extent,legacy=pilot.source_maps(data,result)
    pixel=(extent[1]-extent[0])/(arrays['OFFICIAL'].shape[1]-1)
    similarities={key:source_scores(arrays['OFFICIAL'],arrays[key],pixel) for key in ('MACRO_ONLY','MACRO_HALO')}
    x,y=obs.image_rays();radius=np.hypot(x.mean(-1),y.mean(-1))
    arc=obs.mask&(obs.image>=3*obs.rms)&(radius>.2)
    best=result.best_candidate;halo_source=result.baseline if best is None else best.fit.source
    halo_macro=result.macro_model if best is None else best.fit.macro_model
    candidates=[]
    for candidate in result.candidates:
        f=candidate.fit;edge=bool(np.max(np.abs(np.asarray(f.position)-candidate.seed))>=.97*result.refinement_radius)
        at_cap=bool(f.strength>=.98*result.max_strength)
        pixel_xy=np.array([[f.x/obs.pixel_scale+(obs.image.shape[1]-1)/2,f.y/obs.pixel_scale+(obs.image.shape[0]-1)/2]])
        sky=WCS(data.header).all_pix2world(pixel_xy,0)[0]
        candidates.append(dict(rank=candidate.rank,position_xy_arcsec=list(f.position),ra_deg=float(sky[0]),dec_deg=float(sky[1]),
            strength_arcsec=float(f.strength),delta_chi2=float(f.delta_chi2),penalized_gain=float(f.delta_objective),
            converged=bool(f.converged),strength_at_upper_bound=at_cap,position_at_local_bound=edge,
            review_candidate=bool(f.delta_chi2>=25 and f.delta_objective>0),
            passed_optimizer_and_boundary_checks=bool(f.converged and not at_cap and not edge)))
    ref_radius=effective_sie_radius(data.observation.macro_model)
    metrics=dict(identifier=row['identifier'],tile_id=row['tile_id'],grade='A',inspection=row['inspection'],status='completed',
        seconds=seconds,timings=timing,theta_official_maximum_likelihood=ref_radius,
        theta_official_catalogue_median=float(row['catalogue_row'].get('mass_effective_einstein_radius') or 0),
        theta_image_initializer=effective_sie_radius(initial['macro']),theta_macro=effective_sie_radius(result.macro_model),
        theta_halo_host=effective_sie_radius(halo_macro),macro_converged=result.macro_converged,
        initializer_converged=initial['converged'],broad_macro_converged=macro_record['converged'],
        broad_macro_at_local_bound=macro_record['at_bound'],published_mass_used_for_initialization=False,
        source_similarity=similarities,image_scores=dict(official=image_scores(obs,reference,arc),
            macro=image_scores(obs,result.baseline.image_model,arc),halo=image_scores(obs,halo_source.image_model,arc)),
        candidate_count=len(candidates),pinn_coverage_pixels=int(result.residual_potential.mask.sum()),
        review_candidate_count=sum(c['review_candidate'] for c in candidates),candidates=candidates,best_candidate=None,
        detection=result.detection(),subhalo_existence_confirmed=None,independent_significance_calibrated=False,
        execution=current_execution().to_dict() if current_execution() is not None else dict(backend='reference'),
        single_macro_approximation=row['inspection']=='Multi Galaxy Lens',psf_numerically_usable=psf['numerically_usable'])
    if best:
        metrics['best_candidate']=next(c for c in candidates if c['rank']==best.rank)
    # Keep the common pilot plot schema while reporting the richer batch fields.
    data=replace(data,observation=obs)
    pilot.plot_comparison(row['catalogue_row'],data,result,arrays,extent,metrics)
    write_json(folder/'metrics.json',metrics)
    write_json(folder/'input_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in (
        directory/'dataset'/f'{directory.name}.fits',directory/'dataset/info.json',directory/'model.fits',
        directory/'coolest.json',directory/'vis_pix_source_plane_reconstruction_0.csv')})
    return metrics


def run_one(row):
    torch.set_num_threads(1);folder=OUT/row['identifier'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return json.loads((folder/'metrics.json').read_text())
    started=perf_counter();timing={}
    if row['missing_inputs']:
        record=dict(identifier=row['identifier'],grade='A',inspection=row['inspection'],status='missing_input',missing_inputs=row['missing_inputs'])
        write_json(folder/'metrics.json',record);return record
    try:
        obs,raw,lens,header=read_image(DATA/row['tile_id'],oversample=4)
        source_obs=obs
        if (folder/'image_initializer.json').exists():initial=json.loads((folder/'image_initializer.json').read_text())
        else:
            x,y=make_image_rays(obs.image.shape,obs.pixel_scale,2)
            small=LensingObservation.from_arrays(obs.image,obs.rms,obs.psf,pixel_scale=obs.pixel_scale,mask=obs.mask,
                native_rays=(x,y),macro_model=obs.macro_model,lens_id=row['identifier'])
            initial=ParametricInitializer(small).fit(maxiter=250)
            fits.ImageHDU(initial.pop('image_model'),name='SERSIC_INITIALIZER').writeto(folder/'image_initializer.fits',checksum=True)
            write_json(folder/'image_initializer.json',initial)
        timing['image_initializer_seconds']=initial['seconds']
        obs=with_macro(obs,initial['macro'])
        now=perf_counter()
        if (folder/'source_cv.json').exists():selected=json.loads((folder/'source_cv.json').read_text())['selected']
        else:
            selected,trials,validation=source_cv(obs)
            write_json(folder/'source_cv.json',dict(selected=selected,trials=trials,conditional_not_independent=True))
            fits.ImageHDU(validation.astype(np.uint8),name='VALIDATION').writeto(folder/'source_cv_mask.fits',checksum=True)
        timing['source_cv_seconds']=perf_counter()-now;now=perf_counter()
        if (folder/'macro_fit.json').exists():macro_record=json.loads((folder/'macro_fit.json').read_text())
        else:
            macro_record=refine_macro(obs,selected);write_json(folder/'macro_fit.json',macro_record)
        timing['pixel_macro_seconds']=perf_counter()-now
        obs=with_macro(obs,macro_record['macro']);obs.to_fits(folder/'observation.fits',overwrite=True)
        from run import load_result
        now=perf_counter()
        if (folder/'screen.fits').exists():result=load_result(folder/'screen.fits')
        else:
            result=obs.search_pinn(source_nodes=selected['source_nodes'],regularization=selected['regularization'],**SEARCH)
            result=result.refine_joint(obs.macro_model,maxiter=80,n_starts=2,retry_failed=True)
            result.save_fits(folder/'screen.fits',overwrite=True)
        timing['pinn_search_and_joint_seconds']=perf_counter()-now;now=perf_counter()
        if (folder/'result.fits').exists():result=load_result(folder/'result.fits')
        else:
            if any(c.fit.delta_chi2>=25 and c.fit.delta_objective>0 for c in result.candidates):
                result=result.refine_adaptive(obs.macro_model,**ADAPT)
            result=result.merge_candidates(radius=obs.pixel_scale)
            result.save_fits(folder/'result.fits',overwrite=True);result.save_json(folder/'result.json',overwrite=True)
        timing['adaptive_and_save_seconds']=perf_counter()-now
        metrics=finish(row,result,initial,macro_record,perf_counter()-started,timing)
        print(f"DONE {row['identifier']} {metrics['seconds']:.1f}s Reff={metrics['theta_macro']:.3f} candidates={metrics['review_candidate_count']}",flush=True)
        return metrics
    except Exception as error:
        traceback.print_exc()
        record=dict(identifier=row['identifier'],grade='A',inspection=row['inspection'],status='failed',
            error_type=type(error).__name__,error=str(error),seconds=perf_counter()-started)
        write_json(folder/'failure.json',record);return record


def write_compact_results(records):
    """One row per processed system; failed/missing fits remain unassessed."""
    rows=[]
    for record in records:
        identifier=record['identifier'];status=record['status']
        row=dict(identifier=identifier,status=status,detected=None,x_arcsec=None,y_arcsec=None,
                 strength_arcsec=None,delta_chi2=None,candidate_rank=None,refinement_stage=None)
        if status=='completed':
            full=OUT/'candidate_followup'/identifier/'full_sie/metrics.json'
            source=json.loads(full.read_text()) if full.exists() else record
            decision=summarize_subhalo(source)
            row.update(detected=decision['detected'],strength_arcsec=decision['strength_arcsec'],
                       delta_chi2=decision['delta_chi2'],candidate_rank=decision['candidate_rank'],
                       refinement_stage='full_sie' if full.exists() else 'batch_joint')
            if decision['detected']:row['x_arcsec'],row['y_arcsec']=decision['position_arcsec']
        rows.append(row)
    columns=['identifier','status','detected','x_arcsec','y_arcsec','strength_arcsec','delta_chi2',
             'candidate_rank','refinement_stage']
    destination=OUT/'subhalo_results.csv';temporary=destination.with_suffix('.csv.tmp')
    with temporary.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    temporary.replace(destination)
    write_json(OUT/'subhalo_results_policy.json',dict(
        rule='delta_chi2>=25 and positive penalized gain; greatest penalized gain among passing alternatives',
        stability_is_veto=False,significance_calibrated=False,
        full_sie_results_preferred_when_available=True,
        no_full_sie_available='batch_joint result is explicitly labelled; no new fitting is run by summarize',
        failed_or_missing_input='detected is empty, not False',
        position_units='arcsec in the observation coordinate frame',strength_units='arcsec'))
    return rows


def summarize(manifest,wall_seconds=None):
    records=[]
    for row in manifest:
        path=OUT/row['identifier']/'metrics.json'
        failure=OUT/row['identifier']/'failure.json'
        if path.exists():records.append(json.loads(path.read_text()))
        elif failure.exists():records.append(json.loads(failure.read_text()))
    completed=[r for r in records if r['status']=='completed']
    compact=write_compact_results(records)
    summary=dict(total=len(manifest),completed=len(completed),missing_input=sum(r['status']=='missing_input' for r in records),
        failed=sum(r['status']=='failed' for r in records),pending=len(manifest)-len(records),wall_seconds=wall_seconds,
        screening_candidates=sum(r['review_candidate_count']>0 for r in completed),
        fit_improvement_detections=sum(r['detected'] is True for r in compact),confirmed_subhalos=None,
        confirmation_not_calibrated=True,confirmation_status='not assessed by calibrated detection analysis')
    summary['source_similarity_undefined']=sum(r['source_similarity']['MACRO_ONLY']['cosine'] is None for r in completed)
    if completed:
        summary.update(median_seconds_per_completed_lens=float(np.median([r['seconds'] for r in completed])),
            median_macro_fractional_radius_difference=float(np.median([r['theta_macro']/r['theta_official_maximum_likelihood']-1 for r in completed if r['theta_official_maximum_likelihood']>0])))
    write_json(OUT/'summary.json',summary)
    columns=['identifier','inspection','status','seconds','theta_official_maximum_likelihood','theta_official_catalogue_median',
        'theta_image_initializer','theta_macro','theta_halo_host','initializer_converged','broad_macro_converged','macro_converged',
        'source_cosine_raw','source_cosine_smoothed','chi2_macro','chi2_halo','delta_chi2_best','review_candidate_count','pinn_coverage_pixels']
    with (OUT/'scan_results.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for r in records:
            flat={k:r.get(k) for k in columns}
            if r['status']=='completed':
                sm=r['source_similarity']['MACRO_ONLY'];flat.update(source_cosine_raw=sm['unsmoothed_cosine_same_support'],
                    source_cosine_smoothed=sm['cosine'],chi2_macro=r['image_scores']['macro']['chi2_per_pixel'],
                    chi2_halo=r['image_scores']['halo']['chi2_per_pixel'],delta_chi2_best=(r['best_candidate'] or {}).get('delta_chi2'))
            writer.writerow(flat)
    if not completed:return summary
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    for ax,field,title in zip(axes,['theta_image_initializer','theta_macro'],['Image-only Sersic initialization','Delaunay macro point estimate']):
        for success,color,label in [(True,'#0072B2','Official Success'),(False,'#D55E00','Other official inspection')]:
            group=[r for r in completed if (r['inspection']=='Success')==success and r['theta_official_maximum_likelihood']>0]
            ax.scatter([r['theta_official_maximum_likelihood'] for r in group],[r[field] for r in group],s=20,alpha=.65,color=color,label=f'{label} (n={len(group)})')
        maximum=max(max(r['theta_official_maximum_likelihood'],r[field]) for r in completed)*1.04
        ax.plot([0,maximum],[0,maximum],'k--',lw=1);ax.set(xlim=(0,maximum),ylim=(0,maximum),
            xlabel='Released effective Einstein radius [arcsec]',ylabel='SubLens effective Einstein radius [arcsec]',title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle(f"Part-1 Grade A · {len(completed)}/{len(manifest)} completed · published mass not used as initialization\nPoint estimates; fixed supplied foreground/PSF; inspection and fit failures retained in the table")
    fig.savefig(OUT/'einstein_radius_scatter.png',dpi=170);fig.savefig(OUT/'einstein_radius_scatter.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for success,color,label in [(True,'#0072B2','Official Success'),(False,'#D55E00','Other official inspection')]:
        values=[r['source_similarity']['MACRO_ONLY']['cosine'] for r in completed
            if (r['inspection']=='Success')==success and r['source_similarity']['MACRO_ONLY']['cosine'] is not None]
        ax.hist(values,bins=np.linspace(0,1,21),histtype='step',linewidth=2,color=color,label=f'{label} (n={len(values)})')
    ax.set(xlabel='Source cosine similarity (common 0.02 arcsec smoothing)',ylabel='Number of systems',
        title='Released source versus own Delaunay macro-only source\nComparison only: no source-based selection or fitted alignment')
    ax.legend();ax.grid(alpha=.15);fig.savefig(OUT/'source_similarity_hist.png',dpi=170);fig.savefig(OUT/'source_similarity_hist.pdf');plt.close(fig)
    ranked=sorted((r for r in completed if r['review_candidate_count']),key=lambda r:(r['best_candidate'] or {}).get('delta_chi2',0),reverse=True)
    write_json(OUT/'candidate_review.json',[dict(identifier=r['identifier'],inspection=r['inspection'],best_candidate=r['best_candidate'],
        review_candidates=[c for c in r['candidates'] if c['review_candidate']],
        chi2_macro=r['image_scores']['macro'],chi2_halo=r['image_scores']['halo'],macro_converged=r['macro_converged'],
        plot=str((OUT/r['identifier']/'dr1_nine_panel_comparison.png').relative_to(ROOT))) for r in ranked])
    return summary


def run_worker(arguments):
    row,backend,device,dtype,output=arguments
    global OUT
    OUT=Path(output)
    if (OUT/'STOP').exists():
        return dict(identifier=row['identifier'],status='paused_before_next_system')
    if backend=='torch':
        with torch_backend(device,dtype=dtype):return run_one(row)
    return run_one(row)


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','summarize']);p.add_argument('--workers',type=int)
    p.add_argument('--ids',nargs='*');p.add_argument('--limit',type=int)
    p.add_argument('--backend',choices=['reference','torch'],default='reference');p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=['float32','float64'])
    p.add_argument('--output',type=Path);args=p.parse_args()
    if args.output:OUT=args.output.resolve()
    elif args.backend=='torch':OUT=OUT.with_name(OUT.name+'_torch_'+args.device.replace(':','_'))
    if args.dtype and not args.output:OUT=OUT.with_name(OUT.name+'_'+args.dtype)
    if args.backend=='torch':
        from sublens.execution import resolve_device
        resolved=resolve_device(args.device)
        with torch_backend(args.device,dtype=args.dtype):pass
    else:resolved=torch.device('cpu')
    if args.workers is None:args.workers=1 if resolved.type!='cpu' else 3
    manifest=make_manifest() if args.stage=='prepare' else json.loads((OUT/'manifest.json').read_text())
    if args.stage=='run':
        saved=json.loads((OUT/'protocol.json').read_text())
        if saved.get('backend','reference')!=args.backend:
            raise ValueError('Output directory belongs to another backend; use a separate --output directory')
        if args.backend=='torch' and saved.get('requested_dtype')!=args.dtype:
            raise ValueError('Precision differs from the prepared output directory; use consistent --dtype or a separate --output')
    if args.stage=='prepare':
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(backend=args.backend,requested_device=args.device,requested_dtype=args.dtype)
        write_json(OUT/'protocol.json',protocol)
        print(f'Prepared {len(manifest)} Grade A systems',flush=True);return
    if args.stage=='summarize':print(summarize(manifest),flush=True);return
    selected=[r for r in manifest if not args.ids or r['identifier'] in args.ids]
    if args.limit:selected=selected[:args.limit]
    started=perf_counter();print(f'Running {len(selected)} Grade A systems, {args.workers} workers',flush=True)
    context=multiprocessing.get_context('spawn') if args.backend=='torch' else None
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context) as pool:
        futures={pool.submit(run_worker,(row,args.backend,args.device,args.dtype,str(OUT))):row for row in selected}
        for index,future in enumerate(as_completed(futures),1):
            try:future.result()
            except Exception:traceback.print_exc()
            status=summarize(manifest,perf_counter()-started)
            print(f'PROGRESS {index}/{len(selected)} '+json.dumps(status),flush=True)
    print(json.dumps(summarize(manifest,perf_counter()-started),indent=2),flush=True)

if __name__=='__main__':main()
