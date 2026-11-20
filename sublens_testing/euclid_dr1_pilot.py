"""One ECLIPSE DR1 lens: PSF checks, PINN screening and a nine-panel figure."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, SymLogNorm
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial import Delaunay
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from euclid_grade_a_10 import ProfiledMacroFit, effective_sie_radius, source_shift
from run import SEARCH, ADAPT, save_result, load_result
from strength_scan import write_json, digest
from sublens import LensingObservation
from sublens.grids import make_image_rays
from sublens.profiles.macro import epl_deflection
from sublens.survey.euclid_dr1 import load_euclid_dr1_lens
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.potential import _smooth_convergence

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dr1_catalogue_v5_full'
OUTPUT=ROOT/'sublens_testing/euclid_dr1_one_lens'
CATALOG=ROOT/'downloads/euclid_dr1/dr1_catalogue_15072_master.csv'


def select_lens():
    with CATALOG.open() as f:
        available=[r for r in csv.DictReader(f) if (DATA/r['tile_id']).is_dir()
                   and r['grade']=='A' and r['inspection_consensus_75']=='Success']
    available.sort(key=lambda r:(-float(r['score']),r['unique_identifier']))
    eligible=[r for r in available if .1<=float(r['source_light_effective_radius'] or 0)<=.6
              and .8<=float(r['mass_effective_einstein_radius'])<=2.5]
    return eligible[0], dict(local_successful_grade_a=len(available), resolved_eligible=len(eligible),
        rule='Grade A + official Success; published source Sersic Re 0.1–0.6 arcsec and effective Einstein radius 0.8–2.5 arcsec; highest score, then identifier',
        selection_uses_no_sublens_fit_or_candidate_score=True)


def deflection_check(directory, macro):
    with fits.open(directory/'deflections.fits') as h:
        ay=np.array(h['DEFLECTIONS_Y'].data); ax=np.array(h['DEFLECTIONS_X'].data); header=h[1].header
    x,y=make_image_rays(ax.shape,header['PIXSCAX'],1)
    x=x[...,0]+header.get('ORIGINX',0.)
    y=y[...,0]-header.get('ORIGINY',0.)
    coordinates=torch.tensor(np.column_stack((x.ravel(),y.ravel())),dtype=torch.float64)
    host,_=epl_deflection(coordinates,**{k:macro[k] for k in ('theta_E','q','phi','gamma','center')})
    g1=macro['gamma_ext']*np.cos(2*macro['phi_ext']);g2=macro['gamma_ext']*np.sin(2*macro['phi_ext'])
    # Published shear is about the coordinate origin; map y-components point south.
    pred=host.numpy()+np.column_stack((g1*x.ravel()+g2*y.ravel(),g2*x.ravel()-g1*y.ravel()))
    error=pred-np.column_stack((ax.ravel(),-ay.ravel()))
    maximum=float(np.max(np.abs(error)))
    if maximum>1e-8: raise RuntimeError(f'Released deflection/frame mismatch: {maximum}')
    return dict(maximum_component_error_arcsec=maximum, rms_vector_error_arcsec=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
        source='COOLEST maximum-likelihood model versus released deflections.fits',
        published_y_component_points_south=True, source_shear_origin_translation=source_shift(macro).tolist())


def half_max_width(values):
    i=int(np.argmax(values)); half=values[i]/2
    left=next((j for j in range(i-1,-1,-1) if values[j]<=half),None)
    right=next((j for j in range(i+1,len(values)) if values[j]<=half),None)
    if left is None or right is None:return None
    a=left+(half-values[left])/(values[left+1]-values[left])
    b=right-1+(half-values[right-1])/(values[right]-values[right-1])
    return float(b-a)


def psf_check(data, directory):
    with fits.open(directory/'dataset'/f'{directory.name}.fits') as h: raw=np.array(h['VIS_PSF'].data,dtype=float)
    p=data.observation.psf; yy,xx=np.indices(p.shape); positive=np.maximum(p,0.);positive/=positive.sum()
    cy,cx=(float((a*positive).sum()) for a in (yy,xx))
    radius=np.hypot(xx-cx,yy-cy);order=np.argsort(radius.ravel()); cumulative=np.cumsum(positive.ravel()[order])
    quantiles={str(q):float(radius.ravel()[order[np.searchsorted(cumulative,q)]]) for q in (.5,.8,.95,.98)}
    iy,ix=np.unravel_index(np.argmax(p),p.shape)
    negatives=float(-raw[raw<0].sum()/np.maximum(raw,0).sum())
    checks=[]; interior=data.observation.mask & (distance_transform_edt(data.observation.mask)>max(p.shape)//2)
    if not interior.any():interior=data.observation.mask
    with fits.open(directory/'pre_psf.fits') as h:
        for j,model in ((0,data.lens_light),(1,data.official_lensed_source)):
            native=np.array(h[f'VIS_GALAXY_{j}'].data,dtype=float)
            predicted=fftconvolve(native,p,mode='same')
            flipped=fftconvolve(native,np.flipud(p),mode='same')
            mask=interior
            checks.append(dict(galaxy=j,rel_l2_on_interior=float(np.linalg.norm((predicted-model)[mask])/np.linalg.norm(model[mask])),
                discrepancy_rms_in_noise_units=float(np.sqrt(np.mean(((predicted-model)[mask]/data.observation.rms[mask])**2))),
                vertically_flipped_psf_rel_l2=float(np.linalg.norm((flipped-model)[mask])/np.linalg.norm(model[mask]))))
    center_offset=float(np.hypot(cx-p.shape[1]//2,cy-p.shape[0]//2))
    return dict(shape=list(p.shape),raw_sum=float(raw.sum()),normalized_sum=float(p.sum()),
        raw_minimum=float(raw.min()),negative_absolute_flux_fraction=negatives,
        centroid_offset_xy_pixels=[cx-p.shape[1]//2,cy-p.shape[0]//2],
        half_max_crosscut_fwhm_arcsec=[half_max_width(p[iy,:])*data.observation.pixel_scale,
                                     half_max_width(p[:,ix])*data.observation.pixel_scale],
        enclosed_positive_flux_radius_pixels=quantiles,
        outer_two_pixel_border_flux_fraction=float(positive[(xx<2)|(yy<2)|(xx>=p.shape[1]-2)|(yy>=p.shape[0]-2)].sum()),
        numerically_usable=bool(np.isfinite(p).all() and center_offset<.5 and negatives<1e-3),
        supplied_psf_preserved_without_recentering_or_clipping=True,
        official_pre_psf_to_model_checks=checks,independent_stellar_psf_calibration=False,
        substructure_precision_validated=False,
        caveat='Numerical kernel/model consistency alone cannot establish true PSF accuracy; DR1 MER PSF lacks the more detailed colour-dependent SHE treatment')


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    row,rule=select_lens(); directory=DATA/row['tile_id']
    data=load_euclid_dr1_lens(directory,oversample=4)
    write_json(OUTPUT/'selection.json',dict(lens=row,**rule))
    write_json(OUTPUT/'frame_verification.json',deflection_check(directory,data.observation.macro_model))
    write_json(OUTPUT/'psf_diagnostics.json',psf_check(data,directory))
    if not (OUTPUT/'observation.fits').exists():data.observation.to_fits(OUTPUT/'observation.fits')
    problem=PixelSourceProblem(data.observation.image,data.observation.rms,data.observation.psf,data.observation.mask,
        data.observation.bx,data.observation.by,source_grid=Delaunay(data.source_points),regularization=1.)
    shift=source_shift(data.observation.macro_model)
    reference=problem.render(data.observation.bx-shift[0],data.observation.by-shift[1],data.source_values)
    ref=fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(data.raw_image,name='RAW_DATA'),
        fits.ImageHDU(data.lens_light,name='LENS_LIGHT'),
        fits.ImageHDU(data.official_lensed_source,name='RELEASED_SERSIC_SOURCE'),
        fits.ImageHDU(data.raw_image-data.lens_light-data.official_lensed_source,name='SERSIC_RESIDUAL'),
        fits.ImageHDU(reference,name='REFERENCE_SOURCE'),
        fits.ImageHDU(data.observation.image-reference,name='REFERENCE_RESIDUAL')])
    ref[0].header['REFSTAGE']='vis_pix re-rendered'
    ref[0].header['FGSTAGE']='public Sersic deblend'
    ref.writeto(OUTPUT/'official_reference.fits',overwrite=True,checksum=True)
    files=[CATALOG,directory/'dataset'/f'{directory.name}.fits',directory/'model.fits',directory/'pre_psf.fits',
        directory/'deflections.fits',directory/'coolest.json',directory/'coolest_sersic.json',directory/'dataset/info.json',
        directory/'dataset/positions.json',directory/'dataset/segmentation/artefact_binary.fits',
        directory/'vis_pix_source_plane_reconstruction_0.csv']
    write_json(OUTPUT/'input_sha256.json',{str(p.relative_to(ROOT)):digest(p) for p in files if p.exists()})
    write_json(OUTPUT/'protocol.json',dict(band='VIS',oversample=4,science_mask='released aperture with reflected center; artefact mask and valid finite RMS',
        foreground='released VIS_GALAXY_0 held fixed',macro_starts=4,macro_free='SIE theta_E,q,phi,center_x,center_y,shear_1,shear_2; slope=2',
        later_macro_free='theta_E and both shear components; shape and centre held at broad macro fit',
        search=SEARCH,adaptive=ADAPT,psf_guard='default 98 percent absolute PSF energy, no cap',
        physical_profile=dict(core_arcsec=.035,truncation_arcsec=.22),
        observation_was_not_injected_or_modified_with_a_synthetic_halo=True,
        hypotheses='independently refined single-halo alternatives; not simultaneous detections',
        conjugate_positions_used_for='loaded for provenance only; not imposed as new fitting priors',
        publication='https://zenodo.org/records/19654904',psf_guidance='https://arxiv.org/html/2611.15888v1#S2'))
    write_json(OUTPUT/'reference_stages.json',dict(
        released_model_fits='Sersic photometric deblending, not the vis_pix Delaunay result',
        primary_reference='public vis_pix COOLEST macro + source-node CSV re-rendered with the same 4x4 integration and normalized supplied PSF',
        foreground_for_both_reference_and_our_fit='released model.fits VIS_GALAXY_0 held fixed',
        original_vis_pix_foreground_amplitudes_not_available_in_this_bundle=True,
        primary_reference_is_not_claimed_to_be_the_original_official_pixelized_fit_residual=True,
        producer='https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/blob/main/catalogue/scripts/deblending.py'))
    if Path('/tmp/sublens_dr1_resources.json').exists():shutil.copyfile('/tmp/sublens_dr1_resources.json',OUTPUT/'resources.json')
    return row,data


def source_maps(data,result):
    points,values=data.source_points,data.source_values
    bright=points[values>=.05*values.max()]
    low,high=bright.min(axis=0),bright.max(axis=0)
    center=(low+high)/2;half=max(.3,float(np.max(high-low))*.6)
    axis=np.linspace(-half,half,301);xx,yy=np.meshgrid(axis+center[0],axis+center[1])
    reference=LinearNDInterpolator(points,values,fill_value=0.)(xx,yy)
    best=result.best_candidate
    halo_source=result.baseline if best is None else best.fit.source
    halo_macro=result.macro_model if best is None else best.fit.macro_model
    arrays={'OFFICIAL':reference}
    for key,source,macro in [('MACRO_ONLY',result.baseline,result.macro_model),('MACRO_HALO',halo_source,halo_macro)]:
        shift=source_shift(macro);arrays[key]=source.sample(xx+shift[0],yy+shift[1])
    header=fits.Header()
    for j in (1,2):
        header[f'CTYPE{j}']='LINEAR';header[f'CUNIT{j}']='arcsec';header[f'CRPIX{j}']=151.
        header[f'CRVAL{j}']=center[j-1];header[f'CDELT{j}']=2*half/300
    fits.HDUList([fits.PrimaryHDU()]+[fits.ImageHDU(a,header=header,name=k) for k,a in arrays.items()]).writeto(
        OUTPUT/'source_comparison.fits',overwrite=True,checksum=True)
    smooth=.02/(2*half/300);ref=gaussian_filter(reference,smooth);support=ref>=.05*ref.max()
    scores={}
    for name in ('MACRO_ONLY','MACRO_HALO'):
        a=gaussian_filter(arrays[name],smooth);u,v=ref[support],a[support]
        scores[name]=dict(cosine_on_reference_support=float(u@v/(np.linalg.norm(u)*np.linalg.norm(v))),
            smoothing_arcsec=.02,no_fitted_registration_or_flux_rescaling=True)
    extent=[center[0]-half,center[0]+half,center[1]-half,center[1]+half]
    return arrays,extent,scores


def finalize(row,data,result,seconds):
    arrays,extent,similarities=source_maps(data,result)
    obs=data.observation;best=result.best_candidate
    source=result.baseline if best is None else best.fit.source
    halo_macro=result.macro_model if best is None else best.fit.macro_model
    reference=fits.getdata(OUTPUT/'official_reference.fits','REFERENCE_SOURCE')
    def score(model):
        r=(obs.image-model)/obs.rms;mask=obs.mask;bright=mask&(reference>=3*obs.rms)
        return dict(chi2_per_pixel=float(np.mean(r[mask]**2)),pixels=int(mask.sum()),
            bright_arc_chi2_per_pixel=float(np.mean(r[bright]**2)),bright_pixels=int(bright.sum()))
    metrics=dict(identifier=row['unique_identifier'],tile_id=row['tile_id'],seconds=seconds,
        runtime_note='Full-run wall time not recorded; summary recovered from saved inference' if seconds is None else 'Full inference and initial output time',
        theta_official_maximum_likelihood=effective_sie_radius(obs.macro_model),
        theta_official_catalogue_median=float(row['mass_effective_einstein_radius']),
        theta_macro=effective_sie_radius(result.macro_model),theta_halo_host=effective_sie_radius(halo_macro),
        image_scores=dict(official=score(reference),released_sersic=score(data.official_lensed_source),
                          macro=score(result.baseline.image_model),halo=score(source.image_model)),
        source_similarity=similarities,candidate_count=len(result.candidates),pinn_coverage_pixels=int(result.residual_potential.mask.sum()),
        macro_converged=result.macro_converged,subhalo_existence_confirmed=None,
        best_candidate=None)
    if best:
        f=best.fit;pixel=np.array([[f.x/obs.pixel_scale+(obs.image.shape[1]-1)/2,f.y/obs.pixel_scale+(obs.image.shape[0]-1)/2]])
        ra,dec=WCS(data.header).all_pix2world(pixel,0)[0]
        metrics['best_candidate']=dict(position_xy_arcsec=list(f.position),position_frame='+x west, +y north from cutout centre',
            ra_deg=float(ra),dec_deg=float(dec),strength_arcsec=f.strength,converged=f.converged,
            delta_chi2=f.delta_chi2,penalized_gain=f.delta_objective,
            strength_at_upper_bound=bool(f.strength>=.98*result.max_strength),
            position_at_local_bound=bool(np.max(np.abs(np.asarray(f.position)-best.seed))>=.97*result.refinement_radius))
    write_json(OUTPUT/'metrics.json',metrics)
    plot_comparison(row,data,result,arrays,extent,metrics)
    psf=json.loads((OUTPUT/'psf_diagnostics.json').read_text())
    lines=[f"# {row['unique_identifier']}: one DR1 lens\n",
        f"{row['tile_id']}; Grade {row['grade']}; official inspection {row['inspection_consensus_75']}.\n",
        'The nine-panel comparison uses the same released data, mask, RMS and fixed official foreground model. '
        'All residuals are observed minus model, divided by the original RMS. The last residual adds the optimized '
        'halo to the fitted model; no halo is injected into the observation. Candidate ranks are competing single-halo alternatives.\n',
        '**Reference-stage distinction:** public model.fits and pre_psf.fits are from the Sérsic photometric deblending stage. '
        'Their residual is retained as SERSIC_RESIDUAL in official_reference.fits. Panel 3 instead re-renders the released '
        'vis_pix source CSV and COOLEST mass model with our common 4×4 integration/PSF, holding the same public Sérsic '
        'foreground fixed for the reference and our fit. It is explicitly a re-rendered reference, not the original '
        'official pixelized residual: the original pixelized foreground amplitudes are unavailable in this bundle. '
        'Panel 5 displays the corresponding released Delaunay source.\n',
        f"PSF numerical usability: {psf['numerically_usable']}; crosscut FWHM {psf['half_max_crosscut_fwhm_arcsec']} arcsec. "
        'The supplied kernel is normalized once, with its centroid and small negative interpolation lobes retained. '
        'Kernel consistency does not independently establish its fidelity for dark-matter substructure measurements.\n',
        f"Effective Einstein radius: official maximum-likelihood {metrics['theta_official_maximum_likelihood']:.5f} arcsec; "
        f"catalogue posterior median {metrics['theta_official_catalogue_median']:.5f}; our macro {metrics['theta_macro']:.5f}.\n",
        f"Full-mask chi2/pixel: macro {metrics['image_scores']['macro']['chi2_per_pixel']:.3f} → "
        f"halo {metrics['image_scores']['halo']['chi2_per_pixel']:.3f}. On the fixed reference bright-arc pixels "
        f"(reference source ≥3 RMS), the halo fit is {metrics['image_scores']['halo']['bright_arc_chi2_per_pixel']:.3f} "
        f"versus {metrics['image_scores']['official']['bright_arc_chi2_per_pixel']:.3f} for the re-rendered reference. "
        "The lower full-mask chi2 does not imply a better reconstruction of the arcs. Our source is noticeably smoother, "
        "and structured residuals remain. The candidate gain is not a calibrated substructure detection statistic.\n",
        '```json\n'+json.dumps(metrics,indent=2)+'\n```\n',
        'The broad seven-parameter macro fit has four starts, including the released solution. Later joint refinement '
        'frees theta_E and shear, with q, position angle and centre fixed at the broad macro fit. Sources are reprofiled '
        'and their grids/regularization selected with the established conditional validation workflow. Published conjugate '
        'positions are not imposed as new priors. This is not a posterior uncertainty or subhalo significance analysis.\n',
        'Source CSV y is mirrored from the pipeline frame into the sky frame. The known shear-origin translation '
        'is applied to our source sampling. Both source panels use the same coordinates, limits and colour scale; '
        'there is no fitted registration. Official source nodes are rendered with Delaunay linear interpolation.\n',
        'Run `.venv/bin/python sublens_testing/euclid_dr1_pilot.py`; `--plot-only` redraws cached results.\n',
        '[ECLIPSE release](https://zenodo.org/records/19654904); '
        '[PSF limitations, modelling paper §2](https://arxiv.org/html/2611.15888v1#S2).\n']
    (OUTPUT/'README.md').write_text('\n'.join(lines))
    return metrics


def plot_comparison(row,data,result,sources,source_extent,metrics):
    obs=data.observation;mask=obs.mask; best=result.best_candidate
    fig,axes=plt.subplots(3,3,figsize=(16,15),layout='constrained');axes=axes.ravel()
    image_extent=np.array([-obs.image.shape[1]/2,obs.image.shape[1]/2,-obs.image.shape[0]/2,obs.image.shape[0]/2])*obs.pixel_scale
    limit=min(min(obs.image.shape)*obs.pixel_scale/2,max(2.5,1.7*metrics['theta_official_maximum_likelihood']))
    flux_max=max(float(np.percentile(obs.image[mask],99.5)),float(np.median(obs.rms[mask])))
    norm=AsinhNorm(linear_width=flux_max*.03,vmin=0,vmax=flux_max)
    def image_panel(ax,values,title,*,cmap='magma',norm=None,vmin=None,vmax=None):
        a=ax.imshow(np.where(mask,values,np.nan),origin='lower',extent=image_extent,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
        ax.set(title=title,xlim=(-limit,limit),ylim=(-limit,limit),xlabel='x west [arcsec]',ylabel='y north [arcsec]')
        return a
    a=image_panel(axes[0],obs.image,'1. Masked VIS data − lens light',norm=norm);fig.colorbar(a,ax=axes[0],shrink=.76,label='ADU/s')
    ps=obs.psf;half=np.array([-ps.shape[1]/2,ps.shape[1]/2,-ps.shape[0]/2,ps.shape[0]/2])*obs.pixel_scale
    a=axes[1].imshow(ps,origin='lower',extent=half,cmap='magma',norm=SymLogNorm(linthresh=1e-6,vmin=-1e-6,vmax=ps.max()))
    pd=json.loads((OUTPUT/'psf_diagnostics.json').read_text());fw=pd['half_max_crosscut_fwhm_arcsec']
    axes[1].set(title=f'2. Supplied PSF; FWHM x/y ≈ {fw[0]:.3f}/{fw[1]:.3f}″',xlabel='x [arcsec]',ylabel='y [arcsec]');fig.colorbar(a,ax=axes[1],shrink=.76,label='Normalized kernel')
    reference=fits.getdata(OUTPUT/'official_reference.fits','REFERENCE_SOURCE')
    residuals=[(2,obs.image-reference,'3. Published Delaunay reference, re-rendered','official'),
               (3,result.residual_macro,'4. Our macro-only residual','macro'),
               (8,obs.image-(result.baseline if best is None else best.fit.source).image_model,'9. Residual after adding fitted halo','halo')]
    for index,residual,title,key in residuals:
        chi=metrics['image_scores'][key]['chi2_per_pixel']
        a=image_panel(axes[index],residual/obs.rms,title+f'\nχ²/Npix = {chi:.3f}',cmap='RdBu_r',vmin=-5,vmax=5)
        fig.colorbar(a,ax=axes[index],shrink=.76,label='(Data − model) / RMS')
    vmax=max(1e-15,max(np.percentile(sources[k],99.8) for k in ('OFFICIAL','MACRO_HALO')))
    snorm=AsinhNorm(linear_width=max(vmax*.03,1e-15),vmin=0,vmax=vmax)
    for index,key,title in [(4,'OFFICIAL','5. Their source plane'),(5,'MACRO_HALO','6. Our source plane, macro + halo')]:
        a=axes[index].imshow(sources[key],origin='lower',extent=source_extent,cmap='magma',norm=snorm)
        axes[index].set(title=title,xlabel='βx west [arcsec]',ylabel='βy north [arcsec]');fig.colorbar(a,ax=axes[index],shrink=.76,label='Source intensity')
    kappa=_smooth_convergence(result.residual_potential.convergence,obs.pixel_scale)
    coverage=result.residual_potential.mask; scale=max(float(np.percentile(abs(kappa[coverage]),99)),1e-4) if coverage.any() else 1.
    a=image_panel(axes[6],np.where(coverage,kappa,np.nan),'7. PINN δκ and candidate seeds',cmap='RdBu_r',vmin=-scale,vmax=scale)
    for candidate in result.candidates:
        x,y=candidate.seed;axes[6].plot(x,y,'o',mfc='none',mec='black',ms=9);axes[6].annotate(str(candidate.rank),(x,y),xytext=(5,5),textcoords='offset points',color='black')
    fig.colorbar(a,ax=axes[6],shrink=.76,label='Smoothed δκ; proposal smoothing')
    a=image_panel(axes[7],obs.image,'8. Data − lens light; finalized candidate',norm=norm);fig.colorbar(a,ax=axes[7],shrink=.76,label='ADU/s')
    if best:
        f=best.fit
        for index in (6,7,8):
            axes[index].plot(f.x,f.y,'*',mec='black',mfc='#ffe600',ms=16,label='Final best candidate')
        axes[7].text(.03,.03,f'x={f.x:+.3f}″, y={f.y:+.3f}″\nb={f.strength:.4f}″, Δχ²={f.delta_chi2:.1f}',
            transform=axes[7].transAxes,color='white',fontsize=10,bbox=dict(facecolor='black',alpha=.65))
        axes[7].legend(loc='upper right',fontsize=9)
    else: axes[7].text(.05,.05,'No PINN proposal',transform=axes[7].transAxes,color='white')
    fig.suptitle(f"{row['unique_identifier']} · Grade {row['grade']} / official {row.get('inspection_consensus_75', 'unavailable')}\n"
        f"Effective θE: official {metrics['theta_official_maximum_likelihood']:.3f}″, ours {metrics['theta_macro']:.3f}″ · "
        f"full released mask: {int(mask.sum())} pixels · common fixed foreground · candidates uncalibrated",fontsize=17)
    fig.savefig(OUTPUT/'dr1_nine_panel_comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUTPUT/'dr1_nine_panel_comparison.pdf',bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);started=perf_counter();row,data=prepare();obs=data.observation
    print(f"Selected {row['unique_identifier']}; mask {obs.mask.sum()} pixels; 4×4 rays",flush=True)
    if args.plot_only:
        result=load_result(OUTPUT/'result.fits')
        seconds=json.loads((OUTPUT/'metrics.json').read_text())['seconds'] if (OUTPUT/'metrics.json').exists() else None
    else:
        if (OUTPUT/'result.fits').exists():
            result=load_result(OUTPUT/'result.fits')
            seconds=json.loads((OUTPUT/'metrics.json').read_text())['seconds'] if (OUTPUT/'metrics.json').exists() else None
        else:
            if (OUTPUT/'macro_fit.json').exists():record=json.loads((OUTPUT/'macro_fit.json').read_text());macro=record['macro']
            else:
                fit=ProfiledMacroFit(obs).fit();macro=fit.macro
                record=dict(macro=macro,converged=fit.converged,attempts=fit.attempts,objective=fit.source.objective)
                write_json(OUTPUT/'macro_fit.json',record)
            print(f"Broad macro effective theta_E={effective_sie_radius(macro):.5f}, converged={record['converged']}",flush=True)
            fitted=LensingObservation.from_arrays(obs.image,obs.rms,obs.psf,pixel_scale=obs.pixel_scale,
                macro_model=macro,mask=obs.mask,native_rays=obs.image_rays(),lens_id=obs.lens_id)
            if not (OUTPUT/'fitted_macro_observation.fits').exists():fitted.to_fits(OUTPUT/'fitted_macro_observation.fits')
            if (OUTPUT/'initial_joint.fits').exists():result=load_result(OUTPUT/'initial_joint.fits')
            else:
                result=fitted.search_pinn(**SEARCH);result.save_json(OUTPUT/'initial_search.json',overwrite=True)
                print(f'PINN proposed {len(result.candidates)} alternatives',flush=True)
                result=result.refine_joint(macro,maxiter=120,n_starts=2,retry_failed=True)
                result.save_fits(OUTPUT/'initial_joint.fits',overwrite=True)
            if result.candidates:result=result.refine_adaptive(macro,**ADAPT,verbose=True)
            result=result.merge_candidates(radius=obs.pixel_scale);save_result(result,OUTPUT)
            seconds=perf_counter()-started
    metrics=finalize(row,data,result,seconds)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
