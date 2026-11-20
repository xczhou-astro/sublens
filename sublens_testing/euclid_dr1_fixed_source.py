"""Source-only consistency check at the released DR1 mass and foreground light."""

import csv
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from euclid_dr1_pilot import DATA,OUTPUT as PILOT,source_shift,deflection_check
from strength_scan import write_json,digest
from sublens.survey.euclid_dr1 import load_euclid_dr1_lens
from sublens.inversion.source_plane import PixelSource,PixelSourceProblem
from sublens.io import observation_hdus
from sublens import LensingObservation

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=PILOT/'fixed_official_macro_source'
REGULARIZATIONS=(2e-6,2e-5,2e-4,2e-3,2e-2,.2,2.,20.)


def image_scores(obs,prediction,bright,validation=None):
    r=(obs.image-prediction)/obs.rms
    d=dict(chi2_per_pixel=float(np.mean(r[obs.mask]**2)),
           bright_arc_chi2_per_pixel=float(np.mean(r[bright]**2)))
    if validation is not None:d['validation_chi2_per_pixel']=float(np.mean(r[validation]**2))
    return d


def source_scores(reference,recovered,pixel_scale):
    a,b=[gaussian_filter(v,.02/pixel_scale) for v in (reference,recovered)]
    if (not np.isfinite(a).all() or not np.isfinite(b).all() or a.max()<=0):
        return dict(cosine=None,unsmoothed_cosine_same_support=None,
            unsmoothed_relative_l2_same_support=None,pearson=None,source_relative_l2=None,
            smoothed_peak_offset_mas=None,common_smoothing_arcsec=.02,
            no_fitted_alignment_or_flux_rescaling=True,comparison_defined=False,
            comparison_note='Published source absent/nonpositive or nonfinite comparison arrays')
    support=a>=.05*a.max();u,v=a[support],b[support]
    raw_u,raw_v=reference[support],recovered[support]
    def peak(arr):
        y,x=np.unravel_index(np.argmax(arr),arr.shape);return np.array([x,y])
    norm=np.linalg.norm(u)*np.linalg.norm(v)
    raw_norm=np.linalg.norm(raw_u)*np.linalg.norm(raw_v)
    return dict(cosine=float(u@v/norm) if norm else 0.,
        unsmoothed_cosine_same_support=float(raw_u@raw_v/raw_norm) if raw_norm else 0.,
        unsmoothed_relative_l2_same_support=float(np.linalg.norm(raw_v-raw_u)/np.linalg.norm(raw_u)),
        pearson=float(np.corrcoef(u,v)[0,1]) if u.std()>0 and v.std()>0 else None,
        source_relative_l2=float(np.linalg.norm(v-u)/np.linalg.norm(u)),
        smoothed_peak_offset_mas=float(1000*pixel_scale*np.linalg.norm(peak(a)-peak(b))) if b.max()>0 else None,
        common_smoothing_arcsec=.02,no_fitted_alignment_or_flux_rescaling=True,
        comparison_defined=True,zero_reconstructed_source=bool(b.max()<=0))


def save_fit(name,obs,source,raster,header,metadata):
    h=observation_hdus(obs)
    h[0].header['MASSFIX']=True;h[0].header['FGFIX']=True;h[0].header['HALOFIT']=False
    h[0].header['SRCREG']=metadata['regularization']
    h[0].header['REGKNOWN']=metadata['regularization'] is not None
    h.append(fits.ImageHDU(source.image_model,name='MODEL'))
    h.append(fits.ImageHDU(obs.image-source.image_model,name='RESIDUAL'))
    h.append(fits.ImageHDU(raster,header=header.copy(),name='SOURCE'))
    nodes=[fits.Column(name='X',format='D',array=source.grid.points[:,0]),
           fits.Column(name='Y',format='D',array=source.grid.points[:,1]),
           fits.Column(name='FLUX',format='D',array=source.coefficients)]
    h.append(fits.BinTableHDU.from_columns(nodes,name='SOURCE_NODES'))
    payload=json.dumps(metadata,allow_nan=False)
    h.append(fits.BinTableHDU.from_columns([fits.Column(name='JSON',format=f'{len(payload)}A',array=[payload])],name='METADATA'))
    h.writeto(OUTPUT/f'{name}.fits',overwrite=True,checksum=True)


def plot_comparison(products,obs,source_extent,output):
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    maximum=max(np.percentile(p['raster'],99.8) for p in products)
    norm=AsinhNorm(linear_width=.03*maximum,vmin=0,vmax=maximum)
    image_extent=np.array([-obs.image.shape[1]/2,obs.image.shape[1]/2,-obs.image.shape[0]/2,obs.image.shape[0]/2])*obs.pixel_scale
    for j,p in enumerate(products):
        title=p['title']
        if j:title+=f"\nλ={p['regularization']:g}; cosine raw / 0.02″: "\
                    f"{p['source_metrics']['unsmoothed_cosine_same_support']:.3f} / {p['source_metrics']['cosine']:.3f}"
        artist=axes[0,j].imshow(p['raster'],origin='lower',extent=source_extent,cmap='magma',norm=norm)
        axes[0,j].set(title=title,xlabel='βx west [arcsec]',ylabel='βy north [arcsec]')
        fig.colorbar(artist,ax=axes[0,j],shrink=.75,label='Source intensity')
        residual=(obs.image-p['source'].image_model)/obs.rms
        artist=axes[1,j].imshow(np.where(obs.mask,residual,np.nan),origin='lower',extent=image_extent,cmap='RdBu_r',vmin=-5,vmax=5)
        s=p['image_metrics']
        axes[1,j].set(title=f"Residual: χ²/Npix={s['chi2_per_pixel']:.3f}\nBright arcs: {s['bright_arc_chi2_per_pixel']:.3f}",
            xlim=(-2.5,2.5),ylim=(-2.5,2.5),xlabel='x west [arcsec]',ylabel='y north [arcsec]')
        fig.colorbar(artist,ax=axes[1,j],shrink=.75,label='(Observed − model) / RMS')
    fig.suptitle('EUCL-DR1-L020080 · fixed released mass, foreground light, PSF and mask\n'
                 'Only nonnegative source brightness is fitted; no subhalo',fontsize=16)
    fig.savefig(output,dpi=175,bbox_inches='tight');plt.close(fig)


def verify_outputs(obs,data,products,validation):
    hashes={}
    for product in products:
        file=OUTPUT/f"{product['name']}.fits"
        saved=LensingObservation.from_fits(file)
        # FITS decimal cards can round the last bits of centre coordinates.
        for key,value in obs.macro_model.items():
            np.testing.assert_allclose(saved.macro_model[key],value,rtol=0,atol=1e-14)
        for key in ('image','rms','mask','bx','by'):
            np.testing.assert_array_equal(getattr(saved,key),getattr(obs,key))
        np.testing.assert_allclose(saved.psf,obs.psf,atol=1e-15)
        source=product['source']
        problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,
            source_grid=source.grid,regularization=max(source.regularization,1e-10))
        predicted=problem.render(obs.bx,obs.by,source.coefficients)
        np.testing.assert_allclose(predicted,source.image_model,atol=1e-12)
        with fits.open(file) as h:
            assert h[0].header['MASSFIX'] and h[0].header['FGFIX'] and not h[0].header['HALOFIT']
            np.testing.assert_allclose(h['SCI'].data-h['MODEL'].data,h['RESIDUAL'].data,atol=1e-14)
            np.testing.assert_array_equal(h['SOURCE'].data,product['raster'])
        selected=product.get('conditional_validation')
        if selected is not None:
            grid=Delaunay(data.source_points+source_shift(obs.macro_model)) if product['name']=='released_mesh_cv' else None
            train=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask&~validation,obs.bx,obs.by,
                source_grid=grid,source_nodes=500,seed=22,mesh_strategy='centroidal',regularization=selected['regularization'])
            fit=train.fit(obs.bx,obs.by)
            score=np.mean(((obs.image-fit.image_model)[validation]/obs.rms[validation])**2)
            np.testing.assert_allclose(score,selected['validation_chi2_per_pixel'],atol=1e-10)
    for file in OUTPUT.glob('*.fits'):
        with fits.open(file,checksum=True) as h:
            h.verify('exception')
            assert all(x.verify_checksum()==1 and x.verify_datasum()==1 for x in h),file
        hashes[file.name]=digest(file)
    prior=json.loads((PILOT/'verification.json').read_text())['fits_sha256']
    for name,expected in prior.items():assert digest(PILOT/name)==expected,name
    for name,expected in json.loads((OUTPUT/'input_sha256.json').read_text()).items():
        assert digest(ROOT/name)==expected,name
    write_json(OUTPUT/'verification.json',dict(all_checks_passed=True,fits_verified=len(hashes),
        released_mass_and_psf_and_foreground_and_mask_preserved=True,
        all_source_forward_models_and_residuals_verified=True,
        conditional_validation_choices_reproduced=True,
        earlier_pilot_fits_unchanged=len(prior),fits_sha256=hashes,
        analysis_source_sha256=digest(Path(__file__))))
    print(f'Verified {len(hashes)} FITS products; official mass fixed; earlier {len(prior)} pilot FITS unchanged.',flush=True)


def main():
    torch.set_num_threads(1);started=perf_counter();OUTPUT.mkdir(exist_ok=True)
    row=json.loads((PILOT/'selection.json').read_text())['lens'];directory=DATA/row['tile_id']
    data=load_euclid_dr1_lens(directory,oversample=4);obs=data.observation
    write_json(OUTPUT/'fixed_macro.json',obs.macro_model)
    write_json(OUTPUT/'frame_verification.json',deflection_check(directory,obs.macro_model))
    shift=source_shift(obs.macro_model)
    # The only coordinate change is the analytic shear-origin gauge translation.
    released_grid=Delaunay(data.source_points+shift)
    p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_grid=released_grid,regularization=2.)
    reference_image=p.render(obs.bx,obs.by,data.source_values)
    with fits.open(PILOT/'source_comparison.fits') as h:header=h[1].header.copy()
    n=301
    x=(np.arange(n)+1-header['CRPIX1'])*header['CDELT1']+header['CRVAL1']
    y=(np.arange(n)+1-header['CRPIX2'])*header['CDELT2']+header['CRVAL2']
    xx,yy=np.meshgrid(x,y)
    reference=LinearNDInterpolator(data.source_points,data.source_values,fill_value=0.)(xx,yy)
    extent=[x[0],x[-1],y[0],y[-1]]
    bright=obs.mask&(reference_image>=3*obs.rms)
    rng=np.random.default_rng(20261119);indices=np.flatnonzero(obs.mask)
    validation=np.zeros(obs.mask.shape,dtype=bool)
    validation.ravel()[rng.choice(indices,int(.2*len(indices)),replace=False)]=True
    train=obs.mask&~validation
    fits.ImageHDU(validation.astype(np.uint8),name='VALIDATION').writeto(OUTPUT/'validation_mask.fits',overwrite=True,checksum=True)
    reference_source=PixelSource(data.source_values,reference_image,'delaunay',released_grid,0.,
        image_scores(obs,reference_image,bright)['chi2_per_pixel'])
    products=[dict(name='released_reference',title='Released Delaunay source\nRe-rendered reference',source=reference_source,
        raster=reference,regularization=None,image_metrics=image_scores(obs,reference_image,bright),
        source_metrics=source_scores(reference,reference,header['CDELT1']))]
    regularization_rows=[];cv=[];arrays={}
    for basis in ('our_centroidal','released_mesh'):
        options=dict(source_grid=released_grid) if basis=='released_mesh' else dict(source_nodes=500,mesh_strategy='centroidal',seed=22)
        training=PixelSourceProblem(obs.image,obs.rms,obs.psf,train,obs.bx,obs.by,regularization=2.,**options)
        full=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,regularization=2.,**options)
        trials=[]
        for reg in REGULARIZATIONS:
            fit=training.with_regularization(reg).fit(obs.bx,obs.by)
            metrics=image_scores(obs,fit.image_model,bright,validation)
            trials.append(dict(basis=basis,regularization=reg,validation_chi2_per_pixel=metrics['validation_chi2_per_pixel']))
        selected=min(trials,key=lambda t:t['validation_chi2_per_pixel'])
        selected=dict(selected,at_lowest_tested_regularization=selected['regularization']==min(REGULARIZATIONS))
        cv.extend(trials)
        # Full-data fits are evaluated only after the held-out choice is frozen.
        full_fits={}
        coefficients=[];images=[]
        for reg in REGULARIZATIONS:
            fit=full.with_regularization(reg).fit(obs.bx,obs.by)
            raster=fit.sample(xx+shift[0],yy+shift[1])
            sm=source_scores(reference,raster,header['CDELT1']);im=image_scores(obs,fit.image_model,bright)
            regularization_rows.append(dict(basis=basis,regularization=reg,
                validation_chi2_per_pixel=next(t['validation_chi2_per_pixel'] for t in trials if t['regularization']==reg),
                cosine=sm['cosine'],unsmoothed_cosine=sm['unsmoothed_cosine_same_support'],
                bright_arc_chi2_per_pixel=im['bright_arc_chi2_per_pixel'],chi2_per_pixel=im['chi2_per_pixel'],
                selected_by_validation=reg==selected['regularization']))
            full_fits[reg]=(fit,raster,sm,im)
            coefficients.append(fit.coefficients);images.append(fit.image_model)
        arrays[f'{basis}_points']=full.grid.points
        arrays[f'{basis}_coefficients']=np.array(coefficients)
        arrays[f'{basis}_models']=np.array(images)
        if basis=='our_centroidal':
            fit,raster,sm,im=full_fits[2.]
            products.append(dict(name='our_default_source',title='Our mesh / previous λ=2',source=fit,raster=raster,
                regularization=2.,source_metrics=sm,image_metrics=im))
        fit,raster,sm,im=full_fits[selected['regularization']]
        products.append(dict(name=basis+'_cv',title='Our mesh / pixel-CV λ' if basis=='our_centroidal' else 'Released mesh / pixel-CV λ',
            source=fit,raster=raster,regularization=selected['regularization'],source_metrics=sm,image_metrics=im,
            conditional_validation=selected))
        print(f"{basis}: selected λ={selected['regularization']:g}, cosine={sm['cosine']:.5f}, bright χ²/N={im['bright_arc_chi2_per_pixel']:.4f}",flush=True)
    np.savez_compressed(OUTPUT/'regularization_scan.npz',regularizations=REGULARIZATIONS,**arrays)
    write_json(OUTPUT/'conditional_validation.json',dict(seed=20261119,fraction=.2,rows=cv,
        selected_without_reference_source_similarity=True,
        released_mesh_informed_by_full_official_fit=True,
        selected_lambda_is_scan_minimum_not_a_claim_of_universal_optimum=True,
        validation_is_conditional_diagnostic_not_independent_discovery_test=True))
    with (OUTPUT/'regularization_scan.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(regularization_rows[0]));writer.writeheader();writer.writerows(regularization_rows)
    summary=[]
    for product in products:
        metadata={k:v for k,v in product.items() if k not in ('source','raster')}
        metadata['node_count']=len(product['source'].coefficients)
        metadata['macro_fixed']=obs.macro_model
        save_fit(product['name'],obs,product['source'],product['raster'],header,metadata)
        summary.append(metadata)
    write_json(OUTPUT/'summary.json',dict(identifier=row['unique_identifier'],seconds=perf_counter()-started,
        macro_and_foreground_fixed=True,no_pinn_or_halo_fitted=True,products=summary,
        source_similarity_used_only_for_evaluation=True,
        foreground_stage='released Sersic deblending foreground; original vis_pix foreground unavailable'))
    plot_comparison(products,obs,extent,OUTPUT/'fixed_mass_source_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for basis,label in [('our_centroidal','Our mesh'),('released_mesh','Released mesh')]:
        trace=[r for r in regularization_rows if r['basis']==basis]
        axes[0].semilogx([r['regularization'] for r in trace],[r['validation_chi2_per_pixel'] for r in trace],'o-',label=label)
        axes[1].semilogx([r['regularization'] for r in trace],[r['cosine'] for r in trace],'o-',label=label)
    axes[0].set(xlabel='Source regularization λ',ylabel='Held-out image χ²/pixel')
    axes[1].set(xlabel='Source regularization λ',ylabel='Source cosine versus released source')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUTPUT/'regularization_sensitivity.png',dpi=170,bbox_inches='tight');plt.close(fig)
    lines=['# Fixed official mass and lens light: source consistency\n',
        'All seven released vis_pix SIE+shear parameters, slope=2, foreground lens light, mask, RMS, PSF and 4×4 subpixel integration '
        'are held fixed. Only nonnegative source coefficients are optimized. No PINN, subhalo, image registration or flux rescaling is fitted.\n',
        'The official model.fits foreground is from its Sérsic photometry stage; the mass and source-node CSV are from vis_pix. '
        'The reference residual is therefore re-rendered with this common fixed foreground, not claimed to reproduce the original vis_pix fit residual.\n',
        'Columns compare the released source coefficients, our usual centroidal mesh with λ=2, our mesh with λ chosen from held-out observed pixels, '
        'and a control using the released source vertices with held-out λ. Source similarity never selects λ. '
        'The released mesh was informed by the official full-data fit, so that control is conditional reconstruction rather than blind validation.\n',
        '| Method | Nodes | λ | Raw cosine | 0.02″ smoothed cosine | χ²/pixel | Bright-arc χ²/pixel |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in summary:
        reg='not supplied' if p['regularization'] is None else f"{p['regularization']:g}"
        lines.append(f"| {p['name']} | {p['node_count']} | {reg} | {p['source_metrics']['unsmoothed_cosine_same_support']:.5f} | {p['source_metrics']['cosine']:.5f} | {p['image_metrics']['chi2_per_pixel']:.4f} | {p['image_metrics']['bright_arc_chi2_per_pixel']:.4f} |")
    lines+=['\nSources share identical sky coordinates, view, colour scale and common 0.02-arcsec smoothing for the quantitative comparison. '
        'Source-node coordinates include only the known shear-origin translation. Agreement with the published reconstruction does not establish source truth.\n',
        'Raw and smoothed similarities use the same fixed support derived from the smoothed reference. '
        'The near-unity smoothed score describes large-scale agreement; fine-scale intensities are not reproduced exactly.\n',
        'For this target, λ=2 suppresses resolved structure even when the mass is fixed to the released model. '
        'Reducing λ on the same mesh restores much of that structure; using the released mesh additionally resolves the compact peaks. '
        'Both conditional-validation choices lie at the weakest tested λ=2e-6, where the scan is nearly flat. '
        'This establishes a weak-regularization regime for this diagnostic, not a universally optimal numerical λ.\n',
        'The earlier PINN/macro/halo workflow and its products are preserved. This is a separate source-only diagnostic.\n',
        'Run `.venv/bin/python sublens_testing/euclid_dr1_fixed_source.py`.\n']
    (OUTPUT/'README.md').write_text('\n'.join(lines))
    files=[directory/'model.fits',directory/'coolest.json',directory/'deflections.fits',directory/'dataset'/f'{directory.name}.fits',
        directory/'dataset/info.json',directory/'dataset/segmentation/artefact_binary.fits',directory/'vis_pix_source_plane_reconstruction_0.csv']
    write_json(OUTPUT/'input_sha256.json',{str(p.relative_to(ROOT)):digest(p) for p in files})
    if Path('/tmp/sublens_dr1_fixed_source_resources.json').exists():
        shutil.copyfile('/tmp/sublens_dr1_fixed_source_resources.json',OUTPUT/'resources.json')
    verify_outputs(obs,data,products,validation)


if __name__=='__main__':main()
