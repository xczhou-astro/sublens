"""Report separate joint-chain branches without assigning unmeasured mode weights."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.signal import fftconvolve
from astropy.io import fits
import torch
from euclid_dr1_joint_lambda import OUT,PILOT,ROOT,JointProblem,read_observation,summarize,save_json
from euclid_grade_a_10 import ProfiledMacroFit,source_shift
from euclid_dr1_fixed_source import source_scores


def main():
    torch.set_num_threads(1)
    summarize()
    report=json.loads((OUT/'summary.json').read_text())
    obs,train,test,guard,folder=read_observation();design=np.load(OUT/'design.npz')
    valid_holdout=not bool(np.any(train & test))
    j=JointProblem(obs,train,design['points']);grid=j.problem.grid
    with fits.open(OUT/'joint_reconstruction.fits') as h:
        header=h['SOURCE_MEAN'].header.copy();official=h['OFFICIAL_SOURCE'].data.copy()
    x=(np.arange(301)+1-header['CRPIX1'])*header['CDELT1']+header['CRVAL1']
    y=(np.arange(301)+1-header['CRPIX2'])*header['CDELT2']+header['CRVAL2']
    xx,yy=np.meshgrid(x,y)
    prior=fits.getdata(PILOT/'fixed_official_macro_source/our_centroidal_cv.fits','SOURCE')
    wideaxis=np.linspace(-1.2,1.2,401);wx,wy=np.meshgrid(wideaxis,wideaxis)
    from sublens.survey.euclid_dr1 import load_euclid_dr1_lens
    reference=load_euclid_dr1_lens(folder,oversample=4)
    official_wide=LinearNDInterpolator(reference.source_points,reference.source_values,fill_value=0)(wx,wy)
    records=[];products=[];hdus=[fits.PrimaryHDU()]
    converged=report['posterior_status']=='convergence checks passed'
    hdus[0].header['CONVERGD']=converged
    hdus[0].header['MODEWGHT']=False
    for c in range(4):
        saved=np.load(OUT/f'chain_{c}.npz');burn=int(saved['burn'])
        parameters=saved['parameters'][burn:];coeff=saved['sources'][burn:]
        index=np.linspace(0,len(parameters)-1,60,dtype=int)
        maps=[];wides=[];models=[]
        for i in index:
            macro=ProfiledMacroFit.model(parameters[i,:7]);shift=source_shift(macro)
            interp=LinearNDInterpolator(grid,coeff[i],fill_value=0)
            maps.append(interp(xx+shift[0],yy+shift[1]));wides.append(interp(wx+shift[0],wy+shift[1]))
            native,_=j.geometry(parameters[i,:7])
            models.append(fftconvolve((native@coeff[i]).reshape(obs.image.shape),obs.psf,mode='same'))
        maps=np.array(maps);models=np.array(models)
        mean=maps.mean(0);prediction=models.mean(0);residual=(obs.image-prediction)/obs.rms
        metrics=source_scores(official,mean,header['CDELT1'])
        entry=dict(chain=c,source_comparison=metrics,train_chi2_per_pixel=float(np.mean(residual[train]**2)),
            heldout_chi2_per_pixel=float(np.mean(residual[test]**2)),full_mask_chi2_per_pixel=float(np.mean(residual[obs.mask]**2)),
            heldout_blocks_chi2_per_pixel=[float(np.mean(residual[test&(np.indices(test.shape)[0]<50)]**2)),
                                        float(np.mean(residual[test&(np.indices(test.shape)[0]>=50)]**2))],
            empirical_lambda_q16_q50_q84=np.quantile(10**parameters[:,7],[.16,.5,.84]),
            empirical_Reff_q16_q50_q84=np.quantile(parameters[:,9],[.16,.5,.84]),
            empirical_theta_E_q16_q50_q84=np.quantile(np.exp(parameters[:,0]),[.16,.5,.84]),
            empirical_macro_medians=dict(zip(['ln_theta_E','q','phi','center_x/.2','center_y/.2','g1/.2','g2/.2'],np.median(parameters[:,:7],axis=0))),
            within_chain_lambda_macro_correlations=np.corrcoef(parameters[:,:8].T)[7,:7],
            values_are_credible_intervals=converged)
        records.append(entry)
        products.append(dict(source=mean,wide=np.mean(wides,axis=0),sd=maps.std(0),model=prediction,residual=residual))
        for name,value in [('SOURCE',mean),('SOURCE_SD',maps.std(0)),('MODEL',prediction),('RESIDUAL',obs.image-prediction)]:
            hdus.append(fits.ImageHDU(value,header=header.copy() if name.startswith('SOURCE') else None,name=f'C{c}_{name}'))
    report['chain_branches']=records
    # An unweighted mixture of unmixed chains does not estimate posterior mode weights.
    pool_keys=['source_metrics','marginal_summaries','lambda_quantiles','effective_einstein_radius_quantiles',
               'lambda_macro_correlations','mean_model_train_chi2_per_pixel','mean_model_heldout_chi2_per_pixel',
               'sampled_model_heldout_chi2_quantiles','heldout_standardized_predictive_rms']
    report['unweighted_diagnostic_pool']={key:report.pop(key) for key in pool_keys}
    report['mode_weights_estimated']=False
    report['reference_blocks_are_heldout']=valid_holdout
    report['reported_branch_ranges_are_credible_intervals']=converged
    report['interpretation']='Compare branches separately. Unmixed chains must not be combined as a posterior mean or credible interval.'
    save_json(OUT/'summary.json',report)
    fits.HDUList(hdus).writeto(OUT/'chain_branch_reconstructions.fits',overwrite=True,checksum=True)
    # Main figure contains ALL branches. Source agreement never chooses a branch.
    maximum=max(official.max(),prior.max(),max(p['source'].max() for p in products))
    norm=AsinhNorm(linear_width=.03*maximum,vmin=0,vmax=maximum)
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    panels=[(official,'Released source: comparison only'),(prior,'Previous conditional control\nReleased macro fixed; pixel-CV lambda')]
    for c,(product,r) in enumerate(zip(products,records)):
        panels.append((product['source'],f'Chain {c}: exploratory sample mean\nRaw / smoothed cosine: '
            f'{r["source_comparison"]["unsmoothed_cosine_same_support"]:.3f} / {r["source_comparison"]["cosine"]:.3f}\n'
            f'Reff~{r["empirical_Reff_q16_q50_q84"][1]:.3f} arcsec; lambda~{r["empirical_lambda_q16_q50_q84"][1]:.3g}'))
    extent=[x[0],x[-1],y[0],y[-1]]
    for axis,(image,title) in zip(axes.ravel(),panels):
        artist=axis.imshow(image,origin='lower',extent=extent,cmap='magma',norm=norm)
        axis.set(title=title,xlabel='βx west [arcsec]',ylabel='βy north [arcsec]');fig.colorbar(artist,ax=axis,shrink=.7)
    fig.suptitle(f'{obs.lens_id}: joint macro / positive source / lambda\n'
        f'{report["posterior_status"]}; no posterior weights assigned to branches\n'
        'Shared comparison window; fixed released foreground and PSF; official source excluded from fitting',fontsize=14)
    fig.savefig(OUT/'joint_source_comparison.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
    norm=AsinhNorm(linear_width=.01,vmin=0,vmax=np.percentile(obs.image[obs.mask],99.9))
    for c,(p,r) in enumerate(zip(products,records)):
        artist=axes[0,c].imshow(p['model'],origin='lower',extent=[-5,5,-5,5],cmap='magma',norm=norm)
        axes[0,c].set(title=f'Chain {c}: mean lensed source image');fig.colorbar(artist,ax=axes[0,c],shrink=.7)
        artist=axes[1,c].imshow(np.where(obs.mask,p['residual'],np.nan),origin='lower',extent=[-5,5,-5,5],cmap='RdBu_r',vmin=-5,vmax=5)
        axes[1,c].contour(test,levels=[.5],origin='lower',extent=[-5,5,-5,5],colors=['lime'],linewidths=.8)
        axes[1,c].set(title=f'Residual / RMS\nTrain chi2/N={r["train_chi2_per_pixel"]:.3f}; block={r["heldout_chi2_per_pixel"]:.3f}')
        fig.colorbar(artist,ax=axes[1,c],shrink=.7)
        for a in axes[:,c]:a.set(xlim=(-2.5,2.5),ylim=(-2.5,2.5),xlabel='x west [arcsec]',ylabel='y north [arcsec]')
    fig.suptitle('Conditional image-plane prediction: all macro branches; green = 32 heldout pixels\nFull PSF guards excluded; foreground fixed' if valid_holdout else 'Full-data fit: all pixels used; green blocks are in-sample diagnostics, not validation\nForeground and PSF fixed')
    fig.savefig(OUT/'branch_image_residuals.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(20,4.5),layout='constrained')
    maximum=max(official_wide.max(),max(p['wide'].max() for p in products))
    norm=AsinhNorm(linear_width=.03*maximum,vmin=0,vmax=maximum)
    for axis,image,title in zip(axes,[official_wide]+[p['wide'] for p in products],['Released source']+[f'Chain {c}' for c in range(4)]):
        axis.imshow(image,origin='lower',extent=[-1.2,1.2,-1.2,1.2],cmap='magma',norm=norm)
        axis.set(title=title,xlabel='βx west [arcsec]',ylabel='βy north [arcsec]')
    fig.suptitle('Wide source-plane view: no registration or fitted intensity rescaling; unmixed branches shown separately')
    fig.savefig(OUT/'wide_source_branches.png',dpi=150);plt.close(fig)
    # Mark pooled FITS as diagnostics, not a posterior reconstruction.
    with fits.open(OUT/'joint_reconstruction.fits',mode='update') as h:
        h[0].header['CONVERGD']=converged;h[0].header['MODEWGHT']=False;h[0].header['DIAGPOOL']=True
        for a in h:a.add_checksum()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    old=json.loads((OUT/'previous_fits_sha256.json').read_text())
    for name,digest in old.items():assert sha(ROOT/name)==digest,name
    hashes={}
    for file in OUT.glob('*.fits'):
        with fits.open(file,checksum=True) as h:
            assert all(a.verify_checksum()==1 and a.verify_datasum()==1 for a in h),file
        hashes[file.name]=sha(file)
    input_hashes=json.loads((OUT/'inference_input_sha256.json').read_text())
    for name,digest in input_hashes.items():assert sha(ROOT/name)==digest,name
    assert sha(OUT/'sampler_run_source.py')==sha(ROOT/'sublens_testing/euclid_dr1_joint_lambda.py')
    save_json(OUT/'verification.json',dict(all_fits_checksums_passed=True,fits_sha256=hashes,
        earlier_fits_unchanged=len(old),inference_input_hashes_unchanged=True,
        frozen_sampler_source_sha256=sha(OUT/'sampler_run_source.py'),report_source_sha256=sha(Path(__file__)),
        unmixed_branches_not_assigned_posterior_weights=True))
    (OUT/'README.md').write_text('Conditional joint macro / positive Delaunay source / lambda pilot on EUCL-DR1-L020080.\n\n'
        f'Actual status: {report["posterior_status"]}.\n\n'
        'Read protocol.json and summary.json. Four chains started at four independently optimized image-based initial models. '
        'No official macro or source mesh was supplied; the official source was read after sampling for comparison only. '
        'The published Sersic foreground subtraction and released PSF are fixed. '
        + ('The heldout 32 pixels have full PSF guards. ' if valid_holdout else 'All valid pixels were fitted; the reference blocks are in-sample. ')
        + 'The foreground template used the full image, and noise correlations are not modeled; independent observational validation is not claimed.\n\n'
        'Unmixed chains are displayed separately. The ranges in each branch are descriptive sample quantiles, not validated credible intervals. '
        'Mode weights were not measured. joint_reconstruction.fits contains an explicitly marked unweighted diagnostic pool; '
        'chain_branch_reconstructions.fits and joint_source_comparison.png are the appropriate branch-specific comparisons. '
        'No agreement metric selected priors, lambda, a macro branch or a stopping rule. No subhalo was fitted.\n\n'
        'Reproduce: .venv/bin/python sublens_testing/euclid_dr1_joint_lambda.py prepare; '
        '.venv/bin/python sublens_testing/euclid_dr1_joint_lambda.py sample --steps 3000 --burn 1500; '
        '.venv/bin/python sublens_testing/euclid_dr1_joint_lambda_report.py\n')
    print('Branch report complete',flush=True)
    for r in records:print(json.dumps(r,default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':main()
