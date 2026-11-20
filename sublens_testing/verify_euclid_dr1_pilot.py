"""Verify DR1 data/frame provenance, forward models and the nine-panel products."""

import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

sys.path.insert(0,str(Path(__file__).resolve().parent))
from euclid_dr1_pilot import ROOT,DATA,OUTPUT,select_lens,deflection_check,source_shift,effective_sie_radius
from run import load_result
from strength_scan import digest,write_json
from sublens import LensingObservation
from sublens.survey.euclid_dr1 import load_euclid_dr1_lens
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.adaptive_refinement import _model_rays


def main():
    selection=json.loads((OUTPUT/'selection.json').read_text());row=selection['lens']
    expected,_=select_lens();assert expected==row
    assert row['grade']=='A' and row['inspection_consensus_75']=='Success'
    for name,checksum in json.loads((OUTPUT/'input_sha256.json').read_text()).items():
        assert digest(ROOT/name)==checksum,name
    data=load_euclid_dr1_lens(DATA/row['tile_id'],oversample=4);obs=data.observation
    stored=LensingObservation.from_fits(OUTPUT/'observation.fits')
    for name in ('image','mask','rms','bx','by'):
        np.testing.assert_array_equal(getattr(obs,name),getattr(stored,name))
    np.testing.assert_allclose(obs.psf,stored.psf,atol=1e-15)
    assert obs.bx.shape==(100,100,16)
    assert deflection_check(DATA/row['tile_id'],obs.macro_model)['maximum_component_error_arcsec']<1e-8
    result=load_result(OUTPUT/'result.fits');metrics=json.loads((OUTPUT/'metrics.json').read_text())
    for name in ('image','mask','rms'):
        np.testing.assert_array_equal(getattr(obs,name),getattr(result.observation,name))
    grid=result.baseline.grid
    problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,
        source_grid=grid,regularization=result.baseline.regularization)
    bx,by=_model_rays(obs,result.macro_model)
    np.testing.assert_allclose(problem.render(bx,by,result.baseline.coefficients),result.baseline.image_model,atol=2e-11)
    for candidate in result.candidates:
        fit=candidate.fit;bx,by=_model_rays(obs,fit.macro_model,fit)
        np.testing.assert_allclose(problem.render(bx,by,fit.source.coefficients),fit.source.image_model,atol=2e-11)
        np.testing.assert_allclose(fit.delta_objective,(result.baseline.objective-fit.source.objective)*obs.mask.sum(),atol=2e-9)
    with fits.open(OUTPUT/'official_reference.fits') as h:
        np.testing.assert_array_equal(h['RAW_DATA'].data,data.raw_image)
        np.testing.assert_array_equal(h['LENS_LIGHT'].data,data.lens_light)
        np.testing.assert_array_equal(h['RELEASED_SERSIC_SOURCE'].data,data.official_lensed_source)
        np.testing.assert_allclose(h['SERSIC_RESIDUAL'].data,obs.image-data.official_lensed_source,atol=1e-14)
        ref_problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,
            source_grid=Delaunay(data.source_points),regularization=1.)
        shift=source_shift(obs.macro_model)
        reference=ref_problem.render(obs.bx-shift[0],obs.by-shift[1],data.source_values)
        np.testing.assert_allclose(h['REFERENCE_SOURCE'].data,reference,atol=1e-13)
        np.testing.assert_allclose(h['REFERENCE_RESIDUAL'].data,obs.image-reference,atol=1e-13)
    with fits.open(OUTPUT/'result.fits') as h:
        np.testing.assert_allclose(h['SCI'].data-h['MODEL_MACRO'].data,h['RESID_MACRO'].data,atol=1e-14)
        np.testing.assert_allclose(h['SCI'].data-h['MODEL'].data,h['RESIDUAL'].data,atol=1e-14)
        for ext,key in [('MODEL_MACRO','macro'),('MODEL','halo')]:
            chi=np.mean(((obs.image[obs.mask]-h[ext].data[obs.mask])/obs.rms[obs.mask])**2)
            np.testing.assert_allclose(chi,metrics['image_scores'][key]['chi2_per_pixel'],atol=1e-12)
    with fits.open(OUTPUT/'source_comparison.fits') as h:
        header=h[1].header;n=h[1].data.shape[0]
        x=(np.arange(n)+1-header['CRPIX1'])*header['CDELT1']+header['CRVAL1']
        y=(np.arange(n)+1-header['CRPIX2'])*header['CDELT2']+header['CRVAL2']
        xx,yy=np.meshgrid(x,y)
        reference=LinearNDInterpolator(data.source_points,data.source_values,fill_value=0.)(xx,yy)
        np.testing.assert_allclose(h['OFFICIAL'].data,reference,atol=1e-12)
        best=result.best_candidate
        for ext,source,macro in [('MACRO_ONLY',result.baseline,result.macro_model),
            ('MACRO_HALO',result.baseline if best is None else best.fit.source,result.macro_model if best is None else best.fit.macro_model)]:
            shift=source_shift(macro)
            np.testing.assert_allclose(h[ext].data,source.sample(xx+shift[0],yy+shift[1]),atol=1e-12)
    np.testing.assert_allclose(metrics['theta_macro'],effective_sie_radius(result.macro_model),atol=1e-14)
    assert metrics['subhalo_existence_confirmed'] is None
    assert metrics['candidate_count']==len(result.candidates)
    if result.best_candidate:
        best=result.best_candidate.fit;reported=metrics['best_candidate']
        np.testing.assert_allclose(reported['position_xy_arcsec'],best.position,atol=1e-14)
        pixel=WCS(data.header).all_world2pix([[reported['ra_deg'],reported['dec_deg']]],0)[0]
        position=(pixel-(np.array(obs.image.shape[::-1])-1)/2)*obs.pixel_scale
        np.testing.assert_allclose(position,best.position,atol=2e-8)
    assert json.loads((OUTPUT/'psf_diagnostics.json').read_text())['numerically_usable']
    assert (OUTPUT/'dr1_nine_panel_comparison.png').stat().st_size>100_000
    hashes={}
    for p in sorted(OUTPUT.glob('*.fits')):
        with fits.open(p,checksum=True) as h:
            h.verify('exception')
            assert all(x.verify_checksum()==1 and x.verify_datasum()==1 for x in h),p
        hashes[p.name]=digest(p)
    sources=[ROOT/'sublens/survey/euclid_dr1.py',ROOT/'sublens_testing/euclid_dr1_pilot.py',Path(__file__)]
    write_json(OUTPUT/'verification.json',dict(all_checks_passed=True,lenses_fitted=1,fits_verified=len(hashes),
        raw_inputs_preserved=True,frame_and_source_shear_translation_verified=True,
        released_sersic_and_rerendered_delaunay_references_separated=True,
        model_residual_and_all_candidate_forward_predictions_verified=True,
        sky_position_roundtrip_verified=True,no_subhalo_detection_or_absence_claim=True,
        fits_sha256=hashes,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources}))
    print(f'Verified one DR1 lens, all candidate forward models and {len(hashes)} FITS products.')


if __name__=='__main__':main()
