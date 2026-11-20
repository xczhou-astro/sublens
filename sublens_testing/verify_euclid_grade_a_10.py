"""Read-only verification of the ten real-data inputs and inference products."""

import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from euclid_grade_a_10 import (
    DATA, OUTPUT, ROOT, effective_sie_radius, macro_from_release,
    source_raster, source_shift, similarity,
)
from run import load_result
from strength_scan import digest, write_json
from sublens import LensingObservation
from sublens.inversion.adaptive_refinement import _model_rays
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.survey.euclid import centered_crop, get_available_grade_a_lenses
from sublens.survey.euclid_source import _q1_source_rays


def verify_inputs(selection):
    complete = []
    for r in get_available_grade_a_lenses(DATA / 'q1_discovery_engine_lens_catalog.csv', DATA):
        p = Path(r['dir_path'])
        if all((p/f).exists() for f in ('info.json', 'mask_extra_galaxies.fits',
            'result/lens_light.fits', 'result/source_light.fits', 'result/source_reconstruction.fits')):
            complete.append(r)
    assert selection['lenses'] == complete[:10]
    assert all(r['grade'] == 'A' for r in selection['lenses'])
    rows = []
    for r in selection['lenses']:
        lid = r['id_str']; folder = OUTPUT/lid; raw = Path(r['dir_path'])
        for filename, checksum in json.loads((folder/'input_sha256.json').read_text()).items():
            assert digest(ROOT/filename) == checksum, filename
        obs = LensingObservation.from_fits(folder/'observation.fits')
        mass = json.loads((raw/'result_lens_mass.json').read_text())
        info = json.loads((raw/'info.json').read_text())
        with fits.open(raw/f'{lid}.fits') as h:
            image = centered_crop(h['VIS_FLUX'].data.copy(), 100)
            rms = centered_crop(h['VIS_RMS'].data, 100).astype(float)
            psf = h['VIS_PSF'].data.astype(float)
        image -= centered_crop(fits.getdata(raw/'result/lens_light.fits', 'VIS'), 100)
        x, y = [v.mean(axis=-1) for v in obs.image_rays()]
        cy, cx = info.get('mask_centre', (0, 0))
        mask = (np.hypot(x-cx, y-cy) <= info['mask_radius'])
        mask &= centered_crop(fits.getdata(raw/'mask_extra_galaxies.fits'), 100) == 0
        mask &= np.isfinite(image) & np.isfinite(rms) & (rms > 0)
        np.testing.assert_array_equal(obs.mask, mask)
        np.testing.assert_array_equal(obs.image, image)
        np.testing.assert_array_equal(obs.rms, rms)
        np.testing.assert_allclose(obs.psf, psf/psf.sum(), atol=1e-15)
        assert obs.bx.shape == (100, 100, 4)
        _, _, bx, by = _q1_source_rays(image.shape, obs.pixel_scale, mass, 2)
        shift = source_shift(macro_from_release(mass))
        np.testing.assert_allclose(obs.bx, bx+shift[0], atol=1e-12)
        np.testing.assert_allclose(obs.by, by+shift[1], atol=1e-12)
        with fits.open(folder/'official_reference.fits') as h:
            np.testing.assert_array_equal(h['MODEL'].data,
                centered_crop(fits.getdata(raw/'result/source_light.fits', 'VIS'), 100))
            np.testing.assert_array_equal(h['SOURCE'].data,
                fits.getdata(raw/'result/source_reconstruction.fits', 'VIS'))
        rows.append(dict(lens_id=lid, input_mask_pixels=int(mask.sum()),
            raw_hashes_verified=True, released_ray_conversion_verified=True))
    return rows


def verify_results(rows):
    for row in rows:
        folder = OUTPUT/row['lens_id']
        metrics = json.loads((folder/'metrics.json').read_text())
        result = load_result(folder/'result.fits')
        obs = LensingObservation.from_fits(folder/'observation.fits')
        np.testing.assert_array_equal(result.observation.image, obs.image)
        np.testing.assert_array_equal(result.observation.mask, obs.mask)
        for name in ('rms', 'psf'):
            np.testing.assert_allclose(getattr(result.observation, name), getattr(obs, name), atol=1e-15)
        macro = json.loads((folder/'macro_fit.json').read_text())
        assert len(macro['attempts']) == 4
        assert all(np.isfinite(s['objective']) for s in macro['attempts'])
        assert .2 <= macro['macro']['theta_E'] <= 4.5
        assert metrics['subhalo_presence_confirmed'] is None
        assert metrics['candidate_count'] == len(result.candidates)
        assert metrics['pinn_scoring_pixels'] == int(result.residual_potential.mask.sum())
        np.testing.assert_allclose(metrics['theta_macro_effective_arcsec'],
                                   effective_sie_radius(result.macro_model), atol=1e-13)
        problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
            source_grid=result.baseline.grid, regularization=result.baseline.regularization)
        bx, by = _model_rays(obs, result.macro_model)
        np.testing.assert_allclose(problem.render(bx, by, result.baseline.coefficients),
                                   result.baseline.image_model, atol=2e-11)
        for candidate in result.candidates:
            fit = candidate.fit
            bx, by = _model_rays(obs, fit.macro_model, fit)
            np.testing.assert_allclose(problem.render(bx, by, fit.source.coefficients),
                                       fit.source.image_model, atol=2e-11)
            gain = (result.baseline.objective-fit.source.objective)*obs.mask.sum()
            np.testing.assert_allclose(gain, fit.delta_objective, atol=2e-9)
        with fits.open(folder/'result.fits') as h:
            np.testing.assert_allclose(h['SCI'].data-h['MODEL_MACRO'].data, h['RESID_MACRO'].data, atol=1e-14)
            np.testing.assert_allclose(h['SCI'].data-h['MODEL'].data, h['RESIDUAL'].data, atol=1e-14)
            for model, key in (('MODEL_MACRO', 'macro'), ('MODEL', 'halo')):
                chi = np.mean(((obs.image[obs.mask]-h[model].data[obs.mask])/obs.rms[obs.mask])**2)
                np.testing.assert_allclose(chi, metrics['image_scores'][key]['aperture_chi2_per_pixel'], atol=1e-12)
        with fits.open(folder/'source_comparison.fits') as h:
            reference = h['OFFICIAL'].data
            for key in ('PUBLISHED_MASS', 'MACRO_ONLY', 'MACRO_HALO'):
                recalculated = similarity(reference, h[key].data)
                for metric in ('cosine_support', 'cosine_full', 'pearson_support', 'flux_ratio'):
                    np.testing.assert_allclose(recalculated[metric], metrics['source_similarity'][key][metric], atol=1e-13)
                assert h[key].header['CRPIX1'] == 101 and h[key].header['CDELT2'] == .01
            np.testing.assert_allclose(h['MACRO_ONLY'].data,
                                       source_raster(result.baseline, result.macro_model), atol=1e-13)
            best = result.best_candidate
            source = result.baseline if best is None else best.fit.source
            model = result.macro_model if best is None else best.fit.macro_model
            np.testing.assert_allclose(h['MACRO_HALO'].data, source_raster(source, model), atol=1e-13)
        if result.best_candidate is not None:
            best = result.best_candidate.fit
            candidate = metrics['best_candidate']
            np.testing.assert_allclose(best.position, candidate['position_arcsec'], atol=1e-13)
            np.testing.assert_allclose(best.delta_chi2, candidate['delta_chi2'], atol=1e-12)
            assert candidate['converged'] == best.converged
        row.update(result_and_source_metrics_verified=True,
                   candidate_existence_not_claimed=True, candidate_count=len(result.candidates))


def main():
    selection = json.loads((OUTPUT/'selection.json').read_text())
    rows = verify_inputs(selection)
    verify_results(rows)
    products = {}
    for p in sorted(OUTPUT.rglob('*.fits')):
        with fits.open(p, checksum=True) as h:
            h.verify('exception')
            assert all(hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1 for hdu in h), p
        products[str(p.relative_to(OUTPUT))] = digest(p)
    provenance = json.loads((OUTPUT/'publisher_provenance.json').read_text())
    assert provenance['export_flip_for_ds9'] and provenance['fixed_source_size'] == [201, 201]
    code = [Path(__file__), ROOT/'sublens_testing/euclid_grade_a_10.py',
        *[ROOT/'sublens/inversion'/f for f in ('source_plane.py', 'potential.py', 'halo_search.py',
                                             'joint_refinement.py', 'adaptive_refinement.py')]]
    write_json(OUTPUT/'verification.json', dict(lenses_verified=len(rows), fits_verified=len(products),
        rows=rows, fits_sha256=products, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in code}))
    print(f'Verified {len(rows)} real lenses and {len(products)} FITS products.')


if __name__ == '__main__':
    main()
