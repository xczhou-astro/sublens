"""Conditional follow-up of a frozen list of real DR1 PINN candidates.

All hypotheses reprofile a nonnegative source on the SAME basis and prior per
comparison. Seven SIE parameters (or eight EPL parameters) are free, including
shape and centre. This experiment does not change production search defaults.
Stability and optimizer checks are not calibrated subhalo significance tests.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gc
import json
import math
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import torch
from astropy.io import fits
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, SymLogNorm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import euclid_dr1_grade_a_scan as scan
import euclid_dr1_pilot as pilot
from euclid_dr1_batch_macro import sie_alpha
from run import load_result, residual_metrics
from sublens import LensingObservation, torch_backend, summarize_subhalo
from sublens.execution import current_execution
from sublens.inversion.halo_search import HaloFit
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.torch_source import torch_problem
from sublens.inversion.torch_refinement import minimize_profiled
from sublens.inversion.potential import _smooth_convergence
from sublens.profiles.macro import epl_deflection
from sublens.profiles.subhalo import truncated_subhalo_deflection

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / 'sublens_testing/euclid_dr1_grade_a_part1_torch_mps'
OUT = SCAN / 'candidate_followup'
HALO_CAP = .30
HALO_RADIUS = .70
MAXITER = 140
VARIANT_ORDER=['full_sie','mesh600_seed11','mesh800_seed33','lambda_x10','free_epl_slope',
    'wide_macro_bounds','psf_width_minus5pct','psf_width_plus5pct','foreground_minus2pct','foreground_plus2pct']


def vector(macro, slope=False):
    g = macro.get('gamma_ext', 0.)
    angle = 2 * macro.get('phi_ext', 0.)
    v = [math.log(macro['theta_E']), macro['q'], macro['phi'],
         macro['center'][0] / .2, macro['center'][1] / .2,
         g * math.cos(angle) / .2, g * math.sin(angle) / .2]
    return np.array(v + ([macro.get('gamma', 2.)] if slope else []))


def macro_at(v, slope=False):
    v = np.asarray(v)
    return dict(theta_E=float(np.exp(v[0])), q=float(v[1]),
                phi=float((v[2] + np.pi/2) % np.pi - np.pi/2),
                center=[float(.2*v[3]), float(.2*v[4])],
                gamma_ext=float(.2*np.hypot(v[5], v[6])),
                phi_ext=float(.5*np.arctan2(v[6], v[5])),
                gamma=float(v[7]) if slope else 2.)


def macro_bounds(macro, slope=False, wide=False):
    v = vector(macro, slope)
    bounds = [(v[0]+np.log(.7), v[0]+np.log(1.3)), (.25, .995),
              (v[2]-.7, v[2]+.7), (v[3]-.75, v[3]+.75),
              (v[4]-.75, v[4]+.75),
              (max(-2., v[5]-.6), min(2., v[5]+.6)),
              (max(-2., v[6]-.6), min(2., v[6]+.6))]
    if wide:
        bounds=[(v[0]+np.log(.5),v[0]+np.log(1.6)),(.12,.995),(v[2]-np.pi/2,v[2]+np.pi/2),
                (v[3]-1.75,v[3]+1.75),(v[4]-1.75,v[4]+1.75),
                (max(-2.5,v[5]-1.25),min(2.5,v[5]+1.25)),
                (max(-2.5,v[6]-1.25),min(2.5,v[6]+1.25))]
    return bounds + ([(1.6, 2.4)] if slope else [])


def bounded_start(start,bounds):
    """Unwrap the pi-periodic ellipse angle before applying finite bounds."""
    if torch.is_tensor(start):start=start.detach().cpu().numpy()
    start=np.array(start,dtype=float,copy=True)
    limits=np.asarray(bounds)
    midpoint=float(limits[2].mean())
    start[2]+=np.pi*np.rint((midpoint-start[2])/np.pi)
    return np.clip(start,limits[:,0],limits[:,1])


class FullMacroProblem:
    """Matched source-profile objective; every continuous parameter uses Torch."""
    def __init__(self, obs, problem, slope=False):
        self.obs, self.problem, self.slope = obs, problem, slope
        self.engine = torch_problem(problem)
        self.x, self.y = [self.engine.tensor(a) for a in obs.image_rays()]
        self.coordinates = torch.stack((self.x.flatten(), self.y.flatten()), 1)
        self.nmacro = 8 if slope else 7

    def rays(self, v, halo=False):
        cx, cy = .2*v[3], .2*v[4]
        if self.slope:
            alpha, _ = epl_deflection(self.coordinates, theta_E=v[0].exp(), q=v[1],
                                      phi=v[2], gamma=v[7], center=(cx, cy))
        else:
            alpha = sie_alpha(self.coordinates, v[0].exp(), v[1], v[2], cx, cy)
        dx, dy = self.coordinates[:, 0]-cx, self.coordinates[:, 1]-cy
        shear = .2*torch.stack((v[5]*dx+v[6]*dy, v[6]*dx-v[5]*dy), -1)
        beta = (self.coordinates-alpha-shear).reshape(*self.x.shape, 2)
        if halo:
            hx, hy, strength = v[self.nmacro:]
            ax, ay = truncated_subhalo_deflection(self.x, self.y, hx, hy, strength*HALO_CAP)
            beta = beta-torch.stack((ax, ay), -1)
        return beta[..., 0], beta[..., 1]

    def optimize(self, start, bounds, halo, initial=None, maxiter=MAXITER):
        result = minimize_profiled(self.engine, lambda v: self.rays(v, halo),
                                   bounded_start(start,bounds),
                                   bounds, maxiter=maxiter, initial=initial)
        # Continue a budget-limited fit without loosening convergence tolerances.
        if not result.converged:
            retry = minimize_profiled(self.engine, lambda v: self.rays(v, halo),
                                      result.values, bounds, maxiter=maxiter,
                                      initial=result.source.coefficients)
            if float(retry.source.objective) <= float(result.source.objective)+1e-9:
                result = retry
        return result


def choose(outcomes):
    best = min(float(o.source.objective) for o in outcomes)
    near = [o for o in outcomes if float(o.source.objective) <= best+1e-9]
    return min(near, key=lambda o: (not o.converged, float(o.source.objective)))


def optimum_record(o):
    return dict(objective=float(o.source.objective), converged=bool(o.converged),
                iterations=o.iterations, message=o.message,
                projected_gradient=o.projected_gradient,
                relative_objective_change=o.relative_objective_change,
                parameters=o.values.detach().cpu().numpy().tolist())


def flags(values, bounds):
    v, b = np.asarray(values), np.asarray(bounds)
    return np.minimum((v-b[:, 0])/(b[:, 1]-b[:, 0]), (b[:, 1]-v)/(b[:, 1]-b[:, 0])) < .01


def refit(result, candidates, *, slope=False, nodes=None, seed=42, reg_factor=1.,
          obs=None, starts=3, maxiter=MAXITER, low_strength_start_cap=None, wide=False):
    """Refit matched null/halo models, rescuing null from every halo macro.

    Zero strength is explicitly feasible. Halo hypotheses include the optimized
    null exactly; null rescue reduces optimizer-induced false improvements.
    """
    obs = obs or result.observation
    reference = result.observation.macro_model
    source_macro = result.macro_model or reference
    geometry = scan.with_macro(obs, source_macro)
    baseline = result.baseline
    problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, geometry.bx, geometry.by,
        source_grid=baseline.grid if nodes is None else None,
        source_nodes=nodes or len(baseline.coefficients), seed=seed,
        regularization=baseline.regularization*reg_factor, mesh_strategy='centroidal')
    model = FullMacroProblem(obs, problem, slope)
    bounds = macro_bounds(reference, slope, wide)
    null_starts = [vector(source_macro, slope)]
    null_starts += [vector(c.fit.macro_model or source_macro, slope) for c in candidates]
    v = vector(source_macro, slope)
    perturb = v.copy(); perturb[0] += .025; perturb[1] -= .025; perturb[2] += .06
    perturb[3:5] += [.03, -.03]
    null_starts.append(perturb)
    unique = []
    for s in null_starts:
        if not any(np.allclose(s, old, atol=1e-6) for old in unique): unique.append(s)
    nulls = [model.optimize(s, bounds, False, baseline.coefficients if nodes is None else None,
                           maxiter) for s in unique]
    null = choose(nulls)
    outcomes, logs = [], []
    for c in candidates:
        hb = bounds + [(c.seed[0]-HALO_RADIUS, c.seed[0]+HALO_RADIUS),
                       (c.seed[1]-HALO_RADIUS, c.seed[1]+HALO_RADIUS), (0., 1.)]
        nv = null.values.detach().cpu().numpy()
        old = vector(c.fit.macro_model or source_macro, slope)
        low_start = max(c.fit.strength,.02)
        if low_strength_start_cap is not None: low_start = min(low_start,low_strength_start_cap)
        hstarts = [np.r_[old, c.fit.x, c.fit.y, c.fit.strength/HALO_CAP],
                   np.r_[nv, c.seed, low_start/HALO_CAP],
                   np.r_[nv, c.fit.x+.04, c.fit.y-.04, .03/HALO_CAP]]
        attempts = [model.optimize(s, hb, True, c.fit.source.coefficients if nodes is None
                                  else None, maxiter) for s in hstarts[:starts]]
        # Exact zero-halo branch, evaluated using exactly the same source engine.
        z = np.r_[nv, c.fit.position, 0.]
        zero_bx, zero_by = model.rays(model.engine.tensor(z), True)
        zero_fit = model.engine.fit(zero_bx, zero_by, initial=null.source.coefficients)
        best = choose(attempts)
        # A halo fit may reveal a better macro basin for the null. Always retry it.
        rescue = model.optimize(best.values[:model.nmacro], bounds, False,
                                best.source.coefficients, maxiter)
        nulls.append(rescue); null = choose(nulls)
        outcomes.append((c, best, hb))
        logs.append(dict(rank=c.rank, attempts=[dict(**optimum_record(o),start=s.tolist())
                                                for s,o in zip(hstarts[:starts],attempts)],
                         exact_zero_objective=float(zero_fit.objective),
                         rescued_null=optimum_record(rescue)))
        print(f'  candidate {c.rank}: objective={float(best.source.objective):.7f}; '
              f'null={float(null.source.objective):.7f}', flush=True)
    # All candidates use the best null found anywhere, not a different null per peak.
    source = model.engine.export(null.source)
    null_macro = macro_at(null.values.detach().cpu().numpy(), slope)
    null_chi = float(np.sum(((obs.image-source.image_model)[obs.mask]/obs.rms[obs.mask])**2))
    fits_out, records = [], []
    for c, best, hb in outcomes:
        v = best.values.detach().cpu().numpy()
        if float(best.source.objective) >= float(null.source.objective):
            fitted = source; adjusted = null_macro; v[-1] = 0.; success = null.converged
            v[:model.nmacro] = null.values.detach().cpu().numpy()
        else:
            fitted = model.engine.export(best.source); adjusted = macro_at(v[:model.nmacro], slope)
            success = best.converged
        chi = float(np.sum(((obs.image-fitted.image_model)[obs.mask]/obs.rms[obs.mask])**2))
        f = HaloFit(float(v[-3]), float(v[-2]), float(v[-1]*HALO_CAP), null_chi-chi,
                    float(obs.mask.sum()*(source.objective-fitted.objective)), success,
                    fitted, fitted.image_model, adjusted, best.message, best.iterations)
        fits_out.append(replace(c, fit=f))
        at = flags(v, hb)
        records.append(dict(rank=c.rank, position=list(f.position), strength=f.strength,
            delta_chi2=f.delta_chi2, penalized_gain=f.delta_objective,
            converged=bool(success), null_converged=bool(null.converged),
            macro_at_bound=bool(at[:model.nmacro].any()),
            position_at_bound=bool(at[-3:-1].any()), strength_at_cap=bool(v[-1]>=.98),
            macro=adjusted))
    output = replace(result, observation=obs, baseline=source, candidates=tuple(fits_out),
                     macro_model=null_macro, macro_converged=null.converged,
                     max_strength=HALO_CAP, refinement_radius=HALO_RADIUS,
                     config={**result.config, 'execution': current_execution().to_dict(),
                             'joint_refinement': dict(free_macro_parameters=['theta_E','q','phi','cx','cy','g1','g2']
                                 + (['gamma'] if slope else [])),
                             'candidate_followup': dict(slope_free=slope, nodes=nodes, seed=seed,
                                 regularization_factor=reg_factor, maxiter=maxiter, halo_starts=starts,
                                 low_strength_start_cap=low_strength_start_cap,
                                 widened_macro_bounds=wide,
                                 null_rescued_from_each_halo=True)})
    record = dict(null=optimum_record(null), null_macro=null_macro,
        execution=current_execution().to_dict(),
        null_macro_at_bound=bool(flags(null.values.detach().cpu().numpy(), bounds).any()),
        null_attempts=[optimum_record(o) for o in nulls], candidates=records,
        candidate_attempts=logs, source_nodes=problem.size, regularization=problem.regularization,
        macro_bounds=bounds, halo_cap_arcsec=HALO_CAP, local_halo_radius_arcsec=HALO_RADIUS,
        independent_validation=False, calibrated_detection=False)
    return output, record


def vary_observation(obs, lens, *, psf_width=1., foreground_scale=1.):
    image = obs.image-(foreground_scale-1)*lens
    psf = obs.psf.copy()
    if psf_width != 1.:
        yy, xx = np.indices(psf.shape, dtype=float)
        cy, cx = (np.array(psf.shape)-1)/2
        psf = map_coordinates(psf, [(yy-cy)/psf_width+cy, (xx-cx)/psf_width+cx],
                              order=3, mode='constant', cval=0.)
        psf /= psf.sum()
    return LensingObservation.from_arrays(image, obs.rms, psf, pixel_scale=obs.pixel_scale,
        native_rays=obs.image_rays(), mask=obs.mask, macro_model=obs.macro_model, lens_id=obs.lens_id)


def save_fit(folder, result, record):
    folder.mkdir(parents=True, exist_ok=True)
    record={**record,'cpu_threads':torch.get_num_threads(),'detection':result.detection()}
    result.save_fits(folder/'result.fits', overwrite=True)
    result.save_json(folder/'result.json', overwrite=True)
    scan.write_json(folder/'metrics.json', record)


def numerical_check(result):
    """Float64 evaluation at frozen fitted parameters; no extra candidate selection."""
    obs = result.observation
    with torch_backend('cpu', dtype='float64'):
        base_obs = scan.with_macro(obs, result.macro_model)
        p = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, base_obs.bx, base_obs.by,
            source_grid=result.baseline.grid, regularization=result.baseline.regularization)
        baseline = p.fit(base_obs.bx, base_obs.by)
        rows = []
        for c in result.candidates:
            o = scan.with_macro(obs, c.fit.macro_model)
            x, y = [torch.tensor(a, dtype=torch.float64) for a in obs.image_rays()]
            ax, ay = truncated_subhalo_deflection(x, y, *c.fit.position, c.fit.strength)
            s = p.fit(o.bx-ax.numpy(), o.by-ay.numpy())
            chi0 = np.sum(((obs.image-baseline.image_model)[obs.mask]/obs.rms[obs.mask])**2)
            chi1 = np.sum(((obs.image-s.image_model)[obs.mask]/obs.rms[obs.mask])**2)
            rows.append(dict(rank=c.rank, delta_chi2_cpu64=float(chi0-chi1),
                             delta_chi2_fitted=c.fit.delta_chi2,
                             delta_chi2_difference=float(chi0-chi1-c.fit.delta_chi2)))
        return dict(fixed_parameter_evaluation_only=True, candidates=rows)


def other_viable_ranks(record,best_rank):
    if record['null_macro_at_bound'] or not record['null']['converged']:return []
    return [c['rank'] for c in record['candidates'] if c['rank']!=best_rank
            and c['delta_chi2']>=25 and c['penalized_gain']>0 and c['converged']
            and not (c['macro_at_bound'] or c['position_at_bound'] or c['strength_at_cap'])]


def variant_options(base,lens):
    return [
        ('mesh600_seed11',dict(nodes=600,seed=11)),
        ('mesh800_seed33',dict(nodes=800,seed=33)),
        ('lambda_x10',dict(reg_factor=10.)),
        ('free_epl_slope',dict(slope=True)),
        ('psf_width_minus5pct',dict(obs=vary_observation(base.observation,lens,psf_width=.95))),
        ('psf_width_plus5pct',dict(obs=vary_observation(base.observation,lens,psf_width=1.05))),
        ('foreground_minus2pct',dict(obs=vary_observation(base.observation,lens,foreground_scale=.98))),
        ('foreground_plus2pct',dict(obs=vary_observation(base.observation,lens,foreground_scale=1.02))),
    ]


def run_variants(base,folder,lens,label,early_gate=False):
    best=base.best_candidate
    core_failed=False
    for index,(name,options) in enumerate(variant_options(base,lens)):
        target=folder/name
        if not (target/'metrics.json').exists():
            fitted,record=refit(base,[best],starts=2,maxiter=100,low_strength_start_cap=.06,**options)
            save_fit(target,fitted,record)
            print(f'DONE {label} {name} dchi={fitted.best_candidate.fit.delta_chi2:.2f}',flush=True)
            del fitted
            gc.collect()
            if current_execution().device=='mps':torch.mps.empty_cache()
        else:record=json.loads((target/'metrics.json').read_text())
        c=record['candidates'][0]
        shift=np.linalg.norm(np.array(c['position'])-best.fit.position)
        ratio=c['strength']/best.fit.strength if best.fit.strength>0 else 0.
        if index<4:
            core_failed |= bool(c['delta_chi2']<25 or c['penalized_gain']<=0
                or not c['converged'] or not c['null_converged'] or record['null_macro_at_bound']
                or c['macro_at_bound'] or c['position_at_bound'] or c['strength_at_cap']
                or shift>.15 or not .5<=ratio<=1.5)
        if early_gate and index<4 and core_failed:
            scan.write_json(folder/'stability_stopping.json',dict(
                reason='alternative candidate failed predeclared source/macro stability checks',
                core_variants_completed=index+1,remaining_trials_not_required_after_stability_failure=True,
                this_does_not_rule_out_a_physical_halo=True))
            break


def worker(arguments):
    identifier, stage, device = arguments
    torch.set_num_threads(int(os.environ.get('SUBLENS_FOLLOWUP_CPU_THREADS','1')) if device=='cpu' else 1)
    folder = OUT/identifier
    folder.mkdir(exist_ok=True, parents=True)
    start = perf_counter()
    try:
        original = load_result(SCAN/identifier/'result.fits')
        selection = json.loads((OUT/'selected_candidates.json').read_text())
        item = next(r for r in selection if r['identifier']==identifier)
        ranks = {c['rank'] for c in item['review_candidates']}
        selected = [c for c in original.candidates if c.rank in ranks]
        with torch_backend(device):
            if stage == 'base':
                target = folder/'full_sie'
                if not (target/'metrics.json').exists():
                    fitted, record = refit(original, selected)
                    record['seconds'] = perf_counter()-start
                    record['cpu64_check'] = numerical_check(fitted)
                    save_fit(target, fitted, record)
                    print(f'DONE {identifier} full_sie {record["seconds"]:.1f}s', flush=True)
            elif stage == 'alternate':
                base=load_result(folder/'full_sie/result.fits')
                metrics=json.loads((folder/'full_sie/metrics.json').read_text())
                manifest=json.loads((SCAN/'manifest.json').read_text())
                row=next(r for r in manifest if r['identifier']==identifier)
                lens=fits.getdata(scan.DATA/row['tile_id']/'model.fits','VIS_GALAXY_0').astype(float)
                for rank in other_viable_ranks(metrics,base.best_candidate.rank):
                    candidate=next(c for c in base.candidates if c.rank==rank)
                    track=folder/f'alternative_rank_{rank}'
                    alternate=replace(base,candidates=(candidate,))
                    if not (track/'full_sie/metrics.json').exists():
                        record={**metrics,'candidates':[c for c in metrics['candidates'] if c['rank']==rank]}
                        save_fit(track/'full_sie',alternate,record)
                    run_variants(alternate,track,lens,f'{identifier} rank{rank}',early_gate=True)
            elif stage == 'wide':
                base=load_result(folder/'full_sie/result.fits')
                metrics=json.loads((folder/'full_sie/metrics.json').read_text())
                best=base.best_candidate
                best_row=next(c for c in metrics['candidates'] if c['rank']==best.rank)
                if metrics['null_macro_at_bound'] or best_row['macro_at_bound']:
                    target=folder/'wide_macro_bounds'
                    if not (target/'metrics.json').exists():
                        fitted,record=refit(base,[best],starts=3,wide=True,low_strength_start_cap=.06)
                        save_fit(target,fitted,record)
                        print(f'DONE {identifier} wide_macro_bounds dchi={fitted.best_candidate.fit.delta_chi2:.2f}',flush=True)
            elif stage == 'stability':
                base = load_result(folder/'full_sie/result.fits')
                best = base.best_candidate
                manifest = json.loads((SCAN/'manifest.json').read_text())
                row = next(r for r in manifest if r['identifier']==identifier)
                lens = fits.getdata(scan.DATA/row['tile_id']/'model.fits','VIS_GALAXY_0').astype(float)
                run_variants(base,folder,lens,identifier)
        return dict(identifier=identifier, stage=stage, status='completed', seconds=perf_counter()-start)
    except Exception as e:
        traceback.print_exc()
        failure=dict(identifier=identifier,stage=stage,status='failed',error=str(e))
        scan.write_json(folder/f'{stage}_failure.json',failure)
        return failure


def summarize_system(identifier,folder=None):
    folder=folder or OUT/identifier
    base_path=folder/'full_sie/metrics.json'
    if not base_path.exists():return None
    base=json.loads(base_path.read_text())
    chosen=max(base['candidates'],key=lambda c:c['penalized_gain'])
    variants=[]
    for path in sorted(folder.glob('*/metrics.json')):
        r=json.loads(path.read_text())
        c=next((c for c in r['candidates'] if c['rank']==chosen['rank']),None)
        if c is None:continue
        distance=float(np.linalg.norm(np.array(c['position'])-chosen['position']))
        ratio=c['strength']/chosen['strength'] if chosen['strength']>0 else None
        variants.append(dict(variant=path.parent.name,**c,
            null_macro_at_bound=r['null_macro_at_bound'],
            shift_from_full_sie_arcsec=distance,strength_ratio=ratio,
            chi2_macro=r.get('residual_macro',{}).get('chi2_per_pixel'),
            chi2_halo=r.get('residual_halo',{}).get('chi2_per_pixel'),
            bright_pixel_chi2_macro=r.get('image_scores_macro',{}).get('bright_arc_chi2_per_pixel'),
            bright_pixel_chi2_halo=r.get('image_scores_halo',{}).get('bright_arc_chi2_per_pixel')))
    variants.sort(key=lambda v:VARIANT_ORDER.index(v['variant']))
    # These fixed descriptive tolerances assess numerical/model stability only.
    # They are not an evidence threshold or a calibrated selection function.
    required={'full_sie','mesh600_seed11','mesh800_seed33','lambda_x10','free_epl_slope',
              'psf_width_minus5pct','psf_width_plus5pct','foreground_minus2pct','foreground_plus2pct'}
    early_stopped=(folder/'stability_stopping.json').exists()
    if early_stopped:
        stop=json.loads((folder/'stability_stopping.json').read_text())
        core=['mesh600_seed11','mesh800_seed33','lambda_x10','free_epl_slope']
        required={'full_sie',*core[:stop['core_variants_completed']]}
    if base['null_macro_at_bound'] or chosen['macro_at_bound']:required.add('wide_macro_bounds')
    complete=required <= {v['variant'] for v in variants}
    def passes(v):
        return bool(v['delta_chi2']>=25 and v['penalized_gain']>0
            and v['converged'] and v['null_converged']
            and not v['macro_at_bound'] and not v['null_macro_at_bound']
            and not v['position_at_bound'] and not v['strength_at_cap']
            and v['shift_from_full_sie_arcsec']<=.15
            and v['strength_ratio'] is not None and .5<=v['strength_ratio']<=1.5)
    for v in variants:
        v['passes_descriptive_stability_checks']=passes(v)
        reasons=[]
        if v['delta_chi2']<25 or v['penalized_gain']<=0:reasons.append('below_screening_gain')
        if not v['converged'] or not v['null_converged']:reasons.append('optimizer_not_converged')
        if v['macro_at_bound'] or v['null_macro_at_bound']:reasons.append('macro_parameter_boundary')
        if v['position_at_bound']:reasons.append('halo_position_boundary')
        if v['strength_at_cap']:reasons.append('halo_strength_cap')
        if v['shift_from_full_sie_arcsec']>.15:reasons.append('position_shift_over_0p15_arcsec')
        if v['strength_ratio'] is None or not .5<=v['strength_ratio']<=1.5:reasons.append('strength_ratio_outside_0p5_to_1p5')
        v['stability_failure_reasons']=reasons
    original=next(r for r in json.loads((OUT/'selected_candidates.json').read_text()) if r['identifier']==identifier)
    record=dict(identifier=identifier,inspection=original['inspection'],best_rank=chosen['rank'],
        analysis_directory=str(folder),alternative_track=folder!=OUT/identifier,
        sensitivity_stopped_after_core_failure=early_stopped,
        original_candidates=original['review_candidates'],full_sie_candidates=base['candidates'],
        variants=variants,all_variants_complete=complete,
        passes_all_descriptive_stability_checks=complete and not early_stopped and all(passes(v) for v in variants),
        source_macro_stable=all(passes(v) for v in variants if v['variant'] in
            {'full_sie','mesh600_seed11','mesh800_seed33','lambda_x10','free_epl_slope'})
            and {'mesh600_seed11','mesh800_seed33','lambda_x10','free_epl_slope'}<={v['variant'] for v in variants},
        confirmed_subhalo=None,calibrated_significance=False)
    scan.write_json(folder/'summary.json',record)
    return record


def profile_strength(result):
    """Conditional strength scan: host macro and halo position held fixed.

    Only the source is reprofiled. This is a diagnostic, NOT a confidence interval
    or the joint likelihood profile, and does not replace the matched null fit.
    """
    best=result.best_candidate;obs=result.observation
    host=scan.with_macro(obs,best.fit.macro_model)
    problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,host.bx,host.by,
        source_grid=result.baseline.grid,regularization=result.baseline.regularization)
    x,y=[torch.tensor(a,dtype=torch.float64) for a in obs.image_rays()]
    strengths=np.unique(np.r_[np.linspace(0,min(HALO_CAP,max(.08,2*best.fit.strength)),21),best.fit.strength])
    rows=[]
    for b in strengths:
        ax,ay=truncated_subhalo_deflection(x,y,*best.fit.position,b)
        s=problem.fit(host.bx-ax.numpy(),host.by-ay.numpy())
        rows.append(dict(strength=float(b),penalized_objective_times_n=float(obs.mask.sum()*s.objective)))
    return dict(host_macro_and_position_fixed=True,source_reprofiled=True,confidence_interval=False,points=rows)


def make_plots(identifiers):
    import shutil
    from sublens.survey.euclid_dr1 import load_euclid_dr1_lens
    manifest=json.loads((SCAN/'manifest.json').read_text())
    summaries=[]
    tracks=[(identifier,folder) for identifier in identifiers
            for folder in [OUT/identifier,*sorted((OUT/identifier).glob('alternative_rank_*'))]]
    for identifier,folder in tracks:
        summary=summarize_system(identifier,folder)
        if summary is None:continue
        summaries.append(summary)
        result=load_result(folder/'full_sie/result.fits')
        obs=result.observation;best=result.best_candidate;halo=best.fit
        row=next(r for r in manifest if r['identifier']==identifier)
        # Published reconstruction is opened only after every inference has frozen.
        data=load_euclid_dr1_lens(scan.DATA/row['tile_id'],oversample=4)
        data=replace(data,observation=obs)
        pilot.OUTPUT=folder
        for name in ('official_reference.fits','psf_diagnostics.json'):
            shutil.copy2(SCAN/identifier/name,folder/name)
        sources,extent,_=pilot.source_maps(data,result)
        old=json.loads((SCAN/identifier/'metrics.json').read_text())
        reference=fits.getdata(folder/'official_reference.fits','REFERENCE_SOURCE')
        x,y=obs.image_rays();radius=np.hypot(x.mean(-1),y.mean(-1))
        arc=obs.mask&(obs.image>=3*obs.rms)&(radius>.2)
        image_scores={k:scan.image_scores(obs,m,arc) for k,m in
            [('official',reference),('macro',result.baseline.image_model),('halo',halo.image_model)]}
        metrics=dict(theta_official_maximum_likelihood=old['theta_official_maximum_likelihood'],
            theta_macro=scan.effective_sie_radius(result.macro_model),image_scores=image_scores)
        pilot.plot_comparison(row['catalogue_row'],data,result,sources,extent,metrics)
        for sub in folder.glob('*/metrics.json'):
            fitted=load_result(sub.parent/'result.fits')
            r=json.loads(sub.read_text())
            r.update(residual_macro=residual_metrics(fitted.observation,fitted.baseline.image_model),
                     residual_halo=residual_metrics(fitted.observation,fitted.best_candidate.fit.image_model),
                     image_scores_macro=scan.image_scores(fitted.observation,fitted.baseline.image_model,arc),
                     image_scores_halo=scan.image_scores(fitted.observation,fitted.best_candidate.fit.image_model,arc),
                     execution=fitted.config.get('execution'))
            scan.write_json(sub,r)
        with torch_backend('cpu',dtype='float64'):
            profile=profile_strength(result)
        scan.write_json(folder/'conditional_strength_profile.json',profile)
        comprehensive_plot(identifier,result,reference,sources,extent,summary,profile,image_scores,folder)
        scan.write_json(folder/'image_scores.json',image_scores)
        pixel=np.array([[halo.x/obs.pixel_scale+(obs.image.shape[1]-1)/2,
                         halo.y/obs.pixel_scale+(obs.image.shape[0]-1)/2]])
        ra,dec=scan.WCS(data.header).all_pix2world(pixel,0)[0]
        scan.write_json(folder/'candidate_coordinates.json',dict(rank=best.rank,
            x_west_arcsec=halo.x,y_north_arcsec=halo.y,ra_deg=float(ra),dec_deg=float(dec),
            point_estimate_only=True,subhalo_existence_confirmed=None))
        print(f'PLOTTED {identifier}',flush=True)
    summaries=[summarize_system(r['identifier'],Path(r['analysis_directory'])) for r in summaries]
    scan.write_json(OUT/'summary.json',dict(candidate_tracks=summaries,
        completed_systems=len(set(r['identifier'] for r in summaries)),
        complete_stability_tracks=sum(r['all_variants_complete'] for r in summaries),
        passes_all_descriptive_stability_checks=sum(r['passes_all_descriptive_stability_checks'] for r in summaries),
        confirmed_subhalos=None,confirmation_status='not assessed by calibrated detection analysis'))
    overview_plot(summaries)
    write_report(summaries)


def comprehensive_plot(identifier,result,reference,sources,source_extent,summary,profile,scores,folder):
    obs=result.observation;mask=obs.mask;best=result.best_candidate;f=best.fit
    fig,axes=plt.subplots(4,4,figsize=(21,20),layout='constrained');axes=axes.ravel()
    ex=np.array([-obs.image.shape[1]/2,obs.image.shape[1]/2,-obs.image.shape[0]/2,obs.image.shape[0]/2])*obs.pixel_scale
    limit=min(min(obs.image.shape)*obs.pixel_scale/2,max(2.,1.7*result.observation.einstein_radius))
    flux=max(float(np.percentile(obs.image[mask],99.5)),float(np.median(obs.rms[mask])))
    norm=AsinhNorm(linear_width=.03*flux,vmin=0,vmax=flux)
    r0=(obs.image-result.baseline.image_model)/obs.rms;r1=(obs.image-f.image_model)/obs.rms
    def im(index,array,title,*,cmap='magma',norm=None,vmin=None,vmax=None,label='ADU/s',mark=False):
        a=axes[index].imshow(np.where(mask,array,np.nan),origin='lower',extent=ex,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
        axes[index].set(title=title,xlim=(-limit,limit),ylim=(-limit,limit),xlabel='x west [arcsec]',ylabel='y north [arcsec]')
        fig.colorbar(a,ax=axes[index],shrink=.7,label=label)
        if mark:axes[index].plot(f.x,f.y,'*',ms=14,mfc='#ffe600',mec='black')
    im(0,obs.image,'1. Masked data − supplied lens light',norm=norm)
    ps=obs.psf;psex=np.array([-ps.shape[1]/2,ps.shape[1]/2,-ps.shape[0]/2,ps.shape[0]/2])*obs.pixel_scale
    a=axes[1].imshow(ps,origin='lower',extent=psex,cmap='magma',norm=SymLogNorm(linthresh=1e-6,vmin=min(-1e-6,ps.min()),vmax=ps.max()))
    axes[1].set(title='2. Supplied VIS PSF',xlabel='x [arcsec]',ylabel='y [arcsec]');fig.colorbar(a,ax=axes[1],shrink=.7,label='Normalized kernel')
    im(2,(obs.image-reference)/obs.rms,'3. Re-rendered released-source residual',cmap='RdBu_r',vmin=-5,vmax=5,label='Residual / RMS')
    im(3,r0,f'4. Full SIE+shear null residual\nχ²/N={scores["macro"]["chi2_per_pixel"]:.3f}',cmap='RdBu_r',vmin=-5,vmax=5,label='Residual / RMS')
    im(4,result.baseline.image_model,'5. Full macro-only source image',norm=norm)
    im(5,f.image_model,'6. Macro + candidate source image',norm=norm,mark=True)
    im(6,r1,f'7. Candidate residual\nχ²/N={scores["halo"]["chi2_per_pixel"]:.3f}',cmap='RdBu_r',vmin=-5,vmax=5,label='Residual / RMS',mark=True)
    d=r0*r0-r1*r1;scale=max(1.,float(np.percentile(abs(d[mask]),98)))
    im(7,d,'8. Per-pixel χ² improvement (all refits)',cmap='RdBu_r',vmin=-scale,vmax=scale,label='Null χ² − halo χ²',mark=True)
    vmax=max(1e-15,max(float(np.percentile(sources[k],99.8)) for k in sources))
    snorm=AsinhNorm(linear_width=.03*vmax,vmin=0,vmax=vmax)
    for i,k,t in [(8,'OFFICIAL','9. Released source (comparison only)'),(9,'MACRO_ONLY','10. Our full-macro source'),(10,'MACRO_HALO','11. Our source with candidate')]:
        a=axes[i].imshow(sources[k],origin='lower',extent=source_extent,cmap='magma',norm=snorm)
        axes[i].set(title=t,xlabel='βx west [arcsec]',ylabel='βy north [arcsec]');fig.colorbar(a,ax=axes[i],shrink=.7,label='Source intensity')
    k=_smooth_convergence(result.residual_potential.convergence,obs.pixel_scale);coverage=result.residual_potential.mask
    scale=max(1e-4,float(np.percentile(abs(k[coverage]),99))) if coverage.any() else 1.
    im(11,np.where(coverage,k,np.nan),'12. Original PINN δκ; seeds and refined fit',cmap='RdBu_r',vmin=-scale,vmax=scale,label='δκ',mark=True)
    for c in result.candidates:
        axes[11].plot(*c.seed,'o',mfc='none',mec='black',ms=7)
        axes[11].annotate(str(c.rank),c.seed,xytext=(3,3),textcoords='offset points')
    variants=summary['variants'];colors=plt.cm.tab10(np.arange(len(variants))%10)
    for v,color in zip(variants,colors):
        axes[12].scatter(*v['position'],s=45,color=color,label=v['variant'])
    axes[12].plot(f.x,f.y,'*',ms=16,mfc='#ffe600',mec='black')
    axes[12].set(title='13. Refit positions under model changes',xlabel='x west [arcsec]',ylabel='y north [arcsec]')
    axes[12].xaxis.set_major_locator(MaxNLocator(4));axes[12].yaxis.set_major_locator(MaxNLocator(4))
    axes[12].legend(fontsize=7,loc='best');axes[12].grid(alpha=.2)
    values=[v['delta_chi2'] for v in variants]
    axes[13].barh(np.arange(len(variants)),values,color=colors)
    axes[13].set_yticks(np.arange(len(variants)),[v['variant'].replace('_',' ') for v in variants],fontsize=8)
    axes[13].axvline(25,color='black',ls='--',lw=1)
    axes[13].set(title='14. Matched-null gains; 25 = screening rule',xlabel='Δχ² (uncalibrated)')
    p=profile['points'];loss=np.array([r['penalized_objective_times_n'] for r in p]);loss-=loss.min()
    axes[14].plot([r['strength'] for r in p],loss,'o-',ms=3,color='#0072B2')
    axes[14].axvline(f.strength,color='black',ls='--')
    axes[14].set(title='15. Conditional strength scan; source refit\nHost / position fixed; no confidence interval',xlabel='Strength b [arcsec]',ylabel='N × penalized objective − minimum')
    bins=np.linspace(-8,8,65)
    for r,label,color in [(r0,'Full macro','#0072B2'),(r1,'Macro + candidate','#D55E00')]:
        axes[15].hist(r[mask],bins=bins,density=True,histtype='step',lw=1.5,label=label,color=color)
    axis=np.linspace(-8,8,250);axes[15].plot(axis,np.exp(-axis**2/2)/np.sqrt(2*np.pi),'k--',lw=1,label='Unit Gaussian')
    axes[15].set(title='16. Residual distribution, same mask / RMS',xlabel='Residual / RMS',ylabel='Density');axes[15].legend(fontsize=8)
    fig.suptitle(f'{identifier} / candidate {best.rank} · Grade A / released inspection: {summary["inspection"]}\n'
        f'Full SIE refinement: b={f.strength:.4f}″, (x,y)=({f.x:+.3f}, {f.y:+.3f})″, Δχ²={f.delta_chi2:.1f} · '
        'competing single-halo hypothesis; existence unconfirmed',fontsize=17)
    fig.savefig(folder/'candidate_comprehensive.png',dpi=160,bbox_inches='tight')
    fig.savefig(folder/'candidate_comprehensive.pdf',bbox_inches='tight');plt.close(fig)


def overview_plot(summaries):
    if not summaries:return
    names=[n for n in VARIANT_ORDER if any(v['variant']==n for r in summaries for v in r['variants'])]
    a=np.full((len(summaries),len(names)),np.nan)
    fig,ax=plt.subplots(figsize=(16,max(5,len(summaries)*.6)),layout='constrained')
    for i,r in enumerate(summaries):
        for v in r['variants']:a[i,names.index(v['variant'])]=v['delta_chi2']
    scale=max(25,float(np.nanpercentile(a,95)))
    im=ax.imshow(a,cmap='viridis',vmin=0,vmax=scale,aspect='auto')
    for i,r in enumerate(summaries):
        for j in range(len(names)):
            if not np.isfinite(a[i,j]):ax.text(j,i,'—',ha='center',va='center',fontsize=9,color='grey')
        for v in r['variants']:
            j=names.index(v['variant']);flag='' if v['passes_descriptive_stability_checks'] else '*'
            ax.text(j,i,f'{v["delta_chi2"]:.1f}{flag}',ha='center',va='center',fontsize=8,
                    color='white' if v['delta_chi2']<.45*scale else 'black')
    ax.set_xticks(range(len(names)),[n.replace('_',' ') for n in names],rotation=30,ha='right',fontsize=9)
    ax.set_yticks(range(len(summaries)),[r['identifier']+f' / rank {r["best_rank"]} / '+r['inspection'] for r in summaries],fontsize=9)
    ax.set_title('Candidate follow-up: matched-null Δχ² under model changes\n* = fails a stated stability check; — = not run / not required; not a detection matrix')
    fig.colorbar(im,ax=ax,label='Δχ² (not calibrated significance)')
    fig.savefig(OUT/'candidate_stability_matrix.png',dpi=180);fig.savefig(OUT/'candidate_stability_matrix.pdf');plt.close(fig)


def write_report(summaries):
    trial_columns=['identifier','inspection','variant','rank','x_arcsec','y_arcsec','strength_arcsec',
        'delta_chi2','penalized_gain','converged','null_converged','macro_at_bound','null_macro_at_bound',
        'position_at_bound','strength_at_cap','shift_arcsec','strength_ratio','chi2_macro','chi2_halo',
        'bright_pixel_chi2_macro','bright_pixel_chi2_halo','passes_descriptive_checks']
    with (OUT/'stability_trials.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=trial_columns);writer.writeheader()
        for r in summaries:
            for v in r['variants']:
                row={k:v.get(k) for k in trial_columns}
                row.update(identifier=r['identifier'],inspection=r['inspection'],variant=v['variant'],
                    x_arcsec=v['position'][0],y_arcsec=v['position'][1],strength_arcsec=v['strength'],
                    shift_arcsec=v['shift_from_full_sie_arcsec'],
                    passes_descriptive_checks=v['passes_descriptive_stability_checks'])
                writer.writerow(row)
    columns=['identifier','inspection','rank','original_delta_chi2','full_sie_delta_chi2',
             'x_arcsec','y_arcsec','strength_arcsec','converged','macro_at_bound','position_at_bound','strength_at_cap']
    with (OUT/'full_sie_candidates.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for r in summaries:
            if r['alternative_track']:continue
            for c in r['full_sie_candidates']:
                old=next(o for o in r['original_candidates'] if o['rank']==c['rank'])
                row={k:c.get(k) for k in columns}
                row.update(identifier=r['identifier'],inspection=r['inspection'],
                    original_delta_chi2=old['delta_chi2'],full_sie_delta_chi2=c['delta_chi2'],
                    x_arcsec=c['position'][0],y_arcsec=c['position'][1],strength_arcsec=c['strength'])
                writer.writerow(row)
    lines=['# DR1 Grade A candidate follow-up','',
        'Frozen candidate list: `selected_candidates.json`. This follow-up does not use released mass or source maps for fitting or selection. All released source comparisons are computed after inference. Every hypothesis profiles its own positive source coefficients on exactly the same grid and regularization as its matched null. The full seven-parameter SIE is refitted, with a separate eight-parameter EPL slope sensitivity test.', '',
        'The initial full-SIE run rechecks every originally flagged position with three halo starts, multiple null starts, and null rescue from each fitted halo macro. The exact zero-strength branch is allowed. The source grid is fixed within a comparison. Eight fixed stability variants follow the strongest full-SIE alternative per system. Other candidates with Δχ²≥25, converged null/halo fits and no null-macro, halo-macro, position or strength-cap boundary are also followed. The same four source/macro checks are evaluated in order; once any required stability check fails, the remaining trials can be omitted without changing the failed conjunctive stability verdict. The alternative remains unresolved; a physical halo is not ruled out. These are competing single-halo fits, not simultaneous detections.', '',
        'Sensitivity ranges (PSF width ±5%, supplied foreground amplitude ±2%) are declared perturbation experiments, not measured uncertainty distributions. Two new null-geometry centroidal meshes use 600 nodes/seed 11 and 800 nodes/seed 33. Lambda ×10 tests regularization sensitivity; lambda is not chosen by similarity to the released source. Systems hitting a full-SIE macro boundary also receive a wider-bounds comparison: theta_E factors 0.5–1.6, q≥0.12, angle ±π/2, centre ±0.35 arcsec, shear-component offsets ±0.25 (each component limited to ±0.5). This deliberately broad sensitivity control can expose dependence on an inadequate single-host macro; it is not an observational prior measurement.', '',
        'Descriptive stability checks: Δχ²≥25 and positive penalized gain; null and halo optimizers converge; no macro/local-position/strength-cap boundary; position shift ≤0.15 arcsec and strength ratio 0.5–1.5 relative to the full-SIE fit. These tolerances are screening diagnostics, not Bayes factors, p values, posterior uncertainties, or a detection rule. The full discovery procedure has not been calibrated against realistic null observations. Confirmed-subhalo status remains unassessed (`null`), not an inferred count of zero.', '',
        'All image panels use lens-light-subtracted data or source-only forward models. The released reference residual is re-rendered from its source CSV and maximum-likelihood macro with the same supplied foreground/PSF, not claimed to be the original official pixelized residual. Residual color scales are shared. The conditional strength scan fixes host macro and halo position and refits only source brightness; it is not a joint likelihood profile or confidence interval.', '',
        '| Lens | Released inspection | Full-SIE Δχ² | b [arcsec] | Min variant Δχ² | All descriptive checks |','|---|---|---:|---:|---:|---|']
    for r in summaries:
        b=next(c for c in r['full_sie_candidates'] if c['rank']==r['best_rank'])
        link=Path(r['analysis_directory']).relative_to(OUT)/'candidate_comprehensive.png'
        lines.append(f'| [{r["identifier"]} / rank {r["best_rank"]}]({link}) | {r["inspection"]} | {b["delta_chi2"]:.2f} | {b["strength"]:.5f} | {min(v["delta_chi2"] for v in r["variants"]):.2f} | {r["passes_all_descriptive_stability_checks"]} |')
    lines+=['','Run with the workspace Python:', '```',
        '.venv/bin/python sublens_testing/euclid_dr1_candidate_followup.py base --device mps --workers 2',
        '.venv/bin/python sublens_testing/euclid_dr1_candidate_followup.py stability --device mps --workers 2',
        '.venv/bin/python sublens_testing/euclid_dr1_candidate_followup.py wide --device mps --workers 2',
        '.venv/bin/python sublens_testing/euclid_dr1_candidate_followup.py alternate --device mps --workers 2',
        '.venv/bin/python sublens_testing/euclid_dr1_candidate_followup.py plot','```','',
        'Outputs are separate from the original scan. Full-SIE and all sensitivity variants have result FITS, JSON parameters, objective/optimizer logs, and matched-null metrics. The original full-scan checkpoints are preserved. The `STOP` file in the scan directory prevents new scan objects from starting while follow-up uses the machine.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


def write_simple_results(identifiers, *, min_delta_chi2=25.):
    """Export the nominal full-SIE fits without reinterpreting old diagnostics."""
    rows=[];decisions=[]
    for identifier in identifiers:
        source=OUT/identifier/'full_sie/metrics.json'
        row=dict(identifier=identifier,status='pending_full_refinement',detected=None,
                 x_arcsec=None,y_arcsec=None,strength_arcsec=None,delta_chi2=None)
        if not source.exists() and (OUT/identifier/'base_failure.json').exists():
            row['status']='failed_full_refinement'
        if source.exists():
            data=json.loads(source.read_text())
            decision=summarize_subhalo({'lens_id':identifier,'candidates':data['candidates']},
                                       min_delta_chi2=min_delta_chi2)
            row.update(status='completed',detected=decision['detected'],
                       strength_arcsec=decision['strength_arcsec'],delta_chi2=decision['delta_chi2'])
            if decision['detected']:row['x_arcsec'],row['y_arcsec']=decision['position_arcsec']
            decisions.append(dict(**decision,source_metrics=str(source.relative_to(ROOT)),
                                  refinement_stage='full_sie'))
        rows.append(row)
    folder=OUT/'fit_improvement';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'subhalo_results.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else
            ['identifier','status','detected','x_arcsec','y_arcsec','strength_arcsec','delta_chi2'])
        writer.writeheader();writer.writerows(rows)
    scan.write_json(folder/'results.json',dict(
        decision_rule=dict(name='fit_improvement',min_delta_chi2=float(min_delta_chi2),positive_penalized_gain=True,
                           stability_required=False,significance_calibrated=False),
        source_stage='nominal full_sie only; sensitivity variants do not select or veto a result',
        completed=sum(r['status']=='completed' for r in rows),
        detected=sum(r['detected'] is True for r in rows),records=rows,decisions=decisions))
    lines=['# 子晕简表：按精修后的拟合改善判断','',
        f'判据：完整SIE联合精修后 Δχ²≥{min_delta_chi2:g}，且正则化目标改善为正。'
        '每个系统在满足规则的候选中选择正则化目标改善最大的解，输出一个位置和强度。'
        '稳定性、收敛和边界标记保留在原结果中，不作为本表的否决条件。', '',
        '“检出”指满足本模型规则，未校准为真实子晕的检出显著性；“未检出”不等于证明物理上不存在。'
        '本表固定采用 nominal full_sie 结果，不在不同网格、PSF、斜率或宽边界对照中择优。'
        '原稳定性试验及其结论保持不变。位置和强度都是点估计，单位为角秒。','',
        '| 系统 | 检出 | x | y | b | Δχ² |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        flag='是' if r['detected'] is True else '否' if r['detected'] is False else '未完成'
        values=[f'{r[k]:.4f}' if r[k] is not None else '—' for k in
                ['x_arcsec','y_arcsec','strength_arcsec','delta_chi2']]
        lines.append('| '+ ' | '.join([r['identifier'],flag,*values])+' |')
    (folder/'README_ZH.md').write_text('\n'.join(lines)+'\n')
    print(f"Compact results: {sum(r['detected'] is True for r in rows)}/{sum(r['status']=='completed' for r in rows)} "
          f"meet the fit-improvement rule; {folder/'subhalo_results.csv'}",flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['run','base','summary','stability','wide','alternate','plot'],nargs='?',default='run')
    p.add_argument('--device',default='mps')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--ids',nargs='*')
    p.add_argument('--min-delta-chi2',type=float,default=25.)
    args=p.parse_args()
    if not np.isfinite(args.min_delta_chi2) or args.min_delta_chi2<=0:
        p.error('--min-delta-chi2 must be a positive finite number')
    selected=json.loads((OUT/'selected_candidates.json').read_text())
    identifiers=[r['identifier'] for r in selected if not args.ids or r['identifier'] in args.ids]
    if args.stage=='summary':
        write_simple_results(identifiers,min_delta_chi2=args.min_delta_chi2)
        return
    if args.stage=='plot':
        make_plots(identifiers)
        return
    if args.stage=='run':
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            tasks=[pool.submit(worker,(i,'base',args.device)) for i in identifiers]
            for future in as_completed(tasks):print(json.dumps(future.result()),flush=True)
        write_simple_results(identifiers,min_delta_chi2=args.min_delta_chi2)
        return
    scan.write_json(OUT/'protocol.json',dict(
        frozen_selection='selected_candidates.json',selected_systems=len(selected),
        input_search='original PINN search, no new data or published-source tuning',
        full_sie_free=['theta_E','q','phi','center_x','center_y','shear_1','shear_2'],
        epl_slope_test=[1.6,2.4],halo_cap_arcsec=HALO_CAP,halo_radius_arcsec=HALO_RADIUS,
        null_and_halo_use_same_mesh_regularization_psf_mask_rms=True,
        source_reprofiled_at_every_optimizer_evaluation=True,
        null_starts='baseline, each candidate host macro, deterministic perturbation',
        halo_starts=3,null_rescue_from_each_halo=True,exact_zero_halo_allowed=True,
        stability='strongest full-SIE candidate per system; 600/800 nodes, lambda x10, free slope, PSF width +/-5%, foreground flux +/-2%',
        alternatives='Other positive-gain candidates with delta_chi2>=25, converged null/halo and no bound flags; four source/macro checks in order, short-circuit once any fails; then PSF/foreground only if all four pass',
        nuisance_ranges_are_sensitivity_experiments_not_measured_uncertainties=True,
        production_defaults_modified=False,independent_validation=False,
        null_false_positive_calibration=False,subhalo_existence_confirmed=None))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks=[pool.submit(worker,(i,args.stage,args.device)) for i in identifiers]
        for future in as_completed(tasks): print(json.dumps(future.result()),flush=True)


if __name__=='__main__':main()
