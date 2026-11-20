"""Full macro + compact-halo + source refinement on one matched Delaunay basis."""
from dataclasses import replace
import numpy as np
import torch
from ..execution import current_execution
from .source_plane import PixelSourceProblem
from .torch_source import torch_problem
from .torch_refinement import minimize_profiled
from .halo_search import HaloFit
from .macro_parameters import sie_alpha, vector, macro_at, macro_bounds, bounded_start
from ..profiles.macro import epl_deflection
from ..profiles.subhalo import truncated_subhalo_deflection


class FullMacroProblem:
    """Matched source-profile objective; every continuous parameter uses Torch."""
    def __init__(self, obs, problem, slope=False, max_strength=.30):
        self.obs, self.problem, self.slope = obs, problem, slope
        self.max_strength = max_strength
        self.retry_failed = True
        self.fixed_slope = float(obs.macro_model.get("gamma", 2.))
        self.engine = torch_problem(problem)
        self.x, self.y = [self.engine.tensor(a) for a in obs.image_rays()]
        self.coordinates = torch.stack((self.x.flatten(), self.y.flatten()), 1)
        self.nmacro = 8 if slope else 7

    def rays(self, v, halo=False):
        cx, cy = .2*v[3], .2*v[4]
        if self.slope or self.fixed_slope != 2.:
            alpha, _ = epl_deflection(self.coordinates, theta_E=v[0].exp(), q=v[1],
                                      phi=v[2], gamma=v[7] if self.slope else self.fixed_slope, center=(cx, cy))
        else:
            alpha = sie_alpha(self.coordinates, v[0].exp(), v[1], v[2], cx, cy)
        dx, dy = self.coordinates[:, 0]-cx, self.coordinates[:, 1]-cy
        shear = .2*torch.stack((v[5]*dx+v[6]*dy, v[6]*dx-v[5]*dy), -1)
        beta = (self.coordinates-alpha-shear).reshape(*self.x.shape, 2)
        if halo:
            hx, hy, strength = v[self.nmacro:]
            ax, ay = truncated_subhalo_deflection(self.x, self.y, hx, hy, strength*self.max_strength)
            beta = beta-torch.stack((ax, ay), -1)
        return beta[..., 0], beta[..., 1]

    def optimize(self, start, bounds, halo, initial=None, maxiter=140):
        result = minimize_profiled(self.engine, lambda v: self.rays(v, halo),
                                   bounded_start(start,bounds),
                                   bounds, maxiter=maxiter, initial=initial)
        # Continue a budget-limited fit without loosening convergence tolerances.
        if self.retry_failed and not result.converged:
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


def refine_full_pinn(result, *, macro_model=None, free_slope=False, source_nodes=None, seed=42,
                     regularization_factor=1., n_starts=3, maxiter=140,
                     max_strength=.30, refinement_radius=.70, bounds=None, retry_failed=True):
    """Refit matched null/halo models, rescuing null from every halo macro.

    Zero strength is explicitly feasible. Halo hypotheses include the optimized
    null exactly; null rescue reduces optimizer-induced false improvements.
    """
    if maxiter < 1 or n_starts < 1 or not np.isfinite([max_strength,refinement_radius,regularization_factor]).all() or min(max_strength,refinement_radius,regularization_factor)<=0:
        raise ValueError("Use positive finite strengths, radius, regularization and iteration/start counts")
    slope,nodes,reg_factor,starts = free_slope,source_nodes,regularization_factor,n_starts
    obs = result.observation
    reference = macro_model or result.observation.macro_model
    if reference is None: raise ValueError("Full refinement requires a macro initialization")
    if macro_model is not None:
        obs=obs.with_macro(reference)
        reference=obs.macro_model
    candidates=tuple(result.candidates)
    source_macro = result.macro_model or reference
    geometry = obs.with_macro(source_macro)
    baseline = result.baseline
    problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, geometry.bx, geometry.by,
        source_grid=baseline.grid if nodes is None else None,
        source_nodes=nodes or len(baseline.coefficients), seed=seed,
        regularization=baseline.regularization*reg_factor, mesh_strategy='centroidal')
    model = FullMacroProblem(obs, problem, slope, max_strength)
    model.retry_failed=retry_failed
    bounds = macro_bounds(reference, slope) if bounds is None else bounds
    if len(bounds)!=(8 if slope else 7): raise ValueError("Macro bounds must match the free parameters")
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
        hb = bounds + [(c.seed[0]-refinement_radius, c.seed[0]+refinement_radius),
                       (c.seed[1]-refinement_radius, c.seed[1]+refinement_radius), (0., 1.)]
        nv = null.values.detach().cpu().numpy()
        old = vector(c.fit.macro_model or source_macro, slope)
        low_start = max(c.fit.strength,.02)
        low_start = min(low_start,.06,.5*max_strength)
        hstarts = [np.r_[old, c.fit.x, c.fit.y, c.fit.strength/max_strength],
                   np.r_[nv, c.seed, low_start/max_strength],
                   np.r_[nv, c.fit.x+.04, c.fit.y-.04, min(.03,.25*max_strength)/max_strength]]
        while len(hstarts)<starts:
            trial=hstarts[1].copy();k=len(hstarts)
            trial[-3:-1]=np.asarray(c.fit.position)+[.04*(-1)**k,.04*(-1)**(k//2)]
            trial[-1]=min(1.,.03*(1+k%3)/max_strength)
            hstarts.append(trial)
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
    # All candidates use the best null found anywhere, not a different null per peak.
    source = model.engine.export(null.source)
    null_macro = macro_at(null.values.detach().cpu().numpy(), slope, model.fixed_slope)
    null_chi = float(np.sum(((obs.image-source.image_model)[obs.mask]/obs.rms[obs.mask])**2))
    fits_out, records = [], []
    for c, best, hb in outcomes:
        v = best.values.detach().cpu().numpy()
        if float(best.source.objective) >= float(null.source.objective):
            fitted = source; adjusted = null_macro; v[-1] = 0.; success = null.converged
            v[:model.nmacro] = null.values.detach().cpu().numpy()
        else:
            fitted = model.engine.export(best.source); adjusted = macro_at(v[:model.nmacro], slope, model.fixed_slope)
            success = best.converged
        chi = float(np.sum(((obs.image-fitted.image_model)[obs.mask]/obs.rms[obs.mask])**2))
        f = HaloFit(float(v[-3]), float(v[-2]), float(v[-1]*max_strength), null_chi-chi,
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
                     max_strength=max_strength, refinement_radius=refinement_radius,
                     config={**result.config, 'execution': current_execution().to_dict(),
                             'joint_refinement': dict(macro_model_input=reference,free_macro_parameters=['theta_E','q','phi','cx','cy','g1','g2']
                                 + (['gamma'] if slope else [])),
                             'full_refinement': dict(slope_free=slope, nodes=nodes, seed=seed,
                                 regularization_factor=reg_factor, maxiter=maxiter, halo_starts=starts,
                                 low_strength_start_cap=.06,
                                 null_rescued_from_each_halo=True)})
    record = dict(null=optimum_record(null), null_macro=null_macro,
        execution=current_execution().to_dict(),
        null_macro_at_bound=bool(flags(null.values.detach().cpu().numpy(), bounds).any()),
        null_attempts=[optimum_record(o) for o in nulls], candidates=records,
        candidate_attempts=logs, source_nodes=problem.size, regularization=problem.regularization,
        macro_bounds=bounds, halo_cap_arcsec=max_strength, local_halo_radius_arcsec=refinement_radius,
        independent_validation=False, calibrated_detection=False)
    return replace(output,config={**output.config,"full_refinement_record":record})
