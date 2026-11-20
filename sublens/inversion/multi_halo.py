"""Source-profiled simultaneous compact-halo and macro refinement.

This module fits hypotheses with an explicit number of candidate components.
It does not choose a halo count or calibrate detection significance.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import torch

from .halo_search import _delaunay_native_image
from .source_plane import PixelSource, PixelSourceProblem
from ..profiles.macro import epl_deflection, external_shear_deflection
from ..profiles.subhalo import truncated_subhalo_deflection


@dataclass
class MultiHaloFit:
    """One conditional component-count hypothesis, including its re-solved source."""

    parameters: np.ndarray
    source: PixelSource
    macro_model: dict
    halos: np.ndarray
    converged: bool
    attempts: list
    execution: Optional[dict] = None


class MultiHaloProblem:
    """Fit all candidate halos together on a specified common source basis.

    Parameters are three normalized macro offsets followed by (x,y,b/max_b)
    for each halo. Angular coordinates and strengths are in arcsec. The macro
    input must describe the observation's original EPL+shear rays. Only its
    theta_E and two shear components vary. Source coefficients are re-solved
    at every objective evaluation; no halo truth is accepted by this class.
    """

    def __init__(self, observation, *, problem=None, source_nodes=300,
                 regularization=20., mesh_strategy="centroidal", radius_fraction=.03,
                 shear_offset=.03, max_strength=.12):
        if (not np.isfinite([radius_fraction, shear_offset, max_strength]).all()
                or not 0 < radius_fraction < 1 or min(shear_offset, max_strength) <= 0):
            raise ValueError("Use positive macro ranges and halo strength scale")
        if observation.macro_model is None:
            raise ValueError("The input observation needs an EPL+shear macro model")
        self.obs = observation
        self.macro = dict(observation.macro_model)
        self.radius_fraction, self.shear_offset, self.max_strength = radius_fraction, shear_offset, max_strength
        self.problem = problem if problem is not None else PixelSourceProblem(
            observation.image, observation.rms, observation.psf, observation.mask, observation.bx, observation.by,
            source_nodes=source_nodes, regularization=regularization, mesh_strategy=mesh_strategy)
        if (self.problem.shape != observation.image.shape or
                not np.array_equal(self.problem.mask, observation.mask)):
            raise ValueError("The source problem must use the same image shape and fit mask")
        expected_weight = 1/(observation.rms[observation.mask]*np.sqrt(observation.mask.sum()))
        if (not np.allclose(self.problem.weight, expected_weight) or
                not np.allclose(self.problem.target, observation.image[observation.mask]*expected_weight) or
                self.problem.psf.shape != observation.psf.shape or
                not np.allclose(self.problem.psf, observation.psf/observation.psf.sum())):
            raise ValueError("The source problem must use the same image, noise and PSF")
        self.x, self.y = (torch.as_tensor(a, dtype=torch.float64) for a in observation.image_rays())
        coordinates = torch.stack((self.x.flatten(), self.y.flatten()), dim=1)
        self.base = torch.stack((torch.as_tensor(observation.bx), torch.as_tensor(observation.by)), dim=-1)
        self.slope = float(self.macro.get("gamma", 2.0))
        theta_e = float(self.macro["theta_E"])
        if not np.isfinite([theta_e, self.slope]).all() or theta_e <= 0 or not 1 < self.slope < 3:
            raise ValueError("The macro needs positive theta_E and 1 < gamma < 3")
        center = self.macro.get("center", observation.lens_center)
        host, _ = epl_deflection(coordinates, theta_E=theta_e, q=self.macro.get("q", .8),
                                 phi=self.macro.get("phi", 0.), gamma=self.slope, center=center)
        shear = external_shear_deflection(coordinates, gamma_ext=self.macro.get("gamma_ext", 0.),
                                          phi_ext=self.macro.get("phi_ext", 0.), center=center)
        if not np.allclose((coordinates-host-shear).numpy(), self.base.reshape(-1, 2).numpy(),
                           rtol=1e-5, atol=2e-5):
            raise ValueError("The macro model must describe the observation's original rays")
        self.host = host.reshape(*self.x.shape, 2)
        dx, dy = self.x-center[0], self.y-center[1]
        self.shear_x = torch.stack((dx, -dy), dim=-1)
        self.shear_y = torch.stack((dy, dx), dim=-1)

    def rays(self, parameters):
        """Return differentiable source rays for the complete simultaneous model."""
        if parameters.ndim != 1 or len(parameters) < 3 or (len(parameters)-3) % 3:
            raise ValueError("Parameters must have length 3 + 3 * halo_count")
        key=(str(parameters.device),parameters.dtype)
        cache=getattr(self,'_torch_geometry',None)
        if cache is None:cache={};self._torch_geometry=cache
        if key not in cache:
            cache[key]=tuple(t.to(device=parameters.device,dtype=parameters.dtype)
                for t in (self.base,self.host,self.shear_x,self.shear_y,self.x,self.y))
        base,host,shear_x,shear_y,x_grid,y_grid=cache[key]
        radius, g1, g2 = parameters[:3]
        rays = (base-((1+self.radius_fraction*radius)**(self.slope-1)-1)*host
                - self.shear_offset*(g1*shear_x+g2*shear_y))
        for x, y, amplitude in parameters[3:].reshape(-1, 3):
            ax, ay = truncated_subhalo_deflection(x_grid, y_grid, x, y, amplitude*self.max_strength)
            rays = rays-torch.stack((ax, ay), dim=-1)
        return rays

    def profiled_objective(self, values, initial_source=None):
        """Return objective, envelope gradient and the optimized nonnegative source."""
        from ..execution import current_execution
        if current_execution() is not None:
            from .torch_source import torch_problem
            engine=torch_problem(self.problem)
            parameters=engine.tensor(values).detach().requires_grad_(True)
            bx,by=self.rays(parameters).unbind(-1)
            fitted=engine.fit(bx.detach(),by.detach(),initial=None if initial_source is None else initial_source.coefficients)
            loss=engine.envelope_loss(bx,by,fitted)
            gradient,=torch.autograd.grad(loss,parameters)
            return float(fitted.objective),gradient.detach().cpu().numpy(),engine.export(fitted)
        with torch.enable_grad():
            parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            bx, by = self.rays(parameters).unbind(dim=-1)
            source = self.problem.fit(bx.detach().numpy(), by.detach().numpy(),
                                       initial=None if initial_source is None else initial_source.coefficients)
            residual = source.image_model[self.obs.mask]*self.problem.weight-self.problem.target
            gradient_image = np.zeros(self.obs.image.shape)
            gradient_image[self.obs.mask] = 2*residual*self.problem.weight
            adjoint = fftconvolve(gradient_image, self.problem.convolution_psf[::-1, ::-1], mode="same")
            native = _delaunay_native_image(bx, by, source.grid, source.coefficients)
            gradient, = torch.autograd.grad((native*torch.from_numpy(adjoint)).sum(), parameters)
        return source.objective, gradient.numpy(), source

    def fit(self, initial, anchors, *, initial_source=None, refinement_radius=.35,
            maxiter=120, n_starts=3, retry_failed=True):
        """Jointly refine a candidate configuration using deterministic multiple starts.

        Anchors are proposed locations, not injected positions. They define each
        component's local search box. A zero-strength component is allowed, so
        adding a component does not force a physical detection.
        """
        values = np.asarray(initial, dtype=float)
        anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
        if (values.shape != (3+3*len(anchors),) or not np.isfinite(values).all()
                or not np.isfinite(anchors).all() or not np.isfinite(refinement_radius)
                or refinement_radius <= 0 or min(maxiter, n_starts) < 1):
            raise ValueError("Use matching finite parameters/anchors and positive fit budgets")
        bounds = [(-1., 1.)]*3
        for x, y in anchors:
            bounds.extend(((x-refinement_radius, x+refinement_radius),
                           (y-refinement_radius, y+refinement_radius), (0., 1.)))
        limits = np.array(bounds)
        values = np.clip(values, limits[:, 0], limits[:, 1])
        starts = [values]
        patterns = ((.04, -.04, 1.5), (-.04, .04, .5), (.04, .04, 1.))
        for index in range(n_starts-1):
            trial = values.copy()
            if index == 1 and len(anchors):
                # Include the nested hypothesis, leaving all earlier components
                # unchanged while the newly added component starts at zero.
                trial[-1] = 0.
                starts.append(trial)
                continue
            dx, dy, amplitude_scale = patterns[index % len(patterns)]
            if index == 0 and len(anchors):
                trial[3:].reshape(-1, 3)[:, :2] = anchors
            else:
                trial[3:].reshape(-1, 3)[:, :2] += [dx, dy]
            trial[5::3] = np.maximum(trial[5::3]*amplitude_scale, .015/self.max_strength)
            starts.append(np.clip(trial, limits[:, 0], limits[:, 1]))
        solutions, logs = [], []
        from ..execution import current_execution
        execution=current_execution()
        # A null macro fit has only three offsets; repeated identical starts add no information.
        for start in starts[:1] if not len(anchors) else starts:
            if execution is not None:
                from .torch_source import torch_problem
                from .torch_refinement import minimize_profiled
                engine=torch_problem(self.problem)
                outcome=minimize_profiled(engine,lambda p:self.rays(p).unbind(-1),start,bounds,
                    maxiter=maxiter,initial=None if initial_source is None else initial_source.coefficients)
                retried=False
                if retry_failed and not outcome.converged:
                    retry=minimize_profiled(engine,lambda p:self.rays(p).unbind(-1),outcome.values,bounds,
                        maxiter=2*maxiter,initial=outcome.source.coefficients)
                    if float(retry.source.objective)<float(outcome.source.objective)+1e-9:outcome=retry
                    retried=True
                source=engine.export(outcome.source)
                fitted_values=outcome.values.detach().cpu().numpy().astype(float)
                logs.append(dict(start=start.tolist(),objective=source.objective,converged=outcome.converged,
                    message=outcome.message,iterations=outcome.iterations,retried=retried,
                    projected_gradient=outcome.projected_gradient,relative_objective_change=outcome.relative_objective_change))
                solutions.append((source.objective,fitted_values,source,outcome.converged))
                continue
            best = None
            warm = initial_source

            def objective(parameters):
                nonlocal best
                value, gradient, fitted = self.profiled_objective(parameters, warm)
                if best is None or value < best[0]:
                    best = value, np.array(parameters), fitted
                return value, gradient

            outcome = minimize(objective, start, method="L-BFGS-B", jac=True, bounds=bounds,
                               options={"maxiter": maxiter, "gtol": 1e-5, "ftol": 1e-9})
            retried = False
            if retry_failed and not outcome.success:
                warm = best[2]
                outcome = minimize(objective, best[1], method="L-BFGS-B", jac=True, bounds=bounds,
                                   options={"maxiter": 2*maxiter, "maxls": 40, "gtol": 1e-5, "ftol": 1e-10})
                retried = True
            logs.append(dict(start=start.tolist(), objective=float(best[0]), converged=bool(outcome.success),
                             message=str(outcome.message), iterations=int(outcome.nit), retried=retried))
            solutions.append((*best, bool(outcome.success)))
        objective_min = min(row[0] for row in solutions)
        near = [row for row in solutions if row[0] <= objective_min+1e-7]
        _, values, source, converged = min(near, key=lambda row: (not row[3], row[0]))
        adjusted = dict(self.macro)
        adjusted["theta_E"] *= 1+self.radius_fraction*values[0]
        angle = 2*float(self.macro.get("phi_ext", 0.))
        shear = float(self.macro.get("gamma_ext", 0.))
        g1 = shear*np.cos(angle)+self.shear_offset*values[1]
        g2 = shear*np.sin(angle)+self.shear_offset*values[2]
        adjusted["gamma_ext"] = float(np.hypot(g1, g2))
        adjusted["phi_ext"] = float(np.arctan2(g2, g1)/2)
        halos = values[3:].reshape(-1, 3).copy()
        halos[:, 2] *= self.max_strength
        return MultiHaloFit(values, source, adjusted, halos, converged, logs,
                            execution.to_dict() if execution is not None else None)
