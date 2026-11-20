"""Joint macro, compact halo and source refinement on a fixed Delaunay basis."""

from dataclasses import replace

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from ..inversion.source_plane import PixelSourceProblem
from ..profiles.macro import epl_deflection, external_shear_deflection
from ..profiles.subhalo import truncated_subhalo_deflection
from .halo_search import HaloFit, _delaunay_native_image


def refine_joint_pinn(result, macro_model, *, radius_fraction=0.03, shear_offset=0.03, maxiter=80,
                      n_starts=1, retry_failed=False):
    """Return a new search result with equally refitted null and halo hypotheses."""
    from ..execution import current_execution
    if current_execution() is not None:
        from .torch_refinement import refine_joint_torch
        return refine_joint_torch(result, macro_model, radius_fraction=radius_fraction,
                                  shear_offset=shear_offset, maxiter=maxiter,
                                  n_starts=n_starts, retry_failed=retry_failed)
    if (not np.isfinite([radius_fraction, shear_offset]).all()
            or not 0 < radius_fraction < 1 or shear_offset <= 0 or maxiter < 1 or n_starts < 1):
        raise ValueError("Use 0 < radius_fraction < 1 and positive shear offset and iteration limit")
    obs = result.observation
    macro = dict(macro_model)
    theta_e, slope = float(macro["theta_E"]), float(macro.get("gamma", 2.0))
    if not np.isfinite([theta_e, slope]).all() or theta_e <= 0 or not 1 < slope < 3:
        raise ValueError("The reference EPL needs positive theta_E and 1 < gamma < 3")
    center = macro.get("center", obs.lens_center)
    x, y = (torch.as_tensor(a, dtype=torch.float64) for a in obs.image_rays())
    coordinates = torch.stack((x.flatten(), y.flatten()), dim=1)
    host, _ = epl_deflection(
        coordinates, theta_E=theta_e, q=macro.get("q", 0.8), phi=macro.get("phi", 0.0),
        gamma=slope, center=center,
    )
    shear = external_shear_deflection(
        coordinates, gamma_ext=macro.get("gamma_ext", 0.0),
        phi_ext=macro.get("phi_ext", 0.0), center=center,
    )
    base = torch.stack((torch.as_tensor(obs.bx), torch.as_tensor(obs.by)), dim=-1)
    expected = coordinates - host - shear
    if not np.allclose(expected.numpy(), base.reshape(-1, 2).numpy(), rtol=1e-5, atol=2e-5):
        raise ValueError("macro_model must describe the observation's input EPL and shear rays")

    host = host.reshape(*x.shape, 2)
    dx, dy = x - center[0], y - center[1]
    shear_x = torch.stack((dx, -dy), dim=-1)
    shear_y = torch.stack((dy, dx), dim=-1)
    baseline = result.baseline
    problem = PixelSourceProblem(
        obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
        regularization=baseline.regularization, source_grid=baseline.grid,
        mesh_strategy=baseline.mesh_strategy or "random",
    )
    count = int(obs.mask.sum())
    angle = 2 * float(macro.get("phi_ext", 0.0))
    g1 = float(macro.get("gamma_ext", 0.0)) * np.cos(angle)
    g2 = float(macro.get("gamma_ext", 0.0)) * np.sin(angle)

    def rays_at(parameters, has_halo):
        radius, dg1, dg2 = parameters[:3]
        rays = (base - ((1 + radius_fraction * radius)**(slope - 1) - 1) * host
                - shear_offset * (dg1 * shear_x + dg2 * shear_y))
        if has_halo:
            hx, hy, amplitude = parameters[3:]
            ax, ay = truncated_subhalo_deflection(x, y, hx, hy, amplitude * result.max_strength)
            rays = rays - torch.stack((ax, ay), dim=-1)
        return rays

    attempt_logs = []

    def optimize(start, bounds, initial_source, has_halo):
        best = None

        @torch.enable_grad()
        def objective(values):
            nonlocal best
            parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            rays = rays_at(parameters, has_halo)
            bx, by = rays.unbind(dim=-1)
            source = problem.fit(bx.detach().numpy(), by.detach().numpy(), initial=initial_source.coefficients)
            if best is None or source.objective < best[0]:
                best = source.objective, np.array(values), source
            residual = source.image_model[obs.mask] * problem.weight - problem.target
            image_gradient = np.zeros(obs.image.shape)
            image_gradient[obs.mask] = 2 * residual * problem.weight
            adjoint = fftconvolve(image_gradient, problem.convolution_psf[::-1, ::-1], mode="same")
            native = _delaunay_native_image(bx, by, source.grid, source.coefficients)
            gradient, = torch.autograd.grad((native * torch.from_numpy(adjoint)).sum(), parameters)
            return source.objective, gradient.numpy()

        outcome = minimize(
            objective, start, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": maxiter, "gtol": 1e-5, "ftol": 1e-9},
        )
        retried = False
        if retry_failed and not outcome.success:
            initial_source = best[2]
            outcome = minimize(
                objective, best[1], method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": 2 * maxiter, "maxls": 40, "gtol": 1e-5, "ftol": 1e-10},
            )
            retried = True
        attempt_logs.append(dict(has_halo=has_halo, start=np.asarray(start).tolist(),
                                 converged=bool(outcome.success), message=str(outcome.message),
                                 iterations=int(outcome.nit), retried=retried,
                                 objective=float(best[0])))
        _, values, source = best
        adjusted = dict(macro)
        adjusted["theta_E"] = theta_e * (1 + radius_fraction * values[0])
        shear_1, shear_2 = g1 + shear_offset * values[1], g2 + shear_offset * values[2]
        adjusted["gamma_ext"] = float(np.hypot(shear_1, shear_2))
        adjusted["phi_ext"] = float(np.arctan2(shear_2, shear_1) / 2)
        return values, source, adjusted, bool(outcome.success)

    macro_bounds = [(-1.0, 1.0)] * 3
    null_values, null_source, null_macro, null_success = optimize(np.zeros(3), macro_bounds, baseline, False)
    null_chi2 = np.sum(((obs.image - null_source.image_model)[obs.mask] / obs.rms[obs.mask])**2)
    candidates = []
    for candidate in result.candidates:
        fit = candidate.fit
        initial = np.r_[np.zeros(3), fit.x, fit.y, fit.strength / result.max_strength]
        # Original PINN peak defines the local search region.
        half_width = result.refinement_radius
        bounds = macro_bounds + [
            (candidate.seed[0] - half_width, candidate.seed[0] + half_width),
            (candidate.seed[1] - half_width, candidate.seed[1] + half_width),
            (0.0, 1.0),
        ]
        starts = [initial]
        patterns = [(0., 0., 1.), (.04, -.04, 1.5), (-.04, .04, .5), (.04, .04, 1.)]
        for index in range(n_starts - 1):
            dx, dy, scale = patterns[index % len(patterns)]
            trial = initial.copy()
            if index == 0:
                trial[:3] = null_values
                trial[3:5] = candidate.seed
            else:
                trial[3:5] += [dx, dy]
            trial[5] = max(trial[5] * scale, 0.015 / result.max_strength)
            trial = np.clip(trial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
            starts.append(trial)
        log_start = len(attempt_logs)
        solutions = [optimize(start, bounds, fit.source, True) for start in starts]
        best_objective = min(item[1].objective for item in solutions)
        near_best = [item for item in solutions if item[1].objective <= best_objective + 1e-7]
        chosen = min(near_best, key=lambda item: (not item[3], item[1].objective))
        values, source, fitted_macro, success = chosen
        log = attempt_logs[log_start + next(i for i, item in enumerate(solutions) if item is chosen)]
        if source.objective >= null_source.objective:
            values = np.r_[null_values, values[3:5], 0.0]
            source, fitted_macro, success = null_source, null_macro, null_success
            log = attempt_logs[0]
        chi2 = np.sum(((obs.image - source.image_model)[obs.mask] / obs.rms[obs.mask])**2)
        halo_fit = HaloFit(
            float(values[3]), float(values[4]), float(values[5] * result.max_strength),
            float(null_chi2 - chi2), float(count * (null_source.objective - source.objective)),
            success, source, source.image_model, fitted_macro,
            optimizer_message=log["message"], iterations=log["iterations"],
        )
        candidates.append(replace(candidate, fit=halo_fit))
    config = {
        **result.config,
        "joint_refinement": {
            "macro_model_input": macro, "radius_fraction": radius_fraction,
            "shear_offset": shear_offset, "maxiter": maxiter,
            "n_starts": n_starts, "retry_failed": bool(retry_failed),
            "optimizer_attempts": attempt_logs,
            "free_macro_parameters": ["theta_E", "shear_1", "shear_2"],
            "free_halo_parameters": ["x", "y", "strength"],
        },
    }
    return replace(result, baseline=null_source, candidates=tuple(candidates),
                   macro_model=null_macro, macro_converged=null_success, config=config)
