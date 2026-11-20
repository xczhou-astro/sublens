"""Data-selected Delaunay bases for single-halo source-profiled refinement."""

from dataclasses import replace

import numpy as np
import torch

from ..observation import LensingObservation
from ..profiles.subhalo import truncated_subhalo_deflection
from .source_plane import PixelSourceProblem


def _model_rays(observation, macro_model, fit=None):
    projected = LensingObservation.from_arrays(
        observation.image, observation.rms, observation.psf, pixel_scale=observation.pixel_scale,
        macro_model=macro_model, mask=observation.mask, native_rays=observation.image_rays(),
    )
    bx, by = projected.bx, projected.by
    if fit is not None and fit.strength > 0:
        x, y = (torch.as_tensor(a, dtype=torch.float64) for a in observation.image_rays())
        ax, ay = truncated_subhalo_deflection(x, y, fit.x, fit.y, fit.strength)
        bx, by = bx - ax.numpy(), by - ay.numpy()
    return bx, by


def _select_source(observation, bx, by, validation, nodes, regularizations, seeds, strategy):
    train = observation.mask & ~validation
    rows = []
    for count in nodes:
        for seed in seeds:
            first = PixelSourceProblem(
                observation.image, observation.rms, observation.psf, train, bx, by,
                source_nodes=count, regularization=regularizations[0], seed=seed, mesh_strategy=strategy,
            )
            for regularization in regularizations:
                problem = first if regularization == regularizations[0] else first.with_regularization(regularization)
                source = problem.fit(bx, by)
                residual = (observation.image - source.image_model) / observation.rms
                rows.append(dict(
                    source_nodes=int(count), seed=int(seed), regularization=float(regularization),
                    train_chi2_per_pixel=float(np.mean(residual[train]**2)),
                    validation_chi2_per_pixel=float(np.mean(residual[validation]**2)),
                ))
    selected = min(rows, key=lambda row: (row["validation_chi2_per_pixel"], row["source_nodes"]))
    return selected, rows


def adapt_source_basis_pinn(result, *, source_nodes=(300, 500),
                            regularizations=(2e-6, 2e-4, .02, 2., 20.), seeds=(22, 42),
                            mesh_strategy="centroidal", validation_fraction=.2,
                            validation_seed=20261118):
    """Choose one common source basis without any nonlinear lens/halo refit.

    All candidate positions survive. The current null and each halo are only
    reprofiled in source brightness on the selected grid, with freshly matched
    gains. These are intermediate fixed-lens fits, ready for refine_full().
    Selection is conditional on the current best model, not independent
    discovery validation. No external macro/source products are accepted.
    """
    source_nodes, regularizations, seeds = tuple(source_nodes), tuple(regularizations), tuple(seeds)
    if (not source_nodes or min(source_nodes) < 4 or not regularizations
            or not np.isfinite(regularizations).all() or min(regularizations) <= 0
            or not seeds or not 0 < validation_fraction < 1):
        raise ValueError("Use valid source sizes, positive regularization and a validation fraction in (0,1)")
    obs = result.observation
    null_macro = result.macro_model or obs.macro_model
    if null_macro is None:
        raise ValueError("Source adaptation needs an existing image-fitted macro")
    indices = np.flatnonzero(obs.mask)
    if len(indices) < 2:
        raise ValueError("Source CV needs at least two valid pixels")
    count = min(len(indices)-1, max(1, int(len(indices)*validation_fraction)))
    validation = np.zeros(obs.mask.shape, dtype=bool)
    validation.ravel()[np.random.default_rng(validation_seed).choice(indices, count, replace=False)] = True
    best = result.best_candidate
    fit = None if best is None else best.fit
    geometry_macro = (fit.macro_model if fit is not None else None) or null_macro
    bx, by = _model_rays(obs, geometry_macro, fit)
    selected, trials = _select_source(obs, bx, by, validation, source_nodes,
                                      regularizations, seeds, mesh_strategy)
    problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, bx, by,
        source_nodes=selected['source_nodes'], seed=selected['seed'],
        regularization=selected['regularization'], mesh_strategy=mesh_strategy)
    baseline = problem.fit(*_model_rays(obs, null_macro))
    null_chi2 = np.sum(((obs.image-baseline.image_model)[obs.mask]/obs.rms[obs.mask])**2)
    candidates = []
    for candidate in result.candidates:
        previous = candidate.fit
        source = problem.fit(*_model_rays(obs, previous.macro_model or null_macro, previous))
        chi2 = np.sum(((obs.image-source.image_model)[obs.mask]/obs.rms[obs.mask])**2)
        updated = replace(previous, source=source, image_model=source.image_model,
            delta_chi2=float(null_chi2-chi2),
            delta_objective=float(obs.mask.sum()*(baseline.objective-source.objective)),
            converged=False, iterations=0,
            optimizer_message="Source basis updated; nonlinear parameters await full refinement")
        candidates.append(replace(candidate, fit=updated))
    adaptive = dict(mesh_strategy=mesh_strategy, mesh_rounds=1, source_nodes=list(source_nodes),
        regularizations=list(regularizations), seeds=list(seeds), validation_seed=validation_seed,
        validation_pixel_count=int(validation.sum()), training_pixel_count=int((obs.mask & ~validation).sum()),
        validation_is_conditional_on_fitted_lens=True, final_source_uses_all_fitted_pixels=True,
        nonlinear_refinement=False, boundary_expansion=False,
        history=[dict(round=1, selected=selected, source_trials=trials, expanded_radius=None)])
    return replace(result, baseline=baseline, candidates=tuple(candidates), macro_model=null_macro,
                   macro_converged=False, config={**result.config, 'adaptive_source': adaptive})


def refine_adaptive_pinn(result, macro_model, *, source_nodes=(250, 400, 600),
                         regularizations=(2., 20., 200.), seeds=(11, 22, 33, 44),
                         mesh_strategy="centroidal", validation_mask=None, validation_fraction=.2,
                         validation_seed=20261118, mesh_rounds=2, maxiter=100, n_starts=3,
                         retry_failed=True, expand_boundary=True, max_refinement_radius=.7, verbose=False):
    """Choose source complexity by prediction, then refit null and halo on one basis.

    The source-validation loss is conditional on the currently inferred lens
    geometry; it is not an independent test of the complete discovery pipeline.
    The final solve uses all fitted pixels after hyperparameter selection. No
    injected source, halo location or strength is accepted by this interface.
    """
    source_nodes, regularizations, seeds = tuple(source_nodes), tuple(regularizations), tuple(seeds)
    if (not source_nodes or min(source_nodes) < 4 or not regularizations
            or not np.isfinite(regularizations).all() or min(regularizations) <= 0 or not seeds
            or mesh_rounds < 1 or maxiter < 1 or n_starts < 1 or not 0 < validation_fraction < 1):
        raise ValueError("Use valid source sizes, positive regularization/iterations and a validation fraction in (0,1)")
    obs = result.observation
    if validation_mask is None:
        validation = np.zeros(obs.mask.size, dtype=bool)
        indices = np.flatnonzero(obs.mask)
        rng = np.random.default_rng(validation_seed)
        validation[rng.choice(indices, max(1, int(len(indices)*validation_fraction)), replace=False)] = True
        validation = validation.reshape(obs.mask.shape)
    else:
        validation = np.asarray(validation_mask, dtype=bool)
    if (validation.shape != obs.mask.shape or not validation.any() or np.any(validation & ~obs.mask)
            or not (obs.mask & ~validation).any()):
        raise ValueError("Validation pixels must be a nonempty proper subset of fitted pixels")
    history = []
    current = result
    for iteration in range(mesh_rounds):
        expanded_radius = None
        previous_best = current.best_candidate
        if expand_boundary and previous_best is not None and previous_best.fit.strength > 0:
            offset = np.abs(np.asarray(previous_best.fit.position) - previous_best.seed)
            if offset.max() >= .97 * current.refinement_radius and current.refinement_radius < max_refinement_radius:
                expanded_radius = min(max_refinement_radius, 1.6 * current.refinement_radius)
                current = replace(current, refinement_radius=expanded_radius,
                                  config={**current.config, "refinement_radius_arcsec": expanded_radius}).refine_joint(
                    macro_model, maxiter=maxiter, n_starts=n_starts, retry_failed=retry_failed,
                )
                if verbose:
                    print(f"Expanded boundary-limited proposal radius to {expanded_radius:.3f} arcsec", flush=True)
        best = current.best_candidate
        fit = None if best is None else best.fit
        geometry_macro = ((fit.macro_model if fit is not None else None)
                          or current.macro_model or macro_model)
        bx, by = _model_rays(obs, geometry_macro, fit)
        selected, trials = _select_source(
            obs, bx, by, validation, source_nodes, regularizations, seeds, mesh_strategy,
        )
        if verbose:
            print(f"Adaptive round {iteration+1}: nodes={selected['source_nodes']}, "
                  f"seed={selected['seed']}, lambda={selected['regularization']:g}, "
                  f"validation chi2/pixel={selected['validation_chi2_per_pixel']:.4f}", flush=True)
        problem = PixelSourceProblem(
            obs.image, obs.rms, obs.psf, obs.mask, bx, by, source_nodes=selected["source_nodes"],
            seed=selected["seed"], regularization=selected["regularization"], mesh_strategy=mesh_strategy,
        )
        # Null and every halo hypothesis use exactly the same vertices and prior.
        baseline = problem.fit(obs.bx, obs.by)
        candidates = []
        for candidate in current.candidates:
            previous = candidate.fit
            hx, hy = _model_rays(obs, previous.macro_model or macro_model, previous)
            source = problem.fit(hx, hy)
            candidates.append(replace(candidate, fit=replace(previous, source=source,
                                                             image_model=source.image_model)))
        stage = replace(current, baseline=baseline, candidates=tuple(candidates))
        current = stage.refine_joint(macro_model, maxiter=maxiter, n_starts=n_starts,
                                     retry_failed=retry_failed)
        history.append(dict(round=iteration+1, selected=selected, source_trials=trials,
                            expanded_radius=expanded_radius))
        if verbose and current.best_candidate is not None:
            updated = current.best_candidate.fit
            chi2 = np.mean(((obs.image-updated.image_model)[obs.mask]/obs.rms[obs.mask])**2)
            print(f"  Joint fit: xy={updated.position}, strength={updated.strength:.6f}, "
                  f"chi2/pixel={chi2:.4f}, converged={updated.converged}", flush=True)
    adaptive = dict(
        mesh_strategy=mesh_strategy, mesh_rounds=mesh_rounds, source_nodes=list(source_nodes),
        regularizations=list(regularizations), seeds=list(seeds), validation_seed=validation_seed,
        validation_pixel_count=int(validation.sum()), training_pixel_count=int((obs.mask & ~validation).sum()),
        validation_is_conditional_on_fitted_lens=True, final_source_uses_all_fitted_pixels=True,
        history=history,
    )
    return replace(current, config={**current.config, "adaptive_source": adaptive})
