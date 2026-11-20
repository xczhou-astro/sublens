"""Propose compact deflectors and re-fit their sources on a fixed source basis."""

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from ..inversion.source_plane import PixelSource, PixelSourceProblem
from ..profiles.subhalo import truncated_subhalo_deflection
from .potential import ResidualPotential, propose_residual_peaks
from ..observation import LensingObservation


@dataclass
class HaloFit:
    x: float
    y: float
    strength: float
    delta_chi2: float
    delta_objective: float
    converged: bool
    source: PixelSource
    image_model: np.ndarray
    macro_model: Optional[dict] = None
    optimizer_message: Optional[str] = None
    iterations: Optional[int] = None

    @property
    def position(self):
        return self.x, self.y


@dataclass
class HaloCandidate:
    seed: tuple
    seed_strength: float
    proposal_score: float
    proposal_method: str
    frozen: HaloFit
    refitted: HaloFit


@dataclass
class HaloSearchResult:
    baseline: PixelSource
    candidates: list
    radar_scores: np.ndarray
    radar_x: np.ndarray
    radar_y: np.ndarray
    profile_scores: np.ndarray
    profile_x: np.ndarray
    profile_y: np.ndarray
    residual_potential: Optional[ResidualPotential] = None

    @property
    def best_candidate(self):
        """Candidate with the largest penalized improvement after source refitting."""
        return self.candidates[0] if self.candidates else None


@dataclass
class PinnHaloCandidate:
    rank: int
    seed: tuple
    proposal_score: float
    fit: HaloFit


@dataclass
class PinnHaloSearchResult:
    observation: LensingObservation
    baseline: PixelSource
    residual_potential: ResidualPotential
    candidates: tuple
    search_radius: tuple
    macro_model: Optional[dict] = None
    macro_converged: Optional[bool] = None
    max_strength: float = 0.12
    refinement_radius: float = 0.35
    config: dict = field(default_factory=dict)

    def to_dict(self):
        """Return JSON-compatible parameters, fit metrics and run configuration."""
        from ..result_io import pinn_result_dict

        return pinn_result_dict(self)

    def detection(self, *, min_delta_chi2=None):
        """Return fit-improvement detection, position and strength; no new fit."""
        from ..detection import summarize_subhalo
        return summarize_subhalo(self, min_delta_chi2=min_delta_chi2)

    @property
    def detected(self):
        return self.detection()['detected']

    @property
    def position(self):
        return self.detection()['position_arcsec']

    @property
    def strength(self):
        return self.detection()['strength_arcsec']

    def refine_full(self, *, macro_model=None, **options):
        """Refit all macro parameters, halo and source against a matched null."""
        from ..execution import result_execution,current_execution,torch_backend
        from .full_refinement import refine_full_pinn
        with result_execution(self.config):
            if current_execution() is None:
                with torch_backend('cpu'):
                    return refine_full_pinn(self,macro_model=macro_model,**options)
            return refine_full_pinn(self,macro_model=macro_model,**options)

    def summary(self, *, print_output=True, detailed=False, min_delta_chi2=None):
        """Print the compact decision; detailed=True retains all fit diagnostics."""
        from ..result_io import pinn_result_summary

        text = pinn_result_summary(self, detailed=detailed, min_delta_chi2=min_delta_chi2)
        if print_output:
            print(text)
        return text

    def save_json(self, path, *, overwrite=False):
        """Save candidate parameters, macro models, metrics, settings and seeds."""
        from ..result_io import save_pinn_json

        return save_pinn_json(self, path, overwrite=overwrite)

    def save_fits(self, path, *, source_half_width=1.0, resolution=201, overwrite=False):
        """Save images, residuals, PINN maps, source mesh, candidates and metadata."""
        from ..result_io import save_pinn_fits

        return save_pinn_fits(
            self, path, source_half_width=source_half_width,
            resolution=resolution, overwrite=overwrite,
        )

    @property
    def best_candidate(self):
        """Return the candidate with the greatest gain in the training objective."""
        return max(self.candidates, key=lambda candidate: candidate.fit.delta_objective, default=None)

    @property
    def residual_macro(self):
        """Observed minus the PSF-convolved macro-only fit, in image units."""
        return self.observation.image - self.baseline.image_model

    @property
    def residual_with_subhalo(self):
        """Observed minus the best reprofiled halo fit, or None if no peak was proposed."""
        best = self.best_candidate
        return None if best is None else self.observation.image - best.fit.image_model

    def plot(self, *, truth_position=None, zoom=0.28, smoothed=True, save_path=None):
        """Plot the PINN convergence map, candidates and best physical refinement."""
        from ..utils.visualization import plot_pinn_convergence

        return plot_pinn_convergence(
            self, truth_position=truth_position, zoom=zoom, smoothed=smoothed, save_path=save_path,
        )

    def plot_fit(self, **options):
        """Plot the best reprofiled source and its residual, or the baseline if no peaks were found."""
        best = self.best_candidate
        source = self.baseline if best is None else best.fit.source
        return self.observation._source_result(source, self.observation.mask).plot(**options)

    def plot_residuals(self, *, normalized=True, limit=None, save_path=None):
        """Compare image residuals using the same fit mask and symmetric color scale."""
        from ..utils.visualization import plot_pinn_residuals

        return plot_pinn_residuals(self, normalized=normalized, limit=limit, save_path=save_path)

    def refine_joint(self, macro_model, *, radius_fraction=0.03, shear_offset=0.03, maxiter=80,
                     n_starts=1, retry_failed=False):
        """Refine EPL normalization, both shear components, halo position and strength.

        Source brightness is reprofiled on the existing Delaunay mesh. EPL shape,
        slope and center stay fixed. Refit the macro-only hypothesis with the
        same macro bounds before computing candidate gains.
        """
        from .joint_refinement import refine_joint_pinn
        from ..execution import result_execution
        with result_execution(self.config):
            return refine_joint_pinn(
                self, macro_model, radius_fraction=radius_fraction,
                shear_offset=shear_offset, maxiter=maxiter, n_starts=n_starts, retry_failed=retry_failed,
            )

    def adapt_source_basis(self, **options):
        """Update the common mesh/lambda and source brightness, without lens refits.

        Follow with refine_full() for nonlinear optimization and final decisions.
        """
        from .adaptive_refinement import adapt_source_basis_pinn
        from ..execution import result_execution
        with result_execution(self.config):
            return adapt_source_basis_pinn(self, **options)

    def refine_adaptive(self, macro_model=None, **options):
        """Legacy combined source selection plus local nonlinear joint refinement.

        The production flow uses adapt_source_basis() then refine_full() instead.
        """
        from .adaptive_refinement import refine_adaptive_pinn

        model = macro_model if macro_model is not None else self.observation.macro_model
        if model is None:
            raise ValueError("Adaptive joint refinement requires an input macro model")
        from ..execution import result_execution
        with result_execution(self.config):
            return refine_adaptive_pinn(self, model, **options)

    def merge_candidates(self, *, radius=None, strength_tolerance=0.25):
        """Return a final summary that merges duplicate, converged physical fits.

        Call after joint/source refinement: raw PINN peaks remain independent
        hypotheses during optimization. The strongest fit gain represents each
        group; members remain recorded in config. Zero-strength fits have no
        identified position and are never merged. This does not test existence
        or fit multiple subhaloes simultaneously.
        """
        radius = self.observation.pixel_scale if radius is None else float(radius)
        if (not np.isfinite([radius, strength_tolerance]).all() or radius <= 0
                or not 0 <= strength_tolerance <= 1):
            raise ValueError("Use a positive merge radius and strength tolerance in [0,1]")
        groups = []
        for candidate in sorted(self.candidates, key=lambda c: -c.fit.delta_objective):
            fit = candidate.fit
            matching = None
            if fit.converged and fit.strength > 0:
                for group in groups:
                    other = group[0].fit
                    if (other.converged and other.strength > 0
                            and np.linalg.norm(np.asarray(fit.position)-other.position) <= radius
                            and abs(fit.strength-other.strength) <= strength_tolerance*max(fit.strength, other.strength)):
                        matching = group
                        break
            if matching is None:
                groups.append([candidate])
            else:
                matching.append(candidate)
        previous = {g["representative_rank"]: g["members"]
                    for g in self.config.get("candidate_merge", {}).get("groups", [])}
        records = []
        for group in groups:
            members = []
            for c in group:
                members.extend(previous.get(c.rank, [dict(
                    rank=c.rank, seed_position_arcsec=list(c.seed), position_arcsec=list(c.fit.position),
                    strength_arcsec=c.fit.strength, delta_objective=c.fit.delta_objective,
                )]))
            records.append(dict(representative_rank=group[0].rank, members=members))
        config = {**self.config, "candidate_merge": dict(
            radius_arcsec=radius, relative_strength_tolerance=strength_tolerance,
            input_candidate_count=sum(len(g["members"]) for g in records),
            output_candidate_count=len(groups), groups=records,
            stage="after physical refinement", is_detection_decision=False,
        )}
        kept = tuple(sorted((group[0] for group in groups), key=lambda c:c.rank))
        return replace(self, candidates=kept, config=config)


def _perturbed_rays(bx, by, x, y, halo_x, halo_y, strength):
    ax, ay = truncated_subhalo_deflection(x, y, halo_x, halo_y, strength)
    return bx - ax, by - ay


def _delaunay_native_image(bx, by, mesh, coefficients):
    """Locate triangles on the CPU and differentiate the barycentric interpolation."""
    points = torch.stack((bx.flatten(), by.flatten()), dim=1)
    simplex = mesh.find_simplex(points.detach().numpy())
    inside = simplex >= 0
    index = np.maximum(simplex, 0)
    transform = torch.from_numpy(mesh.transform[index])
    offset = points - transform[:, 2]
    weights = (transform[:, :2] @ offset[..., None]).squeeze(-1)
    vertices = torch.from_numpy(coefficients[mesh.simplices[index]])
    value = vertices[:, 2] + (weights * (vertices[:, :2] - vertices[:, 2:])).sum(dim=1)
    return torch.where(torch.from_numpy(inside), value, 0.0).reshape(bx.shape).mean(dim=-1)


def _radar(problem, baseline, rays, scan_x, scan_y, center, radius, strength):
    scores = np.full((len(scan_y), len(scan_x)), np.nan)
    residual = (problem.target - baseline.image_model[problem.mask] * problem.weight)
    for j, y in enumerate(scan_y):
        for i, x in enumerate(scan_x):
            if not radius[0] <= np.hypot(x - center[0], y - center[1]) <= radius[1]:
                continue
            bx, by = _perturbed_rays(*rays, x, y, strength)
            prediction = problem.render(bx.numpy(), by.numpy(), baseline.coefficients)
            template = (prediction[problem.mask] - baseline.image_model[problem.mask]) * problem.weight
            norm = np.linalg.norm(template)
            if norm > 0:
                scores[j, i] = np.sqrt(problem.mask.sum()) * (residual @ template) / norm
    return scores


def _profile_grid(problem, baseline, rays, xs, ys, center, radius, strengths):
    """Screen a coarse grid by solving the source at each trial deflector."""
    gains = np.full((len(ys), len(xs)), np.nan)
    amplitudes = np.zeros_like(gains)
    count = problem.mask.sum()
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if not radius[0] <= np.hypot(x - center[0], y - center[1]) <= radius[1]:
                continue
            best_gain = -np.inf
            for strength in strengths:
                bx, by = _perturbed_rays(*rays, x, y, strength)
                trial = problem.fit(bx.numpy(), by.numpy(), initial=baseline.coefficients)
                gain = count * (baseline.objective - trial.objective)
                if gain > best_gain:
                    best_gain = gain
                    amplitudes[j, i] = strength
            gains[j, i] = best_gain
    return gains, amplitudes


def _grid_peaks(scores, xs, ys, count, separation):
    x, y = np.meshgrid(xs, ys)
    remaining = np.where(np.isfinite(scores), scores, -np.inf)
    for _ in range(count):
        j, i = np.unravel_index(np.argmax(remaining), remaining.shape)
        if remaining[j, i] <= 0:
            return
        yield j, i
        remaining[np.hypot(x - xs[i], y - ys[j]) < separation] = -np.inf


def _refine(problem, baseline, rays, initial, bounds, max_strength, reprofile, maxiter):
    from ..execution import current_execution
    if current_execution() is not None and reprofile:
        from .torch_refinement import refine_candidate
        return refine_candidate(problem, baseline, rays, initial, bounds, max_strength, maxiter)
    mask = problem.mask
    count = int(mask.sum())
    base_residual = baseline.image_model[mask] * problem.weight - problem.target
    chi2_base = count * (base_residual @ base_residual)
    base_prior = baseline.objective - chi2_base / count
    best = None

    @torch.enable_grad()
    def objective(parameters):
        nonlocal best
        parameters = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
        x, y, amplitude = parameters.unbind()
        bx, by = _perturbed_rays(*rays, x, y, amplitude * max_strength)
        if reprofile:
            source = problem.fit(bx.detach().numpy(), by.detach().numpy(),
                                 initial=baseline.coefficients)
            prediction = source.image_model
        else:
            source = baseline

        native = _delaunay_native_image(bx, by, source.grid, source.coefficients)
        if not reprofile:
            prediction = fftconvolve(native.detach().numpy(), problem.convolution_psf, mode="same")
        residual = prediction[mask] * problem.weight - problem.target
        chi2 = residual @ residual
        value = source.objective if reprofile else chi2 + base_prior
        if best is None or value < best[0]:
            best = (value, parameters.detach().numpy().copy(), chi2, source, prediction)

        # Envelope derivative: hold the optimized coefficients fixed for this
        # derivative, and apply the adjoint of the normalized PSF convolution.
        image_gradient = np.zeros(problem.shape)
        image_gradient[mask] = 2 * residual * problem.weight
        adjoint = fftconvolve(image_gradient, problem.convolution_psf[::-1, ::-1], mode="same")
        gradient, = torch.autograd.grad((native * torch.from_numpy(adjoint)).sum(), parameters)
        return float(value), gradient.numpy()

    result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": maxiter, "gtol": 1e-5, "ftol": 1e-9})
    value, (x, y, amplitude), chi2, source, prediction = best
    if value >= baseline.objective:
        return HaloFit(float(x), float(y), 0.0, 0.0, 0.0, bool(result.success),
                       baseline, baseline.image_model, optimizer_message=str(result.message),
                       iterations=int(result.nit))
    return HaloFit(
        float(x), float(y), float(amplitude * max_strength),
        float(chi2_base - count * chi2), float(count * (baseline.objective - value)),
        bool(result.success), source, prediction, optimizer_message=str(result.message),
        iterations=int(result.nit),
    )


def _prepare_search(obs, source_nodes, regularization, mesh_strategy="random"):
    problem = PixelSourceProblem(
        obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
        source_nodes=source_nodes, regularization=regularization, mesh_strategy=mesh_strategy,
    )
    baseline = problem.fit(obs.bx, obs.by)
    image_rays = obs.image_rays()
    rays = tuple(torch.as_tensor(a, dtype=torch.float64) for a in (obs.bx, obs.by, *image_rays))
    return problem, baseline, image_rays, rays


def search_pinn_subhalos(
    observation: LensingObservation, *, fit_mask=None, search_radius=None,
    source_nodes=400, regularization=0.002, top_k=3, separation=None, gain_radius=0.22,
    pinn_steps=90, seed_strength=0.015, max_strength=0.12,
    refinement_radius=0.35, refine_steps=55, cross_fit=True, mesh_strategy="random",
    psf_guard_max_radius=None,
):
    """Propose halos with a neural potential and reprofile the source at each peak.

    Candidate ranks are in PINN proposal order. The best candidate is selected
    by the penalized fit gain. Strengths are angular (arcsec). The gain aperture
    radius is independent of the exclusion distance, which defaults to two
    native pixels so competing maxima can survive until physical refinement.
    Use merge_candidates() after final refinement to summarize duplicate fits.
    psf_guard_max_radius optionally caps the PSF exclusion guard in arcsec for
    long-wing kernels. The full normalized PSF still enters every forward model;
    the default None retains the 98-percent encircled-energy guard.
    """
    obs = observation if fit_mask is None else replace(observation, mask=observation._fit_mask(fit_mask))
    if search_radius is None:
        if obs.einstein_radius is None:
            raise ValueError("Specify search_radius or an observation Einstein radius")
        search_radius = (0.70 * obs.einstein_radius, 1.30 * obs.einstein_radius)
    inner, outer = search_radius
    if not np.isfinite([inner, outer]).all() or not 0 <= inner < outer:
        raise ValueError("Search radii must be finite with 0 <= inner < outer")
    if min(top_k, pinn_steps, refine_steps) < 1:
        raise ValueError("Peak count and iteration limits must be positive")
    separation = 2 * obs.pixel_scale if separation is None else float(separation)
    scales = (gain_radius, seed_strength, max_strength, refinement_radius)
    if (not np.isfinite([separation, *scales]).all() or min(scales) <= 0
            or separation < 0 or seed_strength > max_strength):
        raise ValueError("Search scales must be positive, separation nonnegative and seed strength <= maximum")
    if psf_guard_max_radius is not None and (not np.isfinite(psf_guard_max_radius) or psf_guard_max_radius <= 0):
        raise ValueError("The optional PSF guard cap must be positive in arcsec")

    problem, baseline, image_rays, rays = _prepare_search(obs, source_nodes, regularization, mesh_strategy)
    potential = propose_residual_peaks(
        obs, *image_rays, radius=(inner, outer), source_nodes=source_nodes,
        regularization=regularization, separation=separation, top_k=top_k, steps=pinn_steps,
        cross_fit=cross_fit,
        mesh_strategy=mesh_strategy,
        gain_radius=gain_radius,
        psf_guard_max_radius=psf_guard_max_radius,
    )

    candidates = []
    for rank, (x, y, gain) in enumerate(potential.peaks, 1):
        bounds = [(x - refinement_radius, x + refinement_radius),
                  (y - refinement_radius, y + refinement_radius), (0.0, 1.0)]
        fit = _refine(problem, baseline, rays, [x, y, seed_strength / max_strength],
                      bounds, max_strength, reprofile=True, maxiter=refine_steps)
        candidates.append(PinnHaloCandidate(rank, (x, y), gain, fit))
    from ..execution import current_execution
    execution = current_execution()
    return PinnHaloSearchResult(
        obs, baseline, potential, tuple(candidates), (inner, outer),
        max_strength=max_strength, refinement_radius=refinement_radius,
        config={
            **({"execution": execution.to_dict()} if execution is not None else {}),
            "source_nodes": source_nodes, "regularization": baseline.regularization,
            "top_k": top_k, "separation_arcsec": separation, "pinn_steps": pinn_steps,
            "gain_radius_arcsec": gain_radius,
            "psf_guard_max_radius_arcsec": psf_guard_max_radius,
            "seed_strength_arcsec": seed_strength, "max_strength_arcsec": max_strength,
            "refinement_radius_arcsec": refinement_radius, "refine_steps": refine_steps,
            "cross_fit": bool(cross_fit),
            "mesh_strategy": mesh_strategy,
            "search_radius_arcsec": (inner, outer),
            "seeds": {"source_mesh": 42, "pinn": 42, "cross_fit_source_mesh": 42},
        },
    )


def search_subhalos(
    observation: LensingObservation, *, search_radius=None, grid_size=27, top_k=3,
    seed_strength=0.015, max_strength=0.12, refinement_radius=0.24, refine_steps=35,
    profile_grid_size=15, source_nodes=500, regularization=None,
    proposal="grid", pinn_steps=90,
):
    """Return ranked candidates from a fixed-macro observation.

    The angular strength is for the softened truncated profile used by the
    survey. Scores and improvements are diagnostics, without a detection threshold.
    """
    obs = observation
    if search_radius is None:
        if obs.einstein_radius is None:
            raise ValueError("Specify search_radius for observations without an Einstein radius")
        search_radius = (0.45 * obs.einstein_radius, 1.55 * obs.einstein_radius)
    inner, outer = search_radius
    if not np.isfinite([inner, outer]).all() or not 0 <= inner < outer:
        raise ValueError("Search radii must be finite with 0 <= inner < outer")
    if grid_size < 3 or top_k < 1 or refine_steps < 1:
        raise ValueError("Search needs at least three grid points, a candidate and a refinement step")
    if profile_grid_size != 0 and profile_grid_size < 3:
        raise ValueError("Profile grid size must be zero or at least three")
    if proposal not in ("grid", "pinn") or pinn_steps < 1:
        raise ValueError("Choose 'grid' or 'pinn' and use at least one PINN step")
    if (not np.isfinite([seed_strength, max_strength, refinement_radius]).all()
            or not 0 < seed_strength <= max_strength or refinement_radius <= 0):
        raise ValueError("Search requires positive refinement radius and 0 < seed strength <= maximum")

    problem, baseline, _, rays = _prepare_search(obs, source_nodes, regularization)
    scan_x = scan_y = profile_x = profile_y = np.empty(0)
    scores = gains = np.empty((0, 0))
    residual_potential = None
    if proposal == "pinn":
        residual_potential = propose_residual_peaks(
            obs, rays[2].numpy(), rays[3].numpy(), radius=search_radius, source_nodes=source_nodes,
            regularization=regularization, separation=2 * refinement_radius, top_k=top_k, steps=pinn_steps,
            gain_radius=refinement_radius,
        )
        seeds = [(x, y, max_strength * 0.3, gain, "pinn") for x, y, gain in residual_potential.peaks]
    else:
        offsets = np.linspace(-outer, outer, grid_size)
        scan_x, scan_y = (offsets + center for center in obs.lens_center)
        scores = _radar(problem, baseline, rays, scan_x, scan_y, obs.lens_center, search_radius, seed_strength)

    if proposal == "grid" and profile_grid_size:
        offsets = np.linspace(-outer, outer, profile_grid_size)
        profile_x, profile_y = (offsets + center for center in obs.lens_center)
        gains, strengths = _profile_grid(
            problem, baseline, rays, profile_x, profile_y, obs.lens_center, search_radius,
            (max_strength * 0.3, max_strength * 0.55),
        )
        seeds = [
            (profile_x[i], profile_y[j], strengths[j, i], gains[j, i], "profile_grid")
            for j, i in _grid_peaks(gains, profile_x, profile_y, top_k, 2 * refinement_radius)
        ]
    elif proposal == "grid":
        seeds = [
            (scan_x[i], scan_y[j], seed_strength, scores[j, i], "radar")
            for j, i in _grid_peaks(scores, scan_x, scan_y, top_k, refinement_radius)
        ]

    candidates = []
    for x, y, strength, score, method in seeds:
        bounds = [(x - refinement_radius, x + refinement_radius),
                  (y - refinement_radius, y + refinement_radius), (0.0, 1.0)]
        initial = [x, y, strength / max_strength]
        frozen = _refine(problem, baseline, rays, initial, bounds, max_strength, False, refine_steps)
        start = initial
        if method == "radar" and frozen.strength:
            start = [frozen.x, frozen.y, frozen.strength / max_strength]
        refitted = _refine(problem, baseline, rays, start, bounds, max_strength, True, refine_steps)
        candidates.append(HaloCandidate((float(x), float(y)), float(strength), float(score),
                                        method, frozen, refitted))
    candidates.sort(key=lambda candidate: candidate.refitted.delta_objective, reverse=True)
    return HaloSearchResult(baseline, candidates, scores, scan_x, scan_y,
                                 gains, profile_x, profile_y, residual_potential)
