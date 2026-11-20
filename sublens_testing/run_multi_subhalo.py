"""Test PINN discovery of multiple subhalos with joint macro and source refinement."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.signal import fftconvolve
import torch

from sublens import LensingObservation, MacroLensModel, truncated_subhalo_deflection
from sublens.inversion.halo_search import _delaunay_native_image
from sublens.inversion.source_plane import PixelSource, PixelSourceProblem
from sublens.io import observation_hdus

from run_pinn_subhalo import MACRO_PARAMS, NOISE_SEED, simulate_observations


CASES = {
    "opposite_arcs": [(-1.28, 0.61, 0.050), (1.10, -0.75, 0.040)],
    "same_arc": [(-1.28, 0.61, 0.050), (-0.80, 1.05, 0.035)],
    "null": [],
}
MAX_STRENGTH = 0.12
REFINEMENT_RADIUS = 0.35


@dataclass
class JointFit:
    parameters: np.ndarray
    source: PixelSource
    converged: bool
    optimizer_stop: str

    @property
    def halos(self):
        values = self.parameters[3:].reshape(-1, 3).copy()
        values[:, 2] *= MAX_STRENGTH
        return values


class JointRefiner:
    """Optimize all discovered halos and macro nuisance parameters on one source mesh."""

    def __init__(self, observation, source):
        self.obs = observation
        self.problem = PixelSourceProblem(
            observation.image, observation.rms, observation.psf, observation.mask,
            observation.bx, observation.by, source_grid=source.grid,
            regularization=source.regularization,
        )
        self.x, self.y = (torch.as_tensor(a, dtype=torch.float64) for a in observation.image_rays())
        self.base = torch.stack((torch.as_tensor(observation.bx), torch.as_tensor(observation.by)), dim=-1)
        coords = torch.stack((self.x.flatten(), self.y.flatten()), dim=-1)
        with torch.no_grad():
            host, _ = MacroLensModel(**MACRO_PARAMS).double()(coords)
        angle = 2 * MACRO_PARAMS["phi_ext"]
        g1 = MACRO_PARAMS["gamma_ext"] * np.cos(angle)
        g2 = MACRO_PARAMS["gamma_ext"] * np.sin(angle)
        cx, cy = observation.lens_center
        dx, dy = self.x - cx, self.y - cy
        self.shear_x = torch.stack((dx, -dy), dim=-1)
        self.shear_y = torch.stack((dy, dx), dim=-1)
        # The cached host excludes the external shear.
        self.host = host.reshape(*self.x.shape, 2) - g1 * self.shear_x - g2 * self.shear_y

    def rays(self, parameters):
        radius, g1, g2 = parameters[:3]
        rays = (self.base - ((1 + 0.03 * radius)**(MACRO_PARAMS["gamma"] - 1) - 1) * self.host
                - 0.03 * (g1 * self.shear_x + g2 * self.shear_y))
        for x, y, amplitude in parameters[3:].reshape(-1, 3):
            ax, ay = truncated_subhalo_deflection(self.x, self.y, x, y, amplitude * MAX_STRENGTH)
            rays = rays - torch.stack((ax, ay), dim=-1)
        return rays

    def fit(self, initial, anchors, source, maxiter):
        bounds = [(-1.0, 1.0)] * 3
        for x, y in anchors:
            bounds.extend([(x - REFINEMENT_RADIUS, x + REFINEMENT_RADIUS),
                           (y - REFINEMENT_RADIUS, y + REFINEMENT_RADIUS), (0.0, 1.0)])
        best = None

        @torch.enable_grad()
        def objective(values):
            nonlocal best
            parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            bx, by = self.rays(parameters).unbind(dim=-1)
            fitted = self.problem.fit(bx.detach().numpy(), by.detach().numpy(),
                                      initial=source.coefficients)
            if best is None or fitted.objective < best[0]:
                best = fitted.objective, np.array(values), fitted
            residual = fitted.image_model[self.obs.mask] * self.problem.weight - self.problem.target
            image_gradient = np.zeros(self.obs.image.shape)
            image_gradient[self.obs.mask] = 2 * residual * self.problem.weight
            adjoint = fftconvolve(image_gradient, self.problem.psf[::-1, ::-1], mode="same")
            native = _delaunay_native_image(bx, by, fitted.grid, fitted.coefficients)
            gradient, = torch.autograd.grad((native * torch.from_numpy(adjoint)).sum(), parameters)
            return fitted.objective, gradient.numpy()

        outcome = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"maxiter": maxiter, "gtol": 1e-5, "ftol": 1e-9})
        if not outcome.success:
            # Restart once from the best fit with its optimized source as a warm start.
            source = best[2]
            outcome = minimize(
                objective, best[1], method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": 2 * maxiter, "maxls": 40, "gtol": 1e-5, "ftol": 1e-9},
            )
        _, parameters, fitted = best
        return JointFit(parameters, fitted, bool(outcome.success), str(outcome.message))


def discover(observation, output, *, max_halos, min_gain, maxiter):
    """Use PINN peaks to grow the model, with no injected positions or count as inputs."""
    proposal = observation.search_pinn()
    proposal.save_json(output / "proposal_01.json", overwrite=True)
    refiner = JointRefiner(observation, proposal.baseline)
    state = refiner.fit(np.zeros(3), [], proposal.baseline, maxiter)
    null_state = state
    anchors, history = [], []
    count = int(observation.mask.sum())
    stop_reason = "maximum_halo_count"
    for iteration in range(1, max_halos + 1):
        if iteration > 1:
            rays = refiner.rays(torch.from_numpy(state.parameters)).numpy()
            conditional = replace(observation, bx=rays[..., 0], by=rays[..., 1], macro_model=None)
            proposal = conditional.search_pinn()
            proposal.save_json(output / f"proposal_{iteration:02d}.json", overwrite=True)
        trials = []
        for candidate in proposal.candidates:
            fit = candidate.fit
            position = fit.position if fit.strength > 0 else candidate.seed
            strength = fit.strength if fit.strength > 0 else 0.015
            if any(np.hypot(position[0] - x, position[1] - y) < 0.15 for x, y, _ in state.halos):
                continue
            trial_anchors = anchors + [candidate.seed]
            initial = np.r_[state.parameters, position, strength / MAX_STRENGTH]
            trial = refiner.fit(initial, trial_anchors, state.source, maxiter)
            gain = count * (state.source.objective - trial.source.objective)
            print(f"  Iteration {iteration}, peak {candidate.rank}, joint gain {gain:.2f}, "
                  f"converged={trial.converged}", flush=True)
            trials.append((gain, candidate, trial, trial_anchors))
        if not trials:
            stop_reason = "no_new_candidates"
            break
        gain, candidate, trial, trial_anchors = max(trials, key=lambda item: item[0])
        history.append(dict(iteration=iteration, gain=float(gain), seed=list(candidate.seed),
                            accepted=bool(gain >= min_gain), converged=trial.converged,
                            optimizer_stop=trial.optimizer_stop))
        if gain < min_gain:
            stop_reason = "gain_below_threshold"
            break
        state, anchors = trial, trial_anchors
        print(f"  Accepted model with {len(state.halos)} halos {state.halos.tolist()}", flush=True)
    return null_state, state, proposal, history, stop_reason


def evaluate(truth, recovered):
    """Match injected and recovered positions only after discovery has finished."""
    if not truth or not len(recovered):
        return [], list(range(len(recovered)))
    truth_array = np.asarray(truth)
    distances = np.linalg.norm(truth_array[:, None, :2] - recovered[None, :, :2], axis=-1)
    rows, cols = linear_sum_assignment(distances)
    matches = []
    matched = set()
    for row, col in zip(rows, cols):
        if distances[row, col] > 0.15:
            continue
        matched.add(int(col))
        matches.append(dict(
            truth_index=int(row), recovered_index=int(col),
            position_error_mas=float(1000 * distances[row, col]),
            strength_error_percent=float(100 * (recovered[col, 2] / truth_array[row, 2] - 1)),
        ))
    return matches, [index for index in range(len(recovered)) if index not in matched]


def save_diagnostics(obs, source_true, sim, null_state, final, proposal, truth, metrics, output):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from astropy.io import fits

    source_axis = sim.src_coords_1d.cpu().numpy()
    sx, sy = np.meshgrid(source_axis, source_axis)
    reconstructed = final.source.sample(sx, sy)
    truth_source = source_true.cpu().numpy()
    residual_null = obs.image - null_state.source.image_model
    residual_final = obs.image - final.source.image_model
    half = sim.N_img * sim.pixel_scale / 2
    extent = (-half, half, -half, half)
    source_half = sim.L_src + sim.L_src / (sim.N_src - 1)
    source_extent = (-source_half, source_half, -source_half, source_half)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5), layout="constrained")
    axes[0, 0].imshow(obs.image, origin="lower", extent=extent, cmap="magma", vmin=0)
    axes[0, 0].set_title("Observation")
    for ax, residual, title in (
        (axes[0, 1], residual_null, "Macro-only residual"),
        (axes[0, 2], residual_final, "Multi-halo residual"),
    ):
        shown = np.full(obs.image.shape, np.nan)
        shown[obs.mask] = residual[obs.mask] / obs.rms[obs.mask]
        artist = ax.imshow(shown, origin="lower", extent=extent, cmap="RdBu_r", vmin=-4, vmax=4)
        score = np.mean(shown[obs.mask]**2)
        ax.set_title(f"{title}\n" + rf"$\chi^2/N_{{\rm pix}}={score:.3f}$")
    fig.colorbar(artist, ax=axes[0, 1:], shrink=0.75, label="Residual / RMS")
    for ax, field, title in (
        (axes[1, 0], truth_source, "True source"),
        (axes[1, 1], reconstructed, "Reconstructed source"),
    ):
        ax.imshow(field, origin="lower", extent=source_extent, cmap="magma", vmin=0, vmax=truth_source.max())
        ax.set(xlim=(-0.65, 0.65), ylim=(-0.65, 0.65), title=title,
               xlabel="βx [arcsec]", ylabel="βy [arcsec]")
    kappa = gaussian_filter(proposal.residual_potential.convergence, max(0.75, 0.06 / obs.pixel_scale))
    shown = np.where(obs.mask, kappa, np.nan)
    limit = max(float(np.nanpercentile(np.abs(shown), 99)), 1e-6)
    artist = axes[1, 2].imshow(shown, origin="lower", extent=extent, cmap="RdBu_r",
                              vmin=-limit, vmax=limit)
    axes[1, 2].set_title("PINN map at final search step")
    fig.colorbar(artist, ax=axes[1, 2], shrink=0.75, label="Smoothed δκ")
    for ax in [*axes[0], axes[1, 2]]:
        ax.set(xlabel="θx [arcsec]", ylabel="θy [arcsec]")
    for ax in (axes[0, 0], axes[1, 2]):
        for i, (x, y, _) in enumerate(truth):
            ax.scatter(x, y, marker="*", s=130, facecolors="none", edgecolors="#00c9c0",
                       linewidths=1.5, label="Injected" if i == 0 else None)
        for i, (x, y, _) in enumerate(final.halos):
            ax.scatter(x, y, marker="+", s=95, color="#e6ff00", linewidths=1.6,
                       label="Recovered" if i == 0 else None)
        if truth or len(final.halos):
            ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output / "multi_subhalo_diagnostics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    hdus = observation_hdus(obs)
    hdus.extend([
        fits.ImageHDU(null_state.source.image_model, name="MODEL_MACRO"),
        fits.ImageHDU(residual_null, name="RESID_MACRO"),
        fits.ImageHDU(final.source.image_model, name="MODEL"),
        fits.ImageHDU(residual_final, name="RESIDUAL"),
        fits.ImageHDU(reconstructed, name="SOURCE"),
        fits.ImageHDU(proposal.residual_potential.convergence, name="PINN_KAPPA"),
    ])
    table = [fits.Column(name=name, format="D", unit="arcsec", array=final.halos[:, index])
             for index, name in enumerate(("X", "Y", "STRENGTH"))]
    hdus.append(fits.BinTableHDU.from_columns(table, name="HALOS"))
    hdus.writeto(output / "multi_subhalo_result.fits", overwrite=True, checksum=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--max-halos", type=int, default=4)
    parser.add_argument("--min-gain", type=float, default=25.0,
                        help="Experimental stopping threshold on the penalized objective gain")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "multi_subhalo_results")
    args = parser.parse_args()
    if args.max_halos < 1 or args.maxiter < 1 or not np.isfinite(args.min_gain) or args.min_gain <= 0:
        parser.error("Use positive model size, iteration limit and gain threshold")
    matplotlib.use("Agg")
    sim, source, _, _ = simulate_observations(args.device)
    with torch.no_grad():
        alpha_macro, _ = MacroLensModel(**MACRO_PARAMS).to(sim.device)(sim.theta_grid)
    x, y = sim.theta_grid.unbind(dim=1)
    for name in args.case or list(CASES):
        started = perf_counter()
        output = args.output_dir.resolve() / name
        output.mkdir(parents=True, exist_ok=True)
        truth = CASES[name]
        with torch.no_grad():
            total = alpha_macro.clone()
            for hx, hy, strength in truth:
                ax, ay = truncated_subhalo_deflection(x, y, hx, hy, strength)
                total += torch.stack((ax, ay), dim=-1)
            image, _ = sim.simulate(source[None, None], total, seed=NOISE_SEED)
        obs = LensingObservation.from_simulation(sim, image, alpha_macro, macro_model=MACRO_PARAMS, lens_id=name)
        xr, yr = obs.image_rays()
        radius = np.hypot(xr[..., 0], yr[..., 0])
        obs = replace(obs, mask=(radius >= 0.50) & (radius <= 1.80))
        obs.to_fits(output / "simulation.fits", overwrite=True)
        print(f"\nCase {name}", flush=True)
        null_state, final, proposal, history, reason = discover(
            obs, output, max_halos=args.max_halos, min_gain=args.min_gain, maxiter=args.maxiter,
        )
        matches, unmatched = evaluate(truth, final.halos)
        metrics = dict(
            case=name, true_halos=truth, recovered_halos=final.halos.tolist(),
            matches=matches, unmatched_recovered_indices=unmatched,
            missing_truth_count=len(truth) - len(matches), stop_reason=reason,
            converged=final.converged, optimizer_stop=final.optimizer_stop, history=history,
            noise_seed=NOISE_SEED, macro_model_input=MACRO_PARAMS,
            macro_offsets=final.parameters[:3].tolist(),
            config=dict(max_halos=args.max_halos, min_gain=args.min_gain, maxiter=args.maxiter),
            seconds=perf_counter() - started,
        )
        save_diagnostics(obs, source, sim, null_state, final, proposal, truth, metrics, output)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
