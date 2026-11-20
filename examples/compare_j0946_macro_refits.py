"""Compare fixed and refitted macromodels on held-out arcs of real J0946."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from sublens.inversion.source_plane import PixelSourceProblem
from sublens.survey.euclid import _halo_deflection
from sublens.survey.euclid_pinn import _arc_masks, _source_gradient

from .j0946_observation import J0946_EPL_SLOPE, J0946_SHEAR, load_j0946_observation


RADIUS_FRACTION = 0.03
SHEAR_OFFSET = 0.03
MAX_HALO_STRENGTH = 0.12


def candidate_positions(path, radar_rank, reference):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["method"] == "refitted" and row["proposal_method"] == "radar"]
    ranked = sorted(rows, key=lambda row: float(row["proposal_score"]), reverse=True)
    if not 1 <= radar_rank <= len(ranked):
        raise ValueError(f"Radar rank {radar_rank} is outside the {len(ranked)} available candidates")
    selected = ranked[radar_rank - 1]
    return {
        "no_halo": None,
        "published_position": tuple(reference),
        "radar_seed": tuple(float(selected[key]) for key in ("seed_x", "seed_y")),
        "wide_refit": tuple(float(selected[key]) for key in ("x", "y")),
    }


class J0946Rays:
    """Vary EPL normalization and external shear while keeping the host shape fixed."""

    def __init__(self, observation, positions):
        x, y = observation.image_rays()
        self.x, self.y = x.mean(axis=-1), y.mean(axis=-1)
        self.base = np.stack((observation.bx, observation.by), axis=-1)
        dx, dy = x - observation.lens_center[0], y - observation.lens_center[1]
        self.shear_x = np.stack((dx, -dy), axis=-1)
        self.shear_y = np.stack((dy, dx), axis=-1)
        self.host = (np.stack((x, y), axis=-1) - self.base
                     - J0946_SHEAR[0] * self.shear_x - J0946_SHEAR[1] * self.shear_y)
        self.halos = {}
        x_tensor, y_tensor = torch.from_numpy(x), torch.from_numpy(y)
        for case, position in positions.items():
            if position is not None:
                ax, ay = _halo_deflection(x_tensor, y_tensor, *position, 1.0)
                self.halos[case] = MAX_HALO_STRENGTH * np.stack((ax.numpy(), ay.numpy()), axis=-1)

    def at(self, parameters, case):
        radius, g1, g2, amplitude = parameters
        ratio = 1 + RADIUS_FRACTION * radius
        slope = J0946_EPL_SLOPE - 1
        derivatives = [-RADIUS_FRACTION * slope * ratio**(slope - 1) * self.host,
                       -SHEAR_OFFSET * self.shear_x, -SHEAR_OFFSET * self.shear_y]
        rays = (self.base - (ratio**slope - 1) * self.host
                + g1 * derivatives[1] + g2 * derivatives[2])
        if case in self.halos:
            rays -= amplitude * self.halos[case]
            derivatives.append(-self.halos[case])
        return rays, derivatives


def fit_hypothesis(problem, geometry, case, allow_macro, start, null_source, maxiter):
    initial = start.copy()
    active = list(range(3)) if allow_macro else []
    if case in geometry.halos:
        active.append(3)
        initial[3] = 0.035 / MAX_HALO_STRENGTH
    if not active:
        return null_source, start, 0, True, "fixed mapping"

    best_source, best_parameters = null_source, start.copy()

    def objective(free):
        nonlocal best_source, best_parameters
        parameters = initial.copy()
        parameters[active] = free
        rays, derivatives = geometry.at(parameters, case)
        source = problem.fit(rays[..., 0], rays[..., 1], initial=best_source.coefficients)
        if source.objective < best_source.objective:
            best_source, best_parameters = source, parameters
        residual = source.image_model[problem.mask] * problem.weight - problem.target
        image_gradient = np.zeros(problem.shape)
        image_gradient[problem.mask] = 2 * residual * problem.weight
        adjoint = fftconvolve(image_gradient, problem.psf[::-1, ::-1], mode="same")
        gradient = _source_gradient(source, rays.reshape(-1, 2)).reshape(rays.shape)
        backprojection = gradient * adjoint[..., None, None] / rays.shape[-2]
        jacobian = [np.sum(backprojection * derivatives[i]) for i in active]
        return source.objective, np.asarray(jacobian)

    bounds = [(-1, 1) if i < 3 else (0, 1) for i in active]
    result = minimize(objective, initial[active], jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter, "gtol": 2e-4, "ftol": 1e-7})
    return best_source, best_parameters, result.nfev, bool(result.success), str(result.message)


def compare_sector(obs, geometry, positions, sector, train, heldout, maxiter):
    problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, train, obs.bx, obs.by)
    baseline = problem.fit(obs.bx, obs.by)
    rows = []
    for mode in ("fixed_macro", "refitted_macro"):
        null_source, null_parameters = baseline, np.zeros(4)
        for case, position in positions.items():
            source, parameters, evaluations, converged, stop = fit_hypothesis(
                problem, geometry, case, mode == "refitted_macro", null_parameters, null_source, maxiter,
            )
            if case == "no_halo":
                null_source, null_parameters = source, parameters
            residual = (obs.image[heldout] - source.image_model[heldout]) / obs.rms[heldout]
            rows.append({
                "sector": sector,
                "mode": mode,
                "case": case,
                "x": position[0] if position is not None else None,
                "y": position[1] if position is not None else None,
                "n_train": int(train.sum()),
                "n_heldout": int(heldout.sum()),
                "train_objective": source.objective,
                "heldout_chi2": residual @ residual,
                "heldout_mean_r2": np.mean(residual**2),
                "theta_e": obs.einstein_radius * (1 + RADIUS_FRACTION * parameters[0]),
                "shear_1": J0946_SHEAR[0] + SHEAR_OFFSET * parameters[1],
                "shear_2": J0946_SHEAR[1] + SHEAR_OFFSET * parameters[2],
                "strength": MAX_HALO_STRENGTH * parameters[3],
                "evaluations": evaluations,
                "converged": converged,
                "optimizer_stop": stop,
            })
            print(f"{sector} {mode:15s} {case:19s} held-out {rows[-1]['heldout_mean_r2']:.4f} "
                  f"train {source.objective:.4f} evaluations {evaluations} converged {converged}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j0946-bundle", required=True, type=Path)
    parser.add_argument("--radar-candidates", required=True, type=Path)
    parser.add_argument("--radar-rank", type=int, default=2)
    parser.add_argument("--reference-position", required=True, type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--sector", type=int, choices=range(4), action="append")
    parser.add_argument("--maxiter", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error("maxiter must be positive")
    obs = load_j0946_observation(args.j0946_bundle)
    positions = candidate_positions(args.radar_candidates, args.radar_rank, args.reference_position)
    geometry = J0946Rays(obs, positions)
    rows = []
    for sector, train, heldout in _arc_masks(obs, geometry.x, geometry.y):
        if args.sector is None or sector in args.sector:
            rows.extend(compare_sector(obs, geometry, positions, sector, train, heldout, args.maxiter))
    if not rows:
        parser.error("no selected sector has shared source coverage")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
