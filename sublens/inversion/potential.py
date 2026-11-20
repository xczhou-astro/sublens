"""Propose compact lenses from cross-fitted image residuals and a neural potential."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from scipy.fft import next_fast_len
from torch import nn
from .source_plane import _image_psf_support


@dataclass
class ResidualPotential:
    mask: np.ndarray
    baseline_residual: np.ndarray
    fitted_residual: np.ndarray
    convergence: np.ndarray
    peaks: list


class _SineLayer(nn.Linear):
    def forward(self, coordinates):
        return torch.sin(super().forward(coordinates))


def _source_gradient(source, points):
    triangle = source.grid.find_simplex(points)
    inside = triangle >= 0
    simplex = triangle[inside]
    values = source.coefficients[source.grid.simplices[simplex]]
    gradient = np.zeros_like(points)
    gradient[inside] = np.einsum(
        "ni,nij->nj", values[:, :2] - values[:, 2:], source.grid.transform[simplex, :2],
    )
    return gradient


def _psf_guard_radius(psf):
    if psf.size > 10000:
        row,col = np.indices(psf.shape)
        radius_sq = (row-psf.shape[0]//2)**2+(col-psf.shape[1]//2)**2
        energy = np.bincount(radius_sq.ravel(),weights=np.abs(psf).ravel())
        return max(1.,float(np.sqrt(np.searchsorted(np.cumsum(energy),.98*energy.sum()))))
    row, col = np.indices(psf.shape)
    radius = np.hypot(row - psf.shape[0] // 2, col - psf.shape[1] // 2).ravel()
    order = np.argsort(radius)
    enclosed = np.cumsum(np.abs(psf.ravel()[order]))
    return max(1.0, radius[order[np.searchsorted(enclosed, 0.98 * enclosed[-1])]])


def _arc_masks(obs, x, y, psf_guard_max_radius=None):
    # Compare predictions only where a different part of the ring constrains the same source.
    angle = np.arctan2(y - obs.lens_center[1], x - obs.lens_center[0])
    points = np.stack((obs.bx, obs.by), axis=-1)
    guard = _psf_guard_radius(obs.psf)
    if psf_guard_max_radius is not None:
        guard = min(guard,max(1.,psf_guard_max_radius/obs.pixel_scale))
    tolerance = max(0.025, 0.5 * obs.pixel_scale)
    sector = np.floor(((angle + np.pi / 4) % (2 * np.pi)) / (np.pi / 2)).astype(int)
    for index, center in enumerate(np.arange(4) * (np.pi / 2)):
        direction = np.cos(angle - center)
        heldout = obs.mask & (sector == index)
        if not heldout.any():
            continue
        train = obs.mask & (distance_transform_edt(~heldout) > guard)
        other_images = train & (direction <= 0)
        if not other_images.any():
            continue
        distances, _ = cKDTree(points[other_images].reshape(-1, 2)).query(points[heldout])
        covered = heldout.copy()
        covered[heldout] = np.max(distances, axis=1) <= tolerance
        if covered.any():
            yield index, train, covered


def _arc_folds(obs, x, y, source_nodes, regularization, cross_fit=True, mesh_strategy="random",
               psf_guard_max_radius=None):
    points = np.stack((obs.bx, obs.by), axis=-1)
    # The ablation keeps the same scoring footprint but fits the source to all
    # pixels, so only the source/residual construction changes between modes.
    full_fit = None if cross_fit else obs.reconstruct(
        source_nodes=source_nodes, regularization=regularization, mesh_strategy=mesh_strategy,
    )
    for _, train, covered in _arc_masks(obs, x, y, psf_guard_max_radius):
        fit = (obs.reconstruct(fit_mask=train, source_nodes=source_nodes, regularization=regularization,
                               mesh_strategy=mesh_strategy)
               if cross_fit else full_fit)
        gradient = _source_gradient(fit.source, points.reshape(-1, 2)).reshape(points.shape)
        yield covered, fit.residual, gradient


def _smooth_convergence(convergence, pixel_scale):
    return gaussian_filter(convergence, max(0.75, 0.06 / pixel_scale))


def _potential_peaks(convergence, gain, x, y, center, radius, pixel_scale, separation, top_k,
                     *, gain_radius=0.22):
    """Rank maxima by aperture gain, independently of the seed exclusion distance."""
    if (not np.isfinite([gain_radius, separation]).all() or gain_radius <= 0 or separation < 0):
        raise ValueError("Use a positive gain radius and nonnegative candidate separation")
    smoothed = _smooth_convergence(convergence, pixel_scale)
    aperture_size = int(np.ceil(gain_radius / pixel_scale))
    axis = np.arange(-aperture_size, aperture_size + 1) * pixel_scale
    aperture = axis[:, None]**2 + axis**2 <= gain_radius**2
    local_gain = convolve(gain, aperture.astype(float), mode="constant")
    distance = np.hypot(x - center[0], y - center[1])
    eligible = ((smoothed == maximum_filter(smoothed, size=3)) & (smoothed > 0)
                & (local_gain > 0) & (distance >= radius[0]) & (distance <= radius[1]))
    rows, cols = np.nonzero(eligible)
    peaks = []
    for index in np.argsort(-local_gain[rows, cols], kind="stable"):
        row, col = rows[index], cols[index]
        px, py = x[row, col], y[row, col]
        if all(np.hypot(px - cx, py - cy) >= separation for cx, cy, _ in peaks):
            peaks.append((float(px), float(py), float(local_gain[row, col])))
        if len(peaks) == top_k:
            break
    return peaks


@torch.enable_grad()
def propose_residual_peaks(obs, x_rays, y_rays, *, radius, source_nodes, regularization,
                           separation, top_k, steps, cross_fit=True, mesh_strategy="random",
                           gain_radius=0.22, psf_guard_max_radius=None):
    """Fit a single neural potential to first-order, PSF-convolved arc displacements."""
    x, y = x_rays.mean(axis=-1), y_rays.mean(axis=-1)
    folds = list(_arc_folds(obs, x, y, source_nodes, regularization, cross_fit=cross_fit,
                            mesh_strategy=mesh_strategy, psf_guard_max_radius=psf_guard_max_radius))
    if not folds:
        empty = np.full(x.shape, np.nan)
        return ResidualPotential(np.zeros(x.shape, dtype=bool), empty, empty.copy(), empty.copy(), [])

    masks, residuals, gradients = map(np.stack, zip(*folds))
    active = masks.any(axis=0)
    rms = np.where(active, obs.rms, 1.0)
    normalized = np.where(masks, residuals / rms, 0.0)
    baseline_map = normalized.sum(axis=0)
    from ..execution import current_execution
    execution = current_execution()
    device = torch.device(execution.device if execution is not None else "cpu")
    tensor_masks = torch.as_tensor(masks, device=device)
    target = torch.as_tensor(normalized[masks], dtype=torch.float32, device=device)
    gradient = torch.as_tensor(gradients, dtype=torch.float32, device=device)
    weight = torch.as_tensor(1 / rms, dtype=torch.float32, device=device)
    kernel = _image_psf_support(obs.psf/obs.psf.sum(),x.shape)
    transform_size = tuple(n+k-1 for n,k in zip(x.shape,kernel.shape))
    if kernel.size>225:
        transform_size=tuple(next_fast_len(n) for n in transform_size)
    psf = torch.as_tensor(kernel, dtype=torch.float32, device=device)
    psf_fft = torch.fft.rfft2(psf, s=transform_size)
    sy, sx = (k//2 for k in kernel.shape)

    field_scale = radius[1]
    center = np.asarray(obs.lens_center)
    ray_points = (np.stack((x_rays, y_rays), axis=-1) - center) / field_scale
    coordinates = torch.tensor(ray_points, dtype=torch.float32, device=device, requires_grad=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = nn.Sequential(_SineLayer(2, 24), _SineLayer(24, 24), _SineLayer(24, 24), nn.Linear(24, 1))
        with torch.no_grad():
            model[0].weight.uniform_(-9.0, 9.0)
            for layer in (model[1], model[2]):
                layer.weight.uniform_(-np.sqrt(6 / 24), np.sqrt(6 / 24))
            model[-1].weight.zero_()
            model[-1].bias.zero_()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    best_loss = float("inf")
    for _ in range(steps + 1):
        optimizer.zero_grad()
        deflection, = torch.autograd.grad(model(coordinates).sum(), coordinates, create_graph=True)
        deflection = deflection * (0.02 / field_scale)
        native = -(gradient * deflection).sum(dim=-1).mean(dim=-1)
        blurred = torch.fft.irfft2(torch.fft.rfft2(native, s=transform_size) * psf_fft, s=transform_size)
        prediction = blurred[:, sy:sy + x.shape[0], sx:sx + x.shape[1]] * weight
        residual = target - prediction[tensor_masks]
        mean_deflection = deflection.mean(dim=-2)
        smoothness = sum(mean_deflection.diff(dim=dim).square().mean() for dim in (0, 1))
        loss = residual.square().mean() + 50 * deflection.square().mean() + 500 * smoothness
        objective = float(loss.detach())
        if not np.isfinite(objective):
            raise RuntimeError("Neural potential fit has a nonfinite objective")
        if objective < best_loss:
            best_loss = objective
            best_parameters = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_residual = residual.detach().cpu().numpy()
        loss.backward(inputs=tuple(model.parameters()))
        optimizer.step()

    model.load_state_dict(best_parameters)
    native_points = (np.stack((x, y), axis=-1) - center) / field_scale
    native_coordinates = torch.tensor(native_points, dtype=torch.float32, device=device, requires_grad=True)
    gradient_x, gradient_y = torch.autograd.grad(model(native_coordinates).sum(), native_coordinates,
                                                 create_graph=True)[0].unbind(dim=-1)
    dxx = torch.autograd.grad(gradient_x.sum(), native_coordinates, retain_graph=True)[0][..., 0]
    dyy = torch.autograd.grad(gradient_y.sum(), native_coordinates)[0][..., 1]
    convergence = (0.01 * (dxx + dyy) / field_scale**2).detach().cpu().numpy()
    by_fold = np.zeros(masks.shape)
    by_fold[masks] = best_residual
    fitted_map = by_fold.sum(axis=0)
    gain = np.sum(normalized**2 - by_fold**2, axis=0)
    peaks = _potential_peaks(convergence, gain, x, y, obs.lens_center, radius, obs.pixel_scale,
                             separation, top_k, gain_radius=gain_radius)
    return ResidualPotential(active, np.where(active, baseline_map, np.nan),
                             np.where(active, fitted_map, np.nan), convergence, peaks)
