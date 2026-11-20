"""Nonnegative semi-linear source inversion at a fixed lens mapping."""

from dataclasses import dataclass
from copy import copy
import math
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.fft import rfftn, irfftn, next_fast_len
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.spatial import Delaunay, cKDTree


def _bilinear_mapping(x, y, axis):
    size = axis.size
    scale = (size - 1) / (axis[-1] - axis[0])
    col = (x.ravel() - axis[0]) * scale
    row = (y.ravel() - axis[0]) * scale
    col0, row0 = np.floor(col).astype(int), np.floor(row).astype(int)
    dx, dy = col - col0, row - row0
    cols = col0[:, None] + [0, 1, 0, 1]
    rows = row0[:, None] + [0, 0, 1, 1]
    weights = np.column_stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy))
    valid = (cols >= 0) & (cols < size) & (rows >= 0) & (rows < size) & (weights > 0)
    return csr_matrix(
        (weights[valid], (np.nonzero(valid)[0], (rows * size + cols)[valid])),
        shape=(x.size, size**2),
    )


def _triangle_mapping(x, y, mesh):
    points = np.column_stack((x.ravel(), y.ravel()))
    simplex = mesh.find_simplex(points)
    inside = simplex >= 0
    transform = mesh.transform[simplex[inside]]
    offset = points[inside] - transform[:, 2]
    weights = np.einsum("ijk,ik->ij", transform[:, :2], offset)
    weights = np.column_stack((weights, 1 - weights.sum(axis=1)))
    return csr_matrix(
        (weights.clip(0, 1).ravel(),
         (np.repeat(np.flatnonzero(inside), 3), mesh.simplices[simplex[inside]].ravel())),
        shape=(x.size, len(mesh.points)),
    )


def _blur_operator(shape, mask, psf):
    """Map native image pixels to masked pixels by convolution."""
    height, width = shape
    row, col = np.nonzero(mask)
    ky, kx = np.nonzero(psf)
    y = row[:, None] + psf.shape[0] // 2 - ky
    x = col[:, None] + psf.shape[1] // 2 - kx
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.broadcast_to(psf[ky, kx], inside.shape)
    return csr_matrix(
        (values[inside], (np.nonzero(inside)[0], (y * width + x)[inside])),
        shape=(len(row), height * width),
    )


def _image_psf_support(psf, shape):
    """Crop impossible offsets without renormalizing a finite image operator."""
    cy,cx = np.array(psf.shape)//2
    ry,rx = min(cy,shape[0]-1),min(cx,shape[1]-1)
    return psf[cy-ry:cy+ry+1,cx-rx:cx+rx+1]


def _gradient_operator(edges, points, size):
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    weights = np.sqrt(np.median(lengths) / lengths / len(edges))
    return csr_matrix(
        (np.column_stack((weights, -weights)).ravel(),
         (np.repeat(np.arange(len(edges)), 2), edges.ravel())),
        shape=(len(edges), size),
    )


def _weighted_centroids(points, weights, nodes, seed, iterations=60):
    """Weighted k-means++ and Lloyd updates without an optional ML dependency."""
    rng = np.random.default_rng(seed)
    centers = [points[rng.choice(len(points), p=weights / weights.sum())]]
    distances = np.sum((points - centers[0])**2, axis=1)
    for _ in range(1, min(nodes, len(points))):
        probability = weights * distances
        if probability.sum() <= np.finfo(float).tiny:
            break
        center = points[rng.choice(len(points), p=probability / probability.sum())]
        centers.append(center)
        distances = np.minimum(distances, np.sum((points - center)**2, axis=1))
    centers = np.array(centers)
    tolerance = 1e-5 * max(float(np.ptp(points, axis=0).max()), 1e-3)
    for _ in range(iterations):
        distance, labels = cKDTree(centers).query(points)
        mass = np.bincount(labels, weights=weights, minlength=len(centers))
        updated = centers.copy()
        occupied = mass > 0
        for axis in (0, 1):
            total = np.bincount(labels, weights=weights * points[:, axis], minlength=len(centers))
            updated[occupied, axis] = total[occupied] / mass[occupied]
        empty = np.flatnonzero(~occupied)
        if empty.size:
            worst = np.argsort(-(weights * distance**2), kind="stable")[:empty.size]
            updated[empty] = points[worst]
        shift = np.max(np.linalg.norm(updated - centers, axis=1))
        centers = updated
        if shift < tolerance:
            break
    return centers


def _adaptive_mesh(bx, by, image, mask, nodes, seed, strategy="random"):
    # The adapt image uses only fitted pixels. Smooth in the observed plane to
    # keep isolated noise peaks from consuming vertices.
    smooth = gaussian_filter(np.where(mask, image, 0), 1.0)
    coverage = gaussian_filter(mask.astype(float), 1.0)
    adapt = np.divide(smooth, coverage, out=np.zeros_like(smooth), where=coverage > 0.1)
    signal = np.maximum(adapt[mask], 0)
    scale = max(signal.max(), np.finfo(float).tiny)
    p = np.broadcast_to((0.01 + (signal / scale)**1.5)[:, None], bx[mask].shape).ravel()
    # The ray samples are uniform within each image pixel.
    points = np.column_stack((bx[mask].ravel(), by[mask].ravel()))
    rng = np.random.default_rng(seed)
    if strategy == "centroidal":
        vertices = _weighted_centroids(points, p, nodes, seed)
    else:
        indices = rng.choice(len(points), size=min(nodes, len(points)), replace=False, p=p / p.sum())
        vertices = points[indices]
    xmin, xmax = bx.min() - 0.01, bx.max() + 0.01
    ymin, ymax = by.min() - 0.01, by.max() + 0.01
    corners = [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]
    return Delaunay(np.unique(np.vstack((vertices, corners)), axis=0))


@dataclass
class PixelSource:
    """A fitted surface-brightness basis and its native image prediction."""

    coefficients: np.ndarray
    image_model: np.ndarray
    method: str
    grid: object
    regularization: float
    objective: float
    iterations: Optional[int] = None
    optimality: Optional[float] = None
    mesh_strategy: Optional[str] = None

    def sample(self, x, y):
        """Evaluate surface brightness at matching arrays of angular coordinates."""
        x, y = np.broadcast_arrays(x, y)
        mapping = _mapping(x, y, self.method, self.grid)
        return (mapping @ self.coefficients).reshape(x.shape)


def _mapping(x, y, method, grid):
    if method == "delaunay":
        return _triangle_mapping(x, y, grid)
    return _bilinear_mapping(x, y, grid)


def _validate_rays(bx, by, shape):
    if bx.ndim != 3 or bx.shape[:2] != shape or by.shape != bx.shape or bx.shape[2] == 0:
        raise ValueError("Rays must have shape (rows, columns, samples)")
    if not np.isfinite(bx).all() or not np.isfinite(by).all():
        raise ValueError("Ray coordinates must be finite")

def _source_grid(image, mask, bx, by, method, source_size, source_nodes, source_half_width, seed,
                 source_grid=None, mesh_strategy="random"):
    if method == "delaunay":
        if source_nodes < 4:
            raise ValueError("The adaptive mesh needs at least four sampled nodes")
        grid = source_grid if source_grid is not None else _adaptive_mesh(
            bx, by, image, mask, source_nodes, seed, strategy=mesh_strategy,
        )
        edges = np.vstack((grid.simplices[:, [0, 1]], grid.simplices[:, [0, 2]], grid.simplices[:, [1, 2]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        points = grid.points
    else:
        if source_size < 2 or not np.isfinite(source_half_width) or source_half_width <= 0:
            raise ValueError("The Cartesian source grid needs at least two pixels and a positive extent")
        grid = np.linspace(-source_half_width, source_half_width, source_size)
        nodes = np.arange(source_size**2).reshape(source_size, source_size)
        edges = np.vstack((
            np.column_stack((nodes[:, :-1].ravel(), nodes[:, 1:].ravel())),
            np.column_stack((nodes[:-1].ravel(), nodes[1:].ravel())),
        ))
        points = np.column_stack(tuple(a.ravel() for a in np.meshgrid(grid, grid)))
    return grid, _gradient_operator(edges, points, len(points))


class PixelSourceProblem:
    """Cache an observation and one source basis for a sequence of lens mappings."""

    def __init__(self, image, rms, psf, mask, bx, by, *, method="delaunay",
                 regularization=None, source_size=96, source_nodes=500,
                 source_half_width=1.0, seed=42, source_grid=None, mesh_strategy="random"):
        image = np.asarray(image, dtype=float)
        rms, psf = np.asarray(rms, dtype=float), np.asarray(psf, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        bx, by = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
        if image.ndim != 2 or rms.shape != image.shape or mask.shape != image.shape:
            raise ValueError("Image, RMS and mask must have the same two-dimensional shape")
        _validate_rays(bx, by, image.shape)
        if (psf.ndim != 2 or not all(n % 2 for n in psf.shape)
                or not np.isfinite(psf).all() or psf.sum() <= 0):
            raise ValueError("PSF must be finite with odd dimensions and a positive sum")
        if (not mask.any() or not np.isfinite(image[mask]).all()
                or not np.isfinite(rms[mask]).all() or (rms[mask] <= 0).any()):
            raise ValueError("Fitted pixels need finite flux and positive finite RMS")
        if method not in ("delaunay", "cartesian"):
            raise ValueError("Source method must be 'delaunay' or 'cartesian'")
        if mesh_strategy not in ("random", "centroidal"):
            raise ValueError("mesh_strategy must be 'random' or 'centroidal'")
        if source_grid is not None and (method != "delaunay" or not isinstance(source_grid, Delaunay)):
            raise ValueError("An existing source_grid must be a Delaunay triangulation")
        if regularization is None:
            regularization = 0.002 if method == "delaunay" else 1.0
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("Regularization must be positive and finite")

        self.method = method
        self.mesh_strategy = mesh_strategy if method == "delaunay" else None
        self.regularization = regularization
        self.shape = image.shape
        self.mask = mask
        self.psf = psf / psf.sum()
        # Offsets beyond the native image diameter cannot contribute to any
        # in-frame prediction. Keep the original normalization when cropping.
        self.convolution_psf = _image_psf_support(self.psf,image.shape)
        self.grid, gradient = _source_grid(
            image, mask, bx, by, method, source_size, source_nodes, source_half_width, seed,
            source_grid, mesh_strategy=mesh_strategy,
        )
        self.size = gradient.shape[1]
        self.flux_scale = max(float(np.percentile(image[mask], 98)), float(np.median(rms[mask])))
        self.weight = 1 / (rms[mask] * np.sqrt(mask.sum()))
        self.target = image[mask] * self.weight
        self._blur = None
        if int(mask.sum())*np.count_nonzero(self.convolution_psf) <= 2_000_000:
            self._blur = _blur_operator(image.shape, mask, self.convolution_psf)
        else:
            self._initialize_fft()
        self._gradient_gram = (gradient.T @ gradient).tocsr()
        self._prior = (regularization*self._gradient_gram).tocsr()
        self._ridge = 0.01 * regularization / self.size
        self._prior_diagonal = self._prior.diagonal() + self._ridge
        self._bounds = [(0, None)] * self.size
        self._operator_cache = {}

    def _initialize_fft(self):
        # Only same-frame outputs are needed: period n+radius is sufficient.
        # Embed signed PSF offsets modulo that period, avoiding a 3*n FFT for
        # kernels whose contributing support spans the whole native image.
        radii=tuple(n//2 for n in self.convolution_psf.shape)
        self._fft_shape=tuple(next_fast_len(n+r) for n,r in zip(self.shape,radii))
        embedded=np.zeros(self._fft_shape)
        iy=(np.arange(self.convolution_psf.shape[0])-radii[0])%self._fft_shape[0]
        ix=(np.arange(self.convolution_psf.shape[1])-radii[1])%self._fft_shape[1]
        embedded[np.ix_(iy,ix)]=self.convolution_psf
        self._psf_fft=rfftn(embedded)

    def with_regularization(self, regularization):
        """Reuse fixed geometry/PSF operators when comparing source priors."""
        if not np.isfinite(regularization) or regularization<=0:
            raise ValueError("Regularization must be positive and finite")
        result=copy(self)
        result.regularization=regularization
        result._prior=(regularization*self._gradient_gram).tocsr()
        result._ridge=.01*regularization/self.size
        result._prior_diagonal=result._prior.diagonal()+result._ridge
        return result

    def _prepared_operator(self,bx,by):
        cache=self._operator_cache
        if (cache and np.array_equal(bx,cache["bx"]) and np.array_equal(by,cache["by"])):
            return cache["native"],cache["operator"]
        native=self._native_mapping(bx,by)
        operator=self._weighted_operator(native)
        cache.clear()
        cache.update(bx=np.array(bx,copy=True),by=np.array(by,copy=True),native=native,operator=operator)
        return native,operator

    def _native_mapping(self, bx, by):
        bx, by = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
        _validate_rays(bx, by, self.shape)
        n_samples = bx.shape[2]
        rays = _mapping(bx, by, self.method, self.grid).tocoo()
        return coo_matrix(
            (rays.data / n_samples, (rays.row // n_samples, rays.col)),
            shape=(np.prod(self.shape), self.size),
        ).tocsr()

    def render(self, bx, by, coefficients):
        """Render a fixed source at new rays with the same pixel integration and PSF."""
        native = self._native_mapping(bx, by)
        return fftconvolve((native @ coefficients).reshape(self.shape), self.convolution_psf, mode="same")

    def _weighted_operator(self, native):
        if self._blur is not None:
            return (self._blur @ native).multiply((self.flux_scale * self.weight)[:, None]).tocsr()
        # Convolve small batches of basis images. Memory scales with source
        # vertices rather than the number of pixels times the PSF stamp area.
        operator = np.empty((int(self.mask.sum()),self.size))
        for start in range(0,self.size,32):
            stop = min(self.size,start+32)
            basis = native[:,start:stop].toarray().reshape(*self.shape,stop-start)
            frequency = rfftn(basis,s=self._fft_shape,axes=(0,1))
            blurred = irfftn(frequency*self._psf_fft[...,None],s=self._fft_shape,axes=(0,1))
            operator[:,start:stop] = blurred[:self.shape[0],:self.shape[1]][self.mask]
        return operator*(self.flux_scale*self.weight)[:,None]

    def fit(self, bx, by, *, initial=None):
        """Solve for nonnegative source coefficients at new rays on the cached basis."""
        from ..execution import current_execution
        if current_execution() is not None:
            from .torch_source import torch_problem
            engine = torch_problem(self)
            return engine.export(engine.fit(bx, by, initial=initial))
        native,operator = self._prepared_operator(bx,by)
        diagonal = (np.asarray(operator.power(2).sum(axis=0)).ravel() if issparse(operator)
                    else np.einsum("ij,ij->j",operator,operator)) + self._prior_diagonal
        scale = 1 / np.sqrt(diagonal)
        transpose = operator.T.tocsr() if issparse(operator) else operator.T

        def loss(coordinates):
            source = scale * coordinates
            residual = operator @ source - self.target
            penalty = self._prior @ source + self._ridge * source
            return residual @ residual + source @ penalty, 2 * scale * (transpose @ residual + penalty)

        start = np.zeros(self.size) if initial is None else initial / (self.flux_scale * scale)
        result = minimize(loss, start, method="L-BFGS-B", jac=True, bounds=self._bounds,
                          options={"maxiter": 1000, "gtol": 1e-5, "ftol": 1e-9})
        if not result.success and np.isfinite(result.fun):
            # Nearby mesh mappings can have weakly constrained coefficients.
            # Continue from the best feasible iterate with a longer history.
            result = minimize(loss, result.x, method="L-BFGS-B", jac=True, bounds=self._bounds,
                              options={"maxiter": 2000, "maxcor": 25, "gtol": 1e-5, "ftol": 1e-9})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Source inversion failed: {result.message}")
        coefficients = self.flux_scale * scale * result.x
        prediction = fftconvolve((native @ coefficients).reshape(self.shape), self.convolution_psf, mode="same")
        return PixelSource(coefficients, prediction, self.method, self.grid,
                           self.regularization, float(result.fun), mesh_strategy=self.mesh_strategy)

    def fit_torch(self, bx, by, *, device="cpu", max_steps=2000, tolerance=1e-5):
        """Fit nonnegative coefficients by accelerated projected gradient in PyTorch.

        Geometry and sparse operator assembly run on CPU. The quadratic solve
        runs on device. The result records the final projected gradient error.
        """
        if max_steps < 1 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("A positive iteration limit and tolerance are required")
        from ..execution import current_execution, resolve_device
        if current_execution() is not None:
            from .torch_source import torch_problem
            engine = torch_problem(self)
            return engine.export(engine.fit(bx, by))
        native,operator = self._prepared_operator(bx,by)
        device = resolve_device(device)
        dtype = torch.float32 if device.type == "mps" else torch.float64

        # Keep the potentially large pixel-to-source operator sparse.
        gram_array = ((operator.T @ operator + self._prior).toarray() if issparse(operator)
                      else operator.T @ operator + self._prior.toarray())
        gram_array.flat[::self.size + 1] += self._ridge
        gram = torch.as_tensor(gram_array, dtype=dtype, device=device)
        scale = gram.diagonal().rsqrt()
        gram = scale[:, None] * gram * scale[None, :]
        rhs = scale * torch.as_tensor(operator.T @ self.target, dtype=dtype, device=device)
        lipschitz = 1.01 * torch.linalg.eigvalsh(gram.cpu())[-1].item()

        current = torch.zeros_like(rhs)
        extrapolated = current
        momentum = 1.0
        grad_scale = rhs.abs().max().clamp_min(torch.finfo(dtype).eps)
        for step in range(1, max_steps + 1):
            nxt = (extrapolated - (gram @ extrapolated - rhs) / lipschitz).clamp_min(0)
            new_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = nxt + ((momentum - 1) / new_momentum) * (nxt - current)
            current, momentum = nxt, new_momentum
            if step % 50 == 0 or step == max_steps:
                projected = (current - (gram @ current - rhs) / lipschitz).clamp_min(0)
                optimality = (lipschitz * (projected - current).abs().max() / grad_scale).item()
                if optimality < tolerance:
                    break

        normalized = (scale * current).cpu().numpy()
        if not np.isfinite(normalized).all():
            raise RuntimeError("PyTorch source inversion produced nonfinite coefficients")
        residual = operator @ normalized - self.target
        penalty = self._prior @ normalized + self._ridge * normalized
        objective = residual @ residual + normalized @ penalty
        coefficients = self.flux_scale * normalized
        prediction = fftconvolve((native @ coefficients).reshape(self.shape), self.convolution_psf, mode="same")
        return PixelSource(coefficients, prediction, self.method, self.grid,
                           self.regularization, float(objective), step, optimality, self.mesh_strategy)


def fit_pixel_source(image, rms, psf, mask, bx, by, **options):
    """Fit a source from ray arrays of shape (rows, columns, samples)."""
    return PixelSourceProblem(image, rms, psf, mask, bx, by, **options).fit(bx, by)
