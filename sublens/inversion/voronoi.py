"""
Voronoi Mesh Source Plane Inversion Engine for Strong Gravitational Lensing.

Implements an adaptive, unstructured Voronoi pixelization with Delaunay
regularization for background source reconstruction, providing direct parity
with PyAutoLens / Hilbert Voronoi mesh modeling.
"""
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import scipy.spatial
import scipy.linalg
from scipy.cluster.vq import kmeans2
from scipy.ndimage import convolve
import torch


class VoronoiSourceInverter:
    """
    Unstructured Voronoi mesh source inversion engine.
    
    Parameters
    ----------
    num_cells : int
        Number of Voronoi mesh cells (centroids) in the source plane. Default: 400.
    reg_lambda : float
        Regularization strength for Delaunay graph Laplacian. Default: 200.0.
    seed : int
        Random seed for reproducible centroid clustering. Default: 42.
    """

    def __init__(
        self,
        num_cells: int = 400,
        reg_lambda: float = 200.0,
        seed: int = 42
    ):
        self.num_cells = num_cells
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.s_sol: Optional[np.ndarray] = None
        self.tree: Optional[scipy.spatial.cKDTree] = None
        self.chi2_red: float = 0.0
        self.fitted_model: Optional[np.ndarray] = None
        self.residual_image: Optional[np.ndarray] = None

    def fit(
        self,
        image_arr: np.ndarray,
        psf_arr: np.ndarray,
        noise_arr: np.ndarray,
        fit_mask: np.ndarray,
        beta_x: np.ndarray,
        beta_y: np.ndarray,
        num_cells: Optional[int] = None,
        reg_lambda: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes linear matrix inversion on the unstructured Voronoi mesh.
        """
        K = num_cells or self.num_cells
        lam = reg_lambda if reg_lambda is not None else self.reg_lambda

        img = np.asarray(image_arr, dtype=np.float32)
        psf = np.asarray(psf_arr, dtype=np.float32)
        noise = np.clip(np.asarray(noise_arr, dtype=np.float32), 1e-5, None)
        mask = np.asarray(fit_mask, dtype=bool)
        bx = np.asarray(beta_x, dtype=np.float32)
        by = np.asarray(beta_y, dtype=np.float32)

        crop_size = img.shape[0]
        m_idx = np.where(mask)
        N_data = len(m_idx[0])

        bx_m = bx[m_idx]
        by_m = by[m_idx]
        d_m = img[m_idx]
        n_m = noise[m_idx]

        # 1. Flux-weighted Voronoi seeds
        w = np.clip(d_m / n_m, 0.05, None)**1.6
        w_prob = w / w.sum()
        rng = np.random.RandomState(self.seed)
        sampled = rng.choice(N_data, size=min(1500, N_data), replace=True, p=w_prob)
        seed_data = np.column_stack([bx_m[sampled], by_m[sampled]])
        centroids, _ = kmeans2(seed_data, K, minit="points", iter=15)
        self.centroids = centroids

        # 2. KDTree mapping
        self.tree = scipy.spatial.cKDTree(centroids)
        _, cell_idx_all = self.tree.query(np.column_stack([bx.ravel(), by.ravel()]))
        cell_grid = cell_idx_all.reshape(crop_size, crop_size)

        # 3. PSF-blurred mapping matrix F (N_data, K)
        psf_norm = psf / psf.sum()
        F_matrix = np.zeros((N_data, K), dtype=np.float32)
        for k in range(K):
            M_k = (cell_grid == k).astype(np.float32)
            F_matrix[:, k] = convolve(M_k, psf_norm, mode="constant")[m_idx]

        # 4. Delaunay Laplacian regularization matrix H (K, K)
        delau = scipy.spatial.Delaunay(centroids)
        H = np.zeros((K, K), dtype=np.float32)
        for simplex in delau.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    dist = np.hypot(centroids[u, 0] - centroids[v, 0], centroids[u, 1] - centroids[v, 1])
                    weight = 1.0 / max(dist, 1e-4)
                    H[u, v] -= weight
                    H[v, u] -= weight
                    H[u, u] += weight
                    H[v, v] += weight

        # 5. Normal equations solve
        inv_cov = (1.0 / (n_m**2))
        Ft_Cin_F = (F_matrix.T * inv_cov[None, :]) @ F_matrix
        rhs = (F_matrix.T * inv_cov[None, :]) @ d_m

        A = Ft_Cin_F + lam * H + 1e-4 * np.eye(K, dtype=np.float32)
        s_sol = np.clip(np.linalg.solve(A, rhs), 0.0, None)
        self.s_sol = s_sol

        # Forward prediction and residual
        pred_m = F_matrix @ s_sol
        pred_full = np.zeros_like(img)
        pred_full[m_idx] = pred_m
        res_full = img - pred_full

        chi2_red = float(((res_full[mask] / noise[mask])**2).mean())
        self.chi2_red = chi2_red
        self.fitted_model = pred_full
        self.residual_image = res_full

        peak_idx = int(np.argmax(s_sol))
        return {
            "centroids": centroids,
            "source_vector": s_sol,
            "peak_brightness": float(s_sol[peak_idx]),
            "peak_position": (float(centroids[peak_idx, 0]), float(centroids[peak_idx, 1])),
            "total_flux": float(s_sol.sum()),
            "chi2_red": chi2_red,
            "model_image": pred_full,
            "residual_image": res_full,
            "num_cells": K,
            "lambda": lam
        }

    def rasterize(
        self,
        extent: Tuple[float, float, float, float],
        resolution: int = 201
    ) -> np.ndarray:
        """
        Rasterizes the Voronoi source tessellation onto a uniform 2D grid.
        
        Parameters
        ----------
        extent : (xmin, xmax, ymin, ymax) in arcseconds
        resolution : int, grid size (default 201x201)
        """
        if self.centroids is None or self.s_sol is None or self.tree is None:
            raise RuntimeError("Must call fit() before rasterize().")

        xmin, xmax, ymin, ymax = extent
        xs = np.linspace(xmin, xmax, resolution)
        ys = np.linspace(ymax, ymin, resolution) # upper row is top (+y)
        grid_x, grid_y = np.meshgrid(xs, ys)

        _, cell_indices = self.tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        raster = self.s_sol[cell_indices].reshape(resolution, resolution)
        return raster
