"""
Gravitational Lensing Forward Simulation Engine.

Handles:
- Coordinate grid creation on image and source planes
- Total deflection evaluation: alpha_total = alpha_macro + alpha_subhalos
- Ray tracing: beta = theta - alpha_total
- High-fidelity interpolation rendering via PyTorch grid_sample
- PSF convolution (Space telescope diffraction / ground AO)
- Realistic instrumental noise injection
"""

import math
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_gaussian_psf(
    kernel_size: int = 15,
    sigma_arcsec: float = 0.045,
    pixel_scale: float = 0.05,
    device: str = "cpu"
) -> torch.Tensor:
    """Creates a normalized 2D Gaussian PSF kernel."""
    sigma_pix = sigma_arcsec / pixel_scale
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    k = torch.exp(-0.5 * (xx**2 + yy**2) / (sigma_pix**2))
    return (k / k.sum()).view(1, 1, kernel_size, kernel_size)


class LensingSimulator:
    """Configures observational grids and performs forward ray tracing."""

    def __init__(
        self,
        N_img: int = 120,
        pixel_scale: float = 0.05,
        N_src: int = 64,
        L_src: float = 1.6,
        psf_fwhm: float = 0.09,
        noise_std: float = 0.012,
        psf_kernel: Optional[Union[torch.Tensor, np.ndarray]] = None,
        noise_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
        device: str = "cpu"
    ):
        self.N_img = N_img
        self.pixel_scale = pixel_scale
        self.N_src = N_src
        self.L_src = L_src
        self.noise_std = noise_std
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Image plane celestial coordinates (arcsec)
        self.L_img = (N_img - 1) / 2.0 * pixel_scale
        coords_1d = torch.linspace(-self.L_img, self.L_img, N_img, device=self.device)
        y_grid, x_grid = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
        self.theta_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
        self.extent_img = [-self.L_img, self.L_img, -self.L_img, self.L_img]
        self.coords_1d = coords_1d

        # PSF kernel setup: Supports custom 2D matrix (from FITS/ePSF) or parametric Gaussian
        if psf_kernel is not None:
            if isinstance(psf_kernel, np.ndarray):
                pk = torch.tensor(psf_kernel, dtype=torch.float32, device=self.device)
            else:
                pk = psf_kernel.to(self.device).float()
            if pk.dim() == 2:
                pk = pk.view(1, 1, pk.shape[0], pk.shape[1])
            self.psf_kernel = pk / pk.sum()
        else:
            psf_sigma = psf_fwhm / 2.355
            self.psf_kernel = build_gaussian_psf(
                kernel_size=15,
                sigma_arcsec=psf_sigma,
                pixel_scale=pixel_scale,
                device=self.device
            )
        self.psf_pad = self.psf_kernel.shape[-1] // 2

        # Noise map setup: Supports pixel-wise RMS/Weight map or uniform scalar
        if noise_map is not None:
            if isinstance(noise_map, np.ndarray):
                nm = torch.tensor(noise_map, dtype=torch.float32, device=self.device)
            else:
                nm = noise_map.to(self.device).float()
            if nm.dim() == 2:
                nm = nm.view(1, 1, nm.shape[0], nm.shape[1])
            self.noise_map = torch.clamp(nm, min=1e-6)
        else:
            self.noise_map = torch.full((1, 1, N_img, N_img), float(noise_std), device=self.device)

        # Source plane grid
        src_coords_1d = torch.linspace(-L_src, L_src, N_src, device=self.device)
        self.src_coords_1d = src_coords_1d
        self.extent_src = [-L_src, L_src, -L_src, L_src]

    def render(self, source_img: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Interpolates source image onto ray-traced coordinates beta."""
        u = beta[:, 0].view(1, self.N_img, self.N_img) / self.L_src
        v = beta[:, 1].view(1, self.N_img, self.N_img) / self.L_src
        grid = torch.stack([u, v], dim=-1)
        return F.grid_sample(source_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    def convolve_psf(self, image: torch.Tensor) -> torch.Tensor:
        """Convolves an image tensor with the instrument PSF."""
        if image.dim() == 2:
            image = image.view(1, 1, self.N_img, self.N_img)
        return F.conv2d(image, self.psf_kernel, padding=self.psf_pad)

    def simulate(
        self,
        source_img: torch.Tensor,
        alpha_total: torch.Tensor,
        add_noise: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes full forward simulation pipeline:
        Ray tracing -> Rendering -> PSF convolution -> Noise injection.
        """
        beta = self.theta_grid - alpha_total
        img_clean = self.render(source_img, beta)
        img_conv = self.convolve_psf(img_clean)

        if add_noise:
            if seed is not None:
                torch.manual_seed(seed)
            noise = self.noise_std * torch.randn_like(img_conv)
            img_obs = img_conv + noise
        else:
            img_obs = img_conv

        return img_obs, img_conv
