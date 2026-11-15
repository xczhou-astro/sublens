"""
Source Galaxy Surface Brightness Profiles.

Supports:
- Parametric Profiles: Single & Multi-component Sersic, Elliptical Gaussians
- Complex Non-Parametric Astrophysics:
  * Exponential disk with spiral arms
  * Compact star-forming HII knots
  * Turbulent fractal ISM (Kolmogorov power spectrum)
  * Asymmetric dust extinction lanes
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np


def sersic_profile(
    sx: torch.Tensor,
    sy: torch.Tensor,
    center: Tuple[float, float] = (0.0, 0.0),
    i_eff: float = 1.0,
    r_eff: float = 0.20,
    n_sersic: float = 1.0,
    q: float = 0.80,
    phi: float = 0.0
) -> torch.Tensor:
    """Standard elliptical Sersic brightness distribution."""
    dx = sx - center[0]
    dy = sy - center[1]
    cos_p = math.cos(phi)
    sin_p = math.sin(phi)
    xr = cos_p * dx + sin_p * dy
    yr = -sin_p * dx + cos_p * dy
    r_ell = torch.sqrt(q * xr**2 + (yr**2) / q + 1e-10)
    b_n = 2.0 * n_sersic - 1.0 / 3.0 + 4.0 / (405.0 * n_sersic)
    return i_eff * torch.exp(-b_n * ((r_ell / r_eff)**(1.0 / n_sersic) - 1.0))


def generate_complex_source(
    N_src: int = 64,
    L_src: float = 1.6,
    seed: int = 42,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a realistic, highly structured star-forming spiral galaxy:
    - Asymmetric exponential stellar disk
    - 2 logarithmic spiral arms
    - 6 compact clumpy star-forming regions (HII knots)
    - Turbulent fractal ISM gas fluctuations
    - Dust extinction lane
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    coords_1d = torch.linspace(-L_src, L_src, N_src, device=device)
    sy, sx = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
    r_src = torch.sqrt(sx**2 + sy**2 + 1e-8)
    theta_src = torch.atan2(sy, sx)

    # 1. Exponential disk
    I_disk = 1.20 * torch.exp(-torch.sqrt((sx + 0.05)**2 / 0.8 + (sy - 0.08)**2 * 0.8) / 0.25)

    # 2. Spiral arms
    spiral_pitch = 0.35
    phase1 = theta_src - torch.log(r_src / 0.15 + 1e-5) / spiral_pitch
    phase2 = (theta_src + math.pi) - torch.log(r_src / 0.15 + 1e-5) / spiral_pitch
    arm1 = torch.exp(-0.5 * (torch.sin(phase1)**2) / (0.18**2)) * torch.exp(-r_src / 0.40)
    arm2 = torch.exp(-0.5 * (torch.sin(phase2)**2) / (0.18**2)) * torch.exp(-r_src / 0.40)
    I_spiral = 0.80 * (arm1 + arm2) * (r_src > 0.06).float()

    # 3. Compact star-forming knots
    knots = [
        (-0.12, +0.18, 0.045, 1.10),
        (+0.15, -0.10, 0.038, 0.95),
        (+0.22, +0.12, 0.040, 0.85),
        (-0.20, -0.15, 0.042, 0.90),
        (+0.02, +0.25, 0.035, 0.75),
        (-0.08, -0.28, 0.038, 0.70),
    ]
    I_knots = torch.zeros_like(I_disk)
    for kx, ky, k_sigma, k_amp in knots:
        dist_sq = (sx - kx)**2 + (sy - ky)**2
        I_knots += k_amp * torch.exp(-0.5 * dist_sq / (k_sigma**2))

    # 4. Turbulent fractal ISM
    freqs = torch.fft.fftfreq(N_src, d=(2 * L_src / N_src)).to(device)
    fy, fx = torch.meshgrid(freqs, freqs, indexing='ij')
    k_perp = torch.sqrt(fx**2 + fy**2)
    k_perp[0, 0] = 1e-5
    power_spec = 1.0 / (k_perp**(11.0 / 6.0))
    power_spec[0, 0] = 0.0
    random_phase = torch.randn(N_src, N_src, dtype=torch.cfloat, device=device)
    turb_field = torch.real(torch.fft.ifft2(random_phase * power_spec))
    turb_field = (turb_field - turb_field.mean()) / (turb_field.std() + 1e-8)
    I_turb = 0.20 * torch.clamp(turb_field, min=-1.5, max=2.0) * torch.exp(-r_src / 0.35)

    # 5. Dust lane extinction
    dust_x = sx * math.cos(0.4) + sy * math.sin(0.4)
    dust_attenuation = 1.0 - 0.45 * torch.exp(-0.5 * ((dust_x + 0.04)**2) / (0.05**2)) * torch.sigmoid(r_src / 0.1)

    # Total emitter
    I_total = (I_disk + I_spiral + I_knots + I_turb) * dust_attenuation
    I_total = torch.clamp(I_total, min=0.0)

    # Smooth outer aperture
    aperture = torch.exp(-0.5 * (r_src**2) / (0.95**2))
    I_final = (I_total * aperture).view(1, 1, N_src, N_src)
    return I_final, coords_1d
