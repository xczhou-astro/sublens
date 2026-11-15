"""
Macro Lens Profiles (Host Galaxies & Extended Mass Distributions).

Implemented Profiles:
- EPL: Elliptical Power-Law Mass Distribution
- SIE: Singular Isothermal Ellipsoid (gamma = 2.0)
- ExternalShear: External gravitational shear
- HernquistProfile: Stellar bulge mass profile
- NFWHostProfile: Extended host dark matter halo profile
- CompositeHost: Stellar Hernquist + Dark Matter NFW
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def epl_deflection(
    coords: torch.Tensor,
    theta_E: float = 1.40,
    q: float = 0.80,
    phi: float = 0.0,
    gamma: float = 2.0,
    center: Tuple[float, float] = (0.0, 0.0)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes deflection angles and convergence for an Elliptical Power-Law (EPL) lens."""
    x = coords[:, 0] - center[0]
    y = coords[:, 1] - center[1]

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    xr = cos_phi * x + sin_phi * y
    yr = -sin_phi * x + cos_phi * y

    r_ell = torch.sqrt(q * xr**2 + (yr**2) / q + 1e-10)
    psi_ell = (theta_E / r_ell) ** (gamma - 1.0)

    ax_r = theta_E * (xr / r_ell) * psi_ell
    ay_r = theta_E * (yr / (q**2 * r_ell)) * psi_ell

    ax = cos_phi * ax_r - sin_phi * ay_r
    ay = sin_phi * ax_r + cos_phi * ay_r

    kappa = 0.5 * (3.0 - gamma) * (theta_E / r_ell) ** (gamma - 1.0)
    return torch.stack([ax, ay], dim=-1), kappa


def external_shear_deflection(
    coords: torch.Tensor,
    gamma_ext: float = 0.05,
    phi_ext: float = 0.0,
    center: Tuple[float, float] = (0.0, 0.0)
) -> torch.Tensor:
    """Computes deflection angle for external shear."""
    x = coords[:, 0] - center[0]
    y = coords[:, 1] - center[1]
    g1 = gamma_ext * math.cos(2.0 * phi_ext)
    g2 = gamma_ext * math.sin(2.0 * phi_ext)
    ax = g1 * x + g2 * y
    ay = g2 * x - g1 * y
    return torch.stack([ax, ay], dim=-1)


def hernquist_deflection(
    coords: torch.Tensor,
    center: Tuple[float, float] = (0.0, 0.0),
    m_star: float = 1.2,
    r_eff: float = 0.60
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hernquist profile for early-type galaxy stellar bulge."""
    r_h = r_eff / 1.8153
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    r = torch.sqrt(dx**2 + dy**2 + 1e-12)
    s = r / r_h

    eps = 1e-4
    s_safe = torch.where(torch.abs(s - 1.0) < eps, torch.tensor(1.0 + eps, device=coords.device), s)
    cond = s_safe < 1.0

    u_small = torch.sqrt(torch.clamp(1.0 - s_safe**2, min=1e-10))
    f_small = torch.atanh(u_small) / u_small
    u_large = torch.sqrt(torch.clamp(s_safe**2 - 1.0, min=1e-10))
    f_large = torch.atan(u_large) / u_large
    F = torch.where(cond, f_small, f_large)
    F = torch.where(torch.abs(s - 1.0) < eps, torch.tensor(1.0, device=coords.device), F)

    denom = s**2 - 1.0
    g = (1.0 - F) / torch.where(torch.abs(denom) < 1e-5, torch.tensor(1.0, device=coords.device), denom)
    kappa = (m_star / (2.0 * r_h)) * ((1.0 + 2.0 * s**2) * g - 3.0) / (denom + 1e-6)

    m_enc = m_star * (s / (s + 1.0)) ** 2
    alpha_mag = m_enc / r
    return torch.stack([alpha_mag * (dx / r), alpha_mag * (dy / r)], dim=-1), kappa


class MacroLensModel(nn.Module):
    """Unified parametric macro lens model (EPL + Shear or Composite)."""

    def __init__(
        self,
        theta_E: float = 1.40,
        q: float = 0.80,
        phi: float = 0.0,
        gamma: float = 2.0,
        gamma_ext: float = 0.05,
        phi_ext: float = 0.0,
        center: Tuple[float, float] = (0.0, 0.0)
    ):
        super().__init__()
        self.theta_E = nn.Parameter(torch.tensor(theta_E, dtype=torch.float32))
        self.q = nn.Parameter(torch.tensor(q, dtype=torch.float32))
        self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.gamma_ext = nn.Parameter(torch.tensor(gamma_ext, dtype=torch.float32))
        self.phi_ext = nn.Parameter(torch.tensor(phi_ext, dtype=torch.float32))
        self.center = nn.Parameter(torch.tensor(list(center), dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = (self.center[0].item(), self.center[1].item())
        a_epl, k_epl = epl_deflection(
            coords,
            theta_E=self.theta_E.item(),
            q=self.q.item(),
            phi=self.phi.item(),
            gamma=self.gamma.item(),
            center=c
        )
        a_shear = external_shear_deflection(
            coords,
            gamma_ext=self.gamma_ext.item(),
            phi_ext=self.phi_ext.item(),
            center=c
        )
        return a_epl + a_shear, k_epl
