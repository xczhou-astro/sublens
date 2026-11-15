"""
Dark Matter Subhalo Profiles and Multi-Subhalo Populations.

Profiles:
- NFWSubhalo: Navarro-Frenk-White profile
- SISSubhalo: Singular Isothermal Sphere profile
- PseudoJaffeSubhalo: Truncated pseudo-Jaffe profile
- MultiSubhaloPopulation: Container for multiple discrete subhalos
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def nfw_subhalo_deflection(
    coords: torch.Tensor,
    center: torch.Tensor,
    kappa_s: torch.Tensor,
    r_s: float = 0.20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes exact analytical deflection and convergence for an NFW subhalo."""
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    r = torch.sqrt(dx**2 + dy**2 + 1e-12)
    x = r / r_s

    eps = 1e-4
    x_safe = torch.where(torch.abs(x - 1.0) < eps, torch.tensor(1.0 + eps, device=coords.device), x)
    cond = x_safe < 1.0

    u_small = torch.sqrt(torch.clamp(1.0 - x_safe**2, min=1e-10))
    f_small = torch.atanh(u_small) / u_small
    u_large = torch.sqrt(torch.clamp(x_safe**2 - 1.0, min=1e-10))
    f_large = torch.atan(u_large) / u_large
    F = torch.where(cond, f_small, f_large)
    F = torch.where(torch.abs(x - 1.0) < eps, torch.tensor(1.0, device=coords.device), F)

    denom = x**2 - 1.0
    g = (1.0 - F) / torch.where(torch.abs(denom) < 1e-5, torch.tensor(1.0, device=coords.device), denom)
    kappa = 2.0 * kappa_s * g

    h = (torch.log(torch.clamp(x / 2.0, min=1e-8)) + F) / torch.clamp(x, min=1e-6)
    alpha_mag = 4.0 * kappa_s * r_s * h
    alpha = torch.stack([alpha_mag * (dx / r), alpha_mag * (dy / r)], dim=-1)
    return alpha, kappa


def sis_subhalo_deflection(
    coords: torch.Tensor,
    center: torch.Tensor,
    theta_E: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes deflection and convergence for a Singular Isothermal Sphere (SIS) subhalo."""
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    r = torch.sqrt(dx**2 + dy**2 + 1e-12)
    alpha = torch.stack([theta_E * (dx / r), theta_E * (dy / r)], dim=-1)
    kappa = 0.5 * theta_E / r
    return alpha, kappa


class Subhalo(nn.Module):
    """Individual dark matter subhalo module with autodiff support."""

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.35,
        r_s: float = 0.20,
        model_type: str = "NFW"
    ):
        super().__init__()
        self.center = nn.Parameter(torch.tensor(list(center), dtype=torch.float32))
        self.kappa_s_raw = nn.Parameter(torch.tensor(math.log(math.exp(kappa_s) - 1.0 + 1e-5), dtype=torch.float32))
        self.r_s = r_s
        self.model_type = model_type

    @property
    def kappa_s(self) -> torch.Tensor:
        return F.softplus(self.kappa_s_raw)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_type.upper() == "NFW":
            return nfw_subhalo_deflection(coords, self.center, self.kappa_s, self.r_s)
        elif self.model_type.upper() == "SIS":
            return sis_subhalo_deflection(coords, self.center, self.kappa_s)
        else:
            raise ValueError(f"Unknown subhalo model: {self.model_type}")


class MultiSubhaloPopulation(nn.Module):
    """Collection of multiple dark matter subhalos."""

    def __init__(self, subhalos: Optional[List[Subhalo]] = None):
        super().__init__()
        self.subhalos = nn.ModuleList(subhalos or [])

    def add_subhalo(self, subhalo: Subhalo):
        self.subhalos.append(subhalo)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.subhalos) == 0:
            zeros_alpha = torch.zeros(coords.shape[0], 2, device=coords.device)
            zeros_kappa = torch.zeros(coords.shape[0], device=coords.device)
            return zeros_alpha, zeros_kappa

        total_alpha = torch.zeros(coords.shape[0], 2, device=coords.device)
        total_kappa = torch.zeros(coords.shape[0], device=coords.device)

        for sub in self.subhalos:
            a, k = sub(coords)
            total_alpha = total_alpha + a
            total_kappa = total_kappa + k

        return total_alpha, total_kappa
