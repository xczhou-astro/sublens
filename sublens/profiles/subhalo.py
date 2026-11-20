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


def truncated_subhalo_deflection(
    x: torch.Tensor,
    y: torch.Tensor,
    halo_x: float,
    halo_y: float,
    strength: torch.Tensor,
    core: float = 0.035,
    truncation: float = 0.22
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deflection of the softened, truncated dark matter halo."""
    dx = x - halo_x
    dy = y - halo_y
    radius_sq = dx**2 + dy**2
    factor = strength * truncation / torch.sqrt((radius_sq + core**2) * (radius_sq + truncation**2) + 1e-12)
    return factor * dx, factor * dy



def nfw_subhalo_deflection(
    coords: torch.Tensor,
    center: torch.Tensor,
    kappa_s: torch.Tensor,
    r_s: float = 0.20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bartelmann (1996) spherical NFW deflection and convergence.

    Uses cancellation-free small-radius and x=1 series, with finite inactive
    branches for autograd. At the exact singular centre alpha=0; convergence is
    evaluated at x=1e-10 (float64) / 1e-6 (float32), not an inferred physical core.
    r_s and coordinates are angular arcsec; kappa_s is dimensionless.
    Reference: https://arxiv.org/abs/astro-ph/9602053, equations 2.8-2.10.
    """
    delta = torch.stack((coords[:, 0]-center[0], coords[:, 1]-center[1]), dim=-1)
    floor = 1e-10 if coords.dtype == torch.float64 else 1e-6
    x = (torch.linalg.vector_norm(delta, dim=-1)/r_s).clamp_min(floor)
    t = x.square()-1
    near = t.abs() < .01
    small = x < .05

    # Safe placeholders keep every unselected branch finite and differentiable.
    lower = torch.where((x < 1) & ~near & ~small, x, .5)
    upper = torch.where((x > 1) & ~near, x, 2.)
    f_lower = torch.acosh(1/lower)/torch.sqrt(1-lower.square())
    z = torch.sqrt(upper.square()-1)
    f_upper = torch.atan(z)/z
    big_f = torch.where(x < 1, f_lower, f_upper)
    safe_t = torch.where(near, torch.ones_like(t), t)
    density = (1-big_f)/safe_t
    mass_over_x2 = (torch.log(x/2)+big_f)/x.square()

    # F = atan(sqrt(t))/sqrt(t) analytically continued through t=0.
    u = t.clamp(-.01, .01)
    near_density = 1/3-u/5+u**2/7-u**3/9+u**4/11-u**5/13
    near_f = 1-u/3+u**2/5-u**3/7+u**4/9-u**5/11+u**6/13
    near_mass = (.5*torch.log1p(u)-math.log(2)+near_f)/(1+u)
    density = torch.where(near, near_density, density)
    mass_over_x2 = torch.where(near, near_mass, mass_over_x2)

    # Small-x expansion of F=acosh(1/x)/sqrt(1-x^2); avoid subtracting log(2/x).
    s = x.clamp_max(.05)
    log = torch.log(2/s)
    s2 = s.square()
    small_mass = (.5*log-.25 + s2*(3/8*log-7/32)
                  + s2**2*(5/16*log-37/192) + s2**3*(35/128*log-533/3072))
    small_density = (log-1 + s2*(3/2*log-5/4)
                     + s2**2*(15/8*log-47/32) + s2**3*(35/16*log-319/192))
    density = torch.where(small, small_density, density)
    mass_over_x2 = torch.where(small, small_mass, mass_over_x2)
    return 4*kappa_s*mass_over_x2[:, None]*delta, 2*kappa_s*density


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
