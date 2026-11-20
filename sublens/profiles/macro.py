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
from typing import Tuple, Union
import torch
import torch.nn as nn

Scalar = Union[float, torch.Tensor]


def epl_deflection(
    coords: torch.Tensor,
    theta_E: Scalar = 1.40,
    q: Scalar = 0.80,
    phi: Scalar = 0.0,
    gamma: Scalar = 2.0,
    center: Tuple[Scalar, Scalar] = (0.0, 0.0)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the Tessore & Metcalf (2015) EPL deflection and convergence.

    theta_E is the circularized Einstein radius, gamma the 3D density slope,
    and phi the major-axis angle in radians. The central singularity is
    regularized with a minimum circularized radius of 1e-5 arcsec.
    """
    x = coords[:, 0] - center[0]
    y = coords[:, 1] - center[1]

    dtype = torch.result_type(coords, 0.0)
    angle = torch.as_tensor(phi, dtype=dtype, device=coords.device)
    cos_phi, sin_phi = angle.cos(), angle.sin()
    xr = cos_phi * x + sin_phi * y
    yr = -sin_phi * x + cos_phi * y

    # Bound the series tail using the maximum ratio between successive terms.
    # Detach only the scalar used to choose the number of terms.
    axis_ratio = float(q.detach()) if torch.is_tensor(q) else q
    if not math.isfinite(axis_ratio) or axis_ratio <= 0:
        raise ValueError("q must be a finite positive axis ratio")
    minor_to_major = min(axis_ratio, 1.0 / axis_ratio)
    gap = 2.0 * minor_to_major / (1.0 + minor_to_major)
    tolerance = torch.finfo(dtype).eps / 8
    terms = 2
    if gap < 1.0:
        terms = max(terms, math.ceil(math.log(tolerance * gap) / math.log1p(-gap)))

    sqrt_q = q**0.5
    ell_x, ell_y = sqrt_q * xr, yr / sqrt_q
    radius = torch.linalg.vector_norm(torch.stack((ell_x, ell_y)), dim=0).clamp_min(1e-5)
    slope = gamma - 1.0
    density_scale = (theta_E / radius) ** slope

    # Real-valued angular recurrence, equations (32)–(33).
    # https://arxiv.org/abs/1507.01819
    term_x, term_y = ell_x / radius, ell_y / radius
    flattening = (1.0 - q) / (1.0 + q)
    rotation_x = -flattening * (term_x.square() - term_y.square())
    rotation_y = -flattening * (2.0 * term_x * term_y)
    angular_x, angular_y = term_x, term_y
    for n in range(1, terms + 1):
        ratio = (2.0 * n - (2.0 - slope)) / (2.0 * n + (2.0 - slope))
        term_x, term_y = (
            ratio * (rotation_x * term_x - rotation_y * term_y),
            ratio * (rotation_y * term_x + rotation_x * term_y),
        )
        angular_x = angular_x + term_x
        angular_y = angular_y + term_y

    radial = (2.0 * sqrt_q / (1.0 + q)) * radius * density_scale
    ax_r, ay_r = radial * angular_x, radial * angular_y
    ax = cos_phi * ax_r - sin_phi * ay_r
    ay = sin_phi * ax_r + cos_phi * ay_r

    kappa = 0.5 * (2.0 - slope) * density_scale
    return torch.stack([ax, ay], dim=-1), kappa


def external_shear_deflection(
    coords: torch.Tensor,
    gamma_ext: Scalar = 0.05,
    phi_ext: Scalar = 0.0,
    center: Tuple[Scalar, Scalar] = (0.0, 0.0)
) -> torch.Tensor:
    """Computes deflection angle for external shear."""
    x = coords[:, 0] - center[0]
    y = coords[:, 1] - center[1]
    dtype = torch.result_type(coords, 0.0)
    angle = 2.0 * torch.as_tensor(phi_ext, dtype=dtype, device=coords.device)
    g1 = gamma_ext * angle.cos()
    g2 = gamma_ext * angle.sin()
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
        center = (self.center[0], self.center[1])
        a_epl, k_epl = epl_deflection(
            coords,
            theta_E=self.theta_E,
            q=self.q,
            phi=self.phi,
            gamma=self.gamma,
            center=center
        )
        a_shear = external_shear_deflection(
            coords,
            gamma_ext=self.gamma_ext,
            phi_ext=self.phi_ext,
            center=center
        )
        return a_epl + a_shear, k_epl
