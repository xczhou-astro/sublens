"""Validated model specifications for forward simulation (angles in arcsec/rad).

These immutable configurations are not priors or input models for analyze_lens.
Existing trainable MacroLensModel/Subhalo modules remain available separately.
"""
from dataclasses import asdict, dataclass, field
import math

import torch

from .macro import epl_deflection, external_shear_deflection
from .subhalo import nfw_subhalo_deflection, truncated_subhalo_deflection


def _numbers(model, *names):
    for name in names:
        raw = getattr(model, name)
        if isinstance(raw, bool):
            raise ValueError(f'{name} must be a finite number')
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
        object.__setattr__(model, name, value)


def _center(model):
    if len(model.center) != 2:
        raise ValueError('center must contain x and y in arcsec')
    center = tuple(float(v) for v in model.center)
    if not all(math.isfinite(v) for v in center):
        raise ValueError('center must be finite')
    object.__setattr__(model, 'center', center)


class _Specification:
    def to_dict(self):
        return dict(model=type(self).__name__, **asdict(self))


@dataclass(frozen=True)
class EPL(_Specification):
    """Elliptical power-law mass plus optional external shear.

    theta_E/center are angular arcsec, phi/phi_ext radians, gamma is the 3-D
    density slope. gamma_ext is the dimensionless shear amplitude.
    """
    theta_E: float = 1.35
    q: float = .8
    phi: float = 0.
    gamma: float = 2.
    center: tuple = (0., 0.)
    gamma_ext: float = 0.
    phi_ext: float = 0.

    def __post_init__(self):
        _numbers(self, 'theta_E', 'q', 'phi', 'gamma', 'gamma_ext', 'phi_ext')
        _center(self)
        if not (self.theta_E > 0 and 0 < self.q <= 1 and 1 < self.gamma < 3
                and 0 <= self.gamma_ext < 1):
            raise ValueError('Use theta_E>0, 0<q<=1, 1<gamma<3 and 0<=gamma_ext<1')

    def deflection(self, coordinates):
        alpha, _ = epl_deflection(coordinates, theta_E=self.theta_E, q=self.q,
            phi=self.phi, gamma=self.gamma, center=self.center)
        return alpha + external_shear_deflection(coordinates, gamma_ext=self.gamma_ext,
            phi_ext=self.phi_ext, center=self.center)


@dataclass(frozen=True)
class SIE(EPL):
    """Isothermal EPL (gamma is fixed at 2 and is not a constructor argument)."""
    gamma: float = field(default=2., init=False)


@dataclass(frozen=True)
class Sersic(_Specification):
    """Analytic source or foreground brightness; i_eff is in image units."""
    center: tuple = (0., 0.)
    i_eff: float = .3
    r_eff: float = .22
    n_sersic: float = 1.25
    q: float = .78
    phi: float = .3

    def __post_init__(self):
        _numbers(self, 'i_eff', 'r_eff', 'n_sersic', 'q', 'phi')
        _center(self)
        if not (self.i_eff >= 0 and self.r_eff > 0 and self.n_sersic > 0 and 0 < self.q <= 1):
            raise ValueError('Use i_eff>=0, r_eff>0, n_sersic>0 and 0<q<=1')

    def brightness(self, coordinates):
        from ..simulation.sources import sersic_profile
        return sersic_profile(coordinates[:, 0], coordinates[:, 1], **asdict(self))


@dataclass(frozen=True)
class TruncatedHalo(_Specification):
    """Existing softened/truncated halo; strength/core/truncation are arcsec.

    This is the profile fitted by the current production PINN physical refiner.
    It is not NFW, and strength is not a mass or NFW kappa_s.
    """
    strength: float = .05
    center: tuple = (0., 0.)
    core: float = .035
    truncation: float = .22

    def __post_init__(self):
        _numbers(self, 'strength', 'core', 'truncation')
        _center(self)
        if not (self.strength >= 0 and 0 < self.core < self.truncation):
            raise ValueError('Use strength>=0 and 0<core<truncation')

    def deflection(self, coordinates):
        ax, ay = truncated_subhalo_deflection(coordinates[:, 0], coordinates[:, 1],
            *self.center, self.strength, core=self.core, truncation=self.truncation)
        return torch.stack((ax, ay), dim=-1)


@dataclass(frozen=True)
class NFW(_Specification):
    """Spherical untruncated NFW lens; r_s/center arcsec, kappa_s dimensionless.

    kappa_s = rho_s*r_s_physical/Sigma_critical, not a mass in solar units.
    Current analyze_lens does not fit NFW parameters; it fits TruncatedHalo.
    """
    kappa_s: float = .2
    r_s: float = .2
    center: tuple = (0., 0.)

    def __post_init__(self):
        _numbers(self, 'kappa_s', 'r_s')
        _center(self)
        if not (self.kappa_s >= 0 and self.r_s > 0):
            raise ValueError('Use kappa_s>=0 and r_s>0')

    def deflection(self, coordinates):
        alpha, _ = nfw_subhalo_deflection(coordinates, self.center, self.kappa_s, self.r_s)
        return alpha
