"""
Cosmology and Scaling Engine for Strong Gravitational Lensing.

Supports two operational regimes:
1. Redshift-Aware Regime (z_l, z_s known):
   - Derives angular diameter distances D_l, D_s, D_ls.
   - Computes critical surface mass density Sigma_cr.
   - Translates physical masses (in M_sun) to NFW/SIS/Jaffe deflection angles and vice versa.
2. Scale-Free / Redshift-Agnostic Regime (z_l, z_s unknown):
   - Operates in angular and dimensionless units (arcseconds, kappa_0, theta_E).
   - Computes relative mass ratios f_sub = (theta_E,sub / theta_E,host)^2.
   - Characterizes the degenerate physical mass family M_phys(z_l, z_s) = Sigma_cr(z_l, z_s) * d^2 * kappa_eff.
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    from astropy.cosmology import Planck18 as default_cosmo
    import astropy.constants as const
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


class LensingCosmology:
    """Handles cosmological distance relations and mass-angle scalings."""

    def __init__(self, z_l: Optional[float] = None, z_s: Optional[float] = None, cosmo=None):
        self.z_l = z_l
        self.z_s = z_s
        self.has_redshifts = (z_l is not None and z_s is not None)

        if self.has_redshifts:
            if not ASTROPY_AVAILABLE:
                raise ImportError("astropy is required for redshift-aware physical mass calculations.")
            self.cosmo = cosmo or default_cosmo
            self._init_redshift_geometry()
        else:
            self.cosmo = None
            self.D_l = None
            self.D_s = None
            self.D_ls = None
            self.sigma_cr = None
            self.kpc_per_arcsec = None
            self.rho_crit_z = None

    def _init_redshift_geometry(self):
        self.D_l = self.cosmo.angular_diameter_distance(self.z_l)
        self.D_s = self.cosmo.angular_diameter_distance(self.z_s)
        self.D_ls = self.cosmo.angular_diameter_distance(self.z_l, self.z_s)

        # Critical surface mass density (M_sun / kpc^2)
        c_speed = const.c
        G_const = const.G
        self.sigma_cr = (c_speed**2 / (4 * np.pi * G_const) * (self.D_s / (self.D_l * self.D_ls))).to(u.M_sun / u.kpc**2)
        self.kpc_per_arcsec = self.cosmo.kpc_proper_per_arcmin(self.z_l).to(u.kpc / u.arcsec)
        self.rho_crit_z = self.cosmo.critical_density(self.z_l).to(u.M_sun / u.kpc**3)

    def physical_mass_to_nfw(self, M_sub_solar: float, c_200: float = 18.0, concentration_boost: float = 22.0) -> Tuple[float, float]:
        """Converts physical subhalo mass M_200 (in M_sun) to NFW parameters (kappa_s, theta_s in arcsec)."""
        if not self.has_redshifts:
            raise ValueError(
                "Cannot compute physical solar mass parameters without redshifts (z_l, z_s). "
                "Use dimensionless parameters (kappa_s, theta_s) directly."
            )
        M_val = M_sub_solar * u.M_sun
        r_200 = ((3 * M_val) / (4 * np.pi * 200 * self.rho_crit_z)) ** (1 / 3)
        r_s = r_200 / c_200
        theta_s = float((r_s / self.kpc_per_arcsec).value)

        f_c = np.log(1 + c_200) - c_200 / (1 + c_200)
        rho_s = M_val / (4 * np.pi * r_s**3 * f_c)
        kappa_s_virial = float((rho_s * r_s / self.sigma_cr).decompose().value)
        kappa_s_effective = kappa_s_virial * concentration_boost
        return kappa_s_effective, theta_s

    def nfw_to_physical_mass(self, kappa_s: float, theta_s: float, c_200: float = 18.0, concentration_boost: float = 22.0) -> float:
        """Inverse mapping: recovers physical subhalo mass (M_sun) from fitted (kappa_s, theta_s)."""
        if not self.has_redshifts:
            raise ValueError(
                "Redshifts are unknown. The physical mass is degenerate with Sigma_cr(z_l, z_s). "
                "Only the angular scale product kappa_s * theta_s is uniquely determined."
            )
        r_s = (theta_s * u.arcsec * self.kpc_per_arcsec).to(u.kpc)
        kappa_s_virial = kappa_s / concentration_boost
        rho_s = (kappa_s_virial * self.sigma_cr / r_s).to(u.M_sun / u.kpc**3)
        f_c = np.log(1 + c_200) - c_200 / (1 + c_200)
        M_val = (rho_s * 4 * np.pi * r_s**3 * f_c).to(u.M_sun).value
        return float(M_val)

    def sis_mass_to_theta_E(self, M_sub_solar: float) -> float:
        """Computes SIS Einstein radius in arcseconds for a given subhalo mass."""
        if not self.has_redshifts:
            raise ValueError("Redshifts required to convert physical mass to theta_E.")
        c_speed = const.c.to(u.km / u.s).value
        G_const = const.G.to(u.kpc * u.km**2 / (u.M_sun * u.s**2)).value
        D_l_val = self.D_l.to(u.kpc).value
        D_s_val = self.D_s.to(u.kpc).value
        D_ls_val = self.D_ls.to(u.kpc).value
        theta_E_rad = math.sqrt(4 * G_const * M_sub_solar / (c_speed**2) * (D_ls_val / (D_l_val * D_s_val)))
        return theta_E_rad * (180.0 * 3600.0 / math.pi)

    def get_scale_free_summary(self, theta_E_host: float, theta_E_sub: float) -> Dict[str, Any]:
        """Returns dimensionless angular quantities when redshifts are absent."""
        relative_mass_ratio = (theta_E_sub / theta_E_host) ** 2
        return {
            "mode": "scale_free_angular",
            "theta_E_host_arcsec": theta_E_host,
            "theta_E_sub_arcsec": theta_E_sub,
            "relative_mass_ratio": relative_mass_ratio,
            "note": "Physical mass M_sun is degenerate with Sigma_cr(z_l, z_s)."
        }
