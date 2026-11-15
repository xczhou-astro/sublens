"""Unit tests for cosmology and scaling."""
import pytest
import math
from sublens.cosmology import LensingCosmology


def test_redshift_aware_cosmology():
    cosmo = LensingCosmology(z_l=0.222, z_s=0.609)
    assert cosmo.has_redshifts is True
    assert cosmo.D_l is not None
    assert cosmo.sigma_cr is not None

    # Test mass to NFW
    kappa_s, theta_s = cosmo.physical_mass_to_nfw(3.51e9)
    assert kappa_s > 0.0
    assert theta_s > 0.0

    # Test roundtrip recovery
    recovered_mass = cosmo.nfw_to_physical_mass(kappa_s, theta_s)
    assert math.isclose(recovered_mass, 3.51e9, rel_tol=1e-3)


def test_scale_free_cosmology():
    cosmo = LensingCosmology(z_l=None, z_s=None)
    assert cosmo.has_redshifts is False

    summary = cosmo.get_scale_free_summary(theta_E_host=1.40, theta_E_sub=0.035)
    assert summary["mode"] == "scale_free_angular"
    assert math.isclose(summary["relative_mass_ratio"], (0.035 / 1.40)**2)
