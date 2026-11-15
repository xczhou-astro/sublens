"""Unit tests for lens and subhalo profiles."""
import torch
import math
from sublens.profiles.macro import epl_deflection, external_shear_deflection
from sublens.profiles.subhalo import nfw_subhalo_deflection, sis_subhalo_deflection, Subhalo, MultiSubhaloPopulation


def test_epl_deflection():
    coords = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    alpha, kappa = epl_deflection(coords, theta_E=1.0, q=1.0, phi=0.0, gamma=2.0)
    assert alpha.shape == (2, 2)
    assert kappa.shape == (2,)
    # Circular SIS at theta = theta_E has alpha = theta_E
    assert torch.allclose(alpha[0], torch.tensor([1.0, 0.0]), atol=1e-3)


def test_subhalo_autodiff():
    coords = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    sub = Subhalo(center=(0.5, 0.5), kappa_s=0.35, r_s=0.20, model_type="NFW")
    alpha, kappa = sub(coords)
    assert alpha.shape == (1, 2)
    # Check that gradient flows to center
    loss = alpha.sum()
    loss.backward()
    assert sub.center.grad is not None
