"""Unit tests for lens and subhalo profiles."""
import torch
from sublens.profiles.macro import MacroLensModel, epl_deflection
from sublens.profiles.subhalo import Subhalo


def test_epl_deflection():
    coords = torch.tensor([[0.5, 0.0], [0.0, 2.0], [-0.3, -0.4]])
    theta_E = 1.35
    alpha, kappa = epl_deflection(coords, theta_E=theta_E, q=1.0, gamma=2.0)
    radii = coords.norm(dim=-1)
    torch.testing.assert_close(alpha, theta_E * coords / radii[:, None])
    torch.testing.assert_close(kappa, 0.5 * theta_E / radii)


def test_elliptical_epl_jacobian_matches_convergence():
    coords = torch.tensor(
        [[0.1, 0.2], [0.7, 0.4], [-0.8, 0.3], [-0.4, -0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    alpha, kappa = epl_deflection(
        coords, theta_E=1.35, q=0.2, phi=0.35, gamma=2.4, center=(0.08, -0.03)
    )
    grad_ax, = torch.autograd.grad(alpha[:, 0].sum(), coords, retain_graph=True)
    grad_ay, = torch.autograd.grad(alpha[:, 1].sum(), coords)

    torch.testing.assert_close(grad_ax[:, 1], grad_ay[:, 0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        0.5 * (grad_ax[:, 0] + grad_ay[:, 1]), kappa, atol=1e-10, rtol=1e-10
    )


def test_macro_model_differentiates_lens_parameters():
    model = MacroLensModel(
        theta_E=1.35, q=0.8, phi=0.35, gamma=2.05,
        gamma_ext=0.045, phi_ext=0.2, center=(0.08, -0.03),
    )
    coords = torch.tensor([[0.7, 0.4], [-0.8, 0.3], [-0.4, -0.6]])
    alpha, kappa = model(coords)
    (alpha.sum() + kappa.sum()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_subhalo_autodiff():
    coords = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    sub = Subhalo(center=(0.5, 0.5), kappa_s=0.35, r_s=0.20, model_type="NFW")
    alpha, kappa = sub(coords)
    assert alpha.shape == (1, 2)
    # Check that gradient flows to center
    loss = alpha.sum()
    loss.backward()
    assert sub.center.grad is not None
