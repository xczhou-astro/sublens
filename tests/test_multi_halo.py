"""Simultaneous deflections, PSF adjoints and nesting of multi-halo hypotheses."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from sublens import LensingObservation, MacroLensModel, MultiHaloProblem, truncated_subhalo_deflection
from sublens.inversion.source_plane import PixelSourceProblem


def observation():
    image = np.exp(-.5*((np.arange(20)[:, None]-9.2)**2+(np.arange(20)[None, :]-9.7)**2)/20)
    psf = np.array([[.01, .03, .01], [.04, .68, .09], [.02, .07, .05]])
    return LensingObservation.from_arrays(image, np.full(image.shape, .06), psf, pixel_scale=.12,
        macro_model=dict(theta_E=.7, q=.88, phi=.18, gamma=2.05, gamma_ext=.03, phi_ext=.2))


def test_simultaneous_rays_match_direct_macro_and_two_halo_sum():
    obs = observation()
    problem = MultiHaloProblem(obs, source_nodes=40, regularization=.1)
    parameters = torch.tensor([.2, -.15, .12, -.62, .24, .20, .55, -.31, .15], dtype=torch.float64)
    expected = torch.stack([torch.from_numpy(a) for a in obs.image_rays()], dim=-1)
    macro = dict(obs.macro_model)
    macro['theta_E'] *= 1+.03*.2
    g1 = macro['gamma_ext']*np.cos(2*macro['phi_ext'])-.03*.15
    g2 = macro['gamma_ext']*np.sin(2*macro['phi_ext'])+.03*.12
    macro.update(gamma_ext=float(np.hypot(g1, g2)), phi_ext=float(np.arctan2(g2, g1)/2))
    alpha, _ = MacroLensModel(**macro).double()(expected.reshape(-1, 2))
    expected = expected-alpha.reshape(expected.shape)
    for x, y, amplitude in parameters[3:].reshape(-1, 3):
        ax, ay = truncated_subhalo_deflection(problem.x, problem.y, x, y, amplitude*.12)
        expected -= torch.stack((ax, ay), dim=-1)
    np.testing.assert_allclose(problem.rays(parameters).numpy(), expected.detach().numpy(), atol=2e-7)


def test_profile_gradient_includes_both_halos_and_asymmetric_psf_adjoint():
    obs = observation()
    problem = MultiHaloProblem(obs, source_nodes=60, regularization=.3)
    values = np.array([.12, -.1, .08, -.63, .27, .2, .57, -.34, .16])
    _, gradient, source = problem.profiled_objective(values)
    def frozen_loss(v):
        rays = problem.rays(torch.from_numpy(v)).numpy()
        image = problem.problem.render(rays[..., 0], rays[..., 1], source.coefficients)
        residual = (image[obs.mask]-obs.image[obs.mask])*problem.problem.weight
        return residual@residual
    for index in range(len(values)):
        delta = np.zeros(len(values)); delta[index] = 2e-6
        numerical = (frozen_loss(values+delta)-frozen_loss(values-delta))/(4e-6)
        np.testing.assert_allclose(gradient[index], numerical, rtol=2e-3, atol=3e-6)


def test_joint_models_allow_zero_amplitudes_and_preserve_nested_objective():
    problem = MultiHaloProblem(observation(), source_nodes=40, regularization=.3)
    null = problem.fit(np.zeros(3), [], n_starts=1, maxiter=15)
    anchors = [(-.63, .27), (.57, -.34)]
    initial = np.r_[null.parameters, -.63, .27, 0., .57, -.34, 0.]
    model = problem.fit(initial, anchors, initial_source=null.source, n_starts=2, maxiter=15)
    assert model.halos.shape == (2, 3) and len(model.attempts) == 2
    assert (model.halos[:, 2] >= 0).all() and (model.halos[:, 2] <= .12).all()
    assert model.source.objective <= null.source.objective+1e-6
    assert abs(model.macro_model['theta_E']/problem.macro['theta_E']-1) <= .0300001


def test_source_problem_must_match_observation_weights():
    obs = observation()
    wrong = PixelSourceProblem(obs.image, 2*obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
                               source_nodes=40, regularization=.1)
    with pytest.raises(ValueError, match='same image, noise and PSF'):
        MultiHaloProblem(obs, problem=wrong)
