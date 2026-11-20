"""Mass normalization, critical area and all seven profiled macro derivatives."""

import numpy as np
import torch

from sublens import LensingObservation, MacroLensModel
from sublens.survey.euclid_source import _q1_source_rays
from sublens_testing.euclid_grade_a_10 import (
    ProfiledMacroFit, effective_sie_radius, macro_from_release, source_shift,
)


def test_release_conversion_preserves_rays_up_to_known_shear_origin_shift():
    mass = dict(ell_comps_0_max_lh=.11, ell_comps_1_max_lh=.18,
        einstein_radius_max_lh=1.3, centre_0_max_lh=-.06, centre_1_max_lh=.04,
        shear_gamma_1_max_lh=.05, shear_gamma_2_max_lh=-.03)
    _, _, bx, by = _q1_source_rays((20, 20), .12, mass, 2)
    macro = macro_from_release(mass)
    obs = LensingObservation.from_arrays(np.ones((20, 20)), np.ones((20, 20)),
        np.ones((1, 1)), pixel_scale=.12, macro_model=macro, oversample=2)
    shift = source_shift(macro)
    np.testing.assert_allclose(obs.bx, bx+shift[0], atol=3e-13)
    np.testing.assert_allclose(obs.by, by+shift[1], atol=3e-13)


def test_area_equivalent_radius_matches_direct_critical_jacobian():
    macro = dict(theta_E=1.2, q=.67, phi=.31, gamma=2., center=[.04, -.07],
                 gamma_ext=.09, phi_ext=-.23)
    angles = np.arange(2048)*2*np.pi/2048
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    coordinates = torch.tensor(direction+macro['center'], dtype=torch.float64, requires_grad=True)
    model = MacroLensModel(**macro).double()
    def alpha(v): return model(v)[0]
    a = alpha(coordinates)
    gradients = [torch.autograd.grad(a[:, j].sum(), coordinates, retain_graph=True)[0].numpy() for j in (0, 1)]
    jacobian = np.stack(gradients, axis=1)
    g1 = macro['gamma_ext']*np.cos(2*macro['phi_ext'])
    g2 = macro['gamma_ext']*np.sin(2*macro['phi_ext'])
    shear = np.array([[g1, g2], [g2, -g1]])
    host = jacobian-shear
    # A scale-free isothermal Hessian scales as 1/r and has rank one.
    transformed = np.linalg.solve(np.eye(2)-shear, host)
    radii = np.trace(transformed, axis1=1, axis2=2)
    # MacroLensModel stores its input constants initially in float32.
    np.testing.assert_allclose(effective_sie_radius(macro), np.sqrt(np.mean(radii**2)), rtol=1e-7)
    np.testing.assert_allclose(effective_sie_radius(dict(macro, q=1., gamma_ext=0.)), 1.2, atol=1e-12)


def test_profile_gradient_for_shape_center_radius_and_shear_with_asymmetric_psf():
    image = np.exp(-.5*((np.arange(20)[:, None]-9.2)**2+(np.arange(20)[None, :]-9.7)**2)/20)
    psf = np.array([[.01, .03, .01], [.04, .68, .09], [.02, .07, .05]])
    obs = LensingObservation.from_arrays(image, np.full(image.shape, .06), psf, pixel_scale=.12,
        macro_model=dict(theta_E=.7, q=.81, phi=.21, gamma=2., center=[.02, -.03],
                         gamma_ext=.03, phi_ext=.2))
    problem = ProfiledMacroFit(obs, nodes=60, regularization=.3)
    values = problem.vector(obs.macro_model)+np.array([.02, -.03, .04, .03, -.02, .02, -.03])
    _, gradient, source = problem.objective(values)
    def frozen_loss(v):
        rays = problem.rays(torch.from_numpy(v)).numpy()
        model = problem.problem.render(rays[..., 0], rays[..., 1], source.coefficients)
        residual = (model[obs.mask]-obs.image[obs.mask])*problem.problem.weight
        return residual@residual
    for index in range(7):
        delta = np.zeros(7); delta[index] = 2e-6
        numerical = (frozen_loss(values+delta)-frozen_loss(values-delta))/(4e-6)
        np.testing.assert_allclose(gradient[index], numerical, rtol=2e-3, atol=3e-6)
