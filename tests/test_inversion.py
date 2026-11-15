"""Unit tests for the three-stage inversion pipeline."""
import torch
import sublens as sl


def test_forward_simulator():
    sim = sl.LensingSimulator(N_img=50, pixel_scale=0.05, N_src=32, L_src=1.0)
    src = torch.ones(1, 1, 32, 32)
    alpha = torch.zeros(50 * 50, 2)
    obs, clean = sim.simulate(src, alpha, add_noise=False)
    assert obs.shape == (1, 1, 50, 50)


def test_astrometric_metrics():
    res = sl.compute_astrometric_error((0.0, 0.0), (0.003, 0.004), pixel_scale=0.05)
    assert round(res["error_arcsec"], 5) == 0.005
    assert round(res["error_mas"], 2) == 5.0
    assert round(res["error_pixels"], 3) == 0.1
