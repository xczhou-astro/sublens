"""Analytic simulation -> observed FITS -> the current image-only PINN API.

Run from the project root with its .venv Python. No simulation model is passed
to inference. Use --simulate-only for generating the observation bundle only.
"""
import argparse
from pathlib import Path
from uuid import uuid4

import torch

from sublens import SIE, Sersic, TruncatedHalo, simulate_lens, load_lens_data, analyze_lens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--simulate-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.output or root/'sublens_testing/one_subhalo/api_demo'/f'run_{uuid4().hex[:8]}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    simulation = simulate_lens(
        lens_mass=SIE(theta_E=1.35, q=.8, phi=.35, gamma_ext=.045, phi_ext=.2),
        source=Sersic(center=(.04,-.03), i_eff=.3, r_eff=.22, n_sersic=1.25, q=.78, phi=.3),
        subhalos=TruncatedHalo(center=(-1.28,.61), strength=.05),
        lens_light=Sersic(i_eff=.18, r_eff=.45, n_sersic=3., q=.85, phi=.35),
        image_size=100, pixel_scale=.05, psf_fwhm=.09, noise=.012,
        seed=314159, device=args.device,
    )
    observation_path = simulation.to_fits(folder/'observation.fits')
    simulation.save_truth(folder/'truth.json')
    print(f'Observation and separate truth file: {folder}', flush=True)
    if args.simulate_only:
        return
    data = load_lens_data(observation_path)
    result = analyze_lens(data, device=args.device, verbose=True)
    result.summary()
    result.save_json(folder/'result.json')
    result.save_fits(folder/'result.fits')
    result.plot_residuals(save_path=folder/'residuals.png')
    result.plot_fit(save_path=folder/'source_reconstruction.png')
    result.plot(save_path=folder/'pinn_candidates.png')


if __name__ == '__main__':
    main()
