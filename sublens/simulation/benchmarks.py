"""
Preconfigured Astronomical Benchmark Systems.

Provides standard reference setups for:
1. SDSS J0946+5106: Vegetti et al. (2010 Nature) confirmed landmark system
2. JVAS B1938+666: Vegetti et al. (2012 Nature) confirmed dwarf satellite system
3. Complex Source Benchmark: Spiral galaxy + starburst clumps + turbulent ISM
4. Scale-Free Synthetic Benchmark: For redshift-agnostic evaluations
5. Astrophysical Systematics Benchmark: Composite core + luminous satellite
"""

import math
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F

from .simulator import LensingSimulator
from .sources import generate_complex_source
from ..profiles.macro import epl_deflection, external_shear_deflection, hernquist_deflection
from ..profiles.subhalo import nfw_subhalo_deflection, sis_subhalo_deflection
from ..cosmology import LensingCosmology


def _resolve_device(device: str):
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def create_j0946_benchmark(device: str = "cpu") -> Dict[str, Any]:
    """Generates the benchmark system for SDSS J0946+5106 (Vegetti et al. 2010 Nature)."""
    device = _resolve_device(device)
    cosmo = LensingCosmology(z_l=0.222, z_s=0.609)
    sim = LensingSimulator(N_img=120, pixel_scale=0.05, N_src=64, L_src=1.8, psf_fwhm=0.09, noise_std=0.012, device=device)

    # Host EPL + Shear
    a_host, k_host = epl_deflection(sim.theta_grid, theta_E=1.43, q=0.77, phi=-1.18, gamma=2.16)
    a_shear = external_shear_deflection(sim.theta_grid, gamma_ext=0.035, phi_ext=-0.45)
    a_macro = a_host + a_shear

    # Ground truth subhalo from Nature 2010
    sub_true_pos = (-0.651, 1.040)
    sub_mass = 3.51e9
    kappa_s, theta_s = cosmo.physical_mass_to_nfw(sub_mass, c_200=18.0, concentration_boost=25.0)
    a_sub, k_sub = nfw_subhalo_deflection(sim.theta_grid, torch.tensor(sub_true_pos, device=device), torch.tensor(kappa_s, device=device), r_s=theta_s)

    # Source galaxy (compact starburst at z_s=0.609)
    sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing='ij')
    vignette = torch.exp(-0.5 * (sx**2 + sy**2) / (1.20**2)).view(1, 1, sim.N_src, sim.N_src)
    I_src = 1.30 * torch.exp(-torch.sqrt((sx + 0.08)**2 / 0.7 + (sy - 0.10)**2 * 0.7) / 0.22)
    I_src += 0.90 * torch.exp(-0.5 * ((sx - 0.12)**2 + (sy + 0.15)**2) / (0.06**2))
    I_src_true = (I_src * vignette.squeeze()).view(1, 1, sim.N_src, sim.N_src)

    alpha_total = a_macro + a_sub
    img_obs, img_clean = sim.simulate(I_src_true, alpha_total, add_noise=True, seed=42)

    return {
        "name": "SDSS J0946+5106 (Nature 2010)",
        "cosmology": cosmo,
        "simulator": sim,
        "obs_image": img_obs,
        "clean_image": img_clean,
        "source_true": I_src_true,
        "alpha_macro": a_macro,
        "sub_true_pos": sub_true_pos,
        "sub_true_mass": sub_mass,
        "sub_kappa_s": kappa_s,
        "sub_theta_s": theta_s,
    }


def create_b1938_benchmark(device: str = "cpu") -> Dict[str, Any]:
    """Generates the benchmark system for JVAS B1938+666 (Vegetti et al. 2012 Nature)."""
    cosmo = LensingCosmology(z_l=0.881, z_s=2.059)
    sim = LensingSimulator(N_img=120, pixel_scale=0.020, N_src=64, L_src=0.8, psf_fwhm=0.06, noise_std=0.008, device=device)

    a_host, k_host = epl_deflection(sim.theta_grid, theta_E=0.445, q=0.872, phi=-0.4217, gamma=2.058)
    a_shear = external_shear_deflection(sim.theta_grid, gamma_ext=0.012, phi_ext=1.9757)
    a_macro = a_host + a_shear

    sub_true_pos = (0.036, 0.576)
    sub_mass = 1.9e8
    kappa_s, theta_s = cosmo.physical_mass_to_nfw(sub_mass, c_200=20.0, concentration_boost=22.0)
    a_sub, k_sub = nfw_subhalo_deflection(sim.theta_grid, torch.tensor(sub_true_pos, device=device), torch.tensor(kappa_s, device=device), r_s=theta_s)

    sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing='ij')
    vignette = torch.exp(-0.5 * (sx**2 + sy**2) / (0.50**2)).view(1, 1, sim.N_src, sim.N_src)
    I_src = 1.20 * torch.exp(-torch.sqrt((sx - 0.01)**2 / 0.8 + (sy - 0.06)**2 * 0.8) / 0.12)
    I_src += 0.85 * torch.exp(-0.5 * ((sx + 0.04)**2 + (sy - 0.02)**2) / (0.035**2))
    I_src_true = (I_src * vignette.squeeze()).view(1, 1, sim.N_src, sim.N_src)

    alpha_total = a_macro + a_sub
    img_obs, img_clean = sim.simulate(I_src_true, alpha_total, add_noise=True, seed=2012)

    return {
        "name": "JVAS B1938+666 (Nature 2012)",
        "cosmology": cosmo,
        "simulator": sim,
        "obs_image": img_obs,
        "clean_image": img_clean,
        "source_true": I_src_true,
        "alpha_macro": a_macro,
        "sub_true_pos": sub_true_pos,
        "sub_true_mass": sub_mass,
        "sub_kappa_s": kappa_s,
        "sub_theta_s": theta_s,
    }


def create_scale_free_benchmark(device: str = "cpu") -> Dict[str, Any]:
    """Generates a scale-free benchmark system when redshifts are completely unknown."""
    cosmo = LensingCosmology(z_l=None, z_s=None)
    sim = LensingSimulator(N_img=100, pixel_scale=0.04, N_src=64, L_src=1.5, psf_fwhm=0.08, noise_std=0.010, device=device)

    # Macro lens: theta_E = 1.20"
    a_host, k_host = epl_deflection(sim.theta_grid, theta_E=1.20, q=0.85, phi=0.35, gamma=2.0)
    a_shear = external_shear_deflection(sim.theta_grid, gamma_ext=0.04, phi_ext=0.20)
    a_macro = a_host + a_shear

    # Subhalo: dimensionless theta_E = 0.035" (~ 0.08% mass ratio)
    sub_pos = (0.75, -0.75)
    theta_E_sub = 0.035
    a_sub, k_sub = sis_subhalo_deflection(sim.theta_grid, torch.tensor(sub_pos, device=device), torch.tensor(theta_E_sub, device=device))

    # Complex source
    I_src_true, _ = generate_complex_source(N_src=sim.N_src, L_src=sim.L_src, seed=101, device=device)

    alpha_total = a_macro + a_sub
    img_obs, img_clean = sim.simulate(I_src_true, alpha_total, add_noise=True, seed=888)

    return {
        "name": "Scale-Free Unknown-Redshift Benchmark",
        "cosmology": cosmo,
        "simulator": sim,
        "obs_image": img_obs,
        "clean_image": img_clean,
        "source_true": I_src_true,
        "alpha_macro": a_macro,
        "sub_true_pos": sub_pos,
        "theta_E_host": 1.20,
        "theta_E_sub": theta_E_sub,
    }
