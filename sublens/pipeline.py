"""
High-Level Unified API for SubLens.

Provides user-friendly, production-ready interfaces:
- SubLensPipeline: End-to-end detection and inversion pipeline supporting FITS files and arrays
- SubLensResult: Rich container holding detection metrics, component extractors, plotting, and FITS export
- detect_subhalo(): Functional entrypoint for quick analysis
"""

import os
import time
import math
from dataclasses import replace
from typing import Dict, Any, Tuple, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .cosmology import LensingCosmology
from .simulation.simulator import LensingSimulator
from .profiles.macro import epl_deflection, external_shear_deflection
from .profiles.subhalo import nfw_subhalo_deflection, sis_subhalo_deflection
from .inversion.macro_fitter import MacroFitter
from .inversion.radar import AstrometricRadar
from .inversion.pinn_optimizer import PINNOptimizer
from .utils.metrics import compute_astrometric_error, compute_image_metrics
from .utils.visualization import plot_diagnostic_figure, plot_real_observation_figure


def _parse_fits_input(
    image_input: Union[str, np.ndarray, torch.Tensor],
    psf_input: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
    noise_input: Optional[Union[str, np.ndarray, torch.Tensor]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Parses flexible inputs:
    - Single multi-extension FITS file containing SCI, PSF, and RMS/WHT
    - Separate FITS files for image, psf, noise
    - In-memory numpy arrays or torch tensors
    Returns: (image_2d, psf_2d, noise_map_2d, header_meta)
    """
    header_meta = {}
    image_arr = None
    psf_arr = None
    noise_arr = None

    # 1. Parse image input
    if isinstance(image_input, (str, os.PathLike)):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"FITS file not found: {image_input}")
        with fits.open(image_input) as hdul:
            ext_names = [h.name.upper() for h in hdul]
            # Science image
            if 'SCI' in ext_names:
                image_arr = hdul['SCI'].data.astype(np.float32)
                hdr = hdul['SCI'].header
            else:
                image_arr = hdul[0].data.astype(np.float32)
                hdr = hdul[0].header

            # Extract header metadata
            header_meta['OBJECT'] = hdr.get('OBJECT', None)
            header_meta['RA'] = float(hdr['RA']) if 'RA' in hdr else None
            header_meta['DEC'] = float(hdr['DEC']) if 'DEC' in hdr else None
            header_meta['GRADE'] = hdr.get('GRADE', None)
            header_meta['EXPSCORE'] = float(hdr['EXPSCORE']) if 'EXPSCORE' in hdr else None
            header_meta['PIXSCALE'] = hdr.get('PIXSCALE', hdr.get('CD1_1', hdr.get('CDELT1', None)))
            if header_meta['PIXSCALE'] is not None:
                header_meta['PIXSCALE'] = abs(float(header_meta['PIXSCALE']))
            header_meta['EXPTIME'] = hdr.get('EXPTIME', None)
            header_meta['FILTER'] = hdr.get('FILTER', None)
            header_meta['TELESCOP'] = hdr.get('TELESCOP', None)
            header_meta['Z_LENS'] = hdr.get('Z_LENS', None)
            header_meta['Z_SOURCE'] = hdr.get('Z_SOURCE', None)
            header_meta['L_SRC'] = float(hdr['L_SRC']) if 'L_SRC' in hdr else None
            header_meta['N_SRC'] = int(hdr['N_SRC']) if 'N_SRC' in hdr else None

            # Auto-extract macro lens parameters if stored in header
            if 'MACRO_TE' in hdr:
                header_meta['macro_model'] = {
                    'theta_E': float(hdr['MACRO_TE']),
                    'center': (float(hdr.get('MACRO_XC', 0.0)), float(hdr.get('MACRO_YC', 0.0))),
                    'q': float(hdr.get('MACRO_Q', 0.8)),
                    'phi': float(hdr.get('MACRO_PA', 0.0)),
                    'gamma': float(hdr.get('MACRO_G', 2.0)),
                    'gamma_ext': float(hdr.get('SHEAR_G', 0.0)),
                    'phi_ext': float(hdr.get('SHEAR_PA', 0.0)),
                    'shear_1': float(hdr.get('SHEAR_1', 0.0)),
                    'shear_2': float(hdr.get('SHEAR_2', 0.0)),
                    'ell_x': float(hdr.get('ELL_X', 0.0)),
                    'ell_y': float(hdr.get('ELL_Y', 0.0))
                }

            # Auto-extract ground truth subhalo properties if stored in header
            if 'SUB_X' in hdr and 'SUB_Y' in hdr:
                header_meta['subhalo_true'] = {
                    'sub_x_true': float(hdr['SUB_X']),
                    'sub_y_true': float(hdr['SUB_Y']),
                    'sub_true_pos': (float(hdr['SUB_X']), float(hdr['SUB_Y'])),
                    'sub_mass_true': float(hdr.get('SUB_MASS', 0.0)),
                    'sub_kappa_s': float(hdr.get('SUB_KS', 0.0)),
                    'sub_theta_s': float(hdr.get('SUB_RS', 0.0))
                }

            # Auto-detect bundled Mask from single multi-extension FITS if present
            if 'MASK' in ext_names:
                header_meta['fit_mask'] = hdul['MASK'].data.astype(bool)

            # Auto-detect bundled PSF from single multi-extension FITS if not explicitly given
            if psf_input is None and 'PSF' in ext_names:
                psf_arr = hdul['PSF'].data.astype(np.float32)

            # Auto-detect bundled Noise/RMS/WHT from single multi-extension FITS if not explicitly given
            if noise_input is None:
                if 'RMS' in ext_names:
                    noise_arr = hdul['RMS'].data.astype(np.float32)
                elif 'WHT' in ext_names:
                    wht = hdul['WHT'].data.astype(np.float32)
                    noise_arr = np.where(wht > 1e-12, 1.0 / np.sqrt(np.maximum(wht, 1e-12)), 1.0)
                elif 'VAR' in ext_names:
                    noise_arr = np.sqrt(np.maximum(hdul['VAR'].data.astype(np.float32), 1e-12))
    elif isinstance(image_input, torch.Tensor):
        image_arr = image_input.squeeze().cpu().numpy().astype(np.float32)
    elif isinstance(image_input, np.ndarray):
        image_arr = image_input.squeeze().astype(np.float32)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # 2. Parse separate PSF input if provided
    if psf_input is not None:
        if isinstance(psf_input, (str, os.PathLike)):
            with fits.open(psf_input) as hdul:
                ext_names = [h.name.upper() for h in hdul]
                if 'PSF' in ext_names:
                    psf_arr = hdul['PSF'].data.astype(np.float32)
                else:
                    psf_arr = hdul[0].data.astype(np.float32)
        elif isinstance(psf_input, torch.Tensor):
            psf_arr = psf_input.squeeze().cpu().numpy().astype(np.float32)
        elif isinstance(psf_input, np.ndarray):
            psf_arr = psf_input.squeeze().astype(np.float32)

    # 3. Parse separate Noise input if provided
    if noise_input is not None:
        if isinstance(noise_input, (str, os.PathLike)):
            with fits.open(noise_input) as hdul:
                ext_names = [h.name.upper() for h in hdul]
                if 'RMS' in ext_names:
                    noise_arr = hdul['RMS'].data.astype(np.float32)
                elif 'WHT' in ext_names:
                    wht = hdul['WHT'].data.astype(np.float32)
                    noise_arr = np.where(wht > 1e-12, 1.0 / np.sqrt(np.maximum(wht, 1e-12)), 1.0)
                elif 'VAR' in ext_names:
                    noise_arr = np.sqrt(np.maximum(hdul['VAR'].data.astype(np.float32), 1e-12))
                else:
                    noise_arr = hdul[0].data.astype(np.float32)
        elif isinstance(noise_input, torch.Tensor):
            noise_arr = noise_input.squeeze().cpu().numpy().astype(np.float32)
        elif isinstance(noise_input, np.ndarray):
            noise_arr = noise_input.squeeze().astype(np.float32)

    return image_arr, psf_arr, noise_arr, header_meta


class SubLensResult:
    """
    Rich container holding detection outputs, extracted data arrays,
    individual component plots, performance metrics, and FITS exporter.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self.subhalo_pos: Tuple[float, float] = data["refined_pos"]
        self.radar_candidate_pos: Tuple[float, float] = data["radar_candidate"]
        self.detection_significance: float = data["radar_score"]
        self.reduced_chi2_initial: float = data["chi2_macro"]
        self.reduced_chi2_final: float = data["chi2_final"]
        self.delta_chi2: float = self.reduced_chi2_initial - self.reduced_chi2_final
        self.elapsed_time: float = data["total_elapsed_time"]
        self.has_redshifts: bool = data["has_redshifts"]
        self.z_l: Optional[float] = data.get("z_l")
        self.z_s: Optional[float] = data.get("z_s")

        # Mass & angular parameters
        self.theta_E_sub: float = data.get("theta_E_sub", 0.0)
        self.physical_mass_solar: Optional[float] = data.get("physical_mass_solar")
        self.relative_mass_ratio: Optional[float] = data.get("relative_mass_ratio")

        # Internal references
        self._sim = data["simulator"]
        self._reconstructed_source: torch.Tensor = data["reconstructed_source"]
        self._modeled_image: torch.Tensor = data["pred_final"]
        self._residual_image: torch.Tensor = data["res_final"]
        self._radar_score_map: torch.Tensor = data["radar_score_map"]
        self._macro_residual: torch.Tensor = data["res_macro"]
        self._obs_image: torch.Tensor = data["obs_image"]

    # ----------------------------------------------------------------------------------
    # 1. Performance Metrics Property
    # ----------------------------------------------------------------------------------
    @property
    def metrics(self) -> Dict[str, Any]:
        """Dictionary of quantitative astronomical performance and accuracy metrics."""
        return {
            "subhalo_pos_arcsec": (float(self.subhalo_pos[0]), float(self.subhalo_pos[1])),
            "subhalo_x_arcsec": float(self.subhalo_pos[0]),
            "subhalo_y_arcsec": float(self.subhalo_pos[1]),
            "radar_candidate_arcsec": (float(self.radar_candidate_pos[0]), float(self.radar_candidate_pos[1])),
            "detection_significance_sigma": float(self.detection_significance),
            "detection_significance_s": float(self.detection_significance),
            "chi2_reduced_initial": float(self.reduced_chi2_initial),
            "chi2_red_initial": float(self.reduced_chi2_initial),
            "chi2_reduced_final": float(self.reduced_chi2_final),
            "chi2_red_final": float(self.reduced_chi2_final),
            "delta_chi2_reduced": float(self.delta_chi2),
            "delta_chi2": float(self._data.get("delta_chi2_total", self.delta_chi2)),
            "physical_mass_solar": self.physical_mass_solar,
            "theta_E_sub_arcsec": float(self.theta_E_sub),
            "relative_mass_ratio": self.relative_mass_ratio,
            "astrometric_error_mas": float(self._data.get("error_mas", 0.0)),
            "source_psnr_db": float(self._data.get("psnr_db", 0.0)),
            "source_ssim": float(self._data.get("ssim", 0.0)),
            "elapsed_seconds": float(self.elapsed_time),
            "has_redshifts": self.has_redshifts,
            "z_lens": self.z_l,
            "z_source": self.z_s,
            "object_id": self._data.get("object_id", self._data.get("lens_id", "Unknown")),
            "lens_id": self._data.get("lens_id", self._data.get("object_id", "Unknown")),
            "ra": self._data.get("ra"),
            "dec": self._data.get("dec"),
            "expert_score": self._data.get("expert_score"),
            "grade": self._data.get("grade", "A"),
            "detection_flag": bool(self._data.get("detection_flag", False))
        }

    # ----------------------------------------------------------------------------------
    # 2. Data Extraction Methods (Numpy / Torch)
    # ----------------------------------------------------------------------------------
    def get_source(self, format: str = "numpy") -> Union[np.ndarray, torch.Tensor]:
        """Extracts the 2D reconstructed pixelated source plane."""
        src = self._reconstructed_source.squeeze()
        return src.cpu().numpy() if format.lower() == "numpy" else src

    def get_model_image(self, format: str = "numpy") -> Union[np.ndarray, torch.Tensor]:
        """Extracts the 2D inverted forward model image (macro + subhalo + PSF)."""
        mod = self._modeled_image.squeeze()
        return mod.cpu().numpy() if format.lower() == "numpy" else mod

    def get_residual_image(self, format: str = "numpy") -> Union[np.ndarray, torch.Tensor]:
        """Extracts the 2D final annihilated image residual (obs - model)."""
        res = self._residual_image.squeeze()
        return res.cpu().numpy() if format.lower() == "numpy" else res

    def get_macro_residual(self, format: str = "numpy") -> Union[np.ndarray, torch.Tensor]:
        """Extracts the 2D smooth macro baseline residual exposing the defect."""
        res = self._macro_residual.squeeze()
        return res.cpu().numpy() if format.lower() == "numpy" else res

    def get_radar_map(self, format: str = "numpy") -> Tuple[Union[np.ndarray, torch.Tensor], List[float]]:
        """Extracts the 2D Astrometric Radar score map and celestial extent [xmin, xmax, ymin, ymax]."""
        smap = self._radar_score_map
        extent = self._data["radar_extent"]
        return (smap.cpu().numpy() if format.lower() == "numpy" else smap), extent

    def get_observation_image(self, format: str = "numpy") -> Union[np.ndarray, torch.Tensor]:
        """Extracts the 2D input observation image."""
        obs = self._obs_image.squeeze()
        return obs.cpu().numpy() if format.lower() == "numpy" else obs

    def get_subhalo_convergence(self, format: str = "numpy") -> Union[np.ndarray, torch.Tensor]:
        """Computes and extracts the 2D convergence field kappa_sub(theta) of the fitted subhalo."""
        device = self._obs_image.device
        sub_center = torch.tensor(self.subhalo_pos, device=device)
        k_val = torch.tensor(self._data.get("final_kappa_s", 0.35), device=device)
        r_s_val = self._data.get("sub_theta_s", 0.15)
        _, kappa_sub = nfw_subhalo_deflection(self._sim.theta_grid, center=sub_center, kappa_s=k_val, r_s=r_s_val)
        kappa_2d = kappa_sub.view(self._sim.N_img, self._sim.N_img)
        return kappa_2d.cpu().numpy() if format.lower() == "numpy" else kappa_2d

    # ----------------------------------------------------------------------------------
    # 3. Individual Component Plotting Methods
    # ----------------------------------------------------------------------------------
    def plot_component(
        self,
        component_name: str,
        save_path: Optional[str] = None,
        cmap: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plots an individual component image:
        - 'observation'
        - 'macro_residual'
        - 'radar_map'
        - 'model'
        - 'final_residual'
        - 'source'
        - 'subhalo_kappa'
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        c_name = component_name.lower().strip()

        if c_name in ['observation', 'obs']:
            data = self.get_observation_image()
            im = ax.imshow(data, origin='lower', extent=self._sim.extent_img, cmap=cmap or 'inferno')
            ax.set_title("Observed Lens Image (SCI)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
        elif c_name in ['macro_residual', 'defect']:
            data = self.get_macro_residual()
            vmax = max(abs(data.min()), abs(data.max()))
            im = ax.imshow(data, origin='lower', extent=self._sim.extent_img, cmap=cmap or 'RdBu_r', vmin=-vmax, vmax=vmax)
            ax.scatter([self.subhalo_pos[0]], [self.subhalo_pos[1]], marker='+', color='black', s=120, label='Subhalo Location')
            ax.set_title(f"Smooth Macro Residual (No Subhalo)\nchi_red^2 = {self.reduced_chi2_initial:.2f}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
            ax.legend(loc='lower left', fontsize=9)
        elif c_name in ['radar_map', 'radar', 'smap']:
            data, extent = self.get_radar_map()
            im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap or 'viridis')
            ax.scatter([self.radar_candidate_pos[0]], [self.radar_candidate_pos[1]], color='yellow', marker='*', s=220, edgecolors='black', label=f"Peak S = {self.detection_significance:.1f} sigma")
            ax.set_title(f"Astrometric Radar Response Map\nPeak S = {self.detection_significance:.1f} sigma", fontsize=11, fontweight='bold')
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
            ax.legend(loc='lower left', fontsize=9)
        elif c_name in ['model', 'final_model']:
            data = self.get_model_image()
            im = ax.imshow(data, origin='lower', extent=self._sim.extent_img, cmap=cmap or 'inferno')
            ax.scatter([self.subhalo_pos[0]], [self.subhalo_pos[1]], facecolors='none', edgecolors='lime', s=160, linewidth=2, label='Inverted Subhalo')
            ax.set_title("Final Joint Inverted Model Image", fontsize=11, fontweight='bold')
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
            ax.legend(loc='lower left', fontsize=9)
        elif c_name in ['final_residual', 'residual']:
            data = self.get_residual_image()
            vmax = max(abs(data.min()), abs(data.max()))
            im = ax.imshow(data, origin='lower', extent=self._sim.extent_img, cmap=cmap or 'RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f"Final Inversion Residual\nchi_red^2 = {self.reduced_chi2_final:.2f} (Noise Limit)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
        elif c_name in ['source', 'reconstructed_source']:
            data = self.get_source()
            im = ax.imshow(data, origin='lower', extent=self._sim.extent_src, cmap=cmap or 'magma')
            ax.set_title("Reconstructed Source Galaxy (TV Regularized)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
        elif c_name in ['subhalo_kappa', 'kappa']:
            data = self.get_subhalo_convergence()
            im = ax.imshow(data, origin='lower', extent=self._sim.extent_img, cmap=cmap or 'plasma')
            ax.scatter([self.subhalo_pos[0]], [self.subhalo_pos[1]], color='cyan', marker='x', s=140, label=f"Center: ({self.subhalo_pos[0]:+.3f}\", {self.subhalo_pos[1]:+.3f}\")")
            ax.set_title("Fitted Dark Matter Subhalo Convergence kappa_sub", fontsize=11, fontweight='bold')
            ax.set_xlabel("Arcsec")
            ax.set_ylabel("Arcsec")
            ax.legend(loc='lower left', fontsize=9)
        else:
            plt.close()
            raise ValueError(f"Unknown component: '{component_name}'. Choose from: 'observation', 'macro_residual', 'radar_map', 'model', 'final_residual', 'source', 'subhalo_kappa'")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi)
            print(f">>> Component plot [{component_name}] saved to: {save_path}")
            plt.close()
        return fig

    def plot_macro_residual(self, save_path: str = "macro_residual.png"):
        """Saves dedicated smooth macro residual plot exposing the defect."""
        return self.plot_component("macro_residual", save_path=save_path)

    def plot_radar_map(self, save_path: str = "radar_smap.png"):
        """Saves dedicated Astrometric Radar S-map plot."""
        return self.plot_component("radar_map", save_path=save_path)

    def plot_final_model(self, save_path: str = "final_model.png"):
        """Saves dedicated final joint inverted model image plot."""
        return self.plot_component("model", save_path=save_path)

    def plot_final_residual(self, save_path: str = "final_residual.png"):
        """Saves dedicated final residual plot showing defect annihilation."""
        return self.plot_component("final_residual", save_path=save_path)

    def plot_reconstructed_source(self, save_path: str = "reconstructed_source.png"):
        """Saves dedicated TV-regularized reconstructed source galaxy plot."""
        return self.plot_component("source", save_path=save_path)

    def plot_subhalo_convergence(self, save_path: str = "subhalo_convergence.png"):
        """Saves dedicated dark matter subhalo convergence field plot."""
        return self.plot_component("subhalo_kappa", save_path=save_path)

    # ----------------------------------------------------------------------------------
    # 4. Multi-Extension FITS Exporter
    # ----------------------------------------------------------------------------------
    def save_fits(self, save_path: str = "sublens_inversion_results.fits", overwrite: bool = True):
        """
        Exports all inversion outputs to a standard Multi-Extension FITS file:
        - HDU 0: MODEL (Inverted lens image)
        - HDU 1: RESIDUAL (Final data - model residual)
        - HDU 2: SOURCE (Reconstructed source plane)
        - HDU 3: RADAR_SMAP (Astrometric radar score map)
        - HDU 4: SUBHALO_KAPPA (Subhalo 2D convergence field)
        - Header keywords containing all converged parameters, astrometry, and chi-squared metrics.
        """
        hdr = fits.Header()
        hdr['TELESCOP'] = ('SubLens', 'Gravitational Lensing Inversion Pipeline')
        hdr['SUB_X']    = (float(self.subhalo_pos[0]), 'Fitted subhalo X (arcsec)')
        hdr['SUB_Y']    = (float(self.subhalo_pos[1]), 'Fitted subhalo Y (arcsec)')
        hdr['RADAR_S']  = (float(self.detection_significance), 'Radar detection score (sigma)')
        hdr['CHI2_INI'] = (float(self.reduced_chi2_initial), 'Smooth macro reduced chi-squared')
        hdr['CHI2_FIN'] = (float(self.reduced_chi2_final), 'Final inverted reduced chi-squared')
        hdr['DEL_CHI2'] = (float(self.delta_chi2), 'Chi-squared improvement')
        hdr['THETA_E']  = (float(self.theta_E_sub), 'Inverted subhalo theta_E (arcsec)')

        if self.has_redshifts:
            hdr['Z_LENS']   = (float(self.z_l), 'Host lens redshift')
            hdr['Z_SOURCE'] = (float(self.z_s), 'Source galaxy redshift')
            if self.physical_mass_solar is not None:
                hdr['M_SOLAR']  = (float(self.physical_mass_solar), 'Inverted subhalo physical mass (M_sun)')
        else:
            hdr['Z_LENS']   = (-1.0, 'Unknown redshift (scale-free)')
            hdr['Z_SOURCE'] = (-1.0, 'Unknown redshift (scale-free)')
            if self.relative_mass_ratio is not None:
                hdr['F_SUB']    = (float(self.relative_mass_ratio), 'Relative mass ratio f_sub')

        # HDUs
        hdu_model = fits.PrimaryHDU(data=self.get_model_image(), header=hdr)
        hdu_model.header['EXTNAME'] = 'MODEL'
        hdu_model.header['COMMENT'] = 'Best-fit forward modeled lens image'

        hdu_res = fits.ImageHDU(data=self.get_residual_image())
        hdu_res.header['EXTNAME'] = 'RESIDUAL'
        hdu_res.header['COMMENT'] = 'Final data - model image residual'

        hdu_src = fits.ImageHDU(data=self.get_source())
        hdu_src.header['EXTNAME'] = 'SOURCE'
        hdu_src.header['COMMENT'] = 'Reconstructed TV-regularized source plane'

        smap, _ = self.get_radar_map()
        hdu_radar = fits.ImageHDU(data=smap)
        hdu_radar.header['EXTNAME'] = 'RADAR_SMAP'
        hdu_radar.header['COMMENT'] = 'Astrometric matched filter S-score map (sigma)'

        hdu_kappa = fits.ImageHDU(data=self.get_subhalo_convergence())
        hdu_kappa.header['EXTNAME'] = 'SUBHALO_KAPPA'
        hdu_kappa.header['COMMENT'] = 'Fitted dark matter subhalo convergence field'

        hdul = fits.HDUList([hdu_model, hdu_res, hdu_src, hdu_radar, hdu_kappa])
        hdul.writeto(save_path, overwrite=overwrite)
        print(f">>> Full inversion results exported to FITS: {save_path}")

    # ----------------------------------------------------------------------------------
    # 5. Reporting and Full Overview Plotting
    # ----------------------------------------------------------------------------------
    def summary(self) -> str:
        """Prints and returns a formatted scientific report of the detection."""
        lines = []
        lines.append("=" * 75)
        lines.append(">>> SubLens Automated Subhalo Discovery Report")
        lines.append("=" * 75)
        lines.append(f"Execution Latency: {self.elapsed_time:.2f} s")
        lines.append(f"Radar Blind Candidate: ({self.radar_candidate_pos[0]:+.4f}\", {self.radar_candidate_pos[1]:+.4f}\")")
        lines.append(f"Radar Detection Significance: {self.detection_significance:.2f} sigma")
        lines.append(f"PINN Converged Position: ({self.subhalo_pos[0]:+.4f}\", {self.subhalo_pos[1]:+.4f}\")")
        lines.append(f"Reduced Chi-Square: {self.reduced_chi2_initial:.2f} -> {self.reduced_chi2_final:.2f} (Delta chi^2 = {self.delta_chi2:.2f})")

        if self.has_redshifts:
            lines.append(f"Cosmology: z_lens = {self.z_l:.3f}, z_source = {self.z_s:.3f}")
            if self.physical_mass_solar is not None:
                lines.append(f"Inverted Physical Mass: {self.physical_mass_solar:.3e} M_sun")
        else:
            lines.append("Cosmology: UNKNOWN REDSHIFTS (Scale-Free Regime)")
            lines.append(f"Inverted Angular Scale: theta_E,sub = {self.theta_E_sub:.4f}\"")
            if self.relative_mass_ratio is not None:
                lines.append(f"Relative Mass Ratio: f_sub = {self.relative_mass_ratio:.4e}")
            lines.append("Physical Mass Degeneracy: M(z_l, z_s) = Sigma_cr(z_l, z_s) * D_l^2 * pi * theta_E^2")

        lines.append("=" * 75)
        report = "\n".join(lines)
        print(report)
        return report

    def plot(self, save_path: str = "sublens_detection_results.png", component: Optional[str] = None, dpi: int = 300):
        """
        Plots either:
        - An individual component if `component` is given (e.g. 'residual', 'radar', 'source', 'model')
        - The complete 8-panel diagnostic overview figure if `component` is None.
        """
        if component is not None:
            return self.plot_component(component, save_path=save_path, dpi=dpi)

        if self._data.get("is_real_observation", False):
            plot_dict = {
                "obs_image": self._obs_image,
                "noise_map": self._data.get("noise_map", self._sim.noise_map if hasattr(self._sim, 'noise_map') else None),
                "pred_macro": self._data.get("pred_macro", self._modeled_image),
                "pred_final": self._modeled_image,
                "res_macro": self._macro_residual,
                "res_final": self._residual_image,
                "source_final": self._reconstructed_source,
                "radar_scores": self._radar_score_map,
                "radar_coords": self._data.get("radar_coords"),
                "radar_pos": self.radar_candidate_pos,
                "pinn_pos": self.subhalo_pos,
                "radar_score": self.detection_significance,
                "chi2_macro": self.reduced_chi2_initial,
                "chi2_final": self.reduced_chi2_final,
                "extent_img": self._sim.extent_img if hasattr(self._sim, 'extent_img') else [-5.0, 5.0, -5.0, 5.0],
                "radar_extent": self._data.get("radar_extent", [-1.5, 1.5, -1.5, 1.5]),
                "source_extent": self._data.get("source_extent", [-self._sim.L_src / 2.0, self._sim.L_src / 2.0, -self._sim.L_src / 2.0, self._sim.L_src / 2.0] if hasattr(self._sim, 'L_src') else [-1.0, 1.0, -1.0, 1.0]),
                "fit_mask": self._data.get("fit_mask"),
                "lens_id": self._data.get("lens_id", "Euclid Q1 Lens"),
                "theta_e": self._data.get("host_theta_E", 1.0)
            }
            plot_real_observation_figure(plot_dict, save_path=save_path, dpi=dpi)
            print(f">>> 8-panel real observation diagnostic overview saved to: {save_path}")
            return

        plot_dict = {
            "simulator": self._sim,
            "obs_image": self._obs_image,
            "res_macro": self._macro_residual,
            "res_final": self._residual_image,
            "pred_final": self._modeled_image,
            "source_true": self._data.get("source_true", self._reconstructed_source),
            "source_final": self._reconstructed_source,
            "radar_scores": self._radar_score_map,
            "radar_extent": self._data["radar_extent"],
            "sub_true_pos": self._data.get("sub_true_pos", self.subhalo_pos),
            "radar_pos": self.radar_candidate_pos,
            "pinn_pos": self.subhalo_pos,
            "radar_score": self.detection_significance,
            "chi2_macro": self.reduced_chi2_initial,
            "chi2_final": self.reduced_chi2_final,
            "error_mas": self._data.get("error_mas", 0.0),
            "psnr_db": self._data.get("psnr_db", 0.0),
            "ssim": self._data.get("ssim", 0.0),
            "title": "SubLens Pipeline Inversion"
        }
        plot_diagnostic_figure(plot_dict, save_path=save_path, dpi=dpi)
        print(f">>> 8-panel diagnostic overview saved to: {save_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Exports detection parameters as a serializable dictionary."""
        return self.metrics


class SubLensPipeline:
    """
    High-Level End-to-End Gravitational Lensing Inversion & Subhalo Pipeline.

    Supports:
    - FITS input files (single multi-extension FITS or separate files for SCI, PSF, RMS)
    - In-memory NumPy arrays and PyTorch tensors
    - Redshift-aware and redshift-agnostic modes
    """

    def __init__(
        self,
        z_l: Optional[float] = None,
        z_s: Optional[float] = None,
        pixel_scale: Optional[float] = None,
        psf_fwhm: float = 0.09,
        noise_std: float = 0.012,
        psf_kernel: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
        noise_map: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
        N_img: int = 120,
        N_src: int = 64,
        L_src: float = 1.8,
        device: str = "auto"
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.z_l = z_l
        self.z_s = z_s
        self.has_redshifts = (z_l is not None and z_s is not None)
        self.cosmology = LensingCosmology(z_l=z_l, z_s=z_s)

        # Parse potential FITS paths, numpy arrays, or tensors in psf_kernel or noise_map
        parsed_psf = None
        if psf_kernel is not None:
            if isinstance(psf_kernel, (np.ndarray, torch.Tensor)):
                parsed_psf = psf_kernel.squeeze().cpu().numpy() if isinstance(psf_kernel, torch.Tensor) else psf_kernel.squeeze()
            elif isinstance(psf_kernel, str):
                p_img, p_psf, _, _ = _parse_fits_input(psf_kernel)
                parsed_psf = p_psf if p_psf is not None else p_img

        parsed_noise = None
        if noise_map is not None:
            if isinstance(noise_map, (np.ndarray, torch.Tensor)):
                parsed_noise = noise_map.squeeze().cpu().numpy() if isinstance(noise_map, torch.Tensor) else noise_map.squeeze()
            elif isinstance(noise_map, str):
                n_img, _, n_noise, _ = _parse_fits_input(noise_map)
                parsed_noise = n_noise if n_noise is not None else n_img

        self.default_pixel_scale = pixel_scale or 0.05
        self.psf_fwhm = psf_fwhm
        self.noise_std = noise_std
        self.N_img = N_img
        self.N_src = N_src
        self.L_src = L_src
        self.psf_kernel_input = parsed_psf
        self.noise_map_input = parsed_noise

        self.simulator = LensingSimulator(
            N_img=N_img,
            pixel_scale=self.default_pixel_scale,
            N_src=N_src,
            L_src=L_src,
            psf_fwhm=psf_fwhm,
            noise_std=noise_std,
            psf_kernel=self.psf_kernel_input,
            noise_map=self.noise_map_input,
            device=self.device
        )

    @property
    def pixel_scale(self) -> float:
        """Instrument pixel scale in arcseconds per pixel."""
        return self.simulator.pixel_scale

    def observation(self, image=None, *, noise=None, psf=None, macro_model=None,
                    fit_mask=None, alpha_macro=None):
        """Load generic lensing data from image/noise/PSF arrays or a FITS bundle."""
        from .observation import LensingObservation

        if isinstance(image, LensingObservation):
            if any(value is not None for value in (noise, psf, macro_model, alpha_macro)):
                raise ValueError("Pass array overrides when creating the observation")
            return image if fit_mask is None else replace(
                image, mask=image._fit_mask(fit_mask),
            )
        if image is None:
            image = getattr(self, "_fits_path", getattr(self, "_cached_image", None))
            noise = getattr(self, "_fits_noise", None) if noise is None else noise
            psf = getattr(self, "_fits_psf", None) if psf is None else psf
        if image is None:
            raise ValueError("Supply an image array or FITS path")
        if isinstance(image, (str, os.PathLike)):
            return LensingObservation.from_fits(
                image, noise=noise, psf=psf, macro_model=macro_model, mask=fit_mask,
            )
        model = macro_model if macro_model is not None else getattr(self, "macro_model", None)
        if noise is None:
            noise = self.simulator.noise_map[0, 0]
            shape = tuple(image.shape[-2:])
            if shape != tuple(noise.shape):
                noise = self.noise_std
        return LensingObservation.from_arrays(
            image, noise,
            self.simulator.psf_kernel[0, 0].detach().cpu().numpy()[::-1, ::-1]
            if psf is None else psf,
            pixel_scale=self.pixel_scale, macro_model=model, alpha_macro=alpha_macro, mask=fit_mask,
        )

    def reconstruct_delaunay(self, image=None, *, noise=None, psf=None, macro_model=None,
                             alpha_macro=None, fit_mask=None, validation_mask=None, **options):
        """Fit a Delaunay source from observation arrays, an object or a FITS path."""
        observation = self.observation(
            image, noise=noise, psf=psf, macro_model=macro_model,
            alpha_macro=alpha_macro, fit_mask=fit_mask,
        )
        options.setdefault("device", self.device)
        return observation.reconstruct_delaunay(validation_mask=validation_mask, **options)

    def search_pinn(self, image=None, *, noise=None, psf=None, macro_model=None,
                    alpha_macro=None, fit_mask=None, joint_refine=False, **options):
        """Run PINN candidate discovery and optional joint macro/source refinement."""
        observation = self.observation(
            image, noise=noise, psf=psf, macro_model=macro_model,
            alpha_macro=alpha_macro, fit_mask=fit_mask,
        )
        if joint_refine and observation.macro_model is None:
            raise ValueError("Joint refinement needs macro_model or macro parameters in the FITS header")
        result = observation.search_pinn(**options)
        return result.refine_joint(observation.macro_model) if joint_refine else result

    def analyze_lens(self, image=None, *, noise=None, psf=None, lens_light=None,
                     mask=None, pixel_scale=None, **options):
        """Complete PINN-only analysis with automatic macro initialization."""
        from .analysis import analyze_lens
        from .data import LensData
        if image is None:image=getattr(self,'_fits_path',None)
        if image is None:raise ValueError('Supply measured image arrays or a FITS path')
        if isinstance(image,LensData):
            options.setdefault('device',self.device)
            return analyze_lens(image,noise=noise,psf=psf,lens_light=lens_light,
                                mask=mask,pixel_scale=pixel_scale,**options)
        if noise is None:noise=self.noise_map_input
        if psf is None:psf=self.psf_kernel_input
        options.setdefault('device',self.device)
        if pixel_scale is None and not isinstance(image,(str,os.PathLike)):
            pixel_scale=self.pixel_scale
        return analyze_lens(image,noise=noise,psf=psf,lens_light=lens_light,mask=mask,
                            pixel_scale=pixel_scale,**options)

    @classmethod
    def from_fits(
        cls,
        fits_path: str,
        psf_path: Optional[str] = None,
        noise_path: Optional[str] = None,
        z_l: Optional[float] = None,
        z_s: Optional[float] = None,
        device: str = "auto",
        **kwargs
    ) -> "SubLensPipeline":
        """
        Factory method to initialize SubLensPipeline directly from a FITS file.
        Automatically parses PIXSCALE, Z_LENS, Z_SOURCE, PSF, and RMS maps if present.
        """
        img_data, psf_data, noise_data, meta = _parse_fits_input(fits_path, psf_path, noise_path)

        pix_scale = meta.get('PIXSCALE') or kwargs.get('pixel_scale', 0.05)
        zl_val = z_l if z_l is not None else meta.get('Z_LENS')
        zs_val = z_s if z_s is not None else meta.get('Z_SOURCE')
        l_src_val = meta.get('L_SRC') or kwargs.get('L_src')
        if l_src_val is None:
            macro_info = meta.get('macro_model', {})
            te = macro_info.get('theta_E', 1.40)
            l_src_val = min(1.8, max(0.6, 1.35 * te))
        n_src_val = meta.get('N_SRC') or kwargs.get('N_src', 64)
        n_img = img_data.shape[0]

        kwargs_clean = kwargs.copy()
        kwargs_clean.pop('pixel_scale', None)
        kwargs_clean.pop('N_src', None)
        kwargs_clean.pop('L_src', None)

        instance = cls(
            z_l=zl_val,
            z_s=zs_val,
            pixel_scale=pix_scale,
            psf_kernel=psf_data,
            noise_map=noise_data,
            N_img=n_img,
            N_src=n_src_val,
            L_src=l_src_val,
            device=device,
            **kwargs_clean
        )
        instance._cached_image = img_data
        instance._fits_path = fits_path
        instance._fits_noise = noise_path
        instance._fits_psf = psf_path
        instance._fits_meta = meta
        if 'fit_mask' in meta:
            instance.fit_mask = meta['fit_mask']
        if 'macro_model' in meta:
            instance.macro_model = meta['macro_model']
        if 'subhalo_true' in meta:
            instance.subhalo_true = meta['subhalo_true']
        return instance

    @classmethod
    def prepare_euclid_fits(
        cls,
        lens_id: str,
        data_dir: str = "archive/data",
        output_path: Optional[str] = None,
        crop_size: int = 100,
        **kwargs
    ) -> str:
        """Standard FITS ingestion for Euclid Q1 Early Release Observations."""
        from .survey.euclid import prepare_euclid_grade_a_fits
        return prepare_euclid_grade_a_fits(lens_id, data_dir=data_dir, output_path=output_path, crop_size=crop_size, **kwargs)


    @classmethod
    def generate_benchmark_fits(
        cls,
        system: str = "j0946",
        mode: str = "single",
        output_dir: str = ".",
        prefix: Optional[str] = None,
        kernel_size: int = 31,
        device: str = "auto",
        seed: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesizes authentic HST astronomical benchmark datasets for:
        - 'j0946': SDSS J0946+5106 (Vegetti et al. 2010 Nature)
        - 'b1938': JVAS B1938+666 (Vegetti et al. 2012 Nature)
        Outputs single multi-extension FITS (mode='single') or 3 separate FITS (mode='separate').
        """
        from .simulation.hst import generate_benchmark_fits
        return generate_benchmark_fits(
            system=system,
            mode=mode,
            output_dir=output_dir,
            prefix=prefix,
            kernel_size=kernel_size,
            device=device,
            seed=seed,
            **kwargs
        )

    def fit(
        self,
        image: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
        macro_model: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
        psf_kernel: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
        noise_map: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
        fit_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        subhalo_type: str = "NFW",
        radar_resolution: int = 45,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None,
        scan_range: Optional[float] = None,
        test_subhalo_mass: Optional[float] = None,
        num_macro_steps: int = 120,
        num_pinn_steps: int = 120,
        ground_truth: Optional[Dict[str, Any]] = None,
        source_fov: Optional[Union[float, str]] = "auto",
        source_resolution: Optional[int] = None,
        fix_macro: bool = True,
        verbose: bool = True,
        proposal: Optional[str] = None,
        joint_refine: bool = False
    ) -> SubLensResult:
        """
        Executes full automated 3-stage inversion:
        1. Smooth macro baseline fit + TV source inversion
        2. Astrometric Radar annular blind scan
        3. PINN continuous autograd refinement

        With proposal="pinn", return PinnHaloSearchResult from Delaunay source
        reconstruction and neural-potential discovery. joint_refine enables
        joint EPL normalization, shear, halo and source refinement.
        """
        if proposal is not None and proposal != "pinn":
            raise ValueError("Use proposal='pinn' for neural-potential discovery")
        if proposal == "pinn":
            return self.search_pinn(
                image, noise=noise_map, psf=psf_kernel, macro_model=macro_model,
                fit_mask=fit_mask, joint_refine=joint_refine, pinn_steps=num_pinn_steps,
            )
        if joint_refine:
            raise ValueError("Use proposal='pinn' with joint_refine=True")
        t0 = time.time()

        # Check for cached image from from_fits if image not explicitly provided
        if image is None:
            if hasattr(self, '_cached_image') and self._cached_image is not None:
                image = self._cached_image
            else:
                raise ValueError("Must provide image either to fit() or via from_fits().")

        # Parse FITS / Array inputs
        img_arr, psf_arr, noise_arr, meta = _parse_fits_input(image, psf_kernel, noise_map)
        if hasattr(self, '_fits_meta') and self._fits_meta:
            meta = {**self._fits_meta, **meta}

        # Auto-extract macro_model if omitted
        if macro_model is None:
            if hasattr(self, 'macro_model') and self.macro_model is not None:
                macro_model = self.macro_model
            elif 'macro_model' in meta:
                macro_model = meta['macro_model']
            else:
                raise ValueError(
                    "macro_model must be provided either explicitly to fit() or via FITS header cards (MACRO_TE, etc.)."
                )

        # Check for fit_mask (for real masked survey observations)
        active_mask = None
        if fit_mask is not None:
            active_mask = fit_mask
        elif hasattr(self, 'fit_mask') and self.fit_mask is not None:
            active_mask = self.fit_mask
        elif 'fit_mask' in meta:
            active_mask = meta['fit_mask']

        if active_mask is not None:
            from .survey.euclid import fit_real_lens_observation
            return fit_real_lens_observation(
                pipeline=self,
                image_arr=img_arr,
                psf_arr=psf_arr if psf_arr is not None else self.simulator.psf_kernel,
                noise_arr=noise_arr if noise_arr is not None else self.simulator.noise_map,
                fit_mask=active_mask,
                macro_model=macro_model,
                meta=meta,
                radar_resolution=radar_resolution,
                num_macro_steps=num_macro_steps,
                num_pinn_steps=num_pinn_steps,
                source_fov=source_fov,
                source_resolution=source_resolution,
                fix_macro=fix_macro,
                verbose=verbose
            )

        # Auto-extract ground_truth if omitted
        if ground_truth is None:
            if hasattr(self, 'subhalo_true') and self.subhalo_true is not None:
                ground_truth = self.subhalo_true
            elif 'subhalo_true' in meta:
                ground_truth = meta['subhalo_true']

        # Update simulator if custom PSF or Noise map arrived with fit()
        if psf_arr is not None or noise_arr is not None or img_arr.shape[0] != self.simulator.N_img:
            pix_scale = meta.get('PIXSCALE') or self.simulator.pixel_scale
            self.simulator = LensingSimulator(
                N_img=img_arr.shape[0],
                pixel_scale=pix_scale,
                N_src=self.N_src,
                L_src=self.L_src,
                psf_fwhm=self.psf_fwhm,
                noise_std=self.noise_std,
                psf_kernel=psf_arr if psf_arr is not None else self.simulator.psf_kernel,
                noise_map=noise_arr if noise_arr is not None else self.simulator.noise_map,
                device=self.device
            )

        image_tensor = torch.tensor(img_arr, dtype=torch.float32, device=self.device)
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.view(1, 1, self.simulator.N_img, self.simulator.N_img)

        # Format macro deflection
        if isinstance(macro_model, dict):
            a_host, _ = epl_deflection(
                self.simulator.theta_grid,
                theta_E=macro_model.get("theta_E", 1.40),
                q=macro_model.get("q", 0.80),
                phi=macro_model.get("phi", 0.0),
                gamma=macro_model.get("gamma", 2.0),
                center=macro_model.get("center", (0.0, 0.0))
            )
            a_shear = external_shear_deflection(
                self.simulator.theta_grid,
                gamma_ext=macro_model.get("gamma_ext", 0.0),
                phi_ext=macro_model.get("phi_ext", 0.0)
            )
            alpha_macro = a_host + a_shear
        else:
            alpha_macro = macro_model.to(self.device)

        # Adapt source scale and scan range
        host_theta_E = 1.40
        if isinstance(macro_model, dict) and "theta_E" in macro_model:
            host_theta_E = macro_model["theta_E"]

        # Auto-adapt source plane extent to compact lenses if not customized
        if self.simulator.L_src > 1.30 and host_theta_E < 0.8:
            adapted_l_src = min(1.8, max(0.6, 1.35 * host_theta_E))
            self.L_src = adapted_l_src
            self.simulator = LensingSimulator(
                N_img=self.simulator.N_img,
                pixel_scale=self.simulator.pixel_scale,
                N_src=self.simulator.N_src,
                L_src=adapted_l_src,
                psf_fwhm=self.psf_fwhm,
                noise_std=self.noise_std,
                psf_kernel=self.simulator.psf_kernel,
                noise_map=self.simulator.noise_map,
                device=self.device
            )

        if scan_range is None:
            scan_range = min(max(1.35 * host_theta_E, host_theta_E + 0.35), 0.90 * self.simulator.L_img)

        if r_min is None:
            r_min = max(0.20, host_theta_E - 0.25) if host_theta_E < 0.8 else max(0.40, host_theta_E - 0.50)
        if r_max is None:
            r_max = host_theta_E + 0.35 if host_theta_E < 0.8 else host_theta_E + 0.55

        if test_subhalo_mass is None:
            if self.has_redshifts:
                test_subhalo_mass = 3.5e9 * min(1.0, (host_theta_E / 1.43)**2)
            else:
                test_subhalo_mass = 0.035 * min(1.0, (host_theta_E / 1.43)**2)

        # Stage 1: Macro Baseline Fit
        if verbose:
            print(">>> [Pipeline Stage 1] Inverting smooth macro baseline (TV pixelated source)...")
        macro_fitter = MacroFitter(self.simulator, num_steps=num_macro_steps)
        macro_res = macro_fitter.fit(image_tensor, alpha_macro, verbose=False)

        # Stage 2: Astrometric Radar Blind Scan
        if verbose:
            print(">>> [Pipeline Stage 2] Scanning Einstein ring manifold with Astrometric Radar...")
        radar = AstrometricRadar(self.simulator, grid_resolution=radar_resolution, r_min=r_min, r_max=r_max)
        test_mass = test_subhalo_mass
        radar_res = radar.scan(
            image_tensor,
            pred_macro=macro_res["pred_macro"],
            res_macro=macro_res["res_macro"],
            source_macro=macro_res["source_reconstructed"],
            alpha_macro=alpha_macro,
            test_subhalo_mass=test_mass,
            subhalo_type=subhalo_type,
            cosmology=self.cosmology,
            scan_range=scan_range,
            verbose=False
        )

        # Stage 3: PINN Continuous Refinement
        if self.has_redshifts and subhalo_type.upper() == "NFW":
            init_k, init_rs = self.cosmology.physical_mass_to_nfw(test_mass, c_200=18.0, concentration_boost=25.0)
        elif subhalo_type.upper() == "NFW":
            init_k, init_rs = 0.35, 0.15
        else:
            init_k, init_rs = 0.035, 0.04

        if verbose:
            print(f">>> [Pipeline Stage 3] Refining subhalo from radar seed ({radar_res['best_candidate'][0]:+.3f}\", {radar_res['best_candidate'][1]:+.3f}\")...")
        pinn = PINNOptimizer(self.simulator, lr_pos=0.015, lr_mass=0.02, lr_src=0.05, num_steps=num_pinn_steps)
        pinn_res = pinn.refine(
            image_tensor,
            init_pos=radar_res["best_candidate"],
            init_src_param=macro_res["source_param"],
            alpha_macro=alpha_macro,
            vignette_mask=macro_res["vignette"],
            subhalo_type=subhalo_type,
            r_s=init_rs,
            init_kappa=init_k,
            verbose=False
        )

        total_time = time.time() - t0

        # Physical & scale-free parameter derivation
        physical_mass = None
        theta_E_sub = pinn_res["final_kappa_s"]
        rel_ratio = None

        if self.has_redshifts:
            try:
                physical_mass = self.cosmology.nfw_to_physical_mass(
                    pinn_res["final_kappa_s"], theta_s=init_rs
                )
            except Exception:
                physical_mass = None
        else:
            rel_ratio = (theta_E_sub / host_theta_E)**2

        # Optional comparison metrics if ground truth is supplied
        err_mas = 0.0
        psnr_val = 0.0
        ssim_val = 0.0
        if ground_truth:
            if "sub_true_pos" in ground_truth:
                err_dict = compute_astrometric_error(
                    pinn_res["refined_pos"], ground_truth["sub_true_pos"], self.simulator.pixel_scale
                )
                err_mas = err_dict["error_mas"]
            if "source_true" in ground_truth:
                img_m = compute_image_metrics(pinn_res["final_source"], ground_truth["source_true"])
                psnr_val = img_m["psnr_db"]
                ssim_val = img_m["ssim"]

        result_dict = {
            "simulator": self.simulator,
            "obs_image": image_tensor,
            "res_macro": macro_res["res_macro"],
            "pred_final": pinn_res["pred_final"],
            "res_final": pinn_res["res_final"],
            "reconstructed_source": pinn_res["final_source"],
            "radar_score_map": radar_res["score_map"],
            "radar_extent": radar_res["extent"],
            "radar_candidate": radar_res["best_candidate"],
            "radar_score": radar_res["max_score"],
            "refined_pos": pinn_res["refined_pos"],
            "chi2_macro": macro_res["chi2_red"],
            "chi2_final": pinn_res["chi2_final"],
            "total_elapsed_time": total_time,
            "has_redshifts": self.has_redshifts,
            "z_l": self.z_l,
            "z_s": self.z_s,
            "theta_E_sub": theta_E_sub,
            "physical_mass_solar": physical_mass,
            "relative_mass_ratio": rel_ratio,
            "error_mas": err_mas,
            "psnr_db": psnr_val,
            "ssim": ssim_val,
            "final_kappa_s": pinn_res.get("final_kappa_s", 0.35),
            "sub_theta_s": init_rs
        }
        if ground_truth:
            result_dict.update(ground_truth)

        res = SubLensResult(result_dict)
        if verbose:
            res.summary()

        return res


def detect_subhalo(
    image: Union[str, torch.Tensor, np.ndarray],
    macro_model: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
    z_l: Optional[float] = None,
    z_s: Optional[float] = None,
    pixel_scale: Optional[float] = None,
    psf_fwhm: float = 0.09,
    noise_std: float = 0.012,
    psf_kernel: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
    noise_map: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
    subhalo_type: str = "NFW",
    device: str = "auto",
    verbose: bool = True,
    **kwargs
) -> SubLensResult:
    """
    Convenience function for one-line subhalo detection.
    Supports single FITS file path, separate FITS files, or NumPy arrays.
    """
    if isinstance(image, str):
        pipeline = SubLensPipeline.from_fits(
            fits_path=image,
            psf_path=psf_kernel if isinstance(psf_kernel, str) else None,
            noise_path=noise_map if isinstance(noise_map, str) else None,
            z_l=z_l,
            z_s=z_s,
            device=device,
            **kwargs
        )
        return pipeline.fit(
            image=None,
            macro_model=macro_model,
            subhalo_type=subhalo_type,
            verbose=verbose,
            **kwargs
        )
    else:
        pipeline = SubLensPipeline(
            z_l=z_l,
            z_s=z_s,
            pixel_scale=pixel_scale or 0.05,
            psf_fwhm=psf_fwhm,
            noise_std=noise_std,
            psf_kernel=psf_kernel,
            noise_map=noise_map,
            device=device
        )
        return pipeline.fit(image, macro_model, subhalo_type=subhalo_type, verbose=verbose, **kwargs)
