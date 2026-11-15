"""
High-Level Unified API for SubLens.

Provides user-friendly, production-ready interfaces:
- SubLensPipeline: End-to-end detection and inversion pipeline
- SubLensResult: Rich container holding detection metrics, plots, and summaries
- detect_subhalo(): Functional entrypoint for quick analysis
"""

import time
import math
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np

from .cosmology import LensingCosmology
from .simulation.simulator import LensingSimulator
from .profiles.macro import epl_deflection, external_shear_deflection
from .inversion.macro_fitter import MacroFitter
from .inversion.radar import AstrometricRadar
from .inversion.pinn_optimizer import PINNOptimizer
from .utils.metrics import compute_astrometric_error, compute_image_metrics
from .utils.visualization import plot_diagnostic_figure


class SubLensResult:
    """Rich container storing all inversion artifacts, parameters, and metrics."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self.subhalo_pos: Tuple[float, float] = data["refined_pos"]
        self.radar_candidate_pos: Tuple[float, float] = data["radar_candidate"]
        self.detection_significance: float = data["radar_score"]
        self.reduced_chi2_initial: float = data["chi2_macro"]
        self.reduced_chi2_final: float = data["chi2_final"]
        self.elapsed_time: float = data["total_elapsed_time"]
        self.has_redshifts: bool = data["has_redshifts"]
        self.z_l: Optional[float] = data.get("z_l")
        self.z_s: Optional[float] = data.get("z_s")

        # Mass & angular parameters
        self.theta_E_sub: float = data.get("theta_E_sub", 0.0)
        self.physical_mass_solar: Optional[float] = data.get("physical_mass_solar")
        self.relative_mass_ratio: Optional[float] = data.get("relative_mass_ratio")

        # Tensors
        self.reconstructed_source: torch.Tensor = data["reconstructed_source"]
        self.modeled_image: torch.Tensor = data["pred_final"]
        self.residual_image: torch.Tensor = data["res_final"]
        self.radar_score_map: torch.Tensor = data["radar_score_map"]

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
        lines.append(f"Reduced Chi-Square: {self.reduced_chi2_initial:.2f} -> {self.reduced_chi2_final:.2f} (Delta chi^2 = {self.reduced_chi2_initial - self.reduced_chi2_final:.2f})")

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

    def plot(self, save_path: str = "sublens_detection_results.png", dpi: int = 300):
        """Generates the 8-panel diagnostic publication figure."""
        plot_dict = {
            "simulator": self._data["simulator"],
            "obs_image": self._data["obs_image"],
            "res_macro": self._data["res_macro"],
            "res_final": self.residual_image,
            "pred_final": self.modeled_image,
            "source_true": self._data.get("source_true", self.reconstructed_source),
            "source_final": self.reconstructed_source,
            "radar_scores": self.radar_score_map,
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
        print(f">>> Diagnostic figure saved to: {save_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Exports detection parameters as a serializable dictionary."""
        return {
            "subhalo_pos_arcsec": self.subhalo_pos,
            "radar_candidate_arcsec": self.radar_candidate_pos,
            "detection_significance_sigma": self.detection_significance,
            "reduced_chi2_initial": self.reduced_chi2_initial,
            "reduced_chi2_final": self.reduced_chi2_final,
            "physical_mass_solar": self.physical_mass_solar,
            "theta_E_sub_arcsec": self.theta_E_sub,
            "relative_mass_ratio": self.relative_mass_ratio,
            "has_redshifts": self.has_redshifts,
            "z_l": self.z_l,
            "z_s": self.z_s,
            "elapsed_seconds": self.elapsed_time
        }


class SubLensPipeline:
    """
    High-Level End-to-End Gravitational Lensing Inversion & Subhalo Pipeline.

    Parameters:
    -----------
    z_l : Optional[float]
        Host lens galaxy redshift (e.g. 0.222). If None, enters scale-free regime.
    z_s : Optional[float]
        Background source galaxy redshift (e.g. 0.609). If None, enters scale-free regime.
    pixel_scale : float
        Detector pixel scale in arcseconds per pixel (default 0.05).
    psf_fwhm : float
        Instrument Point Spread Function FWHM in arcseconds (default 0.09).
    noise_std : float
        Estimated background noise standard deviation (default 0.012).
    device : str
        Computation backend: 'auto' (selects MPS/CUDA/CPU), 'cuda', 'mps', or 'cpu'.
    """

    def __init__(
        self,
        z_l: Optional[float] = None,
        z_s: Optional[float] = None,
        pixel_scale: float = 0.05,
        psf_fwhm: float = 0.09,
        noise_std: float = 0.012,
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

        self.simulator = LensingSimulator(
            N_img=N_img,
            pixel_scale=pixel_scale,
            N_src=N_src,
            L_src=L_src,
            psf_fwhm=psf_fwhm,
            noise_std=noise_std,
            device=self.device
        )

    def fit(
        self,
        image: Union[torch.Tensor, np.ndarray],
        macro_model: Union[torch.Tensor, Dict[str, Any]],
        subhalo_type: str = "NFW",
        radar_resolution: int = 45,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None,
        scan_range: Optional[float] = None,
        num_macro_steps: int = 120,
        num_pinn_steps: int = 120,
        ground_truth: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> SubLensResult:
        """
        Executes full automated 3-stage inversion:
        1. Smooth macro baseline fit + TV source inversion
        2. Astrometric Radar annular blind scan
        3. PINN continuous autograd refinement
        """
        t0 = time.time()

        # Format input image tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)
        else:
            image_tensor = image.to(self.device)

        if image_tensor.dim() == 2:
            image_tensor = image_tensor.view(1, 1, self.simulator.N_img, self.simulator.N_img)

        # Format macro deflection
        if isinstance(macro_model, dict):
            # Parametric dict: e.g. {"theta_E": 1.43, "q": 0.77, "phi": -1.18, "gamma": 2.16, "gamma_ext": 0.035, "phi_ext": -0.45}
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

        # Adapt source scale and scan range if theta_E is specified
        host_theta_E = 1.40
        if isinstance(macro_model, dict) and "theta_E" in macro_model:
            host_theta_E = macro_model["theta_E"]

        if scan_range is None:
            scan_range = min(1.55 * host_theta_E, 0.85 * self.simulator.L_img)

        # Stage 1: Macro Baseline Fit
        if verbose:
            print(">>> [Pipeline Stage 1] Inverting smooth macro baseline (TV pixelated source)...")
        macro_fitter = MacroFitter(self.simulator, num_steps=num_macro_steps)
        macro_res = macro_fitter.fit(image_tensor, alpha_macro, verbose=False)

        # Stage 2: Astrometric Radar Blind Scan
        if verbose:
            print(">>> [Pipeline Stage 2] Scanning Einstein ring manifold with Astrometric Radar...")
        radar = AstrometricRadar(self.simulator, grid_resolution=radar_resolution, r_min=r_min, r_max=r_max)
        test_mass = 3.5e9 if self.has_redshifts else 0.035
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
                    pinn_res["final_kappa_s"], theta_s=0.15
                )
            except Exception:
                physical_mass = None
        else:
            host_theta_E = macro_model.get("theta_E", 1.40) if isinstance(macro_model, dict) else 1.40
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
            "ssim": ssim_val
        }
        if ground_truth:
            result_dict.update(ground_truth)

        res = SubLensResult(result_dict)
        if verbose:
            res.summary()

        return res


def detect_subhalo(
    image: Union[torch.Tensor, np.ndarray],
    macro_model: Union[torch.Tensor, Dict[str, Any]],
    z_l: Optional[float] = None,
    z_s: Optional[float] = None,
    pixel_scale: float = 0.05,
    psf_fwhm: float = 0.09,
    noise_std: float = 0.012,
    subhalo_type: str = "NFW",
    device: str = "auto",
    verbose: bool = True
) -> SubLensResult:
    """
    Convenience function for one-line subhalo detection.
    """
    pipeline = SubLensPipeline(
        z_l=z_l,
        z_s=z_s,
        pixel_scale=pixel_scale,
        psf_fwhm=psf_fwhm,
        noise_std=noise_std,
        device=device
    )
    return pipeline.fit(image, macro_model, subhalo_type=subhalo_type, verbose=verbose)
