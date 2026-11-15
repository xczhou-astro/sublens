"""
Stage 2: Astrometric Matched Filter Radar.

Theoretical Foundation:
A localized subhalo perturbation introduces an image-plane brightness disturbance:
delta_I(theta) approx -nabla I_macro(theta) . delta_alpha_sub(theta)

The Astrometric Radar calculates the spatial matched filter response along the
Einstein ring manifold:
S(x_0, y_0) = <R, delta_I> / (||delta_I|| * sigma_noise)

Where R is the smooth macro residual. S evaluates in units of standard deviations (sigma).
Supports single candidate localization and iterative peeling for multi-subhalo systems.
"""

import time
import math
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from ..profiles.subhalo import nfw_subhalo_deflection, sis_subhalo_deflection


class AstrometricRadar:
    """Fast matched filter scanner on annular gravitational lens manifolds."""

    def __init__(
        self,
        simulator,
        grid_resolution: int = 50,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None
    ):
        self.sim = simulator
        self.grid_res = grid_resolution
        self.r_min = r_min
        self.r_max = r_max

    def scan(
        self,
        obs_image: torch.Tensor,
        pred_macro: torch.Tensor,
        res_macro: torch.Tensor,
        source_macro: torch.Tensor,
        alpha_macro: torch.Tensor,
        test_subhalo_mass: float = 1e9,
        subhalo_type: str = "NFW",
        cosmology=None,
        scan_range: Optional[float] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Executes annular blind scan across the Einstein ring.
        """
        device = obs_image.device
        t0 = time.time()

        if scan_range is None:
            scan_range = 0.85 * self.sim.L_img

        scan_coords = torch.linspace(-scan_range, scan_range, self.grid_res, device=device)
        scores = torch.zeros(self.grid_res, self.grid_res, device=device)

        # Determine annular bounds
        r_inner = self.r_min if self.r_min is not None else 0.40 * scan_range
        r_outer = self.r_max if self.r_max is not None else 1.15 * scan_range

        # Subhalo test kernel parameters
        if subhalo_type.upper() == "NFW":
            if cosmology and cosmology.has_redshifts:
                ks, rs = cosmology.physical_mass_to_nfw(test_subhalo_mass)
            else:
                ks, rs = 0.35, 0.15
        else:
            ks, rs = 0.04, 0.04

        with torch.no_grad():
            for j, cy in enumerate(scan_coords):
                for i, cx in enumerate(scan_coords):
                    r_val = torch.sqrt(cx**2 + cy**2)
                    if r_val < r_inner or r_val > r_outer:
                        continue

                    cand_pos = torch.tensor([cx.item(), cy.item()], device=device)
                    if subhalo_type.upper() == "NFW":
                        a_sub, _ = nfw_subhalo_deflection(
                            self.sim.theta_grid, center=cand_pos, kappa_s=torch.tensor(ks, device=device), r_s=rs
                        )
                    else:
                        a_sub, _ = sis_subhalo_deflection(
                            self.sim.theta_grid, center=cand_pos, theta_E=torch.tensor(ks, device=device)
                        )

                    beta_cand = self.sim.theta_grid - alpha_macro - a_sub
                    u_c = beta_cand[:, 0].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
                    v_c = beta_cand[:, 1].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
                    grid_c = torch.stack([u_c, v_c], dim=-1)

                    I_cand = self.sim.convolve_psf(
                        F.grid_sample(source_macro, grid_c, mode='bilinear', padding_mode='zeros', align_corners=True)
                    )
                    d_I = I_cand - pred_macro
                    # Statistically optimal matched filter with pixel-wise noise map
                    d_I_normed = d_I / self.sim.noise_map
                    norm = torch.sqrt((d_I_normed**2).sum()) + 1e-8
                    scores[j, i] = ((res_macro / self.sim.noise_map) * d_I_normed).sum() / norm

        elapsed = time.time() - t0
        scores_np = scores.cpu().numpy()
        max_idx = np.unravel_index(np.argmax(scores_np), (self.grid_res, self.grid_res))
        best_cand_x = float(scan_coords[max_idx[1]].cpu())
        best_cand_y = float(scan_coords[max_idx[0]].cpu())
        max_score = float(scores_np[max_idx])

        if verbose:
            print(f"[AstrometricRadar] Scan complete in {elapsed:.2f} s. Peak S = {max_score:.2f} sigma at ({best_cand_x:+.3f}, {best_cand_y:+.3f})")

        return {
            "score_map": scores,
            "scan_coords": scan_coords,
            "best_candidate": (best_cand_x, best_cand_y),
            "max_score": max_score,
            "elapsed_time": elapsed,
            "extent": [-scan_range, scan_range, -scan_range, scan_range]
        }
