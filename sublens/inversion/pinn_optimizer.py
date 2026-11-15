"""
Stage 3: Continuous Autodiff PINN Joint Refiner.

Refines subhalo parameters into continuous space:
- Follows exact automatic differentiation gradients.
- Jointly optimizes:
  * Subhalo position (x_sub, y_sub) in continuous coordinates.
  * Subhalo mass parameter kappa_s / theta_E.
  * High-dimensional pixelated source plane S(u, v).
- Drives reduced chi^2 down to the theoretical noise limit.
"""

import time
import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..profiles.subhalo import nfw_subhalo_deflection, sis_subhalo_deflection


class PINNOptimizer:
    """Continuous gradient-flow optimizer for subhalo parameter refinement."""

    def __init__(
        self,
        simulator,
        lr_pos: float = 0.008,
        lr_mass: float = 0.015,
        lr_src: float = 0.04,
        tv_weight: float = 3e-4,
        num_steps: int = 140
    ):
        self.sim = simulator
        self.lr_pos = lr_pos
        self.lr_mass = lr_mass
        self.lr_src = lr_src
        self.tv_weight = tv_weight
        self.num_steps = num_steps

    def refine(
        self,
        obs_image: torch.Tensor,
        init_pos: Tuple[float, float],
        init_src_param: torch.Tensor,
        alpha_macro: torch.Tensor,
        vignette_mask: torch.Tensor,
        subhalo_type: str = "NFW",
        r_s: float = 0.15,
        init_kappa: float = 0.35,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Executes continuous joint autograd optimization.
        """
        device = obs_image.device
        t0 = time.time()

        raw_k_val = math.log(max(math.exp(init_kappa) - 1.0, 1e-5))
        sub_pos = nn.Parameter(torch.tensor([init_pos[0], init_pos[1]], dtype=torch.float32, device=device))
        sub_k = nn.Parameter(torch.tensor([raw_k_val], dtype=torch.float32, device=device))
        src_param = nn.Parameter(init_src_param.clone())

        optimizer = torch.optim.Adam([
            {'params': [sub_pos], 'lr': self.lr_pos},
            {'params': [sub_k], 'lr': self.lr_mass},
            {'params': [src_param], 'lr': self.lr_src}
        ])

        for step in range(self.num_steps):
            optimizer.zero_grad()
            eff_k = F.softplus(sub_k)

            if subhalo_type.upper() == "NFW":
                a_sub, _ = nfw_subhalo_deflection(self.sim.theta_grid, center=sub_pos, kappa_s=eff_k, r_s=r_s)
            else:
                a_sub, _ = sis_subhalo_deflection(self.sim.theta_grid, center=sub_pos, theta_E=eff_k)

            alpha_total = alpha_macro + a_sub
            beta_pred = self.sim.theta_grid - alpha_total

            u_p = beta_pred[:, 0].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
            v_p = beta_pred[:, 1].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
            grid_p = torch.stack([u_p, v_p], dim=-1)

            pos_src = F.softplus(src_param) * vignette_mask
            img_clean = F.grid_sample(pos_src, grid_p, mode='bilinear', padding_mode='zeros', align_corners=True)
            img_conv = self.sim.convolve_psf(img_clean)

            diff_y = torch.abs(pos_src[:, :, 1:, :] - pos_src[:, :, :-1, :])
            diff_x = torch.abs(pos_src[:, :, :, 1:] - pos_src[:, :, :, :-1])
            tv_loss = torch.mean(diff_y) + torch.mean(diff_x)

            # Pixel-wise chi^2 weighted loss using exact noise_map (or uniform noise_std)
            norm_res = (img_conv - obs_image) / self.sim.noise_map
            loss = torch.mean(norm_res**2) + self.tv_weight * tv_loss
            loss.backward()
            optimizer.step()

        elapsed = time.time() - t0
        refined_pos = (float(sub_pos[0].item()), float(sub_pos[1].item()))
        final_kappa = float(F.softplus(sub_k).item())
        final_src = (F.softplus(src_param) * vignette_mask).detach()

        # Final predictions
        if subhalo_type.upper() == "NFW":
            a_sub_final, _ = nfw_subhalo_deflection(self.sim.theta_grid, center=sub_pos.detach(), kappa_s=F.softplus(sub_k).detach(), r_s=r_s)
        else:
            a_sub_final, _ = sis_subhalo_deflection(self.sim.theta_grid, center=sub_pos.detach(), theta_E=F.softplus(sub_k).detach())

        beta_final = self.sim.theta_grid - (alpha_macro + a_sub_final)
        u_f = beta_final[:, 0].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
        v_f = beta_final[:, 1].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
        grid_f = torch.stack([u_f, v_f], dim=-1)

        pred_final = self.sim.convolve_psf(
            F.grid_sample(final_src, grid_f, mode='bilinear', padding_mode='zeros', align_corners=True)
        ).detach()
        res_final = obs_image - pred_final
        chi2_final = torch.mean(((pred_final - obs_image) / self.sim.noise_map)**2).item()

        if verbose:
            print(f"[PINNOptimizer] Refinement complete in {elapsed:.2f} s.")
            print(f"  Converged position: ({refined_pos[0]:+.4f}\", {refined_pos[1]:+.4f}\") | chi^2_red = {chi2_final:.2f}")

        return {
            "refined_pos": refined_pos,
            "final_kappa_s": final_kappa,
            "final_source": final_src,
            "pred_final": pred_final,
            "res_final": res_final,
            "chi2_final": chi2_final,
            "elapsed_time": elapsed
        }
