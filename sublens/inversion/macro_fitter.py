"""
Stage 1: Smooth Macro Lens Baseline Fitter with TV Pixelated Source Inversion.

Under the null hypothesis of a smooth macro lens (no dark subhalos):
- Optimizes pixelated source brightness grid S(u, v) using TV (Total Variation) regularization.
- Computes best-fit image model and exposes localized astrometric defects in the residual.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroFitter:
    """Performs smooth macro baseline inversion and residual defect mapping."""

    def __init__(
        self,
        simulator,
        tv_weight: float = 4e-4,
        lr: float = 0.08,
        num_steps: int = 120
    ):
        self.sim = simulator
        self.tv_weight = tv_weight
        self.lr = lr
        self.num_steps = num_steps

    def fit(
        self,
        obs_image: torch.Tensor,
        alpha_macro: torch.Tensor,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Fits the pixelated source plane given a macro deflection field alpha_macro.
        """
        device = obs_image.device
        beta_macro = self.sim.theta_grid - alpha_macro

        u_m = beta_macro[:, 0].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
        v_m = beta_macro[:, 1].view(1, self.sim.N_img, self.sim.N_img) / self.sim.L_src
        grid_macro = torch.stack([u_m, v_m], dim=-1)

        # Source parameterization (unconstrained log-space with softplus)
        src_param = nn.Parameter(torch.zeros(1, 1, self.sim.N_src, self.sim.N_src, device=device))
        opt = torch.optim.Adam([src_param], lr=self.lr)

        # Vignette mask to avoid border artifacts
        sy, sx = torch.meshgrid(self.sim.src_coords_1d, self.sim.src_coords_1d, indexing='ij')
        vignette = torch.exp(-0.5 * (sx**2 + sy**2) / ((0.75 * self.sim.L_src)**2)).view(1, 1, self.sim.N_src, self.sim.N_src).to(device)

        for step in range(self.num_steps):
            opt.zero_grad()
            pos_src = F.softplus(src_param) * vignette
            pred_clean = F.grid_sample(pos_src, grid_macro, mode='bilinear', padding_mode='zeros', align_corners=True)
            pred_conv = self.sim.convolve_psf(pred_clean)

            # MSE loss + TV regularization
            diff_y = torch.abs(pos_src[:, :, 1:, :] - pos_src[:, :, :-1, :])
            diff_x = torch.abs(pos_src[:, :, :, 1:] - pos_src[:, :, :, :-1])
            tv_loss = torch.mean(diff_y) + torch.mean(diff_x)

            loss = F.mse_loss(pred_conv, obs_image) + self.tv_weight * tv_loss
            loss.backward()
            opt.step()

        final_src = (F.softplus(src_param) * vignette).detach()
        pred_clean = F.grid_sample(final_src, grid_macro, mode='bilinear', padding_mode='zeros', align_corners=True)
        pred_macro = self.sim.convolve_psf(pred_clean).detach()
        res_macro = obs_image - pred_macro

        chi2_red = F.mse_loss(pred_macro, obs_image).item() / (self.sim.noise_std**2)

        if verbose:
            print(f"[MacroFitter] Baseline fit finished. Reduced chi^2 = {chi2_red:.2f}")

        return {
            "source_reconstructed": final_src,
            "source_param": src_param.detach(),
            "pred_macro": pred_macro,
            "res_macro": res_macro,
            "chi2_red": chi2_red,
            "grid_macro": grid_macro,
            "vignette": vignette
        }
