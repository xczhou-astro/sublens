"""
Astrometric and Image Fidelity Evaluation Metrics.
"""

import math
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np


def compute_astrometric_error(
    pred_pos: Tuple[float, float],
    true_pos: Tuple[float, float],
    pixel_scale: float = 0.05
) -> Dict[str, float]:
    """Computes Euclidean astrometric error in arcsec, mas, and pixel units."""
    dx = pred_pos[0] - true_pos[0]
    dy = pred_pos[1] - true_pos[1]
    err_arcsec = math.sqrt(dx**2 + dy**2)
    err_mas = err_arcsec * 1000.0
    err_pix = err_arcsec / pixel_scale
    return {
        "error_arcsec": err_arcsec,
        "error_mas": err_mas,
        "error_pixels": err_pix
    }


def compute_image_metrics(
    recon_img: torch.Tensor,
    true_img: torch.Tensor
) -> Dict[str, float]:
    """Computes PSNR and SSIM between reconstructed and ground truth images."""
    recon_np = recon_img.squeeze().cpu().numpy()
    true_np = true_img.squeeze().cpu().numpy()

    mse = float(np.mean((recon_np - true_np)**2))
    max_val = float(np.max(true_np))
    psnr = 10.0 * math.log10((max_val**2) / (mse + 1e-12))

    mu_x, mu_y = float(np.mean(recon_np)), float(np.mean(true_np))
    sig_x, sig_y = float(np.var(recon_np)), float(np.var(true_np))
    sig_xy = float(np.mean((recon_np - mu_x) * (true_np - mu_y)))
    c1 = (0.01 * max_val)**2
    c2 = (0.03 * max_val)**2
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))

    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "mse": mse
    }
