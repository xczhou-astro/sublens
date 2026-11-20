"""
Euclid Q1 Strong Gravitational Lens Data Ingestion & Survey Pipeline.

Provides high-level facilities to:
1. Extract authentic Euclid Q1 Early Release Observations (ERO / Q1 Discovery Engine).
2. Prepare standardized, publication-ready multi-extension FITS files containing:
   - SCI: Foreground-subtracted lensed arc image
   - PSF: Empirical Euclid VIS point spread function
   - RMS: Exact pixel-level noise uncertainty map
   - MASK: Annular and contaminant-free fit mask
   - Full WCS, astrometric, and macro-lens header metadata.
3. Batch-execute SubLensPipeline across Grade A confirmed lens systems.
4. Compile structured survey catalogs (CSV / FITS tables) and 8-panel diagnostic figures.
"""

import os
import json
import csv
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table

from ..pipeline import SubLensPipeline, SubLensResult
from ..profiles.subhalo import truncated_subhalo_deflection as _halo_deflection


def centered_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Extracts a centered square crop of given size from a 2D image."""
    if image.ndim != 2 or min(image.shape) < size:
        raise ValueError(f"Cannot take a {size}-pixel crop from shape {image.shape}")
    y = (image.shape[0] - size) // 2
    x = (image.shape[1] - size) // 2
    return np.array(image[y:y + size, x:x + size], dtype=np.float32, copy=True)


def get_available_grade_a_lenses(
    catalog_path: Union[str, Path] = "archive/data/q1_discovery_engine_lens_catalog.csv",
    data_dir: Union[str, Path] = "archive/data",
    min_expert_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Finds all Grade A lenses in the Euclid catalog that have complete data files ready for inversion.
    Returns list of metadata dictionaries sorted descending by expert_score.
    """
    catalog_path = Path(catalog_path)
    data_dir = Path(data_dir)
    lens_dir = data_dir / "lens"

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    valid_dirs = set(
        d.name for d in lens_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.fits").exists() and (d / "result_lens_mass.json").exists()
    )

    ready_lenses = []
    with open(catalog_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("grade", "").strip().upper() == "A" and row["id_str"] in valid_dirs:
                score = float(row.get("expert_score", 0.0))
                if score >= min_expert_score:
                    entry = {
                        "id_str": row["id_str"],
                        "object_id": row.get("object_id", ""),
                        "ra": float(row["right_ascension"]),
                        "dec": float(row["declination"]),
                        "expert_score": score,
                        "grade": "A",
                        "dir_path": str(lens_dir / row["id_str"])
                    }
                    ready_lenses.append(entry)

    ready_lenses.sort(key=lambda x: x["expert_score"], reverse=True)
    return ready_lenses


def prepare_euclid_grade_a_fits(
    lens_id: str,
    data_dir: Union[str, Path] = "archive/data",
    output_path: Optional[Union[str, Path]] = None,
    crop_size: int = 100,
    overwrite: bool = True
) -> str:
    """
    Builds a standard, self-contained multi-extension FITS file for a Euclid Q1 Grade A lens.
    
    Structure:
      HDU 0 ('SCI'): Foreground-subtracted lensed arc image (float32)
      HDU 1 ('PSF'): Normalized Euclid VIS PSF kernel (float32)
      HDU 2 ('RMS'): Pixel-wise noise uncertainty map (float32)
      HDU 3 ('MASK'): Binary annular fit mask excluding contaminants (uint8)
    """
    data_dir = Path(data_dir)
    lens_dir = data_dir / "lens" / lens_id
    raw_fits_path = lens_dir / f"{lens_id}.fits"
    mass_json_path = lens_dir / "result_lens_mass.json"

    if not raw_fits_path.exists():
        raise FileNotFoundError(f"Raw Euclid FITS not found: {raw_fits_path}")
    if not mass_json_path.exists():
        raise FileNotFoundError(f"Macro lens parameters JSON not found: {mass_json_path}")

    # 1. Parse raw data
    with fits.open(raw_fits_path) as hdul:
        vis = hdul["VIS_FLUX"]
        raw_img = vis.data
        raw_noise = hdul["VIS_RMS"].data
        raw_psf = hdul["VIS_PSF"].data.astype(np.float32)
        pixel_scale = abs(float(vis.header.get("CD2_2", 0.10 / 3600.0))) * 3600.0
        ra_val = float(vis.header.get("CRVAL1", 0.0))
        dec_val = float(vis.header.get("CRVAL2", 0.0))

    # Look up catalog RA, DEC, expert_score if in catalog
    cat_path = data_dir / "q1_discovery_engine_lens_catalog.csv"
    expert_score = None
    if cat_path.exists():
        with open(cat_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id_str") == lens_id:
                    ra_val = float(row.get("right_ascension", ra_val))
                    dec_val = float(row.get("declination", dec_val))
                    expert_score = float(row.get("expert_score", 0.0))
                    break

    img_cropped = centered_crop(raw_img, crop_size)
    noise_cropped = centered_crop(raw_noise, crop_size)
    psf_norm = (raw_psf / raw_psf.sum()).astype(np.float32)

    # 2. Subtract foreground lens light model
    light_path = lens_dir / "result" / "lens_light.fits"
    if not light_path.exists():
        light_path = lens_dir / "result" / "sersic_lens_light.fits"
    if light_path.exists():
        lens_light = centered_crop(fits.getdata(light_path), crop_size)
        arc_image = img_cropped - lens_light
    else:
        arc_image = img_cropped

    # 3. Contaminant masking
    valid = np.isfinite(arc_image) & np.isfinite(noise_cropped) & (noise_cropped > 0)
    extra_mask_path = lens_dir / "mask_extra_galaxies.fits"
    if extra_mask_path.exists():
        extra_mask = centered_crop(fits.getdata(extra_mask_path), crop_size)
        valid &= (extra_mask == 0)

    # Clean invalid/masked pixels
    arc_image = np.where(valid, arc_image, 0.0)
    noise_cropped = np.where(valid, noise_cropped, 1.0)

    # 4. Parse macro model from JSON
    mass_values = json.loads(mass_json_path.read_text())
    theta_e = mass_values.get("einstein_radius_max_lh")
    if theta_e is None or theta_e <= 0:
        theta_e = mass_values["einstein_radius_effective_max_lh"]
    theta_e = float(theta_e)

    cx = float(mass_values["centre_1_max_lh"])
    cy = float(mass_values["centre_0_max_lh"])
    ell_x = float(mass_values["ell_comps_1_max_lh"])
    ell_y = float(mass_values["ell_comps_0_max_lh"])
    ell = math.hypot(ell_x, ell_y)
    q = (1.0 - ell) / (1.0 + ell) if ell < 1.0 else 0.80
    phi = 0.5 * math.atan2(ell_y, ell_x)

    shear_1 = float(mass_values.get("shear_gamma_1_max_lh", 0.0))
    shear_2 = float(mass_values.get("shear_gamma_2_max_lh", 0.0))
    shear_g = math.hypot(shear_1, shear_2)
    shear_pa = 0.5 * math.atan2(shear_2, shear_1)

    # 5. Annular fit mask
    axis = (np.arange(crop_size, dtype=np.float32) - (crop_size - 1) / 2.0) * pixel_scale
    x_grid, y_grid = np.meshgrid(axis, -axis)
    radius = np.hypot(x_grid - cx, y_grid - cy)

    inner = max(0.25, theta_e - 0.45)
    outer = theta_e + 0.50
    fit_mask = (radius >= inner) & (radius <= outer) & valid

    # Pad edges for PSF convolution footprint
    pad_y = raw_psf.shape[0] // 2
    pad_x = raw_psf.shape[1] // 2
    edge = np.zeros_like(fit_mask, dtype=bool)
    edge[pad_y:crop_size - pad_y, pad_x:crop_size - pad_x] = True
    fit_mask &= edge

    # 6. Build Standard Astronomy FITS
    hdr = fits.Header()
    hdr["TELESCOP"] = ("Euclid", "Space Observatory")
    hdr["INSTRUME"] = ("VIS", "Visible Instrument (550-900nm)")
    hdr["PIXSCALE"] = (float(pixel_scale), "Pixel scale in arcsec/pixel")
    hdr["OBJECT"]   = (lens_id, "Euclid Q1 Lens Catalog Identifier")
    hdr["RA"]       = (ra_val, "Right Ascension (deg, J2000)")
    hdr["DEC"]      = (dec_val, "Declination (deg, J2000)")
    hdr["GRADE"]    = ("A", "Discovery Engine Inspection Grade")
    if expert_score is not None:
        hdr["EXPSCORE"] = (float(expert_score), "Discovery Engine Expert Score")
    hdr["CROPSIZE"] = (crop_size, "Square crop dimension")

    # Macro parameters
    hdr["MACRO_TE"] = (theta_e, "Host Einstein radius theta_E (arcsec)")
    hdr["MACRO_XC"] = (cx, "Host lens x-center (arcsec)")
    hdr["MACRO_YC"] = (cy, "Host lens y-center (arcsec)")
    hdr["MACRO_Q"]  = (q, "Host axis ratio q = b/a")
    hdr["MACRO_PA"] = (phi, "Host position angle (rad)")
    hdr["SHEAR_G"]  = (shear_g, "External shear magnitude gamma_ext")
    hdr["SHEAR_PA"] = (shear_pa, "External shear angle phi_ext (rad)")
    hdr["SHEAR_1"]  = (shear_1, "Catalog external shear gamma_1")
    hdr["SHEAR_2"]  = (shear_2, "Catalog external shear gamma_2")
    hdr["ELL_X"]    = (ell_x, "Catalog ellipticity component e_1")
    hdr["ELL_Y"]    = (ell_y, "Catalog ellipticity component e_2")

    hdu_sci = fits.PrimaryHDU(data=arc_image.astype(np.float32), header=hdr)
    hdu_sci.header["EXTNAME"] = "SCI"

    hdu_psf = fits.ImageHDU(data=psf_norm, name="PSF")
    hdu_psf.header["COMMENT"] = "Normalized Euclid VIS empirical PSF"

    hdu_rms = fits.ImageHDU(data=noise_cropped.astype(np.float32), name="RMS")
    hdu_rms.header["COMMENT"] = "Pixel-level RMS noise map"

    hdu_mask = fits.ImageHDU(data=fit_mask.astype(np.uint8), name="MASK")
    hdu_mask.header["COMMENT"] = "Annular science fit mask (1=fit, 0=masked)"

    hdul_out = fits.HDUList([hdu_sci, hdu_psf, hdu_rms, hdu_mask])

    if output_path is None:
        out_dir = Path("euclid_q1_fits")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{lens_id}_standard.fits"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    hdul_out.writeto(str(output_path), overwrite=overwrite)
    return str(output_path)


class SersicSource(torch.nn.Module):
    """Circular, fixed-index Sérsic model for source initialization."""

    def __init__(self, peak: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.log_peak = torch.nn.Parameter(peak.log())
        self.log_radius = torch.nn.Parameter(peak.new_tensor(math.log(0.15)))
        self.x = torch.nn.Parameter(x)
        self.y = torch.nn.Parameter(y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt((x - self.x).square() + (y - self.y).square() + 1e-8)
        n = 1.2
        return self.log_peak.exp() * torch.exp(
            -(2 * n - 0.324) * (radius / self.log_radius.exp()).pow(1 / n)
        )


def _elliptical_sie(
    x: torch.Tensor,
    y: torch.Tensor,
    q: float,
    cos_phi: float,
    sin_phi: float,
    f: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyAutoLens / Kormann 1994 exact analytic SIE deflection."""
    xr = cos_phi * x + sin_phi * y
    yr = -sin_phi * x + cos_phi * y
    radius = ((q * xr).square() + yr.square()).clamp_min(1e-12).sqrt()
    scale = 2.0 * q / ((1.0 + q) * f)
    ax = scale * torch.atan(f * xr / radius)
    ay = scale * torch.atanh((f * yr / radius).clamp(-0.999999, 0.999999))
    return cos_phi * ax - sin_phi * ay, sin_phi * ax + cos_phi * ay


def _render_source(
    source: torch.Tensor,
    beta_x: torch.Tensor,
    beta_y: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    fov: float,
    psf_kernel: torch.Tensor
) -> torch.Tensor:
    """Renders the pixelated source galaxy to the lens plane with PSF blurring."""
    ray_grid = torch.stack((beta_x - cx, cy - beta_y), dim=-1) * (2.0 / fov)
    arc = torch.nn.functional.grid_sample(
        source[None, None], ray_grid[None], align_corners=True, padding_mode="zeros"
    )[0, 0]
    return torch.nn.functional.conv2d(arc[None, None], psf_kernel, padding="same")[0, 0]


def fit_real_lens_observation(
    pipeline: SubLensPipeline,
    image_arr: np.ndarray,
    psf_arr: np.ndarray,
    noise_arr: np.ndarray,
    fit_mask: np.ndarray,
    macro_model: Dict[str, Any],
    meta: Dict[str, Any],
    radar_resolution: int = 45,
    num_macro_steps: int = 90,
    num_pinn_steps: int = 60,
    source_fov: Optional[Union[float, str]] = "auto",
    source_resolution: Optional[int] = None,
    fix_macro: bool = True,
    verbose: bool = True
) -> SubLensResult:
    """
    Executes the standard 3-stage inversion specifically calibrated for real space telescope lenses:
    1. Sersic warm-start + Total-Variation pixelated source inversion on the masked annulus
    2. Astrometric Radar matched-filter blind scan across the Einstein ring
    3. PINN continuous gradient refinement for candidate localization and significance scoring
    """
    t0 = time.time()
    torch.manual_seed(42)
    device = torch.device(pipeline.device)

    # 1. Coordinate grids
    crop_size = image_arr.shape[0]
    pix_scale = float(pipeline.pixel_scale)
    axis = (torch.arange(crop_size, device=device, dtype=torch.float32) - (crop_size - 1) / 2.0) * pix_scale
    obs_y, obs_x = torch.meshgrid(-axis, axis, indexing="ij")

    # Tensors
    if isinstance(image_arr, torch.Tensor):
        obs_img = image_arr.clone().detach().to(device=device, dtype=torch.float32).squeeze()
    else:
        obs_img = torch.tensor(image_arr, dtype=torch.float32, device=device).squeeze()

    if isinstance(noise_arr, torch.Tensor):
        noise_map = noise_arr.clone().detach().to(device=device, dtype=torch.float32).squeeze().clamp_min(1e-5)
    else:
        noise_map = torch.tensor(noise_arr, dtype=torch.float32, device=device).squeeze().clamp_min(1e-5)

    if isinstance(psf_arr, torch.Tensor):
        psf_norm = psf_arr.clone().detach().to(device=device, dtype=torch.float32).squeeze()
        psf_norm = psf_norm / psf_norm.sum()
        psf_k = torch.flip(psf_norm, dims=[0, 1])[None, None]
    else:
        psf_np = psf_arr.squeeze()
        psf_k = torch.tensor((psf_np[::-1, ::-1] / psf_np.sum()).copy(), dtype=torch.float32, device=device)[None, None]

    if isinstance(fit_mask, torch.Tensor):
        mask_t = fit_mask.clone().detach().to(device=device, dtype=torch.bool).squeeze()
    else:
        mask_t = torch.tensor(fit_mask, dtype=torch.bool, device=device).squeeze()

    # 2. Macro deflection
    theta_e_val = float(macro_model.get("theta_E", 1.0))
    mcx = float(macro_model.get("center", (0.0, 0.0))[0])
    mcy = float(macro_model.get("center", (0.0, 0.0))[1])
    ell_x = float(macro_model.get("ell_x", 0.0))
    ell_y = float(macro_model.get("ell_y", 0.0))
    shear_1 = float(macro_model.get("shear_1", 0.0))
    shear_2 = float(macro_model.get("shear_2", 0.0))

    ell = math.hypot(ell_x, ell_y)
    q = max(0.05, (1.0 - ell) / (1.0 + ell)) if ell < 1.0 else 0.80
    phi = 0.5 * math.atan2(ell_y, ell_x)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    f = math.sqrt(max(1e-6, 1.0 - q**2))

    x_rel = obs_x - mcx
    y_rel = obs_y - mcy
    ax_sie, ay_sie = _elliptical_sie(x_rel, y_rel, q, cos_phi, sin_phi, f)
    ax_shear = shear_1 * x_rel + shear_2 * y_rel
    ay_shear = shear_2 * x_rel - shear_1 * y_rel

    beta_x_init = obs_x - theta_e_val * ax_sie - ax_shear
    beta_y_init = obs_y - theta_e_val * ay_sie - ay_shear

    # 3. Source warm start & baseline inversion
    if verbose:
        print(f">>> [SubLens Survey] Processing {meta.get('OBJECT', 'Euclid Lens')} (theta_E = {theta_e_val:.3f}\")...")
        print("  Stage 1: Inverting smooth macro baseline (Sersic + TV pixelated source)...")

    weights = obs_img[mask_t].clamp_min(0.0)
    w_sum = weights.sum().clamp_min(1e-8)
    flux_unit = torch.maximum(torch.quantile(weights, 0.98), noise_map[mask_t].median())

    cx_init = (beta_x_init[mask_t] * weights).sum() / w_sum
    cy_init = (beta_y_init[mask_t] * weights).sum() / w_sum

    sersic = SersicSource(flux_unit.new_tensor(2.0), cx_init, cy_init)
    if fix_macro:
        opt_sersic = torch.optim.Adam(sersic.parameters(), lr=0.025)
        for _ in range(60):
            opt_sersic.zero_grad()
            arc = torch.nn.functional.conv2d((flux_unit * sersic(beta_x_init, beta_y_init))[None, None], psf_k, padding="same")[0, 0]
            loss = ((obs_img - arc) / noise_map)[mask_t].square().mean()
            loss.backward()
            opt_sersic.step()
        final_te = torch.tensor(theta_e_val, device=device)
        bx = beta_x_init
        by = beta_y_init
    else:
        log_theta_scale = torch.nn.Parameter(torch.zeros((), device=device))
        opt_sersic = torch.optim.Adam([
            {"params": sersic.parameters(), "lr": 0.025},
            {"params": [log_theta_scale], "lr": 0.004}
        ])
        for _ in range(60):
            opt_sersic.zero_grad()
            cur_te = theta_e_val * log_theta_scale.exp()
            b_x = obs_x - cur_te * ax_sie - ax_shear
            b_y = obs_y - cur_te * ay_sie - ay_shear
            arc = torch.nn.functional.conv2d((flux_unit * sersic(b_x, b_y))[None, None], psf_k, padding="same")[0, 0]
            loss = ((obs_img - arc) / noise_map)[mask_t].square().mean() + 10.0 * log_theta_scale.square()
            loss.backward()
            opt_sersic.step()
        final_te = (theta_e_val * log_theta_scale.exp()).detach()
        bx = (obs_x - final_te * ax_sie - ax_shear).detach()
        by = (obs_y - final_te * ay_sie - ay_shear).detach()

    with torch.no_grad():
        cx = sersic.x.detach()
        cy = sersic.y.detach()
        r_eff = float(sersic.log_radius.exp().item())

        if source_fov is None or source_fov == "auto":
            arc_flux = obs_img[mask_t]
            f_thresh = max(float(noise_map[mask_t].median().item()) * 2.5, float(arc_flux.max().item()) * 0.15)
            sig_mask = mask_t & (obs_img > f_thresh)
            if sig_mask.sum() > 15:
                dx_sig = (bx[sig_mask] - cx).abs()
                dy_sig = (by[sig_mask] - cy).abs()
                q_span = max(float(torch.quantile(dx_sig, 0.90).item()), float(torch.quantile(dy_sig, 0.90).item()))
            else:
                q_span = 0.30

            ray_span = max(0.5, q_span * 2.2)
            sersic_span = max(0.6, r_eff * 5.5)
            effective_fov = min(1.6, max(0.65, max(ray_span, sersic_span)))
        else:
            effective_fov = float(source_fov)

        if source_resolution is None:
            source_res = int(round(effective_fov / 0.010))
            source_res = max(64, min(96, source_res))
            if source_res % 2 != 0:
                source_res += 1
        else:
            source_res = int(source_resolution)

        axis_src = torch.linspace(-effective_fov / 2.0, effective_fov / 2.0, source_res, device=device)
        sy, sx = torch.meshgrid(-axis_src + cy, axis_src + cx, indexing="ij")
        init_src = sersic(sx, sy).detach().clamp_min(1e-4)

        # Spatial coherence: Gaussian filter kernel enforcing smooth galaxy morphology
        sigma_pix = 2.0
        ksize = int(2 * round(3 * sigma_pix) + 1)
        kx = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
        g1d = torch.exp(-0.5 * (kx / sigma_pix)**2)
        g2d = (g1d[:, None] * g1d[None, :])
        g2d = (g2d / g2d.sum())[None, None]

        raw_src = torch.nn.Parameter(torch.log(torch.exp(init_src) - 1.0 + 1e-6))

    opt_src = torch.optim.Adam([raw_src], lr=0.04)
    r_border = torch.sqrt((sx - cx)**2 + (sy - cy)**2) / (effective_fov / 2.0)
    border_mask = (r_border > 0.85).float()

    for _ in range(num_macro_steps):
        opt_src.zero_grad()
        blurred = torch.nn.functional.conv2d(raw_src[None, None], g2d, padding=ksize//2)[0, 0]
        src_pix = torch.nn.functional.softplus(blurred)
        arc = _render_source(flux_unit * src_pix, bx, by, cx, cy, effective_fov, psf_k)
        res = (obs_img - arc) / noise_map
        data_loss = res[mask_t].square().mean()

        # Edge damping: penalize unconstrained flux drifting at the grid boundary
        border_penalty = (src_pix * border_mask).square().mean() * 5.0

        # Curvature regularization: penalize second spatial derivatives
        d2x = src_pix[:, 2:] - 2 * src_pix[:, 1:-1] + src_pix[:, :-2]
        d2y = src_pix[2:, :] - 2 * src_pix[1:-1, :] + src_pix[:-2, :]
        curv_loss = (d2x.square().mean() + d2y.square().mean()) * 0.5

        loss = data_loss + border_penalty + curv_loss
        loss.backward()
        opt_src.step()

    with torch.no_grad():
        blurred = torch.nn.functional.conv2d(raw_src[None, None], g2d, padding=ksize//2)[0, 0]
        final_src = (flux_unit * torch.nn.functional.softplus(blurred)).detach()
        I_base = _render_source(final_src, bx, by, cx, cy, effective_fov, psf_k).detach()
        res_macro = (obs_img - I_base).detach()
        norm_res_macro = (res_macro / noise_map).detach()
        chi2_base = norm_res_macro[mask_t].square().mean().item()

    # 4. Astrometric Radar blind scan
    if verbose:
        print(f"  Stage 2: Scanning Einstein ring with Astrometric Radar (chi2_base = {chi2_base:.3f})...")

    scan_coords_1d = torch.linspace(-1.3 * final_te.item(), 1.3 * final_te.item(), radar_resolution, device=device)
    scores = torch.zeros((radar_resolution, radar_resolution), device=device)
    delta_chi2_map = torch.zeros((radar_resolution, radar_resolution), device=device)

    with torch.no_grad():
        for j, ty in enumerate(scan_coords_1d):
            for i, tx in enumerate(scan_coords_1d):
                r_c = math.hypot(tx.item() - mcx, ty.item() - mcy)
                if r_c < 0.45 * final_te.item() or r_c > 1.35 * final_te.item():
                    continue

                dax, day = _halo_deflection(obs_x, obs_y, tx.item(), ty.item(), strength=torch.tensor(0.015, device=device))
                bx_pert = bx - dax
                by_pert = by - day
                I_pert = _render_source(final_src, bx_pert, by_pert, cx, cy, effective_fov, psf_k)

                d_I = (I_pert - I_base) / noise_map
                d_I_masked = torch.where(mask_t, d_I, torch.zeros_like(d_I))
                norm = torch.sqrt(torch.sum(d_I_masked**2)) + 1e-8
                res_masked = torch.where(mask_t, norm_res_macro, torch.zeros_like(norm_res_macro))

                score = torch.sum(res_masked * d_I_masked) / norm
                scores[j, i] = score

                res_pert = (obs_img - I_pert) / noise_map
                dchi2 = (norm_res_macro[mask_t].square().sum() - res_pert[mask_t].square().sum()).item()
                delta_chi2_map[j, i] = dchi2

    idx_max = np.unravel_index(np.argmax(scores.cpu().numpy()), (radar_resolution, radar_resolution))
    cand_x = float(scan_coords_1d[idx_max[1]].cpu())
    cand_y = float(scan_coords_1d[idx_max[0]].cpu())
    max_score = float(scores[idx_max].cpu())

    # 5. Continuous autograd refinement (PINN)
    if verbose:
        print(f"  Stage 3: Refining subhalo from seed ({cand_x:+.3f}\", {cand_y:+.3f}\") [Radar S = {max_score:.2f} sigma]...")

    halo_pos = torch.nn.Parameter(torch.tensor([cand_x, cand_y], dtype=torch.float32, device=device))
    log_strength = torch.nn.Parameter(torch.tensor(math.log(0.015), dtype=torch.float32, device=device))
    opt_pinn = torch.optim.Adam([
        {"params": [halo_pos], "lr": 0.005},
        {"params": [log_strength], "lr": 0.02}
    ])

    for _ in range(num_pinn_steps):
        opt_pinn.zero_grad()
        str_val = log_strength.exp()
        dax, day = _halo_deflection(obs_x, obs_y, halo_pos[0], halo_pos[1], strength=str_val)
        bx_pert = bx - dax
        by_pert = by - day
        I_pert = _render_source(final_src, bx_pert, by_pert, cx, cy, effective_fov, psf_k)
        loss = ((obs_img - I_pert) / noise_map)[mask_t].square().mean()
        loss.backward()
        opt_pinn.step()

    with torch.no_grad():
        final_pos = (float(halo_pos[0].cpu().item()), float(halo_pos[1].cpu().item()))
        final_str = float(log_strength.exp().cpu().item())
        dax_f, day_f = _halo_deflection(obs_x, obs_y, halo_pos[0], halo_pos[1], strength=log_strength.exp())
        I_final = _render_source(final_src, bx - dax_f, by - day_f, cx, cy, effective_fov, psf_k).detach()
        res_final = (obs_img - I_final).detach()
        chi2_final = ((res_final / noise_map)[mask_t].square().mean()).cpu().item()
        delta_chi2_total = (chi2_base - chi2_final) * mask_t.sum().item()

    elapsed = time.time() - t0
    detection_flag = (max_score >= 4.0 and delta_chi2_total >= 10.0)

    if verbose:
        status_str = "CONFIRMED SUBHALO DETECTION" if detection_flag else "NO SUBHALO (SMOOTH LENS / UPPER LIMIT)"
        print(f"  >>> Inversion Complete in {elapsed:.2f}s | chi2: {chi2_base:.3f} -> {chi2_final:.3f} (Delta chi2 = {delta_chi2_total:.2f}) | {status_str}")

    cx_val = float(cx.cpu().item())
    cy_val = float(cy.cpu().item())
    src_extent = [
        cx_val - effective_fov / 2.0,
        cx_val + effective_fov / 2.0,
        cy_val - effective_fov / 2.0,
        cy_val + effective_fov / 2.0
    ]

    # Synchronize pipeline simulator geometry if present
    if hasattr(pipeline, "simulator") and pipeline.simulator is not None:
        pipeline.simulator.L_src = effective_fov
        pipeline.simulator.N_src = source_res

    # Build SubLensResult container
    data = {
        "refined_pos": final_pos,
        "radar_candidate": (cand_x, cand_y),
        "radar_score": max_score,
        "chi2_macro": chi2_base,
        "chi2_final": chi2_final,
        "delta_chi2_total": delta_chi2_total,
        "total_elapsed_time": elapsed,
        "has_redshifts": False,
        "theta_E_sub": final_str,
        "reconstructed_source": final_src,
        "pred_final": I_final,
        "res_final": res_final,
        "res_macro": res_macro,
        "pred_macro": I_base,
        "radar_score_map": scores,
        "radar_extent": [-1.3 * final_te.item(), 1.3 * final_te.item(), -1.3 * final_te.item(), 1.3 * final_te.item()],
        "obs_image": obs_img,
        "noise_map": noise_map,
        "simulator": pipeline.simulator,
        "is_real_observation": True,
        "fit_mask": mask_t,
        "source_extent": src_extent,
        "source_fov": effective_fov,
        "source_resolution": source_res,
        "source_center": (cx_val, cy_val),
        "lens_id": meta.get("OBJECT", "Euclid Q1 Lens"),
        "object_id": meta.get("OBJECT", "Euclid Q1 Lens"),
        "ra": meta.get("RA"),
        "dec": meta.get("DEC"),
        "expert_score": meta.get("EXPSCORE"),
        "grade": meta.get("GRADE", "A"),
        "detection_flag": detection_flag,
        "host_theta_E": float(final_te.item()),
        "radar_coords": scan_coords_1d.cpu().numpy()
    }
    return SubLensResult(data)


def run_euclid_grade_a_survey(
    catalog_path: Union[str, Path] = "archive/data/q1_discovery_engine_lens_catalog.csv",
    data_dir: Union[str, Path] = "archive/data",
    output_dir: Union[str, Path] = "euclid_q1_diagnostic_plots",
    table_output_csv: str = "euclid_q1_grade_a_catalog.csv",
    table_output_fits: str = "euclid_q1_grade_a_catalog.fits",
    max_systems: Optional[int] = 10,
    radar_resolution: int = 45,
    crop_size: int = 100,
    save_fits_products: bool = True
) -> Tuple[Table, List[SubLensResult]]:
    """
    Standard batch execution API across Euclid Q1 Grade A confirmed gravitational lenses.
    
    1. Prepares standard FITS images (SCI, PSF, RMS, MASK)
    2. Runs SubLensPipeline.from_fits() and .fit()
    3. Generates publication-grade 8-panel diagnostic figures
    4. Compiles structured Astropy Table and exports both CSV and FITS catalogs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_dir = Path("euclid_q1_fits")
    fits_dir.mkdir(parents=True, exist_ok=True)

    available = get_available_grade_a_lenses(catalog_path=catalog_path, data_dir=data_dir)
    if max_systems is not None:
        target_lenses = available[:max_systems]
    else:
        target_lenses = available

    print("=" * 85)
    print(f">>> Launching Euclid Q1 Grade A Strong Lensing Survey ({len(target_lenses)} targets)...")
    print("=" * 85)

    catalog_rows = []
    results = []

    for idx, target in enumerate(target_lenses):
        lid = target["id_str"]
        print(f"\n[{idx+1}/{len(target_lenses)}] Ingesting Euclid Grade A Lens: {lid} (Score: {target['expert_score']:.2f})")

        # 1. Prepare standard FITS
        fits_file = prepare_euclid_grade_a_fits(lid, data_dir=data_dir, crop_size=crop_size)

        # 2. Standard API call
        pipeline = SubLensPipeline.from_fits(fits_file)
        res = pipeline.fit(radar_resolution=radar_resolution)
        results.append(res)

        # 3. Export diagnostic plot
        plot_path = output_dir / f"{lid}_diagnostic.png"
        res.plot(save_path=str(plot_path))

        # 4. Export inversion products FITS
        if save_fits_products:
            prod_path = output_dir / f"{lid}_products.fits"
            res.save_fits(str(prod_path))

        # 5. Record row for catalog table
        m = res.metrics
        status = "DETECTED" if m.get("detection_flag") else "SMOOTH_LIMIT"
        row = {
            "LENS_ID": lid,
            "RA": float(target["ra"]),
            "DEC": float(target["dec"]),
            "EXPERT_SCORE": float(target["expert_score"]),
            "THETA_E": float(res._data.get("host_theta_E", 1.0)),
            "RADAR_SIGNIFICANCE_SIGMA": float(m["detection_significance_s"]),
            "SUBHALO_X_ARCSEC": float(m["subhalo_x_arcsec"]),
            "SUBHALO_Y_ARCSEC": float(m["subhalo_y_arcsec"]),
            "CHI2_BASE": float(m["chi2_red_initial"]),
            "CHI2_SUB": float(m["chi2_red_final"]),
            "DELTA_CHI2": float(res._data.get("delta_chi2_total", m["delta_chi2"])),
            "DETECTION_FLAG": bool(m.get("detection_flag", False)),
            "STATUS": status,
            "RUNTIME_SEC": float(m["elapsed_seconds"]),
            "DIAGNOSTIC_PLOT": str(plot_path.resolve())
        }
        catalog_rows.append(row)

    # 6. Compile Astropy Table
    table = Table(rows=catalog_rows)

    # Export CSV & FITS tables
    table.write(table_output_csv, format="ascii.csv", overwrite=True)
    table.write(table_output_fits, format="fits", overwrite=True)

    print("\n" + "=" * 85)
    print(">>> Survey Complete! Catalogs & Diagnostics Exported:")
    print(f"  * CSV Catalog:  {os.path.abspath(table_output_csv)}")
    print(f"  * FITS Catalog: {os.path.abspath(table_output_fits)}")
    print(f"  * Diagnostics:  {os.path.abspath(output_dir)}")
    print("=" * 85)

    return table, results

