"""Reports and portable JSON/FITS products for PINN halo-search results."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from astropy.io import fits

from .io import observation_hdus


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _chi2(observation, prediction):
    residual = (observation.image[observation.mask] - prediction[observation.mask])
    return float(np.sum((residual / observation.rms[observation.mask])**2))


def pinn_result_dict(result):
    """Describe the search without embedding the full image arrays."""
    from . import __version__
    from .detection import _detection_from_candidates

    obs = result.observation
    count = int(obs.mask.sum())
    best = result.best_candidate
    input_macro = result.config.get("joint_refinement", {}).get("macro_model_input", obs.macro_model)
    baseline_macro = result.macro_model if result.macro_model is not None else input_macro
    baseline_chi2 = _chi2(obs, result.baseline.image_model)
    candidates = []
    for candidate in result.candidates:
        fit = candidate.fit
        chi2 = _chi2(obs, fit.image_model)
        candidates.append({
            "rank": candidate.rank, "is_best": candidate is best,
            "seed_position_arcsec": candidate.seed, "proposal_score": candidate.proposal_score,
            "position_arcsec": fit.position, "strength_arcsec": fit.strength,
            "chi2": chi2, "chi2_per_pixel": chi2 / count,
            "delta_chi2": fit.delta_chi2, "delta_objective": fit.delta_objective,
            "source_objective": fit.source.objective, "converged": fit.converged,
            "optimizer_message": fit.optimizer_message, "iterations": fit.iterations,
            "macro_model": fit.macro_model if fit.macro_model is not None else input_macro,
        })
    return _plain({
        "schema_version": 1, "sublens_version": __version__,
        "lens_id": obs.lens_id,
        "mode": "joint_refinement" if result.macro_model is not None else "fixed_macro",
        "units": {"position": "arcsec", "strength": "arcsec", "angle": "rad",
                  "residual": "image units", "noise": "image units (RMS)",
                  "pinn_cross_fit_residual": "RMS units", "convergence": "dimensionless"},
        "image_shape": obs.image.shape, "pixel_scale_arcsec": obs.pixel_scale,
        "fit_pixel_count": count, "lens_center_arcsec": obs.lens_center,
        "macro_model_input": input_macro, "macro_model_baseline": baseline_macro,
        "macro_converged": result.macro_converged,
        "baseline": {
            "chi2": baseline_chi2, "chi2_per_pixel": baseline_chi2 / count,
            "source_objective": result.baseline.objective,
            "source_node_count": len(result.baseline.coefficients),
            "regularization": result.baseline.regularization,
            "mesh_strategy": result.baseline.mesh_strategy,
        },
        "best_candidate_rank": None if best is None else best.rank,
        "detection": _detection_from_candidates(candidates, lens_id=obs.lens_id,
            min_delta_chi2=result.config.get('detection_min_delta_chi2',25.)),
        "max_strength_arcsec": result.max_strength,
        "refinement_radius_arcsec": result.refinement_radius,
        "candidates": candidates, "config": result.config,
    })


def pinn_result_summary(result, *, detailed=False, min_delta_chi2=None):
    data = pinn_result_dict(result)
    if not detailed:
        from .detection import summarize_subhalo
        decision = summarize_subhalo(data, min_delta_chi2=min_delta_chi2)
        flag = 'yes' if decision['detected'] else 'no'
        if not data['candidates']:
            flag += ' (No PINN candidates)'
        if decision['detected']:
            x, y = decision['position_arcsec']
            position = f'({x:+.5f}, {y:+.5f}) arcsec'
            strength = f"{decision['strength_arcsec']:.6f} arcsec"
        else:
            position, strength = '—', '—'
        return (f"{data['lens_id']}: subhalo detected = {flag}\n"
                f"Position (x, y): {position}\nStrength b: {strength}")
    lines = [
        f"PINN search on {data['lens_id']} ({data['mode']})",
        f"Fitted pixels {data['fit_pixel_count']}, candidates {len(data['candidates'])}",
        f"Macro-only chi2/pixel {data['baseline']['chi2_per_pixel']:.4f}",
    ]
    best = next((item for item in data["candidates"] if item["is_best"]), None)
    if best is None:
        lines.append("No PINN candidates")
    else:
        x, y = best["position_arcsec"]
        lines += [
            f"Best peak {best['rank']}, position ({x:+.5f}, {y:+.5f}) arcsec",
            f"Strength {best['strength_arcsec']:.6f} arcsec",
            f"Macro + subhalo chi2/pixel {best['chi2_per_pixel']:.4f}",
            f"Delta chi2 {best['delta_chi2']:.2f}, penalized gain {best['delta_objective']:.2f}",
            f"Halo refinement converged {best['converged']}",
        ]
    if data["macro_converged"] is not None:
        lines.append(f"Macro-only refinement converged {data['macro_converged']}")
    return "\n".join(lines)


@contextmanager
def _atomic_output(path, overwrite):
    destination = Path(path)
    with NamedTemporaryFile(dir=destination.parent, suffix=destination.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_pinn_json(result, path, *, overwrite=False):
    payload = json.dumps(pinn_result_dict(result), ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    with _atomic_output(path, overwrite) as temporary:
        temporary.write_text(payload, encoding="utf-8")
    return Path(path)


def _candidate_table(data):
    items = data["candidates"]
    definitions = [
        ("RANK", "J", [item["rank"] for item in items], None),
        ("BEST", "L", [item["is_best"] for item in items], None),
        ("SEED_X", "D", [item["seed_position_arcsec"][0] for item in items], "arcsec"),
        ("SEED_Y", "D", [item["seed_position_arcsec"][1] for item in items], "arcsec"),
        ("SCORE", "D", [item["proposal_score"] for item in items], None),
        ("X", "D", [item["position_arcsec"][0] for item in items], "arcsec"),
        ("Y", "D", [item["position_arcsec"][1] for item in items], "arcsec"),
        ("STRENGTH", "D", [item["strength_arcsec"] for item in items], "arcsec"),
        ("CHI2", "D", [item["chi2"] for item in items], None),
        ("CHI2_PIX", "D", [item["chi2_per_pixel"] for item in items], None),
        ("DCHI2", "D", [item["delta_chi2"] for item in items], None),
        ("DOBJECT", "D", [item["delta_objective"] for item in items], None),
        ("OBJECTIVE", "D", [item["source_objective"] for item in items], None),
        ("CONVERGED", "L", [item["converged"] for item in items], None),
    ]
    columns = [fits.Column(name=name, format=fmt, array=values, unit=unit)
               for name, fmt, values, unit in definitions]
    for key, unit in (("theta_E", "arcsec"), ("q", None), ("phi", "rad"),
                      ("gamma", None), ("gamma_ext", None), ("phi_ext", "rad")):
        values = [(item["macro_model"] or {}).get(key, np.nan) for item in items]
        columns.append(fits.Column(name=key.upper(), format="D", array=values, unit=unit))
    for axis, index in (("CENTER_X", 0), ("CENTER_Y", 1)):
        values = [(item["macro_model"] or {}).get("center", [np.nan, np.nan])[index] for item in items]
        columns.append(fits.Column(name=axis, format="D", array=values, unit="arcsec"))
    return fits.BinTableHDU.from_columns(columns, name="CANDIDATES")


def save_pinn_fits(result, path, *, source_half_width=1.0, resolution=201, overwrite=False):
    if not np.isfinite(source_half_width) or source_half_width <= 0 or resolution < 2:
        raise ValueError("Source raster needs a positive half-width and at least two pixels")
    data = pinn_result_dict(result)
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False)
    hdus = observation_hdus(result.observation)
    header = hdus[0].header
    header["RESULTV"] = data["schema_version"]
    header["SLVER"] = data["sublens_version"]
    header["FITMODE"] = data["mode"]
    header["NPEAKS"] = len(data["candidates"])
    header["NPXFIT"] = data["fit_pixel_count"]
    header["CHI2MAC"] = data["baseline"]["chi2"]
    header["HALODET"] = (data["detection"]["detected"], "Passes fit-improvement rule")
    header["DCHIMIN"] = (data["detection"]["rule"]["min_delta_chi2"], "Minimum delta chi2 for HALODET")
    header["DETCAL"] = (False, "Detection significance is not calibrated")
    if data["detection"]["detected"]:
        header["DETRANK"] = data["detection"]["candidate_rank"]
    header["RESIDSIG"] = ("OBS-MOD", "Residual sign convention")
    if data["best_candidate_rank"] is not None:
        header["BESTRANK"] = data["best_candidate_rank"]

    best = result.best_candidate
    source = result.baseline if best is None else best.fit.source
    potential = result.residual_potential
    images = [
        ("MODEL_MACRO", result.baseline.image_model), ("RESID_MACRO", result.residual_macro),
        ("MODEL", source.image_model), ("RESIDUAL", result.observation.image - source.image_model),
        ("PINN_KAPPA", potential.convergence), ("PINN_MASK", potential.mask.astype(np.uint8)),
        ("PINN_BASE_RES", potential.baseline_residual), ("PINN_FIT_RES", potential.fitted_residual),
    ]
    wcs = fits.Header({key: value for key, value in header.items()
                       if key.startswith(("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT"))})
    for name, array in images:
        image = fits.ImageHDU(array, header=wcs.copy(), name=name)
        if name in ("PINN_KAPPA", "PINN_BASE_RES", "PINN_FIT_RES"):
            image.header["BUNIT"] = "1"
        hdus.append(image)

    axis = np.linspace(-source_half_width, source_half_width, resolution)
    raster = fits.ImageHDU(source.sample(*np.meshgrid(axis, axis)), name="SOURCE")
    for index in (1, 2):
        raster.header[f"CTYPE{index}"] = "LINEAR"
        raster.header[f"CUNIT{index}"] = "arcsec"
        raster.header[f"CRPIX{index}"] = (resolution + 1) / 2
        raster.header[f"CRVAL{index}"] = 0.0
        raster.header[f"CDELT{index}"] = 2 * source_half_width / (resolution - 1)
    hdus.append(raster)
    mesh = source.grid
    columns = [
        fits.Column(name="X", format="D", array=mesh.points[:, 0], unit="arcsec"),
        fits.Column(name="Y", format="D", array=mesh.points[:, 1], unit="arcsec"),
        fits.Column(name="BASE_FLUX", format="D", array=result.baseline.coefficients),
        fits.Column(name="FLUX", format="D", array=source.coefficients),
    ]
    for candidate in result.candidates:
        columns.append(fits.Column(name=f"FLUX_{candidate.rank}", format="D",
                                   array=candidate.fit.source.coefficients))
    hdus.append(fits.BinTableHDU.from_columns(columns, name="SOURCE_NODES"))
    triangles = fits.ImageHDU(mesh.simplices, name="SOURCE_TRI")
    triangles.header["INDEXBAS"] = (0, "Vertex indices start at zero in SOURCE_NODES")
    hdus.append(triangles)
    hdus.append(_candidate_table(data))
    metadata = fits.Column(name="JSON", format=f"{len(payload)}A", array=[payload])
    hdus.append(fits.BinTableHDU.from_columns([metadata], name="METADATA"))
    with _atomic_output(path, overwrite) as temporary:
        hdus.writeto(temporary, overwrite=True, checksum=True)
    return Path(path)
